"""Phoenix (ECCV 2026) mask refinement on the Oslo map — section U's third refiner.

Phoenix (https://phoenix-eccv26.github.io, naver-ai/Phoenix) is a model-agnostic
mask refiner: image + noisy binary mask -> cleaner mask. It is the same slot in
section U as guided filtering (U1/U1b) and SAMRefiner (U2), and it inherits
SAMRefiner's prompt sampler outright, so it lands on a branch this ledger has
already closed twice. It is nevertheless *not* an averaging scheme — the claimed
contribution is training-side (adversarial mask perturbation + contrastive
refinement learning), i.e. a refiner that has seen realistic segmentation errors
rather than random geometric ones — so the closing argument of Iteration 8 does
not automatically cover it, exactly as it did not cover SAMRefiner. Measured, not
assumed.

The regimes, because they ask different questions
------------------------------------------------
``--regime classmap`` (default) is the one that produces a usable product: a
transition map with Phoenix applied, carrying the input's codes and legend, so it
opens in QGIS beside the map it came from. Only the **change** classes are
refined, each component inside its own crop window; the stable classes are left
as the ground they are. Pixels the refiner drops fall back to the stable
transition for the state they started in (``X -> Y`` becomes ``X -> X``), because
in a transition map "no change here" is a class, not an absence.

``--regime semantic`` reproduces the paper's own semantic-segmentation protocol
(``phoenix/eval/evaluators.evaluate_semantic``): every connected component of
every class is refined and an arg-max over class-probability channels resolves
overlaps. It is Phoenix applied as its authors apply it, and on this map it is
destructive — the stable classes are one connected mass each, not objects, and
what comes back wins the arg-max over most of the AOI. Kept as the measurement,
not as a way to make a map.

``--regime binary`` and ``--regime crop`` refine the change mask alone and write
a stable/change raster: the diagnostic reads behind sections U5's numbers, at
whole-tile and per-component scale respectively.

What can and cannot be concluded
--------------------------------
Oslo has zero labelled plots, so nothing here is accuracy. It is structure
(``map_detail_metrics``: does the boundary land on a real image edge, do objects
stay coherent) plus the change-pixel count, which is reported beside every row
because a refiner that buys alignment by deleting the minority class is not
doing what it claims. **The yardstick is the map's own reproducibility floor** —
``compare_map_iou.py`` on two disjoint seed blocks — since an edit smaller than
the seed noise is not an edit worth reading.

Runtime
-------
The geo env's torch (2.5.1+cu121) has no kernels for this machine's sm_120 GPU,
so Phoenix runs from a separate venv; see ``--phoenix-root``. FastGeodis does not
build here and is stubbed with scipy, which is exact for the lamb=0 calls
SAMRefiner's sampler makes (see the shim's docstring).

Usage::

    PHX=/home/geethen.singh/.cache/phoenix-test
    PYTHONPATH=$PHX/shim $PHX/venv/bin/python src/phoenix_refine_map.py \\
        --map data/inference/s2_20260731_100710/oslo_s2off_centre_m3s3_bf_merged2.tif \\
        --image data/inference/s2_20260727_130926/oslo_s2_rgb_2024.tif \\
        --reference data/inference/s2_20260727_130926/oslo_s2_nir_2024.tif \\
        --regime semantic binary --output-dir data/inference/phoenix_oslo
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_detail_metrics import metrics_for  # noqa: E402

NODATA = 255
DEFAULT_PHOENIX_ROOT = Path("/home/geethen.singh/.cache/phoenix-test")


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def qml_labels(tif: Path) -> dict[int, str]:
    """{code: label} from the .qml sidecar.

    The codes are the model's *sorted* class list, not the palette order, so the
    sidecar is the only trustworthy source; assuming order once counted stable
    Vegetation as change.
    """
    qml = tif.with_suffix(".qml")
    if not qml.exists():
        raise SystemExit(f"no {qml.name} beside the raster; cannot trust class codes")
    return {int(v): lab for v, lab in
            re.findall(r'paletteEntry value="(\d+)"[^>]*label="([^"]*)"',
                       qml.read_text(encoding="utf-8"))}


def is_change(label: str) -> bool:
    """True for a transition label whose endpoints differ."""
    if " -> " not in label:
        return False
    a, b = label.split(" -> ")[0], label.split(" -> ")[-1]
    return a != b


def load_map(path: Path):
    with rasterio.open(path) as src:
        codes = src.read(1)
        profile = src.profile
        try:
            colormap = src.colormap(1)
        except ValueError:
            colormap = None
    nodata = profile.get("nodata")
    valid = np.ones(codes.shape, bool) if nodata is None else codes != nodata
    return codes, valid, profile, colormap


def rgb_uint8(path: Path, valid: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 from the S2 composite, per-band 2-98% stretched.

    Phoenix's encoder is a natural-image backbone with fixed ImageNet-scale
    normalisation, so the composite has to be brought onto an 8-bit photographic
    range before it means anything to the network. The stretch is per band and
    over valid pixels only.
    """
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
        nod = src.nodata
    if arr.shape[0] < 3:
        arr = np.repeat(arr[:1], 3, axis=0)
    arr = arr[:3]
    out = np.zeros((*arr.shape[1:], 3), "uint8")
    for b in range(3):
        band = arr[b]
        good = valid & np.isfinite(band)
        if nod is not None:
            good &= band != nod
        lo, hi = np.percentile(band[good], [2, 98])
        out[..., b] = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)[...] * 255
    return out


def tile_origins(size: int, tile: int) -> list[int]:
    """Tile starts covering ``size`` with ``tile``-wide windows, last one flush.

    Phoenix's transform resizes the longest side to 1024, so a 1024 tile is the
    only window the encoder sees at native 10 m resolution — handing it the whole
    1722 px AOI would silently downsample the imagery by 0.6x in a *detail*
    experiment.
    """
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile, tile)) + [size - tile]
    return sorted(set(starts))


def nearest_tile_index(shape, windows) -> np.ndarray:
    """Per-pixel index of the tile whose centre is nearest — the mosaic rule.

    Overlapping tiles both predict the overlap; taking the nearer centre keeps
    every pixel as far as possible from the window edge, where a refiner has the
    least context.
    """
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    best = np.full(shape, -1, "int16")
    best_d = np.full(shape, np.inf, "float32")
    for i, (r0, c0, r1, c1) in enumerate(windows):
        d = np.hypot(rows - (r0 + r1) / 2, cols - (c0 + c1) / 2).astype("float32")
        inside = np.zeros(shape, bool)
        inside[r0:r1, c0:c1] = True
        take = inside & (d < best_d)
        best[take] = i
        best_d[take] = d[take]
    return best


# --------------------------------------------------------------------------
# refinement
# --------------------------------------------------------------------------
def components(mask: np.ndarray, merge: bool):
    """Connected components of a binary mask, optionally via SAMRefiner's merge.

    ``merge_regions`` is what the paper's semantic evaluator uses: it drops
    components under 5 px and fuses ones whose boxes overlap heavily, so a
    fragmented class is not handed to the decoder as hundreds of specks. Applied
    here for the same reason and with the same code, since the alternative is a
    different protocol from the one being tested.
    """
    import cv2
    from phoenix.utils.samrefiner import merge_regions

    m = mask.astype("uint8")
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return []
    if not merge:
        return [(labels == k) for k in range(1, num)]
    num_m, _, merged = merge_regions(num - 1, stats[1:], centroids[1:], labels)
    if merged is not None:
        return [np.asarray(m_) for m_ in merged]
    return [(labels == k) for k in range(1, num_m + 1)]


def component_windows(mask: np.ndarray, merge: bool, min_px: int = 1):
    """Yield ``(bbox, local_mask)`` per component, without materialising H×W masks.

    The change mask has ~1,000 components over the AOI; one full-size boolean
    array each is ~3 GB, so the crop regime works from the label image and each
    component's bounding box instead.

    ``merge`` reproduces SAMRefiner's ``merge_regions``, which is what the paper's
    protocol uses — but note what it does to *this* mask: it fuses components
    whose boxes overlap, and on a fragmented change class that means 550 blobs
    become 41 regions sprawling across the AOI. Those are not localised objects,
    so a window centred on one covers a fraction of it. That is why the crop
    regime defaults to raw components: cropping only means anything if the thing
    cropped is compact.
    """
    import cv2

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype("uint8"), connectivity=8)
    if num <= 1:
        return []
    if merge:
        from phoenix.utils.samrefiner import merge_regions

        _, _, merged = merge_regions(num - 1, stats[1:], stats[1:, :2], labels)
        out = []
        for m in merged or []:
            m = np.asarray(m)
            rr, cc = np.nonzero(m)
            if rr.size < min_px:
                continue
            box = (rr.min(), cc.min(), rr.max() + 1, cc.max() + 1)
            out.append((box, m[box[0]:box[2], box[1]:box[3]]))
        return out
    out = []
    for k in range(1, num):
        x, y, w, h, area = stats[k]
        if area < min_px:
            continue
        box = (y, x, y + h, x + w)
        out.append((box, labels[box[0]:box[2], box[1]:box[3]] == k))
    return out


def stable_counterpart(labels: dict[int, str]) -> dict[int, int]:
    """{change code: the ``X -> X`` code for the same starting state}.

    Needed because removing change from a *transition* map is not the same
    operation as removing it from a binary mask: a pixel the refiner drops has
    to become some class, and the only defensible one is the stable transition
    for the state it started in — the refiner's claim is "nothing happened
    here", which means the 2018 state persisted. Falling back to the input class
    instead would make the refiner incapable of ever removing change.
    """
    by_name = {lab: code for code, lab in labels.items()}
    out = {}
    for code, lab in labels.items():
        if not is_change(lab):
            continue
        src = lab.split(" -> ")[0]
        stable = by_name.get(f"{src} -> {src}")
        if stable is not None:
            out[code] = stable
    return out


def refine_classmap(refiner, image, codes, valid, labels, window, refine_iters,
                    smooth_kernel, merge, min_px, records=None):
    """Phoenix applied to a transition map, returning a transition map.

    Only the **change** classes are refined. That is the whole difference from
    the paper's semantic protocol, which refines every class and destroys this
    map: the stable classes cover 99.4% of the AOI in one connected mass each,
    they are not objects, and handing them to an object refiner returns garbage
    that then wins the arg-max everywhere. Here they are the ground the refined
    change sits on, exactly as in the input.

    Composition, in order: a pixel claimed by one or more refined change
    components takes the class of the most confident one; a pixel that was
    change in the input and is claimed by nothing takes its stable counterpart;
    everything else keeps its input class. Codes and legend are the input's
    throughout, so the output drops straight into QGIS beside the map it came
    from.
    """
    change_codes = [c for c, lab in sorted(labels.items()) if is_change(lab)]
    fallback = stable_counterpart(labels)
    missing = [labels[c] for c in change_codes if c not in fallback]
    if missing:
        print(f"  note: no stable counterpart for {missing}; those keep their "
              f"input class where Phoenix drops them")

    prob = np.zeros((len(change_codes), *codes.shape), "float32")
    n_comp = 0
    for i, code in enumerate(change_codes):
        mask = (codes == code) & valid
        if not mask.any():
            continue
        rec = [] if records is not None else None
        _, n = refine_crop(refiner, image, mask, window, refine_iters,
                           smooth_kernel, merge, rec, min_px, prob_out=prob[i])
        n_comp += n
        if records is not None:
            for r in rec:
                r["class"] = labels[code]
            records.extend(rec)
        print(f"  {labels[code]:28s} {int(mask.sum()):>8,} px  {n:>4d} components"
              f" -> {int((prob[i] > 0).sum()):>8,} px", flush=True)

    out = codes.copy()
    claimed = prob.max(0) > 0
    winner = np.array(change_codes, dtype="uint8")[prob.argmax(0)]
    was_change = np.isin(codes, change_codes)
    dropped = was_change & ~claimed & valid
    for code, stable in fallback.items():
        out[dropped & (codes == code)] = stable
    out[claimed & valid] = winner[claimed & valid]
    return np.where(valid, out, NODATA), n_comp


def refine_crop(refiner, image, mask, window, refine_iters, smooth_kernel,
                merge, records=None, min_px=1, prob_out=None):
    """Refine each change component inside its own ``window``-px crop.

    The control every negative result at this scale has to answer. Phoenix's
    transform resizes the longest side to 1024, so a whole-tile pass shows the
    encoder a 20 px change blob at 20 px — far below anything a natural-image
    refiner was trained on. Cropping a small window around the component and
    letting that same transform *upsample* it is the standard way SAM-family
    models are given small objects, and it is the most favourable framing the
    method has here. If it fails at 1024 and succeeds cropped, the finding is
    about scale; if it fails both ways, it is about the method.

    Costs one encoder pass per component instead of one per tile, which is why
    it is not the default.
    """
    comps = component_windows(mask, merge, min_px)
    out = np.zeros(mask.shape, bool)
    half = window // 2
    H, W = mask.shape
    for (br0, bc0, br1, bc1), local in comps:
        r_mid, c_mid = (br0 + br1) // 2, (bc0 + bc1) // 2
        r0 = int(np.clip(r_mid - half, 0, max(H - window, 0)))
        c0 = int(np.clip(c_mid - half, 0, max(W - window, 0)))
        r1, c1 = min(r0 + window, H), min(c0 + window, W)
        sub_mask = np.zeros((r1 - r0, c1 - c0), bool)
        # A component wider than the window is clipped by the paste, not dropped;
        # the alternative is silently skipping the largest objects, which are the
        # ones the method should handle best.
        rr0, cc0 = max(br0, r0), max(bc0, c0)
        rr1, cc1 = min(br1, r1), min(bc1, c1)
        sub_mask[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = \
            local[rr0 - br0:rr1 - br0, cc0 - bc0:cc1 - bc0]
        if not sub_mask.any():
            continue
        refiner.set_image(np.ascontiguousarray(image[r0:r1, c0:c1]))
        result = refiner.refine_current(
            sub_mask.astype("uint8"), refine_iters=refine_iters,
            smooth_kernel=smooth_kernel, binarize_input=True,
            return_logits=prob_out is not None)
        refined, logits = result if prob_out is not None else (result, None)
        out[r0:r1, c0:c1] |= refined
        if prob_out is not None:
            # Components of one class can overlap after refinement; keep the
            # most confident claim so the later arg-max across classes is
            # comparing like with like.
            # `logits` is already sigmoided in Predictor.predict_torch
            # (`out_dict["logits"] = torch.sigmoid(masks)`), so it is a
            # probability in [0, 1] and must not be squashed a second time.
            view = prob_out[r0:r1, c0:c1]
            np.maximum(view, np.where(refined, logits, 0.0), out=view)
        if records is not None:
            union = (sub_mask | refined).sum()
            records.append({
                "input_px": int(sub_mask.sum()),
                "refined_px": int(refined.sum()),
                "area_ratio": float(refined.sum() / max(sub_mask.sum(), 1)),
                "iou": float((sub_mask & refined).sum() / union) if union else np.nan,
            })
    return out, len(comps)


def refine_semantic(refiner, image, codes, labels, refine_iters, smooth_kernel,
                    merge, max_batch):
    """Paper protocol: refine every component of every class, compose by prob.

    Returns (codes_out, n_components).
    """
    keys = sorted(labels)
    prob_stack = np.zeros((len(keys), *codes.shape), "float32")
    refiner.set_image(image)
    total = 0
    for i, code in enumerate(keys):
        comps = components(codes == code, merge)
        total += len(comps)
        if not comps:
            continue
        masks, probs = refiner.refine_batch(
            [c.astype("uint8") for c in comps], refine_iters=refine_iters,
            smooth_kernel=smooth_kernel, binarize_input=True,
            return_logits=True, max_batch=max_batch,
        )
        for refined, prob in zip(masks, probs):
            prob_stack[i] = np.where(refined, prob, prob_stack[i])
    claimed = prob_stack.max(0) > 0
    winner = np.array(keys, dtype="uint8")[prob_stack.argmax(0)]
    return np.where(claimed, winner, codes), total


def refine_binary(refiner, image, mask, refine_iters, smooth_kernel, merge,
                  max_batch, records=None):
    """Refine one binary mask's components and union them back. Returns (mask, n).

    ``records`` collects one row per component (input area, refined area, IoU).
    That per-component read is what separates the three ways a refiner can fail
    here, which the unioned raster cannot: a no-op returns its input (U2a's IoU
    1.000), a dilating refiner returns something much larger, and a genuine
    refinement returns a similar area with a different boundary. Size is carried
    with each row because the whole question at 10 m is whether an "object" is
    big enough for a natural-image refiner to have an opinion about it.
    """
    refiner.set_image(image)
    comps = components(mask, merge)
    if not comps:
        return np.zeros_like(mask), 0
    masks = refiner.refine_batch(
        [c.astype("uint8") for c in comps], refine_iters=refine_iters,
        smooth_kernel=smooth_kernel, binarize_input=True, max_batch=max_batch,
    )
    out = np.zeros(mask.shape, bool)
    for comp, refined in zip(comps, masks):
        out |= refined
        if records is not None:
            comp = comp.astype(bool)
            union = (comp | refined).sum()
            records.append({
                "input_px": int(comp.sum()),
                "refined_px": int(refined.sum()),
                "area_ratio": float(refined.sum() / max(comp.sum(), 1)),
                "iou": float((comp & refined).sum() / union) if union else np.nan,
            })
    return out, len(comps)


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------
def write_like(path: Path, codes: np.ndarray, profile, colormap,
               src_qml: Path | None):
    """Write a class raster sharing the source's codes, palette and .qml.

    Copying the sidecar rather than regenerating it is what keeps the refined map
    readable with the same code->label table as its input; re-deriving the codes
    from whichever classes happen to survive is how a refined raster ends up
    silently permuted against the map it is being compared to. **Only valid when
    the output really does carry the source's codes** — pass ``src_qml=None``
    otherwise and write a sidecar that matches (see ``write_qml``).
    """
    from rasterio.enums import Resampling

    prof = dict(profile)
    prof.update(driver="GTiff", count=1, dtype="uint8", nodata=NODATA,
                compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(codes.astype("uint8"), 1)
        if colormap:
            dst.write_colormap(1, colormap)
        dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
        dst.update_tags(ns="rio_overview", resampling="nearest")
    if src_qml is not None and src_qml.exists():
        path.with_suffix(".qml").write_text(src_qml.read_text(encoding="utf-8"),
                                            encoding="utf-8")


BINARY_LABELS = {0: ("stable", (225, 225, 225)), 1: ("change", (200, 30, 30))}


def write_qml(path: Path, entries: dict[int, tuple[str, tuple[int, int, int]]]):
    """Paletted-raster style naming each code. Written, never copied.

    Copying the source map's sidecar onto a raster with different codes is the
    failure this project has already paid for once: a two-value change mask
    inheriting the four-class merged2 legend renders value 1 as
    "Artificial -> Vegetation" in QGIS when it means "change", and the map looks
    plausible while being read wrong.
    """
    rows = "\n".join(
        f'          <paletteEntry value="{v}" '
        f'color="#{c[0]:02x}{c[1]:02x}{c[2]:02x}" alpha="255" label="{lab}"/>'
        for v, (lab, c) in sorted(entries.items()))
    path.write_text(
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        '<qgis version="3.34" styleCategories="AllStyleCategories">\n'
        '  <pipe>\n    <rasterrenderer type="paletted" band="1" opacity="1">\n'
        "      <colorPalette>\n" + rows + "\n      </colorPalette>\n"
        "    </rasterrenderer>\n  </pipe>\n</qgis>\n", encoding="utf-8")


def write_binary(path: Path, mask: np.ndarray, valid, profile, src_qml: Path):
    codes = np.where(valid, mask.astype("uint8"), NODATA)
    colormap = {v: (*c, 255) for v, (_, c) in BINARY_LABELS.items()}
    colormap[NODATA] = (0, 0, 0, 0)
    # src_qml is deliberately not passed through: this raster's codes are
    # stable/change, not the source map's transitions.
    write_like(path, codes, profile, colormap, src_qml=None)
    write_qml(path.with_suffix(".qml"), BINARY_LABELS)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", type=Path, required=True, help="merged2 class raster")
    ap.add_argument("--image", type=Path, required=True, help="S2 RGB composite")
    ap.add_argument("--reference", type=Path, default=None,
                    help="image for boundary_align (the ledger uses S2 NIR 2024)")
    ap.add_argument("--regime", nargs="+", default=["classmap"],
                    choices=["classmap", "semantic", "binary", "crop"])
    ap.add_argument("--crop-window", type=int, default=256,
                    help="crop regime: window around each change component. "
                         "Phoenix upsamples it to 1024, so 256 magnifies 4x")
    ap.add_argument("--crop-merge", action="store_true",
                    help="crop regime: fuse components with merge_regions first "
                         "(the paper's protocol; produces non-compact regions here)")
    ap.add_argument("--crop-min-px", type=int, default=1,
                    help="crop regime: skip components below this size")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--encoder", default="efficientvit_xl1")
    ap.add_argument("--phoenix-root", type=Path, default=DEFAULT_PHOENIX_ROOT,
                    help="checkout holding Phoenix/ and ckpt/")
    ap.add_argument("--tile", type=int, default=1024,
                    help="1024 is the encoder's native size; anything larger is "
                         "downsampled by Phoenix's own transform")
    ap.add_argument("--refine-iters", type=int, default=5)
    ap.add_argument("--smooth-kernel", type=int, default=1)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--no-merge-regions", action="store_true",
                    help="skip SAMRefiner's small-component merge (the paper's "
                         "semantic protocol uses it; this is the ablation)")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, str(args.phoenix_root / "Phoenix"))
    import phoenix

    ckpt = args.checkpoint or (args.phoenix_root / "ckpt" /
                               f"phoenix_{args.encoder}.pt")
    out_dir = args.output_dir or args.map.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    codes, valid, profile, colormap = load_map(args.map)
    labels = qml_labels(args.map)
    change_codes = [c for c, lab in labels.items() if is_change(lab)]
    image = rgb_uint8(args.image, valid)
    if image.shape[:2] != codes.shape:
        raise SystemExit(f"image {image.shape[:2]} does not match map {codes.shape}")
    base_change = np.isin(codes, change_codes) & valid

    rows = tile_origins(codes.shape[0], args.tile)
    cols = tile_origins(codes.shape[1], args.tile)
    windows = [(r, c, r + args.tile, c + args.tile) for r in rows for c in cols]
    owner = nearest_tile_index(codes.shape, windows)
    print(f"map {args.map.name}: {valid.sum():,} valid px, "
          f"{base_change.sum():,} change px ({base_change.sum()/valid.sum():.2%})")
    print(f"classes {labels}")
    print(f"{len(windows)} tiles of {args.tile} px "
          f"({len(rows)}x{len(cols)}, overlapping)\n")

    model = phoenix.build_phoenix(checkpoint=str(ckpt), encoder=args.encoder,
                                  device=args.device)
    refiner = phoenix.PhoenixRefiner(model)
    merge = not args.no_merge_regions

    summary = {"map": str(args.map), "image": str(args.image),
               "checkpoint": str(ckpt), "encoder": args.encoder,
               "tile": args.tile, "refine_iters": args.refine_iters,
               "merge_regions": merge, "base_change_px": int(base_change.sum()),
               "valid_px": int(valid.sum()), "regimes": {}}

    for regime in args.regime:
        t0 = time.time()
        out_codes = codes.copy()
        out_mask = base_change.copy()
        n_comp = 0
        records = [] if regime in ("binary", "crop", "classmap") else None

        if regime == "classmap":
            out_codes, n_comp = refine_classmap(
                refiner, image, codes, valid, labels, args.crop_window,
                args.refine_iters, args.smooth_kernel, args.crop_merge,
                args.crop_min_px, records)
            windows_iter = []
        elif regime == "crop":
            # Whole-AOI components, not per tile: the crop is the window, so
            # there is nothing for tiling to buy and splitting a component
            # across a tile seam would only invent boundaries.
            # merge_regions is off by default here even though the paper's
            # protocol uses it: it fuses this mask's blobs into AOI-spanning
            # regions, which a crop window cannot represent (see
            # component_windows). --crop-merge restores it for the ablation.
            out_mask, n_comp = refine_crop(
                refiner, image, base_change, args.crop_window, args.refine_iters,
                args.smooth_kernel, args.crop_merge, records, args.crop_min_px)
            print(f"  {n_comp} components, window {args.crop_window} px, "
                  f"{time.time()-t0:.0f}s", flush=True)
            windows_iter = []
        else:
            windows_iter = list(enumerate(windows))

        for i, (r0, c0, r1, c1) in windows_iter:
            take = owner[r0:r1, c0:c1] == i
            tile_img = np.ascontiguousarray(image[r0:r1, c0:c1])
            if regime == "semantic":
                tile_out, n = refine_semantic(
                    refiner, tile_img, codes[r0:r1, c0:c1], labels,
                    args.refine_iters, args.smooth_kernel, merge, args.max_batch)
                out_codes[r0:r1, c0:c1] = np.where(take, tile_out,
                                                   out_codes[r0:r1, c0:c1])
            else:
                tile_out, n = refine_binary(
                    refiner, tile_img, base_change[r0:r1, c0:c1],
                    args.refine_iters, args.smooth_kernel, merge, args.max_batch,
                    records)
                out_mask[r0:r1, c0:c1] = np.where(take, tile_out,
                                                  out_mask[r0:r1, c0:c1])
            n_comp += n
            print(f"  tile {i+1}/{len(windows)} [{r0}:{r1},{c0}:{c1}] "
                  f"{n} components  {time.time()-t0:.0f}s", flush=True)

        if regime in ("semantic", "classmap"):
            out_codes = np.where(valid, out_codes, NODATA)
            new_change = np.isin(out_codes, change_codes) & valid
            # `_phoenix` beside the source stem, because the output is the
            # same product as its input -- same codes, same legend -- and is
            # meant to be opened next to it.
            suffix = "_phoenix" if regime == "classmap" else "_phoenix_semantic"
            path = out_dir / f"{args.map.stem}{suffix}.tif"
            write_like(path, out_codes, profile, colormap,
                       args.map.with_suffix(".qml"))
        else:
            new_change = out_mask & valid
            path = out_dir / f"{args.map.stem}_phoenix_{regime}_change.tif"
            write_binary(path, new_change, valid, profile,
                         args.map.with_suffix(".qml"))

        stats = metrics_for(path, args.reference)
        # The binary map's structure metrics are on a 2-class raster and are not
        # comparable to the 4-class merged2 ones; its own input is the baseline.
        base_path = args.map
        if regime in ("binary", "crop"):
            base_path = out_dir / f"{args.map.stem}_change_input.tif"
            if not base_path.exists():
                write_binary(base_path, base_change, valid, profile,
                             args.map.with_suffix(".qml"))
        base_stats = metrics_for(base_path, args.reference)

        per_class = {}
        if regime in ("semantic", "classmap"):
            for code, lab in sorted(labels.items()):
                per_class[lab] = {
                    "input_px": int(((codes == code) & valid).sum()),
                    "refined_px": int(((out_codes == code) & valid).sum()),
                    "iou": iou((codes == code) & valid, (out_codes == code) & valid),
                }

        rec = {
            "raster": str(path),
            "seconds": round(time.time() - t0, 1),
            "components": n_comp,
            "change_px": int(new_change.sum()),
            "change_delta": float(new_change.sum() / max(base_change.sum(), 1) - 1),
            "change_iou_with_input": iou(new_change, base_change),
            "px_moved_frac": float(((out_codes != codes) & valid).sum() / valid.sum())
                             if regime in ("semantic", "classmap")
                             else float((new_change != base_change).sum() / valid.sum()),
            "input": {k: base_stats[k] for k in
                      ("edge_density", "median_segment_px", "boundary_align")},
            "refined": {k: stats[k] for k in
                        ("edge_density", "median_segment_px", "boundary_align")},
            "per_class": per_class,
        }
        if records:
            import pandas as pd

            comp_frame = pd.DataFrame(records)
            comp_csv = out_dir / f"{args.map.stem}_phoenix_{regime}_components.csv"
            comp_frame.to_csv(comp_csv, index=False)
            # Binned by input size: the question is not "does it dilate" but
            # "at what object size does it stop dilating", and a single median
            # over a size distribution spanning 5 px to 5,000 px hides that.
            bins = [0, 10, 30, 100, 300, 1000, 10 ** 9]
            comp_frame["bin"] = pd.cut(comp_frame["input_px"], bins,
                                       labels=["<10", "10-30", "30-100",
                                               "100-300", "300-1k", ">1k"])
            grouped = comp_frame.groupby("bin", observed=True).agg(
                n=("iou", "size"), median_iou=("iou", "median"),
                median_area_ratio=("area_ratio", "median"),
                median_input_px=("input_px", "median"))
            rec["components_by_size"] = json.loads(grouped.to_json(orient="index"))
            rec["median_component_iou"] = float(comp_frame["iou"].median())
            rec["median_area_ratio"] = float(comp_frame["area_ratio"].median())
            print("\n  per component (input size bin):")
            print(grouped.round(3).to_string())
            print(f"  -> {comp_csv}")
        summary["regimes"][regime] = rec

        print(f"\n[{regime}] {rec['seconds']}s  {n_comp} components")
        print(f"  change px {base_change.sum():,} -> {new_change.sum():,} "
              f"({rec['change_delta']:+.1%}), IoU with input "
              f"{rec['change_iou_with_input']:.4f}")
        print(f"  edge {base_stats['edge_density']:.4f} -> {stats['edge_density']:.4f}"
              f"   medseg {base_stats['median_segment_px']:.0f} -> "
              f"{stats['median_segment_px']:.0f}"
              f"   align {base_stats['boundary_align']:.4f} -> "
              f"{stats['boundary_align']:.4f}")
        for lab, d in per_class.items():
            print(f"    {lab:28s} {d['input_px']:>9,} -> {d['refined_px']:>9,}  "
                  f"IoU {d['iou']:.4f}")
        print(f"  -> {path}\n", flush=True)

    out_json = out_dir / f"{args.map.stem}_phoenix_summary.json"
    out_json.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"-> {out_json}")


if __name__ == "__main__":
    main()
