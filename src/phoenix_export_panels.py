"""Export the image panels behind the Phoenix comparison page.

The rasters in `phoenix_oslo_explore/` are the thing to load in QGIS; this exists
for the reading that QGIS makes slow — flipping between the same window under the
input map and under each Phoenix arm, at several places at once, without setting
up layer visibility by hand.

Windows are chosen by where the two maps *disagree most*, not at random and not
by eye: the product of the local "change in both" and "added by Phoenix"
densities, greedily de-overlapped. That picks places where Phoenix kept something
and grew it, which is where the method's behaviour is legible; sampling uniformly
would mostly return the 99.4% of the AOI where nothing happens.

Panels are JPEG for the photographic ones and PNG for the flat-colour delta,
because a 4-colour PNG of a delta is a few kB while its JPEG is both larger and
wrong (ringing on hard edges).
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phoenix_delta_map import DELTA_COLORS  # noqa: E402
from phoenix_refine_map import is_change, load_map, qml_labels, rgb_uint8  # noqa: E402

CHANGE_RGB = (255, 45, 45)


def pick_windows(delta: np.ndarray, size: int, n: int) -> list[tuple[int, int]]:
    from scipy import ndimage

    score = (ndimage.uniform_filter((delta == 2).astype("float32"), size)
             * ndimage.uniform_filter((delta == 1).astype("float32"), size))
    picks, sc = [], score.copy()
    h, w = delta.shape
    for _ in range(n):
        r, c = np.unravel_index(sc.argmax(), sc.shape)
        picks.append((int(np.clip(r - size // 2, 0, h - size)),
                      int(np.clip(c - size // 2, 0, w - size))))
        sc[max(0, r - size):r + size, max(0, c - size):c + size] = 0
    return picks


def overlay(img: np.ndarray, mask: np.ndarray, colour, alpha: float = 0.55):
    out = img.astype("float32").copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(colour, "float32")
    return out.astype("uint8")


def encode(arr: np.ndarray, fmt: str, quality: int = 82) -> str:
    import io

    buf = io.BytesIO()
    im = Image.fromarray(arr)
    if fmt == "jpeg":
        im.save(buf, "JPEG", quality=quality, subsampling=0)
    else:
        im.convert("P", palette=Image.ADAPTIVE, colors=8).save(buf, "PNG",
                                                               optimize=True)
    mime = "jpeg" if fmt == "jpeg" else "png"
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--delta", type=Path, required=True,
                    help="delta raster used to choose the windows")
    ap.add_argument("--arm", action="append", nargs=2, metavar=("NAME", "TIF"),
                    required=True, help="label and refined change raster; repeatable")
    ap.add_argument("--window", type=int, default=384)
    ap.add_argument("--n-windows", type=int, default=6)
    ap.add_argument("--scale", type=int, default=2,
                    help="nearest-neighbour upscale, so 10 m pixels stay visible")
    ap.add_argument("--output", type=Path, required=True,
                    help="HTML page; the payload is inlined into it")
    args = ap.parse_args()

    codes, valid, _, _ = load_map(args.map)
    labels = qml_labels(args.map)
    base = np.isin(codes, [c for c, l in labels.items() if is_change(l)]) & valid
    rgb = rgb_uint8(args.image, valid)
    with rasterio.open(args.delta) as src:
        delta = src.read(1)
    arms = {name: (rasterio.open(tif).read(1) == 1) & valid
            for name, tif in args.arm}

    dcolors = np.zeros((256, 3), "uint8")
    for i, c in enumerate(DELTA_COLORS):
        dcolors[i] = DELTA_COLORS[c]

    picks = pick_windows(delta, args.window, args.n_windows)
    scale, W = args.scale, args.window
    up = lambda a: np.kron(a, np.ones((scale, scale, 1), a.dtype)) \
        if a.ndim == 3 else np.kron(a, np.ones((scale, scale), a.dtype))

    out = {"window_px": W, "scale": scale, "windows": []}
    for r, c in picks:
        sl = (slice(r, r + W), slice(c, c + W))
        img = rgb[sl]
        panels = {
            "imagery": encode(up(img), "jpeg"),
            "input": encode(up(overlay(img, base[sl], CHANGE_RGB)), "jpeg"),
        }
        for name, mask in arms.items():
            panels[name] = encode(up(overlay(img, mask[sl], CHANGE_RGB)), "jpeg")
        panels["delta"] = encode(up(dcolors[delta[sl]]), "png")
        w = delta[sl]
        out["windows"].append({
            "row": r, "col": c,
            "stats": {"kept": int((w == 1).sum()), "added": int((w == 2).sum()),
                      "removed": int((w == 3).sum())},
            "panels": panels,
        })

    # Whole-AOI overview at 1:2, so the page opens on context rather than on a
    # crop with no indication of where in Oslo it sits.
    small = slice(None, None, 2)
    out["overview"] = {
        "imagery": encode(rgb[small, small], "jpeg", 78),
        "input": encode(overlay(rgb, base, CHANGE_RGB)[small, small], "jpeg", 78),
        "delta": encode(dcolors[delta][small, small], "png"),
        "shape": [int(np.ceil(rgb.shape[0] / 2)), int(np.ceil(rgb.shape[1] / 2))],
    }
    for name, mask in arms.items():
        out["overview"][name] = encode(
            overlay(rgb, mask, CHANGE_RGB)[small, small], "jpeg", 78)

    # Inlined rather than fetched: a published artifact runs under a CSP that
    # blocks every request off-host, so the payload has to travel in the file.
    # Same template-substitution pattern as umap_page_template.html.
    template = Path(__file__).resolve().parent / "phoenix_compare_template.html"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        template.read_text(encoding="utf-8").replace("__PHOENIX_DATA__",
                                                     json.dumps(out)),
        encoding="utf-8")
    mb = args.output.stat().st_size / 1e6
    print(f"{len(picks)} windows + overview -> {args.output} ({mb:.1f} MB)")
    if mb > 15:
        print("  WARNING: over the 16 MB artifact limit; drop --n-windows "
              "or --scale")


if __name__ == "__main__":
    main()
