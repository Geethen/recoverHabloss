"""Map the recommended Sentinel-2 models over an AOI, for visual inspection.

S2_DETAIL_RESEARCH.md ends with three models and no single winner, because they
optimise different things. This maps them over the same pixels so the choice can
be made by eye on the artefact rather than from a metric table:

    baseline_aef     AlphaEarth only -- the incumbent, and the map the user
                     already judged best on stable built-up
    aef_builtfrac    + the NDVI built-fraction covariate (flat, no second tower).
                     Best built-up numbers on the board: artStab 0.657,
                     art->veg 0.192, at no change-F1 cost
    mc_s2_drop0.7    AlphaEarth tower + Sentinel-2 detail tower behind a
                     STOCHASTIC gate, 16 MC passes. Best S2 change-F1 0.6656

The three share a pipeline, an AOI, a geobox and a palette, so any visible
difference is the model and not the plumbing.

Train/serve skew is the risk here, and it is guarded rather than hoped for
--------------------------------------------------------------------------
Training reads S2 features from a stored 64x64 patch per plot
(`build_s2_features.features_for_year`); inference must read the *same*
statistics from a raster, per pixel. Those are different code paths over
different array layouts, which is exactly how a silent skew gets in.

`raster_features` therefore computes every family as a sliding window over the
AOI -- means and standard deviations at 3/9/25 px, local contrast, Sobel
gradient, built fraction at the calibrated NDVI cut -- and `--self-check`
verifies it: it runs the raster path over each stored plot patch as if it were a
tiny raster and compares its centre pixel against the training path's value for
that plot, column by column. Any drift above tolerance aborts before a map is
written. NaN-aware windowing matters here (the training path uses nanmean/nanstd,
so absent pixels must be skipped, not propagated).

What the deployed read costs, and what it used to
-------------------------------------------------
`s2off_deploy` never touches Sentinel-2 at inference, so its whole serving cost
is: fetch AlphaEarth, arrange it, run the AlphaEarth tower once per ensemble
member. Arranging it turned out to be five sixths of that. At Oslo scale
(2.95 Mpx, 5 seeds), measured on the deployed path:

    stage                       before    after
    AlphaEarth pixel matrix      12.3 s     1.8 s   band-major (stack_aef_bands)
    per-model re-stack           11.1 s     0.0 s   the same array, not a copy
    predict                      42.8 s     4.7 s   no DataFrame, standardise on GPU
    total                        66.2 s     6.5 s

None of it changed a number: the fast path is asserted bit-identical to the
general one (`tests/test_s2off_fastpath.py`). The model was never the cost --
pandas construction and five numpy standardisations per batch were 84% of
`predict`, against 98 ms of actual tower per batch.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
from rasterio.crs import CRS as RioCRS
from rasterio.windows import Window, from_bounds

from aef_loader import AEFIndex, DataSource, VirtualTiffReader, aoi_geobox

import build_s2_features as BSF
import extract_s2_points as EXS
from experiment_merged_legend import LEGENDS, build_frame, target_for_legend
from infer_cities import CITY_AOIS, CLASS_COLORS, load_year_embeddings
from infer_twotower import MERGED_COLORS, NODATA, write_class_raster
from model_zoo import DEFAULT_INPUT, HierarchicalSoftmaxNN, to_merged_label
from project_paths import project_data_dir
from twotower_lab import (S2_MASK, S2_SUBSET_DESC, S2_SUBSETS, _state_pool,
                          attach_s2, s2_base_columns, s2_subset_columns)

YEARS = (2018, 2024)
MC_PASSES = 16   # only for --mc-sampling, the legacy reproduction path
MC_KEEP = 0.5    # P(detail tower is trusted); the quantity the MC passes averaged
# Training-time P(a present modality is dropped). For the deployed gate-off
# recipe this is the one knob that matches training to serving: the model is
# served at "detail tower never fires", so it should be *trained* near that.
# Swept in `experiment_s2off_training.py`; see S16.
MODALITY_DROPOUT = 0.5
CHANGE_COLORS = {"stable": (225, 225, 225), "change": (200, 30, 30)}


# --------------------------------------------------------------------------
# Sentinel-2 composite over the AOI
# --------------------------------------------------------------------------
def _seasonal_scenes(client, bbox, year, per_year):
    """Least-cloudy scene per season for the AOI, grouped by granule."""
    from collections import defaultdict

    items = []
    for limit in EXS.CLOUD_STEPS:
        try:
            items = list(client.search(
                collections=[EXS.COLLECTION], bbox=list(bbox),
                datetime=f"{year}-01-01/{year}-12-31",
                query={"eo:cloud_cover": {"lt": limit}}, max_items=400).items())
        except Exception:
            items = []
        if items:
            break
    by_tile = defaultdict(list)
    for it in items:
        tile = (it.properties.get("grid:code")
                or it.properties.get("s2:mgrs_tile") or "single")
        by_tile[tile].append(it)
    chosen = []
    for tile_items in by_tile.values():
        chosen.extend(EXS.pick_seasonal(tile_items, per_year))
    return chosen


def _read_scene_onto(item, band, geobox, cache_dir):
    """One band of one scene, reprojected onto the target geobox (NaN outside)."""
    href = item.assets[band].href
    dst = np.full((geobox.shape.y, geobox.shape.x), np.nan, "float32")
    try:
        with rasterio.open(href) as src:
            left, bottom, right, top = geobox.extent.boundingbox
            bounds = rasterio.warp.transform_bounds(
                f"EPSG:{geobox.crs.epsg}", src.crs, left, bottom, right, top)
            window = from_bounds(*bounds, transform=src.transform)
            window = window.round_offsets().round_lengths()
            # Pad by a tile so the resample has neighbours at the edges.
            window = Window(window.col_off - 8, window.row_off - 8,
                            window.width + 16, window.height + 16)
            arr = src.read(1, window=window, boundless=True, fill_value=0)
            if arr.size == 0:
                return dst
            src_transform = src.window_transform(window)
            src_crs = src.crs
    except Exception:
        return dst
    rasterio.warp.reproject(
        source=arr.astype("float32"), destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=geobox.transform,
        dst_crs=RioCRS.from_epsg(geobox.crs.epsg),
        src_nodata=0, dst_nodata=np.nan,
        resampling=rasterio.warp.Resampling.nearest)
    return dst


def s2_composite(bbox, geobox, year, per_year, cache_dir):
    """(4, H, W) median VNIR composite on the geobox, SCL-masked. NaN = no data."""
    client = EXS._client()
    items = _seasonal_scenes(client, bbox, year, per_year)
    if not items:
        return np.full((len(EXS.BANDS), geobox.shape.y, geobox.shape.x),
                       np.nan, "float32")
    stack = []
    for item in items:
        scl = _read_scene_onto(item, EXS.SCL_BAND, geobox, cache_dir)
        clear = np.isin(np.nan_to_num(scl, nan=0).astype("int16"), EXS.SCL_CLEAR)
        if not clear.any():
            continue
        bands = []
        for band in EXS.BANDS:
            arr = _read_scene_onto(item, band, geobox, cache_dir)
            arr[~clear] = np.nan
            bands.append(arr)
        stack.append(np.stack(bands))
    if not stack:
        return np.full((len(EXS.BANDS), geobox.shape.y, geobox.shape.x),
                       np.nan, "float32")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(np.stack(stack), axis=0).astype("float32")


# --------------------------------------------------------------------------
# Per-pixel features -- the sliding-window twin of build_s2_features
# --------------------------------------------------------------------------
def _nan_window(arr, size, want_std=False):
    """NaN-aware windowed mean (and std), matching nanmean/nanstd on a block."""
    from scipy import ndimage

    finite = np.isfinite(arr)
    filled = np.where(finite, arr, 0.0).astype("float64")
    count = ndimage.uniform_filter(finite.astype("float64"), size, mode="nearest")
    total = ndimage.uniform_filter(filled, size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1e-12), np.nan)
    if not want_std:
        return mean, None
    sq = ndimage.uniform_filter(np.where(finite, arr.astype("float64") ** 2, 0.0),
                                size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        msq = np.where(count > 0, sq / np.maximum(count, 1e-12), np.nan)
        var = np.maximum(msq - mean ** 2, 0.0)
    return mean, np.sqrt(var)


def _sobel_raster(arr):
    """Sobel magnitude per pixel, same kernel as build_s2_features._sobel_centre.

    NaN is left to propagate rather than filled, because the training path is
    plain arithmetic over the 3x3 centre block and therefore returns NaN when any
    neighbour is absent. Filling with zero would invent a strong false edge at
    every cloud margin -- and, worse, would differ from what the model was fitted
    on. ``_prepare`` maps the resulting non-finite value to the column mean, which
    is the behaviour the training rows already had.
    """
    from scipy import ndimage

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], "float64")
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], "float64")
    values = arr.astype("float64")
    gx = ndimage.convolve(values, kx, mode="nearest")
    gy = ndimage.convolve(values, ky, mode="nearest")
    return np.sqrt(gx ** 2 + gy ** 2)


def raster_features(patch, year):
    """Every S2 feature family for a (4, H, W) composite, as {column: (H, W)}.

    Mirrors ``build_s2_features.features_for_year`` statistic for statistic; the
    difference is only that each is evaluated at every pixel rather than at a
    patch centre. Verified against the training path by ``--self-check``.
    """
    cube = BSF._channels(patch[None])[0]  # (C, H, W)
    out: dict[str, np.ndarray] = {}
    for ci, name in enumerate(BSF.CHANNELS):
        out[f"S2c_{name}_{year}"] = cube[ci].astype("float64")

    means = {}
    for size in BSF.SCALES:
        means[size] = {}
        for ci, name in enumerate(BSF.CHANNELS):
            mean, std = _nan_window(cube[ci], size, want_std=True)
            out[f"S2m{size}_{name}_{year}"] = mean
            out[f"S2s{size}_{name}_{year}"] = std
            means[size][name] = mean
    for ci, name in enumerate(BSF.CHANNELS):
        out[f"S2lc_{name}_{year}"] = out[f"S2c_{name}_{year}"] - means[9][name]
        out[f"S2g_{name}_{year}"] = _sobel_raster(cube[ci])

    # Built fraction must copy the training path's NaN convention exactly, and
    # that convention is a quirk worth stating: `np.nanmean(block < CUT)` first
    # evaluates the comparison, and `NaN < CUT` is **False**, so an absent pixel
    # is counted as *not built* and still occupies the denominator. `nanmean` then
    # has no NaN left to skip. Excluding absent pixels instead -- the more
    # defensible statistic in isolation -- shifts the feature by up to 0.32 in a
    # [0, 1] range and would serve a different feature than the one fitted. The
    # self-check catches precisely this.
    from scipy import ndimage

    ndvi = cube[BSF.CHANNELS.index("ndvi")]
    built = np.where(np.isfinite(ndvi), ndvi < BSF.NDVI_VEG_CUT, False)
    built = built.astype("float64")
    for w in (3, 5, 9, 25, 64):
        # An even window has no true centre, and the two implementations break
        # the tie differently: the training block is [lo, lo+w) with
        # lo = (H - w) // 2, i.e. [centre-31, centre+32] for w=64, while scipy's
        # default is [centre-32, centre+31]. origin=-1 shifts scipy's window onto
        # the training convention. Odd windows are unambiguous and need no shift.
        origin = -1 if w % 2 == 0 else 0
        out[f"S2bf{w}_{year}"] = ndimage.uniform_filter(
            built, w, mode="nearest", origin=origin)
    return out


def s2_feature_stack(comp18, comp24, columns):
    """(H*W, len(columns)) in the model's column order, from two composites."""
    per_name = raster_features(comp18, YEARS[0])
    per_name.update(raster_features(comp24, YEARS[1]))
    last = {k: v for k, v in per_name.items() if k.endswith(f"_{YEARS[1]}")}
    for key, values in last.items():
        stem = key[: -len(str(YEARS[1]))]
        other = f"{stem}{YEARS[0]}"
        if other in per_name:
            per_name[f"{stem}diff"] = values - per_name[other]
    missing = [c for c in columns if c not in per_name]
    if missing:
        raise ValueError(f"S2 columns not reconstructable: {missing[:5]}")
    shape = per_name[columns[0]].shape
    stack = np.stack([per_name[c] for c in columns], axis=-1)
    return stack.reshape(shape[0] * shape[1], len(columns)).astype("float32")


def self_check(shard_dir, s2_path, n=24, tol=1e-3):
    """Run the raster path over stored plot patches; compare centres to training.

    The strongest available guard against train/serve skew: same plots, same
    numbers, two independent implementations. Aborts the run if they disagree.
    """
    patches, plotid, years = BSF.load_shards(shard_dir)
    truth = pd.read_parquet(s2_path).set_index("PLOTID")
    take = min(n, len(plotid))
    cols = [c for c in truth.columns
            if c.startswith(("S2c_", "S2m", "S2s", "S2lc_", "S2g_", "S2bf"))
            and not c.endswith("_diff")]
    worst, worst_col = 0.0, None
    for i in range(take):
        pid = plotid[i]
        if pid not in truth.index:
            continue
        got = {}
        for yi, year in enumerate(years):
            got.update(raster_features(patches[i, yi], year))
        # Must match build_s2_features._centre_block(cube, 1), which lands on
        # (h - 1) // 2 = 31 for a 64-wide patch, NOT h // 2 = 32. Every windowed
        # family is centred on that same index, so reading 32 here compares two
        # different pixels and manufactures a disagreement that is not real.
        centre = (patches.shape[-1] - 1) // 2
        for col in cols:
            if col not in got:
                continue
            a = float(got[col][centre, centre])
            b = float(truth.loc[pid, col])
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            denom = max(abs(b), 1.0)
            err = abs(a - b) / denom
            if err > worst:
                worst, worst_col = err, col
    return worst, worst_col


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def _file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _source_fingerprint(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path), "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def _model_code_fingerprints() -> dict:
    src_dir = Path(__file__).resolve().parent
    return {
        name: _source_fingerprint(src_dir / name)
        for name in ("infer_s2.py", "model_zoo.py", "twotower_lab.py")
    }


def _cache_metadata(name: str, spec: dict, s2_path: Path, seed: int, n_seeds: int) -> dict:
    return {
        "recipe": name,
        "seed": seed,
        "n_seeds": n_seeds,
        "input": _file_fingerprint(DEFAULT_INPUT),
        "s2": _file_fingerprint(s2_path),
        "columns": spec["columns"],
        "kwargs": spec["kwargs"],
        "deploy": spec.get("deploy"),
        "mc": spec.get("mc", False),
        "code": _model_code_fingerprints(),
    }


def _cache_path(cache_dir: Path, metadata: dict) -> Path:
    key = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return cache_dir / f"{metadata['recipe']}_seed{metadata['seed']}_n{metadata['n_seeds']}_{key}.pt"


def _move_loaded_models(entry: dict) -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for model in entry["models"]:
        model.device = device
        for module in getattr(model, "_modules_", ()):  # torch modules
            module.to(device)
        for attr, value in list(vars(model).items()):
            if isinstance(value, torch.Tensor):
                setattr(model, attr, value.to(device))


_MODEL_INIT_KEYS = (
    "arch", "loss", "epochs", "level_weights", "hidden", "gamma", "head",
    "noise_rate", "ssl", "ssl_weight", "ssl_threshold", "seed", "batch_size",
    "early_stop", "val_fraction", "patience", "mixup_alpha", "sampler", "gh_m",
    "n_experts", "moe_k", "moe_aux", "expert_dim", "noise_std", "noise_schedule",
    "noise_period", "noise_sites", "noise_gradscale", "aef_columns", "tess_columns",
    "mask_column", "modality_dropout", "tower_dim", "aef_mask_column", "fusion",
    "tess_gate", "dropout_tess", "tess_width", "align_weight", "align_temperature",
    "distill_weight", "distill_temperature", "endpoint_weight",
    # The siamese block. Absent from this tuple a cached model is REBUILT as a
    # flat two-tower and the state_dict load then fails on key names -- loudly,
    # but only on the second run of a recipe, since the first writes the cache
    # rather than reading it. Every siam_* recipe here needs them.
    "aef_siam", "siam_dim", "siam_combine", "siam_year_adapter",
    "siam_crfe", "siam_pyramid",
    "siam_cos_weight", "siam_cos_margin",
    "siam_barlow_weight", "siam_barlow_lambda",
    "siam_columns_18", "siam_columns_24", "siam_extra_columns",
)

_MODEL_LEARNED_ATTRS = (
    "mu", "sd", "mu_a", "sd_a", "mu_t", "sd_t", "year_columns",
    "fine_classes_", "merged_classes_", "gate_classes_", "base_classes_",
    "from_idx_", "to_idx_", "merged_from_idx_", "merged_to_idx_", "state_classes_",
)


def _module_state_dicts(model) -> list[dict]:
    return [
        {key: value.detach().cpu() for key, value in module.state_dict().items()}
        for module in model._modules_
    ]


def _serialise_model(model) -> dict:
    if model.arch == "gru":
        raise ValueError("infer_s2 model cache does not support arch='gru'")
    state = {
        "columns": list(model.columns),
        "init": {key: getattr(model, key) for key in _MODEL_INIT_KEYS},
        "learned": {
            key: getattr(model, key)
            for key in _MODEL_LEARNED_ATTRS
            if hasattr(model, key)
        },
        "input_dim": (len(model.aef_columns) + len(model.tess_columns) + 2
                      if model.arch == "two_tower" else len(model.columns)),
        "modules": _module_state_dicts(model),
        "M": model._M.detach().cpu(),
        "G": model._G.detach().cpu(),
    }
    if hasattr(model, "fine_bias"):
        state["fine_bias"] = model.fine_bias.detach().cpu()
    return state


def _deserialise_model(state: dict):
    import torch

    model = HierarchicalSoftmaxNN(state["columns"], **state["init"])
    for key, value in state["learned"].items():
        setattr(model, key, value)
    model.device = "cuda" if torch.cuda.is_available() else "cpu"
    model.trunk, rep_dim = model._flat_trunk(state["input_dim"])
    model.trunk.to(model.device)
    head_modules = model._build_head(rep_dim, len(model.fine_classes_))
    model._from_idx = torch.tensor(model.from_idx_, device=model.device)
    model._to_idx = torch.tensor(model.to_idx_, device=model.device)
    model._modules_ = (model.trunk,) + head_modules
    for module, module_state in zip(model._modules_, state["modules"]):
        module.load_state_dict(module_state)
        module.to(model.device)
    if "fine_bias" in state:
        with torch.no_grad():
            model.fine_bias.copy_(state["fine_bias"].to(model.device))
    model._M = state["M"].to(model.device)
    model._G = state["G"].to(model.device)
    model._T = None
    return model


def _serialise_entry(entry: dict) -> dict:
    return {"spec": entry["spec"],
            "models": [_serialise_model(model) for model in entry["models"]]}


def _deserialise_entry(entry: dict) -> dict:
    return {"spec": entry["spec"],
            "models": [_deserialise_model(model) for model in entry["models"]]}


def _load_cached_entry(path: Path, metadata: dict):
    import torch

    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("metadata") != metadata:
        return None
    return _deserialise_entry(payload["entry"])


def _save_cached_entry(path: Path, metadata: dict, entry: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "entry": _serialise_entry(entry)}, path)


def fit_models(s2_path, seed, which, n_seeds=1, cache_dir: Path | None = None,
               use_cache: bool = True):
    """Fit each requested recipe on every labelled plot (no CV)."""
    frame, aef_cols = build_frame(DEFAULT_INPUT)
    frame = frame.copy()
    frame["aef_present"] = np.float32(1.0)
    frame, s2_stat, s2_texture, s2_patch, s2_names, s2_built = attach_s2(
        frame, s2_path)
    target = target_for_legend(frame, LEGENDS["coarse3"], 20)
    # `s2_stat` is the table's whole stat block, which now carries eleven 10 m
    # channels. `s2_full` is the published seven-channel 204 that
    # `mc_s2_drop0.7` and `s2off_deploy` were measured on and must keep meaning;
    # named subsets pick their own channels out of `s2_stat`.
    s2_full = s2_base_columns(s2_stat)
    detail = [c for c in s2_full if c not in set(s2_built)]

    recipes = {
        "baseline_aef": dict(columns=aef_cols,
                             kwargs=dict(arch="wide", loss="focal", epochs=30)),
        "aef_builtfrac": dict(columns=aef_cols + s2_built,
                              kwargs=dict(arch="wide", loss="focal", epochs=30)),
        "mc_s2_drop0.7": dict(
            columns=aef_cols + s2_full,
            kwargs=dict(arch="two_tower", loss="focal", epochs=30, tower_dim=256,
                        aef_columns=aef_cols, tess_columns=s2_full,
                        mask_column=S2_MASK, aef_mask_column="aef_present",
                        fusion="gated_mean", modality_dropout=0.5,
                        dropout_tess=0.7),
            mc=True),
        # THE DEPLOYED RECIPE. Trained with both towers, served with the detail
        # gate off -- Sentinel-2 is privileged information at training time and
        # is never touched at inference. `modality_dropout` is what makes this
        # coherent: it trains the AlphaEarth tower to stand alone on rows where
        # both modalities exist, so the served configuration is one the network
        # was explicitly fitted for rather than a degraded read of it.
        #
        # The consequence for cost is the whole point. `deploy="aef_only"` lets
        # `run_aoi` skip the Sentinel-2 composite fetch and the sliding-window
        # features altogether, and `predict` route through
        # `model.probs_aef_only`, which never builds the detail block or runs its
        # tower. Sentinel-2 becomes a one-off training cost instead of a
        # per-tile one -- see S16 in S2_DETAIL_RESEARCH.md for the price paid.
        "s2off_deploy": dict(
            columns=aef_cols + s2_full,
            kwargs=dict(arch="two_tower", loss="focal", epochs=30, tower_dim=256,
                        aef_columns=aef_cols, tess_columns=s2_full,
                        mask_column=S2_MASK, aef_mask_column="aef_present",
                        fusion="gated_mean", modality_dropout=MODALITY_DROPOUT,
                        dropout_tess=0.7),
            deploy="aef_only"),
        # S17: the same recipe with the detail tower cut from 204 columns to 15.
        # `optimise_s2off.py`, 15 seeds, gate-off read: every Sentinel-2 subset
        # tried -- built fraction alone, centre reflectance alone, texture alone,
        # single-date, no-diff -- lands within +/-0.0005 change-F1 of the full
        # 204, against a seed spread of 0.004. The block is not carrying 204
        # columns' worth of information into the shared head; it is carrying at
        # most one family's.
        #
        # Built fraction is the family to keep, and not only because it is the
        # smallest: it is the only subset that *improves* the two built-up
        # metrics the user judges the Oslo map on (art_stable_recall 0.6420 ->
        # 0.6539, art->veg 0.1921 -> 0.1814) while tying change-F1 (0.6555 ->
        # 0.6553) and macro-F1 (0.6935 both). That is the same lever
        # `aef_builtfrac` found flat, arriving here through the privileged tower.
        #
        # What it saves is upstream, since serving already skips Sentinel-2
        # entirely: the windowed mean/std/contrast/gradient families are never
        # computed, and the extraction only needs red and NIR (the NDVI cut) plus
        # SCL rather than all four VNIR bands -- 3 windowed COG reads per scene
        # instead of 5, on the one stage of this project that is measured in
        # hours.
        #
        # It is NOT a drop-in. On Oslo it carries 17.4% fewer change pixels than
        # `s2off_deploy` (14,824 vs 17,941, merged2 agreeing on 97.9%) while the
        # plots call the two indistinguishable on change (recall 0.7254 vs
        # 0.7296). That is the same unadjudicable situation as T2 -- no labelled
        # plot lies inside the AOI -- so judge it on the map before switching.
        "s2off_slim": dict(
            columns=aef_cols + s2_built,
            kwargs=dict(arch="two_tower", loss="focal", epochs=30, tower_dim=256,
                        aef_columns=aef_cols, tess_columns=s2_built,
                        mask_column=S2_MASK, aef_mask_column="aef_present",
                        fusion="gated_mean", modality_dropout=MODALITY_DROPOUT,
                        dropout_tess=0.7),
            deploy="aef_only"),
    }

    # S18: the same gate-off recipe over each named detail-tower subset, so the
    # reporting question ("204 engineered features" is not a sentence anyone
    # wants to defend) can be answered on map evidence rather than on taste.
    # Every one of these is `s2off_deploy` with a different `tess_columns`; the
    # AlphaEarth tower, the training schedule and the served code path are
    # identical, so a difference between two of these maps is the feature set
    # and nothing else.
    #
    # `s2off_centre_m3s3_bf` (78 columns) is the one to report -- selected on the
    # user's visual read of the coarse3 map, which is the right arbiter because
    # the quantitative metrics genuinely tie. A 5-seed ensemble reproduces itself
    # at only change-IoU 0.84 across disjoint seed blocks, and 57 and 78 sit
    # inside that floor, so no map metric separates them. What the numbers do say
    # is that 78 is never worse: on the coarse3 read it carries more structure
    # *and* better boundary alignment than the full 204 (edge 0.0970 vs 0.0949,
    # align 1.6071 vs 1.6013), and it matches the full block to the fourth
    # decimal on every plot metric. Everything below 57 is separable and worse;
    # `bf` alone (s2off_slim, 15 cols) is below the floor at 0.77. See S18.
    for subset in S2_SUBSETS:
        recipes[f"s2off_{subset}"] = dict(
            columns=aef_cols + s2_subset_columns(s2_stat, subset),
            kwargs=dict(arch="two_tower", loss="focal", epochs=30, tower_dim=256,
                        aef_columns=aef_cols,
                        tess_columns=s2_subset_columns(s2_stat, subset),
                        mask_column=S2_MASK, aef_mask_column="aef_present",
                        fusion="gated_mean", modality_dropout=MODALITY_DROPOUT,
                        dropout_tess=0.7),
            deploy="aef_only")

    # N8b (docs/research/SIAMESE_RESEARCH.md). `s2off_centre_m3s3_bf` with its
    # AlphaEarth tower replaced by a SHARED ENDPOINT ENCODER -- one encoder
    # applied to the 2018 and 2024 blocks, head on
    # [z18, z24, z24-z18, |z24-z18|, cos] -- plus the gate-supervised cosine
    # objective that pulls a stable plot's two embeddings together and pushes a
    # change plot's apart.
    #
    # Everything about the deployment is unchanged: same 78-column privileged
    # detail tower, same mask gating, same modality dropout, `deploy="aef_only"`,
    # so Sentinel-2 is still never read at inference and this serves at exactly
    # the deployed model's cost. The swap is entirely inside `aef_tower`, which
    # is the one module `probs_aef_only_matrix` runs, so the fast path stays
    # exact (`_assert_aef_only_ok` passes: two_tower / gated_mean / mask gate).
    #
    # On plots, 5 seeds, blocked CV: change-F1 0.6644 +/-0.0024 against 0.6568,
    # macro-F1 0.7067 against 0.6943, and every commissioned transition up
    # except Nature->Artificial, which plain `siam_cos` reads better. It is
    # WORSE on stable built-up returned as vegetation (0.225 vs 0.196) and
    # better on stable built-up returned as spurious change (0.129 vs 0.167).
    #
    # This is NOT a replacement for the deployed model. CLAUDE.md settles that
    # choice on the user's visual read of the map, and no map evidence exists
    # for this recipe yet -- that is what running it produces.
    recipes["siam_s2off_cos"] = dict(
        columns=aef_cols + s2_subset_columns(s2_stat, "centre_m3s3_bf"),
        kwargs=dict(arch="two_tower", loss="focal", epochs=30, tower_dim=256,
                    aef_columns=aef_cols,
                    tess_columns=s2_subset_columns(s2_stat, "centre_m3s3_bf"),
                    mask_column=S2_MASK, aef_mask_column="aef_present",
                    fusion="gated_mean", modality_dropout=MODALITY_DROPOUT,
                    dropout_tess=0.7, aef_siam=True, siam_dim=128,
                    siam_combine="conc", siam_cos_weight=0.3,
                    siam_cos_margin=0.3),
        deploy="aef_only")

    # Q7b (docs/research/SIAMESE_RESEARCH.md, section Q). `siam_s2off_cos` with
    # the CRFE module from Zhang et al.'s burned-area Swin network on its head
    # block: the elementwise SUM of the two endpoint embeddings appended
    # alongside their difference, and a squeeze-and-excitation channel gate over
    # the assembled block. Their spatial-attention branch has no form at a plot
    # and is not here.
    #
    # Everything about the deployment is again unchanged -- both modules live
    # INSIDE `aef_tower`, which is the one module `probs_aef_only_matrix` runs,
    # so `deploy="aef_only"` stays exact and no Sentinel-2 is read at inference.
    # Serving cost is one SE bottleneck (128+512 -> ~46 -> 641) per pixel over
    # `siam_s2off_cos`.
    #
    # Why it is here: it is the only thing in four sections of ledger to move
    # STABLE BUILT-UP, which CLAUDE.md records as the frontier the deployed
    # model still owns. On plots at 15 seeds against `siam_s2off_cos`:
    # `art_stable_recall` 0.669 vs 0.644, stable built-up read as vegetation
    # 0.208 vs 0.230, spurious change on built-up 0.123 vs 0.126, for
    # change-F1 -0.0010 / macro -0.0010 / focus -0.0019, all inside seed noise.
    # Against the DEPLOYED model on the same 15 seeds: artStab +0.027,
    # spurious change on built-up -0.041 (0.123 vs 0.164), `veg_stable_as_art`
    # 0.0346 vs 0.0354 -- so it is not Artificial flooding -- and it keeps
    # +0.004 change-F1 / +0.009 macro / +0.013 focus. It costs 0.014 on
    # `Artificial -> Nature`, the recovery class.
    #
    # NOT a replacement for the deployed model; CLAUDE.md settles that on the
    # user's visual read, and this recipe exists so there is a map to read.
    recipes["siam_s2off_crfe_full"] = dict(
        columns=aef_cols + s2_subset_columns(s2_stat, "centre_m3s3_bf"),
        kwargs=dict(arch="two_tower", loss="focal", epochs=30, tower_dim=256,
                    aef_columns=aef_cols,
                    tess_columns=s2_subset_columns(s2_stat, "centre_m3s3_bf"),
                    mask_column=S2_MASK, aef_mask_column="aef_present",
                    fusion="gated_mean", modality_dropout=MODALITY_DROPOUT,
                    dropout_tess=0.7, aef_siam=True, siam_dim=128,
                    siam_combine="conc", siam_crfe="full",
                    siam_cos_weight=0.3, siam_cos_margin=0.3),
        deploy="aef_only")

    # W1 / W1b (docs/research/SIAMESE_RESEARCH.md, section W). `siam_s2off_cos`
    # with CLASS-BALANCED focal in place of plain focal -- Cui et al. (2019)
    # effective-number weights (beta 0.999, normalised to mean 1) multiplying the
    # same focal modulation. One kwarg. Nothing about the deployment changes:
    # same 78-column privileged detail tower, same mask gating, same modality
    # dropout, `deploy="aef_only"`, so Sentinel-2 is still never read at
    # inference and both serve at exactly the deployed model's cost. The change
    # is entirely in the training objective and leaves the served graph
    # identical, so `probs_aef_only_matrix` stays exact.
    #
    # `cb_levels="fine"` (W1b) keeps the weights off the 2-class gate and
    # merged2 and puts them only on the nine coarse3 classes. The pair separates
    # the mechanism, and the separation is the reason both are here: on plots at
    # 5 seeds the MERGED2/GATE weights are what move stable built-up (W1
    # `art_stable_as_veg` 0.151, W1b 0.187, `siam_s2off_cos` 0.225) while the
    # FINE weights are what break `Artificial -> Cropland` off zero.
    #
    # Why there is a map to read at all. W1's 0.151 is the largest move on
    # stable-built-up-read-as-vegetation in the ledger -- past the DEPLOYED
    # model's 0.196 and past the cost-gated deployed model's 0.165, a frontier
    # CLAUDE.md records as still the deployed model's after four sections -- and
    # it is the only instrument that moves it at all, because O3, V1 and every
    # conformal read re-score the coarse3 arg-max and leave every merged2 metric
    # bit-identical. It is NOT free: change-F1 0.6499 against 0.6644, macro-F1
    # 0.6962 against 0.7067, change precision 0.559 against 0.651, and
    # `veg_stable_as_art` 0.029 -> 0.041, which is the false-built-up direction
    # the user's map judgement weights most.
    #
    # That last trade is exactly what no plot metric can adjudicate here -- Oslo
    # has zero labelled plots inside the AOI (G3/G4) -- and it is why these two
    # recipes exist. Read the change precision loss as pixels before believing
    # the built-up gain. NOT a replacement for the deployed model; CLAUDE.md
    # settles that on the user's visual read.
    for _w_name, _w_extra in (("siam_s2off_cb", {}),
                              ("siam_s2off_cb_fine", {"cb_levels": "fine"})):
        recipes[_w_name] = dict(
            columns=aef_cols + s2_subset_columns(s2_stat, "centre_m3s3_bf"),
            kwargs=dict(arch="two_tower", loss="cb_focal", epochs=30,
                        tower_dim=256, aef_columns=aef_cols,
                        tess_columns=s2_subset_columns(s2_stat, "centre_m3s3_bf"),
                        mask_column=S2_MASK, aef_mask_column="aef_present",
                        fusion="gated_mean", modality_dropout=MODALITY_DROPOUT,
                        dropout_tess=0.7, aef_siam=True, siam_dim=128,
                        siam_combine="conc", siam_cos_weight=0.3,
                        siam_cos_margin=0.3, **_w_extra),
            deploy="aef_only")

    # O3 (docs/research/SIAMESE_RESEARCH.md, section O). The AlphaEarth-ONLY
    # shared-endpoint siamese with the cosine objective -- `siam_cos`, N2 -- read
    # through the coarse3 decision-cost gate.
    #
    # No Sentinel-2 anywhere: not at training and not at inference. This model
    # never had a detail tower, so unlike `s2off_*` there is no gate to force off
    # and the AlphaEarth-only property is structural rather than a serving
    # choice. It cannot use `probs_aef_only_matrix` (that fast path is proved for
    # `two_tower` + `gated_mean` + mask gate, and asserts as much), so it runs the
    # DataFrame path -- slower per pixel, and still no composite fetch, because
    # `needs_s2` keys off whether any requested model reads a non-AlphaEarth
    # column.
    #
    # `c3_costs` names the shipped decision-cost vector (fit_coarse3_costs.py).
    # On the plots at 5 seeds this gate takes `focus_macro_f1` 0.3815 -> 0.4412
    # under nested CV and `Artificial -> Cropland` 0.000 -> 0.2727, with every
    # merged2 aggregate unchanged BY CONSTRUCTION -- it re-reads only the coarse3
    # arg-max. The fitted vector is a single multiplier on that one class, so
    # the `*_merged2.tif` map is bit-identical to the ungated model's and only
    # `*_coarse3_gated.tif` differs.
    recipes["siam_cos"] = dict(
        columns=aef_cols,
        kwargs=dict(
            arch="siamese", loss="focal", epochs=30, tower_dim=256,
            siam_columns_18=sorted(c for c in aef_cols if c.endswith("_2018")),
            siam_columns_24=sorted(c for c in aef_cols if c.endswith("_2024")),
            siam_extra_columns=sorted(c for c in aef_cols if c.endswith("_diff")),
            siam_dim=128, siam_combine="conc",
            siam_cos_weight=0.3, siam_cos_margin=0.3),
        c3_costs="base_siam_cos_fine")

    # P7e/P7f (docs/research/SIAMESE_RESEARCH.md, section P7). `siam_s2off_cos`
    # whose shared endpoint encoder is PRETRAINED on single-date land-cover
    # states -- 30 epochs of g(f(x)) -> {Nature, Cropland, Artificial} over the
    # 13,118-unit GLanCE 2018 pool -- before the transition loss is ever seen.
    # The state head is discarded after that phase and the auxiliary term is off
    # for the whole fit, so this is a training-time-only change: the served graph
    # is `siam_s2off_cos`'s exactly, `deploy="aef_only"` stays exact, and no
    # Sentinel-2 and no GLanCE is read at inference.
    #
    # Why it is here. On plots at 5 seeds it takes `Artificial -> Cropland` from
    # 0.000 to 0.1075 and `focus_macro_f1` 0.3847 -> 0.4157 at flat change-F1
    # (0.6643 vs 0.6644), and it beats BOTH controls -- the endogenous one
    # (no new data) and a shuffled-label one that lands exactly on baseline, so
    # the effect is the labels' land-cover content and not the extra phase.
    # Read with the gate (`c3_costs`) it is 0.4383 +/-0.008 against O3's 0.4318
    # +/-0.023 on the same base: nominally the ledger's best, inside the noise.
    #
    # What the map has to answer, and the reason this recipe exists. On plots it
    # moves stable built-up (`art_stable_recall` 0.646 -> 0.658, read-as-
    # vegetation 0.225 -> 0.204) but pays `art_stable_as_change` 0.129 -> 0.138
    # -- more fabricated habitat-loss events, the wrong direction for this
    # product. Section W is the precedent: a plot-level built-up win there did
    # not survive the map's 0.5% change base rate. The `Artificial -> Cropland`
    # pixel count is the direct read on whether the rescued class is real.
    #
    # NOT a replacement for the deployed model. CLAUDE.md settles that on the
    # user's visual read; this produces something to read.
    recipes["siam_s2off_state_pre"] = dict(
        columns=aef_cols + s2_subset_columns(s2_stat, "centre_m3s3_bf"),
        kwargs=dict(arch="two_tower", loss="focal", epochs=30, tower_dim=256,
                    aef_columns=aef_cols,
                    tess_columns=s2_subset_columns(s2_stat, "centre_m3s3_bf"),
                    mask_column=S2_MASK, aef_mask_column="aef_present",
                    fusion="gated_mean", modality_dropout=MODALITY_DROPOUT,
                    dropout_tess=0.7, aef_siam=True, siam_dim=128,
                    siam_combine="conc", siam_cos_weight=0.3,
                    siam_cos_margin=0.3,
                    siam_state_pretrain=30, siam_state_source="external"),
        deploy="aef_only", state_pool=True,
        c3_costs="siam_s2off_state_pre")

    # Y3 (docs/research/STATE_PRETRAIN_RESEARCH.md section Y). `siam_s2off_state_pre`
    # with ONE change: the state-pretraining head's cross-entropy is weighted by
    # 1/class frequency. Nothing else moves -- same 78 columns, same cosine
    # objective, same 30 pretrain epochs, same `external` pool, same
    # `deploy="aef_only"`, so it serves at the deployed model's cost and the A/B
    # against `siam_s2off_state_pre` isolates the reweighting.
    #
    # It exists because of the user's read of the Oslo map: cropland has room to
    # grow, but it must take pixels from Nature and NOT from the built-up
    # classes. Section Y measured that as three numbers, and this is the only
    # arm that moved all three the right way at both fold counts -- `f1_cropland`
    # +0.0063/+0.0039, `nature_as_cropland` +0.0080/+0.0088 (where the growth
    # should come from) and `artificial_as_cropland` -0.0038/-0.0037 (where it
    # should not). Four capacity arms spending up to 3.8x the encoder parameters
    # moved the *sum* of those errors and never the ratio (Y1/Y2).
    #
    # The state-level trade is -0.0014/-0.0048 macro-F1, and P7i already showed
    # the pretraining phase itself is negative on plot change-F1 (0.6450 against
    # 0.6644 unpretrained). So this map is NOT a candidate to replace the
    # deployed model and must not be read against it -- the comparison that means
    # anything is against `siam_s2off_state_pre`, its own baseline, which is why
    # both are run together. CLAUDE.md settles the deployment on the user's
    # visual read; this produces the pair to read.
    recipes["siam_s2off_state_pre_cw"] = dict(
        columns=aef_cols + s2_subset_columns(s2_stat, "centre_m3s3_bf"),
        kwargs=dict(arch="two_tower", loss="focal", epochs=30, tower_dim=256,
                    aef_columns=aef_cols,
                    tess_columns=s2_subset_columns(s2_stat, "centre_m3s3_bf"),
                    mask_column=S2_MASK, aef_mask_column="aef_present",
                    fusion="gated_mean", modality_dropout=MODALITY_DROPOUT,
                    dropout_tess=0.7, aef_siam=True, siam_dim=128,
                    siam_combine="conc", siam_cos_weight=0.3,
                    siam_cos_margin=0.3,
                    siam_state_pretrain=30, siam_state_source="external",
                    siam_state_class_weight="balanced"),
        deploy="aef_only", state_pool=True,
        c3_costs="siam_s2off_state_pre_cw")

    # Q10f (docs/research/SIAMESE_RESEARCH.md, sections Q10-Q11). P7e above with
    # SNIIF-Net's feature-interaction gate on the shared endpoint encoder:
    # z18 <- z18 * (1 + tanh(W [z18 | z24])), one shared W used with the inputs
    # swapped for the other date, zero-initialised. It sits UPSTREAM of the
    # endpoint subtraction, so unlike section Q's CRFE gate it changes
    # z24 - z18, the cosine feature and what the pair losses read. Serving cost
    # is one 256x128 matmul per pixel per date; `deploy="aef_only"` is
    # unaffected and no Sentinel-2 is read at inference.
    #
    # Why it is here, stated at the strength the plots actually support. On 15
    # seeds it is the best arg-max `focus_macro_f1` in the section (0.4427 vs
    # P7e's 0.4170) and takes `Artificial -> Cropland` 0.115 -> 0.225 with its
    # seed spread nearly halved, at change-F1 -0.0010 and macro -0.0012, both
    # inside noise. It also moves stable built-up the right way on all three
    # rows at once -- `art_stable_recall` 0.657 -> 0.669, read-as-vegetation
    # 0.206 -> 0.199, read-as-change 0.136 -> 0.132 -- which is what P7e could
    # not do (P7e paid `art_stable_as_change` 0.129 -> 0.138) and is the single
    # reason a map of this is worth fetching.
    #
    # Two things the plots also say, and they cut the other way. Read
    # gate-to-gate against the incumbent P7f the whole gain is +0.0035, inside
    # +/-0.005 (Q10h). And its own control -- the same gate reading each date
    # twice instead of the pair -- reproduces 63% of it and BEATS it on
    # built-up, so the cross-branch interaction is not the mechanism (Q10g).
    #
    # NOT a replacement for the deployed model, and not a candidate on plot
    # metrics alone. `veg_stable_as_art` rises 0.0328 -> 0.0346 (still under the
    # deployed 0.0354) and section W's lesson stands: a plot-level built-up gain
    # measured at the plots' ~25% change base rate has twice failed to survive
    # the map's 0.5%. This produces the raster that says which it is here.
    recipes["siam_s2off_state_pre_fiim"] = dict(
        columns=aef_cols + s2_subset_columns(s2_stat, "centre_m3s3_bf"),
        kwargs=dict(arch="two_tower", loss="focal", epochs=30, tower_dim=256,
                    aef_columns=aef_cols,
                    tess_columns=s2_subset_columns(s2_stat, "centre_m3s3_bf"),
                    mask_column=S2_MASK, aef_mask_column="aef_present",
                    fusion="gated_mean", modality_dropout=MODALITY_DROPOUT,
                    dropout_tess=0.7, aef_siam=True, siam_dim=128,
                    siam_combine="conc", siam_cos_weight=0.3,
                    siam_cos_margin=0.3, siam_fiim="cross",
                    siam_state_pretrain=30, siam_state_source="external"),
        deploy="aef_only", state_pool=True,
        c3_costs="siam_s2off_state_pre_fiim")

    out = {}
    for name in which:
        spec = recipes[name]
        cache_file = None
        metadata = None
        if use_cache and cache_dir is not None:
            metadata = _cache_metadata(name, spec, Path(s2_path), seed, n_seeds)
            cache_file = _cache_path(cache_dir, metadata)
            cached = _load_cached_entry(cache_file, metadata)
            if cached is not None:
                print(f"  loaded {name} seed {seed}..{seed + n_seeds - 1} "
                      f"from {cache_file}", flush=True)
                out[name] = cached
                continue
        # Seed ensembling (S12/S12b). Averaging several torch seeds' posteriors
        # is worth +0.0038 change-F1 and +0.0032 macro-F1 on this recipe and
        # shrinks the seed spread 2.6x (0.0052 -> 0.002). The variance is the
        # real argument for a *map*: a one-seed raster is a single draw from a
        # distribution wide enough to move visible pixels between runs.
        models = []
        for offset in range(n_seeds):
            model = HierarchicalSoftmaxNN(spec["columns"], seed=seed + offset,
                                          **spec["kwargs"])
            print(f"  fitting {name} seed {seed + offset} "
                  f"({len(spec['columns'])} cols) ...", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # A state-pretrained recipe gets the WHOLE pool here, where the
                # lab cuts it to each fold's training blocks: there is no
                # held-out block at deployment, so the split that protects the
                # blocked-CV estimate has nothing to protect and would only
                # discard labels. No labelled plot is at risk either way -- zero
                # pool points fall within 100 m of one (N14a).
                model.fit(frame, target.to_numpy(),
                          state_frame=_state_pool() if spec.get("state_pool")
                          else None)
            models.append(model)
        entry = {"models": models, "spec": spec}
        if cache_file is not None and metadata is not None:
            _save_cached_entry(cache_file, metadata, entry)
            print(f"  saved {name} seed {seed}..{seed + n_seeds - 1} "
                  f"to {cache_file}", flush=True)
        out[name] = entry
    return out, aef_cols, s2_stat, s2_built, detail


def _coarse3_costs(spec: dict, fine_classes: list):
    """The shipped coarse3 decision-cost vector for a recipe, or None.

    Aligned to *this* model's coarse3 class order by NAME. The stored vector
    carries its own class list precisely so the two can be checked rather than
    assumed: a silently mis-aligned cost vector would multiply the wrong class
    and produce a map that looks entirely plausible, which is the same failure
    mode the ensemble class-order check exists to prevent.
    """
    name = spec.get("c3_costs")
    if not name:
        return None
    path = project_data_dir("analysis_results") / f"coarse3_costs__{name}.json"
    if not path.exists():
        raise SystemExit(
            f"recipe wants coarse3 costs from {path.name}, which does not exist; "
            f"run: python src/fit_coarse3_costs.py --idea {name}")
    blob = json.loads(path.read_text())
    lookup = dict(zip(blob["fine_classes"], blob["costs"]))
    missing = [c for c in fine_classes if c not in lookup]
    if missing:
        raise SystemExit(f"{path.name} has no cost for coarse3 classes {missing}")
    return np.array([lookup[c] for c in fine_classes], dtype="float64")


def predict(entry, aef_bands, s2_mat, s2_present, batch, seed, want_off=False,
            mc_sampling=False):
    """Posteriors for every pixel, at both levels of the hierarchy and both views.

    Returns ``(views, classes, fine_classes)`` where ``views`` maps a view name
    to ``(merged_probs, fine_probs)``. ``"on"`` is the deployed read; ``"off"``
    is the same model with the Sentinel-2 gate forced off (the T2
    counterfactual), and is produced only when ``want_off``.

    ``aef_bands`` is **band-major**: ``(n_aef_columns, n_pixels)``, one row per
    AlphaEarth column in the model's column order. ``s2_mat`` stays pixel-major
    ``(n_pixels, n_s2_columns)`` because that is what ``s2_feature_stack``
    produces and only the non-deployed models read it. The asymmetry is
    deliberate: a raster arrives band by band, so band-major is the layout it is
    already in, and materialising the pixel-major transpose cost 11.8 s and a
    second 2.3 GB allocation per AOI -- a strided scatter over 192 columns that
    is pure memory-system punishment. Both consumers want a contiguous slice out
    of this layout instead: the deployed path takes ``[:, start:stop]`` and
    transposes 154 MB on the device, and the DataFrame path takes ``[i, start:stop]``,
    which is a contiguous row rather than the strided column it used to copy.

    Two levels for one pass
    -----------------------
    ``_probs`` returns ``(p_fine, p_merged)`` with ``p_merged = p_fine @ M``, so
    keeping the coarse3 posterior costs an accumulator, not a second pass.

    The MC gate is marginalised exactly, not sampled
    -----------------------------------------------
    The deployed recipe reads the detail tower through a *stochastic* gate --
    each pass keeps Sentinel-2 with probability ``MC_KEEP`` -- because a
    deterministic gate suppresses change (S2_DETAIL_RESEARCH.md, iteration 4).
    It was implemented as 16 sampled passes, but sampling is unnecessary here:
    at inference the modules are in ``eval()`` mode, so torch dropout is off and
    BatchNorm uses running statistics, leaving every pixel independent of every
    other. The only stochastic element is ``S2_MASK``, and it is **binary**. So
    the 16-pass average is a Monte-Carlo estimate of a two-valued expectation

        E[p] = keep * f(mask=1) + (1 - keep) * f(mask=0)   (S2 present)
        E[p] =                                 f(mask=0)   (S2 absent)

    which is computable in **two** forward passes, exactly. That is 8x less work
    *and* strictly more accurate: 16 Bernoulli draws put the realised keep-rate
    anywhere in +/-0.5 of 0.5, and that sampling noise moved visible pixels
    between runs. It also makes the map deterministic -- no RNG, no seed.

    And the ``mask=0`` branch **is** the counterfactual view, so the "gate off"
    map that used to cost a second full sweep now falls out of this one.

    ``mc_sampling=True`` restores the original sampled loop. Keep it: it is how
    the equivalence above is verified against the maps already published, and it
    is the only way to reproduce those runs bit for bit.
    """
    import pandas as pd
    import torch

    members = entry["models"]
    spec = entry["spec"]
    columns = spec["columns"]
    n_aef, n = aef_bands.shape
    classes = members[0].merged_classes_
    fine_classes = members[0].fine_classes_
    # Averaging posteriors across seeds is only meaningful if every member orders
    # its classes identically; otherwise column k means a different transition in
    # different members and the sum silently permutes classes. Both levels must
    # be checked -- the fine vector is the one with nine ways to go wrong.
    if any(list(m.merged_classes_) != list(classes) for m in members):
        raise SystemExit("seed members disagree on class order; refusing to average")
    if any(list(m.fine_classes_) != list(fine_classes) for m in members):
        raise SystemExit("seed members disagree on fine class order; refusing to average")

    mc = bool(spec.get("mc"))
    aef_only = spec.get("deploy") == "aef_only"
    if aef_only:
        # The deployed read. Only the AlphaEarth block is materialised and only
        # its tower runs; `probs_aef_only` is bit-identical to `_probs` with the
        # gate zeroed (tests/test_s2off_fastpath.py). There is no "off" view to
        # produce because this read *is* the gate-off view -- the counterfactual
        # would be comparing the map against itself.
        columns = [c for c in columns if c in set(members[0].aef_columns)]
        want_off = False
        # `aef_bands` is handed straight to the tower, so its rows must already
        # BE the model's AlphaEarth columns in the model's order. Checked rather
        # than assumed: a mismatch would permute the embedding channels and
        # produce a map that looks entirely plausible.
        if list(columns) != list(members[0].aef_columns):
            raise SystemExit("aef_only: band matrix rows are not the model's "
                             "aef_columns in order")
    # The two blocks are split positionally inside the batch loop, so their row
    # and column counts have to account for every column the model reads.
    if n_aef + s2_mat.shape[1] != len(columns):
        raise SystemExit(f"aef_bands has {n_aef} rows and s2_mat "
                         f"{s2_mat.shape[1]} columns, model reads {len(columns)}")
    names = ["on"] + (["off"] if want_off else [])
    views = {v: (np.zeros((n, len(classes)), "float32"),
                 np.zeros((n, len(fine_classes)), "float32")) for v in names}
    rng = np.random.default_rng(seed)

    def _probs_at(model, chunk, mask):
        chunk[S2_MASK] = mask
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model._probs(chunk)  # (fine, merged)

    for start in range(0, n, batch):
        stop = min(start + batch, n)
        if aef_only:
            # No DataFrame at all on the deployed path. The raw AlphaEarth block
            # goes to the device once and every ensemble member standardises it
            # there; pandas construction plus five numpy standardisations were
            # 84% of this loop's time and none of it was the model. The transpose
            # back to pixel-major rides along as a device-side view.
            chunk = torch.as_tensor(
                np.ascontiguousarray(aef_bands[:, start:stop])
            ).to(members[0].device, torch.float32, non_blocking=True).T
        else:
            block = {}
            for i, col in enumerate(columns):
                block[col] = (aef_bands[i, start:stop] if i < n_aef
                              else s2_mat[start:stop, i - n_aef])
            # One DataFrame per batch, not one per pass. `_probs` re-runs
            # `_prepare` (396 columns out of pandas, standardised) on every call,
            # and that -- not the matmuls -- was most of the old runtime at 80
            # calls per batch.
            chunk = pd.DataFrame(block)
            chunk["aef_present"] = 1.0
        present = s2_present[start:stop].astype("float32")
        off = np.zeros_like(present)
        if not aef_only:
            chunk[S2_MASK] = present
        # Average over seeds in one accumulator. Seed variance (which model you
        # happened to fit) and gate variance (how much the detail tower is
        # trusted) are different things; the deployed read integrates over both,
        # the second one analytically.
        acc = {v: [np.zeros((stop - start, len(classes)), "float64"),
                   np.zeros((stop - start, len(fine_classes)), "float64")]
               for v in names}
        for model in members:
            if aef_only:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fine, merged = model.probs_aef_only_matrix(chunk)
                acc["on"][0] += merged
                acc["on"][1] += fine
                continue
            if mc and mc_sampling:
                for _ in range(MC_PASSES):
                    keep = rng.random(stop - start) >= (1.0 - MC_KEEP)
                    fine, merged = _probs_at(
                        model, chunk,
                        np.where(present > 0.5, keep.astype("float32"), 0.0))
                    acc["on"][0] += merged / MC_PASSES
                    acc["on"][1] += fine / MC_PASSES
                if want_off:
                    fine, merged = _probs_at(model, chunk, off)
                    acc["off"][0] += merged
                    acc["off"][1] += fine
                continue
            fine0, merged0 = _probs_at(model, chunk, off)
            if mc:
                fine1, merged1 = _probs_at(model, chunk, present)
                w = (present > 0.5).astype("float64")[:, None] * MC_KEEP
                acc["on"][0] += w * merged1 + (1.0 - w) * merged0
                acc["on"][1] += w * fine1 + (1.0 - w) * fine0
            else:
                # A flat recipe never reads the mask column, so one pass is the
                # whole answer and `off` is the same numbers (see T2b).
                fine1, merged1 = _probs_at(model, chunk, present)
                acc["on"][0] += merged1
                acc["on"][1] += fine1
            if want_off:
                acc["off"][0] += merged0
                acc["off"][1] += fine0
        for v in names:
            views[v][0][start:stop] = (acc[v][0] / len(members)).astype("float32")
            views[v][1][start:stop] = (acc[v][1] / len(members)).astype("float32")
    return views, classes, fine_classes


# --------------------------------------------------------------------------
def stack_aef_bands(emb18, emb24, aef_cols):
    """``(len(aef_cols), n_pixels)`` float32, one row per AlphaEarth column.

    The training columns are ``A{i}_2018``, ``A{i}_2024`` and their difference,
    which is a linear function of the other two and so is materialised here
    rather than fetched. Writing each row as one contiguous sequential store is
    the whole point: the pixel-major equivalent (``np.stack(..., -1)``) writes
    the same 2.3 GB as 192 strided scatters and takes 11.8 s against 1.2 s for
    this, on identical numbers (``tests/test_s2off_fastpath.py``).
    """
    n = emb18.shape[1] * emb18.shape[2]
    out = np.empty((len(aef_cols), n), "float32")
    at = {c: j for j, c in enumerate(aef_cols)}
    for i in range(emb18.shape[0]):
        a = emb18[i].reshape(-1)
        b = emb24[i].reshape(-1)
        out[at[f"A{i:02d}_2018"]] = a
        out[at[f"A{i:02d}_2024"]] = b
        np.subtract(b, a, out=out[at[f"A{i:02d}_diff"]])
    return out


async def run_aoi(name, bbox, index, reader, models, aef_cols, s2_stat, s2_built,
                  resolution, batch, seed, per_year, cache_dir, out_dir,
                  save_probs=False, mc_sampling=False, s2_backdrop=False):
    # Target grid in the AOI's own UTM zone, snapped to the global 10 m lattice
    # so 2018 and 2024 share pixel coordinates -- the same construction
    # infer_cities.py uses, so these maps overlay its outputs exactly.
    probe = await index.query(bbox=bbox, years=YEARS[0])
    epsg = probe[0].crs_epsg
    geobox = aoi_geobox(bbox, crs=f"EPSG:{epsg}", resolution=resolution,
                        bbox_crs="EPSG:4326")
    height, width = geobox.shape.y, geobox.shape.x
    print(f"\n[{name}] geobox {height}x{width} @ {resolution} m "
          f"(EPSG:{epsg})", flush=True)

    emb = {}
    for year in YEARS:
        emb[year] = await load_year_embeddings(index, reader, bbox, year, geobox)
    aef_bands = stack_aef_bands(emb[YEARS[0]], emb[YEARS[1]], aef_cols)
    aef_valid = np.isfinite(aef_bands).all(0)
    print(f"[{name}] AlphaEarth ready, {aef_valid.mean():.1%} of pixels valid",
          flush=True)

    # Does any requested model actually READ Sentinel-2 at inference? The
    # deployed recipe does not -- it is served with the detail gate off -- so for
    # a wide-area run the composite fetch (the one network-bound, cloud-dependent
    # stage in the pipeline) and the sliding-window features are pure waste.
    # This is the single largest saving available and it is a consequence of the
    # deployment choice, not an approximation of it.
    needs_s2 = any(
        entry["spec"].get("deploy") != "aef_only"
        and any(c not in set(aef_cols) for c in entry["spec"]["columns"])
        for entry in models.values())
    t0 = time.time()
    if needs_s2 or s2_backdrop:
        comps = {y: s2_composite(bbox, geobox, y, per_year, cache_dir)
                 for y in YEARS}
        for year, comp in comps.items():
            print(f"[{name}] S2 {year}: {np.isfinite(comp[0]).mean():.1%} pixels "
                  f"({time.time() - t0:.0f}s)", flush=True)
    else:
        comps = {}
        print(f"[{name}] no model reads Sentinel-2 at inference -- skipping the "
              f"composite fetch and features (pass --s2-backdrop for the "
              f"true-colour reference layers)", flush=True)

    # Export the composite itself: it is the visual reference for judging whether
    # a map's boundaries land on real edges, and it is the input
    # `map_detail_metrics.py --reference` needs to compute `boundary_align`.
    # Written as a plain 3-band RGB (red/green/blue) uint16 so QGIS opens it as a
    # true-colour backdrop under the class maps.
    for year, comp in comps.items():
        rgb = np.stack([comp[2], comp[1], comp[0]])  # red, green, blue
        rgb = np.nan_to_num(rgb, nan=0).clip(0, 65535).astype("uint16")
        with rasterio.open(
            out_dir / f"{name}_s2_rgb_{year}.tif", "w", driver="GTiff",
            height=height, width=width, count=3, dtype="uint16",
            crs=RioCRS.from_epsg(geobox.crs.epsg), transform=geobox.transform,
            nodata=0, compress="deflate", tiled=True, blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(rgb)
            dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)
        # NIR as a single band -- the channel that actually separates built from
        # vegetated, and the one the detail metrics take their gradient from.
        nir = np.nan_to_num(comp[3], nan=0).clip(0, 65535).astype("uint16")
        with rasterio.open(
            out_dir / f"{name}_s2_nir_{year}.tif", "w", driver="GTiff",
            height=height, width=width, count=1, dtype="uint16",
            crs=RioCRS.from_epsg(geobox.crs.epsg), transform=geobox.transform,
            nodata=0, compress="deflate", tiled=True,
        ) as dst:
            dst.write(nir, 1)

    results = {}
    timings = {"composite_and_export_s": round(time.time() - t0, 1)}
    for model_name, entry in models.items():
        columns = entry["spec"]["columns"]
        # Stage timings, because "make it efficient globally" needs a budget and
        # not an impression. At AOI scale the composite fetch and the sliding
        # windows are the standing cost; the forward passes stopped being it once
        # the gate was marginalised exactly (predict()).
        tf = time.time()
        s2_cols = [c for c in columns if c not in set(aef_cols)]
        # A gate-off model owns S2 columns but never reads them, so the feature
        # stack is not built for it at all -- the columns are not materialised
        # anywhere on this path.
        reads_s2 = s2_cols and entry["spec"].get("deploy") != "aef_only"
        s2_mat = (s2_feature_stack(comps[YEARS[0]], comps[YEARS[1]], s2_cols)
                  if reads_s2 else np.zeros((height * width, 0), "float32"))
        s2_present = (np.isfinite(s2_mat).all(1) if reads_s2
                      else np.zeros(height * width, bool))
        s2_mat = np.nan_to_num(s2_mat, nan=0.0)
        # `aef_bands` already holds every AlphaEarth column in `aef_cols` order at
        # 192 float32 per pixel -- 2.3 GB at Oslo's size. Re-stacking it per model
        # allocated a second copy of the same numbers (11 s and 2.3 GB per model
        # at AOI scale, and the term that would decide whether a global tile fits
        # in memory at all), so a model that wants exactly those columns in
        # exactly that order gets a view rather than a copy. A model that wants a
        # subset gets a row selection, which is still band-major and still 3x
        # cheaper than the pixel-major stack it replaced.
        model_aef = [c for c in columns if c in set(aef_cols)]
        aef_block = (aef_bands if model_aef == list(aef_cols) else
                     aef_bands[[list(aef_cols).index(c) for c in model_aef]])

        # T2, the within-AOI counterfactual. Oslo has zero validation plots, so
        # accuracy cannot be measured there at all; what *can* be measured is the
        # same trained model on the same pixels with the Sentinel-2 gate forced
        # off. Tessera's equivalent read was **-16.0%** -- it removed a sixth of
        # the change class, and with no labels in the AOI there was no way to say
        # whether that was commission correctly cleaned or real change lost. This
        # asks the identical question of S2, and it is the reason the gate is a
        # column rather than a baked-in constant.
        # Only a *gated* model has a counterfactual here. A flat recipe
        # (arch='wide', e.g. aef_builtfrac) never reads the mask column, so
        # forcing it to zero changes nothing and would report a meaningless
        # "+0.0%" that reads like evidence of no suppression. For those models
        # the honest comparison is against baseline_aef, i.e. the same model
        # without the S2 columns at all.
        # This used to be a second full sweep over the AOI. It is now the
        # `mask=0` branch that the exact gate marginalisation already computes,
        # so the diagnostic costs nothing.
        gated = entry["spec"]["kwargs"].get("arch") == "two_tower"
        aef_only = entry["spec"].get("deploy") == "aef_only"
        want_off = bool(reads_s2 and gated)
        t_feat = time.time() - tf
        tp = time.time()
        view_probs, classes, fine_classes = predict(
            entry, aef_block, s2_mat, s2_present, batch, seed,
            want_off=want_off, mc_sampling=mc_sampling)
        t_pred = time.time() - tp
        n_pass = (1 if aef_only
                  else (MC_PASSES + (1 if want_off else 0))
                  if (mc_sampling and gated)
                  else (2 if gated else 1)) * len(entry["models"])
        mode = ("gate off, detail tower skipped" if aef_only
                else f"{'sampled' if mc_sampling else 'exact'} gate")
        timings[model_name] = {"features_s": round(t_feat, 1),
                               "predict_s": round(t_pred, 1),
                               "forward_passes": n_pass, "mode": mode}
        print(f"[{name}] {model_name:16s} features {t_feat:.0f}s, "
              f"predict {t_pred:.0f}s over {n_pass} forward passes "
              f"({mode})", flush=True)

        view_codes = {}
        for view, (probs, fine_probs) in view_probs.items():
            # Everything below stays in *index* space and only the 4- and 9-entry
            # class lists are ever touched as strings. The labels-then-compare
            # version did the same work per pixel: an object-dtype array of 2.95 M
            # Python strings, one boolean pass per class over it, and -- worst --
            # a per-pixel `m.split(" -> ")` to decide change, which alone was
            # 1.3 s of the 1.7 s spent turning one posterior into three rasters.
            # Same rasters, 20x less time (0.08 s per view).
            merged_idx = probs.argmax(1)

            suffix = "" if view == "on" else "_s2off"
            keep = list(classes)
            codes = np.where(aef_valid, merged_idx, NODATA).astype("uint8")
            write_class_raster(
                out_dir / f"{name}_{model_name}{suffix}_merged2.tif",
                codes.reshape(height, width), geobox, keep, MERGED_COLORS)

            # The coarse3 read: the same forward pass, taken at the fine head
            # instead of after aggregation. Nine transitions
            # (Nature/Cropland/Artificial x two endpoints) rather than four, which
            # is what distinguishes a field going under tarmac from a forest doing
            # the same. Codes follow the *sorted* class list handed to the writer,
            # never the palette's insertion order -- read them back from the .qml.
            fine_idx = fine_probs.argmax(1)
            fine_codes = np.where(aef_valid, fine_idx, NODATA).astype("uint8")
            write_class_raster(
                out_dir / f"{name}_{model_name}{suffix}_coarse3.tif",
                fine_codes.reshape(height, width), geobox, list(fine_classes),
                CLASS_COLORS)

            # O3: the coarse3 decision-cost gate, written as a SECOND raster
            # beside the plain arg-max rather than replacing it. Both come from
            # the same forward pass, so the pair is an exact counterfactual for
            # what the gate does on this AOI -- which is the only way to see it,
            # since Oslo has no labelled plots to score either read against
            # (G3/G4). The merged2 raster is untouched: the gate acts on the
            # coarse3 arg-max alone.
            costs = _coarse3_costs(entry["spec"], list(fine_classes))
            if costs is not None:
                gated_idx = (fine_probs * costs).argmax(1)
                write_class_raster(
                    out_dir / f"{name}_{model_name}{suffix}_coarse3_gated.tif",
                    np.where(aef_valid, gated_idx, NODATA)
                    .astype("uint8").reshape(height, width),
                    geobox, list(fine_classes), CLASS_COLORS)
                moved = int((gated_idx != fine_idx)[aef_valid].sum())
                print(f"[{name}] {model_name:16s}{suffix or '      '} coarse3 gate "
                      f"moved {moved:,} px ({moved / max(int(aef_valid.sum()), 1):.3%})",
                      flush=True)

            # Consistency of the two reads. p_merged = p_fine @ M, but arg-max
            # does not commute with that sum: three fine classes at 0.30/0.30/0.35
            # elect the singleton at the fine head and the pair after aggregation.
            # So the agreement rate is an empirical fact about this map, not an
            # identity -- print it, because a low value means the two rasters
            # disagree about the same pixels and the user should know which to
            # trust. The mapping is resolved once over the nine class names,
            # never per pixel.
            fine_to_merged = np.array(
                [keep.index(to_merged_label(str(c))) for c in fine_classes])
            same = (fine_to_merged[fine_idx][aef_valid] == merged_idx[aef_valid])
            print(f"[{name}] {model_name:16s}{suffix or '      '} coarse3->merged2 "
                  f"agreement {same.mean():.4%} "
                  f"({int((~same).sum()):,} px differ)", flush=True)
            if view == "on":
                counts = np.bincount(fine_idx[aef_valid],
                                     minlength=len(fine_classes))
                coarse3_stats = {
                    "agreement_with_merged2": float(same.mean()),
                    "px_by_class": {str(c): int(counts[i])
                                    for i, c in enumerate(fine_classes)},
                }

            is_change = np.array(
                ["->" in str(c) and str(c).split(" -> ")[0] != str(c).split(" -> ")[-1]
                 for c in keep])
            ch_codes = np.where(aef_valid, is_change[merged_idx],
                                NODATA).astype("uint8")
            ch_of = {"stable": 0, "change": 1}
            write_class_raster(
                out_dir / f"{name}_{model_name}{suffix}_change.tif",
                ch_codes.reshape(height, width), geobox, list(ch_of),
                CHANGE_COLORS)
            view_codes[view] = ch_codes

            # Persist the merged2 posteriors. U1 showed that refining a *hard*
            # class map removes 11.7% of the change class, because a one-hot
            # neighbourhood arg-max is a majority vote a 0.45% class always
            # loses. A probability-level refinement is the strictly stronger
            # version -- a confident change pixel can outvote its neighbours --
            # and it needs the posteriors, not the labels. Band order is written
            # to the sidecar so nothing downstream has to guess it.
            if save_probs and view == "on":
                for tag, stack, names in (
                    ("probs", probs, classes),
                    ("coarse3_probs", fine_probs, fine_classes),
                ):
                    prob_path = out_dir / f"{name}_{model_name}_{tag}.tif"
                    with rasterio.open(
                        prob_path, "w", driver="GTiff", height=height, width=width,
                        count=len(names), dtype="float32",
                        crs=RioCRS.from_epsg(geobox.crs.epsg),
                        transform=geobox.transform, nodata=np.nan,
                        compress="deflate", predictor=3, tiled=True,
                    ) as dst:
                        for band, cls in enumerate(names, start=1):
                            layer = stack[:, band - 1].reshape(height, width)
                            dst.write(np.where(aef_valid.reshape(height, width),
                                               layer, np.nan).astype("float32"),
                                      band)
                            dst.set_band_description(band, str(cls))
                    prob_path.with_suffix(".classes.json").write_text(
                        json.dumps([str(c) for c in names], indent=2))

        ch_codes = view_codes["on"]
        n_change = int((ch_codes == 1).sum())
        if "off" in view_codes:
            off = view_codes["off"]
            n_off = int((off == 1).sum())
            turned_off = int(((off == 1) & (ch_codes == 0)).sum())
            turned_on = int(((off == 0) & (ch_codes == 1)).sum())
            delta = (n_change - n_off) / max(n_off, 1)
            print(f"[{name}] {model_name:16s} T2 counterfactual: "
                  f"{n_off:,} change without S2 -> {n_change:,} with "
                  f"({delta:+.1%}); {turned_off:,} px removed, "
                  f"{turned_on:,} added  [Tessera's read was -16.0%]", flush=True)
            results.setdefault("_counterfactual", {})[model_name] = {
                "change_px_s2_off": n_off, "change_px_s2_on": n_change,
                "delta_frac": delta, "px_turned_off": turned_off,
                "px_turned_on": turned_on,
            }

        results[model_name] = {
            "change_px": n_change,
            "valid_px": int(aef_valid.sum()),
            "change_frac": n_change / max(int(aef_valid.sum()), 1),
            "s2_present_frac": float(s2_present.mean()),
            "coarse3": coarse3_stats,
        }
        print(f"[{name}] {model_name:16s} change {n_change:,} px "
              f"({results[model_name]['change_frac']:.2%})", flush=True)
    results["_timings"] = timings
    return results


async def main_async(args):
    import pandas as pd  # noqa: F401  (used by predict)

    # The self-check guards the *raster* S2 feature path against the training
    # table. A run where no model reads S2 rasters never executes that path, so
    # running the check would cost time and, worse, print a guarantee about code
    # this map does not use. Skip it, and say so.
    gate_off = {"baseline_aef", "s2off_deploy", "s2off_slim"}
    gate_off |= {f"s2off_{s}" for s in S2_SUBSETS}
    reads_s2_raster = any(m not in gate_off
                          for m in args.models) or args.s2_backdrop
    if not args.no_self_check and reads_s2_raster:
        worst, col = self_check(args.shard_dir, args.s2)
        print(f"self-check: worst relative disagreement {worst:.2e} on {col}")
        if worst > args.tolerance:
            raise SystemExit(
                f"ABORT: raster features disagree with the training path by "
                f"{worst:.2e} (> {args.tolerance:.0e}) on {col}. A map built on "
                "this would not be the model that was scored.")
        print("self-check passed -- raster and training feature paths agree\n")
    elif not args.no_self_check:
        print("self-check skipped -- no requested model reads Sentinel-2 "
              "rasters at inference\n")

    models, aef_cols, s2_stat, s2_built, detail = fit_models(
        args.s2, args.seed, args.models, args.seeds, args.model_cache_dir,
        not args.no_model_cache)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"s2_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    index = AEFIndex(source=DataSource.SOURCE_COOP)
    await index.download()
    index.load()
    summary = {}
    async with VirtualTiffReader(manifest_cache_dir=args.manifest_cache) as reader:
        for name in args.aois:
            summary[name] = await run_aoi(
                name, CITY_AOIS[name], index, reader, models, aef_cols, s2_stat,
                s2_built, args.resolution, args.batch, args.seed,
                args.scenes_per_year, args.cache_dir, out_dir, args.save_probs,
                args.mc_sampling, args.s2_backdrop)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n-> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aois", nargs="+", default=["oslo"])
    parser.add_argument("--models", nargs="+",
                        default=["baseline_aef", "aef_builtfrac", "mc_s2_drop0.7"])
    parser.add_argument("--s2", type=Path,
                        default=project_data_dir("embeddings",
                                                 "s2_features_habloss_recover_10m.parquet"))
    parser.add_argument("--shard-dir", type=Path,
                        default=project_data_dir("embeddings", "s2_shards"))
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--batch", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1,
                        help="number of torch seeds to fit and average "
                             "(S12: 5 gives +0.0038 change-F1 and 2.6x "
                             "tighter seed spread). Now that the data plumbing "
                             "is gone this IS the serving cost -- one member per "
                             "pass over every pixel. S17's sizing puts the knee "
                             "at 3: the seed spread falls 0.0051 -> 0.0025 by "
                             "3 members and 3 -> 5 is inside its own resolution")
    parser.add_argument("--scenes-per-year", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--no-self-check", action="store_true")
    parser.add_argument("--s2-backdrop", action="store_true",
                        help="write the S2 true-colour/NIR reference layers even "
                             "when no model reads Sentinel-2 at inference. Costs "
                             "the composite fetch (~180 s/AOI), so it is off by "
                             "default for the deployed gate-off recipe")
    parser.add_argument("--mc-sampling", action="store_true",
                        help="sample the stochastic gate with 16 MC passes "
                             "instead of marginalising it exactly in 2. Slower "
                             "(8x) and noisier; kept to reproduce earlier runs "
                             "and to verify the exact path against them")
    parser.add_argument("--save-probs", action="store_true",
                        help="also write the merged2 posteriors (float32, one band "
                             "per class) for probability-level refinement")
    parser.add_argument("--cache-dir", type=Path,
                        default=project_data_dir("embeddings", "s2_shards", "cache"))
    parser.add_argument("--output-dir", type=Path, default=project_data_dir("inference"))
    parser.add_argument("--manifest-cache", type=Path,
                        default=project_data_dir("inference", "manifest_cache"))
    parser.add_argument("--model-cache-dir", type=Path,
                        default=project_data_dir("models", "s2_ensembles"),
                        help="directory for fitted seed-ensemble artifacts. The "
                             "cache key includes recipe, seed block, training "
                             "inputs, model kwargs and model code fingerprints, "
                             "so stale fits are ignored")
    parser.add_argument("--no-model-cache", action="store_true",
                        help="refit the requested seed ensemble even if a matching "
                             "artifact exists")
    args = parser.parse_args()
    asyncio.run(main_async(args))


import pandas as pd  # noqa: E402  (module-level for self_check)

if __name__ == "__main__":
    main()
