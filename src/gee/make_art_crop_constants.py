"""Regenerate the constants embedded in `art_to_cropland_similarity.js`.

The RECOVER GEE sample assets carry no transition labels (only PLOTID and `r`),
so the 46 `Artificial -> Cropland` plots cannot be selected inside Earth Engine.
This script fits the retrieval prototypes locally and prints them as JavaScript
literals to paste into the script's CONFIG block.

    /home/geethen.singh/.pixi/envs/geo/bin/python src/gee/make_art_crop_constants.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

FRAME = "data/embeddings/embeddings_habloss_recover.parquet"
N_PROTO, N_NOVELTY, SEED = 3, 24, 0


def coarse(label: str) -> str:
    return "Nature" if str(label).startswith("Nature") else label


def load():
    df = pd.read_parquet(FRAME)
    c18 = [f"A{i:02d}_2018" for i in range(64)]
    c24 = [f"A{i:02d}_2024" for i in range(64)]
    # pandas 3.x round-trips parquet floats as nullable extension dtypes
    e18 = df[c18].astype("float64").to_numpy()
    e24 = df[c24].astype("float64").to_numpy()
    ok = np.isfinite(e18).all(1) & np.isfinite(e24).all(1)
    df, e18, e24 = df[ok].reset_index(drop=True), e18[ok], e24[ok]
    tgt = ((df.lc_2018.map(coarse) == "Artificial")
           & (df.lc_2024.map(coarse) == "Cropland")).to_numpy()
    diff = e24 - e18
    unit = diff / np.clip(np.linalg.norm(diff, axis=1, keepdims=True), 1e-9, None)
    return df, diff, unit, tgt


def fit(unit, rows, k):
    centres = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(
        unit[rows]).cluster_centers_
    return centres / np.linalg.norm(centres, axis=1, keepdims=True)


def as_js(name, arr):
    rows = ",\n".join("    [" + ", ".join(f"{v:.4f}" for v in r) + "]" for r in arr)
    return f"var {name} = [\n{rows}\n  ];"


def main():
    df, diff, unit, tgt = load()
    proto = fit(unit, tgt, N_PROTO)
    novelty = fit(unit, np.ones(len(unit), bool), N_NOVELTY)
    mu, sigma = diff.mean(1), diff.std(1)
    print(f"// fitted on {len(df)} plots, {tgt.sum()} of them Artificial -> Cropland")
    print(as_js("PROTOTYPES", proto))
    print(as_js("NOVELTY_CENTROIDS", novelty))
    print(f"var FA_SCALE = {{mu_min: {mu.min():.5f}, mu_max: {mu.max():.5f}, "
          f"sd_min: {sigma.min():.5f}, sd_max: {sigma.max():.5f}}};")
    sim = (unit @ proto.T).max(1)
    print(f"// similarity: background p50 {np.percentile(sim, 50):.3f} "
          f"p90 {np.percentile(sim, 90):.3f} p99 {np.percentile(sim, 99):.3f}; "
          f"target p10 {np.percentile(sim[tgt], 10):.3f} "
          f"p50 {np.percentile(sim[tgt], 50):.3f}")


if __name__ == "__main__":
    main()
