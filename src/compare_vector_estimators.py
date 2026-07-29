"""Vector-valued (3-class) design-based and PPI composition on real data.

Estimand: for land observed leaving Artificial in the RECOVER frame, the joint
composition of its 2024 destination over the coarse classes Nature, Cropland,
and (residual) still-Artificial. Everything is estimated jointly so the full
covariance matrix is retained, which is what makes the derived Cropland-vs-
Nature share and their difference more precise than running a scalar estimator
once per class.

Three estimators are compared:

* ``scalar_per_class`` -- the current approach: run the scalar stratified
  proportion once per class and pretend the classes are independent;
* ``design_vector``    -- the joint multinomial design-based estimator;
* ``ppi_vector``       -- the joint PPI estimator using per-class predicted
  scores from the predicted-only sample (requires the yhat_nature/cropland/
  artificial columns from a fresh ``extract_ppi_gee`` run).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_ppi_subsets import (  # noqa: E402
    DEFAULT_INPUT,
    DEFAULT_RECOVER,
    build_lookup,
    read_areas,
    read_design_strata,
)
from analyse_transition_composition import (  # noqa: E402
    read_clean_recover_labels,
)
from estimators import (  # noqa: E402
    composition_ratio_ci,
    contrast_ci,
    optimal_lam_multinomial_diag,
    stratified_multinomial,
    stratified_ppi_multinomial,
    stratified_prop,
)
from project_paths import project_data_dir  # noqa: E402


# The estimand is the composition of EXITS from Artificial land: of all area
# that left Artificial between 2018 and 2024, the joint split into Nature and
# Cropland destinations. Each class is an area indicator on the full sample;
# the composition is each class area over their sum, read off the joint
# covariance. This mirrors the scalar ratio already reported in the repo but
# keeps both destinations (and their negative covariance) in one estimator.
CLASSES = ["to_nature", "to_cropland"]
PRED_COLUMNS = {
    "to_nature": "yhat_nature",
    "to_cropland": "yhat_cropland",
}
DEFAULT_OUTPUT = project_data_dir("analysis_results")
Z_95 = 1.959963984540054


def exit_indicators(frame: pd.DataFrame) -> np.ndarray:
    """Area indicators for Artificial->Nature and Artificial->Cropland exits."""
    before = frame["lc_2018"].astype(str).str.strip().str.lower()
    after = frame["lc_2024"].astype(str).str.strip().str.lower()
    left_artificial = before == "artificial"
    columns = [
        (left_artificial & after.str.startswith("nature")).astype(float),
        (left_artificial & (after == "cropland")).astype(float),
    ]
    return np.column_stack(columns)


def load_labelled(input_dir: Path, recover_root: Path) -> tuple[pd.DataFrame, bool]:
    """Return the exits-from-Artificial labelled table and a PPI-ready flag.

    Prefer the PPI ``labelled_predictions.csv`` when it carries land cover and
    per-class predicted scores (so vector PPI can run). Otherwise fall back to
    the authoritative cleaned RECOVER labels, which always carry lc_2018/lc_2024
    but no per-class predictions -- enough for the design-based vector rows.
    """
    labelled = pd.read_csv(input_dir / "labelled_predictions.csv")
    if "stratum" not in labelled:
        labelled = labelled.merge(
            read_design_strata(recover_root),
            on="PLOTID",
            how="left",
            validate="one_to_one",
        )
    have_lc = "lc_2018" in labelled and "lc_2024" in labelled
    have_pred = all(c in labelled.columns for c in PRED_COLUMNS.values())
    if have_lc:
        labelled = labelled.dropna(subset=["stratum", "lc_2018", "lc_2024"]).copy()
        labelled["stratum"] = labelled["stratum"].astype(str)
        return labelled, have_pred

    # The GEE labelled file carries per-class predictions but not the full
    # lc_2018/lc_2024 transition (the samples asset only stores PLOTID and the
    # coarse label ``r``). The true transition lives in the cleaned RECOVER
    # sampler CSVs. Join the per-class predictions onto those labels by PLOTID
    # so vector PPI runs on the same design-based estimand.
    _, unique_labels, _ = read_clean_recover_labels(recover_root)
    design = read_design_strata(recover_root)
    merged = unique_labels.merge(
        design, on="PLOTID", how="left", validate="many_to_one"
    )
    if have_pred:
        pred_cols = ["PLOTID", *PRED_COLUMNS.values()]
        predictions = labelled[pred_cols].dropna().drop_duplicates("PLOTID")
        merged = merged.merge(
            predictions, on="PLOTID", how="left", validate="one_to_one"
        )
    merged = merged.dropna(subset=["stratum", "lc_2018", "lc_2024"]).copy()
    merged["stratum"] = merged["stratum"].astype(str)
    # PPI is only usable where the per-class predictions actually joined.
    joined_pred = have_pred and merged[list(PRED_COLUMNS.values())].notna().all(
        axis=1
    ).any()
    return merged, bool(joined_pred)


def _row(name, quantity, value, se, ci):
    return {
        "estimator": name,
        "quantity": quantity,
        "estimate": float(value),
        "se": float(se),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
    }


def summarise(name: str, p: np.ndarray, Sigma: np.ndarray) -> list[dict]:
    """Report the exit composition (shares of total exits) from the area vector.

    ``p`` is the 2-vector of exit areas as fractions of total mapped area:
    (Artificial->Nature, Artificial->Cropland). The reported quantities are the
    composition shares of all exits, each a ratio over the vector sum, plus
    their difference -- all propagated through the joint covariance ``Sigma``.
    """
    rows = []
    # Raw exit-area proportions (of total mapped area), for provenance.
    for k, cls in enumerate(CLASSES):
        se = float(np.sqrt(max(0.0, Sigma[k, k])))
        rows.append(
            _row(name, f"exit_area_prop_{cls}", p[k], se, (p[k] - Z_95 * se, p[k] + Z_95 * se))
        )
    # Composition: each destination as a share of all exits from Artificial.
    nat_share, se_n, ci_n = composition_ratio_ci(p, Sigma, [0], [0, 1])
    rows.append(_row(name, "nature_share_of_exits", nat_share, se_n, ci_n))
    crop_share, se_c, ci_c = composition_ratio_ci(p, Sigma, [1], [0, 1])
    rows.append(_row(name, "cropland_share_of_exits", crop_share, se_c, ci_c))
    # Difference of the two shares (equivalently 2*nature_share - 1); computed
    # on the ratio scale via a small delta-method contrast of the two shares.
    diff = nat_share - crop_share
    # Var(nature_share - cropland_share); since shares sum to 1, this is
    # 4*Var(nature_share).
    se_d = 2.0 * se_n
    rows.append(_row(name, "nature_minus_cropland_share", diff, se_d, (diff - Z_95 * se_d, diff + Z_95 * se_d)))
    return rows


def scalar_per_class(Y: np.ndarray, strat: np.ndarray, Nh: dict) -> tuple:
    """Baseline: independent scalar estimator per class (diagonal covariance)."""
    K = Y.shape[1]
    p = np.zeros(K)
    Sigma = np.zeros((K, K))
    for k in range(K):
        pk, sek, _ = stratified_prop(Y[:, k], strat, Nh)
        p[k] = pk
        Sigma[k, k] = sek**2
    return p, Sigma


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--recover-root", type=Path, default=DEFAULT_RECOVER)
    parser.add_argument("--predicted-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    lookup = build_lookup(args.recover_root)
    areas = read_areas(
        args.recover_root, lookup, args.input_dir / "stratum_areas.csv"
    )
    labelled, have_pred = load_labelled(args.input_dir, args.recover_root)
    labelled = labelled[labelled["stratum"].isin(areas)].copy()
    # Area indicators are defined on the full labelled sample, so every stratum
    # with labelled observations keeps its full mapped-area weight. Restrict Nh
    # to strata that carry labels (the multinomial estimator requires nh >= 1).
    present = set(labelled["stratum"])
    Nh = {h: a for h, a in areas.items() if h in present}
    Y = exit_indicators(labelled)
    strat = labelled["stratum"].to_numpy()
    n_exits = int(Y.sum())

    rows: list[dict] = []
    p_scalar, S_scalar = scalar_per_class(Y, strat, Nh)
    rows += summarise("scalar_per_class", p_scalar, S_scalar)

    p_design, S_design = stratified_multinomial(Y, strat, Nh)
    rows += summarise("design_vector", p_design, S_design)

    # Vector PPI, only if per-class predicted scores are available.
    predicted_path = args.predicted_file or args.input_dir / "predicted_100000.csv"
    have_pred_cols = have_pred
    ppi_note = ""
    if have_pred_cols and predicted_path.exists():
        predicted = pd.read_csv(predicted_path)
        if all(c in predicted.columns for c in PRED_COLUMNS.values()):
            pred_names = [PRED_COLUMNS[c] for c in CLASSES]
            # Restrict labelled rows to those where per-class predictions joined.
            lab = labelled.dropna(subset=pred_names).copy()
            predicted["stratum_id"] = predicted["stratum_id"].astype(int)
            predicted = predicted.merge(
                lookup, on="stratum_id", how="left", validate="many_to_one"
            )
            predicted = predicted.dropna(subset=pred_names)
            # PPI needs every stratum to have BOTH a labelled point and a
            # predicted-only observation; use that intersection as the frame.
            common = (
                set(lab["stratum"]) & set(predicted["stratum"]) & set(Nh)
            )
            lab = lab[lab["stratum"].isin(common)].copy()
            predicted = predicted[predicted["stratum"].isin(common)]
            Nh_ppi = {h: a for h, a in Nh.items() if h in common}
            Y_ppi = exit_indicators(lab)
            strat_ppi = lab["stratum"].to_numpy()
            Yhat = lab[pred_names].to_numpy(float)
            pop_mean, pop_cov, pop_n = {}, {}, {}
            for h, group in predicted.groupby("stratum"):
                mat = group[pred_names].to_numpy(float)
                pop_mean[h] = mat.mean(axis=0)
                pop_cov[h] = np.cov(mat, rowvar=False, ddof=1)
                pop_n[h] = len(mat)
            lam = optimal_lam_multinomial_diag(
                Y_ppi, Yhat, strat_ppi, Nh_ppi, pop_cov, pop_n
            )
            p_ppi, S_ppi = stratified_ppi_multinomial(
                Y_ppi, Yhat, strat_ppi, Nh_ppi, pop_mean, pop_cov, pop_n, lam=lam
            )
            rows += summarise("ppi_vector", p_ppi, S_ppi)
            ppi_note = (
                f"ppi_vector: {len(lab)} labelled, {len(predicted):,} predicted, "
                f"{len(Nh_ppi)} strata; lambda per class "
                f"= {np.round(lam, 3).tolist()}"
            )
        else:
            ppi_note = (
                f"{predicted_path.name} lacks per-class yhat columns; "
                "re-run extract_ppi_gee.py."
            )
    else:
        ppi_note = (
            "Per-class predicted scores not found. Re-run extract_ppi_gee.py to "
            "attach yhat_nature/yhat_cropland/yhat_artificial, then rerun."
        )

    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "vector_composition_comparison.csv"
    result.to_csv(out, index=False)

    result["ci_width"] = result["ci95_high"] - result["ci95_low"]
    display = result.copy()
    for col in ["estimate", "se", "ci95_low", "ci95_high", "ci_width"]:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    print(f"RECOVER: {len(labelled)} labelled plots, {len(Nh)} strata, "
          f"{n_exits} observed exits from Artificial")
    print(display.to_string(index=False))
    if ppi_note:
        print(ppi_note)

    # Headline: how the joint covariance changes each derived interval relative
    # to estimating the destination shares as if independent. For a SHARE or a
    # DIFFERENCE the two exit destinations are negatively correlated, so the
    # honest joint interval is wider than the naive independent one -- the
    # scalar version understates the variance. The covariance only narrows
    # COMBINED quantities (a sum of shares), which for a 2-part composition is
    # the trivial constant 1. The value of the vector estimator here is a
    # correct interval and a reusable covariance for any downstream contrast,
    # not an automatic precision gain.
    def width(est, q):
        sub = result[(result.estimator == est) & (result.quantity == q)]
        return float(sub["ci_width"].iloc[0]) if len(sub) else float("nan")

    for q in ["cropland_share_of_exits", "nature_minus_cropland_share"]:
        w_scalar = width("scalar_per_class", q)
        w_vector = width("design_vector", q)
        if np.isfinite(w_scalar) and np.isfinite(w_vector) and w_scalar > 0:
            direction = "wider" if w_vector > w_scalar else "narrower"
            print(
                f"{q}: naive-independent CI width {w_scalar:.4f} vs honest "
                f"joint {w_vector:.4f} ({100 * (w_vector / w_scalar - 1):+.1f}%, "
                f"{direction})"
            )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
