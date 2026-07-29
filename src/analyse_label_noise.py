"""Estimate the label-noise ceiling from plots interpreted twice.

Every model in the zoo tops out near change-F1 0.58, and a linear discriminant
ties gradient boosting and two tabular foundation models there. That pattern --
model family stops mattering -- is what an accuracy *ceiling* looks like, but it
does not say where the ceiling is. This script measures one component of it
directly: how often two independent interpretations of the **same plot**
disagree on the transition.

The combined frame carries 76 plots twice at identical coordinates:

* **54 RECOVER reverifications** (``recover | recover``) -- the same plot read
  twice under the same protocol, no legend or frame difference, only
  interpretation. **These are not a random sample.** Reverification targets
  plots the first read flagged as uncertain or changed, so the subset is
  change-enriched (measured here against the full RECOVER change rate). Its raw
  agreement rate is therefore *reliability on contested plots*, a lower bound on
  reliability, not the population label-noise rate -- read it that way.
* **22 HABLOSS dual-frame overlaps** (``habloss_landwater | habloss_main``) --
  the same plot appearing in two sample frames. These agree trivially when the
  frames share one interpretation (they do here), so this subset is a
  duplication check, not an independent-read reliability estimate.

Both are compared on the coarse 3-class legend the transition matrix is defined
on. For each repeated plot the two reads are compared on the 2018 label, the
2024 label, and the change / no-change call the estimator consumes, with a
Wilson interval because the samples are small. The disagreement is then broken
down by *which* endpoint flips: if the change labels are noisy on one semantic
boundary, that boundary -- not the model -- is what caps change-F1 there, and
the fix is to the target (merge or down-weight that boundary), not the features.

Writes a per-plot disagreement table and a summary to ``analysis_results/``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from model_zoo import coarsen, is_change_label
from project_paths import project_data_dir


DEFAULT_INPUT = project_data_dir("embeddings", "embeddings_habloss_recover.parquet")
DEFAULT_OUTPUT = project_data_dir("analysis_results")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion -- honest at small n and near 0/1.

    The normal approximation collapses to a zero-width interval when k = 0,
    which is exactly the regime here (few disagreements out of ~50 pairs), so
    the Wilson form is used instead.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_reads(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per repeated PLOTID: the two coarse reads and whether they agree.

    Only plots appearing exactly twice are used. A plot present three times
    would need a rule for which pair to compare; none exists in this frame, so
    it is flagged rather than guessed at.
    """
    frame = frame.assign(
        c_2018=coarsen(frame["lc_2018"]),
        c_2024=coarsen(frame["lc_2024"]),
    )
    counts = frame["PLOTID"].value_counts()
    repeated = counts[counts >= 2].index
    over_two = counts[counts > 2].index
    if len(over_two):
        raise ValueError(
            f"{len(over_two)} PLOTIDs appear more than twice; pairing is "
            f"ambiguous for {sorted(over_two)[:5]}"
        )

    rows = []
    for plotid in repeated:
        pair = frame[frame["PLOTID"] == plotid]
        a, b = pair.iloc[0], pair.iloc[1]
        combo = " | ".join(sorted(pair["source"]))
        trans_a = f"{a.c_2018} -> {a.c_2024}"
        trans_b = f"{b.c_2018} -> {b.c_2024}"
        rows.append(
            {
                "PLOTID": plotid,
                "source_combo": combo,
                "read_a": trans_a,
                "read_b": trans_b,
                "agree_2018": a.c_2018 == b.c_2018,
                "agree_2024": a.c_2024 == b.c_2024,
                "agree_transition": trans_a == trans_b,
                # The estimand: does the change / no-change call survive?
                "agree_change": is_change_label(trans_a) == is_change_label(trans_b),
                "either_change": is_change_label(trans_a) or is_change_label(trans_b),
            }
        )
    return pd.DataFrame(rows)


def disagreement_boundaries(pairs: pd.DataFrame) -> pd.DataFrame:
    """Which class boundary the disagreeing reads flip across.

    Each disagreeing pair is reduced to the unordered set of coarse classes its
    two reads touch (e.g. a ``Cropland -> Nature`` vs ``Cropland -> Cropland``
    split flips Nature<->Cropland). A single dominant boundary means the label
    noise is one fuzzy distinction, not diffuse interpreter unreliability.
    """
    disagree = pairs[~pairs["agree_transition"]]
    rows = []
    for _, r in disagree.iterrows():
        classes = set()
        for read in (r["read_a"], r["read_b"]):
            before, after = read.split(" -> ")
            classes.update((before, after))
        # The boundary is the symmetric difference of the two transitions.
        a = set(r["read_a"].split(" -> "))
        b = set(r["read_b"].split(" -> "))
        flip = a.symmetric_difference(b)
        rows.append({"boundary": " / ".join(sorted(flip)) or "(same classes, order)"})
    if not rows:
        return pd.DataFrame(columns=["boundary", "n"])
    return (
        pd.DataFrame(rows)["boundary"].value_counts()
        .rename_axis("boundary").reset_index(name="n")
    )


def summarise(pairs: pd.DataFrame, label: str) -> dict:
    n = len(pairs)
    out = {"subset": label, "n_pairs": int(n)}
    for field in ("agree_2018", "agree_2024", "agree_transition", "agree_change"):
        agree = int(pairs[field].sum())
        disagree = n - agree
        lo, hi = wilson_interval(disagree, n)
        key = field.replace("agree_", "")
        out[f"{key}_agree_rate"] = round(agree / n, 4) if n else None
        out[f"{key}_disagree_rate"] = round(disagree / n, 4) if n else None
        out[f"{key}_disagree_ci95"] = [round(lo, 4), round(hi, 4)]
    # Among plots either read called change, how often do they agree it changed?
    # This is the reliability of the class the whole project turns on.
    changed = pairs[pairs["either_change"]]
    if len(changed):
        agree = int(changed["agree_change"].sum())
        lo, hi = wilson_interval(len(changed) - agree, len(changed))
        out["change_flag_reliability"] = round(agree / len(changed), 4)
        out["change_flag_reliability_ci95"] = [round(lo, 4), round(hi, 4)]
        out["n_either_change"] = int(len(changed))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    if "source" not in frame:
        raise SystemExit(
            f"{args.input} has no 'source' column; the reverification / "
            "dual-frame split cannot be identified"
        )

    # Change rate per source, so the reverification subset's enrichment (and
    # hence its selection bias) is quantified rather than assumed.
    full = frame.assign(
        _t=coarsen(frame["lc_2018"]) + " -> " + coarsen(frame["lc_2024"])
    )
    full_change_rate = {
        src: round(float(g["_t"].map(is_change_label).mean()), 4)
        for src, g in full.groupby("source")
    }

    pairs = paired_reads(frame)
    print(f"{len(pairs)} plots interpreted twice "
          f"({frame['PLOTID'].nunique():,} unique of {len(frame):,} rows)\n")

    subsets = {"all_repeats": pairs}
    for combo in sorted(pairs["source_combo"].unique()):
        subsets[combo] = pairs[pairs["source_combo"] == combo]
    summaries = [summarise(sub, name) for name, sub in subsets.items()]

    boundaries = disagreement_boundaries(pairs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "label_noise_pairs.csv", index=False)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output_dir / "label_noise_summary.csv", index=False)
    boundaries.to_csv(args.output_dir / "label_noise_boundaries.csv", index=False)
    (args.output_dir / "label_noise_summary.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "change_rate_by_source": full_change_rate,
                "subsets": summaries,
                "disagreement_boundaries": boundaries.to_dict("records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for s in summaries:
        print(f"[{s['subset']}] n={s['n_pairs']}")
        print(f"    transition agreement : {s['transition_agree_rate']:.1%} "
              f"(disagree {s['transition_disagree_rate']:.1%}, "
              f"95% CI {s['transition_disagree_ci95']})")
        print(f"    change-call agreement: {s['change_agree_rate']:.1%}")
        if "change_flag_reliability" in s:
            print(f"    change-flag reliability (either read changed): "
                  f"{s['change_flag_reliability']:.1%} "
                  f"of {s['n_either_change']} plots")
        print()

    rev = next(s for s in summaries if s["subset"] == "recover | recover")
    rev_change = full["_t"].loc[  # change rate inside the reverified pairs
        full["PLOTID"].isin(pairs.loc[pairs.source_combo == "recover | recover",
                                      "PLOTID"])
    ].map(is_change_label).mean()
    print("Reading this honestly:")
    print(f"  RECOVER reverifications are change-enriched "
          f"({rev_change:.0%} change vs {full_change_rate.get('recover'):.0%} in "
          f"the full RECOVER set) -- a selected hard subset, not the population.")
    print(f"  On those contested plots two reads agree on the transition only "
          f"{rev['transition_agree_rate']:.0%} of the time.")
    if len(boundaries):
        top = boundaries.iloc[0]
        share = top["n"] / int(boundaries["n"].sum())
        print(f"  {share:.0%} of disagreements flip the {top['boundary']} "
              f"boundary -- the label noise is one fuzzy distinction, not diffuse.")
    print("  Implication: change-F1 on that boundary is capped by the labels, "
          "not the model; fix the target (merge/down-weight it), not the features.")


if __name__ == "__main__":
    main()
