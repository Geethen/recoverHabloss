"""Fit the F1-tuned LDA transition models on the whole labelled set and persist.

The leaderboard scores models out-of-fold; nothing there is a *fitted* artefact
you can point at a new scene. This script trains the two tuned LDA framings on
every labelled plot and writes them to disk, so ``infer_cities.py`` can apply the
exact same transforms to AlphaEarth pixels over Johannesburg and Oslo:

* **direct** (``lda``): one 7-way discriminant over the joint 2018+2024+diff
  vector (lsqr solver, 0.3 covariance shrinkage, empirical priors, standardised).
* **post-classification** (``lda_pc``): label each date on its own with the
  post-class-tuned LDA (Ledoit-Wolf shrinkage, uniform priors) and read the
  transition off the pair. Persisted as a fitted ``PostClassification``.

Both share the same class-code space (the direct model's class list), so the two
map into the same colours and the same GeoTIFF codes. The feature column order
and class list travel in JSON sidecars, because a pixel matrix built in a
different column order would be silently misaligned against the standardiser.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from model_zoo import (
    DEFAULT_INPUT,
    RARE_LABEL,
    PostClassification,
    build_registry,
    classical_models,
    feature_columns,
    load,
)
from project_paths import project_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir", type=Path, default=project_data_dir("models")
    )
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument(
        "--model", default="lda",
        help="Registered direct model to fit (default: the tuned 'lda')",
    )
    parser.add_argument(
        "--postclass-base", default="lda_pc",
        help="Base model for the post-classification framing (default 'lda_pc')",
    )
    args = parser.parse_args()

    frame, target, groups = load(args.input, args.min_class_count)
    columns = feature_columns(frame)

    registry = build_registry(columns, balanced=False, postclass_bases=[])
    if args.model not in registry:
        raise SystemExit(f"{args.model} not in registry: {sorted(registry)}")
    factory, needs_frame = registry[args.model]
    if needs_frame:
        raise SystemExit(f"{args.model} needs the frame; not a direct pixel model")

    X = frame[columns].to_numpy()
    y = target.to_numpy()
    classes = sorted(set(y))
    change_classes = [
        c for c in classes
        if c == RARE_LABEL or c.split(" -> ")[0] != c.split(" -> ")[1]
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- direct 7-way model -------------------------------------------------
    direct = factory().fit(X, y)
    direct_path = args.output_dir / f"transition_{args.model}_direct.joblib"
    joblib.dump(direct, direct_path)

    base_sidecar = {
        "input": str(args.input),
        "n_train": int(len(y)),
        "classes": classes,
        "change_classes": change_classes,
        "rare_label": RARE_LABEL,
        "embedding_years": [2018, 2024],
    }
    (args.output_dir / f"transition_{args.model}_direct.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "framing": "direct",
                **base_sidecar,
                "feature_columns": columns,
                "n_features": len(columns),
                "feature_layout": (
                    "per channel A{i:02d}: _2018, _2024, _diff (= 2024 - 2018); "
                    "columns are sorted() of these names -- match at inference"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # --- post-classification model -----------------------------------------
    pc_bases = classical_models(balanced=False)
    if args.postclass_base not in pc_bases:
        raise SystemExit(
            f"postclass base '{args.postclass_base}' not in {sorted(pc_bases)}"
        )
    postclass = PostClassification(
        columns, pc_bases[args.postclass_base], shared=True
    ).fit(frame, y)
    pc_path = args.output_dir / f"transition_{args.postclass_base}_postclass.joblib"
    joblib.dump(postclass, pc_path)
    (args.output_dir / f"transition_{args.postclass_base}_postclass.json").write_text(
        json.dumps(
            {
                "model": args.postclass_base,
                "framing": "postclass",
                **base_sidecar,
                "columns_2018": postclass.columns_2018,
                "columns_2024": postclass.columns_2024,
                "retained_transitions": sorted(postclass.retained_),
                "feature_layout": (
                    "per-date arrays: x18 columns in columns_2018 order, x24 in "
                    "columns_2024 order (both channel order A00..A63)"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Trained on {len(y):,} plots, {len(columns)} features")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Wrote {direct_path}")
    print(f"Wrote {pc_path}")


if __name__ == "__main__":
    main()
