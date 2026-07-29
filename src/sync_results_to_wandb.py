"""Sync generated analysis results to Weights & Biases.

Examples:
    uv run python src/sync_results_to_wandb.py --mode offline
    uv run python src/sync_results_to_wandb.py --project recover-habloss --mode online
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from project_paths import project_data_dir
from wandb_results import log_result_files

RESULT_SUFFIXES = {".csv", ".json", ".md"}


def experiment_key(path: Path) -> str:
    stem = path.stem
    replacements = [
        ("model_zoo_leaderboard_", "model_zoo_"),
        ("model_zoo_meta_", "model_zoo_"),
        ("map_efficiency_pairwise_", "map_efficiency_"),
        ("map_efficiency_meta_", "map_efficiency_"),
        ("merged_legend_perclass_", "merged_legend_"),
        ("hier_change_recall_curve_", "hier_change_recall_"),
        ("hier_change_recall_meta_", "hier_change_recall_"),
        ("hier_variants_meta_", "hier_variants_"),
        ("hier_novel_meta_", "hier_novel_"),
        ("hier_gh_fair_meta_", "hier_gh_fair_"),
        ("hier_gh_meta_", "hier_gh_"),
        ("gate_threshold_", "gate_threshold_"),
        ("tune_lda_winners", "tune_lda"),
    ]
    for old, new in replacements:
        if stem.startswith(old):
            return new + stem.removeprefix(old)
    if stem.startswith("hier_moe_noise_meta_"):
        return "hier_moe_noise_" + stem.removeprefix("hier_moe_noise_meta_")
    if stem.startswith(("hier_moe_", "hier_noise_")):
        tag = stem.rsplit("_", 1)[-1]
        return f"hier_moe_noise_{tag}"
    return stem


def collect_results(results_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(results_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in RESULT_SUFFIXES:
            groups[experiment_key(path)].append(path)
    return dict(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=project_data_dir("analysis_results"))
    parser.add_argument("--project", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--group", default="analysis-results")
    parser.add_argument("--mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--only", nargs="+", default=None, help="Experiment keys to sync.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    groups = collect_results(args.results_dir)
    if args.only:
        selected = set(args.only)
        groups = {key: files for key, files in groups.items() if key in selected}

    if not groups:
        raise SystemExit(f"no result files found in {args.results_dir}")

    for key, files in groups.items():
        names = ", ".join(path.name for path in files)
        if args.dry_run:
            print(f"{key}: {names}")
            continue
        print(f"syncing {key}: {names}", flush=True)
        log_result_files(
            run_name=key,
            files=files,
            project=args.project,
            entity=args.entity,
            group=args.group,
            mode=args.mode,
            config={"results_dir": str(args.results_dir)},
        )


if __name__ == "__main__":
    main()
