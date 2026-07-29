"""Utilities for logging experiment result files to Weights & Biases."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


DEFAULT_PROJECT = "recover-habloss"


def _import_wandb():
    try:
        import wandb  # type: ignore
    except ImportError as error:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "wandb is not installed. Run `uv sync` or `uv add wandb` first."
        ) from error
    return wandb


def safe_wandb_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return name.strip(".-") or "experiment-results"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {"value": data}


def log_result_files(
    *,
    run_name: str,
    files: Iterable[Path],
    project: str | None = None,
    entity: str | None = None,
    group: str | None = None,
    mode: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Log CSV/JSON/Markdown result files as one W&B run.

    CSV files are logged both as artifact files and as W&B tables. JSON objects are
    folded into the run config under their filename stem.
    """
    wandb = _import_wandb()
    paths = [Path(path) for path in files]
    paths = [path for path in paths if path.exists()]
    if not paths:
        raise ValueError(f"no result files found for {run_name}")

    merged_config: dict[str, Any] = dict(config or {})
    for path in paths:
        if path.suffix.lower() == ".json":
            try:
                merged_config[path.stem] = read_json(path)
            except json.JSONDecodeError:
                merged_config[path.stem] = {"unparsed_json": str(path)}

    run = wandb.init(
        project=project or os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
        entity=entity or os.environ.get("WANDB_ENTITY"),
        name=safe_wandb_name(run_name),
        group=group,
        job_type="analysis-result-sync",
        mode=mode or os.environ.get("WANDB_MODE"),
        config=merged_config,
    )
    try:
        artifact = wandb.Artifact(safe_wandb_name(run_name), type="analysis-results")
        table_payload: dict[str, Any] = {}
        for path in paths:
            artifact.add_file(str(path), name=path.name)
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
                table_key = safe_wandb_name(path.stem).replace("-", "_")
                table_payload[table_key] = wandb.Table(dataframe=frame)
                for column in frame.select_dtypes(include="number").columns:
                    if len(frame[column].dropna()):
                        table_payload[f"{table_key}/{column}_max"] = float(frame[column].max())
        if table_payload:
            run.log(table_payload)
        run.log_artifact(artifact)
    finally:
        run.finish()
