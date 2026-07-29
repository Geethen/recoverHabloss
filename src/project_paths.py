"""Shared filesystem defaults for running analyses from any checkout."""
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _first_existing(candidates: list[Path], fallback: Path) -> Path:
    for candidate in candidates:
        if (candidate / "data").exists():
            return candidate
    return fallback


def default_habloss_root() -> Path:
    env = _env_path("HABLOSS_ROOT")
    if env is not None:
        return env
    candidates = [
        Path("/data/P-Prosjekter2/155069_habloss/R"),
        Path("P:/155069_habloss/R"),
    ]
    return _first_existing(candidates, candidates[0])


def default_recover_root() -> Path:
    env = _env_path("RECOVER_ROOT")
    if env is not None:
        return env
    candidates = [
        Path("/data/P-Prosjekter2/155020_recover/WP1/R"),
        Path("P:/155020_recover/WP1/R"),
    ]
    return _first_existing(candidates, candidates[0])


def project_data_dir(*parts: str) -> Path:
    return REPO_ROOT.joinpath("data", *parts)
