"""Write a dataset arm out as a drop-in ``state_frame`` for the real pretraining phase.

``model_zoo._prepare_state_pool`` asks a state pool for exactly two things: a
``state`` column in the model's endpoint vocabulary, and the 64 **2018-block**
columns ``A00_2018..A63_2018``. It does not ask what year those numbers were
measured in -- ``_pretrain_state`` encodes the external block with
``encode_single(..., "2018")`` unconditionally.

That is the whole hook. A 2021 AlphaEarth vector written into the ``A**_2018``
columns is, to the phase, one more single-date state label; the temporal
augmentation reaches the deployed encoder as a **bigger pool file** and not as a
line of changed code in ``model_zoo``. Nothing in the settled path moves.

``block_id`` rides along because ``twotower_lab.cv_probs_state`` cuts the pool to
each fold's training blocks, and the endogenous arms *are* RECOVER plots -- so
without it a fold would pretrain on the held-out block's own plots. With it, the
existing filter removes them and the blocked CV keeps measuring what it says.

Usage::

    P=/home/geethen.singh/.pixi/envs/geo/bin/python
    $P -m statepre.export --arm stable_years
    # -> data/embeddings/state_labels_stable_years_2018.parquet

then, in a twotower_lab idea, hand ``_state_pool("state_labels_stable_years_2018.parquet")``
to the recipe that currently gets ``STATE_POOL``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from project_paths import project_data_dir
from statepre import data as sp_data


def to_state_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The long frame in ``model_zoo``'s pool format, dropping pseudo rows.

    A ``pseudo`` row has no state until a fold fills it, and that filling happens
    inside ``llto.py``. There is no honest way to put one in a file, so
    ``all_years_pseudo`` exports as ``stable_years`` plus nothing.
    """
    frame = frame.loc[frame["kind"] != "pseudo"].reset_index(drop=True)
    out = pd.DataFrame({
        "sid": frame["source"] + ":" + frame["loc_id"].astype(str)
               + ":" + frame["year"].astype(str),
        "state": frame["state"].astype(str),
        "lon": frame["lon"], "lat": frame["lat"],
        "block_id": frame["block_id"],
        "source": frame["source"], "year": frame["year"], "kind": frame["kind"],
    })
    bands = pd.DataFrame(frame[sp_data.FEATURES].to_numpy("float64"),
                         columns=[f"A{i:02d}_2018" for i in range(64)])
    return pd.concat([out, bands], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="stable_years", choices=list(sp_data.ARMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("embeddings"))
    args = parser.parse_args()

    plots = sp_data.load_plots()
    long = sp_data.build(args.arm, plots, seed=args.seed)
    frame = to_state_frame(long)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"state_labels_{args.arm}_2018.parquet"
    frame.to_parquet(out, index=False)

    meta = {
        "arm": args.arm, "seed": args.seed,
        "description": sp_data.ARMS[args.arm][1],
        "n_rows": int(len(frame)),
        "n_locations": int(long.loc[long["kind"] != "pseudo", "loc_id"].nunique()),
        "years_written_into_the_2018_block": sorted(
            int(y) for y in frame["year"].unique()),
        "kind_counts": frame["kind"].value_counts().to_dict(),
        "state_counts": frame["state"].value_counts().to_dict(),
        "n_blocks": int(frame["block_id"].nunique()),
    }
    (args.output_dir / f"metadata_state_labels_{args.arm}_2018.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {len(frame):,} rows to {out}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
