#!/usr/bin/env python
"""Re-cut the reader assignment of batches that are ALREADY BUILT.

WHY THIS IS NOT ``build_label_batches.py --double-frac``
-------------------------------------------------------
Re-running the builder would redraw the points from the candidate table, and a
point that moves takes its baked evidence with it: the chip sprites
(``<batch>_chips/``, ~3 MB and ~36 min of Earth Engine per 100 points) and the
dense series (``<batch>_dense/``) are keyed by point id. Changing *who reads a
point* must not cost a re-bake, so this script rewrites the two assignment
fields in place and leaves point identity, order and every sidecar untouched.

It imports ``assign`` from the builder rather than reimplementing it. There is
one definition of what an assignment is, and a second copy here would be free to
drift from the one that cuts the next round -- the same argument as
``S2_SUBSETS`` in ``twotower_lab``.

    $G src/reassign_batches.py --double-frac 1.0            # every listed batch
    $G src/reassign_batches.py --double-frac 1.0 --batches cov001,rar001
    $G src/reassign_batches.py --double-frac 1.0 --dry-run

100% OVERLAP, AND WHY ROUND ONE IS CUT THAT WAY
-----------------------------------------------
``--double-frac 1.0`` puts every point in front of every expert. That halves the
distinct points per interpreter-hour, which is a real cost against AL5's
"more labels is the lever" (+0.026 change-F1 per doubling) -- but at round one's
200 points it buys the campaign's first measurement of its own noise for ~4
interpreter-hours, and the literature is clear that redundancy pays exactly
here: a small budget, and annotators whose accuracy is not yet known (Lin,
Mausam & Weld, HCOMP 2014). It is a round-one decision, not a standing policy;
set it from the measured agreement once round one closes.

Note that ``assign`` gives a doubled point TWO readers -- the primary and the
next expert round-robin. With the campaign's two experts that is everybody, so
1.0 really is 100%. With three or more it would not be, and this script says so
rather than silently cutting a third of the overlap you asked for.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_label_batches import BATCH_DIR, assign, assignment_counts, show


def listed_batches(outdir: Path) -> list[str]:
    """Batch ids in the manifest, which is what the app will actually serve.

    Deliberately not ``glob("*.json")``: an unlisted batch (b001, the beta
    stress-test draw) is not part of the campaign and re-cutting its readers
    would quietly put it back in somebody's workload if it were ever relisted.
    """
    path = outdir / "index.json"
    if not path.exists():
        return []
    return [e["batch_id"] for e in json.loads(path.read_text()).get("batches", [])]


def reassign(path: Path, double_frac: float, seed: int) -> dict:
    """Rewrite one batch file's assignment. Returns its manifest fields."""
    batch = json.loads(path.read_text())
    experts = [str(e) for e in (batch.get("experts") or [])]
    if not experts:
        raise ValueError(f"{path} names no experts; nothing to assign")
    points = batch.get("points") or []

    before = assignment_counts(points)
    assign(points, experts, double_frac, seed=seed)
    counts = assignment_counts(points)

    doubled = sum(1 for p in points
                  if len(p.get("required_readers") or []) > 1)
    full = sum(1 for p in points
               if set(p.get("required_readers") or []) >= set(experts))
    path.write_text(json.dumps(batch, indent=1) + "\n")
    print(f"  {show(path)}  {len(points)} points, {doubled} double-read "
          f"({100 * doubled / max(len(points), 1):.0f}%), "
          f"{full} read by all {len(experts)}")
    print(f"      readings {dict(sorted(before.items()))} "
          f"-> {dict(sorted(counts.items()))}")
    return {"assigned": counts, "assigned_to": ", ".join(sorted(counts))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--double-frac", type=float, required=True,
                        help="fraction of each batch given a second reading; "
                             "1.0 is total overlap")
    parser.add_argument("--batches",
                        help="comma-separated batch ids (default: every batch "
                             "in index.json)")
    parser.add_argument("--seed", type=int, default=0,
                        help="the builder's assignment seed (default 0, which "
                             "is what cut these batches)")
    parser.add_argument("--outdir", type=Path, default=BATCH_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="say what would change and write nothing")
    args = parser.parse_args()

    wanted = ([b.strip() for b in args.batches.split(",") if b.strip()]
              if args.batches else listed_batches(args.outdir))
    if not wanted:
        raise SystemExit("no batches to re-assign (empty or missing index.json)")

    if args.dry_run:
        for batch_id in wanted:
            path = args.outdir / f"{batch_id}.json"
            batch = json.loads(path.read_text())
            experts = batch.get("experts") or []
            n = len(batch.get("points") or [])
            print(f"  would re-assign {show(path)}: {n} points over "
                  f"{len(experts)} experts at double_frac={args.double_frac}")
        return

    manifest_path = args.outdir / "index.json"
    manifest = json.loads(manifest_path.read_text())
    entries = {e["batch_id"]: e for e in manifest.get("batches", [])}

    for batch_id in wanted:
        path = args.outdir / f"{batch_id}.json"
        if not path.exists():
            raise SystemExit(f"{path} does not exist")
        fields = reassign(path, args.double_frac, args.seed)
        # The manifest's per-expert counts are what the app reads to answer
        # "resume my batch" and to draw "N for you" WITHOUT downloading the
        # batch file. Leaving them stale would show every expert the old
        # workload on the picker and the right one after they opened it.
        if batch_id in entries:
            entries[batch_id].update(fields)

    manifest_path.write_text(json.dumps(
        {"batches": list(entries.values())}, indent=1) + "\n")
    print(f"  {show(manifest_path)}  {len(entries)} batches")


if __name__ == "__main__":
    main()
