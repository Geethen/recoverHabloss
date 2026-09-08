"""Round one is cut at 100% overlap, and each expert walks it in its own order.

Two properties, and they fail in opposite directions:

  * **Every point is read by both experts.** If the assignment silently reverts
    to a 5% sample, the campaign still runs, the app still looks right, and the
    agreement number comes back computed over ten points instead of two hundred
    -- which is exactly the silent-denominator failure §AL8 was written about.
  * **The two experts do NOT walk it in the same order.** Shared order means
    shared fatigue and shared anchoring on the same points, which inflates
    agreement without either reader doing anything wrong. That error is
    invisible to the agreement number by construction, so it has to be
    prevented here rather than detected later.

The order function is loaded from `app/js/app.js` and run in **node**, not
re-implemented in Python: §AL8's standing lesson is that a Python double
polices a contract the JavaScript may not have signed. Skips cleanly with no
node, so `pytest -q` stays green on a bare checkout.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
APP = ROOT / "app" / "js" / "app.js"
BATCHES = ROOT / "app" / "batches"
NODE = shutil.which("node") or shutil.which("nodejs")

#: The batches round one actually serves. Read from the manifest rather than
#: globbed: an unlisted batch (b001) is not part of the campaign and must not
#: drag this suite red when its assignment is deliberately left alone.
def listed() -> list[str]:
    index = BATCHES / "index.json"
    if not index.exists():
        return []
    return [e["batch_id"] for e in json.loads(index.read_text())["batches"]]


def _js_block(start: str) -> str:
    """Source from `start` to the first line that is just `}` or `};`."""
    text = APP.read_text()
    i = text.index(start)
    m = re.compile(r"^\};?$", re.M).search(text, i)
    assert m, f"no closing brace for {start!r}"
    return text[i:m.end()]


def _run_js(expr: str, expert: str, campaign: str = "recover-habloss"):
    """Evaluate `expr` against the app's real ordering functions in node.

    The browser globals those functions close over are stubbed to the two
    values they actually read -- the campaign and the current expert id.
    """
    preamble = (
        "const CFG = { campaign: " + json.dumps(campaign) + " };"
        "const Expert = { id: () => " + json.dumps(expert) + " };"
        + _js_block("function seedFrom(text) {")
        + _js_block("function rngFrom(seed) {")
        + _js_block("function orderedForExpert(points, batchId, expertId) {")
        + _js_block("function isFullOverlap(batch, points) {")
    )
    run = subprocess.run(
        [NODE, "-e", preamble + "console.log(JSON.stringify(" + expr + "));"],
        capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr.strip()[:2000]
    return json.loads(run.stdout)


def _ids(batch_id: str, expert: str) -> list[str]:
    """The order `expert` walks `batch_id` in, straight out of the app's code."""
    points = json.loads((BATCHES / f"{batch_id}.json").read_text())["points"]
    thin = json.dumps([{"id": p["id"]} for p in points])
    return _run_js(
        f"orderedForExpert({thin}, {json.dumps(batch_id)}).map(p => p.id)",
        expert)


# ── the assignment: 100% overlap ────────────────────────────────────────────
@pytest.mark.parametrize("batch_id", listed())
def test_every_point_is_read_by_every_expert(batch_id):
    """`--double-frac 1.0`: 200 points, 400 readings, and no point read once."""
    batch = json.loads((BATCHES / f"{batch_id}.json").read_text())
    experts = set(batch["experts"])
    assert len(experts) >= 2, f"{batch_id} names fewer than two experts"
    for point in batch["points"]:
        readers = set(point.get("required_readers") or [])
        assert readers == experts, (
            f"{batch_id}/{point['id']} is read by {sorted(readers)}, "
            f"not by all of {sorted(experts)}")


@pytest.mark.parametrize("batch_id", listed())
def test_the_manifest_counts_match_the_batch_file(batch_id):
    """The picker draws "N for you" from the manifest WITHOUT downloading the
    batch, so a stale count shows one workload on the picker and another after
    the batch opens."""
    entry = next(e for e in json.loads((BATCHES / "index.json").read_text())
                 ["batches"] if e["batch_id"] == batch_id)
    batch = json.loads((BATCHES / f"{batch_id}.json").read_text())
    counts: dict[str, int] = {}
    for point in batch["points"]:
        for reader in point.get("required_readers") or []:
            counts[reader] = counts.get(reader, 0) + 1
    assert entry["assigned"] == counts
    assert all(n == len(batch["points"]) for n in counts.values())


# ── the walk order: per expert, stable, and a real permutation ──────────────
@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("batch_id", listed())
def test_each_expert_gets_a_permutation_not_a_subset(batch_id):
    """Losing or duplicating a point here would silently change the workload."""
    filed = [p["id"] for p in
             json.loads((BATCHES / f"{batch_id}.json").read_text())["points"]]
    for expert in ("e1", "e2"):
        walk = _ids(batch_id, expert)
        assert sorted(walk) == sorted(filed), \
            f"{expert}'s order of {batch_id} is not a permutation of it"


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("batch_id", listed())
def test_the_two_experts_do_not_share_an_order(batch_id):
    """The whole point of the shuffle. A shared order correlates the two
    readings of every point through fatigue and through the anchoring of a hard
    call by whatever preceded it -- and inflated agreement reads as agreement."""
    e1, e2 = _ids(batch_id, "e1"), _ids(batch_id, "e2")
    assert e1 != e2, f"both experts walk {batch_id} in the same order"
    # Not merely "not identical": a rotation or a swapped pair would pass that
    # and still hand them the same run of points. Most positions must differ.
    same = sum(1 for a, b in zip(e1, e2) if a == b)
    assert same < 0.2 * len(e1), \
        f"{same}/{len(e1)} positions of {batch_id} are shared between experts"


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_order_is_stable_across_processes():
    """DERIVED, not stored. The same person must get the same order after a
    reload and on another machine, or the progress strip stops being a place
    and a half-finished batch reshuffles under them."""
    batch_id = listed()[0]
    assert _ids(batch_id, "e1") == _ids(batch_id, "e1")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_order_moves_with_the_batch_and_the_campaign():
    """Seeded on all three, so one expert does not meet the same permutation
    in every batch of the round."""
    a, b = listed()[0], listed()[-1]
    assert a != b
    points = json.dumps([{"id": f"p{i:04d}"} for i in range(60)])
    walk_a = _run_js(f"orderedForExpert({points}, {json.dumps(a)}).map(p => p.id)", "e1")
    walk_b = _run_js(f"orderedForExpert({points}, {json.dumps(b)}).map(p => p.id)", "e1")
    assert walk_a != walk_b


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_an_unidentified_reader_gets_the_file_order():
    """Nobody selected is not a reader yet, and reordering a batch for them
    would make the order change the moment they say who they are."""
    points = [{"id": f"p{i:04d}"} for i in range(20)]
    walk = _run_js(f"orderedForExpert({json.dumps(points)}, 'b').map(p => p.id)", "")
    assert walk == [p["id"] for p in points]


# ── the queue collapses when everyone reads everything ──────────────────────
@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_full_overlap_is_detected_and_a_partial_one_is_not():
    both = {"experts": ["e1", "e2"]}
    full = [{"id": "a", "required_readers": ["e1", "e2"]},
            {"id": "b", "required_readers": ["e2", "e1"]}]
    part = [{"id": "a", "required_readers": ["e1", "e2"]},
            {"id": "b", "required_readers": ["e2"]}]
    assert _run_js(f"isFullOverlap({json.dumps(both)}, {json.dumps(full)})", "e1")
    assert not _run_js(f"isFullOverlap({json.dumps(both)}, {json.dumps(part)})", "e1")
    # A single-expert batch has no overlap to collapse, whatever it lists.
    solo = {"experts": ["e1"]}
    assert not _run_js(
        f"isFullOverlap({json.dumps(solo)}, "
        f"{json.dumps([{'id': 'a', 'required_readers': ['e1']}])})", "e1")


def test_the_queue_split_is_bypassed_at_full_overlap():
    """`primary_expert` stays round-robin in the file, so without this the two
    queues split the batch arbitrarily and the app tells an expert their
    primary samples are done with half of it left."""
    src = _js_block("function queueOf(point) {")
    assert "if (S.fullOverlap) return 'mine';" in src, \
        "queueOf still splits mine/second when everyone reads everything"
    assert "S.fullOverlap = isFullOverlap(" in APP.read_text(), \
        "nothing sets S.fullOverlap when a batch is adopted"


def test_only_a_fully_overlapped_batch_is_reordered():
    """A batch file's order is not arbitrary -- for an acquisition channel it is
    the ranking, and a split batch is not all read anyway. Reshuffling one to
    decorrelate the 5% that is double-read is the trade backwards."""
    src = _js_block("function adoptBatch(batch) {")
    assert "S.points = S.fullOverlap" in src and "orderedForExpert" in src, \
        "adoptBatch reorders regardless of whether the batch is overlapped"
    assert ": batch.points.slice()" in src, \
        "a partially-assigned batch no longer keeps the file's order"


def test_the_walk_order_is_re_derived_when_the_identity_changes():
    """Every per-expert store moves on switchExpert; the order is one of them,
    and it must be re-cut from the FILE's order rather than from the outgoing
    expert's permutation of it."""
    src = _js_block("function switchExpert(id, name) {")
    assert "orderedForExpert(S.batch.points" in src, \
        "switchExpert leaves the previous expert's walk order in place"
