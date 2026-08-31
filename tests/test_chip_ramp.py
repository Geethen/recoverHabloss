"""The chip display ramp, and the two places it is written down.

The bake renders a sprite through `combo_bounds()` in Python; the app paints the
tint under that sprite, and every dot on the annual chart, through
`comboBounds()` in JavaScript. They are one contract, and a disagreement is
invisible in exactly the way this project keeps getting caught by: the picture
still appears, it is just a different picture from the colour behind it.

So this does not re-implement the JavaScript in Python -- §AL8's lesson is that
a Python double polices a contract the JavaScript may not have signed. It runs
the app's own function, extracted from `label_app.html`, in node.

Skips cleanly where there is no node, so `pytest -q` stays green on a bare
checkout.
"""
from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
APP = ROOT / "app" / "label_app.html"
sys.path.insert(0, str(ROOT / "src"))

import build_batch_chips as C          # noqa: E402
import build_batch_dense as D          # noqa: E402

NODE = shutil.which("node") or shutil.which("nodejs")


def _js_const(name: str) -> str:
    """The literal a top-level `const NAME = ...;` is assigned."""
    m = re.search(rf"^const {re.escape(name)} = (.+?);\s*$",
                  APP.read_text(), re.M)
    assert m, f"{name} not found in label_app.html"
    return m.group(1)


def _js_block(start: str) -> str:
    """Source from `start` to the first line that is just `}` or `};`.

    The app indents everything inside a top-level declaration, so a closing
    brace at column 0 ends it. `};` as well as `}`: an object literal closes
    with the semicolon, and matching only `}` ran straight past CHIP_RGB into
    the end of the next function.
    """
    text = APP.read_text()
    i = text.index(start)
    m = re.compile(r"^\};?$", re.M).search(text, i)
    assert m, f"no closing brace for {start!r}"
    return text[i:m.end()]


def test_the_bake_version_the_app_looks_for_is_the_one_the_baker_writes():
    """A mismatch is not cosmetic: chip1 sprites were rendered through the
    fixed global bounds, and painting a chip2 tint under one shows two
    different exposures of the same pixel."""
    assert _js_const("CHIP_BAKE_VERSION") == f"'{C.CHIP_BAKE_VERSION}'"
    assert _js_const("DENSE_BAKE_VERSION") == f"'{D.DENSE_BAKE_VERSION}'"


def test_the_batch_keys_the_bakers_write_survive_parsebatch():
    """`parseBatch` is an ALLOW-LIST.

    A batch-level key it does not name is dropped without a word, and the
    feature reverts to its fallback -- live Earth Engine, or nothing. Caught
    once already on `chips`; `dense` is the same shape of mistake.
    """
    src = _js_block("function parseBatch(text, name) {")
    for key in ("chips", "dense", "evidence_schema", "evidence_version"):
        assert f"{key}: obj.{key}" in src, f"parseBatch drops batch['{key}']"


def test_a_rebake_busts_the_cache():
    """A re-bake writes new bytes to the SAME path.

    Without a stamp on the URL, every browser and CDN that already fetched a
    sprite keeps serving the old one -- and from a fresh profile everything
    looks correct, so the report comes back as "it still looks the same" with
    nothing to see on this side.
    """
    src = APP.read_text()
    assert "'?v=' + b.built" in src, "the sprite URL carries no bake stamp"
    for mod in (C, D):
        assert '"built": int(time.time())' in (
            Path(mod.__file__).read_text()), f"{mod.__name__} writes no stamp"


def test_the_fallback_says_which_condition_it_failed():
    """Four ways a bake is silently not used, and a strip of flat tints looks
    identical for all of them -- and identical to no bake at all."""
    src = _js_block("function chipBakeMiss(p) {")
    for cond in ("version", "S.batchUrl", "combos", "width_m"):
        assert cond in src, f"chipBakeMiss does not name {cond}"
    assert "chipBakeMiss(p)" in _js_block("function liveChips(p, years, gen, why) {")


def test_the_min_span_guard_is_the_same_number_on_both_sides():
    assert float(_js_const("STRETCH_MIN_SPAN")) == C.STRETCH_MIN_SPAN


def test_the_app_and_the_baker_agree_on_the_combos_they_can_bake():
    """A scheme the app thinks is baked and the baker never renders is a 404
    per point, which degrades to live Earth Engine -- silently, and only for
    whoever picked that scheme."""
    rgb = _js_block("const CHIP_RGB = {")
    for combo in C.COMBOS:
        assert f"'{combo}'" in rgb, f"{combo} is bakeable but not in CHIP_RGB"


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_the_two_ramps_are_the_same_ramp():
    """`combo_bounds` (Python, bakes the sprite) against `comboBounds`
    (JavaScript, paints the tint and drives the live request), on random
    stretch tables including the degenerate ones the guard exists for."""
    src = "\n".join([
        _js_block("const CHIP_RGB = {"),
        _js_block("const CHIP_INDEX = {"),
        "const STRETCH_MIN_SPAN = " + _js_const("STRETCH_MIN_SPAN") + ";",
        "function chipIsIndex(c) { return c in CHIP_INDEX; }",
        _js_block("function chipSpec(combo) {"),
        # The app reads the table off S.batch; the test hands it in directly.
        "let STRETCH = null;",
        "function chipStretch() { return STRETCH; }",
        _js_block("function comboBounds(p, combo) {"),
        "const cases = JSON.parse(process.argv[2]);",
        "const out = cases.map(c => { STRETCH = c.stretch;",
        "  const b = comboBounds({}, c.combo);",
        "  return [b.min, b.max]; });",
        "console.log(JSON.stringify(out));",
    ])

    rng = random.Random(11)
    cases = []
    for _ in range(300):
        combo = rng.choice(sorted(C.COMBOS))
        if rng.random() < 0.15:
            stretch = None                                  # no bake
        else:
            stretch = {}
            for band in C.S2_BANDS:
                if rng.random() < 0.08:
                    stretch[band] = None                    # no clear pixel
                else:
                    lo = rng.uniform(0, 6000)
                    stretch[band] = [round(lo, 1),
                                     round(lo + rng.choice([2, 40, 300, 3000]), 1)]
        cases.append({"combo": combo, "stretch": stretch})

    script = Path(__file__).parent / "_ramp.mjs"
    script.write_text(src)
    try:
        got = json.loads(subprocess.run(
            [NODE, str(script), json.dumps(cases)],
            capture_output=True, text=True, check=True).stdout)
    finally:
        script.unlink()

    for case, js in zip(cases, got):
        _, lo, hi = C.combo_bounds(case["combo"], case["stretch"])
        assert [lo, hi] == pytest.approx(js, abs=1e-6), (
            f"{case['combo']} with {case['stretch']}: python {(lo, hi)} "
            f"vs javascript {tuple(js)}")


def test_the_dense_recipe_is_the_same_recipe():
    """`series_for` is `denseFetchLive` in Python. The sidecar it bakes is
    served to half the batch and the live call to the other half, so a drift
    here puts two different measurements on one chart."""
    live = _js_block("function denseFetchLive(p) {")
    assert f"'CLOUDY_PIXEL_PERCENTAGE', {D.CLOUDY_MAX}" in live
    assert f"lt({D.CLDPRB_MAX})" in live
    assert f"buffer({D.BUFFER_M})" in live
    assert f", {D.SCALE_M})" in live
    for band in D.BANDS:
        assert f"'{band}'" in live
    assert "MSK_CLDPRB" in live and "s2cloudless" not in live.lower(), (
        "the dense series must not use the s2cloudless join -- it made a "
        "first attempt take 103 s for 269 scenes")
