"""The Esri Wayback capture-date lookup, which was returning no date at all.

WHAT WAS WRONG, measured against the live service on 2026-09-11

`_wbFetchMeta` probed the metadata sublayers `[L0, L0+1, L0+2, L0+3, 13]` --
the band matched to the working zoom, the three next-coarser, and the base that
bounds the walk -- and then took `hits.find(Boolean)`: the first band that
returned a FEATURE, dated or not.

Band 13 (and 12) is Esri's 15 m TerraColor global base. It answers at *every*
point on Earth, with `SRC_DATE` and `SRC_DATE2` both null. So:

  * a point with no footprint in the four bands around the working zoom picked
    up the base-map row and reported the literal string "unknown date", and
  * nothing FINER than the matched band was ever requested. Band coverage is
    patchy in both directions -- Oslo under the 2018-07-25 release carries
    DigitalGlobe footprints at bands 4-9 and nothing at all at 10-11 -- so a
    point one notch less covered has only finer bands to offer. NB: no such
    point turned up in the 120 reads below; that half is a hole closed by
    argument and by the `OSLO_2018_FINER_ONLY` fixture, not an observed
    failure.

On round one's own draw, 5 of 120 (point, side) reads came back without a date,
every one of them an Arctic point (79.07N -31.74, 66.55N -45.66, 66.31N 178.16)
where no band carries a date because there genuinely is no dated survey -- only
the 15 m base. That is an ANSWER about the point and it reads completely
differently from a failed lookup, which is what "unknown date" sounded like.

The fixtures below are recorded from the live service. The test runs the app's
own JavaScript in node against them, rather than re-implementing it in Python:
§AL8's rule is that a Python double cannot police a contract the JavaScript may
not have signed.

Skips cleanly where there is no node, so `pytest -q` stays green on a bare
checkout.
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
NODE = shutil.which("node") or shutil.which("nodejs")

pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


# ---------------------------------------------------------------------------
# Loading the real functions into node
# ---------------------------------------------------------------------------
def _fn(name: str) -> str:
    """One top-level function declaration, by brace matching.

    `_js_block`'s "first line that is just `}`" trick in test_chip_ramp does
    not work here: `_wbFetchMeta` contains a `};` at column 2 and the arrow
    functions inside it close at other depths. Counting braces outside strings,
    comments and regexes is overkill for hand-written source, so this counts
    braces and skips line comments and string literals, which is all this file
    needs.
    """
    text = APP.read_text()
    m = re.search(rf"^(?:async )?function {re.escape(name)}\(", text, re.M)
    assert m, f"{name} not found in {APP.name}"
    i = text.index("{", m.start())
    depth, j, n = 0, i, len(text)
    while j < n:
        c = text[j]
        if c == "/" and text[j:j + 2] == "//":
            j = text.index("\n", j)
        elif c == "/" and text[j:j + 2] == "/*":
            j = text.index("*/", j) + 1
        elif c in "'\"`":
            q, j = c, j + 1
            while j < n and text[j] != q:
                j += 2 if text[j] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[m.start():j + 1]
        j += 1
    raise AssertionError(f"no closing brace for {name}")


#: Everything `_wbFetchMeta` reaches for that lives elsewhere in the app. The
#: query itself is the seam: the test hands it recorded rows.
HARNESS = """
const wbMetaCache = {};
const wbMetaCtl = { signal: {} };
function wbMetaZoom() { return ZOOM; }
function AnySignal(a, b) { return a || b; }
function wbTimeoutSignal(ms, outer) { return { signal: outer, done: () => {} }; }
const ASKED = [];
async function _wbQueryLayer(rel, pt, layerId, signal) {
  ASKED.push(layerId);
  return ROWS[String(layerId)] || null;
}
"""


def _run(rows: dict, zoom: int = 15, expr: str = "out") -> dict:
    """Run `_wbFetchMeta` over recorded `rows` and return its answer.

    Also returns which sublayers were asked for, because the number of requests
    per (release, point) is the cost this lookup is budgeted on -- §AL9's
    fan-out note is about exactly this function.
    """
    code = "\n".join([
        f"const ROWS = {json.dumps(rows)};",
        f"const ZOOM = {zoom};",
        HARNESS,
        _fn("wbRowDate"),
        _fn("wbRowInfo"),
        _fn("wbIso"),
        _fn("wbDateWords"),
        _fn("_wbFetchMeta"),
        "(async () => {",
        "  const out = await _wbFetchMeta({ num: 1, metaUrl: 'x' },"
        "                                 { lng: 10.75, lat: 59.91 }, 'k');",
        '  console.log(JSON.stringify({ out: out, asked: ASKED }));',
        "})();",
    ])
    run = subprocess.run([NODE, "-e", code], capture_output=True, text=True,
                         check=False)
    assert run.returncode == 0, run.stderr.strip()[:3000]
    return json.loads(run.stdout)


# ---------------------------------------------------------------------------
# Recorded from the live service, 2026-09-11
# ---------------------------------------------------------------------------
#: The 15 m global base. Present at every point on Earth, with NO date -- which
#: is the row the old code reported as "unknown date".
BASE = {"SRC_DATE": None, "SRC_DATE2": None, "NICE_DESC":
        "Earthstar Geographics", "SRC_DESC": None, "SRC_RES": 15,
        "SAMP_RES": 15}

#: Oslo, 59.9139 10.7522, Wayback release 2024-08-15. Note bands 10 and 11
#: carry the same Maxar footprint as 8-9 here, and 4-7 a DIFFERENT, earlier
#: scene -- so "which band answered" is a real choice and not a formality.
OSLO_2024 = {
    **{str(L): {"SRC_DATE": 20230822, "SRC_DATE2": 1692662400000,
                "NICE_DESC": "Maxar", "SRC_DESC": None, "SRC_RES": 0.46,
                "SAMP_RES": 0.3} for L in (4, 5, 6, 7)},
    **{str(L): {"SRC_DATE": 20230610, "SRC_DATE2": 1686355200000,
                "NICE_DESC": "Maxar", "SRC_DESC": None, "SRC_RES": 0.31,
                "SAMP_RES": 0.6} for L in (8, 9, 10, 11)},
    "12": BASE, "13": BASE,
}

#: Oslo under release 2018-07-25. The shape the old band set could not see:
#: 10 and 11 answer with NOTHING, so at the working zoom's L0=8 a point one
#: notch less well covered than this has only finer bands to offer.
OSLO_2018_FINER_ONLY = {
    **{str(L): {"SRC_DATE": 20170715, "SRC_DATE2": 1500076800000,
                "NICE_DESC": "DigitalGlobe", "SRC_DESC": None,
                "SRC_RES": 0.46, "SAMP_RES": 0.46} for L in (4, 5, 6, 7)},
    "12": BASE, "13": BASE,
}

#: cov001/c026832 at 79.071 -31.740, and the other two Arctic points behave
#: identically: not one of the fourteen sublayers carries a date.
ARCTIC = {"12": BASE, "13": BASE}


# ---------------------------------------------------------------------------
# The fault
# ---------------------------------------------------------------------------
def test_the_global_base_map_is_not_reported_as_a_capture_date():
    """The whole bug, in one assertion.

    5 of 120 reads on round one's draw hit this. `hits.find(Boolean)` took the
    base-map row because it was the only row, and the app said "unknown date"
    -- indistinguishable from a lookup that broke.
    """
    r = _run(ARCTIC)["out"]
    assert r["date"] is None, r
    assert r["base"] is True, r
    assert r["res"] == 15
    # and it says so in words, rather than as an absence
    assert "no dated survey" in _words(r)


def test_a_footprint_finer_than_the_working_zoom_is_found():
    """Nothing finer than the matched band was ever requested.

    At the configured zoom 15 the matched band is 8, and this point's only
    footprints are at 4-7. The old set -- 8, 9, 10, 11, 13 -- could only reach
    the base map, so a dated 0.46 m DigitalGlobe scene became "unknown date".
    """
    got = _run(OSLO_2018_FINER_ONLY)
    assert got["out"]["date"] == "2017-07-15", got["out"]
    assert got["out"].get("base") is not True
    assert got["out"]["provider"] == "DigitalGlobe"
    # it took a second wave to get there, and the finer bands were in it
    assert set(got["asked"]) >= {4, 5, 6, 7}, got["asked"]


def test_band_12_is_no_longer_skipped():
    """`L0..L0+3` then a jump to 13 left band 12 out of the walk entirely."""
    asked = _run({"12": BASE, "13": BASE})["asked"]
    assert 12 in asked, asked


# ---------------------------------------------------------------------------
# What it must not cost
# ---------------------------------------------------------------------------
def test_the_common_case_still_costs_one_wave():
    """§AL9's fan-out note is about this function: it opened ~40 requests per
    point. A well-covered point -- 115 of 120 on round one's draw -- must
    resolve in the same five requests it always did, with the finer bands never
    asked for."""
    got = _run(OSLO_2024)
    assert got["out"]["date"] == "2023-06-10", got["out"]
    assert sorted(got["asked"]) == [8, 9, 10, 11, 13], got["asked"]


def test_the_band_matched_to_the_working_zoom_wins():
    """Oslo 2024 carries two different Maxar scenes: 2023-08-22 in bands 4-7
    and 2023-06-10 in 8-11. A tile at zoom 15 is drawn from band 8, so 8's date
    is the one that describes what is on screen -- picking the finest available
    would caption the picture with a scene the interpreter is not looking at."""
    assert _run(OSLO_2024)["out"]["band"] == 8


# ---------------------------------------------------------------------------
# Date parsing: the service does not only issue YYYYMMDD
# ---------------------------------------------------------------------------
def _words(meta) -> str:
    code = "\n".join([
        _fn("wbDateWords"),
        f"console.log(wbDateWords({json.dumps(meta)}));",
    ])
    run = subprocess.run([NODE, "-e", code], capture_output=True, text=True,
                         check=False)
    assert run.returncode == 0, run.stderr.strip()[:2000]
    return run.stdout.strip()


def _row_date(row):
    code = "\n".join([
        _fn("wbRowDate"),
        f"console.log(JSON.stringify(wbRowDate({json.dumps(row)})));",
    ])
    run = subprocess.run([NODE, "-e", code], capture_output=True, text=True,
                         check=False)
    assert run.returncode == 0, run.stderr.strip()[:2000]
    return json.loads(run.stdout)


@pytest.mark.parametrize("raw,date,precision", [
    (20230610, "2023-06-10", "day"),
    ("20230610", "2023-06-10", "day"),
    (201907, "2019-07", "month"),          # a month is still provenance
    (2014, "2014", "year"),                # and so is a year
    ("", None, None),
    (None, None, None),
    (20239999, None, None),                # not a date, whatever it is
])
def test_partial_dates_survive(raw, date, precision):
    """`/^\\d{8}$/` was the only accepted form, and a month or a year fell
    through to `SRC_DATE2`, which on those same rows is null. A 2019 date is a
    perfectly good note on a 2018-vs-2024 call and was being discarded."""
    got = _row_date({"SRC_DATE": raw, "SRC_DATE2": None})
    if date is None:
        assert got is None
    else:
        assert got == {"date": date, "precision": precision}


def test_epoch_milliseconds_are_still_the_fallback():
    """SRC_DATE2 is epoch ms and carries the date on rows where SRC_DATE does
    not. `new Date(str)` on a numeric string is not the same parse, so it is
    coerced."""
    assert _row_date({"SRC_DATE": None, "SRC_DATE2": 1686355200000}) == {
        "date": "2023-06-10", "precision": "day"}
    assert _row_date({"SRC_DATE": None, "SRC_DATE2": "1686355200000"}) == {
        "date": "2023-06-10", "precision": "day"}


# ---------------------------------------------------------------------------
# Downstream: a null date must not reach the record or the read-out as "null"
# ---------------------------------------------------------------------------
def test_partial_dates_are_usable_downstream():
    """The gap warning and the release snap both tested for a full ISO day and
    threw a month or a year away. `wbIso` is the one place that decides."""
    code = "\n".join([
        _fn("wbIso"),
        "const t = m => wbIso(m);",
        "console.log(JSON.stringify(["
        "t({date:'2023-06-10'}), t({date:'2019-07'}), t({date:'2014'}),"
        "t({date:null, base:true}), t(null), t({date:'unknown date'})]));",
    ])
    run = subprocess.run([NODE, "-e", code], capture_output=True, text=True,
                         check=False)
    assert run.returncode == 0, run.stderr.strip()[:2000]
    assert json.loads(run.stdout) == [
        "2023-06-10", "2019-07", "2014", None, None, None]


def test_the_readout_never_interpolates_a_null_date():
    """`'<b>' + m.date + '</b>'` printed the string "null" for the base-map
    case, which is what the report of this bug looked like on screen."""
    src = APP.read_text()
    i = src.index("const fmtM = m =>")
    fmt = src[i:src.index("\n\n", i)]
    assert "wbDateWords" in fmt, "fmtM no longer routes through wbDateWords"
    assert "'<b>' + m.date" not in fmt


def test_the_record_distinguishes_absent_from_unknown():
    """`imagery_a`/`imagery_b` are the provenance of a call. "no dated survey"
    and "we could not find out" are different facts about a point and an
    interpreter comparing two calls needs to know which it was."""
    src = _fn("wbCaptureNote")
    assert "m.base" in src, "wbCaptureNote cannot tell the base map apart"
    assert "no dated survey" in src


# ---------------------------------------------------------------------------
# The chip lightbox: the play button, and the frame stack that makes it smooth
# ---------------------------------------------------------------------------
def test_the_lightbox_keeps_one_frame_per_year():
    """Stepping a year used to blank the <img>, show the sprite at a DIFFERENT
    size, then reveal the img the moment `src` was assigned -- before a byte of
    it had decoded. Four repaints and two reflows per arrow press, repeated in
    full when stepping back to a year already seen. The fix is a frame per
    year, kept, plus a fixed square stage."""
    src = APP.read_text()
    assert "const lbFrames = new Map()" in src
    # the frame is appended on `onload`, never on `src =`
    load = _fn("lbLoadFrame")
    assert "img.onload" in load and "appendChild" in load
    assert "img.src = url" in load
    assert load.index("img.onload") < load.index("img.src = url"), \
        "the handler must be bound before the src, or a cached image misses it"
    # and nothing blanks the picture any more -- the header keeps the old line
    # as the record of what the fault was, so this asks the code, not the file
    for fn in ("renderLightbox", "lbShowFrame", "lbStep"):
        assert "removeAttribute('src')" not in _fn(fn), fn


def test_the_film_prefetches_capped_frames_and_upgrades_only_the_one_read():
    """A cost decision the film changed, and the old choice was right for one
    year and wrong for nine.

    The lightbox fetched the UNCAPPED composite, on AL9's principle that an
    interpreter who has deliberately enlarged a year is waiting on purpose. AL9
    also timed uncapped chips at 15-41 s each against 5-6 s for the 12-scene
    cap, for a median 0.002 relative reflectance difference at the plot -- so
    nine of them is minutes of Earth Engine per lightbox open, mostly on years
    nobody looks at, and the film cannot start until they land.
    """
    load = _fn("lbLoadFrame")
    # capped unless asked otherwise, and the ask is the caller's
    assert "const o = full ? { full: true, dim: CHIP_DIM_BIG }" in load
    assert ": { dim: CHIP_DIM_BIG }" in load
    # the prefetch does NOT ask for full; the year on screen does
    assert "full" not in _fn("lbPrefetch")
    assert "lbLoadFrame(y, { full: true })" in _fn("renderLightbox")
    # and NOT while the film is running -- a year the film passes through is not
    # a year being read, and upgrading each in turn would request all nine
    # uncapped composites over one play-through, which is the whole spend the
    # capped prefetch exists to avoid
    assert "!lbPlaying()" in _fn("renderLightbox")
    assert "{ full: true }" in _fn("lbPause"), \
        "stopping on a year is choosing to read it; that is when to upgrade"
    # and an upgrade replaces the capped frame rather than stacking on it
    assert "old.remove()" in load
    assert "lbFrames.set(year, img)" in load


def test_a_prefetch_slot_is_released_even_if_its_year_never_arrives():
    """`chipUrl` has no timeout of its own. Two unanswered years would hold both
    slots and the rest of the film would never be asked for at all."""
    src = APP.read_text()
    assert "const LB_SLOT_MS" in src
    assert "lbSlot(lbLoadFrame(y))" in _fn("lbPrefetch")
    slot = _fn("lbSlot")
    assert "setTimeout(done, LB_SLOT_MS)" in slot
    # it frees the SLOT, not the request -- a late picture is still usable
    assert "abort" not in slot


def test_the_lightbox_has_a_play_button_on_a_timeout_chain():
    """setInterval keeps firing while a frame is still decoding and the film
    runs ahead of its pictures."""
    src = APP.read_text()
    assert "function lbPlay()" in src and "function lbPause()" in src
    assert "setInterval" not in _fn("lbPlay")
    assert "setTimeout(tick, lbSpeed)" in _fn("lbPlay")
    # closing the lightbox, or navigating away from it, must stop the film
    assert "lbPause();" in _fn("closeLightbox")


def test_the_film_skips_years_with_no_composite():
    """A year with no cloud-free growing-season composite renders BLACK, which
    reads as a broken image rather than as an absence. The strip already marks
    those cells `nodata`; the film must not stop on one and neither should the
    arrow keys."""
    assert "lbNoData" in _fn("lbStep")
    assert "evVisColor" in _fn("lbNoData")


def test_a_scheme_or_width_change_drops_the_frames():
    """A frame is a picture of one point under one scheme at one width. Held
    across a scheme change, the new caption sits over the old bands."""
    src = _fn("applyChipVis")
    assert "lbDropFrames()" in src
    assert "lbFrameKey()" in src
