"""The labelling cell, and the two places it is written down.

`src/label_cell.py` bakes the cell into the batch and reduces over it; `s2Cell`
in `app/js/cell.js` draws it, and computes it outright for a file dropped on the
window. §AL8's rule applies: a Python double cannot police a contract the
JavaScript disagrees with, so this runs the app's own functions in node against
pyproj — not against a re-implementation.

The two are allowed to differ by a millimetre (Snyder's series against PROJ);
they are NOT allowed to disagree about which pixel that is, so the snap is
compared exactly and only skipped where the point lands within 5 cm of a cell
edge, where a millimetre legitimately decides.

Skips cleanly with no node, so `pytest -q` stays green on a bare checkout.
"""
from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
#: The app's own file, loaded whole. This used to be `app/label_app.html` with
#: the functions cut back out of it by regular expression -- see the header of
#: app/js/cell.js for why that was worth ending.
CELL_JS = ROOT / "app" / "js" / "cell.js"
sys.path.insert(0, str(ROOT / "src"))

import label_cell as LC  # noqa: E402

NODE = shutil.which("node") or shutil.which("nodejs")
pyproj = pytest.importorskip("pyproj")


#: Places chosen for the things that break a zone rule, not for coverage:
#: 32V (Bergen, Stavanger), the Svalbard row, both hemispheres, the antimeridian,
#: a zone edge, and Oslo — which is what the deployed map is cut on.
PLACES = [
    (10.7522, 59.9139),      # Oslo
    (5.3221, 60.3913),       # Bergen — inside 32V, zone 31 by the naive rule
    (5.7331, 58.9700),       # Stavanger — 32V's southern half
    (15.6469, 78.2232),      # Longyearbyen — the Svalbard exception
    (20.9, 78.5),            # Svalbard, the 33X/35X boundary
    (2.9999, 60.0),          # just outside 32V, longitudinally
    (12.0001, 60.0),         # just outside 32V, the other side
    (5.9999, 55.9),          # just below 32V, latitudinally
    (-70.6483, -33.4569),    # Santiago — southern hemisphere false northing
    (28.0473, -26.2041),     # Johannesburg
    (112.7773, 37.7726),     # p0000 of b001
    (179.99, -16.5),         # the antimeridian, east side
    (-179.99, 64.5),         # the antimeridian, west side
    (-3.0001, 0.0),          # the equator, a zone edge
    (-2.9999, 0.0),
    (33.0, 71.99),           # just below the Svalbard row
]


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_the_labelled_pixel_is_the_same_size_in_both_languages():
    """The one number the paste-up used to supply from the Python side, so it
    could not disagree. It can now, so check it."""
    run = subprocess.run(
        [NODE, "-e", "console.log(require(" + json.dumps(str(CELL_JS))
         + ").LABEL_CELL_M)"], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr.strip()[:2000]
    assert int(run.stdout.strip()) == int(LC.CELL_M)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_the_cell_is_one_pixel_in_both_languages():
    # The app's file, whole, rather than a paste-up of slices of it. The old
    # version also DEFINED `LABEL_CELL_M` from the Python side before pasting,
    # which quietly made the two agree about the one number they most need to
    # be checked on; it is asserted below instead.
    src = "\n".join([
        "const C = require(" + json.dumps(str(CELL_JS)) + ");",
        "const pts = JSON.parse(process.argv[2]);",
        "console.log(JSON.stringify(pts.map(p => C.s2Cell(p[0], p[1]))));",
    ])

    rng = random.Random(20260831)
    pts = list(PLACES)
    while len(pts) < 400:
        pts.append([round(rng.uniform(-180, 180), 6),
                    round(rng.uniform(-80, 84), 6)])
    pts = [list(p) for p in pts]

    script = Path(__file__).parent / "_cell.cjs"
    script.write_text(src)
    try:
        run = subprocess.run([NODE, str(script), json.dumps(pts)],
                             capture_output=True, text=True)
        # Not check=True: a CalledProcessError prints the whole 400-point
        # argument vector and buries node's one-line reason in it.
        assert run.returncode == 0, run.stderr.strip()[:2000]
        got = json.loads(run.stdout)
    finally:
        script.unlink()

    edge_skips = 0
    for (lon, lat), js in zip(pts, got):
        py = LC.cell(lon, lat)
        assert js["epsg"] == py["epsg"], f"zone disagreement at {lon},{lat}"

        # Where the forward transforms agree to a millimetre but the point sits
        # a millimetre from a pixel edge, the two floors legitimately differ.
        # Count those rather than pretend; assert the rest exactly.
        fwd = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{py['epsg']}",
                                          always_xy=True)
        x, y = fwd.transform(lon, lat)
        if min(x % 10, 10 - x % 10, y % 10, 10 - y % 10) < 0.05:
            edge_skips += 1
            continue
        assert (js["x0"], js["y0"]) == (py["x0"], py["y0"]), (
            f"different pixel at {lon},{lat}")
        # Compared in METRES, not degrees: a degree tolerance is a different
        # tolerance at 78 N from the one it is at the equator, and the corners
        # are a picture on the ground. Snyder's series is sub-millimetre within
        # 3 deg of a central meridian and centimetre-level at the edge of the
        # 12 deg-wide Svalbard zones, which is the number this allows for.
        for a, b in zip(js["ring"], py["ring"]):
            dx = (a[0] - b[0]) * 111320 * math.cos(math.radians(b[1]))
            dy = (a[1] - b[1]) * 110574
            assert math.hypot(dx, dy) < 0.05, (
                f"corners {math.hypot(dx, dy):.3f} m apart at {lon},{lat}")
    assert edge_skips < len(pts) * 0.05


def test_the_cell_is_snapped_and_not_centred():
    """The property the whole change is for: the ring is a pixel of the grid
    the deployed raster is written on, and the point is somewhere inside it
    rather than at its centre."""
    c = LC.cell(10.7522, 59.9139)
    assert c["x0"] % LC.CELL_M == 0 and c["y0"] % LC.CELL_M == 0
    fwd = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{c['epsg']}",
                                      always_xy=True)
    x, y = fwd.transform(10.7522, 59.9139)
    assert c["x0"] <= x < c["x0"] + LC.CELL_M
    assert c["y0"] <= y < c["y0"] + LC.CELL_M
    # ... and it is NOT the point-centred square, which is what §AL11 drew.
    assert abs((c["x0"] + LC.CELL_M / 2) - x) > 1e-9


def test_two_points_in_one_pixel_get_one_cell():
    """Two addresses of the same pixel are the same labelling unit — which is
    the property that makes a repeat reading a repeat reading."""
    a = LC.cell(10.7522, 59.9139)
    fwd = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:%d" % a["epsg"],
                                      always_xy=True)
    inv = pyproj.Transformer.from_crs("EPSG:%d" % a["epsg"], "EPSG:4326",
                                      always_xy=True)
    x, y = fwd.transform(10.7522, 59.9139)
    lon2, lat2 = inv.transform(a["x0"] + 9.4, a["y0"] + 0.6)
    b = LC.cell(lon2, lat2)
    assert (a["x0"], a["y0"], a["epsg"]) == (b["x0"], b["y0"], b["epsg"])
