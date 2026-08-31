"""Can the deployed labelling app draw its auxiliary layers with nobody signed in?

    $G src/check_ee_service.py --url 'https://script.google.com/.../exec' \
        [--token <submitToken>] [--point lon,lat]

The app serves the Earth Engine overlays on the campaign's **service account**
(`eeAuthMode: 'auto'|'service'`, `app/apps_script/Code.gs` mints the token), so
"do the overlays draw" is a question about one identity's IAM and not about the
page. It has already been answered wrongly once, in the expensive direction: the
deployment self-test probed `value:compute`, passed, and every overlay in the
app came back empty because **`earthengine.maps.create` is a separate IAM
permission from `earthengine.computations.create`** -- found by a labeller
rather than by a test (see `ACTIVE_LEARNING.md` §AL9).

`Code.gs` grew `eeTokenSelfTest()` and `eeMapsSelfTest()` for the two halves,
but both live inside the Apps Script editor, which means they are run by hand,
by whoever remembers, and never in CI. This is the same walk from a terminal,
against the deployment as a labeller's browser sees it -- and it adds the step
neither of those makes: **fetching an actual tile**. `getMapId` validates the
request; a band name that does not match, an empty collection or a memory limit
all mint cleanly and fail per tile, which is the other silent failure the same
section closed.

Six steps, each printed with what a failure means:

  1. the deployment answers at all                    (`action=ping`)
  2. it mints a service-account token                 (`action=ee_token`)
  3. that token may compute            `earthengine.computations.create`
                                     + `serviceusage.services.use`
  4. that token may mint map tiles     `earthengine.maps.create`
  5. a tile actually renders
  6. a real overlay from the panel renders  (Dynamic World 2024, as `dw24`
     builds it)

Exit status is 0 only if every step passes, so it can gate a campaign.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

EE_ROOT = "https://earthengine.googleapis.com/v1"

#: Step 5 only asks whether the tile endpoint answers for the minted map, and
#: `ee.Image(1)` is defined everywhere, so any valid z/x/y will do.
WORLD_TILE = (0, 0, 0)

OK, BAD, INFO = "✓", "✗", "·"


def say(state: str, label: str, detail: str = "") -> None:
    print(f"{state} {label}" + (f" — {detail}" if detail else ""), flush=True)


def advice(msg: str) -> str:
    """The terminal half of `eeFailureAdvice` in `label_app.html`.

    Kept in step with it deliberately: a labeller reading the panel and the
    person reading this should be told the same thing about the same 403.
    """
    low = msg.lower()
    if "401" in low or "unauthenticated" in low or "invalid authentication" in low:
        return ("This is the TOKEN, not IAM: Earth Engine did not accept it at "
                "all. Check EE_SERVICE_ACCOUNT_KEY in the Apps Script's Script "
                "Properties -- a key that came through a form with its newlines "
                "eaten, or one whose account has since been deleted, both land "
                "here.")
    if any(w in low for w in ("permission", "denied", "forbidden",
                             "not authori", "403")):
        return (
            "This is IAM on the service account, not the app. Grant it "
            "roles/serviceusage.serviceUsageConsumer AND an Earth Engine role "
            "on the project. If the message names earthengine.maps.create and "
            "roles/earthengine.viewer is already granted, grant "
            "roles/earthengine.writer: minting map tiles is a different "
            "permission from running a computation and the account can hold "
            "one without the other. A new grant takes a minute or two to "
            "propagate.")
    if any(w in low for w in ("quota", "rate limit", "429", "too many")):
        return "Earth Engine is rate-limiting this project; wait and re-run."
    return ""


def get_json(url: str, timeout: int = 60) -> tuple[int, object]:
    req = urllib.request.Request(url, headers={"Cache-Control": "no-store"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            body = res.read().decode("utf-8", "replace")
            code = res.getcode()
    except urllib.error.HTTPError as err:                      # 4xx/5xx
        body, code = err.read().decode("utf-8", "replace"), err.code
    except (urllib.error.URLError, OSError) as err:
        # A typo in the URL, a dropped network, a deployment that was deleted.
        # This is the most likely first failure and the one that must not
        # arrive as a traceback.
        return 0, {"error": f"could not reach it: {err}"}
    try:
        return code, json.loads(body)
    except json.JSONDecodeError:
        return code, body


def post_json(url: str, payload: dict, token: str, project: str,
              timeout: int = 120) -> tuple[int, object]:
    """POST to Earth Engine exactly as the browser SDK does.

    `X-Goog-User-Project` is the header that makes this test mean anything: it
    is what the SDK sets from `ee.initialize`'s project argument and what
    triggers the *serviceusage* check. Without it an account that is missing
    Service Usage Consumer answers 200 here and 403 to every call the app makes
    -- i.e. the probe passes on exactly the deployment that is broken.
    """
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
        "X-Goog-User-Project": project,
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.getcode(), json.loads(res.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", "replace")
        try:
            return err.code, json.loads(body)
        except json.JSONDecodeError:
            return err.code, body
    except (urllib.error.URLError, OSError) as err:
        return 0, {"error": f"could not reach Earth Engine: {err}"}


def error_text(body: object) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if err:
            return str(err)
        return json.dumps(body)[:300]
    return str(body)[:300]


def exec_url(base: str, action: str, token: str | None) -> str:
    q = {"action": action}
    if token:
        q["token"] = token
    sep = "&" if "?" in base else "?"
    return base + sep + urllib.parse.urlencode(q)


def a_point(spec: str | None) -> tuple[float, float, float]:
    """lon, lat, cell_km — from `--point`, else the first baked batch point."""
    if spec:
        lon, lat = (float(x) for x in spec.split(",")[:2])
        return lon, lat, 5.0
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in sorted(glob.glob(os.path.join(here, "app", "batches", "b*.json"))):
        with open(path) as fh:
            batch = json.load(fh)
        pts = batch.get("points") or []
        if pts:
            p = pts[0]
            return float(p["lon"]), float(p["lat"]), float(p.get("cell_km") or 5)
    return 10.75, 59.91, 5.0          # Oslo, if there is no batch to read


# ── the steps ───────────────────────────────────────────────────────────────
def step_ping(base: str, token: str | None) -> bool:
    code, body = get_json(exec_url(base, "ping", token))
    if code == 200 and isinstance(body, dict) and body.get("ok"):
        say(OK, "the deployment answers",
            f"{body.get('service', '?')} v{body.get('version', '?')}")
        return True
    say(BAD, "the deployment answers",
        error_text(body) if code == 0 else f"HTTP {code}: {error_text(body)}")
    if code == 200:
        print("  A 200 that is not JSON is usually the Apps Script sign-in "
              "page: re-deploy the web app with 'Who has access: Anyone'.")
    return False


def step_token(base: str, token: str | None) -> dict | None:
    code, body = get_json(exec_url(base, "ee_token", token))
    if code != 200 or not isinstance(body, dict):
        say(BAD, "it mints a service-account token",
            f"HTTP {code}: {error_text(body)}")
        return None
    if not body.get("access_token"):
        if body.get("configured") is False or body.get("ok") is True:
            # `ok:true` with no token is an older Code.gs answering with its
            # ping, which is the same thing as far as the app is concerned.
            say(BAD, "it mints a service-account token",
                "this deployment has NO service account")
            print("  The app will fall back to the Google sign-in button, "
                  "which is what the campaign account exists to avoid. Paste "
                  "the key into Script Properties as EE_SERVICE_ACCOUNT_KEY "
                  "and RE-DEPLOY the web app (a new version, not just a save "
                  "-- /exec keeps serving the version it was deployed with).")
        else:
            say(BAD, "it mints a service-account token", error_text(body))
        return None
    say(OK, "it mints a service-account token",
        f"{body.get('client_email', '?')} on {body.get('project', '?')}, "
        f"{body.get('expires_in', '?')} s left")
    if body.get("scope") and body["scope"] != \
            "https://www.googleapis.com/auth/earthengine":
        say(INFO, "  scope is not earthengine alone", str(body["scope"]))
    return body


def step_compute(tok: dict) -> bool:
    code, body = post_json(
        f"{EE_ROOT}/projects/{tok['project']}/value:compute",
        {"expression": {"values": {"0": {"constantValue": 1}}, "result": "0"}},
        tok["access_token"], tok["project"])
    if code == 200:
        say(OK, "the token may compute", "earthengine.computations.create")
        return True
    msg = error_text(body)
    say(BAD, "the token may compute", f"HTTP {code}: {msg}")
    hint = advice(f"HTTP {code} {msg}")
    if hint:
        print("  " + hint)
    return False


def step_maps(tok: dict) -> str | None:
    """`maps.create` on `ee.Image(1)`, serialised as the client serialises it.

    A constant is not accepted here -- it has to be an IMAGE -- which is the
    detail that made the original self-test look sufficient when it was not.
    """
    code, body = post_json(
        f"{EE_ROOT}/projects/{tok['project']}/maps",
        {"expression": {"result": "0", "values": {"0": {
            "functionInvocationValue": {
                "functionName": "Image.constant",
                "arguments": {"value": {"constantValue": 1}}}}}},
         # REQUIRED, and its absence is indistinguishable from the failure this
         # step exists to find: without it `maps.create` answers HTTP 400
         # "IMAGE_FILE_FORMAT_UNSPECIFIED" however complete the account's IAM
         # is, so the probe can never go green and every run reads as "cannot
         # draw". The client sends it too -- `convert_to_image_file_format
         # (None)` in ee/_cloud_api_utils.py is exactly this value.
         "fileFormat": "AUTO_JPEG_PNG"},
        tok["access_token"], tok["project"])
    if code == 200 and isinstance(body, dict) and body.get("name"):
        say(OK, "the token may mint map tiles", "earthengine.maps.create")
        return str(body["name"])
    msg = error_text(body)
    say(BAD, "the token may mint map tiles", f"HTTP {code}: {msg}")
    hint = advice(f"HTTP {code} {msg}")
    if hint:
        print("  " + hint)
    if code == 400:
        print("  A 400 is the REQUEST, not the account: this probe is malformed "
              "and the answer says nothing about what the service account may "
              "do. Fix the payload above before reading anything into it.")
    else:
        print("  This is THE failure that emptied every overlay in the app "
              "while the compute probe above passed.")
    return None


def tile_xy(lon: float, lat: float, zoom: int) -> tuple[int, int, int]:
    """Web-mercator tile containing a point, so step 6 asks for the tile the
    labeller would actually be looking at rather than one at zoom 0."""
    n = 2 ** zoom
    lat_r = math.radians(lat)
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi)
            / 2.0 * n)
    return zoom, x, y


def step_tile(name: str, label: str = "a tile actually renders",
              tile: tuple[int, int, int] = WORLD_TILE) -> bool:
    z, x, y = tile
    url = f"{EE_ROOT}/{name}/tiles/{z}/{x}/{y}"
    # No Authorization header, deliberately: MapLibre loads rasters as IMAGES
    # and sends none, the map id carrying its own credential. Adding one here
    # would test a request the app never makes.
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            blob = res.read()
        say(OK, label, f"{len(blob)} bytes of imagery")
        return True
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", "replace")[:300]
        say(BAD, label, f"HTTP {err.code}: {body}")
        hint = advice(f"HTTP {err.code} {body}")
        print("  " + (hint or "The map minted and the tiles do not render: "
                              "this is the expression failing at render time, "
                              "which is what MapLibre reports to eeTileError() "
                              "in the app."))
        return False
    except Exception as err:                      # noqa: BLE001 - reported
        say(BAD, label, str(err))
        return False


def step_real_layer(tok: dict, lon: float, lat: float, km: float) -> bool | None:
    """`dw24` from the panel, built the way `dwMode`/`eeBox` build it.

    Steps 4 and 5 prove the account may draw *something*. This one is the
    question the labeller actually asked -- an overlay from the panel, over a
    real point -- and it is the one that would catch an expression that mints
    and then fails per tile.

    @return None when the `ee` client is not importable, which is a skip.
    """
    try:
        import ee                                   # noqa: PLC0415 - optional
        from google.oauth2.credentials import Credentials  # noqa: PLC0415
    except ImportError:
        say(INFO, "a real overlay renders (dw24)",
            "skipped: no `ee` client in this interpreter")
        return None
    try:
        ee.Initialize(credentials=Credentials(tok["access_token"]),
                      project=tok["project"])
        span = km * 1.4
        d_lat = span / 2 / 110.574
        d_lon = span / 2 / (111.320 * abs(math.cos(math.radians(lat))))
        box = ee.Geometry.Rectangle(
            [lon - d_lon, lat - d_lat, lon + d_lon, lat + d_lat], None, False)
        img = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
               .filterDate("2024-01-01", "2025-01-01")
               .filterBounds(box).select("label").mode().clip(box))
        got = img.getMapId({"min": 0, "max": 8})
    except Exception as err:                        # noqa: BLE001 - reported
        msg = str(err)
        say(BAD, "a real overlay renders (dw24)", msg[:300])
        hint = advice(msg)
        if hint:
            print("  " + hint)
        return False
    say(OK, "a real overlay mints (dw24)", "Dynamic World 2024 at the point")
    # And the half getMapId cannot answer for: a tile at the point itself.
    return step_tile(got["mapid"], "its tiles render at the point",
                     tile_xy(lon, lat, 13))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", required=True,
                    help="the Apps Script web app /exec URL from config.js")
    ap.add_argument("--token", default=os.environ.get("LABEL_SUBMIT_TOKEN", ""),
                    help="submitToken, if the deployment sets SUBMIT_TOKEN")
    ap.add_argument("--point", help="lon,lat to test a real overlay over "
                                    "(default: the first point of the first "
                                    "baked batch)")
    args = ap.parse_args()

    base = args.url.strip()
    print(f"deployment: {base.split('?')[0]}")
    if not step_ping(base, args.token):
        return 1
    tok = step_token(base, args.token)
    if not tok:
        return 1
    if not tok.get("project"):
        say(BAD, "the deployment names a project",
            "no `project` in the token response; set EE_PROJECT in Script "
            "Properties or use a key whose project_id is the registered one")
        return 1

    ok = step_compute(tok)
    name = step_maps(tok)
    ok = bool(name) and ok
    if name:
        ok = step_tile(name) and ok
    if ok:
        lon, lat, km = a_point(args.point)
        if step_real_layer(tok, lon, lat, km) is False:
            ok = False
    else:
        say(INFO, "a real overlay renders (dw24)",
            "skipped: fix the above first, this can only repeat it")

    print()
    if ok:
        print("The campaign account can draw the auxiliary layers. Labellers "
              "will not be asked to sign in.")
    else:
        print("The auxiliary layers will NOT draw for labellers. Everything "
              "else in the app -- the baked chips, the dense series, the "
              "evidence rows, saving -- is unaffected.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
