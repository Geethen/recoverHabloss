/**
 * Deployment config for label_app.html.
 *
 * This file is separate from the app on purpose: re-deploying label_app.html
 * must never clobber the Sheet URL or the OAuth client id. Edit here, leave the
 * app alone. Every field is optional -- the app runs with all of them empty and
 * you export labels by hand.
 *
 * WHAT THIS FILE IS AND IS NOT. It configures a LOCAL serve. The published
 * Pages app does not read it: .github/workflows/pages.yml OVERWRITES it with
 * the LABEL_APP_CONFIG_JS Actions secret at build time, and a missing secret
 * fails the build rather than publishing an app with no backend. So a blank
 * field here is not a statement about the deployment -- ask the deployment:
 *
 *     curl '<sheetUrl>?action=ping'
 *       -> "token_required": …, "ee_service_account": …
 *
 * The Earth Engine private key is NOT here and cannot be -- see `eeAuthMode`
 * below. It lives in the Apps Script's Script Properties, server-side, and the
 * page is handed a one-hour token instead.
 */
window.LABEL_APP_CONFIG = {

  // ── Google Sheet backend ──────────────────────────────────────────────────
  // The /exec URL from Deploy > New deployment > Web app in the Sheet's Apps
  // Script editor (see apps_script/Code.gs). Empty => local-only mode: labels
  // are kept in the browser and exported with the "export" button.
  sheetUrl: '',

  // Shared secret matching SUBMIT_TOKEN in apps_script/Code.gs. NOT a secret --
  // it ships inside this file, which the browser downloads -- but it stops
  // anyone who merely finds the /exec URL from writing rows to your sheet.
  // Leave both sides empty to disable the check.
  submitToken: '',

  // ── Earth Engine ──────────────────────────────────────────────────────────
  // How the page gets an Earth Engine token.
  //
  //   'auto'    (default) the deployment's own service account when the Apps
  //             Script has one, and the sign-in button when it does not
  //   'service' the service account only -- the button retries it and never
  //             offers a Google sign-in
  //   'oauth'   the old behaviour: everyone signs in with their own account
  //   'off'     no Earth Engine at all
  //
  // 'auto' and 'service' are what stop a labeller ever seeing a Google popup.
  // The key itself is NOT here and cannot be: see "Earth Engine without a
  // sign-in" in README.md, and the note at the top of the Earth Engine section
  // in label_app.html for why the SDK refuses a browser-side private key. It
  // lives in the Apps Script's Script Properties and this page is handed a
  // one-hour token.
  eeAuthMode: 'service',

  // Where that token comes from. Empty => the Apps Script already in
  // `sheetUrl`, which is where apps_script/Code.gs serves it. Set this only if
  // you moved the broker somewhere else (a Cloud Function, say).
  eeTokenUrl: '',

  // The Cloud project registered for Earth Engine. With a service account this
  // is filled in from the key and nobody has to know it; it stays here for the
  // sign-in fallback, where it is the one field the person signing in supplies.
  eeProject: 'ee-gsingh',

  // The OAuth fallback, used when `eeAuthMode` is 'oauth' or when it is 'auto'
  // and the deployment has no service account. An OAuth *Web application*
  // client id from that project, with every origin
  // this app is served from listed under "Authorised JavaScript origins". Made
  // ONCE and then shared by everyone who opens the page; the panel walks you
  // through making it and remembers it per browser, so this field is the way to
  // set it for everybody at once. It cannot be borrowed from another app (a
  // web client validates the origin exactly), which is why there is no default.
  eeClientId: '',

  // ── the expert roster ─────────────────────────────────────────────────────
  // Every label row is keyed by (campaign, batch, point, expert_id), and
  // `expert_id` is what comes from this list. A typed name is not an identity:
  // "Ann", "ann", "Ann " and "Anne" are four experts to a groupby and the
  // failure is silent until the round report. So the header is a dropdown fed
  // from here, `?expert=e1` is a per-person bookmark, and free text is an
  // explicit fallback rather than the default.
  //
  // `id` is permanent and goes in the sheet; `name` is a display label and may
  // be changed at will. Never re-use an id for a different person.
  experts: [],

  // ── campaign ──────────────────────────────────────────────────────────────
  campaign: 'recover-habloss',
  manifest: 'batches/index.json',

  // The campaign's full class definitions, linked from the legend fold and from
  // its summary. The app falls back to the RECOVER cribsheet when this is
  // empty, so a dragged-in copy of app/ still has the definitions one click
  // from the call. Point it elsewhere for a campaign on a different typology --
  // and if you do, change CLASSES[].hint in the app to match, because the two
  // teaching the same interpreter two legends is the failure the calibration
  // batches exist to catch.
  cribsheetUrl: '',

  // Zoom the map jumps to for each new point. 15 shows roughly the 5 km cell;
  // the interpreter's own zoom is remembered after the first adjustment, so
  // this is a starting scale, not a cage.
  pointZoom: 15
};
