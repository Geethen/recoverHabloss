// ESLint flat config for the labelling app.
//
// Named `.mjs` on purpose: this repository has no `package.json`, so node reads
// a bare `.js` as CommonJS and `export default` would fail. Keeping it that way
// is the point -- the app is served as static files with no build step and no
// `node_modules`, and `pre-commit` supplies eslint from its own managed
// environment. Nothing here obliges anybody to install a JavaScript toolchain
// to work on the app.
//
// THE GLOBALS ARE WRITTEN OUT RATHER THAN IMPORTED FROM THE `globals` PACKAGE.
// Importing it does not work: eslint resolves the config's imports from the
// repository, which has no node_modules, so the package pre-commit installs
// into its own environment is not visible here. Writing them out is better
// anyway. This list is not a guess -- it is every free identifier eslint
// actually reports in the app's JavaScript, and there were exactly 25 of them.
// A blanket `globals.browser` would also declare `name`, `top`, `length`,
// `status`, `event`, `open` and `find`, which are the classic accidental
// globals: declaring those tells eslint that a typo referring to one of them
// is fine, which is most of what `no-undef` is for.
//
// Re-derive it, rather than adding to it by hand, if the app starts using a
// browser API that is not here: lint with an empty `globals` and read off the
// `no-undef` names.

export default [
  {
    // Vendored library, baked batch data. The batches are also JSON-with-a-
    // wrapper in places and are checked by `check-json` instead.
    ignores: ["app/vendor/**", "app/batches/**"],
  },
  {
    files: ["app/**/*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      // CLASSIC scripts, not modules. app/label_app.html loads cell.js,
      // chips.js and app.js with plain <script> tags and they share one global
      // scope, which is what the single inline block they came from did. See
      // the header of js/app.js for why `type="module"` is a separate job.
      sourceType: "script",
      globals: {
        // --- the DOM and the page
        window: "readonly",
        document: "readonly",
        location: "readonly",
        history: "readonly",
        navigator: "readonly",
        localStorage: "readonly",
        console: "readonly",
        alert: "readonly",
        Image: "readonly",
        Event: "readonly",
        ResizeObserver: "readonly",
        requestAnimationFrame: "readonly",
        setTimeout: "readonly",
        clearTimeout: "readonly",
        setInterval: "readonly",
        // --- fetching
        fetch: "readonly",
        Blob: "readonly",
        URL: "readonly",
        URLSearchParams: "readonly",
        AbortController: "readonly",
        AbortSignal: "readonly",
        TextEncoder: "readonly",
        btoa: "readonly",
        // --- loaded as classic scripts before the app module
        maplibregl: "readonly",
        // The Earth Engine client, loaded on demand for the overlays only.
        ee: "readonly",
        // The `typeof module !== 'undefined'` export guards at the foot of
        // cell.js and chips.js, which exist so the tests can load those files
        // in node. Undefined in the browser; that is the point.
        module: "readonly",
      },
    },
    rules: {
      "no-undef": "error",
      "no-redeclare": "error",
      "no-dupe-keys": "error",
      "no-dupe-args": "error",
      "no-dupe-else-if": "error",
      "no-duplicate-case": "error",
      "no-unsafe-negation": "error",
      "no-unreachable": "error",
      "no-fallthrough": "error",
      "no-self-assign": "error",
      "no-constant-condition": ["error", { checkLoops: false }],
      "valid-typeof": "error",
      "use-isnan": "error",
      // `args: "none"` because a handler that ignores its event argument is
      // normal and not a defect.
      //
      // `caughtErrors: "none"` for the same reason, and it is not a default:
      // eslint 9 changed it to "all", which reports every one of this app's
      // deliberate `catch (e) { /* private mode */ }` -- 28 of them, all
      // correct. The alternative is renaming each to `catch { }`, which is a
      // sweep through the file to silence a linter, i.e. exactly the kind of
      // churn a linter is supposed to save you from.
      "no-unused-vars": ["error", { args: "none", caughtErrors: "none" }],
    },
  },
  // ---------------------------------------------------------------------------
  // The seams between the app's three script files. They share a global scope
  // in the browser, so a name defined in one and used in another is correct --
  // but eslint reads one file at a time and would call every one of them
  // undefined. Declaring them here per file is what keeps `no-undef` useful:
  // it says exactly what crosses the boundary, and a typo still fails.
  //
  // Re-derive rather than guess: lint with these entries removed and read off
  // the `no-undef` names.
  {
    files: ["app/js/app.js"],
    languageOptions: {
      globals: {
        // from js/cell.js
        LABEL_CELL_M: "readonly",
        s2Cell: "readonly",
        // from js/chips.js
        CHIP_RGB: "readonly",
        CHIP_INDEX: "readonly",
        INDEX_RAMP: "readonly",
        chipIsIndex: "readonly",
        chipSpec: "readonly",
        comboBounds: "readonly",
        indexComboForSeries: "readonly",
        rampColor: "readonly",
      },
    },
  },
  {
    files: ["app/js/chips.js"],
    languageOptions: {
      globals: {
        // Both live in js/app.js because both read app state. `chipStretch` is
        // the seam the ramp test stubs; see that file's header.
        chipStretch: "readonly",
        evSeries: "readonly",
      },
    },
  },
  {
    // config.js is a classic script that assigns to `window`, loaded by
    // js/boot.js before the app.
    files: ["app/config.js"],
    languageOptions: { sourceType: "script" },
  },
];
