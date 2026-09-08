#!/usr/bin/env python3
"""Static checks on `app/label_app.html`, run by pre-commit AND by CI.

One script with two callers on purpose. These checks used to live as heredocs
inside .github/workflows/tests.yml, which meant they ran only after a push and
could only be fixed by editing YAML. They now run on the commit that breaks
them, and the workflow calls the same file, so the two cannot drift.

Two things are checked:

  1. EVERY INLINE <script> PARSES. `node --check` per block. A syntax error in
     the app is otherwise a page that loads to a blank screen, and the Playwright
     suite reports it as a timeout rather than as the parse failure it is.

  2. NO INLINE EVENT HANDLERS. `onclick="..."` needs every function it names to
     be a global, is invisible to eslint, and is what makes a `script-src`
     Content-Security-Policy impossible. They were removed in favour of one
     delegated dispatcher; this keeps them out.

Both checks tolerate an app with no inline script at all -- that is the
direction this file is moving in, not a reason to fail.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "label_app.html"

#: An inline block is a <script> with no `src=`. `[^>]*` deliberately excludes
#: `>` so an attribute cannot swallow the tag.
INLINE_SCRIPT = re.compile(r"<script(?![^>]*\ssrc=)[^>]*>(.*?)</script>", re.S)

#: HTML comments are stripped before any of that. The page's own comments talk
#: ABOUT `<script>` -- the CSP note explains which inline blocks it used to have
#: -- and matching those made this script report a syntax error in English
#: prose. Comments are stripped rather than skipped over so that line-based
#: checks below stay aligned: each one is replaced by as many blank lines as it
#: spanned.
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


def strip_comments(text: str) -> str:
    return HTML_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)

#: The handlers that were removed. Kept as a list rather than `on\w+` so a
#: legitimate attribute like `only="..."` cannot trip it.
HANDLER = re.compile(
    r"""\son(?:click|change|input|load|error|submit|focus|blur|keydown|keyup"""
    r"""|mouse[a-z]+|pointer[a-z]+|touch[a-z]+)\s*=\s*["']""",
    re.I,
)

IN_CI = bool(os.environ.get("GITHUB_ACTIONS"))


def _err(msg: str) -> None:
    print(f"::error::{msg}" if IN_CI else f"error: {msg}", file=sys.stderr)


def check_inline_scripts(text: str) -> int:
    node = shutil.which("node") or shutil.which("nodejs")
    blocks = INLINE_SCRIPT.findall(text)
    if not blocks:
        print("inline scripts: none to check")
        return 0
    if not node:
        # In CI node is installed by the workflow, so its absence is a broken
        # job rather than a bare checkout, and must not pass quietly.
        if IN_CI:
            _err("node is not installed, so the inline scripts were not checked")
            return 1
        print("inline scripts: skipped, node is not installed")
        return 0

    bad = 0
    for i, block in enumerate(blocks):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".js", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(block)
            tmp = fh.name
        try:
            done = subprocess.run(
                [node, "--check", tmp], capture_output=True, text=True, check=False
            )
            if done.returncode:
                _err(f"inline script block {i} of {APP.name} does not parse")
                print(done.stderr, file=sys.stderr)
                bad = 1
        finally:
            os.unlink(tmp)
    if not bad:
        print(f"inline scripts: {len(blocks)} block(s) parse")
    return bad


def check_inline_handlers(text: str) -> int:
    hits = [
        (i, line)
        for i, line in enumerate(text.splitlines(), 1)
        if HANDLER.search(line)
    ]
    if not hits:
        print("inline handlers: none")
        return 0
    for i, line in hits:
        _err(f"{APP.name}:{i}: inline event handler -- add a data-act entry to "
             f"LOADING_ACTIONS, or wire it in JS: {line.strip()[:80]}")
    return 1


def main() -> int:
    if not APP.exists():
        _err(f"{APP} does not exist")
        return 1
    text = strip_comments(APP.read_text(encoding="utf-8"))
    return check_inline_scripts(text) | check_inline_handlers(text)


if __name__ == "__main__":
    raise SystemExit(main())
