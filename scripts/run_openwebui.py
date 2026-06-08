#!/usr/bin/env python
"""Repo-local Open WebUI launcher that avoids broken Windows console shims."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from open_webui import app
    except ImportError:
        print(
            "Error: open-webui is not installed in this environment. "
            "Install it into .venv first.",
            file=sys.stderr,
        )
        return 1

    app(prog_name="open-webui")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())