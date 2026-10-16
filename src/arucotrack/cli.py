"""`arucotrack run <config.yaml> [key=value ...]`: video -> trajectory files. `run` is the only command."""
from __future__ import annotations

import sys

USAGE = """usage: arucotrack run <config.yaml> [key=value ...] [--data-root DIR] [--out DIR] [--work DIR] [--force] [--debug]
       arucotrack --version | --help

commands:
  run    video -> detection -> refined corners -> pose -> floor -> trajectory files (YAML configuration)

`arucotrack run --help` describes the options. Configuration fields: docs/CONFIG.md."""


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("--version", "-V"):
        import arucotrack
        print(f"arucotrack {arucotrack.__version__}")
        return 0
    if argv and argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if not argv or argv[0] != "run":
        if argv:
            print(f"arucotrack: unknown command {argv[0]!r}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    from . import run
    return int(run.main(argv[1:]) or 0)


if __name__ == "__main__":
    sys.exit(main())
