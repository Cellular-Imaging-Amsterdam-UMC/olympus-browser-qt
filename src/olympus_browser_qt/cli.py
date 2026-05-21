from __future__ import annotations

import argparse

from .olympus_browser_dialog import run_dialog_as_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open the Olympus Qt browser and print selections as JSON.")
    parser.add_argument("paths", nargs="*", help="Files or folders to browse.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--single", action="store_true", help="Select one image. This is the default.")
    group.add_argument("--multi", "--multiple", dest="multiple", action="store_true", help="Select images.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_dialog_as_json(args.paths, multiple=bool(args.multiple))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

