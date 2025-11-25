"""
Command line interface for generating professional-grade visualizations.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from . import render_suite


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GIFT professional visualizations.")
    parser.add_argument(
        "--figure",
        "-f",
        action="append",
        help="Figure key to render (repeat to render multiple). Defaults to all.",
    )
    parser.add_argument(
        "--output-dir",
        default="assets/visualizations/outputs/pro",
        help="Directory for exported figures.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to config.json override.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive windows after rendering.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    render_suite(
        figures=args.figure,
        output_dir=Path(args.output_dir),
        config_path=args.config,
        show=args.show,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


