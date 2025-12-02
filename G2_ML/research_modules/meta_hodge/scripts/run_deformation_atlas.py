#!/usr/bin/env python
"""Run the K7_GIFT deformation atlas exploration.

This script explores a grid of deformation parameters (sigma, s, alpha) around
the baseline K7_GIFT structure and maps out the stability region.

Usage:
    python run_deformation_atlas.py                    # Default 5x5x5 grid
    python run_deformation_atlas.py --quick            # Quick 3x3x3 grid
    python run_deformation_atlas.py --fine             # Fine 7x7x7 grid
    python run_deformation_atlas.py --output ./my_out  # Custom output directory

See K7_DEFORMATION_ATLAS.md for theoretical background.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent directories to path for imports
script_dir = Path(__file__).resolve().parent
meta_hodge_dir = script_dir.parent
g2_ml_dir = meta_hodge_dir.parent
if str(g2_ml_dir) not in sys.path:
    sys.path.insert(0, str(g2_ml_dir))

from meta_hodge.deformation_explorer import (
    DeformationConfig,
    load_baseline_data,
    explore_grid,
    save_results,
    summarize_results,
)


def make_config(preset: str = "default") -> DeformationConfig:
    """Create a DeformationConfig based on preset."""
    if preset == "quick":
        # 3x3x3 = 27 points for quick testing
        return DeformationConfig(
            sigma_values=(0.7, 1.0, 1.3),
            s_values=(0.7, 1.0, 1.3),
            alpha_values=(-0.2, 0.0, 0.2),
        )
    elif preset == "fine":
        # 7x7x7 = 343 points for detailed mapping
        return DeformationConfig(
            sigma_values=(0.5, 0.67, 0.83, 1.0, 1.17, 1.33, 1.5),
            s_values=(0.5, 0.67, 0.83, 1.0, 1.17, 1.33, 1.5),
            alpha_values=(-0.4, -0.27, -0.13, 0.0, 0.13, 0.27, 0.4),
        )
    elif preset == "tiny":
        # 2x2x2 = 8 points for debugging
        return DeformationConfig(
            sigma_values=(0.8, 1.2),
            s_values=(0.8, 1.2),
            alpha_values=(-0.1, 0.1),
        )
    else:
        # Default 5x5x5 = 125 points
        return DeformationConfig()


def progress_callback(i: int, total: int, result) -> None:
    """Progress callback for exploration."""
    status = "stable" if result.stable else ("posdef" if result.g_posdef else "FAIL")
    bar_len = 30
    filled = int(bar_len * i / total)
    bar = "=" * filled + "-" * (bar_len - filled)
    print(f"\r[{bar}] {i}/{total} ({result.sigma:.2f},{result.s:.2f},{result.alpha:.2f}) -> {status}    ", end="")
    if i == total:
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Run K7_GIFT deformation atlas exploration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use quick 3x3x3 grid (27 points)",
    )
    parser.add_argument(
        "--fine", action="store_true",
        help="Use fine 7x7x7 grid (343 points)",
    )
    parser.add_argument(
        "--tiny", action="store_true",
        help="Use tiny 2x2x2 grid (8 points) for debugging",
    )
    parser.add_argument(
        "--version", default="1_6",
        help="Baseline model version (default: 1_6)",
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Number of samples to use (default: all available)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output directory (default: artifacts/deformation_atlas/)",
    )
    parser.add_argument(
        "--yukawa", action="store_true",
        help="Enable Yukawa computation (slow)",
    )
    parser.add_argument(
        "--exact-metric", action="store_true",
        help="Use exact metric computation (slow)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Determine preset
    if args.tiny:
        preset = "tiny"
    elif args.quick:
        preset = "quick"
    elif args.fine:
        preset = "fine"
    else:
        preset = "default"

    config = make_config(preset)
    config.compute_yukawa = args.yukawa
    config.use_exact_metric = args.exact_metric

    # Output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = meta_hodge_dir / "artifacts" / "deformation_atlas" / timestamp

    # Banner
    n_points = len(config.sigma_values) * len(config.s_values) * len(config.alpha_values)
    if not args.quiet:
        print("=" * 60)
        print("K7_GIFT Deformation Atlas Exploration")
        print("=" * 60)
        print(f"Preset: {preset}")
        print(f"Grid size: {len(config.sigma_values)}x{len(config.s_values)}x{len(config.alpha_values)} = {n_points} points")
        print(f"Baseline version: {args.version}")
        print(f"Output: {output_dir}")
        print(f"Yukawa: {'enabled' if config.compute_yukawa else 'disabled'}")
        print(f"Exact metric: {'enabled' if config.use_exact_metric else 'disabled'}")
        print("-" * 60)

    # Load baseline
    if not args.quiet:
        print("Loading baseline data...")
    try:
        baseline = load_baseline_data(version=args.version, num_samples=args.samples)
        if not args.quiet:
            print(f"  Loaded {baseline.coords.shape[0]} samples from {baseline.notes}")
            print(f"  phi_local shape: {baseline.phi_local.shape}")
            print(f"  phi_global shape: {baseline.phi_global.shape}")
    except Exception as e:
        print(f"ERROR loading baseline: {e}")
        print("Try running with --version 1_5 or ensure v1.6 data is available")
        return 1

    # Run exploration
    if not args.quiet:
        print(f"\nExploring {n_points} grid points...")

    cb = None if args.quiet else progress_callback
    results = explore_grid(baseline, config, progress_callback=cb)

    # Summary
    if not args.quiet:
        print("\n" + summarize_results(results))

    # Save results
    if not args.quiet:
        print(f"\nSaving results to {output_dir}...")
    paths = save_results(results, output_dir)

    # Also save summary
    summary_path = output_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write(summarize_results(results))
        f.write("\n\nConfig:\n")
        f.write(f"  sigma_values: {config.sigma_values}\n")
        f.write(f"  s_values: {config.s_values}\n")
        f.write(f"  alpha_values: {config.alpha_values}\n")
        f.write(f"  baseline_version: {args.version}\n")
        f.write(f"  num_samples: {baseline.coords.shape[0]}\n")

    if not args.quiet:
        print(f"  CSV: {paths.get('csv')}")
        print(f"  JSON: {paths.get('json')}")
        print(f"  Summary: {summary_path}")
        print("\nDone!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
