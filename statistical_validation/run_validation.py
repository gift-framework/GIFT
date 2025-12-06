#!/usr/bin/env python3
"""
Script to run the advanced statistical validation for GIFT framework.

This script executes the comprehensive validation that tests thousands of
alternative G2 manifold configurations to demonstrate that only the specific
E8×E8/K7 configuration yields the observed 0.128% mean deviation from experiment.
"""

import argparse
import sys
from pathlib import Path

# Add the statistical_validation directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from advanced_validation import StatisticalValidator

def main():
    parser = argparse.ArgumentParser(
        description="Run advanced statistical validation for GIFT framework"
    )
    parser.add_argument(
        "--n-configs",
        type=int,
        default=10000,
        help="Number of alternative configurations to test (default: 10000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results (default: 42)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ADVANCED STATISTICAL VALIDATION - GIFT FRAMEWORK")
    print("=" * 70)
    print()
    print("This validation addresses the overfitting/post-hoc pattern concern by testing")
    print(f"{args.n_configs:,} alternative G2 manifold configurations against experimental data.")
    print()
    print("If GIFT predictions result from overfitting, many alternatives should")
    print("give similar agreement. If genuine, only E8×E8/K7 should yield 0.128% mean deviation.")
    print()

    # Set random seed for reproducibility
    import numpy as np
    np.random.seed(args.seed)

    # Run validation
    validator = StatisticalValidator()

    try:
        results_df = validator.run_validation(n_configs=args.n_configs)

        print()
        print("✓ Validation completed successfully!")
        print(f"✓ Results saved to: statistical_validation/{args.output_dir}/")
        print("✓ Run 'python -m statistical_validation.analyze_results' to see detailed analysis")
    except Exception as e:
        print(f"✗ Error during validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
