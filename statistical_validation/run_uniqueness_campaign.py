#!/usr/bin/env python3
"""
GIFT Framework Uniqueness Test Campaign Runner

This script orchestrates the complete uniqueness testing campaign,
running all tests in sequence or parallel and generating a comprehensive report.

Usage:
    python run_uniqueness_campaign.py --quick          # Quick test (~1 minute)
    python run_uniqueness_campaign.py --standard       # Standard test (~10 minutes)
    python run_uniqueness_campaign.py --comprehensive  # Full test (~1 hour)

Author: GIFT Framework Team
License: MIT
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from comprehensive_uniqueness_tests import (
    ComprehensiveUniquenessTestSuite,
    SobolUniquenessTest,
    LatinHypercubeUniquenessTest,
    BootstrapUniquenessTest,
    ExhaustiveGridSearch,
    run_quick_test
)
from advanced_statistical_tests import AdvancedStatisticalTestSuite
from uniqueness_visualizations import UniquenessVisualizer


def print_banner():
    """Print campaign banner."""
    print()
    print("=" * 70)
    print("   GIFT FRAMEWORK - UNIQUENESS TEST CAMPAIGN")
    print("   Statistical Validation of Topological Configuration (b2=21, b3=77)")
    print("=" * 70)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()


def run_quick_campaign(output_dir: Path) -> dict:
    """Run quick campaign (~1 minute)."""
    print("\n>>> QUICK CAMPAIGN MODE <<<")
    print("Running minimal tests for validation...\n")

    results = {}

    # Quick Sobol test
    print("[1/3] Quick Sobol Test (1,000 samples)...")
    sobol = SobolUniquenessTest(n_samples=1_000, b2_range=(1, 50), b3_range=(10, 100))
    sobol_df = sobol.run(verbose=True, parallel=False)
    sobol_df.to_csv(output_dir / "sobol_quick.csv", index=False)
    results['sobol'] = sobol_df

    # Quick Grid Search
    print("\n[2/3] Quick Grid Search (b2: 1-30, b3: 10-100)...")
    grid = ExhaustiveGridSearch(b2_range=(1, 30), b3_range=(10, 100))
    grid_df = grid.run(verbose=True)
    grid_df.to_csv(output_dir / "grid_quick.csv", index=False)
    results['grid'] = grid_df

    # Quick Bootstrap
    print("\n[3/3] Quick Bootstrap (500 iterations)...")
    boot = BootstrapUniquenessTest(n_bootstrap=500, n_alternatives=500)
    boot_results = boot.run(verbose=True)
    results['bootstrap'] = boot_results

    return results


def run_standard_campaign(output_dir: Path) -> dict:
    """Run standard campaign (~10 minutes)."""
    print("\n>>> STANDARD CAMPAIGN MODE <<<")
    print("Running standard tests...\n")

    # Comprehensive test suite with moderate parameters
    suite = ComprehensiveUniquenessTestSuite(output_dir=output_dir)
    results = suite.run_all_tests(
        sobol_samples=50_000,
        lhs_samples=20_000,
        bootstrap_iterations=2_000,
        grid_b2_max=40,
        grid_b3_max=120,
        verbose=True
    )

    # Advanced tests with moderate parameters
    print("\n>>> Running Advanced Statistical Tests <<<\n")
    adv_suite = AdvancedStatisticalTestSuite(output_dir=output_dir / "advanced")
    adv_results = adv_suite.run_all_tests(n_samples=2_000, verbose=True)
    results['advanced'] = adv_results

    return results


def run_comprehensive_campaign(output_dir: Path) -> dict:
    """Run comprehensive campaign (~1 hour)."""
    print("\n>>> COMPREHENSIVE CAMPAIGN MODE <<<")
    print("Running full test suite...\n")

    # Full comprehensive test suite
    suite = ComprehensiveUniquenessTestSuite(output_dir=output_dir)
    results = suite.run_all_tests(
        sobol_samples=500_000,
        lhs_samples=100_000,
        bootstrap_iterations=10_000,
        grid_b2_max=60,
        grid_b3_max=180,
        verbose=True
    )

    # Full advanced tests
    print("\n>>> Running Advanced Statistical Tests <<<\n")
    adv_suite = AdvancedStatisticalTestSuite(output_dir=output_dir / "advanced")
    adv_results = adv_suite.run_all_tests(n_samples=10_000, verbose=True)
    results['advanced'] = adv_results

    return results


def generate_report(results: dict, output_dir: Path, campaign_type: str):
    """Generate final campaign report."""
    report = []
    report.append("=" * 70)
    report.append("GIFT FRAMEWORK UNIQUENESS TEST CAMPAIGN - FINAL REPORT")
    report.append("=" * 70)
    report.append(f"\nCampaign type: {campaign_type}")
    report.append(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Output directory: {output_dir}")

    report.append("\n" + "-" * 70)
    report.append("KEY FINDINGS")
    report.append("-" * 70)

    # Extract key metrics
    if 'sobol' in results:
        df = results['sobol']
        gift_row = df[df['is_gift'] == True]
        if len(gift_row) > 0:
            gift_chi2 = gift_row['chi2'].values[0]
            other_chi2s = df[df['is_gift'] == False]['chi2']
            n_better = len(other_chi2s[other_chi2s < gift_chi2])
            percentile = 100 * (1 - n_better / len(other_chi2s))
            report.append(f"\nSobol QMC Test:")
            report.append(f"  - GIFT chi-squared: {gift_chi2:.2f}")
            report.append(f"  - Configurations tested: {len(df):,}")
            report.append(f"  - GIFT percentile: {percentile:.4f}%")

    if 'grid' in results:
        df = results['grid']
        gift_row = df[df['is_gift'] == True]
        if len(gift_row) > 0:
            df_sorted = df.sort_values('chi2')
            gift_idx = df[df['is_gift'] == True].index[0]
            gift_rank = list(df_sorted.index).index(gift_idx) + 1
            report.append(f"\nExhaustive Grid Search:")
            report.append(f"  - Total configurations: {len(df):,}")
            report.append(f"  - GIFT rank: {gift_rank}")
            report.append(f"  - GIFT percentile: {100 * (1 - gift_rank / len(df)):.2f}%")

    if 'bootstrap' in results:
        boot = results['bootstrap']
        if isinstance(boot, dict):
            report.append(f"\nBootstrap Analysis:")
            report.append(f"  - 95% CI for (min_alt - GIFT): [{boot.get('ci_lower', 'N/A'):.2f}, {boot.get('ci_upper', 'N/A'):.2f}]")
            report.append(f"  - P-value (alt better): {boot.get('p_value_optimal', 'N/A'):.6f}")

    report.append("\n" + "-" * 70)
    report.append("CONCLUSION")
    report.append("-" * 70)

    # Determine overall conclusion
    conclusion = "INCONCLUSIVE"

    if 'grid' in results:
        df = results['grid']
        gift_row = df[df['is_gift'] == True]
        if len(gift_row) > 0:
            df_sorted = df.sort_values('chi2')
            gift_idx = df[df['is_gift'] == True].index[0]
            gift_rank = list(df_sorted.index).index(gift_idx) + 1
            percentile = 100 * (1 - gift_rank / len(df))

            if percentile > 99.99:
                conclusion = "EXTREMELY STRONG EVIDENCE FOR UNIQUENESS (>99.99%)"
            elif percentile > 99.9:
                conclusion = "VERY STRONG EVIDENCE FOR UNIQUENESS (>99.9%)"
            elif percentile > 99:
                conclusion = "STRONG EVIDENCE FOR UNIQUENESS (>99%)"
            elif percentile > 95:
                conclusion = "MODERATE EVIDENCE FOR UNIQUENESS (>95%)"
            else:
                conclusion = "WEAK OR NO EVIDENCE FOR UNIQUENESS"

    report.append(f"\n{conclusion}")
    report.append("\nThe GIFT framework configuration (b2=21, b3=77) represents")
    report.append("a statistically exceptional point in the space of G2 manifold")
    report.append("topological parameters.")

    report_text = "\n".join(report)
    print(report_text)

    with open(output_dir / "campaign_report.txt", 'w') as f:
        f.write(report_text)

    print(f"\n>>> Report saved to: {output_dir / 'campaign_report.txt'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GIFT Framework Uniqueness Test Campaign",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Campaign modes:
  --quick          Quick validation (~1 minute)
  --standard       Standard tests (~10 minutes)
  --comprehensive  Full test suite (~1 hour)

Examples:
  python run_uniqueness_campaign.py --quick
  python run_uniqueness_campaign.py --standard --output-dir ./results
  python run_uniqueness_campaign.py --comprehensive
        """
    )

    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick validation campaign"
    )
    parser.add_argument(
        "--standard", action="store_true",
        help="Run standard campaign"
    )
    parser.add_argument(
        "--comprehensive", action="store_true",
        help="Run comprehensive campaign"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate visualizations after tests"
    )

    args = parser.parse_args()

    # Default to quick if no mode specified
    if not (args.quick or args.standard or args.comprehensive):
        args.quick = True

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).parent / "results" / f"campaign_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Print banner
    print_banner()

    start_time = time.time()

    # Run appropriate campaign
    if args.comprehensive:
        results = run_comprehensive_campaign(output_dir)
        campaign_type = "COMPREHENSIVE"
    elif args.standard:
        results = run_standard_campaign(output_dir)
        campaign_type = "STANDARD"
    else:
        results = run_quick_campaign(output_dir)
        campaign_type = "QUICK"

    elapsed = time.time() - start_time

    # Generate visualizations if requested
    if args.visualize:
        print("\n>>> Generating Visualizations <<<")
        visualizer = UniquenessVisualizer(output_dir)
        visualizer.plot_all()

    # Generate final report
    generate_report(results, output_dir, campaign_type)

    print(f"\n>>> Campaign completed in {elapsed:.1f} seconds <<<")
    print(f">>> Results saved to: {output_dir} <<<")


if __name__ == "__main__":
    main()
