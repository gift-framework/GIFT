#!/usr/bin/env python3
"""
Analysis script for GIFT statistical validation results.
"""

import json
import pandas as pd
from pathlib import Path
import sys

def load_results():
    """Load validation results and summary"""
    results_dir = Path("statistical_validation/results")

    if not results_dir.exists():
        print("Results directory not found. Run validation first.")
        sys.exit(1)

    summary_file = results_dir / "summary.json"
    if not summary_file.exists():
        print("Summary file not found.")
        sys.exit(1)

    with open(summary_file, 'r') as f:
        summary = json.load(f)

    results_file = results_dir / "validation_results.csv"
    if not results_file.exists():
        print("Results file not found.")
        sys.exit(1)

    df = pd.read_csv(results_file)
    return summary, df

def print_summary_analysis(summary, df):
    """Print detailed statistical analysis"""
    print("=" * 80)
    print("GIFT FRAMEWORK - STATISTICAL VALIDATION ANALYSIS")
    print("=" * 80)

    ref = summary['reference_config']
    alt = summary['alternative_configs']
    sig = summary['statistical_significance']

    print("\nReference configuration (GIFT E8×E8/K7):")
    print(f"   b2 = {ref['b2']}, b3 = {ref['b3']}")
    print(".4f")

    print(f"\nAlternative configurations tested: {alt['count']:,}")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")

    print("\nStatistical significance:")
    print(".2f")
    print(".2e")
    print(".1f")

    print("\nConclusion:")
    print(f"   {summary['conclusion']}")

    print("\nDetailed breakdown:")
    ref_row = df[df['is_reference'] == True].iloc[0]
    alt_rows = df[df['is_reference'] == False]

    print("\nTop 5 alternative configurations:")
    top_alt = alt_rows.nsmallest(5, 'mean_deviation')[['config_name', 'b2', 'b3', 'mean_deviation']]
    for _, row in top_alt.iterrows():
        print(".4f")

    print("\nWorst 5 alternative configurations:")
    worst_alt = alt_rows.nlargest(5, 'mean_deviation')[['config_name', 'b2', 'b3', 'mean_deviation']]
    for _, row in worst_alt.iterrows():
        print(".4f")

def main():
    """Main analysis function"""
    print("Loading validation results...")

    try:
        summary, df = load_results()
        print(f"Loaded {len(df)} configurations")

        print_summary_analysis(summary, df)

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()