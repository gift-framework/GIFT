#!/usr/bin/env python3
"""
Quick test of the statistical validation system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from advanced_validation import StatisticalValidator

print("Running quick statistical validation test...")
print("Testing 50 alternative configurations...")

validator = StatisticalValidator()
results_df = validator.run_validation(n_configs=50)

print("\nResults summary:")
print(f"Total configurations: {len(results_df)}")
print(f"Reference config: {results_df[results_df['is_reference'] == True]['mean_deviation'].values[0]:.4f}%")
print(f"Alternative mean: {results_df[results_df['is_reference'] == False]['mean_deviation'].mean():.4f}%")

print("\n✅ Quick test completed successfully!")
