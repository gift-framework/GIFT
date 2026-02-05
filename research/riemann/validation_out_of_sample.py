#!/usr/bin/env python3
"""
Out-of-Sample Validation: Does 31/21 Predict Better?
=====================================================

The ultimate test: train on first 30k zeros, predict on next 20k.
Compare exact 31/21 vs free fit vs other candidates.
"""

import numpy as np
from pathlib import Path
import json

# =============================================================================
# LOAD ZEROS
# =============================================================================

def load_zeros(max_zeros=100000):
    """Load Riemann zeros from data files."""
    zeros = []
    zeros_dir = Path(__file__).parent
    for i in range(1, 11):
        zeros_file = zeros_dir / f"zeros{i}"
        if zeros_file.exists():
            with open(zeros_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        zeros.append(float(line))
                        if len(zeros) >= max_zeros:
                            return np.array(zeros)
    return np.array(zeros)

# =============================================================================
# FIT AND PREDICT
# =============================================================================

def fit_on_train(zeros_train, lag1, lag2):
    """Fit recurrence on training data, return (a, b, c)."""
    max_lag = max(lag1, lag2)
    n = len(zeros_train) - max_lag

    X1 = zeros_train[max_lag - lag1:max_lag - lag1 + n]
    X2 = zeros_train[max_lag - lag2:max_lag - lag2 + n]
    X = np.column_stack([X1, X2, np.ones(n)])
    y = zeros_train[max_lag:max_lag + n]

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs[0], coeffs[1], coeffs[2]

def predict_with_coefficients(zeros_full, a, b, c, lag1, lag2, start_idx, end_idx):
    """
    Predict zeros from start_idx to end_idx using fixed coefficients.
    Return (predictions, actual, errors).
    """
    predictions = []
    actuals = []

    for n in range(start_idx, end_idx):
        pred = a * zeros_full[n - lag1] + b * zeros_full[n - lag2] + c
        predictions.append(pred)
        actuals.append(zeros_full[n])

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    errors = actuals - predictions

    return predictions, actuals, errors

def compute_metrics(errors, actuals):
    """Compute various error metrics."""
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / actuals)) * 100
    max_error = np.max(np.abs(errors))

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'max_error': max_error
    }

# =============================================================================
# MAIN VALIDATION
# =============================================================================

def main():
    print("=" * 70)
    print("OUT-OF-SAMPLE VALIDATION: DOES 31/21 PREDICT BETTER?")
    print("=" * 70)

    # Load zeros
    zeros = load_zeros(50000)
    print(f"\n✓ Loaded {len(zeros)} Riemann zeros")

    # Split: train on first 30k, test on next 20k
    train_size = 30000
    test_start = train_size
    test_end = len(zeros)

    zeros_train = zeros[:train_size]
    lag1, lag2 = 8, 21

    print(f"\nTrain: zeros 1-{train_size}")
    print(f"Test:  zeros {test_start+1}-{test_end}")

    # ==========================================================================
    # Fit on training data
    # ==========================================================================

    a_fit, b_fit, c_fit = fit_on_train(zeros_train, lag1, lag2)
    print(f"\nFitted on training data:")
    print(f"  a = {a_fit:.6f}")
    print(f"  b = {b_fit:.6f}")
    print(f"  c = {c_fit:.6f}")

    # ==========================================================================
    # Compare candidates
    # ==========================================================================

    # For each candidate, we need to estimate c from training data
    candidates = {
        '31/21 (k=6 exact)': (31/21, -10/21),
        '3/2 (simple)': (3/2, -1/2),
        '19/13 (k=5)': (19/13, -6/13),
        '50/34 (k=7)': (50/34, -16/34),
        'φ-φ⁻⁴ (limit)': (1.4721, -0.4721),
        'FREE FIT': (a_fit, b_fit),
    }

    print("\n" + "-" * 70)
    print("OUT-OF-SAMPLE PREDICTION ERRORS")
    print("-" * 70)

    print(f"\n{'Candidate':<22} {'MAE':<12} {'RMSE':<12} {'MAPE %':<12} {'Max Err':<12}")
    print("-" * 70)

    results = {}

    for name, (a, b) in candidates.items():
        # Estimate c from training data with fixed a, b
        max_lag = max(lag1, lag2)
        n_train = train_size - max_lag
        X1 = zeros_train[max_lag - lag1:max_lag - lag1 + n_train]
        X2 = zeros_train[max_lag - lag2:max_lag - lag2 + n_train]
        y_train = zeros_train[max_lag:max_lag + n_train]
        pred_train = a * X1 + b * X2
        c = np.mean(y_train - pred_train)

        # Predict on test set
        _, actuals, errors = predict_with_coefficients(
            zeros, a, b, c, lag1, lag2, test_start, test_end
        )
        metrics = compute_metrics(errors, actuals)

        print(f"{name:<22} {metrics['mae']:<12.6f} {metrics['rmse']:<12.6f} {metrics['mape']:<12.6f} {metrics['max_error']:<12.4f}")

        results[name] = {
            'a': float(a),
            'b': float(b),
            'c': float(c),
            **{k: float(v) for k, v in metrics.items()}
        }

    # ==========================================================================
    # Statistical comparison
    # ==========================================================================

    print("\n" + "-" * 70)
    print("STATISTICAL COMPARISON")
    print("-" * 70)

    # Find best by MAE
    best_mae = min(results.items(), key=lambda x: x[1]['mae'])
    best_rmse = min(results.items(), key=lambda x: x[1]['rmse'])

    print(f"\nBest by MAE:  {best_mae[0]} ({best_mae[1]['mae']:.6f})")
    print(f"Best by RMSE: {best_rmse[0]} ({best_rmse[1]['rmse']:.6f})")

    # Compare 31/21 vs free fit
    mae_31_21 = results['31/21 (k=6 exact)']['mae']
    mae_free = results['FREE FIT']['mae']
    diff_percent = (mae_31_21 - mae_free) / mae_free * 100

    print(f"\n31/21 vs FREE FIT:")
    print(f"  31/21 MAE: {mae_31_21:.6f}")
    print(f"  FREE MAE:  {mae_free:.6f}")
    print(f"  Difference: {diff_percent:+.2f}%")

    if abs(diff_percent) < 1:
        print(f"  → Practically equivalent (< 1% difference)")
    elif diff_percent < 0:
        print(f"  → 31/21 is BETTER out-of-sample!")
    else:
        print(f"  → FREE FIT is better, but difference is small")

    # ==========================================================================
    # Test if 31/21 is within statistical noise of optimal
    # ==========================================================================

    print("\n" + "-" * 70)
    print("IS 31/21 STATISTICALLY DISTINGUISHABLE FROM OPTIMAL?")
    print("-" * 70)

    # Bootstrap test: sample prediction errors and compare
    _, actuals_31, errors_31 = predict_with_coefficients(
        zeros, 31/21, -10/21,
        results['31/21 (k=6 exact)']['c'],
        lag1, lag2, test_start, test_end
    )

    _, actuals_free, errors_free = predict_with_coefficients(
        zeros, a_fit, b_fit,
        results['FREE FIT']['c'],
        lag1, lag2, test_start, test_end
    )

    # Paired comparison
    diff_errors = np.abs(errors_31) - np.abs(errors_free)
    mean_diff = np.mean(diff_errors)
    std_diff = np.std(diff_errors) / np.sqrt(len(diff_errors))
    t_stat = mean_diff / std_diff
    p_value = 2 * (1 - __import__('scipy').stats.norm.cdf(abs(t_stat)))

    print(f"\nPaired t-test (|error_31/21| - |error_free|):")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Standard error:  {std_diff:.6f}")
    print(f"  t-statistic:     {t_stat:.2f}")
    print(f"  p-value:         {p_value:.4f}")

    if p_value > 0.05:
        print(f"\n  → NO significant difference (p > 0.05)")
        print(f"  → 31/21 is STATISTICALLY EQUIVALENT to optimal fit")
    else:
        if mean_diff < 0:
            print(f"\n  → 31/21 is SIGNIFICANTLY BETTER (p < 0.05)")
        else:
            print(f"\n  → FREE FIT is significantly better (p < 0.05)")

    # ==========================================================================
    # Window analysis: does the advantage persist?
    # ==========================================================================

    print("\n" + "-" * 70)
    print("WINDOW ANALYSIS: DOES 31/21 ADVANTAGE PERSIST?")
    print("-" * 70)

    windows = [
        (30000, 35000, "30k-35k"),
        (35000, 40000, "35k-40k"),
        (40000, 45000, "40k-45k"),
        (45000, 50000, "45k-50k"),
    ]

    print(f"\n{'Window':<12} {'MAE 31/21':<14} {'MAE FREE':<14} {'Winner':<12}")
    print("-" * 55)

    for start, end, name in windows:
        _, _, err_31 = predict_with_coefficients(
            zeros, 31/21, -10/21,
            results['31/21 (k=6 exact)']['c'],
            lag1, lag2, start, end
        )
        _, _, err_free = predict_with_coefficients(
            zeros, a_fit, b_fit,
            results['FREE FIT']['c'],
            lag1, lag2, start, end
        )

        mae_31 = np.mean(np.abs(err_31))
        mae_fr = np.mean(np.abs(err_free))
        winner = "31/21" if mae_31 < mae_fr else "FREE"

        print(f"{name:<12} {mae_31:<14.6f} {mae_fr:<14.6f} {winner:<12}")

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
OUT-OF-SAMPLE VALIDATION RESULTS:

1. PREDICTION ACCURACY:
   - 31/21 MAE: {mae_31_21:.6f}
   - FREE MAE:  {mae_free:.6f}
   - Difference: {diff_percent:+.2f}%

2. STATISTICAL TEST:
   - p-value: {p_value:.4f}
   - Verdict: {'NO significant difference' if p_value > 0.05 else 'Significant difference'}

3. INTERPRETATION:
""")

    if p_value > 0.05 and abs(diff_percent) < 2:
        print("   ✅ The exact Fibonacci formula 31/21 is STATISTICALLY EQUIVALENT")
        print("      to the empirically optimized coefficients.")
        print()
        print("   This means: the data DOES NOT REJECT the theoretical prediction.")
        print("   The Fibonacci structure is consistent with optimal performance.")
    else:
        print("   ⚠️ There is a measurable difference between 31/21 and optimal.")

    # Save results
    output = {
        'train_size': train_size,
        'test_size': test_end - test_start,
        'candidates': results,
        'statistical_test': {
            'mean_diff': float(mean_diff),
            't_stat': float(t_stat),
            'p_value': float(p_value),
            'equivalent': bool(p_value > 0.05)
        }
    }

    with open(Path(__file__).parent / "validation_oos_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n✓ Results saved to validation_oos_results.json")

if __name__ == "__main__":
    main()
