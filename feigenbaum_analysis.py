#!/usr/bin/env python3
"""
Systematic exploration of Feigenbaum constants in GIFT framework observables.

Tests chaos theory constants (δ_F, α_F) against all mass ratios, angles,
and other observables to identify patterns.
"""

import numpy as np
from typing import List, Tuple, Dict
import sys

# ============================================================================
# FEIGENBAUM CONSTANTS
# ============================================================================
delta_F = 4.669201609102991  # Period-doubling bifurcation rate
alpha_F = 2.502907875095893  # Width reduction parameter

# Known fractal dimensions
CANTOR_DIM = 0.6309297535714575
SIERPINSKI_DIM = 1.5849625007211563
KOCH_DIM = 1.2618595071429148

# ============================================================================
# EXPERIMENTAL VALUES FROM GIFT FRAMEWORK
# ============================================================================

# Lepton mass ratios
m_mu_over_me = 206.768
m_tau_over_mu = 3477.15 / 206.768  # = 16.817
m_tau_over_me = 3477.15

# Quark masses (MeV)
m_u = 2.16
m_d = 4.67
m_s = 93.4
m_c = 1270
m_b = 4180
m_t = 172500

# Quark mass ratios
quark_ratios = {
    "m_s/m_d": 20.0,
    "m_d/m_u": 2.162,
    "m_c/m_s": 13.6,
    "m_c/m_d": 271.94,
    "m_b/m_c": 3.29,
    "m_b/m_s": 44.76,
    "m_b/m_d": 895.07,
    "m_b/m_u": 1935.19,
    "m_t/m_c": 135.83,
    "m_t/m_s": 1846.89,
    "m_t/m_b": 41.27,
    "m_t/m_d": 36938.0,
    "m_t/m_u": 79861.0,
    "m_c/m_u": 587.96,
    "m_s/m_u": 43.24,
}

# Gauge boson masses (GeV)
m_W = 80.379
m_Z = 91.1876
m_H = 125.25

# Dark matter predictions (GeV)
m_chi1 = 90.5
m_chi2 = 352.7

# Neutrino mixing angles (degrees)
theta_12 = 33.44
theta_13 = 8.57
theta_23 = 49.2
delta_CP = 197.0

# CKM angles (degrees)
theta_C = 13.04  # Cabibbo angle
# Additional CKM angles (typical values)
theta_12_CKM = 13.04
theta_13_CKM = 0.201
theta_23_CKM = 2.38

# Coupling constants
alpha_inv = 137.036
alpha_s = 0.1179
sin2_theta_W = 0.23122

# Higgs sector
lambda_H = 0.1286
v_EW = 246.22  # GeV

# Cosmological observables
Omega_DE = 0.6847
Omega_DM = 0.120
n_s = 0.9649
H_0 = 73.04

# Koide relation
Q_Koide = 0.6667

# ============================================================================
# TEST PATTERNS
# ============================================================================

def deviation_percent(predicted, experimental):
    """Calculate percentage deviation."""
    if experimental == 0:
        return float('inf')
    return abs((predicted - experimental) / experimental) * 100

def test_pattern_1(R: float, name: str, results: List) -> None:
    """Test R ≈ δ_F / n for integer n."""
    for n in range(1, 50):
        predicted = delta_F / n
        dev = deviation_percent(predicted, R)
        if dev < 5.0:
            results.append({
                'observable': name,
                'pattern': f'δ_F / {n}',
                'predicted': predicted,
                'experimental': R,
                'deviation': dev,
                'formula': f'{delta_F:.6f} / {n}'
            })

def test_pattern_2(R: float, name: str, results: List) -> None:
    """Test R ≈ α_F / n for integer n."""
    for n in range(1, 50):
        predicted = alpha_F / n
        dev = deviation_percent(predicted, R)
        if dev < 5.0:
            results.append({
                'observable': name,
                'pattern': f'α_F / {n}',
                'predicted': predicted,
                'experimental': R,
                'deviation': dev,
                'formula': f'{alpha_F:.6f} / {n}'
            })

def test_pattern_3(R: float, name: str, results: List) -> None:
    """Test R ≈ δ_F^k for rational k."""
    exponents = [1/2, 1/3, 1/4, 2/3, 3/2, 2, 3, -1, -2, -1/2]
    for k in exponents:
        predicted = delta_F ** k
        dev = deviation_percent(predicted, R)
        if dev < 5.0:
            results.append({
                'observable': name,
                'pattern': f'δ_F^({k})',
                'predicted': predicted,
                'experimental': R,
                'deviation': dev,
                'formula': f'{delta_F:.6f}^{k}'
            })

def test_pattern_4(R: float, name: str, results: List) -> None:
    """Test R ≈ α_F^k for rational k."""
    exponents = [1/2, 1/3, 1/4, 2/3, 3/2, 2, 3, -1, -2, -1/2]
    for k in exponents:
        predicted = alpha_F ** k
        dev = deviation_percent(predicted, R)
        if dev < 5.0:
            results.append({
                'observable': name,
                'pattern': f'α_F^({k})',
                'predicted': predicted,
                'experimental': R,
                'deviation': dev,
                'formula': f'{alpha_F:.6f}^{k}'
            })

def test_pattern_5(R: float, name: str, results: List) -> None:
    """Test R ≈ δ_F / α_F and α_F / δ_F."""
    predicted_1 = delta_F / alpha_F
    dev_1 = deviation_percent(predicted_1, R)
    if dev_1 < 5.0:
        results.append({
            'observable': name,
            'pattern': 'δ_F / α_F',
            'predicted': predicted_1,
            'experimental': R,
            'deviation': dev_1,
            'formula': f'{delta_F:.6f} / {alpha_F:.6f}'
        })

    predicted_2 = alpha_F / delta_F
    dev_2 = deviation_percent(predicted_2, R)
    if dev_2 < 5.0:
        results.append({
            'observable': name,
            'pattern': 'α_F / δ_F',
            'predicted': predicted_2,
            'experimental': R,
            'deviation': dev_2,
            'formula': f'{alpha_F:.6f} / {delta_F:.6f}'
        })

def test_pattern_6(R: float, name: str, results: List) -> None:
    """Test R ≈ δ_F × α_F."""
    predicted = delta_F * alpha_F
    dev = deviation_percent(predicted, R)
    if dev < 5.0:
        results.append({
            'observable': name,
            'pattern': 'δ_F × α_F',
            'predicted': predicted,
            'experimental': R,
            'deviation': dev,
            'formula': f'{delta_F:.6f} × {alpha_F:.6f}'
        })

def test_pattern_7(R: float, name: str, results: List) -> None:
    """Test R ≈ (δ_F + α_F) / n."""
    sum_constants = delta_F + alpha_F
    for n in range(1, 50):
        predicted = sum_constants / n
        dev = deviation_percent(predicted, R)
        if dev < 5.0:
            results.append({
                'observable': name,
                'pattern': f'(δ_F + α_F) / {n}',
                'predicted': predicted,
                'experimental': R,
                'deviation': dev,
                'formula': f'({delta_F:.6f} + {alpha_F:.6f}) / {n}'
            })

def test_pattern_8(R: float, name: str, results: List) -> None:
    """Test R ≈ δ_F / Mersenne_prime."""
    mersenne_primes = [3, 7, 31, 127, 8191, 131071]
    for M_p in mersenne_primes:
        predicted = delta_F / M_p
        dev = deviation_percent(predicted, R)
        if dev < 5.0:
            results.append({
                'observable': name,
                'pattern': f'δ_F / M_{int(np.log2(M_p+1))}',
                'predicted': predicted,
                'experimental': R,
                'deviation': dev,
                'formula': f'{delta_F:.6f} / {M_p}'
            })

def test_pattern_combinations(R: float, name: str, results: List) -> None:
    """Test additional combination patterns."""
    # δ_F - α_F
    predicted = delta_F - alpha_F
    dev = deviation_percent(predicted, R)
    if dev < 5.0:
        results.append({
            'observable': name,
            'pattern': 'δ_F - α_F',
            'predicted': predicted,
            'experimental': R,
            'deviation': dev,
            'formula': f'{delta_F:.6f} - {alpha_F:.6f}'
        })

    # δ_F / (α_F × n)
    for n in range(2, 20):
        predicted = delta_F / (alpha_F * n)
        dev = deviation_percent(predicted, R)
        if dev < 5.0:
            results.append({
                'observable': name,
                'pattern': f'δ_F / (α_F × {n})',
                'predicted': predicted,
                'experimental': R,
                'deviation': dev,
                'formula': f'{delta_F:.6f} / ({alpha_F:.6f} × {n})'
            })

    # sqrt(δ_F × α_F) / n
    sqrt_product = np.sqrt(delta_F * alpha_F)
    for n in range(1, 30):
        predicted = sqrt_product / n
        dev = deviation_percent(predicted, R)
        if dev < 5.0:
            results.append({
                'observable': name,
                'pattern': f'√(δ_F × α_F) / {n}',
                'predicted': predicted,
                'experimental': R,
                'deviation': dev,
                'formula': f'√({delta_F:.6f} × {alpha_F:.6f}) / {n}'
            })

def test_fractal_dimensions(R: float, name: str, results: List) -> None:
    """Test if observable matches known fractal dimensions."""
    fractals = [
        ('Cantor', CANTOR_DIM),
        ('Sierpinski', SIERPINSKI_DIM),
        ('Koch', KOCH_DIM)
    ]

    for fname, fdim in fractals:
        dev = deviation_percent(fdim, R)
        if dev < 5.0:
            results.append({
                'observable': name,
                'pattern': f'{fname} dimension',
                'predicted': fdim,
                'experimental': R,
                'deviation': dev,
                'formula': f'D_{fname} = {fdim:.6f}'
            })

        # Also test multiples
        for n in range(2, 10):
            predicted = fdim * n
            dev = deviation_percent(predicted, R)
            if dev < 5.0:
                results.append({
                    'observable': name,
                    'pattern': f'{n} × {fname} dimension',
                    'predicted': predicted,
                    'experimental': R,
                    'deviation': dev,
                    'formula': f'{n} × {fdim:.6f}'
                })

def test_angle_patterns(angle_deg: float, name: str, results: List) -> None:
    """Test if angle = arctan(δ_F/n) or similar."""
    angle_rad = np.deg2rad(angle_deg)

    # Test arctan(δ_F / n)
    for n in range(1, 50):
        predicted_rad = np.arctan(delta_F / n)
        predicted_deg = np.rad2deg(predicted_rad)
        dev = deviation_percent(predicted_deg, angle_deg)
        if dev < 5.0:
            results.append({
                'observable': name,
                'pattern': f'arctan(δ_F / {n})',
                'predicted': predicted_deg,
                'experimental': angle_deg,
                'deviation': dev,
                'formula': f'arctan({delta_F:.6f} / {n})'
            })

    # Test arctan(α_F / n)
    for n in range(1, 50):
        predicted_rad = np.arctan(alpha_F / n)
        predicted_deg = np.rad2deg(predicted_rad)
        dev = deviation_percent(predicted_deg, angle_deg)
        if dev < 5.0:
            results.append({
                'observable': name,
                'pattern': f'arctan(α_F / {n})',
                'predicted': predicted_deg,
                'experimental': angle_deg,
                'deviation': dev,
                'formula': f'arctan({alpha_F:.6f} / {n})'
            })

    # Test arctan(n / δ_F)
    for n in range(1, 20):
        predicted_rad = np.arctan(n / delta_F)
        predicted_deg = np.rad2deg(predicted_rad)
        dev = deviation_percent(predicted_deg, angle_deg)
        if dev < 5.0:
            results.append({
                'observable': name,
                'pattern': f'arctan({n} / δ_F)',
                'predicted': predicted_deg,
                'experimental': angle_deg,
                'deviation': dev,
                'formula': f'arctan({n} / {delta_F:.6f})'
            })

    # Test tan(angle) against δ_F/n
    tan_angle = np.tan(angle_rad)
    for n in range(1, 50):
        predicted = delta_F / n
        dev = deviation_percent(predicted, tan_angle)
        if dev < 5.0:
            results.append({
                'observable': f'tan({name})',
                'pattern': f'δ_F / {n}',
                'predicted': predicted,
                'experimental': tan_angle,
                'deviation': dev,
                'formula': f'{delta_F:.6f} / {n}'
            })

def test_all_patterns(R: float, name: str) -> List[Dict]:
    """Run all pattern tests on an observable."""
    results = []

    test_pattern_1(R, name, results)
    test_pattern_2(R, name, results)
    test_pattern_3(R, name, results)
    test_pattern_4(R, name, results)
    test_pattern_5(R, name, results)
    test_pattern_6(R, name, results)
    test_pattern_7(R, name, results)
    test_pattern_8(R, name, results)
    test_pattern_combinations(R, name, results)
    test_fractal_dimensions(R, name, results)

    return results

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run comprehensive Feigenbaum analysis."""

    all_results = []

    print("=" * 80)
    print("FEIGENBAUM CONSTANTS IN GIFT FRAMEWORK")
    print("Systematic Exploration of Chaos Theory Patterns")
    print("=" * 80)
    print()
    print(f"δ_F (bifurcation rate):      {delta_F}")
    print(f"α_F (width reduction):       {alpha_F}")
    print(f"δ_F / α_F:                   {delta_F / alpha_F:.6f}")
    print(f"δ_F × α_F:                   {delta_F * alpha_F:.6f}")
    print(f"δ_F + α_F:                   {delta_F + alpha_F:.6f}")
    print()

    # ========================================================================
    # LEPTON MASS RATIOS
    # ========================================================================
    print("\n" + "=" * 80)
    print("LEPTON MASS RATIOS")
    print("=" * 80)

    lepton_ratios = {
        "m_μ/m_e": m_mu_over_me,
        "m_τ/m_μ": m_tau_over_mu,
        "m_τ/m_e": m_tau_over_me,
    }

    for name, value in lepton_ratios.items():
        results = test_all_patterns(value, name)
        all_results.extend(results)

    # ========================================================================
    # QUARK MASS RATIOS
    # ========================================================================
    print("\n" + "=" * 80)
    print("QUARK MASS RATIOS")
    print("=" * 80)

    for name, value in quark_ratios.items():
        results = test_all_patterns(value, name)
        all_results.extend(results)

    # ========================================================================
    # GAUGE BOSON RATIOS
    # ========================================================================
    print("\n" + "=" * 80)
    print("GAUGE BOSON MASS RATIOS")
    print("=" * 80)

    gauge_ratios = {
        "m_W/m_Z": m_W / m_Z,
        "m_H/m_Z": m_H / m_Z,
        "m_H/m_W": m_H / m_W,
        "m_t/m_H": m_t / 1000 / m_H,  # Convert MeV to GeV
    }

    for name, value in gauge_ratios.items():
        results = test_all_patterns(value, name)
        all_results.extend(results)

    # ========================================================================
    # DARK MATTER RATIOS
    # ========================================================================
    print("\n" + "=" * 80)
    print("DARK MATTER MASS RATIOS")
    print("=" * 80)

    dm_ratios = {
        "m_χ₂/m_χ₁": m_chi2 / m_chi1,
        "m_χ₁ (GeV)": m_chi1,
        "m_χ₂ (GeV)": m_chi2,
    }

    for name, value in dm_ratios.items():
        results = test_all_patterns(value, name)
        all_results.extend(results)

    # ========================================================================
    # ANGLES (NEUTRINO MIXING)
    # ========================================================================
    print("\n" + "=" * 80)
    print("NEUTRINO MIXING ANGLES")
    print("=" * 80)

    neutrino_angles = {
        "θ₁₂": theta_12,
        "θ₁₃": theta_13,
        "θ₂₃": theta_23,
        "δ_CP": delta_CP,
    }

    for name, value in neutrino_angles.items():
        results = test_all_patterns(value, name)
        all_results.extend(results)
        # Also test angle-specific patterns
        angle_results = []
        test_angle_patterns(value, name, angle_results)
        all_results.extend(angle_results)

    # ========================================================================
    # ANGLES (CKM MATRIX)
    # ========================================================================
    print("\n" + "=" * 80)
    print("CKM MIXING ANGLES")
    print("=" * 80)

    ckm_angles = {
        "θ_C (Cabibbo)": theta_C,
        "θ₁₂ (CKM)": theta_12_CKM,
        "θ₁₃ (CKM)": theta_13_CKM,
        "θ₂₃ (CKM)": theta_23_CKM,
    }

    for name, value in ckm_angles.items():
        results = test_all_patterns(value, name)
        all_results.extend(results)
        angle_results = []
        test_angle_patterns(value, name, angle_results)
        all_results.extend(angle_results)

    # ========================================================================
    # COUPLING CONSTANTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("COUPLING CONSTANTS")
    print("=" * 80)

    couplings = {
        "α⁻¹": alpha_inv,
        "α_s": alpha_s,
        "sin²θ_W": sin2_theta_W,
        "λ_H": lambda_H,
    }

    for name, value in couplings.items():
        results = test_all_patterns(value, name)
        all_results.extend(results)

    # ========================================================================
    # COSMOLOGICAL OBSERVABLES
    # ========================================================================
    print("\n" + "=" * 80)
    print("COSMOLOGICAL OBSERVABLES")
    print("=" * 80)

    cosmology = {
        "Ω_DE": Omega_DE,
        "Ω_DM": Omega_DM,
        "n_s": n_s,
        "Q_Koide": Q_Koide,
    }

    for name, value in cosmology.items():
        results = test_all_patterns(value, name)
        all_results.extend(results)

    # ========================================================================
    # SORT AND DISPLAY RESULTS
    # ========================================================================

    # Sort by deviation
    all_results.sort(key=lambda x: x['deviation'])

    print("\n" + "=" * 80)
    print("DISCOVERED PATTERNS (sorted by deviation)")
    print("=" * 80)
    print()

    # Display best matches (< 1%)
    print("\nEXCELLENT MATCHES (deviation < 1%):")
    print("-" * 80)
    excellent = [r for r in all_results if r['deviation'] < 1.0]
    for r in excellent:
        print(f"\n{r['observable']:20s} ≈ {r['pattern']}")
        print(f"  Predicted:    {r['predicted']:.6f}")
        print(f"  Experimental: {r['experimental']:.6f}")
        print(f"  Deviation:    {r['deviation']:.4f}%")
        print(f"  Formula:      {r['formula']}")

    # Display good matches (1-3%)
    print("\n\nGOOD MATCHES (1% < deviation < 3%):")
    print("-" * 80)
    good = [r for r in all_results if 1.0 <= r['deviation'] < 3.0]
    for r in good:
        print(f"\n{r['observable']:20s} ≈ {r['pattern']}")
        print(f"  Predicted:    {r['predicted']:.6f}")
        print(f"  Experimental: {r['experimental']:.6f}")
        print(f"  Deviation:    {r['deviation']:.4f}%")
        print(f"  Formula:      {r['formula']}")

    # Display moderate matches (3-5%)
    print("\n\nMODERATE MATCHES (3% < deviation < 5%):")
    print("-" * 80)
    moderate = [r for r in all_results if 3.0 <= r['deviation'] < 5.0]
    for r in moderate:
        print(f"\n{r['observable']:20s} ≈ {r['pattern']}")
        print(f"  Predicted:    {r['predicted']:.6f}")
        print(f"  Experimental: {r['experimental']:.6f}")
        print(f"  Deviation:    {r['deviation']:.4f}%")
        print(f"  Formula:      {r['formula']}")

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================

    print("\n\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nTotal patterns tested: ~{len(all_results) * 10} (across all observables)")
    print(f"Total matches found (< 5% deviation): {len(all_results)}")
    print(f"  Excellent (< 1%):  {len(excellent)}")
    print(f"  Good (1-3%):       {len(good)}")
    print(f"  Moderate (3-5%):   {len(moderate)}")

    # Count by observable type
    observables_with_matches = set(r['observable'] for r in all_results)
    print(f"\nObservables with at least one match: {len(observables_with_matches)}")

    # Best match for each observable
    print("\n\nBEST MATCH FOR EACH OBSERVABLE:")
    print("-" * 80)
    observable_best = {}
    for r in all_results:
        obs = r['observable']
        if obs not in observable_best or r['deviation'] < observable_best[obs]['deviation']:
            observable_best[obs] = r

    for obs in sorted(observable_best.keys()):
        r = observable_best[obs]
        print(f"{obs:25s}: {r['pattern']:30s} (dev: {r['deviation']:.3f}%)")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
