#!/usr/bin/env python3
"""
Refined analysis of odd zeta discoveries with focus on:
1. Meaningful patterns (not trivial approximations to 1)
2. Priority cosmological and gauge observables
3. Statistical significance calculations
4. Comparison with existing formulas
"""

import math
import csv
from typing import Dict, List, Tuple
from pathlib import Path

# High-precision zeta values
ZETA = {
    3: 1.2020569031595942,
    5: 1.0369277551433699,
    7: 1.0083492773819228,
    9: 1.0020083928260822,
    11: 1.0004941886041194,
    13: 1.0001224140313004,
    15: 1.0000122713347577,
    17: 1.0000061275061856,
    19: 1.0000030588236307,
    21: 1.0000015282259408,
}

# Framework parameters
P = {
    'ln2': math.log(2),
    'pi': math.pi,
    'tau': 10416/2673,
    'xi': 5*math.pi/16,
    'delta': 2*math.pi/25,
    'gamma': 0.5772156649,
    'gamma_GIFT': 511/884,
    'phi': (1+math.sqrt(5))/2,
    'delta_F': 4.669201609,
    'Weyl': 5,
    'rank': 8,
    'b2': 21,
    'b3': 77,
    'H_star': 99,
    'M2': 3,
    'M3': 7,
    'M5': 31,
    'dim_G2': 14,
    'dim_K7': 7,
}

def analyze_down_quark():
    """Detailed analysis of m_d patterns."""
    print("\n" + "="*80)
    print("DOWN QUARK MASS (m_d) - DETAILED ANALYSIS")
    print("="*80)

    m_d_exp = 4.67
    m_d_unc = 0.48

    # Current formula
    m_d_ln107 = math.log(107)
    dev_ln107 = abs(m_d_ln107 - m_d_exp) / m_d_exp * 100
    sigma_ln107 = abs(m_d_ln107 - m_d_exp) / m_d_unc

    # New zeta formula
    m_d_zeta = ZETA[13] * P['delta_F']
    dev_zeta = abs(m_d_zeta - m_d_exp) / m_d_exp * 100
    sigma_zeta = abs(m_d_zeta - m_d_exp) / m_d_unc

    # Best product formula
    m_d_product = ZETA[13] * ZETA[15] * P['delta_F']
    dev_product = abs(m_d_product - m_d_exp) / m_d_exp * 100
    sigma_product = abs(m_d_product - m_d_exp) / m_d_unc

    print(f"\nExperimental: {m_d_exp} ± {m_d_unc} MeV")
    print("\nFormula Comparison:")
    print(f"\n1. CURRENT: m_d = ln(107)")
    print(f"   Value: {m_d_ln107:.10f}")
    print(f"   Deviation: {dev_ln107:.6f}%")
    print(f"   Significance: {sigma_ln107:.2f}σ")

    print(f"\n2. NEW: m_d = ζ(13) × δ_F")
    print(f"   Value: {m_d_zeta:.10f}")
    print(f"   Deviation: {dev_zeta:.6f}%")
    print(f"   Significance: {sigma_zeta:.2f}σ")
    print(f"   Improvement: {dev_ln107/dev_zeta:.1f}× better precision")

    print(f"\n3. PRODUCT: m_d = ζ(13) × ζ(15) × δ_F")
    print(f"   Value: {m_d_product:.10f}")
    print(f"   Deviation: {dev_product:.6f}%")
    print(f"   Significance: {sigma_product:.2f}σ")

    print("\nPhysical Interpretation:")
    print("  - ζ(13) connects to M₁₃ = 8191 (dark matter mass scale)")
    print("  - δ_F is Feigenbaum constant (chaos/fractal dynamics)")
    print("  - Suggests down quark mass involves both:")
    print("    (a) Topological structure (ζ(13))")
    print("    (b) Chaotic/fractal dynamics (δ_F)")
    print(f"  - ζ(13) ≈ {ZETA[13]:.10f} (very close to 1)")
    print("  - Essentially: m_d ≈ δ_F with small ζ(13) correction")

    return {
        'ln107': (m_d_ln107, dev_ln107),
        'zeta13': (m_d_zeta, dev_zeta),
        'product': (m_d_product, dev_product),
    }


def analyze_priority_cosmology():
    """Analyze priority cosmological observables."""
    print("\n" + "="*80)
    print("PRIORITY COSMOLOGICAL OBSERVABLES")
    print("="*80)

    results = []

    # Omega_b (baryon density)
    print("\n1. Ω_b (Baryon Density)")
    print("-" * 40)
    Omega_b_exp = 0.0486
    Omega_b_unc = 0.001

    # Test patterns
    patterns_Omega_b = [
        ('ζ(9) / b2', ZETA[9] / P['b2']),
        ('ζ(13) × M2 / rank', ZETA[13] * P['M2'] / P['rank']),
        ('ζ(7) / (pi × M3)', ZETA[7] / (P['pi'] * P['M3'])),
        ('ln(2) × M2 / rank', P['ln2'] * P['M2'] / P['rank']),  # For comparison
        ('ζ(5) / b2', ZETA[5] / P['b2']),
        ('ζ(7) / b2', ZETA[7] / P['b2']),
    ]

    for formula, value in patterns_Omega_b:
        dev = abs(value - Omega_b_exp) / Omega_b_exp * 100
        sigma = abs(value - Omega_b_exp) / Omega_b_unc
        if dev < 5:
            print(f"  {formula}")
            print(f"    Value: {value:.6f}")
            print(f"    Deviation: {dev:.4f}%")
            print(f"    Significance: {sigma:.2f}σ")
            results.append(('Omega_b', formula, value, Omega_b_exp, dev, sigma))

    # H_0 (Hubble constant) - test various corrections
    print("\n2. H₀ (Hubble Constant)")
    print("-" * 40)
    H0_exp = 73.04
    H0_unc = 1.04
    H0_CMB = 67.36

    patterns_H0 = [
        ('H0_CMB × ζ(3)^(pi/8)', H0_CMB * ZETA[3]**(P['pi']/8)),
        ('H0_CMB × ζ(5)^(pi/8)', H0_CMB * ZETA[5]**(P['pi']/8)),
        ('H0_CMB × (ζ(3)/xi)^(pi/8)', H0_CMB * (ZETA[3]/P['xi'])**(P['pi']/8)),
        ('100 × ζ(3) / phi', 100 * ZETA[3] / P['phi']),
        ('100 × ζ(5) / phi', 100 * ZETA[5] / P['phi']),
        ('H0_CMB × ζ(9)', H0_CMB * ZETA[9]),
    ]

    for formula, value in patterns_H0:
        dev = abs(value - H0_exp) / H0_exp * 100
        sigma = abs(value - H0_exp) / H0_unc
        if dev < 5:
            print(f"  {formula}")
            print(f"    Value: {value:.6f}")
            print(f"    Deviation: {dev:.4f}%")
            print(f"    Significance: {sigma:.2f}σ")
            results.append(('H_0', formula, value, H0_exp, dev, sigma))

    # sigma_8 (matter fluctuations)
    print("\n3. σ₈ (Matter Fluctuation Amplitude)")
    print("-" * 40)
    sigma8_exp = 0.811
    sigma8_unc = 0.006

    patterns_sigma8 = [
        ('ζ(5) / phi', ZETA[5] / P['phi']),
        ('ζ(3) / phi', ZETA[3] / P['phi']),
        ('ζ(7) / phi', ZETA[7] / P['phi']),
        ('ln(2) × gamma', P['ln2'] * P['gamma']),
        ('4 / Weyl', 4 / P['Weyl']),
        ('ζ(9) - (pi - M2)/10', ZETA[9] - (P['pi'] - P['M2'])/10),
    ]

    for formula, value in patterns_sigma8:
        dev = abs(value - sigma8_exp) / sigma8_exp * 100
        sigma = abs(value - sigma8_exp) / sigma8_unc
        if dev < 5:
            print(f"  {formula}")
            print(f"    Value: {value:.6f}")
            print(f"    Deviation: {dev:.4f}%")
            print(f"    Significance: {sigma:.2f}σ")
            results.append(('sigma_8', formula, value, sigma8_exp, dev, sigma))

    return results


def analyze_gauge_couplings():
    """Analyze gauge coupling patterns."""
    print("\n" + "="*80)
    print("GAUGE COUPLINGS - EXTENDED SEARCH")
    print("="*80)

    results = []

    # alpha_s
    print("\n1. α_s (Strong Coupling)")
    print("-" * 40)
    alpha_s_exp = 0.1179
    alpha_s_unc = 0.0001

    # Known: sqrt(2)/12 ≈ 0.1178
    alpha_s_known = math.sqrt(2) / 12

    patterns_alpha_s = [
        ('sqrt(2) / 12', alpha_s_known),
        ('ζ(7) × gamma / Weyl', ZETA[7] * P['gamma'] / P['Weyl']),
        ('ζ(9) × gamma / Weyl', ZETA[9] * P['gamma'] / P['Weyl']),
        ('1 / (rank + 1/ζ(13))', 1 / (P['rank'] + 1/ZETA[13])),
        ('ζ(5) / (rank + 1)', ZETA[5] / (P['rank'] + 1)),
        ('ζ(7) / (rank + 1/2)', ZETA[7] / (P['rank'] + 0.5)),
    ]

    for formula, value in patterns_alpha_s:
        dev = abs(value - alpha_s_exp) / alpha_s_exp * 100
        sigma = abs(value - alpha_s_exp) / alpha_s_unc
        print(f"  {formula}")
        print(f"    Value: {value:.8f}")
        print(f"    Deviation: {dev:.4f}%")
        print(f"    Significance: {sigma:.2f}σ")
        results.append(('alpha_s', formula, value, alpha_s_exp, dev, sigma))

    # Hypercharge coupling g_Y
    print("\n2. g_Y (Hypercharge Coupling)")
    print("-" * 40)
    # g_Y ≈ 0.357 (from sin²θ_W and α)
    g_Y_exp = 0.357
    g_Y_unc = 0.002

    patterns_g_Y = [
        ('sqrt(ζ(3) / Weyl)', math.sqrt(ZETA[3] / P['Weyl'])),
        ('ζ(5) / M2', ZETA[5] / P['M2']),
        ('ζ(7) / M2', ZETA[7] / P['M2']),
        ('1 / sqrt(rank)', 1 / math.sqrt(P['rank'])),
        ('sqrt(Weyl / 39)', math.sqrt(P['Weyl'] / 39)),
    ]

    for formula, value in patterns_g_Y:
        dev = abs(value - g_Y_exp) / g_Y_exp * 100
        sigma = abs(value - g_Y_exp) / g_Y_unc
        if dev < 5:
            print(f"  {formula}")
            print(f"    Value: {value:.6f}")
            print(f"    Deviation: {dev:.4f}%")
            print(f"    Significance: {sigma:.2f}σ")
            results.append(('g_Y', formula, value, g_Y_exp, dev, sigma))

    # SU(2) coupling g_2
    print("\n3. g₂ (SU(2) Coupling)")
    print("-" * 40)
    g_2_exp = 0.653
    g_2_unc = 0.003

    patterns_g_2 = [
        ('sqrt(ζ(3) / M2)', math.sqrt(ZETA[3] / P['M2'])),
        ('ζ(5) / phi', ZETA[5] / P['phi']),
        ('ζ(3) / M2', ZETA[3] / P['M2']),
        ('2 / M2', 2 / P['M2']),
        ('M3 / (pi × M2)', P['M3'] / (P['pi'] * P['M2'])),
    ]

    for formula, value in patterns_g_2:
        dev = abs(value - g_2_exp) / g_2_exp * 100
        sigma = abs(value - g_2_exp) / g_2_unc
        if dev < 5:
            print(f"  {formula}")
            print(f"    Value: {value:.6f}")
            print(f"    Deviation: {dev:.4f}%")
            print(f"    Significance: {sigma:.2f}σ")
            results.append(('g_2', formula, value, g_2_exp, dev, sigma))

    return results


def analyze_neutrino_masses():
    """Analyze neutrino mass squared differences."""
    print("\n" + "="*80)
    print("NEUTRINO MASS SQUARED DIFFERENCES")
    print("="*80)

    results = []

    # Δm²₂₁ (solar)
    print("\n1. Δm²₂₁ (Solar)")
    print("-" * 40)
    Dm21_exp = 7.53e-5  # eV²
    Dm21_unc = 0.18e-5

    # Try to find patterns involving small numbers
    patterns_Dm21 = [
        ('ζ(13) × 1e-4 / phi', ZETA[13] * 1e-4 / P['phi']),
        ('1e-4 × Weyl / rank', 1e-4 * P['Weyl'] / P['rank']),
        ('1e-4 / (ζ(5) × phi)', 1e-4 / (ZETA[5] * P['phi'])),
        ('ζ(21) × 1e-4 / phi', ZETA[21] * 1e-4 / P['phi']),
    ]

    for formula, value in patterns_Dm21:
        dev = abs(value - Dm21_exp) / Dm21_exp * 100
        sigma = abs(value - Dm21_exp) / Dm21_unc
        if dev < 5:
            print(f"  {formula}")
            print(f"    Value: {value:.6e}")
            print(f"    Deviation: {dev:.4f}%")
            print(f"    Significance: {sigma:.2f}σ")
            results.append(('Delta_m21_sq', formula, value, Dm21_exp, dev, sigma))

    # Δm²₃₁ (atmospheric)
    print("\n2. Δm²₃₁ (Atmospheric)")
    print("-" * 40)
    Dm31_exp = 2.453e-3  # eV²
    Dm31_unc = 0.033e-3

    patterns_Dm31 = [
        ('ζ(5) × 1e-3 / (M2 × phi)', ZETA[5] * 1e-3 / (P['M2'] * P['phi'])),
        ('1e-3 × M2 / phi', 1e-3 * P['M2'] / P['phi']),
        ('1e-3 / (ζ(3) × phi)', 1e-3 / (ZETA[3] * P['phi'])),
    ]

    for formula, value in patterns_Dm31:
        dev = abs(value - Dm31_exp) / Dm31_exp * 100
        sigma = abs(value - Dm31_exp) / Dm31_unc
        if dev < 5:
            print(f"  {formula}")
            print(f"    Value: {value:.6e}")
            print(f"    Deviation: {dev:.4f}%")
            print(f"    Significance: {sigma:.2f}σ")
            results.append(('Delta_m31_sq', formula, value, Dm31_exp, dev, sigma))

    return results


def analyze_ckm_angles():
    """Analyze CKM mixing angles."""
    print("\n" + "="*80)
    print("CKM MATRIX ANGLES")
    print("="*80)

    results = []

    # Cabibbo angle (already known: θ_C ≈ θ₁₃ × sqrt(7/3))
    print("\n1. θ_C (Cabibbo Angle)")
    print("-" * 40)
    theta_C_exp = 13.04  # degrees
    theta_C_unc = 0.05

    # sin(θ_C) ≈ 0.2254
    sin_theta_C_exp = 0.2254

    patterns_sinC = [
        ('ζ(3) / (Weyl + 1/ζ(7))', ZETA[3] / (P['Weyl'] + 1/ZETA[7])),
        ('ζ(5) / (Weyl - 1/ζ(9))', ZETA[5] / (P['Weyl'] - 1/ZETA[9])),
        ('sqrt(ζ(7) / b2)', math.sqrt(ZETA[7] / P['b2'])),
        ('1 / (Weyl - 1/ζ(11))', 1 / (P['Weyl'] - 1/ZETA[11])),
    ]

    print("  Testing sin(θ_C) patterns:")
    for formula, value in patterns_sinC:
        dev = abs(value - sin_theta_C_exp) / sin_theta_C_exp * 100
        if dev < 5:
            angle_deg = math.degrees(math.asin(value))
            sigma = abs(angle_deg - theta_C_exp) / theta_C_unc
            print(f"    {formula}")
            print(f"      sin(θ_C): {value:.6f}")
            print(f"      θ_C: {angle_deg:.4f}°")
            print(f"      Deviation: {dev:.4f}%")
            print(f"      Significance: {sigma:.2f}σ")
            results.append(('sin_theta_C', formula, value, sin_theta_C_exp, dev, sigma))

    return results


def main():
    """Run refined analysis."""

    print("\n" + "="*80)
    print("REFINED ODD ZETA ANALYSIS")
    print("Focus: Meaningful patterns in priority observables")
    print("="*80)

    all_results = []

    # Run all analyses
    md_results = analyze_down_quark()
    all_results.extend(analyze_priority_cosmology())
    all_results.extend(analyze_gauge_couplings())
    all_results.extend(analyze_neutrino_masses())
    all_results.extend(analyze_ckm_angles())

    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL NEW PATTERNS")
    print("="*80)

    if all_results:
        # Sort by deviation
        all_results.sort(key=lambda x: x[4])

        print(f"\nTotal patterns found: {len(all_results)}")
        print("\nTop 10 by precision:")
        print("-" * 80)

        for i, (obs, formula, pred, exp, dev, sigma) in enumerate(all_results[:10], 1):
            print(f"\n{i}. {obs}")
            print(f"   Formula: {formula}")
            print(f"   Predicted: {pred:.8f}")
            print(f"   Experimental: {exp:.8f}")
            print(f"   Deviation: {dev:.4f}%")
            print(f"   Significance: {sigma:.2f}σ")

        # Save to CSV
        data_dir = Path(__file__).resolve().parent.parent / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        output_file = data_dir / 'refined_zeta_patterns.csv'
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Observable', 'Formula', 'Predicted', 'Experimental',
                           'Deviation_%', 'Sigma'])
            for row in all_results:
                writer.writerow(row)

        print(f"\n\nResults saved to: {output_file}")

    else:
        print("\nNo patterns found within tolerance.")


if __name__ == '__main__':
    main()
