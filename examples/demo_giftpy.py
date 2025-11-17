#!/usr/bin/env python3
"""
GIFTpy Demonstration Script

This script showcases the main features of the GIFTpy package.
"""
import giftpy

def main():
    print("=" * 70)
    print(" " * 20 + "GIFTpy Demonstration")
    print("=" * 70)
    print()

    # Initialize GIFT framework
    print("🔧 Initializing GIFT framework...")
    gift = giftpy.GIFT()
    print(f"✓ GIFTpy version {gift.version}")
    print()

    # Display topological constants
    print("📐 Topological Constants")
    print("-" * 70)
    print(f"  b₂(K₇) = {gift.constants.b2} (harmonic 2-forms)")
    print(f"  b₃(K₇) = {gift.constants.b3} (harmonic 3-forms)")
    print(f"  dim(E₈) = {gift.constants.dim_E8}")
    print(f"  dim(G₂) = {gift.constants.dim_G2}")
    print(f"  β₀ = {gift.constants.beta0:.10f}")
    print(f"  ξ = {gift.constants.xi:.10f} (DERIVED!)")
    print()

    # Gauge sector predictions
    print("⚛️  Gauge Sector Predictions")
    print("-" * 70)

    alpha_s = gift.gauge.alpha_s()
    print(f"  α_s(M_Z) = {alpha_s:.6f}")
    print(f"    Formula: √2/12")
    print(f"    Experimental: 0.1179 ± 0.0010")
    print(f"    Deviation: {abs(alpha_s - 0.1179)/0.1179*100:.3f}%")
    print()

    sin2 = gift.gauge.sin2theta_W()
    print(f"  sin²θ_W(M_Z) = {sin2:.6f}")
    print(f"    Formula: 3/13")
    print(f"    Experimental: 0.23122 ± 0.00004")
    print(f"    Deviation: {abs(sin2 - 0.23122)/0.23122*100:.3f}%")
    print()

    alpha_inv = gift.gauge.alpha_inv()
    print(f"  α⁻¹(M_Z) = {alpha_inv:.6f}")
    print(f"    Formula: 2⁷ - 1/24")
    print(f"    Experimental: 127.952 ± 0.001")
    print(f"    Deviation: {abs(alpha_inv - 127.952)/127.952*100:.4f}%")
    print()

    # Lepton sector - The spectacular predictions!
    print("⚡ Lepton Sector - Spectacular Predictions!")
    print("-" * 70)

    Q_Koide = gift.lepton.Q_Koide()
    print(f"  Q_Koide = {Q_Koide}")
    print(f"    Formula: dim(G₂)/b₂(K₇) = 14/21 = 2/3 (EXACT!)")
    print(f"    Experimental: 0.666661 ± 0.000007")
    print(f"    Deviation: {abs(Q_Koide - 0.666661)/0.666661*100:.4f}% 🎯")
    print()

    m_mu_m_e = gift.lepton.m_mu_m_e()
    print(f"  m_μ/m_e = {m_mu_m_e:.6f}")
    print(f"    Formula: 27^φ (φ = golden ratio)")
    print(f"    Experimental: 206.7682827 ± 0.0000046")
    print(f"    Deviation: {abs(m_mu_m_e - 206.7682827)/206.7682827*100:.4f}%")
    print()

    # Neutrino sector
    print("🔄 Neutrino Sector")
    print("-" * 70)

    import numpy as np
    delta_CP = gift.neutrino.delta_CP()
    delta_CP_deg = np.degrees(delta_CP)
    print(f"  δ_CP = {delta_CP_deg:.1f}°")
    print(f"    Formula: ζ(3) + √5")
    print(f"    Experimental: 197° ± 24°")
    print(f"    Deviation: {abs(delta_CP_deg - 197)/197*100:.3f}%")
    print()

    # Cosmology
    print("🌌 Cosmology")
    print("-" * 70)

    Omega_DE = gift.cosmology.Omega_DE()
    print(f"  Ω_DE = {Omega_DE:.6f}")
    print(f"    Formula: ln(2)")
    print(f"    Experimental: 0.6847 ± 0.0073")
    print(f"    Deviation: {abs(Omega_DE - 0.6847)/0.6847*100:.2f}%")
    print()

    # Full validation
    print("📊 Full Validation")
    print("-" * 70)
    validation = gift.validate(verbose=False)
    print(f"  Total observables: {validation.n_observables}")
    print(f"  Mean deviation: {validation.mean_deviation:.4f}%")
    print(f"  Median deviation: {validation.median_deviation:.4f}%")
    print(f"  Max deviation: {validation.max_deviation:.4f}%")
    print(f"  Exact (<0.01%): {validation.n_exact}")
    print(f"  Exceptional (<0.1%): {validation.n_exceptional}")
    print(f"  Excellent (<0.5%): {validation.n_excellent}")
    print()

    # Display all results
    print("📋 All Predictions")
    print("-" * 70)
    results = gift.compute_all()
    print(results[['name', 'value', 'experimental', 'deviation_%']].to_string(index=False))
    print()

    print("=" * 70)
    print("✓ Demo complete!")
    print("=" * 70)
    print()
    print("For more information:")
    print("  - Documentation: https://github.com/gift-framework/GIFT")
    print("  - Theory: publications/gift_main.md")
    print("  - Issues: https://github.com/gift-framework/GIFT/issues")
    print()


if __name__ == "__main__":
    main()
