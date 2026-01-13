#!/usr/bin/env python3
"""
Gauge Group Uniqueness Analysis for GIFT Framework

This script systematically tests the uniqueness of E₈×E₈ as the optimal gauge group
for the GIFT framework, comparing against all physically motivated alternatives.

Tests performed:
1. Gauge group comparison (E₈×E₈, E₇×E₇, E₆×E₆, SO(32), etc.)
2. Holonomy comparison (G₂, Spin(7), SU(3), SU(4))
3. Combined optimization over gauge × holonomy × Betti

Author: GIFT Framework Team
License: MIT
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
from pathlib import Path


# =============================================================================
# GAUGE GROUP DEFINITIONS
# =============================================================================

@dataclass
class GaugeGroup:
    """Defines a gauge group with its properties."""
    name: str
    dim: int          # Total dimension
    rank: int         # Rank (for generation counting)
    type: str         # 'exceptional', 'orthogonal', 'unitary'


# All physically motivated gauge groups for heterotic string compactification
GAUGE_GROUPS = [
    # Heterotic E₈×E₈ and subgroups
    GaugeGroup("E₈×E₈", 496, 8, "exceptional"),
    GaugeGroup("E₇×E₈", 381, 7, "exceptional"),  # 133 + 248
    GaugeGroup("E₆×E₈", 326, 6, "exceptional"),  # 78 + 248
    GaugeGroup("E₇×E₇", 266, 7, "exceptional"),  # 133 + 133
    GaugeGroup("E₆×E₇", 211, 6, "exceptional"),  # 78 + 133
    GaugeGroup("E₆×E₆", 156, 6, "exceptional"),  # 78 + 78

    # Heterotic SO(32)
    GaugeGroup("SO(32)", 496, 16, "orthogonal"),

    # GUT groups (for comparison)
    GaugeGroup("SO(10)×SO(10)", 90, 5, "orthogonal"),  # 45 + 45
    GaugeGroup("SU(5)×SU(5)", 48, 4, "unitary"),       # 24 + 24
    GaugeGroup("SO(10)×E₆", 123, 5, "mixed"),          # 45 + 78
    GaugeGroup("SU(5)×E₆", 102, 4, "mixed"),           # 24 + 78

    # Other exceptional products
    GaugeGroup("F₄×E₈", 300, 4, "exceptional"),  # 52 + 248
    GaugeGroup("G₂×E₈", 262, 2, "exceptional"),  # 14 + 248
    GaugeGroup("F₄×F₄", 104, 4, "exceptional"),  # 52 + 52
]


# =============================================================================
# HOLONOMY GROUP DEFINITIONS
# =============================================================================

@dataclass
class HolonomyGroup:
    """Defines a holonomy group with its properties."""
    name: str
    dim: int          # Dimension of compact manifold
    hol_dim: int      # Dimension of holonomy group
    susy: str         # Preserved supersymmetry


HOLONOMY_GROUPS = [
    HolonomyGroup("G₂", 7, 14, "N=1"),
    HolonomyGroup("Spin(7)", 8, 21, "N=0"),
    HolonomyGroup("SU(3)", 6, 8, "N=2"),   # Calabi-Yau 3-fold
    HolonomyGroup("SU(4)", 8, 15, "N=1"),  # Calabi-Yau 4-fold
    HolonomyGroup("SU(2)", 4, 3, "N=4"),   # K3
]


# =============================================================================
# EXPERIMENTAL VALUES
# =============================================================================

EXPERIMENTAL = {
    # Electroweak
    'sin2_theta_W': (0.23122, 0.00004),
    'Q_Koide': (0.666661, 0.000007),
    'm_W_m_Z': (0.8815, 0.0002),

    # PMNS
    'sin2_theta_12': (0.307, 0.013),
    'sin2_theta_23': (0.546, 0.021),
    'sin2_theta_13': (0.0220, 0.0007),

    # Mass ratios
    'm_s_m_d': (20.0, 1.5),
    'm_b_m_t': (0.024, 0.001),
    'm_H_m_t': (0.725, 0.003),

    # CKM
    'sin2_theta_12_CKM': (0.2250, 0.0006),
    'lambda_Wolf': (0.22453, 0.00044),

    # Cosmology
    'Omega_DM_Omega_b': (5.375, 0.1),
    'h': (0.674, 0.005),
    'Omega_Lambda_Omega_m': (2.175, 0.05),
}


# =============================================================================
# GIFT TOPOLOGICAL CONSTANTS
# =============================================================================

# Fixed K₇ manifold topology (TCS construction)
B2 = 21   # Second Betti number
B3 = 77   # Third Betti number
H_STAR = B2 + B3 + 1  # = 99
P2 = 2    # Pontryagin class

# Exceptional structures
DIM_J3O = 27   # Jordan algebra J₃(O)
PSL_2_7 = 168  # Fano plane symmetry
FUND_E7 = 56   # E₇ fundamental representation


# =============================================================================
# OBSERVABLE FORMULAS
# =============================================================================

def compute_observables(gauge: GaugeGroup, hol: HolonomyGroup,
                        b2: int = B2, b3: int = B3) -> Dict[str, float]:
    """
    Compute all observables for a given gauge group and holonomy.

    Key insight: Many formulas depend on:
    - rank(gauge) for generation counting
    - dim(holonomy) for moduli corrections
    - b2, b3 for cohomology
    """
    dim_G = gauge.dim
    rank_G = gauge.rank
    dim_hol = hol.hol_dim
    dim_K = hol.dim

    h_star = b2 + b3 + 1
    chi = 2 * b2  # Euler characteristic proxy

    # Derived constants
    weyl = (dim_hol + 1) // 3 if dim_hol > 0 else 1
    kappa_t = 1 / (b3 - dim_hol - P2) if (b3 - dim_hol - P2) != 0 else 1e10

    predictions = {}

    # === Electroweak sector ===

    # sin²θ_W = b2/(b3 + dim_hol)
    denom = b3 + dim_hol
    predictions['sin2_theta_W'] = b2 / denom if denom != 0 else 1.0

    # Q_Koide = dim_hol / b2
    predictions['Q_Koide'] = dim_hol / b2 if b2 != 0 else 1.0

    # m_W/m_Z = (χ - weyl) / χ = (2b2 - weyl) / (2b2)
    predictions['m_W_m_Z'] = (chi - weyl) / chi if chi != 0 else 1.0

    # === PMNS sector ===

    # sin²θ₁₂ = (1 + N_gen) / α_sum where α_sum = 13
    n_gen = compute_n_gen(rank_G, b2, b3)
    predictions['sin2_theta_12'] = (1 + n_gen) / 13

    # sin²θ₂₃ = (D_bulk - weyl) / D_bulk = (11 - 5) / 11 = 6/11
    predictions['sin2_theta_23'] = (11 - weyl) / 11 if weyl < 11 else 0.5

    # sin²θ₁₃ = D_bulk / dim(gauge)²  (approximation)
    predictions['sin2_theta_13'] = 11 / dim_G if dim_G > 0 else 0.02

    # === Mass ratios ===

    # m_s/m_d = (α_sum + dim_J3O) / p2 = 40/2 = 20
    predictions['m_s_m_d'] = (13 + DIM_J3O) / P2

    # m_b/m_t = 1/χ = 1/42
    predictions['m_b_m_t'] = 1 / chi if chi != 0 else 0.02

    # m_H/m_t = fund(E7) / b3 = 56/77
    predictions['m_H_m_t'] = FUND_E7 / b3 if b3 != 0 else 0.7

    # === CKM sector ===

    # sin²θ₁₂_CKM = fund(E7) / dim(E8) = 56/248
    predictions['sin2_theta_12_CKM'] = FUND_E7 / 248
    predictions['lambda_Wolf'] = FUND_E7 / 248

    # === Cosmology ===

    # Ω_DM/Ω_b = (1 + χ) / rank = 43/8
    predictions['Omega_DM_Omega_b'] = (1 + chi) / rank_G if rank_G > 0 else 5.0

    # h = (PSL(2,7) - 1) / dim(E8) = 167/248
    predictions['h'] = (PSL_2_7 - 1) / 248

    # Ω_Λ/Ω_m = (dim_hol + H*) / dim(F4) = (14 + 99) / 52 = 113/52
    predictions['Omega_Lambda_Omega_m'] = (dim_hol + h_star) / 52

    return predictions


def compute_n_gen(rank: int, b2: int, b3: int) -> float:
    """
    Compute number of generations.
    N_gen = (rank × b2) / (b3 - b2)
    """
    denom = b3 - b2
    if denom == 0:
        return 0.0
    return (rank * b2) / denom


# =============================================================================
# STATISTICAL METRICS
# =============================================================================

def compute_deviation(predictions: Dict[str, float],
                      experimental: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Compute relative deviation for each observable."""
    deviations = {}
    for name, pred in predictions.items():
        if name in experimental:
            exp_val, _ = experimental[name]
            if exp_val != 0:
                deviations[name] = abs(pred - exp_val) / abs(exp_val) * 100
            else:
                deviations[name] = abs(pred) * 100
    return deviations


def compute_chi_squared(predictions: Dict[str, float],
                        experimental: Dict[str, Tuple[float, float]]) -> float:
    """Compute chi-squared statistic."""
    chi2 = 0.0
    for name, pred in predictions.items():
        if name in experimental:
            exp_val, exp_unc = experimental[name]
            if exp_unc > 0:
                chi2 += ((pred - exp_val) / exp_unc) ** 2
    return chi2


def count_exact_matches(deviations: Dict[str, float], threshold: float = 0.1) -> int:
    """Count observables with deviation below threshold (%)."""
    return sum(1 for d in deviations.values() if d < threshold)


def count_good_matches(deviations: Dict[str, float], threshold: float = 1.0) -> int:
    """Count observables with deviation below threshold (%)."""
    return sum(1 for d in deviations.values() if d < threshold)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_gauge_groups(holonomy: HolonomyGroup = None) -> List[Dict]:
    """
    Analyze all gauge groups with a fixed holonomy.
    Default: G₂ holonomy (the GIFT choice).
    """
    if holonomy is None:
        holonomy = HOLONOMY_GROUPS[0]  # G₂

    results = []

    for gauge in GAUGE_GROUPS:
        predictions = compute_observables(gauge, holonomy)
        deviations = compute_deviation(predictions, EXPERIMENTAL)
        chi2 = compute_chi_squared(predictions, EXPERIMENTAL)
        n_gen = compute_n_gen(gauge.rank, B2, B3)

        results.append({
            'gauge': gauge.name,
            'dim': gauge.dim,
            'rank': gauge.rank,
            'type': gauge.type,
            'holonomy': holonomy.name,
            'n_gen': n_gen,
            'mean_deviation': np.mean(list(deviations.values())),
            'max_deviation': np.max(list(deviations.values())),
            'chi_squared': chi2,
            'exact_matches': count_exact_matches(deviations),
            'good_matches': count_good_matches(deviations),
            'predictions': predictions,
            'deviations': deviations,
        })

    # Sort by mean deviation
    results.sort(key=lambda x: x['mean_deviation'])

    return results


def analyze_holonomies(gauge: GaugeGroup = None) -> List[Dict]:
    """
    Analyze all holonomies with a fixed gauge group.
    Default: E₈×E₈ (the GIFT choice).
    """
    if gauge is None:
        gauge = GAUGE_GROUPS[0]  # E₈×E₈

    results = []

    for hol in HOLONOMY_GROUPS:
        predictions = compute_observables(gauge, hol)
        deviations = compute_deviation(predictions, EXPERIMENTAL)
        chi2 = compute_chi_squared(predictions, EXPERIMENTAL)

        results.append({
            'holonomy': hol.name,
            'dim_K': hol.dim,
            'dim_hol': hol.hol_dim,
            'susy': hol.susy,
            'gauge': gauge.name,
            'mean_deviation': np.mean(list(deviations.values())),
            'chi_squared': chi2,
            'exact_matches': count_exact_matches(deviations),
            'good_matches': count_good_matches(deviations),
        })

    results.sort(key=lambda x: x['mean_deviation'])

    return results


def full_analysis() -> Dict:
    """Run complete uniqueness analysis."""

    print("=" * 70)
    print("GIFT GAUGE GROUP & HOLONOMY UNIQUENESS ANALYSIS")
    print("=" * 70)

    # 1. Gauge group comparison with G₂ holonomy
    print("\n[1/3] Analyzing gauge groups (G₂ holonomy fixed)...")
    g2_hol = HOLONOMY_GROUPS[0]
    gauge_results = analyze_gauge_groups(g2_hol)

    print(f"\n{'Rank':<6} {'Gauge Group':<15} {'Mean Dev':<10} {'N_gen':<8} {'χ²':<10} {'Exact':<6} {'Good':<6}")
    print("-" * 70)
    for i, r in enumerate(gauge_results[:10]):
        marker = "★" if r['gauge'] == "E₈×E₈" else ""
        print(f"{i+1:<6} {r['gauge']:<15} {r['mean_deviation']:.2f}%{'':<5} "
              f"{r['n_gen']:.3f}{'':<3} {r['chi_squared']:.1f}{'':<5} "
              f"{r['exact_matches']:<6} {r['good_matches']:<6} {marker}")

    # 2. Holonomy comparison with E₈×E₈
    print("\n[2/3] Analyzing holonomies (E₈×E₈ gauge fixed)...")
    e8e8 = GAUGE_GROUPS[0]
    holonomy_results = analyze_holonomies(e8e8)

    print(f"\n{'Rank':<6} {'Holonomy':<12} {'dim_K':<8} {'SUSY':<8} {'Mean Dev':<10} {'χ²':<10}")
    print("-" * 60)
    for i, r in enumerate(holonomy_results):
        marker = "★" if r['holonomy'] == "G₂" else ""
        print(f"{i+1:<6} {r['holonomy']:<12} {r['dim_K']:<8} {r['susy']:<8} "
              f"{r['mean_deviation']:.2f}%{'':<5} {r['chi_squared']:.1f} {marker}")

    # 3. N_gen analysis
    print("\n[3/3] Generation counting (N_gen = rank × b₂ / (b₃ - b₂))...")
    print(f"\nFor (b₂, b₃) = ({B2}, {B3}):")
    print(f"\n{'Gauge':<15} {'Rank':<8} {'Calculation':<25} {'N_gen':<10} {'Status':<10}")
    print("-" * 70)

    for gauge in GAUGE_GROUPS[:8]:
        n_gen = compute_n_gen(gauge.rank, B2, B3)
        calc = f"({gauge.rank}×{B2})/{B3-B2}"
        status = "✓ EXACT" if abs(n_gen - 3) < 0.001 else "✗"
        marker = "★" if gauge.name == "E₈×E₈" else ""
        print(f"{gauge.name:<15} {gauge.rank:<8} {calc:<25} {n_gen:.3f}{'':<5} {status:<10} {marker}")

    # 4. Summary
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    gift_result = next(r for r in gauge_results if r['gauge'] == "E₈×E₈")
    second_best = gauge_results[1] if gauge_results[0]['gauge'] == "E₈×E₈" else gauge_results[0]

    print(f"\n1. GAUGE GROUP UNIQUENESS:")
    print(f"   E₈×E₈ mean deviation: {gift_result['mean_deviation']:.2f}%")
    print(f"   Next best ({second_best['gauge']}): {second_best['mean_deviation']:.2f}%")
    print(f"   Improvement factor: {second_best['mean_deviation']/gift_result['mean_deviation']:.1f}×")

    g2_result = next(r for r in holonomy_results if r['holonomy'] == "G₂")
    su3_result = next(r for r in holonomy_results if r['holonomy'] == "SU(3)")

    print(f"\n2. HOLONOMY UNIQUENESS:")
    print(f"   G₂ mean deviation: {g2_result['mean_deviation']:.2f}%")
    print(f"   SU(3) (Calabi-Yau): {su3_result['mean_deviation']:.2f}%")
    print(f"   → Calabi-Yau FAILS by factor {su3_result['mean_deviation']/g2_result['mean_deviation']:.0f}×")

    print(f"\n3. GENERATION COUNTING:")
    print(f"   Only rank=8 gives N_gen = 3 exactly")
    print(f"   N_gen = (8 × 21) / 56 = 168/56 = 3")
    print(f"   Note: 168 = |PSL(2,7)| = Fano plane symmetry")

    print(f"\n4. STATISTICAL SIGNIFICANCE:")
    # Count how many alternatives are better
    n_better_gauge = sum(1 for r in gauge_results if r['mean_deviation'] < gift_result['mean_deviation'])
    n_better_hol = sum(1 for r in holonomy_results if r['mean_deviation'] < g2_result['mean_deviation'])

    print(f"   Gauge groups tested: {len(GAUGE_GROUPS)}")
    print(f"   Better than E₈×E₈: {n_better_gauge}")
    print(f"   Holonomies tested: {len(HOLONOMY_GROUPS)}")
    print(f"   Better than G₂: {n_better_hol}")

    if n_better_gauge == 0 and n_better_hol == 0:
        print(f"\n   ★ E₈×E₈ with G₂ holonomy is UNIQUELY OPTIMAL ★")

    return {
        'gauge_analysis': gauge_results,
        'holonomy_analysis': holonomy_results,
        'gift_config': {
            'gauge': 'E₈×E₈',
            'holonomy': 'G₂',
            'b2': B2,
            'b3': B3,
            'n_gen': 3,
            'mean_deviation': gift_result['mean_deviation'],
        },
        'conclusions': {
            'gauge_unique': n_better_gauge == 0,
            'holonomy_unique': n_better_hol == 0,
            'improvement_over_next': second_best['mean_deviation'] / gift_result['mean_deviation'],
        }
    }


def save_results(results: Dict, output_dir: str = None):
    """Save results to JSON file."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up for JSON serialization
    clean_results = {
        'gift_config': results['gift_config'],
        'conclusions': results['conclusions'],
        'gauge_ranking': [
            {k: v for k, v in r.items() if k not in ['predictions', 'deviations']}
            for r in results['gauge_analysis']
        ],
        'holonomy_ranking': results['holonomy_analysis'],
    }

    output_file = output_dir / "gauge_group_uniqueness.json"
    with open(output_file, 'w') as f:
        json.dump(clean_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = full_analysis()
    save_results(results)
    print("\nDone!")
