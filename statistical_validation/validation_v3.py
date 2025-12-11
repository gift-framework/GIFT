#!/usr/bin/env python3
"""
Statistical Validation for GIFT Framework v3.0

Tests the 18 dimensionless predictions against 10,000+ alternative G2 manifold
configurations by varying topological parameters (b2, b3) and computing
predictions using the ACTUAL topological formulas.

Key improvement over v2.3: Uses real formula calculations, not random perturbations.
"""

import numpy as np
from scipy import stats
from scipy.special import zeta
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
from pathlib import Path
import time

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2

@dataclass
class G2Config:
    """G2 manifold configuration with topological parameters"""
    name: str
    b2: int      # Second Betti number
    b3: int      # Third Betti number
    dim_g2: int = 14      # G2 dimension (fixed)
    dim_e8: int = 248     # E8 dimension (fixed)
    rank_e8: int = 8      # E8 rank (fixed)
    dim_k7: int = 7       # K7 dimension (fixed)
    dim_j3o: int = 27     # Jordan algebra dimension (fixed)
    p2: int = 2           # Binary duality (fixed)
    weyl: int = 5         # Weyl factor (fixed)
    d_bulk: int = 11      # Bulk dimension (fixed)

    @property
    def h_star(self) -> float:
        """Effective cohomology H* = b2 + b3 + 1"""
        return self.b2 + self.b3 + 1

    @property
    def n_gen(self) -> float:
        """Number of generations from index theorem"""
        # N_gen from: (rank + N_gen) * b2 = N_gen * b3
        # => N_gen = rank * b2 / (b3 - b2)
        if self.b3 <= self.b2:
            return float('inf')
        return self.rank_e8 * self.b2 / (self.b3 - self.b2)

    @property
    def kappa_t(self) -> float:
        """Torsion magnitude kappa_T = 1/(b3 - dim_G2 - p2)"""
        denom = self.b3 - self.dim_g2 - self.p2
        if denom <= 0:
            return float('inf')
        return 1.0 / denom

    @property
    def det_g(self) -> float:
        """Metric determinant det(g) = p2 + 1/(b2 + dim_G2 - N_gen)"""
        # For alternative configs, use derived N_gen
        n = self.n_gen if np.isfinite(self.n_gen) and self.n_gen > 0 else 3
        denom = self.b2 + self.dim_g2 - n
        if denom <= 0:
            return float('inf')
        return self.p2 + 1.0 / denom


# Experimental values for v3.0 (18 dimensionless observables)
EXPERIMENTAL_V3 = {
    # Structural (4)
    'N_gen': {'value': 3.0, 'uncertainty': 0.0, 'has_exp': True},
    'tau': {'value': 3.897, 'uncertainty': 0.001, 'has_exp': False},  # Internal
    'kappa_T': {'value': 0.0164, 'uncertainty': 0.0001, 'has_exp': False},  # Internal
    'det_g': {'value': 2.03125, 'uncertainty': 0.001, 'has_exp': False},  # Internal

    # Gauge (2)
    'sin2_theta_W': {'value': 0.23122, 'uncertainty': 0.00004, 'has_exp': True},
    'alpha_s': {'value': 0.1179, 'uncertainty': 0.0009, 'has_exp': True},

    # Lepton masses (3)
    'Q_Koide': {'value': 0.666661, 'uncertainty': 0.000007, 'has_exp': True},
    'm_tau_m_e': {'value': 3477.15, 'uncertainty': 0.05, 'has_exp': True},
    'm_mu_m_e': {'value': 206.768, 'uncertainty': 0.001, 'has_exp': True},

    # Quark masses (1)
    'm_s_m_d': {'value': 20.0, 'uncertainty': 1.0, 'has_exp': True},

    # Neutrino (4)
    'delta_CP': {'value': 197.0, 'uncertainty': 24.0, 'has_exp': True},
    'theta_13': {'value': 8.54, 'uncertainty': 0.12, 'has_exp': True},
    'theta_23': {'value': 49.3, 'uncertainty': 1.0, 'has_exp': True},
    'theta_12': {'value': 33.41, 'uncertainty': 0.75, 'has_exp': True},

    # Higgs & Cosmology (3)
    'lambda_H': {'value': 0.129, 'uncertainty': 0.003, 'has_exp': True},
    'Omega_DE': {'value': 0.6847, 'uncertainty': 0.0073, 'has_exp': True},
    'n_s': {'value': 0.9649, 'uncertainty': 0.0042, 'has_exp': True},

    # Fine structure (1)
    'alpha_inv': {'value': 137.035999, 'uncertainty': 0.000001, 'has_exp': True},
}


def compute_predictions(cfg: G2Config) -> Dict[str, float]:
    """
    Compute all 18 dimensionless predictions from topological parameters.
    Uses the ACTUAL GIFT formulas.
    """
    preds = {}

    # === STRUCTURAL (4) ===

    # 1. N_gen: from index theorem
    preds['N_gen'] = cfg.n_gen

    # 2. tau: hierarchy parameter
    # tau = dim(E8×E8) * b2 / (dim(J3O) * H*)
    preds['tau'] = (2 * cfg.dim_e8 * cfg.b2) / (cfg.dim_j3o * cfg.h_star)

    # 3. kappa_T: torsion magnitude
    preds['kappa_T'] = cfg.kappa_t

    # 4. det(g): metric determinant
    preds['det_g'] = cfg.det_g

    # === GAUGE SECTOR (2) ===

    # 5. sin²θ_W = b2/(b3 + dim_G2)
    preds['sin2_theta_W'] = cfg.b2 / (cfg.b3 + cfg.dim_g2)

    # 6. α_s = √2/12 (fixed - doesn't depend on b2/b3)
    preds['alpha_s'] = np.sqrt(2) / 12

    # === LEPTON SECTOR (3) ===

    # 7. Q_Koide = dim_G2/b2
    preds['Q_Koide'] = cfg.dim_g2 / cfg.b2

    # 8. m_tau/m_e = dim_K7 + 10*dim_E8 + 10*H*
    preds['m_tau_m_e'] = cfg.dim_k7 + 10 * cfg.dim_e8 + 10 * cfg.h_star

    # 9. m_mu/m_e = 27^φ (fixed - doesn't depend on b2/b3)
    preds['m_mu_m_e'] = 27 ** PHI

    # === QUARK SECTOR (1) ===

    # 10. m_s/m_d = p2² × Weyl (fixed)
    preds['m_s_m_d'] = cfg.p2 ** 2 * cfg.weyl

    # === NEUTRINO SECTOR (4) ===

    # 11. δ_CP = dim_K7 × dim_G2 + H*
    preds['delta_CP'] = cfg.dim_k7 * cfg.dim_g2 + cfg.h_star

    # 12. θ₁₃ = π/b2 (in degrees)
    preds['theta_13'] = (np.pi / cfg.b2) * (180 / np.pi)

    # 13. θ₂₃ = (rank_E8 + b3)/H* radians -> degrees
    theta_23_rad = (cfg.rank_e8 + cfg.b3) / cfg.h_star
    preds['theta_23'] = theta_23_rad * (180 / np.pi)

    # 14. θ₁₂ = arctan(sqrt(delta/gamma_GIFT))
    delta = 2 * np.pi / (cfg.weyl ** 2)
    gamma_gift = (2 * cfg.rank_e8 + 5 * cfg.h_star) / (10 * cfg.dim_g2 + 3 * cfg.dim_e8)
    if gamma_gift > 0:
        preds['theta_12'] = np.arctan(np.sqrt(delta / gamma_gift)) * (180 / np.pi)
    else:
        preds['theta_12'] = float('inf')

    # === HIGGS & COSMOLOGY (3) ===

    # 15. λ_H = √(dim_G2 + N_gen) / 2^Weyl
    n = cfg.n_gen if np.isfinite(cfg.n_gen) and 0 < cfg.n_gen < 10 else 3
    preds['lambda_H'] = np.sqrt(cfg.dim_g2 + n) / (2 ** cfg.weyl)

    # 16. Ω_DE = ln(2) × (b2+b3)/H*
    preds['Omega_DE'] = np.log(2) * (cfg.b2 + cfg.b3) / cfg.h_star

    # 17. n_s = ζ(11)/ζ(5) (fixed)
    preds['n_s'] = zeta(11) / zeta(5)

    # === FINE STRUCTURE (1) ===

    # 18. α⁻¹ = (dim_E8 + rank_E8)/2 + H*/D_bulk + det(g)*kappa_T
    base = (cfg.dim_e8 + cfg.rank_e8) / 2
    bulk_term = cfg.h_star / cfg.d_bulk
    torsion_term = cfg.det_g * cfg.kappa_t if np.isfinite(cfg.det_g * cfg.kappa_t) else 0
    preds['alpha_inv'] = base + bulk_term + torsion_term

    return preds


def compute_mean_deviation(predictions: Dict[str, float],
                           experimental: Dict[str, dict],
                           use_exp_only: bool = True) -> float:
    """
    Compute mean relative deviation from experimental values.

    Args:
        predictions: Dictionary of predicted values
        experimental: Dictionary of experimental values
        use_exp_only: If True, only use observables with experimental data
    """
    deviations = []

    for obs_name, pred_val in predictions.items():
        if obs_name not in experimental:
            continue

        exp_data = experimental[obs_name]
        if use_exp_only and not exp_data.get('has_exp', True):
            continue

        exp_val = exp_data['value']

        # Skip invalid predictions
        if not np.isfinite(pred_val) or exp_val == 0:
            deviations.append(100.0)  # Penalize invalid predictions
            continue

        rel_dev = abs(pred_val - exp_val) / abs(exp_val) * 100
        deviations.append(min(rel_dev, 100.0))  # Cap at 100%

    return np.mean(deviations) if deviations else float('inf')


def generate_alternative_configs(n_configs: int, seed: int = 42) -> List[G2Config]:
    """
    Generate alternative G2 manifold configurations.
    Varies b2 and b3 within physically plausible ranges.
    """
    np.random.seed(seed)
    configs = []

    for i in range(n_configs):
        # Sample b2 in range [1, 50]
        b2 = np.random.randint(1, 51)

        # Sample b3 in range [b2+5, 150] to ensure b3 > b2
        b3_min = b2 + 5
        b3_max = 150
        if b3_min >= b3_max:
            b3 = b3_min + np.random.randint(1, 20)
        else:
            b3 = np.random.randint(b3_min, b3_max + 1)

        configs.append(G2Config(name=f"alt_{i:05d}", b2=b2, b3=b3))

    return configs


def run_validation(n_configs: int = 10000) -> dict:
    """
    Run the complete statistical validation for GIFT v3.0.
    """
    print("=" * 60)
    print("GIFT v3.0 Statistical Validation")
    print("Testing 18 dimensionless predictions")
    print("=" * 60)
    print()

    start_time = time.time()

    # Reference configuration (GIFT E8×E8/K7)
    ref_config = G2Config(name="E8×E8/K7", b2=21, b3=77)

    # Compute reference predictions
    ref_predictions = compute_predictions(ref_config)
    ref_deviation = compute_mean_deviation(ref_predictions, EXPERIMENTAL_V3)

    print(f"Reference configuration: b2={ref_config.b2}, b3={ref_config.b3}")
    print(f"Reference mean deviation: {ref_deviation:.4f}%")
    print()

    # Generate and test alternative configurations
    print(f"Generating {n_configs} alternative configurations...")
    alt_configs = generate_alternative_configs(n_configs)

    alt_deviations = []
    alt_details = []

    for i, cfg in enumerate(alt_configs):
        if (i + 1) % 2000 == 0:
            print(f"  Progress: {i + 1}/{n_configs} ({100*(i+1)/n_configs:.0f}%)")

        predictions = compute_predictions(cfg)
        deviation = compute_mean_deviation(predictions, EXPERIMENTAL_V3)

        alt_deviations.append(deviation)
        alt_details.append({
            'name': cfg.name,
            'b2': cfg.b2,
            'b3': cfg.b3,
            'mean_deviation': deviation
        })

    alt_deviations = np.array(alt_deviations)

    # Statistical analysis
    alt_mean = np.mean(alt_deviations)
    alt_std = np.std(alt_deviations)
    alt_min = np.min(alt_deviations)
    alt_max = np.max(alt_deviations)

    # Z-score (how many sigmas away is GIFT?)
    z_score = (ref_deviation - alt_mean) / alt_std

    # P-value
    p_value = stats.norm.cdf(z_score)

    elapsed = time.time() - start_time

    # Results summary
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print()
    print(f"Reference (GIFT E8×E8/K7):")
    print(f"  b2 = {ref_config.b2}, b3 = {ref_config.b3}")
    print(f"  Mean deviation = {ref_deviation:.4f}%")
    print()
    print(f"Alternative configurations ({n_configs} tested):")
    print(f"  Mean deviation = {alt_mean:.2f}%")
    print(f"  Std deviation = {alt_std:.2f}%")
    print(f"  Min deviation = {alt_min:.2f}%")
    print(f"  Max deviation = {alt_max:.2f}%")
    print()
    print(f"Statistical significance:")
    print(f"  Z-score = {z_score:.2f}σ")
    print(f"  P-value = {p_value:.2e}")
    print(f"  Separation = {abs(z_score):.2f}σ")
    print()
    print(f"Elapsed time: {elapsed:.1f}s")

    # Build results dictionary
    results = {
        'version': '3.0',
        'date': time.strftime('%Y-%m-%d'),
        'n_observables': 18,
        'n_configs_tested': n_configs,
        'reference': {
            'name': ref_config.name,
            'b2': ref_config.b2,
            'b3': ref_config.b3,
            'mean_deviation_pct': round(ref_deviation, 4),
            'predictions': {k: round(v, 6) if np.isfinite(v) else None
                          for k, v in ref_predictions.items()}
        },
        'alternatives': {
            'mean_deviation_pct': round(alt_mean, 2),
            'std_deviation_pct': round(alt_std, 2),
            'min_deviation_pct': round(alt_min, 2),
            'max_deviation_pct': round(alt_max, 2)
        },
        'statistical_significance': {
            'z_score': round(z_score, 2),
            'sigma_separation': round(abs(z_score), 2),
            'p_value': float(f"{p_value:.2e}")
        },
        'methodology': {
            'b2_range': '[1, 50]',
            'b3_range': '[b2+5, 150]',
            'formulas': 'Actual topological formulas (not perturbations)',
            'observables': 'Dimensionless only (v3.0 catalog)'
        }
    }

    return results, alt_details


def save_results(results: dict, details: List[dict], output_dir: Path):
    """Save validation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    with open(output_dir / "validation_v3_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Save per-observable deviations for reference config
    ref_preds = results['reference']['predictions']
    obs_details = []
    for obs_name, pred_val in ref_preds.items():
        if obs_name in EXPERIMENTAL_V3 and pred_val is not None:
            exp_data = EXPERIMENTAL_V3[obs_name]
            exp_val = exp_data['value']
            dev = abs(pred_val - exp_val) / abs(exp_val) * 100 if exp_val != 0 else 0
            obs_details.append({
                'observable': obs_name,
                'predicted': pred_val,
                'experimental': exp_val,
                'uncertainty': exp_data['uncertainty'],
                'deviation_pct': round(dev, 4)
            })

    with open(output_dir / "validation_v3_observables.json", 'w') as f:
        json.dump(obs_details, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


def main():
    """Main entry point."""
    # Run validation
    results, details = run_validation(n_configs=10000)

    # Save results
    output_dir = Path(__file__).parent / "results"
    save_results(results, details, output_dir)

    # Print interpretation
    z = abs(results['statistical_significance']['z_score'])
    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    if z > 5:
        print(f"✓ HIGHLY SIGNIFICANT ({z:.1f}σ separation)")
        print("  The GIFT configuration is exceptional among tested alternatives.")
        print("  Strong evidence against overfitting within this parameter space.")
    elif z > 3:
        print(f"✓ SIGNIFICANT ({z:.1f}σ separation)")
        print("  Good evidence that GIFT is not simply overfitting to b2/b3.")
    else:
        print(f"⚠ MARGINAL ({z:.1f}σ separation)")
        print("  Results warrant further investigation.")


if __name__ == "__main__":
    main()
