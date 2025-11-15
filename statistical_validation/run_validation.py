"""
GIFT Framework v2.0 - Statistical Validation Runner

Standalone script for running comprehensive statistical validation.
Optimized for cloud/GPU execution.

Usage:
    python run_validation.py --mc-samples 1000000 --bootstrap 10000 --sobol 10000
    python run_validation.py --quick  # Fast test run
    python run_validation.py --full   # Full analysis (default)

Author: GIFT Framework Team
Date: 2025-11-13
"""

import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# === GIFT FRAMEWORK IMPLEMENTATION ===

class GIFTFrameworkStatistical:
    """GIFT Framework with uncertainty propagation capabilities."""

    def __init__(self, p2=2.0, Weyl_factor=5, tau=None):
        # Fundamental parameters
        self.p2 = p2
        self.Weyl_factor = Weyl_factor
        self.tau = tau if tau is not None else 10416 / 2673

        # Topological integers
        self.b2_K7 = 21
        self.b3_K7 = 77
        self.H_star = 99
        self.dim_E8 = 248
        self.dim_G2 = 14
        self.dim_K7 = 7
        self.dim_J3O = 27
        self.rank_E8 = 8

        # Derived parameters
        self.beta0 = np.pi / self.rank_E8
        self.xi = (self.Weyl_factor / self.p2) * self.beta0
        self.delta = 2 * np.pi / (self.Weyl_factor ** 2)
        self.gamma_GIFT = 511 / 884

        # Mathematical constants
        self.zeta2 = np.pi**2 / 6
        self.zeta3 = 1.2020569031595942
        self.zeta5 = 1.0369277551433699
        self.zeta11 = 1.0004941886041195
        self.phi = (1 + np.sqrt(5)) / 2

        # Experimental data
        self.experimental_data = {
            'alpha_inv_MZ': (127.955, 0.01),
            'sin2thetaW': (0.23122, 0.00004),
            'alpha_s_MZ': (0.1179, 0.0011),
            'theta12': (33.44, 0.77),
            'theta13': (8.61, 0.12),
            'theta23': (49.2, 1.1),
            'delta_CP': (197.0, 24.0),
            'Q_Koide': (0.6667, 0.0001),
            'm_mu_m_e': (206.768, 0.001),
            'm_tau_m_e': (3477.0, 0.1),
            'm_s_m_d': (20.0, 1.0),
            'lambda_H': (0.129, 0.002),
            'Omega_DE': (0.6847, 0.0056),
            'n_s': (0.9649, 0.0042),
            'H0': (73.04, 1.04)
        }

    def compute_all_observables(self):
        """Compute all dimensionless observables."""
        obs = {}

        # Gauge sector
        obs['alpha_inv_MZ'] = 2**(self.rank_E8 - 1) - 1/24
        obs['sin2thetaW'] = self.zeta2 - np.sqrt(2)
        obs['alpha_s_MZ'] = np.sqrt(2) / 12

        # Neutrino sector
        obs['theta12'] = np.arctan(np.sqrt(self.delta / self.gamma_GIFT)) * 180 / np.pi
        obs['theta13'] = (np.pi / self.b2_K7) * 180 / np.pi
        theta23_rad = (self.rank_E8 + self.b3_K7) / self.H_star
        obs['theta23'] = theta23_rad * 180 / np.pi
        obs['delta_CP'] = 7 * self.dim_G2 + self.H_star

        # Lepton sector
        obs['Q_Koide'] = self.dim_G2 / self.b2_K7
        obs['m_mu_m_e'] = self.dim_J3O ** self.phi
        obs['m_tau_m_e'] = self.dim_K7 + 10 * self.dim_E8 + 10 * self.H_star

        # Quark ratios
        obs['m_s_m_d'] = self.p2**2 * self.Weyl_factor

        # Higgs & Cosmology
        obs['lambda_H'] = np.sqrt(17) / 32
        obs['Omega_DE'] = np.log(2) * 98 / 99
        obs['n_s'] = self.zeta11 / self.zeta5

        # Hubble constant
        H0_Planck = 67.36
        obs['H0'] = H0_Planck * (self.zeta3 / self.xi)**self.beta0

        return obs


# === PARAMETER UNCERTAINTIES ===

PARAM_UNCERTAINTIES = {
    'p2': {'central': 2.0, 'uncertainty': 0.001},
    'Weyl_factor': {'central': 5, 'uncertainty': 0.1},
    'tau': {'central': 10416 / 2673, 'uncertainty': 0.01}
}


# === MONTE CARLO ANALYSIS ===

def monte_carlo_uncertainty_propagation(n_samples=1000000, seed=42):
    """
    Monte Carlo uncertainty propagation.

    Returns:
        distributions, statistics
    """
    np.random.seed(seed)

    print(f"\n{'='*80}")
    print("MONTE CARLO UNCERTAINTY PROPAGATION")
    print(f"{'='*80}")
    print(f"Samples: {n_samples:,}")
    print(f"Seed: {seed}\n")

    # Sample parameters
    p2_samples = np.random.normal(
        PARAM_UNCERTAINTIES['p2']['central'],
        PARAM_UNCERTAINTIES['p2']['uncertainty'],
        n_samples
    )

    Weyl_samples = np.random.normal(
        PARAM_UNCERTAINTIES['Weyl_factor']['central'],
        PARAM_UNCERTAINTIES['Weyl_factor']['uncertainty'],
        n_samples
    )

    tau_samples = np.random.normal(
        PARAM_UNCERTAINTIES['tau']['central'],
        PARAM_UNCERTAINTIES['tau']['uncertainty'],
        n_samples
    )

    # Get observable names
    gift_temp = GIFTFrameworkStatistical()
    obs_names = list(gift_temp.compute_all_observables().keys())

    # Storage
    observable_distributions = {name: np.zeros(n_samples) for name in obs_names}

    # Propagate
    batch_size = 10000
    n_batches = n_samples // batch_size

    for batch in tqdm(range(n_batches), desc="MC Propagation"):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size

        for i in range(start_idx, end_idx):
            gift = GIFTFrameworkStatistical(
                p2=p2_samples[i],
                Weyl_factor=Weyl_samples[i],
                tau=tau_samples[i]
            )

            obs = gift.compute_all_observables()

            for name, value in obs.items():
                observable_distributions[name][i] = value

    # Compute statistics
    statistics = {}
    for name, dist in observable_distributions.items():
        statistics[name] = {
            'mean': float(np.mean(dist)),
            'std': float(np.std(dist)),
            'median': float(np.median(dist)),
            'q16': float(np.percentile(dist, 16)),
            'q84': float(np.percentile(dist, 84)),
            'q025': float(np.percentile(dist, 2.5)),
            'q975': float(np.percentile(dist, 97.5)),
            'min': float(np.min(dist)),
            'max': float(np.max(dist))
        }

    print("\nMonte Carlo complete!")
    return observable_distributions, statistics


# === BOOTSTRAP VALIDATION ===

def bootstrap_experimental_validation(n_bootstrap=10000, seed=42):
    """Bootstrap validation on experimental data."""
    np.random.seed(seed)

    print(f"\n{'='*80}")
    print("BOOTSTRAP EXPERIMENTAL VALIDATION")
    print(f"{'='*80}")
    print(f"Samples: {n_bootstrap:,}\n")

    gift = GIFTFrameworkStatistical()
    predictions = gift.compute_all_observables()

    bootstrap_deviations = {name: [] for name in predictions.keys()
                           if name in gift.experimental_data}

    for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
        for obs_name, pred_value in predictions.items():
            if obs_name in gift.experimental_data:
                exp_val, exp_unc = gift.experimental_data[obs_name]
                exp_sample = np.random.normal(exp_val, exp_unc)
                dev_pct = abs(pred_value - exp_sample) / exp_sample * 100
                bootstrap_deviations[obs_name].append(dev_pct)

    # Statistics
    bootstrap_stats = {}
    for obs_name, devs in bootstrap_deviations.items():
        devs_array = np.array(devs)
        bootstrap_stats[obs_name] = {
            'mean': float(np.mean(devs_array)),
            'median': float(np.median(devs_array)),
            'std': float(np.std(devs_array)),
            'q025': float(np.percentile(devs_array, 2.5)),
            'q975': float(np.percentile(devs_array, 97.5))
        }

    print("\nBootstrap complete!")
    return bootstrap_stats


# === SOBOL ANALYSIS ===

def sobol_sensitivity_analysis(n_samples=10000, seed=42):
    """Sobol global sensitivity analysis."""
    try:
        from SALib.sample import saltelli
        from SALib.analyze import sobol
    except ImportError:
        print("\nWARNING: SALib not installed. Skipping Sobol analysis.")
        print("Install with: pip install SALib")
        return None

    np.random.seed(seed)

    print(f"\n{'='*80}")
    print("SOBOL GLOBAL SENSITIVITY ANALYSIS")
    print(f"{'='*80}")
    print(f"Base samples: {n_samples:,}")
    print(f"Total evaluations: {n_samples * 8:,}\n")

    # Define problem
    problem = {
        'num_vars': 3,
        'names': ['p2', 'Weyl_factor', 'tau'],
        'bounds': [
            [PARAM_UNCERTAINTIES['p2']['central'] - 3*PARAM_UNCERTAINTIES['p2']['uncertainty'],
             PARAM_UNCERTAINTIES['p2']['central'] + 3*PARAM_UNCERTAINTIES['p2']['uncertainty']],
            [PARAM_UNCERTAINTIES['Weyl_factor']['central'] - 3*PARAM_UNCERTAINTIES['Weyl_factor']['uncertainty'],
             PARAM_UNCERTAINTIES['Weyl_factor']['central'] + 3*PARAM_UNCERTAINTIES['Weyl_factor']['uncertainty']],
            [PARAM_UNCERTAINTIES['tau']['central'] - 3*PARAM_UNCERTAINTIES['tau']['uncertainty'],
             PARAM_UNCERTAINTIES['tau']['central'] + 3*PARAM_UNCERTAINTIES['tau']['uncertainty']]
        ]
    }

    # Generate samples
    param_values = saltelli.sample(problem, n_samples, calc_second_order=True)

    # Get observable names
    gift_temp = GIFTFrameworkStatistical()
    obs_names = list(gift_temp.compute_all_observables().keys())

    # Storage
    Y = {name: np.zeros(len(param_values)) for name in obs_names}

    # Evaluate
    for i, params in enumerate(tqdm(param_values, desc="Sobol Evaluation")):
        gift = GIFTFrameworkStatistical(
            p2=params[0],
            Weyl_factor=params[1],
            tau=params[2]
        )
        obs = gift.compute_all_observables()
        for name, value in obs.items():
            Y[name][i] = value

    # Analyze
    sobol_indices = {}
    for obs_name in obs_names:
        Si = sobol.analyze(problem, Y[obs_name], calc_second_order=True)
        sobol_indices[obs_name] = {
            'S1': [float(x) for x in Si['S1']],
            'ST': [float(x) for x in Si['ST']]
        }

    print("\nSobol analysis complete!")
    return sobol_indices


# === MAIN RUNNER ===

def main():
    parser = argparse.ArgumentParser(description='GIFT Statistical Validation')
    parser.add_argument('--mc-samples', type=int, default=1000000,
                       help='Monte Carlo samples (default: 1M)')
    parser.add_argument('--bootstrap', type=int, default=10000,
                       help='Bootstrap samples (default: 10k)')
    parser.add_argument('--sobol', type=int, default=10000,
                       help='Sobol samples (default: 10k)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test run (100k MC, 1k bootstrap, 1k Sobol)')
    parser.add_argument('--full', action='store_true',
                       help='Full analysis (default)')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    if args.quick:
        mc_samples = 100000
        bootstrap_samples = 1000
        sobol_samples = 1000
    else:
        mc_samples = args.mc_samples
        bootstrap_samples = args.bootstrap
        sobol_samples = args.sobol

    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("GIFT FRAMEWORK v2.0 - COMPREHENSIVE STATISTICAL VALIDATION")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"\nConfiguration:")
    print(f"  Monte Carlo samples: {mc_samples:,}")
    print(f"  Bootstrap samples: {bootstrap_samples:,}")
    print(f"  Sobol samples: {sobol_samples:,}")
    print(f"{'='*80}\n")

    # Run analyses
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'framework_version': '2.0',
            'mc_samples': mc_samples,
            'bootstrap_samples': bootstrap_samples,
            'sobol_samples': sobol_samples,
            'seed': args.seed
        }
    }

    # 1. Monte Carlo
    mc_distributions, mc_stats = monte_carlo_uncertainty_propagation(
        n_samples=mc_samples,
        seed=args.seed
    )
    results['monte_carlo_statistics'] = mc_stats

    # 2. Bootstrap
    bootstrap_stats = bootstrap_experimental_validation(
        n_bootstrap=bootstrap_samples,
        seed=args.seed
    )
    results['bootstrap_statistics'] = bootstrap_stats

    # 3. Sobol
    sobol_results = sobol_sensitivity_analysis(
        n_samples=sobol_samples,
        seed=args.seed
    )
    if sobol_results:
        results['sobol_indices'] = sobol_results

    # Save results
    output_path = os.path.join(args.output_dir, 'validation_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_path}")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"{'Observable':<20} {'MC Mean':>12} {'MC Std':>12} {'Bootstrap Dev%':>15}")
    print(f"{'-'*80}")

    for obs_name in mc_stats.keys():
        mc_mean = mc_stats[obs_name]['mean']
        mc_std = mc_stats[obs_name]['std']
        boot_dev = bootstrap_stats.get(obs_name, {}).get('mean', np.nan)
        print(f"{obs_name:<20} {mc_mean:>12.6f} {mc_std:>12.6f} {boot_dev:>15.4f}")

    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
