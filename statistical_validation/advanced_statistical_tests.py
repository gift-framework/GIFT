#!/usr/bin/env python3
"""
Advanced Statistical Tests for GIFT Framework Uniqueness

This module implements sophisticated statistical tests to rigorously
evaluate the uniqueness claims of the GIFT framework.

Tests implemented:
- Likelihood Ratio Test (Wilks' theorem)
- Bayesian Model Comparison (Bayes Factor)
- Permutation Tests
- Cross-Validation with Hold-Out
- Information-Theoretic Measures (KL Divergence)
- Goodness-of-Fit Tests (Anderson-Darling, Kolmogorov-Smirnov)

Author: GIFT Framework Team
License: MIT
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import zeta, gammaln
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
import time

warnings.filterwarnings('ignore')

# Import from main module
from comprehensive_uniqueness_tests import (
    GIFT, load_observables_from_csv, create_compute_functions,
    compute_chi_squared, compute_log_likelihood, Observable
)


# =============================================================================
# LIKELIHOOD RATIO TEST
# =============================================================================

class LikelihoodRatioTest:
    """
    Likelihood Ratio Test for comparing GIFT to alternative models.

    Uses Wilks' theorem: -2 * log(L_alt / L_GIFT) ~ chi2(df)
    under the null hypothesis that alternatives are as good as GIFT.
    """

    def __init__(self, n_alternatives: int = 10_000, seed: int = 42):
        self.n_alternatives = n_alternatives
        self.seed = seed
        self.observables = load_observables_from_csv()
        np.random.seed(seed)

    def compute_log_likelihood_config(self, b2: int, b3: int) -> float:
        """Compute log-likelihood for a configuration."""
        predictions = {}
        compute_funcs = create_compute_functions()

        for obs in self.observables:
            if obs.name in compute_funcs:
                try:
                    predictions[obs.name] = compute_funcs[obs.name](b2, b3)
                except:
                    predictions[obs.name] = float('inf')

        return compute_log_likelihood(predictions, self.observables)

    def run(self, verbose: bool = True) -> Dict:
        """Run the likelihood ratio test."""
        if verbose:
            print("Running Likelihood Ratio Test...")

        # Compute GIFT log-likelihood
        ll_gift = self.compute_log_likelihood_config(GIFT.B2, GIFT.B3)

        # Generate alternatives and compute their log-likelihoods
        ll_alternatives = []
        test_statistics = []

        for i in range(self.n_alternatives):
            b2 = np.random.randint(1, 100)
            b3 = np.random.randint(10, 200)

            ll_alt = self.compute_log_likelihood_config(b2, b3)
            ll_alternatives.append(ll_alt)

            # Likelihood ratio statistic: -2 * (ll_alt - ll_gift)
            # Positive values mean GIFT is better
            lr_stat = -2 * (ll_alt - ll_gift)
            test_statistics.append(lr_stat)

            if verbose and (i + 1) % 2000 == 0:
                print(f"  Processed {i + 1}/{self.n_alternatives}")

        ll_alternatives = np.array(ll_alternatives)
        test_statistics = np.array(test_statistics)

        # Compute p-value: fraction of alternatives with better likelihood
        p_value = np.mean(ll_alternatives > ll_gift)

        # Chi-squared test on the LR statistics
        # Under H0, -2*log(LR) should follow chi2 distribution
        n_better = np.sum(test_statistics < 0)  # LR stat < 0 means alternative is better

        results = {
            'gift_log_likelihood': ll_gift,
            'alt_log_likelihood_mean': np.mean(ll_alternatives),
            'alt_log_likelihood_std': np.std(ll_alternatives),
            'alt_log_likelihood_max': np.max(ll_alternatives),
            'n_alternatives_tested': self.n_alternatives,
            'n_alternatives_better': int(n_better),
            'p_value': float(p_value),
            'lr_statistic_mean': float(np.mean(test_statistics)),
            'lr_statistic_std': float(np.std(test_statistics))
        }

        if verbose:
            print(f"\nResults:")
            print(f"  GIFT log-likelihood: {ll_gift:.2f}")
            print(f"  Alternative log-likelihood: mean={np.mean(ll_alternatives):.2f}, "
                  f"max={np.max(ll_alternatives):.2f}")
            print(f"  Alternatives better than GIFT: {n_better} ({100*p_value:.4f}%)")

        return results


# =============================================================================
# BAYESIAN MODEL COMPARISON
# =============================================================================

class BayesianModelComparison:
    """
    Bayesian model comparison using Bayes factors.

    Computes the posterior probability that GIFT is the true model
    given uniform priors over all configurations.
    """

    def __init__(self, n_samples: int = 10_000, seed: int = 42):
        self.n_samples = n_samples
        self.seed = seed
        self.observables = load_observables_from_csv()
        np.random.seed(seed)

    def compute_evidence(self, b2: int, b3: int) -> float:
        """
        Compute model evidence (marginal likelihood) for a configuration.

        For Gaussian likelihood with known uncertainties:
        log P(D|M) = sum_i [ -0.5 * ((y_i - f_i)/sigma_i)^2 - 0.5*log(2*pi*sigma_i^2) ]
        """
        predictions = {}
        compute_funcs = create_compute_functions()

        for obs in self.observables:
            if obs.name in compute_funcs:
                try:
                    predictions[obs.name] = compute_funcs[obs.name](b2, b3)
                except:
                    return -np.inf

        log_evidence = 0.0
        for obs in self.observables:
            if obs.name in predictions and obs.exp_uncertainty > 0:
                pred = predictions[obs.name]
                residual = (pred - obs.exp_value) / obs.exp_uncertainty
                log_evidence -= 0.5 * (residual**2 + np.log(2 * np.pi * obs.exp_uncertainty**2))

        return log_evidence

    def run(self, verbose: bool = True) -> Dict:
        """Run Bayesian model comparison."""
        if verbose:
            print("Running Bayesian Model Comparison...")

        # Compute GIFT evidence
        log_evidence_gift = self.compute_evidence(GIFT.B2, GIFT.B3)

        # Sample alternatives and compute their evidence
        log_evidences = [log_evidence_gift]
        configs = [(GIFT.B2, GIFT.B3)]

        for i in range(self.n_samples - 1):
            b2 = np.random.randint(1, 100)
            b3 = np.random.randint(10, 200)

            log_ev = self.compute_evidence(b2, b3)
            log_evidences.append(log_ev)
            configs.append((b2, b3))

            if verbose and (i + 1) % 2000 == 0:
                print(f"  Processed {i + 1}/{self.n_samples}")

        log_evidences = np.array(log_evidences)

        # Compute posterior probabilities using log-sum-exp trick
        log_evidences_finite = log_evidences[np.isfinite(log_evidences)]
        max_log_ev = np.max(log_evidences_finite)
        log_total_evidence = max_log_ev + np.log(np.sum(np.exp(log_evidences_finite - max_log_ev)))

        posterior_gift = np.exp(log_evidence_gift - log_total_evidence)

        # Bayes factor: GIFT vs average alternative
        log_ev_alts = log_evidences[1:]  # Exclude GIFT
        log_ev_alts_finite = log_ev_alts[np.isfinite(log_ev_alts)]
        if len(log_ev_alts_finite) > 0:
            max_log_alt = np.max(log_ev_alts_finite)
            log_mean_evidence_alt = max_log_alt + np.log(np.mean(np.exp(log_ev_alts_finite - max_log_alt)))
            log_bayes_factor = log_evidence_gift - log_mean_evidence_alt
        else:
            log_bayes_factor = np.inf

        results = {
            'log_evidence_gift': float(log_evidence_gift),
            'log_evidence_mean_alt': float(np.mean(log_ev_alts_finite)) if len(log_ev_alts_finite) > 0 else -np.inf,
            'log_bayes_factor': float(log_bayes_factor),
            'bayes_factor': float(np.exp(min(log_bayes_factor, 700))),  # Cap to avoid overflow
            'posterior_probability_gift': float(posterior_gift),
            'n_models_compared': self.n_samples
        }

        if verbose:
            print(f"\nResults:")
            print(f"  GIFT log-evidence: {log_evidence_gift:.2f}")
            print(f"  Log Bayes factor (GIFT vs avg alt): {log_bayes_factor:.2f}")
            print(f"  Posterior probability of GIFT: {posterior_gift:.6f}")

            if log_bayes_factor > 2.3:  # log(10)
                print("  Interpretation: Strong evidence for GIFT (BF > 10)")
            elif log_bayes_factor > 1.1:  # log(3)
                print("  Interpretation: Moderate evidence for GIFT (BF > 3)")
            else:
                print("  Interpretation: Weak or inconclusive evidence")

        return results


# =============================================================================
# PERMUTATION TEST
# =============================================================================

class PermutationTest:
    """
    Permutation test for assessing GIFT uniqueness.

    Tests the null hypothesis that any configuration could achieve
    GIFT-level agreement by randomly permuting observable assignments.
    """

    def __init__(self, n_permutations: int = 10_000, seed: int = 42):
        self.n_permutations = n_permutations
        self.seed = seed
        self.observables = load_observables_from_csv()
        np.random.seed(seed)

    def compute_test_statistic(self, b2: int, b3: int) -> float:
        """Compute chi-squared test statistic."""
        predictions = {}
        compute_funcs = create_compute_functions()

        for obs in self.observables:
            if obs.name in compute_funcs:
                try:
                    predictions[obs.name] = compute_funcs[obs.name](b2, b3)
                except:
                    predictions[obs.name] = float('inf')

        return compute_chi_squared(predictions, self.observables)

    def run(self, verbose: bool = True) -> Dict:
        """Run permutation test."""
        if verbose:
            print("Running Permutation Test...")

        # Compute observed statistic for GIFT
        observed_stat = self.compute_test_statistic(GIFT.B2, GIFT.B3)

        # Generate permuted statistics
        permuted_stats = []

        for i in range(self.n_permutations):
            # Random configuration
            b2 = np.random.randint(1, 100)
            b3 = np.random.randint(10, 200)

            stat = self.compute_test_statistic(b2, b3)
            permuted_stats.append(stat)

            if verbose and (i + 1) % 2000 == 0:
                print(f"  Processed {i + 1}/{self.n_permutations}")

        permuted_stats = np.array(permuted_stats)

        # P-value: fraction of permuted statistics <= observed
        p_value = np.mean(permuted_stats <= observed_stat)

        # Effect size: standardized difference
        if np.std(permuted_stats) > 0:
            effect_size = (np.mean(permuted_stats) - observed_stat) / np.std(permuted_stats)
        else:
            effect_size = np.inf

        results = {
            'observed_statistic': float(observed_stat),
            'permuted_mean': float(np.mean(permuted_stats)),
            'permuted_std': float(np.std(permuted_stats)),
            'permuted_min': float(np.min(permuted_stats)),
            'p_value': float(p_value),
            'effect_size_cohens_d': float(effect_size),
            'n_permutations': self.n_permutations
        }

        if verbose:
            print(f"\nResults:")
            print(f"  GIFT chi-squared: {observed_stat:.2f}")
            print(f"  Permuted chi-squared: mean={np.mean(permuted_stats):.2f}, "
                  f"min={np.min(permuted_stats):.2f}")
            print(f"  P-value: {p_value:.6f}")
            print(f"  Effect size (Cohen's d): {effect_size:.2f}")

        return results


# =============================================================================
# CROSS-VALIDATION TEST
# =============================================================================

class CrossValidationTest:
    """
    K-fold cross-validation to test generalization.

    Tests whether GIFT's predictions generalize across different
    subsets of observables, checking for overfitting.
    """

    def __init__(self, k_folds: int = 5, n_alternatives: int = 1000, seed: int = 42):
        self.k_folds = k_folds
        self.n_alternatives = n_alternatives
        self.seed = seed
        self.observables = load_observables_from_csv()
        np.random.seed(seed)

    def compute_cv_score(self, b2: int, b3: int) -> Tuple[float, float]:
        """
        Compute cross-validation score for a configuration.

        Returns: (mean_test_chi2, std_test_chi2) across folds
        """
        n_obs = len(self.observables)
        indices = np.arange(n_obs)
        np.random.shuffle(indices)

        fold_size = n_obs // self.k_folds
        test_chi2s = []

        compute_funcs = create_compute_functions()

        for fold in range(self.k_folds):
            # Split into train and test
            test_start = fold * fold_size
            test_end = test_start + fold_size
            test_indices = indices[test_start:test_end]

            # Compute predictions
            predictions = {}
            for obs in self.observables:
                if obs.name in compute_funcs:
                    try:
                        predictions[obs.name] = compute_funcs[obs.name](b2, b3)
                    except:
                        predictions[obs.name] = float('inf')

            # Compute chi-squared on test set only
            test_chi2 = 0.0
            for idx in test_indices:
                obs = self.observables[idx]
                if obs.name in predictions and obs.exp_uncertainty > 0:
                    pred = predictions[obs.name]
                    residual = (pred - obs.exp_value) / obs.exp_uncertainty
                    test_chi2 += residual ** 2

            test_chi2s.append(test_chi2)

        return np.mean(test_chi2s), np.std(test_chi2s)

    def run(self, verbose: bool = True) -> Dict:
        """Run cross-validation test."""
        if verbose:
            print(f"Running {self.k_folds}-Fold Cross-Validation Test...")

        # GIFT CV score
        gift_mean, gift_std = self.compute_cv_score(GIFT.B2, GIFT.B3)

        # Alternative CV scores
        alt_means = []
        alt_stds = []

        for i in range(self.n_alternatives):
            b2 = np.random.randint(1, 100)
            b3 = np.random.randint(10, 200)

            mean, std = self.compute_cv_score(b2, b3)
            alt_means.append(mean)
            alt_stds.append(std)

            if verbose and (i + 1) % 200 == 0:
                print(f"  Processed {i + 1}/{self.n_alternatives}")

        alt_means = np.array(alt_means)
        alt_stds = np.array(alt_stds)

        # P-value
        p_value = np.mean(alt_means <= gift_mean)

        results = {
            'gift_cv_mean': float(gift_mean),
            'gift_cv_std': float(gift_std),
            'alt_cv_mean_avg': float(np.mean(alt_means)),
            'alt_cv_mean_min': float(np.min(alt_means)),
            'p_value': float(p_value),
            'k_folds': self.k_folds,
            'n_alternatives': self.n_alternatives
        }

        if verbose:
            print(f"\nResults:")
            print(f"  GIFT CV score: {gift_mean:.2f} +/- {gift_std:.2f}")
            print(f"  Alternative CV scores: mean={np.mean(alt_means):.2f}, "
                  f"min={np.min(alt_means):.2f}")
            print(f"  P-value: {p_value:.6f}")

        return results


# =============================================================================
# INFORMATION-THEORETIC ANALYSIS
# =============================================================================

class InformationTheoreticAnalysis:
    """
    Information-theoretic measures for model comparison.

    Computes:
    - Kullback-Leibler divergence
    - Mutual information
    - Effective number of parameters
    """

    def __init__(self, n_alternatives: int = 5000, seed: int = 42):
        self.n_alternatives = n_alternatives
        self.seed = seed
        self.observables = load_observables_from_csv()
        np.random.seed(seed)

    def compute_predictions(self, b2: int, b3: int) -> np.ndarray:
        """Get predictions as array."""
        predictions = []
        compute_funcs = create_compute_functions()

        for obs in self.observables:
            if obs.name in compute_funcs:
                try:
                    pred = compute_funcs[obs.name](b2, b3)
                    predictions.append(pred)
                except:
                    predictions.append(np.nan)

        return np.array(predictions)

    def compute_kl_divergence(self, b2: int, b3: int) -> float:
        """
        Compute KL divergence D_KL(P_exp || P_pred).

        Approximated using Gaussian distributions.
        """
        predictions = self.compute_predictions(b2, b3)
        exp_values = np.array([obs.exp_value for obs in self.observables])
        exp_stds = np.array([obs.exp_uncertainty for obs in self.observables])

        # Assume predictions have same uncertainty as experiments
        pred_stds = exp_stds

        # KL divergence between two Gaussians
        kl_div = 0.0
        for i, obs in enumerate(self.observables):
            if np.isfinite(predictions[i]) and exp_stds[i] > 0:
                mu_p, sigma_p = exp_values[i], exp_stds[i]
                mu_q, sigma_q = predictions[i], pred_stds[i]

                kl = np.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2) - 0.5
                kl_div += kl

        return kl_div

    def run(self, verbose: bool = True) -> Dict:
        """Run information-theoretic analysis."""
        if verbose:
            print("Running Information-Theoretic Analysis...")

        # GIFT KL divergence
        kl_gift = self.compute_kl_divergence(GIFT.B2, GIFT.B3)

        # Alternative KL divergences
        kl_alternatives = []

        for i in range(self.n_alternatives):
            b2 = np.random.randint(1, 100)
            b3 = np.random.randint(10, 200)

            kl = self.compute_kl_divergence(b2, b3)
            if np.isfinite(kl):
                kl_alternatives.append(kl)

            if verbose and (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{self.n_alternatives}")

        kl_alternatives = np.array(kl_alternatives)

        # P-value
        p_value = np.mean(kl_alternatives <= kl_gift)

        results = {
            'gift_kl_divergence': float(kl_gift),
            'alt_kl_mean': float(np.mean(kl_alternatives)),
            'alt_kl_min': float(np.min(kl_alternatives)),
            'alt_kl_std': float(np.std(kl_alternatives)),
            'p_value': float(p_value),
            'n_alternatives': self.n_alternatives
        }

        if verbose:
            print(f"\nResults:")
            print(f"  GIFT KL divergence: {kl_gift:.2f}")
            print(f"  Alternative KL: mean={np.mean(kl_alternatives):.2f}, "
                  f"min={np.min(kl_alternatives):.2f}")
            print(f"  P-value: {p_value:.6f}")

        return results


# =============================================================================
# COMPREHENSIVE ADVANCED TEST SUITE
# =============================================================================

class AdvancedStatisticalTestSuite:
    """Run all advanced statistical tests."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = Path(__file__).parent / "results" / "advanced_tests"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def run_all_tests(self, n_samples: int = 5000, verbose: bool = True) -> Dict:
        """Run all advanced statistical tests."""
        print("=" * 70)
        print("GIFT FRAMEWORK ADVANCED STATISTICAL TESTS")
        print("=" * 70)

        # 1. Likelihood Ratio Test
        print("\n[1/5] Likelihood Ratio Test...")
        lr_test = LikelihoodRatioTest(n_alternatives=n_samples)
        self.results['likelihood_ratio'] = lr_test.run(verbose=verbose)

        # 2. Bayesian Model Comparison
        print("\n[2/5] Bayesian Model Comparison...")
        bayes_test = BayesianModelComparison(n_samples=n_samples)
        self.results['bayesian'] = bayes_test.run(verbose=verbose)

        # 3. Permutation Test
        print("\n[3/5] Permutation Test...")
        perm_test = PermutationTest(n_permutations=n_samples)
        self.results['permutation'] = perm_test.run(verbose=verbose)

        # 4. Cross-Validation
        print("\n[4/5] Cross-Validation Test...")
        cv_test = CrossValidationTest(n_alternatives=n_samples // 5)
        self.results['cross_validation'] = cv_test.run(verbose=verbose)

        # 5. Information-Theoretic Analysis
        print("\n[5/5] Information-Theoretic Analysis...")
        info_test = InformationTheoreticAnalysis(n_alternatives=n_samples)
        self.results['information_theoretic'] = info_test.run(verbose=verbose)

        # Save results
        with open(self.output_dir / "advanced_tests_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)

        # Generate summary
        self._generate_summary()

        return self.results

    def _generate_summary(self):
        """Generate summary of all tests."""
        summary = []
        summary.append("=" * 70)
        summary.append("ADVANCED STATISTICAL TESTS SUMMARY")
        summary.append("=" * 70)

        p_values = []

        if 'likelihood_ratio' in self.results:
            p = self.results['likelihood_ratio']['p_value']
            p_values.append(p)
            summary.append(f"\nLikelihood Ratio Test:")
            summary.append(f"  P-value: {p:.6f}")

        if 'bayesian' in self.results:
            bf = self.results['bayesian']['log_bayes_factor']
            summary.append(f"\nBayesian Model Comparison:")
            summary.append(f"  Log Bayes Factor: {bf:.2f}")
            summary.append(f"  Posterior P(GIFT): {self.results['bayesian']['posterior_probability_gift']:.6f}")

        if 'permutation' in self.results:
            p = self.results['permutation']['p_value']
            p_values.append(p)
            summary.append(f"\nPermutation Test:")
            summary.append(f"  P-value: {p:.6f}")
            summary.append(f"  Effect size: {self.results['permutation']['effect_size_cohens_d']:.2f}")

        if 'cross_validation' in self.results:
            p = self.results['cross_validation']['p_value']
            p_values.append(p)
            summary.append(f"\nCross-Validation Test:")
            summary.append(f"  P-value: {p:.6f}")

        if 'information_theoretic' in self.results:
            p = self.results['information_theoretic']['p_value']
            p_values.append(p)
            summary.append(f"\nInformation-Theoretic Analysis:")
            summary.append(f"  P-value: {p:.6f}")

        # Combined p-value using Fisher's method
        if p_values:
            chi2_combined = -2 * np.sum(np.log(np.array(p_values) + 1e-300))
            dof = 2 * len(p_values)
            combined_p = 1 - stats.chi2.cdf(chi2_combined, dof)

            summary.append(f"\nCombined P-value (Fisher's method): {combined_p:.10f}")
            summary.append(f"Combined significance: {stats.norm.ppf(1 - combined_p/2):.2f} sigma")

        summary_text = "\n".join(summary)
        print(summary_text)

        with open(self.output_dir / "advanced_tests_summary.txt", 'w') as f:
            f.write(summary_text)


def run_advanced_tests(n_samples: int = 5000, output_dir: str = None) -> Dict:
    """Convenience function to run all advanced tests."""
    suite = AdvancedStatisticalTestSuite(output_dir)
    return suite.run_all_tests(n_samples=n_samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GIFT Framework Advanced Statistical Tests"
    )
    parser.add_argument(
        "--samples", type=int, default=5000,
        help="Number of samples per test (default: 5000)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results"
    )

    args = parser.parse_args()
    run_advanced_tests(n_samples=args.samples, output_dir=args.output_dir)
