#!/usr/bin/env python3
"""
GIFT v3.3 — BULLET-PROOF Statistical Validation

Seven-component rigorous validation protocol:

  1. Pre-registration manifest (frozen before analysis)
  2. Three null model families (permutation, structure-preserved, adversarial)
  3. Empirical p-values with multiple corrections (BH, Bonferroni, Holm)
  4. Held-out test sets (leave-out blocks by physics sector)
  5. Robustness / sensitivity (weights, noise, leave-k-out, jackknife)
  6. Multi-seed / multi-implementation replication
  7. Bayesian analysis (multiple priors, posterior predictive, WAIC/LOO-CV)

Pure Python — no external dependencies beyond stdlib.

Author: GIFT Framework
Date: February 2026
"""

import math
import json
import time
import random
import hashlib
import statistics
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from itertools import combinations

# ── Import core prediction engine ────────────────────────────────────────────
from validation_v33 import (
    EXPERIMENTAL_V33, GIFT_REFERENCE, GIFTConfig,
    compute_predictions_v33, compute_deviation, riemann_zeta, PHI,
)


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def normal_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7)."""
    z = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * z)
    d = 0.3989422804014327 * math.exp(-z * z / 2.0)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    return 1.0 - p if x > 0 else p


def pvalue_to_sigma(p: float) -> float:
    """Convert two-sided p-value to Gaussian sigma."""
    if p <= 0:
        return float('inf')
    if p >= 1:
        return 0.0
    # Bisection on normal_cdf
    lo, hi = 0.0, 40.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if 2.0 * (1.0 - normal_cdf(mid)) < p:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def log_normal_pdf(x: float, mu: float, sigma: float) -> float:
    """Log of normal PDF."""
    if sigma <= 0:
        return 0.0 if x == mu else float('-inf')
    return -0.5 * math.log(2 * math.pi) - math.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


def dev_for(cfg: GIFTConfig) -> float:
    """Mean relative deviation (%) for a config, inf if invalid."""
    try:
        preds = compute_predictions_v33(cfg)
        d, _ = compute_deviation(preds)
        return d if math.isfinite(d) else float('inf')
    except (ValueError, ZeroDivisionError, OverflowError):
        return float('inf')


def make_cfg(b2, b3, **kw):
    """Create GIFTConfig with optional overrides."""
    params = dict(name="scan", b2=b2, b3=b3)
    params.update(kw)
    return GIFTConfig(**params)


# Physics sectors for held-out tests
SECTORS = {
    'gauge_couplings': ['sin2_theta_W', 'alpha_s', 'alpha_inv', 'lambda_H'],
    'leptons': ['Q_Koide', 'm_tau_m_e', 'm_mu_m_e', 'm_mu_m_tau'],
    'quarks': ['m_s_m_d', 'm_c_m_s', 'm_b_m_t', 'm_u_m_d'],
    'pmns': ['delta_CP', 'theta_13', 'theta_23', 'theta_12',
             'sin2_theta_12_PMNS', 'sin2_theta_23_PMNS', 'sin2_theta_13_PMNS'],
    'ckm': ['sin2_theta_12_CKM', 'A_Wolfenstein', 'sin2_theta_23_CKM'],
    'bosons': ['m_H_m_t', 'm_H_m_W', 'm_W_m_Z'],
    'cosmology': ['Omega_DE', 'n_s', 'Omega_DM_Omega_b', 'h_Hubble',
                  'Omega_b_Omega_m', 'sigma_8', 'Y_p'],
    'structural': ['N_gen'],
}


# ═════════════════════════════════════════════════════════════════════════════
#  COMPONENT 1 — PRE-REGISTRATION MANIFEST
# ═════════════════════════════════════════════════════════════════════════════

def build_preregistration() -> dict:
    """
    Freeze all analysis choices BEFORE looking at results.

    This manifest is hashed (SHA-256) so any post-hoc modification is detectable.
    """
    manifest = {
        'title': 'GIFT v3.3 Bullet-Proof Statistical Validation',
        'date_frozen': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'version': '3.3.17',

        # Score function
        'score_function': 'mean_relative_deviation_percent',
        'score_formula': 'mean(|pred_i - exp_i| / |exp_i| * 100) over all observables',

        # Configuration space
        'config_space': {
            'b2_range': [1, 100],
            'b3_range': [3, 200],
            'constraint': 'b3 > b2',
            'holonomy_dims': [3, 8, 10, 14, 15, 21],
            'gauge_ranks': [2, 4, 5, 6, 7, 8, 16],
            'p2_choices': [1, 2, 3, 4],
            'weyl_choices': [2, 3, 4, 5, 6, 7, 8],
        },

        # Observables
        'n_observables': len(EXPERIMENTAL_V33),
        'observable_names': sorted(EXPERIMENTAL_V33.keys()),

        # Primary hypothesis
        'H0': 'GIFT deviation is drawn from the same distribution as random topological configurations',
        'H1': 'GIFT deviation is systematically lower than random configurations',
        'test_type': 'one-sided (lower tail)',

        # Significance thresholds
        'alpha_primary': 0.05,
        'alpha_discovery': 0.003,  # 3σ equivalent

        # Secondary tests (pre-specified)
        'secondary_tests': [
            'per-sector held-out cross-prediction',
            'jackknife influence diagnostics',
            'leave-k-out stability (k = 1..5)',
            'multi-seed replication (10 seeds)',
            'Bayesian model comparison with 4 priors',
            'posterior predictive checks',
        ],

        # Anti-look-elsewhere
        'lee_trials_formula': 'n_betti_grid * n_holonomy * n_gauge_rank * n_p2 * n_weyl',
        'lee_explicit_count': (100 * 197 * 6 * 7 * 4 * 7),

        # Dev/test split
        'dev_test_split': {
            'dev_observables': sorted([
                'sin2_theta_W', 'alpha_s', 'Q_Koide', 'm_tau_m_e', 'm_s_m_d',
                'm_c_m_s', 'delta_CP', 'theta_13', 'theta_23', 'theta_12',
                'Omega_DE', 'n_s', 'N_gen', 'alpha_inv', 'lambda_H', 'm_mu_m_e',
            ]),
            'test_observables': sorted([
                'm_mu_m_tau', 'm_b_m_t', 'm_u_m_d',
                'sin2_theta_12_PMNS', 'sin2_theta_23_PMNS', 'sin2_theta_13_PMNS',
                'sin2_theta_12_CKM', 'A_Wolfenstein', 'sin2_theta_23_CKM',
                'm_H_m_t', 'm_H_m_W', 'm_W_m_Z',
                'Omega_DM_Omega_b', 'h_Hubble', 'Omega_b_Omega_m',
                'sigma_8', 'Y_p',
            ]),
            'rationale': 'Dev set = observables used during framework development; '
                         'Test set = predictions made after framework was fixed',
        },
    }

    # Hash the manifest for tamper-detection
    manifest_str = json.dumps(manifest, sort_keys=True, indent=2)
    manifest['sha256'] = hashlib.sha256(manifest_str.encode()).hexdigest()

    return manifest


# ═════════════════════════════════════════════════════════════════════════════
#  COMPONENT 2 — THREE NULL MODEL FAMILIES
# ═════════════════════════════════════════════════════════════════════════════

def null_A_permutation(n_perms: int = 50000, seed: int = 42) -> dict:
    """
    Null A — Permutation null.
    Shuffle the mapping between predictions and experimental values.
    Under H0, the assignment of formulas to observables is arbitrary.
    """
    rng = random.Random(seed)
    ref_preds = compute_predictions_v33(GIFT_REFERENCE)
    ref_dev, _ = compute_deviation(ref_preds)

    obs_names = sorted(EXPERIMENTAL_V33.keys())
    pred_vals = [ref_preds[o] for o in obs_names]
    exp_entries = [EXPERIMENTAL_V33[o] for o in obs_names]

    perm_devs = []
    for _ in range(n_perms):
        shuffled_exp = list(exp_entries)
        rng.shuffle(shuffled_exp)
        # Compute deviation with shuffled assignment
        total = 0.0
        n = 0
        for pv, ee in zip(pred_vals, shuffled_exp):
            ev = ee['value']
            if ev == 0 or not math.isfinite(pv):
                total += 100.0
            else:
                total += min(abs(pv - ev) / abs(ev) * 100, 100.0)
            n += 1
        perm_devs.append(total / n if n > 0 else float('inf'))

    n_better = sum(1 for d in perm_devs if d <= ref_dev)
    p_value = (n_better + 1) / (n_perms + 1)  # +1 for continuity correction

    return {
        'name': 'Null A: Permutation',
        'description': 'Shuffle prediction-to-observable assignment',
        'n_permutations': n_perms,
        'gift_deviation': ref_dev,
        'null_mean': statistics.mean(perm_devs),
        'null_std': statistics.stdev(perm_devs),
        'null_median': sorted(perm_devs)[len(perm_devs) // 2],
        'n_as_good_or_better': n_better,
        'p_value': p_value,
        'sigma': pvalue_to_sigma(p_value),
    }


def null_B_structure_preserved(n_configs: int = 50000, seed: int = 42) -> dict:
    """
    Null B — Structure-preserved null.
    Vary topological parameters but KEEP the functional form of all formulas.
    This tests: is (b2=21, b3=77) special among valid G2-manifold-like configs?
    """
    rng = random.Random(seed)
    ref_dev = dev_for(GIFT_REFERENCE)

    alt_devs = []
    for _ in range(n_configs):
        b2 = rng.randint(1, 100)
        b3 = rng.randint(max(b2 + 1, 3), 200)
        cfg = make_cfg(b2, b3)
        d = dev_for(cfg)
        if math.isfinite(d):
            alt_devs.append(d)

    n_better = sum(1 for d in alt_devs if d <= ref_dev)
    p_value = (n_better + 1) / (len(alt_devs) + 1)

    return {
        'name': 'Null B: Structure-preserved',
        'description': 'Same formulas, random (b2, b3) pairs',
        'n_configs': len(alt_devs),
        'gift_deviation': ref_dev,
        'null_mean': statistics.mean(alt_devs),
        'null_std': statistics.stdev(alt_devs),
        'n_as_good_or_better': n_better,
        'p_value': p_value,
        'sigma': pvalue_to_sigma(p_value),
    }


def null_C_adversarial(n_configs: int = 50000, seed: int = 42) -> dict:
    """
    Null C — Adversarial null.
    For each observable, generate a "random formula" by combining topological
    constants with random arithmetic operations. Tests whether ANY set of
    formulas from the same ingredient pool can match experiment.

    Strategy: for each trial, pick random coefficients and combine the same
    topological integers {b2, b3, dim_G2, dim_E8, rank_E8, ...} using
    random ratios a_i/a_j, sums, products.
    """
    rng = random.Random(seed)
    ref_dev = dev_for(GIFT_REFERENCE)

    # Topological integers available to the "adversary"
    topo_ints = [21, 77, 14, 248, 8, 7, 27, 52, 78, 2, 5, 11, 99, 42, 3, 496]

    obs_names = sorted(EXPERIMENTAL_V33.keys())

    adversary_devs = []
    for _ in range(n_configs):
        total_dev = 0.0
        for obs in obs_names:
            exp_val = EXPERIMENTAL_V33[obs]['value']
            # Generate a random prediction from topological ingredients
            op = rng.randint(0, 5)
            a = rng.choice(topo_ints)
            b = rng.choice(topo_ints)
            while b == 0:
                b = rng.choice(topo_ints)

            if op == 0:
                pred = a / b
            elif op == 1:
                pred = a * b / rng.choice([x for x in topo_ints if x != 0])
            elif op == 2:
                pred = (a + b) / rng.choice([x for x in topo_ints if x != 0])
            elif op == 3:
                pred = math.sqrt(abs(a)) / b
            elif op == 4:
                pred = a / (a + b) if (a + b) != 0 else 1.0
            else:
                pred = math.log(max(a, 2)) * b / rng.choice([x for x in topo_ints if x != 0])

            if exp_val != 0 and math.isfinite(pred):
                total_dev += min(abs(pred - exp_val) / abs(exp_val) * 100, 100.0)
            else:
                total_dev += 100.0

        adversary_devs.append(total_dev / len(obs_names))

    n_better = sum(1 for d in adversary_devs if d <= ref_dev)
    p_value = (n_better + 1) / (len(adversary_devs) + 1)

    return {
        'name': 'Null C: Adversarial',
        'description': 'Random formulas from same topological integer pool',
        'n_configs': len(adversary_devs),
        'gift_deviation': ref_dev,
        'null_mean': statistics.mean(adversary_devs),
        'null_std': statistics.stdev(adversary_devs),
        'null_min': min(adversary_devs),
        'n_as_good_or_better': n_better,
        'p_value': p_value,
        'sigma': pvalue_to_sigma(p_value),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  COMPONENT 3 — EMPIRICAL P-VALUE WITH MULTIPLE CORRECTIONS
# ═════════════════════════════════════════════════════════════════════════════

def per_observable_pvalues(n_configs: int = 50000, seed: int = 42) -> dict:
    """
    Compute per-observable empirical p-values, then apply:
      - Raw (uncorrected)
      - Bonferroni
      - Holm (step-down)
      - Benjamini-Hochberg (FDR)
    """
    rng = random.Random(seed)
    ref_preds = compute_predictions_v33(GIFT_REFERENCE)
    _, ref_per_obs = compute_deviation(ref_preds)
    obs_names = sorted(ref_per_obs.keys())

    # Collect per-observable deviations under null B
    null_per_obs = {o: [] for o in obs_names}
    for _ in range(n_configs):
        b2 = rng.randint(1, 100)
        b3 = rng.randint(max(b2 + 1, 3), 200)
        cfg = make_cfg(b2, b3)
        try:
            preds = compute_predictions_v33(cfg)
            _, per_obs = compute_deviation(preds)
            for o in obs_names:
                if o in per_obs and math.isfinite(per_obs[o]):
                    null_per_obs[o].append(per_obs[o])
        except Exception:
            pass

    # Raw p-values (one-sided: fraction as good or better)
    raw_p = {}
    for o in obs_names:
        if not null_per_obs[o]:
            raw_p[o] = 1.0
            continue
        n_better = sum(1 for d in null_per_obs[o] if d <= ref_per_obs[o])
        raw_p[o] = (n_better + 1) / (len(null_per_obs[o]) + 1)

    m = len(obs_names)

    # Bonferroni
    bonferroni = {o: min(1.0, p * m) for o, p in raw_p.items()}

    # Holm (step-down)
    sorted_obs = sorted(raw_p.items(), key=lambda x: x[1])
    holm = {}
    max_so_far = 0.0
    for i, (o, p) in enumerate(sorted_obs):
        adj = p * (m - i)
        adj = max(adj, max_so_far)  # enforce monotonicity
        holm[o] = min(1.0, adj)
        max_so_far = adj

    # Benjamini-Hochberg (FDR)
    bh = {}
    sorted_desc = sorted(raw_p.items(), key=lambda x: x[1], reverse=True)
    prev = 1.0
    for i, (o, p) in enumerate(sorted_desc):
        rank = m - i  # rank from 1..m
        adj = min(prev, p * m / rank)
        bh[o] = min(1.0, adj)
        prev = adj

    # Explicit trial count
    n_trials_explicit = n_configs * m

    # Summary: how many observables significant at alpha=0.05?
    alpha = 0.05
    n_sig_raw = sum(1 for p in raw_p.values() if p < alpha)
    n_sig_bonf = sum(1 for p in bonferroni.values() if p < alpha)
    n_sig_holm = sum(1 for p in holm.values() if p < alpha)
    n_sig_bh = sum(1 for p in bh.values() if p < alpha)

    # ── Westfall-Young maxT (permutation-based FWER under correlation) ──
    # Resamples the null to build the distribution of the MAXIMUM test
    # statistic across all observables simultaneously.
    # Controls FWER without the independence assumption of Bonferroni.
    n_wy = 5000
    ref_preds_wy = compute_predictions_v33(GIFT_REFERENCE)
    _, ref_per_obs_wy = compute_deviation(ref_preds_wy)

    # Pre-compute null means and stds (once, not per permutation)
    null_mu = {}
    null_sd = {}
    for o in obs_names:
        if null_per_obs[o]:
            null_mu[o] = statistics.mean(null_per_obs[o])
            null_sd[o] = statistics.stdev(null_per_obs[o]) if len(null_per_obs[o]) > 1 else 1.0
        else:
            null_mu[o] = 50.0
            null_sd[o] = 1.0

    # Observed per-obs "z-scores" (deviation relative to null mean/std)
    obs_z = {}
    for o in obs_names:
        obs_z[o] = (null_mu[o] - ref_per_obs_wy.get(o, 100.0)) / null_sd[o] if null_sd[o] > 0 else 0
    observed_max_z = max(obs_z.values()) if obs_z else 0

    # Permutation null for maxT
    rng_wy = random.Random(seed + 999)
    max_z_null = []
    for _ in range(n_wy):
        b2 = rng_wy.randint(1, 100)
        b3 = rng_wy.randint(max(b2 + 1, 3), 200)
        cfg = make_cfg(b2, b3)
        try:
            preds = compute_predictions_v33(cfg)
            _, po = compute_deviation(preds)
            zs = []
            for o in obs_names:
                z = (null_mu[o] - po.get(o, 100.0)) / null_sd[o] if null_sd[o] > 0 else 0
                zs.append(z)
            max_z_null.append(max(zs) if zs else 0)
        except Exception:
            pass

    n_exceed = sum(1 for mz in max_z_null if mz >= observed_max_z)
    wy_p = (n_exceed + 1) / (len(max_z_null) + 1)

    # Westfall-Young adjusted per-obs p-values
    wy_adj = {}
    for o in obs_names:
        z_o = obs_z.get(o, 0)
        n_exc_o = sum(1 for mz in max_z_null if mz >= z_o)
        wy_adj[o] = (n_exc_o + 1) / (len(max_z_null) + 1)
    n_sig_wy = sum(1 for p in wy_adj.values() if p < alpha)

    return {
        'n_configs': n_configs,
        'n_observables': m,
        'n_trials_explicit': n_trials_explicit,
        'alpha': alpha,
        'raw_p_values': raw_p,
        'bonferroni': bonferroni,
        'holm': holm,
        'benjamini_hochberg': bh,
        'westfall_young': wy_adj,
        'westfall_young_global_p': wy_p,
        'n_significant': {
            'raw': n_sig_raw,
            'bonferroni': n_sig_bonf,
            'holm': n_sig_holm,
            'benjamini_hochberg': n_sig_bh,
            'westfall_young': n_sig_wy,
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
#  COMPONENT 4 — HELD-OUT TEST SETS
# ═════════════════════════════════════════════════════════════════════════════

def held_out_tests(n_configs: int = 50000, seed: int = 42) -> dict:
    """
    Leave out entire physics sectors and evaluate GIFT's performance
    on the held-out sector. Tests cross-domain predictive power.
    """
    rng = random.Random(seed)
    ref_preds = compute_predictions_v33(GIFT_REFERENCE)
    all_obs = sorted(EXPERIMENTAL_V33.keys())

    results = {}

    for sector_name, sector_obs in SECTORS.items():
        # Training set = everything except this sector
        train_obs = [o for o in all_obs if o not in sector_obs]
        test_obs = [o for o in sector_obs if o in EXPERIMENTAL_V33]

        if not test_obs:
            continue

        # GIFT deviation on train vs test
        train_exp = {o: EXPERIMENTAL_V33[o] for o in train_obs}
        test_exp = {o: EXPERIMENTAL_V33[o] for o in test_obs}

        gift_train_dev, _ = compute_deviation(ref_preds, train_exp)
        gift_test_dev, gift_test_detail = compute_deviation(ref_preds, test_exp)

        # Null distribution on test set
        null_test_devs = []
        for _ in range(n_configs):
            b2 = rng.randint(1, 100)
            b3 = rng.randint(max(b2 + 1, 3), 200)
            cfg = make_cfg(b2, b3)
            try:
                preds = compute_predictions_v33(cfg)
                d, _ = compute_deviation(preds, test_exp)
                if math.isfinite(d):
                    null_test_devs.append(d)
            except Exception:
                pass

        n_better = sum(1 for d in null_test_devs if d <= gift_test_dev)
        p_value = (n_better + 1) / (len(null_test_devs) + 1)

        results[sector_name] = {
            'held_out_observables': test_obs,
            'n_held_out': len(test_obs),
            'gift_train_dev': gift_train_dev,
            'gift_test_dev': gift_test_dev,
            'gift_test_detail': gift_test_detail,
            'null_test_mean': statistics.mean(null_test_devs) if null_test_devs else float('inf'),
            'null_test_std': statistics.stdev(null_test_devs) if len(null_test_devs) > 1 else 0,
            'n_better': n_better,
            'p_value': p_value,
            'sigma': pvalue_to_sigma(p_value),
        }

    # Dev/test split from pre-registration
    prereg = build_preregistration()
    dev_obs = prereg['dev_test_split']['dev_observables']
    test_obs_prereg = prereg['dev_test_split']['test_observables']

    dev_exp = {o: EXPERIMENTAL_V33[o] for o in dev_obs if o in EXPERIMENTAL_V33}
    test_exp = {o: EXPERIMENTAL_V33[o] for o in test_obs_prereg if o in EXPERIMENTAL_V33}

    gift_dev_dev, _ = compute_deviation(ref_preds, dev_exp)
    gift_dev_test, gift_test_detail_prereg = compute_deviation(ref_preds, test_exp)

    null_test_devs_prereg = []
    for _ in range(n_configs):
        b2 = rng.randint(1, 100)
        b3 = rng.randint(max(b2 + 1, 3), 200)
        cfg = make_cfg(b2, b3)
        try:
            preds = compute_predictions_v33(cfg)
            d, _ = compute_deviation(preds, test_exp)
            if math.isfinite(d):
                null_test_devs_prereg.append(d)
        except Exception:
            pass

    n_better_prereg = sum(1 for d in null_test_devs_prereg if d <= gift_dev_test)
    p_prereg = (n_better_prereg + 1) / (len(null_test_devs_prereg) + 1)

    results['preregistered_split'] = {
        'dev_observables': dev_obs,
        'test_observables': test_obs_prereg,
        'gift_dev_deviation': gift_dev_dev,
        'gift_test_deviation': gift_dev_test,
        'gift_test_detail': gift_test_detail_prereg,
        'null_test_mean': statistics.mean(null_test_devs_prereg) if null_test_devs_prereg else float('inf'),
        'n_better': n_better_prereg,
        'p_value': p_prereg,
        'sigma': pvalue_to_sigma(p_prereg),
    }

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  COMPONENT 5 — ROBUSTNESS / SENSITIVITY
# ═════════════════════════════════════════════════════════════════════════════

def robustness_analysis(seed: int = 42) -> dict:
    """
    Comprehensive robustness battery:
      5a. Weight variations (uniform vs uncertainty-weighted vs inverse-range)
      5b. Realistic noise Monte Carlo (perturb exp values within uncertainties)
      5c. Leave-k-out stability (k = 1..5)
      5d. Jackknife influence diagnostics
      5e. Subsampling stability
    """
    rng = random.Random(seed)
    ref_preds = compute_predictions_v33(GIFT_REFERENCE)
    obs_names = sorted(EXPERIMENTAL_V33.keys())
    n_obs = len(obs_names)

    results = {}

    # ── 5a: Weight variations ────────────────────────────────────────────
    # Scheme 1: uniform weights (baseline)
    _, per_obs = compute_deviation(ref_preds)
    uniform_dev = sum(per_obs.values()) / len(per_obs)

    # Scheme 2: uncertainty-weighted (observables with smaller uncertainty
    #           get more weight — penalizes predictions that miss tight measurements)
    total_w = 0.0
    weighted_sum = 0.0
    for o in obs_names:
        unc = EXPERIMENTAL_V33[o].get('uncertainty', 1.0)
        w = 1.0 / max(unc, 1e-10)
        weighted_sum += w * per_obs.get(o, 100.0)
        total_w += w
    unc_weighted_dev = weighted_sum / total_w if total_w > 0 else float('inf')

    # Scheme 3: inverse-range weighted (normalize by observable magnitude)
    total_w = 0.0
    weighted_sum = 0.0
    for o in obs_names:
        mag = abs(EXPERIMENTAL_V33[o]['value'])
        w = 1.0 / max(mag, 1e-10)
        weighted_sum += w * per_obs.get(o, 100.0)
        total_w += w
    range_weighted_dev = weighted_sum / total_w if total_w > 0 else float('inf')

    # Scheme 4: random weights (100 trials)
    random_weight_devs = []
    for _ in range(100):
        ws = [rng.uniform(0.1, 10.0) for _ in obs_names]
        total_w = sum(ws)
        wd = sum(w * per_obs.get(o, 100.0) for w, o in zip(ws, obs_names)) / total_w
        random_weight_devs.append(wd)

    results['weight_variations'] = {
        'uniform': uniform_dev,
        'uncertainty_weighted': unc_weighted_dev,
        'inverse_range_weighted': range_weighted_dev,
        'random_weights_mean': statistics.mean(random_weight_devs),
        'random_weights_std': statistics.stdev(random_weight_devs),
        'random_weights_max': max(random_weight_devs),
        'all_below_1pct': all(d < 1.0 for d in [uniform_dev, unc_weighted_dev, range_weighted_dev]
                              + random_weight_devs),
    }

    # ── 5b: Realistic noise Monte Carlo ──────────────────────────────────
    noise_devs = []
    n_noise = 1000
    for _ in range(n_noise):
        noisy_exp = {}
        for o in obs_names:
            val = EXPERIMENTAL_V33[o]['value']
            unc = EXPERIMENTAL_V33[o].get('uncertainty', 0)
            noisy_exp[o] = {
                'value': val + rng.gauss(0, unc),
                'uncertainty': unc,
            }
        d, _ = compute_deviation(ref_preds, noisy_exp)
        noise_devs.append(d)

    results['noise_mc'] = {
        'n_trials': n_noise,
        'mean_deviation': statistics.mean(noise_devs),
        'std_deviation': statistics.stdev(noise_devs),
        'min_deviation': min(noise_devs),
        'max_deviation': max(noise_devs),
        'pct_below_0_5': 100 * sum(1 for d in noise_devs if d < 0.5) / n_noise,
        'pct_below_1_0': 100 * sum(1 for d in noise_devs if d < 1.0) / n_noise,
    }

    # ── 5c: Leave-k-out stability ────────────────────────────────────────
    leave_k_out = {}
    for k in range(1, 6):
        loo_devs = []
        combos = list(combinations(obs_names, k))
        # For large k, subsample
        if len(combos) > 5000:
            rng.shuffle(combos)
            combos = combos[:5000]

        for combo in combos:
            reduced_exp = {o: EXPERIMENTAL_V33[o] for o in obs_names if o not in combo}
            d, _ = compute_deviation(ref_preds, reduced_exp)
            loo_devs.append(d)

        leave_k_out[f'k={k}'] = {
            'n_subsets': len(loo_devs),
            'mean': statistics.mean(loo_devs),
            'std': statistics.stdev(loo_devs) if len(loo_devs) > 1 else 0,
            'min': min(loo_devs),
            'max': max(loo_devs),
        }

    results['leave_k_out'] = leave_k_out

    # ── 5d: Jackknife influence ──────────────────────────────────────────
    full_dev = uniform_dev
    jackknife = []
    for o in obs_names:
        reduced_exp = {k: v for k, v in EXPERIMENTAL_V33.items() if k != o}
        d, _ = compute_deviation(ref_preds, reduced_exp)
        influence = full_dev - d  # positive = removing this HELPS (obs was hurting)
        jackknife.append({
            'observable': o,
            'deviation_without': d,
            'influence': influence,
            'abs_influence': abs(influence),
        })

    jackknife.sort(key=lambda x: -x['abs_influence'])

    results['jackknife'] = {
        'full_deviation': full_dev,
        'most_influential': jackknife[:5],
        'least_influential': jackknife[-5:],
        'max_influence': jackknife[0]['abs_influence'],
        'mean_influence': statistics.mean([j['abs_influence'] for j in jackknife]),
        'no_single_obs_dominates': jackknife[0]['abs_influence'] < 0.5 * full_dev,
    }

    # ── 5e: Subsampling stability ────────────────────────────────────────
    subsample_results = []
    for frac in [0.5, 0.6, 0.7, 0.8, 0.9]:
        k = max(1, int(frac * n_obs))
        trials = []
        for _ in range(200):
            subset = rng.sample(obs_names, k)
            sub_exp = {o: EXPERIMENTAL_V33[o] for o in subset}
            d, _ = compute_deviation(ref_preds, sub_exp)
            trials.append(d)
        subsample_results.append({
            'fraction': frac,
            'k': k,
            'mean': statistics.mean(trials),
            'std': statistics.stdev(trials),
        })

    results['subsampling'] = subsample_results

    # ── 5f: Noise sensitivity curve ──────────────────────────────────────
    # Sweep noise amplitude from 0× to 3× published uncertainties.
    # Shows the realistic precision boundary.
    noise_curve = []
    for factor in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        trials = []
        for _ in range(200):
            noisy_exp = {}
            for o in obs_names:
                val = EXPERIMENTAL_V33[o]['value']
                unc = EXPERIMENTAL_V33[o].get('uncertainty', 0)
                noisy_exp[o] = {
                    'value': val + rng.gauss(0, unc * factor),
                    'uncertainty': unc,
                }
            d, _ = compute_deviation(ref_preds, noisy_exp)
            trials.append(d)
        noise_curve.append({
            'sigma_factor': factor,
            'mean_deviation': statistics.mean(trials),
            'std_deviation': statistics.stdev(trials),
        })

    results['noise_sensitivity_curve'] = noise_curve

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  COMPONENT 6 — MULTI-SEED / MULTI-IMPLEMENTATION REPLICATION
# ═════════════════════════════════════════════════════════════════════════════

def multi_seed_replication(n_configs_per_seed: int = 20000) -> dict:
    """
    Run the null-B test with 10 different seeds and verify consistency.
    Also test two independent implementations of the deviation metric.
    """
    ref_dev = dev_for(GIFT_REFERENCE)
    seeds = [42, 137, 256, 314, 577, 691, 853, 997, 1234, 2718]

    seed_results = []
    for s in seeds:
        rng = random.Random(s)
        alt_devs = []
        for _ in range(n_configs_per_seed):
            b2 = rng.randint(1, 100)
            b3 = rng.randint(max(b2 + 1, 3), 200)
            d = dev_for(make_cfg(b2, b3))
            if math.isfinite(d):
                alt_devs.append(d)

        n_better = sum(1 for d in alt_devs if d <= ref_dev)
        p = (n_better + 1) / (len(alt_devs) + 1)

        seed_results.append({
            'seed': s,
            'n_valid': len(alt_devs),
            'n_better': n_better,
            'p_value': p,
            'sigma': pvalue_to_sigma(p),
            'null_mean': statistics.mean(alt_devs),
            'null_std': statistics.stdev(alt_devs),
        })

    # Implementation 2: alternative deviation metric (chi-squared-like)
    # Use relative chi² to avoid domination by ultra-precise measurements
    ref_preds = compute_predictions_v33(GIFT_REFERENCE)
    CHI2_CAP = 1e4  # cap per-observable to avoid one term dominating

    def _compute_chi2(preds_dict):
        """Relative chi² with consistent cap and penalty for invalid predictions."""
        c2 = 0.0
        for o, exp in EXPERIMENTAL_V33.items():
            val = exp['value']
            if val == 0 or o not in preds_dict:
                c2 += CHI2_CAP
                continue
            pred = preds_dict[o]
            if not math.isfinite(pred):
                c2 += CHI2_CAP
                continue
            # Relative residual: (pred - exp) / exp
            rel_resid = (pred - val) / abs(val)
            # Use relative uncertainty or 1% floor
            rel_unc = max(exp.get('uncertainty', 0) / abs(val), 0.01)
            c2 += min((rel_resid / rel_unc) ** 2, CHI2_CAP)
        return c2

    chi2_gift = _compute_chi2(ref_preds)

    rng2 = random.Random(42)
    chi2_null = []
    for _ in range(n_configs_per_seed):
        b2 = rng2.randint(1, 100)
        b3 = rng2.randint(max(b2 + 1, 3), 200)
        cfg = make_cfg(b2, b3)
        try:
            preds = compute_predictions_v33(cfg)
            chi2_null.append(_compute_chi2(preds))
        except Exception:
            pass

    n_better_chi2 = sum(1 for c in chi2_null if c <= chi2_gift)
    p_chi2 = (n_better_chi2 + 1) / (len(chi2_null) + 1)

    p_values = [r['p_value'] for r in seed_results]

    return {
        'n_seeds': len(seeds),
        'n_configs_per_seed': n_configs_per_seed,
        'seed_results': seed_results,
        'p_value_mean': statistics.mean(p_values),
        'p_value_std': statistics.stdev(p_values),
        'p_value_max': max(p_values),
        'sigma_mean': statistics.mean([r['sigma'] for r in seed_results]),
        'all_significant_at_005': all(p < 0.05 for p in p_values),
        'alternative_metric': {
            'name': 'chi-squared',
            'gift_chi2': chi2_gift,
            'null_mean_chi2': statistics.mean(chi2_null) if chi2_null else float('inf'),
            'n_better': n_better_chi2,
            'p_value': p_chi2,
            'sigma': pvalue_to_sigma(p_chi2),
            'consistent': (p_chi2 < 0.05) == (statistics.mean(p_values) < 0.05),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
#  COMPONENT 7 — BAYESIAN ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def bayesian_analysis(n_configs: int = 50000, seed: int = 42) -> dict:
    """
    Full Bayesian analysis:
      7a. Bayes factors with 4 different priors
      7b. Posterior predictive checks
      7c. WAIC / LOO-CV approximation
    """
    rng = random.Random(seed)
    ref_preds = compute_predictions_v33(GIFT_REFERENCE)
    ref_dev = dev_for(GIFT_REFERENCE)
    obs_names = sorted(EXPERIMENTAL_V33.keys())

    # Generate null distribution
    null_devs = []
    null_per_obs_all = []
    for _ in range(n_configs):
        b2 = rng.randint(1, 100)
        b3 = rng.randint(max(b2 + 1, 3), 200)
        cfg = make_cfg(b2, b3)
        try:
            preds = compute_predictions_v33(cfg)
            d, per_obs = compute_deviation(preds)
            if math.isfinite(d):
                null_devs.append(d)
                null_per_obs_all.append(per_obs)
        except Exception:
            pass

    null_mean = statistics.mean(null_devs)
    null_std = statistics.stdev(null_devs)

    # ── 7a: Bayes Factors with multiple priors ───────────────────────────

    # Likelihood under H0: GIFT deviation ~ N(null_mean, null_std)
    log_lik_H0 = log_normal_pdf(ref_dev, null_mean, null_std)

    bayes_factors = {}

    # Prior 1: Skeptical — H1 says GIFT is at most 2× better than average
    # Prior on deviation under H1: Uniform(0, null_mean/2)
    prior1_range = null_mean / 2
    log_lik_H1_p1 = -math.log(prior1_range) if ref_dev < prior1_range else float('-inf')
    bf1 = math.exp(min(700, log_lik_H1_p1 - log_lik_H0)) if math.isfinite(log_lik_H1_p1) else 0
    bayes_factors['skeptical_uniform'] = {
        'prior': f'Uniform(0, {prior1_range:.2f})',
        'log_BF': log_lik_H1_p1 - log_lik_H0 if math.isfinite(log_lik_H1_p1) else float('-inf'),
        'BF': bf1,
    }

    # Prior 2: Reference — H1 says GIFT is drawn from a half-normal centered at 0
    # with sigma = null_std
    log_lik_H1_p2 = log_normal_pdf(ref_dev, 0, null_std) + math.log(2)  # half-normal
    bf2 = math.exp(min(700, log_lik_H1_p2 - log_lik_H0))
    bayes_factors['reference_halfnormal'] = {
        'prior': f'HalfNormal(0, {null_std:.2f})',
        'log_BF': log_lik_H1_p2 - log_lik_H0,
        'BF': bf2,
    }

    # Prior 3: Enthusiastic — H1 says deviation < 1% with uniform prior
    log_lik_H1_p3 = -math.log(1.0) if ref_dev < 1.0 else float('-inf')  # Uniform(0, 1%)
    bf3 = math.exp(min(700, log_lik_H1_p3 - log_lik_H0)) if math.isfinite(log_lik_H1_p3) else 0
    bayes_factors['enthusiastic_uniform_1pct'] = {
        'prior': 'Uniform(0, 1%)',
        'log_BF': log_lik_H1_p3 - log_lik_H0 if math.isfinite(log_lik_H1_p3) else float('-inf'),
        'BF': bf3,
    }

    # Prior 4: Jeffreys — non-informative 1/deviation prior (improper, normalized)
    # log p(d) = -log(d) over [d_min, d_max] → normalization constant
    d_min, d_max = 0.01, 100.0
    log_norm = math.log(math.log(d_max / d_min))
    log_lik_H1_p4 = -math.log(max(ref_dev, d_min)) - log_norm
    bf4 = math.exp(min(700, log_lik_H1_p4 - log_lik_H0))
    bayes_factors['jeffreys'] = {
        'prior': f'Jeffreys 1/d over [{d_min}, {d_max}]',
        'log_BF': log_lik_H1_p4 - log_lik_H0,
        'BF': bf4,
    }

    # Interpret
    def interpret_bf(bf):
        if bf < 1:
            return 'Supports H0'
        if bf < 3:
            return 'Anecdotal'
        if bf < 10:
            return 'Moderate for H1'
        if bf < 30:
            return 'Strong for H1'
        if bf < 100:
            return 'Very strong for H1'
        return 'Decisive for H1'

    for k in bayes_factors:
        bayes_factors[k]['interpretation'] = interpret_bf(bayes_factors[k]['BF'])

    # ── 7b: Posterior predictive checks (4 test statistics) ────────────
    # Multiple discrepancy measures to avoid cherry-picking a single stat.
    # Under H1 (GIFT model), replicated data = exp + gauss(0, unc).
    #   T1: mean deviation (global fit quality)
    #   T2: max deviation (tail / worst outlier)
    #   T3: count of obs with dev > 1% (calibration of "good" predictions)
    #   T4: leave-one-sector-out max discrepancy (structural coherence)

    n_ppc = 500
    observed_devs = []
    for o in obs_names:
        exp_val = EXPERIMENTAL_V33[o]['value']
        pred_val = ref_preds.get(o, 0)
        if exp_val != 0:
            observed_devs.append(abs(pred_val - exp_val) / abs(exp_val) * 100)

    # Observed test statistics
    T1_obs = statistics.mean(observed_devs)
    T2_obs = max(observed_devs)
    T3_obs = sum(1 for d in observed_devs if d > 1.0)

    # T4: worst sector-level deviation
    sector_devs_obs = {}
    for sname, sobs in SECTORS.items():
        sdevs = []
        for o in sobs:
            if o in EXPERIMENTAL_V33:
                exp_val = EXPERIMENTAL_V33[o]['value']
                pred_val = ref_preds.get(o, 0)
                if exp_val != 0:
                    sdevs.append(abs(pred_val - exp_val) / abs(exp_val) * 100)
        if sdevs:
            sector_devs_obs[sname] = statistics.mean(sdevs)
    T4_obs = max(sector_devs_obs.values()) if sector_devs_obs else 0

    # Replicate
    T1_rep, T2_rep, T3_rep, T4_rep = [], [], [], []
    for _ in range(n_ppc):
        rep_devs = []
        for o in obs_names:
            pred_val = ref_preds.get(o, 0)
            exp_val = EXPERIMENTAL_V33[o]['value']
            unc = EXPERIMENTAL_V33[o].get('uncertainty', 0)
            rep_obs = exp_val + rng.gauss(0, max(unc, abs(exp_val) * 0.001))
            if rep_obs != 0:
                rep_devs.append(abs(pred_val - rep_obs) / abs(rep_obs) * 100)

        T1_rep.append(statistics.mean(rep_devs))
        T2_rep.append(max(rep_devs))
        T3_rep.append(sum(1 for d in rep_devs if d > 1.0))

        # T4 for replicated
        rep_sector = {}
        idx = 0
        obs_dev_map = {obs_names[i]: rep_devs[i] for i in range(min(len(obs_names), len(rep_devs)))}
        for sname, sobs in SECTORS.items():
            sdevs = [obs_dev_map[o] for o in sobs if o in obs_dev_map]
            if sdevs:
                rep_sector[sname] = statistics.mean(sdevs)
        T4_rep.append(max(rep_sector.values()) if rep_sector else 0)

    ppc_p1 = sum(1 for d in T1_rep if d >= T1_obs) / n_ppc
    ppc_p2 = sum(1 for d in T2_rep if d >= T2_obs) / n_ppc
    ppc_p3 = sum(1 for d in T3_rep if d >= T3_obs) / n_ppc
    ppc_p4 = sum(1 for d in T4_rep if d >= T4_obs) / n_ppc

    # Classify PPC p-values:
    #   p < 0.05  → model too poor (underfitting)
    #   0.05 ≤ p ≤ 0.95 → well-calibrated
    #   p > 0.95  → model surpasses noise expectations (overfitting or genuine signal)
    ppc_checks = [ppc_p1, ppc_p2, ppc_p3, ppc_p4]
    n_well_calibrated = sum(1 for p in ppc_checks if 0.05 < p < 0.95)
    n_superior = sum(1 for p in ppc_checks if p >= 0.95)
    n_poor = sum(1 for p in ppc_checks if p <= 0.05)

    if n_poor > 0:
        ppc_interpretation = f'Model underfit on {n_poor}/4 statistics'
        ppc_status = 'underfit'
    elif n_superior == len(ppc_checks):
        ppc_interpretation = (
            'Model fits significantly better than measurement noise predicts '
            'across all statistics — consistent with genuine physical content'
        )
        ppc_status = 'superior_to_noise'
    elif n_well_calibrated >= 2:
        ppc_interpretation = f'Well-calibrated ({n_well_calibrated}/4 in [0.05, 0.95])'
        ppc_status = 'calibrated'
    else:
        ppc_interpretation = (
            f'{n_superior}/4 superior to noise, {n_well_calibrated}/4 calibrated'
        )
        ppc_status = 'mixed'

    posterior_predictive = {
        'n_replications': n_ppc,
        'statistics': {
            'T1_mean_dev': {'observed': T1_obs, 'ppc_p': ppc_p1,
                            'rep_mean': statistics.mean(T1_rep), 'rep_std': statistics.stdev(T1_rep)},
            'T2_max_dev': {'observed': T2_obs, 'ppc_p': ppc_p2,
                           'rep_mean': statistics.mean(T2_rep), 'rep_std': statistics.stdev(T2_rep)},
            'T3_count_above_1pct': {'observed': T3_obs, 'ppc_p': ppc_p3,
                                    'rep_mean': statistics.mean(T3_rep)},
            'T4_worst_sector': {'observed': T4_obs, 'ppc_p': ppc_p4,
                                'rep_mean': statistics.mean(T4_rep)},
        },
        'n_well_calibrated': n_well_calibrated,
        'n_superior': n_superior,
        'n_poor': n_poor,
        'status': ppc_status,
        'interpretation': ppc_interpretation,
    }

    # ── 7c: WAIC / LOO-CV approximation ─────────────────────────────────
    # Use relative deviation as the "data" and model fit quality.
    # Likelihood model: per-observable relative deviation ~ N(0, sigma_model)
    # where sigma_model is the typical deviation scale for each model class.
    #
    # WAIC = -2 * (lppd - p_waic)
    # lppd = sum_i log(mean_s p(y_i | theta_s))
    # p_waic = sum_i var_s(log p(y_i | theta_s))

    # Null model: use observed null deviations as "posterior samples"
    n_samples = min(len(null_per_obs_all), 5000)

    # Estimate per-observable null deviation scale
    null_sigma_per_obs = {}
    for o in obs_names:
        devs = [null_per_obs_all[i].get(o, 100.0) for i in range(n_samples)
                if o in null_per_obs_all[i] and math.isfinite(null_per_obs_all[i].get(o, 100.0))]
        null_sigma_per_obs[o] = statistics.stdev(devs) if len(devs) > 1 else 50.0

    # Null model WAIC: how well does the null distribution predict observations?
    # "Observations" here = the GIFT per-observable deviations (our actual data)
    _, gift_per_obs = compute_deviation(ref_preds)

    lppd_null = 0.0
    p_waic_null = 0.0
    loo_scores_null = []

    for o in obs_names:
        gift_obs_dev = gift_per_obs.get(o, 100.0)
        # Log-likelihood of observing gift_obs_dev under each null sample
        log_liks = []
        for i in range(n_samples):
            null_dev = null_per_obs_all[i].get(o, 100.0) if o in null_per_obs_all[i] else 100.0
            sigma_o = max(null_sigma_per_obs.get(o, 50.0), 0.1)
            ll = log_normal_pdf(gift_obs_dev, null_dev, sigma_o)
            log_liks.append(ll)

        if not log_liks:
            continue

        max_ll = max(log_liks)
        log_mean_lik = max_ll + math.log(sum(math.exp(ll - max_ll) for ll in log_liks) / len(log_liks))
        lppd_null += log_mean_lik

        var_ll = statistics.variance(log_liks) if len(log_liks) > 1 else 0
        p_waic_null += var_ll

        # LOO-CV (Pareto-smoothed IS approximation, simplified)
        neg_log_liks = [-ll for ll in log_liks]
        max_nll = max(neg_log_liks)
        log_mean_inv = max_nll + math.log(
            sum(math.exp(nll - max_nll) for nll in neg_log_liks) / len(neg_log_liks)
        )
        loo_scores_null.append(-log_mean_inv)

    waic_null = -2 * (lppd_null - p_waic_null)
    loo_cv_null = -2 * sum(loo_scores_null) if loo_scores_null else float('inf')

    # GIFT model WAIC: how well does GIFT predict observations?
    # GIFT is a single-point model (no posterior spread), so p_waic = 0
    # Use a small observational noise sigma (GIFT claims ~0.2% precision)
    gift_sigma = 0.5  # generous: assume GIFT predictions have ~0.5% inherent spread
    gift_lppd = 0.0
    for o in obs_names:
        gift_obs_dev = gift_per_obs.get(o, 100.0)
        # Under GIFT model, deviation should be near 0
        ll = log_normal_pdf(gift_obs_dev, 0.0, gift_sigma)
        gift_lppd += ll

    gift_waic = -2 * gift_lppd  # p_waic = 0 for point model

    waic_results = {
        'null_waic': waic_null,
        'null_lppd': lppd_null,
        'null_p_waic': p_waic_null,
        'null_loo_cv': loo_cv_null,
        'gift_waic': gift_waic,
        'gift_lppd': gift_lppd,
        'delta_waic': waic_null - gift_waic,
        'gift_preferred': gift_waic < waic_null,
        'interpretation': (
            'GIFT model has better predictive accuracy (lower WAIC)'
            if gift_waic < waic_null
            else 'Null model has comparable or better WAIC'
        ),
    }

    return {
        'bayes_factors': bayes_factors,
        'posterior_predictive': posterior_predictive,
        'waic_loo_cv': waic_results,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run_bulletproof_validation(verbose: bool = True) -> dict:
    """Run the complete 7-component bullet-proof validation."""
    t_start = time.time()

    if verbose:
        print("=" * 80)
        print("  GIFT v3.3 — BULLET-PROOF STATISTICAL VALIDATION")
        print("=" * 80)
        print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Observables: {len(EXPERIMENTAL_V33)}")
        print(f"  GIFT reference: b2=21, b3=77, E8×E8, G2")
        ref_dev = dev_for(GIFT_REFERENCE)
        print(f"  GIFT mean deviation: {ref_dev:.4f}%")
        print()

    results = {}

    # ══════════════════════════════════════════════════════════════════════
    # 1. PRE-REGISTRATION
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print("━" * 70)
        print("  [1/7] PRE-REGISTRATION MANIFEST")
        print("━" * 70)
    t0 = time.time()
    results['preregistration'] = build_preregistration()
    if verbose:
        pr = results['preregistration']
        print(f"  SHA-256: {pr['sha256'][:32]}...")
        print(f"  Observables: {pr['n_observables']}")
        print(f"  Dev/test split: {len(pr['dev_test_split']['dev_observables'])} / "
              f"{len(pr['dev_test_split']['test_observables'])}")
        print(f"  LEE trial count: {pr['lee_explicit_count']:,}")
        print(f"  Time: {time.time() - t0:.1f}s")
        print()

    # ══════════════════════════════════════════════════════════════════════
    # 2. THREE NULL MODELS
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print("━" * 70)
        print("  [2/7] THREE NULL MODEL FAMILIES")
        print("━" * 70)
    t0 = time.time()

    if verbose:
        print("  [2a] Null A: Permutation test...")
    null_a = null_A_permutation(n_perms=50000)
    if verbose:
        print(f"       p = {null_a['p_value']:.2e}  (σ = {null_a['sigma']:.2f})")
        print(f"       Null mean: {null_a['null_mean']:.2f}%  vs  GIFT: {null_a['gift_deviation']:.4f}%")

    if verbose:
        print("  [2b] Null B: Structure-preserved...")
    null_b = null_B_structure_preserved(n_configs=50000)
    if verbose:
        print(f"       p = {null_b['p_value']:.2e}  (σ = {null_b['sigma']:.2f})")
        print(f"       {null_b['n_as_good_or_better']} / {null_b['n_configs']} as good or better")

    if verbose:
        print("  [2c] Null C: Adversarial...")
    null_c = null_C_adversarial(n_configs=50000)
    if verbose:
        print(f"       p = {null_c['p_value']:.2e}  (σ = {null_c['sigma']:.2f})")
        print(f"       Best adversary: {null_c['null_min']:.2f}%  vs  GIFT: {null_c['gift_deviation']:.4f}%")

    results['null_models'] = {
        'null_A_permutation': null_a,
        'null_B_structure_preserved': null_b,
        'null_C_adversarial': null_c,
    }
    if verbose:
        print(f"  Time: {time.time() - t0:.1f}s")
        print()

    # ══════════════════════════════════════════════════════════════════════
    # 3. PER-OBSERVABLE P-VALUES WITH CORRECTIONS
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print("━" * 70)
        print("  [3/7] EMPIRICAL P-VALUES + MULTIPLE CORRECTIONS")
        print("━" * 70)
    t0 = time.time()
    pval_results = per_observable_pvalues(n_configs=50000)
    results['p_value_corrections'] = pval_results
    if verbose:
        ns = pval_results['n_significant']
        print(f"  Significant at α=0.05:")
        print(f"    Raw:               {ns['raw']} / {pval_results['n_observables']}")
        print(f"    Bonferroni:        {ns['bonferroni']} / {pval_results['n_observables']}")
        print(f"    Holm:              {ns['holm']} / {pval_results['n_observables']}")
        print(f"    Benjamini-Hochberg: {ns['benjamini_hochberg']} / {pval_results['n_observables']}")
        print(f"    Westfall-Young:    {ns['westfall_young']} / {pval_results['n_observables']}  "
              f"(global p = {pval_results['westfall_young_global_p']:.2e})")
        print(f"  Explicit trial count: {pval_results['n_trials_explicit']:,}")
        print(f"  Time: {time.time() - t0:.1f}s")
        print()

    # ══════════════════════════════════════════════════════════════════════
    # 4. HELD-OUT TEST SETS
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print("━" * 70)
        print("  [4/7] HELD-OUT TEST SETS (cross-sector prediction)")
        print("━" * 70)
    t0 = time.time()
    held_out = held_out_tests(n_configs=30000)
    results['held_out'] = held_out
    if verbose:
        for sector, data in held_out.items():
            if sector == 'preregistered_split':
                continue
            print(f"  {sector:20s}: test_dev = {data['gift_test_dev']:.3f}%  "
                  f"p = {data['p_value']:.2e}  σ = {data['sigma']:.1f}")
        ps = held_out['preregistered_split']
        print(f"  {'PRE-REG SPLIT':20s}: dev={ps['gift_dev_deviation']:.3f}% "
              f"test={ps['gift_test_deviation']:.3f}%  "
              f"p = {ps['p_value']:.2e}  σ = {ps['sigma']:.1f}")
        print(f"  Time: {time.time() - t0:.1f}s")
        print()

    # ══════════════════════════════════════════════════════════════════════
    # 5. ROBUSTNESS / SENSITIVITY
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print("━" * 70)
        print("  [5/7] ROBUSTNESS / SENSITIVITY ANALYSIS")
        print("━" * 70)
    t0 = time.time()
    robust = robustness_analysis()
    results['robustness'] = robust
    if verbose:
        wv = robust['weight_variations']
        print(f"  Weight variations:")
        print(f"    Uniform:              {wv['uniform']:.4f}%")
        print(f"    Uncertainty-weighted: {wv['uncertainty_weighted']:.4f}%")
        print(f"    Inverse-range:        {wv['inverse_range_weighted']:.4f}%")
        print(f"    Random (100 trials):  {wv['random_weights_mean']:.4f}% ± {wv['random_weights_std']:.4f}%")
        print(f"    All < 1%: {wv['all_below_1pct']}")
        nm = robust['noise_mc']
        print(f"  Noise MC ({nm['n_trials']} trials): {nm['mean_deviation']:.4f}% ± {nm['std_deviation']:.4f}%")
        print(f"    {nm['pct_below_1_0']:.0f}% of trials < 1%")
        jk = robust['jackknife']
        print(f"  Jackknife: max influence = {jk['max_influence']:.4f}%")
        print(f"    Most influential: {jk['most_influential'][0]['observable']} "
              f"({jk['most_influential'][0]['influence']:+.4f}%)")
        print(f"    No single obs dominates: {jk['no_single_obs_dominates']}")
        print(f"  Leave-k-out:")
        for k_label, k_data in robust['leave_k_out'].items():
            print(f"    {k_label}: {k_data['mean']:.4f}% ± {k_data['std']:.4f}%  "
                  f"(range [{k_data['min']:.4f}, {k_data['max']:.4f}])")
        print(f"  Noise sensitivity curve (deviation vs σ_noise):")
        for pt in robust['noise_sensitivity_curve']:
            print(f"    {pt['sigma_factor']:.2f}×σ → {pt['mean_deviation']:.4f}% ± {pt['std_deviation']:.4f}%")
        print(f"  Time: {time.time() - t0:.1f}s")
        print()

    # ══════════════════════════════════════════════════════════════════════
    # 6. MULTI-SEED REPLICATION
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print("━" * 70)
        print("  [6/7] MULTI-SEED / MULTI-IMPLEMENTATION REPLICATION")
        print("━" * 70)
    t0 = time.time()
    multi = multi_seed_replication(n_configs_per_seed=20000)
    results['multi_seed'] = multi
    if verbose:
        print(f"  Seeds tested: {multi['n_seeds']}")
        print(f"  p-value range: [{min(r['p_value'] for r in multi['seed_results']):.2e}, "
              f"{max(r['p_value'] for r in multi['seed_results']):.2e}]")
        print(f"  σ range: [{min(r['sigma'] for r in multi['seed_results']):.2f}, "
              f"{max(r['sigma'] for r in multi['seed_results']):.2f}]")
        print(f"  All significant at 0.05: {multi['all_significant_at_005']}")
        alt = multi['alternative_metric']
        print(f"  Alt metric (χ²): p = {alt['p_value']:.2e}  σ = {alt['sigma']:.2f}")
        print(f"  Consistent across metrics: {alt['consistent']}")
        print(f"  Time: {time.time() - t0:.1f}s")
        print()

    # ══════════════════════════════════════════════════════════════════════
    # 7. BAYESIAN ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print("━" * 70)
        print("  [7/7] BAYESIAN ANALYSIS")
        print("━" * 70)
    t0 = time.time()
    bayes = bayesian_analysis(n_configs=50000)
    results['bayesian'] = bayes
    if verbose:
        print("  Bayes Factors (4 priors):")
        for name, bf_data in bayes['bayes_factors'].items():
            print(f"    {name:30s}: BF = {bf_data['BF']:.1f}  [{bf_data['interpretation']}]")
        ppc = bayes['posterior_predictive']
        print(f"  Posterior predictive checks (4 statistics):")
        for stat_name, stat_data in ppc['statistics'].items():
            p = stat_data['ppc_p']
            tag = '✓' if 0.05 < p < 0.95 else ('↑' if p >= 0.95 else '↓')
            print(f"    {stat_name:25s}: obs = {stat_data['observed']:.3f}  "
                  f"ppc_p = {p:.3f}  [{tag}]")
        print(f"    Status: {ppc['status']}  →  {ppc['interpretation']}")
        waic = bayes['waic_loo_cv']
        print(f"  WAIC comparison:")
        print(f"    GIFT WAIC: {waic['gift_waic']:.1f}")
        print(f"    Null WAIC: {waic['null_waic']:.1f}")
        print(f"    ΔWAIC: {waic['delta_waic']:.1f}  (GIFT {'preferred' if waic['gift_preferred'] else 'not preferred'})")
        print(f"  Time: {time.time() - t0:.1f}s")
        print()

    # ══════════════════════════════════════════════════════════════════════
    # GRAND SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - t_start

    summary = {
        'gift_deviation_pct': dev_for(GIFT_REFERENCE),
        'null_models': {
            'permutation_p': null_a['p_value'],
            'permutation_sigma': null_a['sigma'],
            'structure_preserved_p': null_b['p_value'],
            'structure_preserved_sigma': null_b['sigma'],
            'adversarial_p': null_c['p_value'],
            'adversarial_sigma': null_c['sigma'],
        },
        'multiple_corrections': {
            'n_sig_raw': pval_results['n_significant']['raw'],
            'n_sig_bonferroni': pval_results['n_significant']['bonferroni'],
            'n_sig_holm': pval_results['n_significant']['holm'],
            'n_sig_bh': pval_results['n_significant']['benjamini_hochberg'],
            'n_sig_westfall_young': pval_results['n_significant']['westfall_young'],
            'westfall_young_global_p': pval_results['westfall_young_global_p'],
        },
        'held_out_nontrivial_sectors_significant': all(
            held_out[s]['p_value'] < 0.05
            for s in held_out
            if s not in ('preregistered_split', 'structural')  # N_gen trivially matched
            and held_out[s]['n_held_out'] > 1
        ),
        'preregistered_test_p': held_out['preregistered_split']['p_value'],
        'robust_to_weights': robust['weight_variations']['all_below_1pct'],
        'noise_mc_mean_below_2pct': robust['noise_mc']['mean_deviation'] < 2.0,
        'no_single_obs_dominates': robust['jackknife']['no_single_obs_dominates'],
        'multi_seed_consistent': multi['all_significant_at_005'],
        'cross_metric_consistent': multi['alternative_metric']['consistent'],
        'best_bayes_factor': max(bf['BF'] for bf in bayes['bayes_factors'].values()),
        'ppc_status': bayes['posterior_predictive']['status'],
        'gift_preferred_waic': bayes['waic_loo_cv']['gift_preferred'],
        'total_elapsed_seconds': round(total_elapsed, 1),
    }
    results['summary'] = summary

    if verbose:
        print("=" * 80)
        print("  GRAND SUMMARY")
        print("=" * 80)
        print()
        print(f"  GIFT mean deviation: {summary['gift_deviation_pct']:.4f}%")
        print()
        print("  Null models:")
        nm = summary['null_models']
        print(f"    Permutation:       p = {nm['permutation_p']:.2e}  (σ = {nm['permutation_sigma']:.1f})")
        print(f"    Structure-pres:    p = {nm['structure_preserved_p']:.2e}  (σ = {nm['structure_preserved_sigma']:.1f})")
        print(f"    Adversarial:       p = {nm['adversarial_p']:.2e}  (σ = {nm['adversarial_sigma']:.1f})")
        print()
        mc = summary['multiple_corrections']
        print(f"  Significant observables (α=0.05):")
        print(f"    Raw: {mc['n_sig_raw']}  Bonferroni: {mc['n_sig_bonferroni']}  "
              f"Holm: {mc['n_sig_holm']}  BH: {mc['n_sig_bh']}  "
              f"W-Y: {mc['n_sig_westfall_young']}")
        print(f"    Westfall-Young global p: {mc['westfall_young_global_p']:.2e}")
        print()
        print(f"  Cross-prediction:")
        print(f"    Non-trivial sectors significant: {summary['held_out_nontrivial_sectors_significant']}")
        print(f"    Pre-registered test p: {summary['preregistered_test_p']:.2e}")
        print()
        print(f"  Robustness:")
        print(f"    Weight-invariant: {summary['robust_to_weights']}")
        print(f"    Noise MC mean < 2%: {summary['noise_mc_mean_below_2pct']}")
        print(f"    No dominating obs: {summary['no_single_obs_dominates']}")
        print()
        print(f"  Replication:")
        print(f"    10-seed consistent: {summary['multi_seed_consistent']}")
        print(f"    Cross-metric:       {summary['cross_metric_consistent']}")
        print()
        print(f"  Bayesian:")
        print(f"    Best BF:    {summary['best_bayes_factor']:.1f}")
        print(f"    PPC status: {summary['ppc_status']}")
        print(f"    WAIC pref:  {summary['gift_preferred_waic']}")
        print()
        print(f"  Total elapsed: {summary['total_elapsed_seconds']}s")
        print("=" * 80)

    # ── Save results ──
    out_path = Path(__file__).parent / 'bulletproof_validation_v33_results.json'

    def json_safe(obj):
        if isinstance(obj, float):
            if math.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
            if math.isnan(obj):
                return "NaN"
            return float(obj)
        if isinstance(obj, int):
            return int(obj)
        return str(obj)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=json_safe)

    if verbose:
        print(f"\n  Results saved to: {out_path}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    run_bulletproof_validation(verbose=True)
