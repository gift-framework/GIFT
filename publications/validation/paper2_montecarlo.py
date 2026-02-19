#!/usr/bin/env python3
"""
Statistical Validation Suite for the Explicit G₂ Metric
========================================================

Implements five independent tests to validate the numerical claims
of the PINN-reconstructed metric on K₇:

    g_mean: 7×7 SPD matrix with det(g) = 65/32, κ = 1.01518,
    7 eigenvalues matched to 8 significant figures.

Tests:
    1. PERMUTATION TEST: Shuffle metric entries, measure det deviation null dist.
    2. MONTE CARLO RATIONAL: Random SPD matrices near identity, probability of
       det landing near any "nice" rational p/q.
    3. SOBOL SENSITIVITY: Which Cholesky components control det, κ, torsion?
    4. BOOTSTRAP STABILITY: Noise at measurement precision, propagate to
       det, κ, eigenvalues.
    5. LOOK-ELSEWHERE: Among all p/q with small denominators, how special
       is 65/32?

Data: The 7×7 metric tensor from PINN v3 (Table 5.1 of Paper 2).

Author: GIFT Framework
Date: 2026-02-08
"""

import numpy as np
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────
N_PERM = 10_000          # permutation test trials
N_MC = 100_000           # Monte Carlo random matrices
N_BOOTSTRAP = 5_000      # bootstrap resamples
N_SOBOL = 8_192          # 2^13 Sobol points

DET_TARGET = 65 / 32     # = 2.03125
KAPPA_TARGET = 1.01518
DET_TOLERANCE = 4e-8     # fractional deviation (4 × 10⁻⁸ %)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Data: the explicit 7×7 metric from Paper 2 ───────────────────────
G_MEAN = np.array([
    [1.11332, +0.00098, -0.00072, -0.00019, +0.00341, +0.00285, -0.00305],
    [+0.00098, 1.11055, -0.00081, +0.00123, -0.00419, +0.00018, -0.00325],
    [-0.00072, -0.00081, 1.10908, +0.00461, +0.00085, +0.00269, +0.00069],
    [-0.00019, +0.00123, +0.00461, 1.10430, -0.00069, +0.00010, -0.00135],
    [+0.00341, -0.00419, +0.00085, -0.00069, 1.10263, +0.00154, -0.00001],
    [+0.00285, +0.00018, +0.00269, +0.00010, +0.00154, 1.10385, -0.00066],
    [-0.00305, -0.00325, +0.00069, -0.00135, -0.00001, -0.00066, 1.10217],
])

# Target eigenvalues (from PINN v3)
EIGENVALUES_TARGET = np.array([
    1.09926643, 1.10004584, 1.10124313, 1.10334338,
    1.11246355, 1.11358841, 1.11595127,
])

# Measurement precision (from comparison table, max ~3 × 10⁻⁷)
MEASUREMENT_NOISE = 3e-7


# ── Utility functions ─────────────────────────────────────────────────
def det_deviation(g, target=DET_TARGET):
    """Fractional deviation of det(g) from target, in percent."""
    return abs(np.linalg.det(g) - target) / target * 100


def condition_number(g):
    """Condition number κ = λ_max / λ_min."""
    eigs = np.linalg.eigvalsh(g)
    return eigs[-1] / eigs[0]


def cholesky_perturb(L0, delta_L):
    """Reconstruct metric from perturbed Cholesky factor."""
    L = L0 + delta_L
    return L @ L.T


def random_spd_near_identity(rng, diag_range=(1.09, 1.12), off_scale=0.005):
    """Generate a random 7×7 SPD matrix with structure similar to G_MEAN."""
    diag = rng.uniform(diag_range[0], diag_range[1], 7)
    off = rng.uniform(-off_scale, off_scale, (7, 7))
    off = (off + off.T) / 2  # symmetrize
    np.fill_diagonal(off, 0)
    g = np.diag(diag) + off
    # Ensure SPD by projecting to nearest SPD
    eigs, vecs = np.linalg.eigh(g)
    eigs = np.maximum(eigs, 0.01)
    return vecs @ np.diag(eigs) @ vecs.T


# ── Test 1: Permutation Test on Off-Diagonal Structure ───────────────
def test_permutation(n_perm=N_PERM):
    """
    Permutation test: shuffle off-diagonal entries of G_MEAN,
    recompute det deviation. Tests whether the specific arrangement
    of off-diagonal entries matters for achieving det = 65/32.
    """
    print("\n" + "=" * 70)
    print("TEST 1: PERMUTATION TEST (off-diagonal structure)")
    print("=" * 70)

    det_real = np.linalg.det(G_MEAN)
    dev_real = det_deviation(G_MEAN)
    print(f"  Original det(g) = {det_real:.10f}")
    print(f"  Original deviation = {dev_real:.2e} %")

    # Extract off-diagonal entries (upper triangle, 21 values)
    triu_idx = np.triu_indices(7, k=1)
    off_diag = G_MEAN[triu_idx].copy()
    diag = np.diag(G_MEAN).copy()

    rng = np.random.default_rng(42)
    dev_null = np.zeros(n_perm)

    t0 = time.time()
    for i in range(n_perm):
        perm = rng.permutation(len(off_diag))
        g_perm = np.diag(diag).copy()
        shuffled = off_diag[perm]
        g_perm[triu_idx] = shuffled
        g_perm[(triu_idx[1], triu_idx[0])] = shuffled  # symmetrize
        dev_null[i] = det_deviation(g_perm)

        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{n_perm} permutations done...")

    elapsed = time.time() - t0

    # How many random arrangements achieve equal or better det?
    p_value = float(np.mean(dev_null <= dev_real))
    z_score = (np.mean(dev_null) - dev_real) / np.std(dev_null) if np.std(dev_null) > 0 else float('inf')

    result = {
        "det_original": float(det_real),
        "det_deviation_pct": float(dev_real),
        "dev_null_mean": float(np.mean(dev_null)),
        "dev_null_std": float(np.std(dev_null)),
        "dev_null_min": float(np.min(dev_null)),
        "dev_null_max": float(np.max(dev_null)),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "n_permutations": n_perm,
        "runtime_s": float(elapsed),
    }

    print(f"  Null distribution: mean dev = {result['dev_null_mean']:.4e} %, "
          f"std = {result['dev_null_std']:.4e} %")
    print(f"  Null min deviation: {result['dev_null_min']:.4e} %")
    print(f"  Z-score = {z_score:.1f}")
    print(f"  p-value = {p_value:.4e}")
    print(f"  Runtime: {elapsed:.1f}s")

    return result


# ── Test 2: Monte Carlo — Random SPD Matrices ────────────────────────
def test_monte_carlo_rational(n_mc=N_MC):
    """
    Generate random 7×7 SPD matrices with similar structure to G_MEAN.
    Count how many have det within tolerance of ANY nice rational p/q
    (q ≤ 64, 1 < p/q < 4). Tests whether det ≈ 65/32 is special.
    """
    print("\n" + "=" * 70)
    print("TEST 2: MONTE CARLO — RANDOM SPD NEAR IDENTITY")
    print("=" * 70)

    # Build set of "nice" rationals p/q with q ≤ 64 in range [1.5, 2.5]
    rationals = set()
    for q in range(1, 65):
        for p in range(1, 4 * q + 1):
            r = p / q
            if 1.5 < r < 2.5:
                rationals.add((p, q, r))

    # Remove duplicates (same float value)
    rat_values = {}
    for p, q, r in rationals:
        key = round(r, 12)
        if key not in rat_values or q < rat_values[key][1]:
            rat_values[key] = (p, q, r)
    unique_rationals = list(rat_values.values())
    rat_array = np.array([r for _, _, r in unique_rationals])

    print(f"  Nice rationals p/q (q ≤ 64) in [1.5, 2.5]: {len(unique_rationals)}")

    rng = np.random.default_rng(123)

    # Track: how many random matrices have det close to ANY nice rational
    tol = 1e-3  # 0.1% — generous tolerance
    tol_tight = 1e-5  # 0.001% — tight tolerance
    tol_paper = 4e-8  # 4 × 10⁻⁸ % — Paper 2 level

    n_any_rational_loose = 0
    n_any_rational_tight = 0
    n_any_rational_paper = 0
    n_6532_loose = 0
    n_6532_tight = 0
    n_6532_paper = 0
    dets = np.zeros(n_mc)

    t0 = time.time()
    for i in range(n_mc):
        g = random_spd_near_identity(rng)
        d = np.linalg.det(g)
        dets[i] = d

        # Check proximity to any nice rational
        frac_devs = np.abs(d - rat_array) / rat_array * 100
        min_dev = np.min(frac_devs)

        if min_dev < tol * 100:      # convert to %
            n_any_rational_loose += 1
        if min_dev < tol_tight * 100:
            n_any_rational_tight += 1
        if min_dev < tol_paper * 100:
            n_any_rational_paper += 1

        # Check proximity to 65/32 specifically
        dev_6532 = abs(d - DET_TARGET) / DET_TARGET * 100
        if dev_6532 < tol * 100:
            n_6532_loose += 1
        if dev_6532 < tol_tight * 100:
            n_6532_tight += 1
        if dev_6532 < tol_paper * 100:
            n_6532_paper += 1

        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{n_mc} matrices ({elapsed:.0f}s)...")

    elapsed = time.time() - t0

    result = {
        "n_matrices": n_mc,
        "n_nice_rationals_tested": len(unique_rationals),
        "det_mean": float(np.mean(dets)),
        "det_std": float(np.std(dets)),
        "det_range": [float(np.min(dets)), float(np.max(dets))],
        "any_rational_within_0.1pct": int(n_any_rational_loose),
        "any_rational_within_0.001pct": int(n_any_rational_tight),
        "any_rational_within_paper_tol": int(n_any_rational_paper),
        "exact_6532_within_0.1pct": int(n_6532_loose),
        "exact_6532_within_0.001pct": int(n_6532_tight),
        "exact_6532_within_paper_tol": int(n_6532_paper),
        "frac_6532_loose": float(n_6532_loose / n_mc),
        "frac_6532_tight": float(n_6532_tight / n_mc),
        "frac_6532_paper": float(n_6532_paper / n_mc),
        "runtime_s": float(elapsed),
    }

    print(f"  det distribution: mean={result['det_mean']:.4f}, "
          f"std={result['det_std']:.4f}")
    print(f"  det ∈ [{result['det_range'][0]:.3f}, {result['det_range'][1]:.3f}]")
    print(f"  --- Any nice rational p/q ---")
    print(f"    Within 0.1%:     {n_any_rational_loose:,} / {n_mc:,} "
          f"({n_any_rational_loose/n_mc:.4%})")
    print(f"    Within 0.001%:   {n_any_rational_tight:,} / {n_mc:,} "
          f"({n_any_rational_tight/n_mc:.4%})")
    print(f"    Within paper tol: {n_any_rational_paper:,} / {n_mc:,} "
          f"({n_any_rational_paper/n_mc:.6%})")
    print(f"  --- Specifically 65/32 ---")
    print(f"    Within 0.1%:     {n_6532_loose:,} / {n_mc:,} "
          f"({n_6532_loose/n_mc:.4%})")
    print(f"    Within 0.001%:   {n_6532_tight:,} / {n_mc:,} "
          f"({n_6532_tight/n_mc:.6%})")
    print(f"    Within paper tol: {n_6532_paper:,} / {n_mc:,} "
          f"({n_6532_paper/n_mc:.8%})")
    print(f"  Runtime: {elapsed:.1f}s")

    return result


# ── Test 3: Sobol Sensitivity on Cholesky Components ─────────────────
def sobol_sequence_2d(n):
    """Generate a 2D Sobol-like quasi-random sequence (Van der Corput)."""
    def vdc(n_pts, base=2):
        seq = np.zeros(n_pts)
        for i in range(n_pts):
            f, r = 1.0, 0.0
            val = i + 1
            while val > 0:
                f /= base
                r += f * (val % base)
                val //= base
            seq[i] = r
        return seq
    return np.column_stack([vdc(n, 2), vdc(n, 3)])


def test_sobol_sensitivity(n_sobol=N_SOBOL):
    """
    Sobol-like sensitivity analysis: perturb individual Cholesky components
    and measure the response of det, κ, and eigenvalue spread.

    Groups the 28 Cholesky DOF into diagonal (7) and off-diagonal (21).
    """
    print("\n" + "=" * 70)
    print("TEST 3: SOBOL SENSITIVITY (Cholesky components)")
    print("=" * 70)

    L0 = np.linalg.cholesky(G_MEAN)

    # We'll perturb each of the 28 lower-triangular entries independently
    n_lt = 28  # 7 diag + 21 off-diag
    lt_indices = []
    for i in range(7):
        for j in range(i + 1):
            lt_indices.append((i, j))

    # Classify: diagonal vs off-diagonal
    diag_mask = np.array([i == j for i, j in lt_indices])

    rng = np.random.default_rng(77)
    perturbation_scale = 1e-4  # small perturbation to Cholesky entries

    # For each component, perturb and measure output
    n_samples = n_sobol
    det_responses = np.zeros((n_lt, n_samples))
    kappa_responses = np.zeros((n_lt, n_samples))

    t0 = time.time()
    for k, (ci, cj) in enumerate(lt_indices):
        perturbations = rng.uniform(-perturbation_scale, perturbation_scale,
                                    n_samples)
        for s in range(n_samples):
            dL = np.zeros((7, 7))
            dL[ci, cj] = perturbations[s]
            g = cholesky_perturb(L0, dL)
            det_responses[k, s] = np.linalg.det(g)
            eigs = np.linalg.eigvalsh(g)
            kappa_responses[k, s] = eigs[-1] / eigs[0]

        if (k + 1) % 7 == 0:
            print(f"    Component {k+1}/{n_lt} done...")

    elapsed = time.time() - t0

    # Compute sensitivity: variance of output due to each component
    det_var_per_component = np.var(det_responses, axis=1)
    kappa_var_per_component = np.var(kappa_responses, axis=1)

    det_var_total = np.sum(det_var_per_component)
    kappa_var_total = np.sum(kappa_var_per_component)

    det_frac_diag = float(np.sum(det_var_per_component[diag_mask]) / det_var_total) \
        if det_var_total > 0 else 0
    det_frac_offdiag = 1.0 - det_frac_diag

    kappa_frac_diag = float(np.sum(kappa_var_per_component[diag_mask]) / kappa_var_total) \
        if kappa_var_total > 0 else 0
    kappa_frac_offdiag = 1.0 - kappa_frac_diag

    # Top 5 most sensitive components for det
    top5_det = np.argsort(det_var_per_component)[::-1][:5]
    top5_det_info = [
        {"index": int(k), "entry": list(lt_indices[k]),
         "is_diagonal": bool(diag_mask[k]),
         "variance_fraction": float(det_var_per_component[k] / det_var_total)}
        for k in top5_det
    ]

    result = {
        "n_sobol_per_component": n_samples,
        "n_components": n_lt,
        "perturbation_scale": perturbation_scale,
        "det_sensitivity": {
            "frac_diagonal": float(det_frac_diag),
            "frac_off_diagonal": float(det_frac_offdiag),
            "top5_components": top5_det_info,
        },
        "kappa_sensitivity": {
            "frac_diagonal": float(kappa_frac_diag),
            "frac_off_diagonal": float(kappa_frac_offdiag),
        },
        "runtime_s": float(elapsed),
    }

    print(f"  det(g) sensitivity:")
    print(f"    Diagonal entries:     {det_frac_diag:.1%} of variance")
    print(f"    Off-diagonal entries: {det_frac_offdiag:.1%} of variance")
    print(f"  κ sensitivity:")
    print(f"    Diagonal entries:     {kappa_frac_diag:.1%} of variance")
    print(f"    Off-diagonal entries: {kappa_frac_offdiag:.1%} of variance")
    print(f"  Top 5 det-sensitive components:")
    for info in top5_det_info:
        print(f"    L[{info['entry'][0]},{info['entry'][1]}] "
              f"({'diag' if info['is_diagonal'] else 'off'}): "
              f"{info['variance_fraction']:.1%}")
    print(f"  Runtime: {elapsed:.1f}s")

    return result


# ── Test 4: Bootstrap Stability ───────────────────────────────────────
def test_bootstrap(n_boot=N_BOOTSTRAP):
    """
    Bootstrap: add Gaussian noise at the measurement precision level
    to each entry of G_MEAN. Propagate to det, κ, and all 7 eigenvalues.
    Tests: are the reported values stable under measurement uncertainty?
    """
    print("\n" + "=" * 70)
    print("TEST 4: BOOTSTRAP STABILITY (measurement noise)")
    print("=" * 70)

    rng = np.random.default_rng(999)

    dets = np.zeros(n_boot)
    kappas = np.zeros(n_boot)
    all_eigs = np.zeros((n_boot, 7))

    t0 = time.time()
    for i in range(n_boot):
        noise = rng.normal(0, MEASUREMENT_NOISE, (7, 7))
        noise = (noise + noise.T) / 2  # symmetrize
        g_noisy = G_MEAN + noise
        # Ensure SPD (noise is tiny, so this is a safety check)
        eigs = np.linalg.eigvalsh(g_noisy)
        if np.min(eigs) > 0:
            dets[i] = np.linalg.det(g_noisy)
            kappas[i] = eigs[-1] / eigs[0]
            all_eigs[i] = eigs
        else:
            dets[i] = np.nan
            kappas[i] = np.nan
            all_eigs[i] = np.nan

        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{n_boot} bootstrap resamples...")

    elapsed = time.time() - t0

    valid = ~np.isnan(dets)
    n_valid = int(np.sum(valid))

    # det stability
    det_mean = float(np.mean(dets[valid]))
    det_std = float(np.std(dets[valid]))
    det_ci95 = [float(np.percentile(dets[valid], 2.5)),
                float(np.percentile(dets[valid], 97.5))]

    # κ stability
    kappa_mean = float(np.mean(kappas[valid]))
    kappa_std = float(np.std(kappas[valid]))

    # Eigenvalue stability
    eig_means = np.mean(all_eigs[valid], axis=0)
    eig_stds = np.std(all_eigs[valid], axis=0)
    eig_max_shift = float(np.max(np.abs(eig_means - EIGENVALUES_TARGET)))

    # How many sig figs of det are stable?
    if det_std > 0:
        det_sig_figs = int(-np.log10(det_std / det_mean))
    else:
        det_sig_figs = 15

    result = {
        "n_bootstrap": n_boot,
        "n_valid": n_valid,
        "noise_level": MEASUREMENT_NOISE,
        "det": {
            "mean": det_mean,
            "std": det_std,
            "ci95": det_ci95,
            "stable_sig_figs": det_sig_figs,
            "contains_target": bool(det_ci95[0] <= DET_TARGET <= det_ci95[1]),
        },
        "kappa": {
            "mean": kappa_mean,
            "std": kappa_std,
        },
        "eigenvalues": {
            "mean": eig_means.tolist(),
            "std": eig_stds.tolist(),
            "max_shift_from_target": eig_max_shift,
            "all_within_1e7": bool(eig_max_shift < 1e-7),
        },
        "runtime_s": float(elapsed),
    }

    print(f"  Noise level: σ = {MEASUREMENT_NOISE:.0e}")
    print(f"  Valid resamples: {n_valid}/{n_boot}")
    print(f"  det(g): {det_mean:.10f} ± {det_std:.2e} "
          f"({det_sig_figs} stable sig figs)")
    print(f"  det 95% CI: [{det_ci95[0]:.10f}, {det_ci95[1]:.10f}]")
    print(f"  Contains 65/32: {result['det']['contains_target']}")
    print(f"  κ: {kappa_mean:.6f} ± {kappa_std:.2e}")
    print(f"  Eigenvalue max shift: {eig_max_shift:.2e}")
    print(f"  Runtime: {elapsed:.1f}s")

    return result


# ── Test 5: Look-Elsewhere for Rational Determinant ──────────────────
def test_look_elsewhere():
    """
    Among all "nice" rationals p/q with q ≤ Q_MAX, how likely is it
    that a random det value near 2.03 would land within tolerance of
    one of them? This quantifies the look-elsewhere effect for the
    claim det(g) = 65/32.
    """
    print("\n" + "=" * 70)
    print("TEST 5: LOOK-ELSEWHERE (rational det target)")
    print("=" * 70)

    Q_MAX = 128  # denominators up to 128

    # Enumerate all reduced fractions p/q in [1.5, 2.5] with q ≤ Q_MAX
    from math import gcd
    rationals = []
    seen = set()
    for q in range(1, Q_MAX + 1):
        for p in range(int(1.5 * q), int(2.5 * q) + 1):
            g = gcd(p, q)
            pr, qr = p // g, q // g
            if (pr, qr) not in seen:
                seen.add((pr, qr))
                rationals.append((pr, qr, pr / qr))

    rationals.sort(key=lambda x: x[2])
    n_rats = len(rationals)
    print(f"  Reduced fractions p/q (q ≤ {Q_MAX}) in [1.5, 2.5]: {n_rats}")

    # What fraction of the interval [1.5, 2.5] is within tolerance of
    # some rational? (measure of the "catch basin")
    det_actual = np.linalg.det(G_MEAN)
    interval_length = 1.0  # [1.5, 2.5]

    # At paper tolerance (4 × 10⁻⁸ %)
    tol_abs_paper = DET_TARGET * DET_TOLERANCE / 100
    catch_paper = 2 * tol_abs_paper * n_rats / interval_length

    # At looser tolerances
    tol_levels = {
        "0.1%": 0.001,
        "0.01%": 0.0001,
        "0.001%": 0.00001,
        "paper (4e-8%)": DET_TOLERANCE / 100,
    }

    catch_basins = {}
    for label, tol_frac in tol_levels.items():
        # Total "catch" measure: sum of 2*tol*r for each rational r
        total_catch = sum(2 * tol_frac * r for _, _, r in rationals)
        # Probability of landing in catch basin (assuming uniform det)
        p_catch = min(1.0, total_catch / interval_length)
        catch_basins[label] = float(p_catch)

    # Is 65/32 special among the rationals? Check: smallest q giving
    # det within tolerance
    det_dist = [(abs(det_actual - r), p, q) for p, q, r in rationals]
    det_dist.sort()

    closest = det_dist[:10]

    # Bonferroni: n_rats trials, each with p = catch_paper
    p_bonferroni = min(1.0, n_rats * catch_basins.get("paper (4e-8%)", 0))

    result = {
        "n_rationals_tested": n_rats,
        "q_max": Q_MAX,
        "det_actual": float(det_actual),
        "catch_basin_probabilities": catch_basins,
        "closest_rationals": [
            {"p": int(p), "q": int(q), "value": float(p/q),
             "distance": float(d)}
            for d, p, q in closest
        ],
        "target_65_32_rank": int(next(
            i for i, (_, p, q) in enumerate(det_dist) if p == 65 and q == 32
        )) + 1,
        "p_bonferroni_paper_tol": float(p_bonferroni),
    }

    print(f"  Catch basin probabilities (uniform det in [1.5, 2.5]):")
    for label, p in catch_basins.items():
        print(f"    {label:>20s}: {p:.4e}")
    print(f"  Closest rationals to det(g):")
    for info in result["closest_rationals"][:5]:
        print(f"    {info['p']}/{info['q']} = {info['value']:.10f} "
              f"(dist = {info['distance']:.2e})")
    print(f"  65/32 rank among closest: #{result['target_65_32_rank']}")
    print(f"  Bonferroni p (paper tol): {p_bonferroni:.4e}")

    return result


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("STATISTICAL VALIDATION SUITE")
    print("Explicit G₂ Metric on K₇ (Paper 2)")
    print("=" * 70)

    # ── Verify input data ──
    print("\n[1/6] Verifying metric data...")
    det_g = np.linalg.det(G_MEAN)
    eigs = np.linalg.eigvalsh(G_MEAN)
    kappa = eigs[-1] / eigs[0]
    print(f"  det(G_MEAN) = {det_g:.10f} (target: {DET_TARGET})")
    print(f"  κ(G_MEAN) = {kappa:.6f} (target: {KAPPA_TARGET})")
    print(f"  Eigenvalues: {eigs}")
    print(f"  Positive definite: {np.all(eigs > 0)}")
    print(f"  Symmetric: {np.allclose(G_MEAN, G_MEAN.T)}")

    # ── Run tests ──
    print("\n[2/6] Running statistical tests...")
    results = {"metadata": {
        "det_target": DET_TARGET,
        "kappa_target": KAPPA_TARGET,
        "det_actual": float(det_g),
        "kappa_actual": float(kappa),
        "eigenvalues_actual": eigs.tolist(),
        "eigenvalues_target": EIGENVALUES_TARGET.tolist(),
        "measurement_noise": MEASUREMENT_NOISE,
        "date": "2026-02-08",
    }}

    # Test 1: Permutation
    results["permutation"] = test_permutation(n_perm=N_PERM)

    # Test 2: Monte Carlo rational
    results["monte_carlo"] = test_monte_carlo_rational(n_mc=N_MC)

    # Test 3: Sobol sensitivity
    results["sobol"] = test_sobol_sensitivity(n_sobol=N_SOBOL)

    # Test 4: Bootstrap
    results["bootstrap"] = test_bootstrap(n_boot=N_BOOTSTRAP)

    # Test 5: Look-elsewhere
    results["look_elsewhere"] = test_look_elsewhere()

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Metric: det = {det_g:.10f}, κ = {kappa:.6f}")
    print(f"  T1 Permutation:    Z = {results['permutation']['z_score']:.1f}, "
          f"p = {results['permutation']['p_value']:.4e}")
    print(f"  T2 MC rational:    {results['monte_carlo']['frac_6532_tight']:.6%} "
          f"hit 65/32 at 0.001%")
    print(f"  T3 Sobol:          {results['sobol']['det_sensitivity']['frac_diagonal']:.1%} "
          f"det variance from diagonal")
    print(f"  T4 Bootstrap:      det stable to "
          f"{results['bootstrap']['det']['stable_sig_figs']} sig figs")
    print(f"  T5 Look-elsewhere: p = "
          f"{results['look_elsewhere']['p_bonferroni_paper_tol']:.4e}")

    # ── Save ──
    out_path = RESULTS_DIR / "paper2_validation_results.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean = json.loads(json.dumps(results, default=convert))
    with open(out_path, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
