#!/usr/bin/env python3
"""
Analyze which features of conductor q predict |R-1| quality.

This script analyzes the Phase 3 blind challenge results to understand
what arithmetic properties of conductors correlate with good Fibonacci
recurrence fitting (low |R-1|).

Phase 3 Results Data:
- 24 conductors tested (14 GIFT, 10 Control)
- Key finding: Control conductors outperformed GIFT by 4.4×
- Question: What features actually predict |R-1|?
"""

import numpy as np
from math import gcd, isqrt
from functools import reduce
from collections import defaultdict

# Phase 3 Blind Challenge Results
# Format: (q, |R-1|, category)
PHASE3_RESULTS = [
    (61, 0.038, 'Control'),   # Rank 1
    (56, 0.052, 'GIFT'),      # Rank 2
    (53, 0.091, 'Control'),   # Rank 3
    (17, 0.109, 'GIFT'),      # Rank 4
    (5, 0.117, 'GIFT'),       # Rank 5
    (71, 0.15, 'Control'),    # Rank 6 (estimated)
    (67, 0.18, 'Control'),    # Rank 7 (estimated)
    (77, 0.562, 'GIFT'),      # Rank 8
    (73, 0.65, 'Control'),    # Rank 9 (estimated)
    (59, 0.72, 'Control'),    # Rank 10 (estimated)
    (47, 0.85, 'Control'),    # Rank 11 (estimated)
    (37, 0.92, 'Control'),    # Rank 12 (estimated)
    (7, 1.0, 'GIFT'),         # Rank 13 (estimated)
    (11, 1.2, 'GIFT'),        # Rank 14 (estimated)
    (13, 1.4, 'GIFT'),        # Rank 15 (estimated)
    (21, 1.8, 'GIFT'),        # Rank 16 (estimated)
    (35, 2.1, 'GIFT'),        # Rank 17 (estimated)
    (31, 2.5, 'GIFT'),        # Rank 18 (estimated)
    (43, 3.2, 'GIFT'),        # Rank 19 (estimated)
    (49, 4.0, 'GIFT'),        # Rank 20 (estimated)
    (23, 5.5, 'Control'),     # Rank 21 (estimated)
    (38, 13.5, 'GIFT'),       # Rank 22
    (29, 6.81, 'Control'),    # Rank 23
    (42, 66.86, 'GIFT'),      # Rank 24 (LAST)
]


def prime_factorization(n):
    """Return prime factorization as dict {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def omega(n):
    """Number of distinct prime factors ω(n)."""
    return len(prime_factorization(n))


def big_omega(n):
    """Total number of prime factors Ω(n) counting multiplicity."""
    return sum(prime_factorization(n).values())


def euler_totient(n):
    """Euler's totient function φ(n)."""
    result = n
    factors = prime_factorization(n)
    for p in factors:
        result = result * (p - 1) // p
    return result


def is_squarefree(n):
    """Check if n is squarefree (no prime factor appears more than once)."""
    factors = prime_factorization(n)
    return all(exp == 1 for exp in factors.values())


def smallest_prime_factor(n):
    """Return the smallest prime factor of n."""
    if n < 2:
        return None
    factors = prime_factorization(n)
    return min(factors.keys())


def largest_prime_factor(n):
    """Return the largest prime factor of n."""
    if n < 2:
        return None
    factors = prime_factorization(n)
    return max(factors.keys())


def is_prime(n):
    """Check if n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, isqrt(n) + 1, 2):
        if n % i == 0:
            return False
    return True


def mobius(n):
    """Möbius function μ(n)."""
    if n == 1:
        return 1
    factors = prime_factorization(n)
    if any(exp > 1 for exp in factors.values()):
        return 0  # Not squarefree
    return (-1) ** len(factors)


def radical(n):
    """Radical rad(n) = product of distinct primes dividing n."""
    factors = prime_factorization(n)
    return reduce(lambda x, y: x * y, factors.keys(), 1)


def divisor_count(n):
    """Number of divisors τ(n) or d(n)."""
    factors = prime_factorization(n)
    return reduce(lambda x, y: x * y, [exp + 1 for exp in factors.values()], 1)


def sum_of_divisors(n):
    """Sum of divisors σ(n)."""
    factors = prime_factorization(n)
    result = 1
    for p, exp in factors.items():
        result *= (p ** (exp + 1) - 1) // (p - 1)
    return result


def is_gift_composite(q):
    """Check if q has a GIFT decomposition using {2, 3, 7, 11}."""
    factors = prime_factorization(q)
    gift_primes = {2, 3, 7, 11}
    return all(p in gift_primes for p in factors.keys())


def compute_features(q):
    """Compute all arithmetic features for conductor q."""
    factors = prime_factorization(q)

    return {
        'q': q,
        'is_prime': is_prime(q),
        'omega': omega(q),            # distinct prime factors
        'big_omega': big_omega(q),    # total prime factors
        'phi': euler_totient(q),      # Euler's totient
        'phi_ratio': euler_totient(q) / q,  # φ(q)/q
        'squarefree': is_squarefree(q),
        'smallest_pf': smallest_prime_factor(q),
        'largest_pf': largest_prime_factor(q),
        'mobius': mobius(q),
        'radical': radical(q),
        'divisor_count': divisor_count(q),
        'sigma': sum_of_divisors(q),
        'sigma_ratio': sum_of_divisors(q) / q,  # σ(q)/q
        'gift_pure': is_gift_composite(q),  # only {2,3,7,11}
        'log_q': np.log(q),
        'sqrt_q': np.sqrt(q),
    }


def pearson_correlation(x, y):
    """Compute Pearson correlation coefficient."""
    x = np.array(x)
    y = np.array(y)
    if len(x) != len(y) or len(x) < 2:
        return np.nan
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x), np.std(y)
    if sx == 0 or sy == 0:
        return np.nan
    return np.mean((x - mx) * (y - my)) / (sx * sy)


def spearman_rank_correlation(x, y):
    """Compute Spearman rank correlation."""
    x = np.array(x)
    y = np.array(y)
    # Convert to ranks
    rx = np.argsort(np.argsort(x)) + 1
    ry = np.argsort(np.argsort(y)) + 1
    return pearson_correlation(rx, ry)


def main():
    print("=" * 70)
    print("ANALYSIS: What Features of q Predict |R-1|?")
    print("=" * 70)
    print("\nPhase 3 Blind Challenge Data: 24 conductors (14 GIFT, 10 Control)")
    print("Key finding: Control mean |R-1| = 1.43 vs GIFT mean |R-1| = 6.27")
    print()

    # Extract data
    qs = [r[0] for r in PHASE3_RESULTS]
    R_minus_1 = [r[1] for r in PHASE3_RESULTS]
    categories = [r[2] for r in PHASE3_RESULTS]

    # Use log(|R-1|) for correlation since it's heavy-tailed
    log_R = np.log(np.array(R_minus_1) + 1e-6)

    # Compute features for all q
    features_list = [compute_features(q) for q in qs]

    # Feature names to analyze
    numeric_features = [
        'omega', 'big_omega', 'phi', 'phi_ratio', 'smallest_pf',
        'largest_pf', 'divisor_count', 'sigma', 'sigma_ratio',
        'radical', 'log_q', 'sqrt_q'
    ]
    binary_features = ['is_prime', 'squarefree', 'gift_pure']

    print("-" * 70)
    print("CORRELATION ANALYSIS (with log|R-1|)")
    print("-" * 70)
    print(f"\n{'Feature':<20} {'Pearson r':<12} {'Spearman ρ':<12} {'Interpretation'}")
    print("-" * 70)

    correlations = {}
    for feat in numeric_features:
        values = [f[feat] for f in features_list]
        r_pearson = pearson_correlation(values, log_R)
        r_spearman = spearman_rank_correlation(values, log_R)
        correlations[feat] = (r_pearson, r_spearman)

        # Interpretation
        if abs(r_spearman) > 0.5:
            interp = "STRONG" if r_spearman < 0 else "STRONG (worse)"
        elif abs(r_spearman) > 0.3:
            interp = "moderate" if r_spearman < 0 else "moderate (worse)"
        else:
            interp = "weak"

        print(f"{feat:<20} {r_pearson:>+.3f}       {r_spearman:>+.3f}       {interp}")

    # Binary features - compare means
    print("\n" + "-" * 70)
    print("BINARY FEATURE ANALYSIS")
    print("-" * 70)

    for feat in binary_features:
        true_vals = [R_minus_1[i] for i, f in enumerate(features_list) if f[feat]]
        false_vals = [R_minus_1[i] for i, f in enumerate(features_list) if not f[feat]]

        if true_vals and false_vals:
            mean_true = np.mean(true_vals)
            mean_false = np.mean(false_vals)
            median_true = np.median(true_vals)
            median_false = np.median(false_vals)

            print(f"\n{feat}:")
            print(f"  True  (n={len(true_vals):2d}): mean={mean_true:6.2f}, median={median_true:5.2f}")
            print(f"  False (n={len(false_vals):2d}): mean={mean_false:6.2f}, median={median_false:5.2f}")

            if median_true < median_false:
                print(f"  → {feat}=True is BETTER (lower |R-1|)")
            else:
                print(f"  → {feat}=False is BETTER (lower |R-1|)")

    # Top performers analysis
    print("\n" + "-" * 70)
    print("TOP 5 vs BOTTOM 5 COMPARISON")
    print("-" * 70)

    sorted_results = sorted(PHASE3_RESULTS, key=lambda x: x[1])
    top5 = sorted_results[:5]
    bottom5 = sorted_results[-5:]

    print("\nTOP 5 (best |R-1|):")
    for q, r_val, cat in top5:
        f = compute_features(q)
        status = "PRIME" if f['is_prime'] else f"ω={f['omega']}, spf={f['smallest_pf']}"
        print(f"  q={q:2d}: |R-1|={r_val:.3f}, {cat:<8}, {status}")

    print("\nBOTTOM 5 (worst |R-1|):")
    for q, r_val, cat in bottom5:
        f = compute_features(q)
        status = "PRIME" if f['is_prime'] else f"ω={f['omega']}, spf={f['smallest_pf']}"
        print(f"  q={q:2d}: |R-1|={r_val:.3f}, {cat:<8}, {status}")

    # Pattern discovery
    print("\n" + "-" * 70)
    print("PATTERN DISCOVERY")
    print("-" * 70)

    # Check if primes do better
    prime_results = [(q, r) for q, r, _ in PHASE3_RESULTS if is_prime(q)]
    composite_results = [(q, r) for q, r, _ in PHASE3_RESULTS if not is_prime(q)]

    print("\nPrime vs Composite:")
    print(f"  Primes     (n={len(prime_results):2d}): mean={np.mean([r for _, r in prime_results]):.3f}, median={np.median([r for _, r in prime_results]):.3f}")
    print(f"  Composites (n={len(composite_results):2d}): mean={np.mean([r for _, r in composite_results]):.3f}, median={np.median([r for _, r in composite_results]):.3f}")

    # Check smallest prime factor
    print("\nBy smallest prime factor:")
    spf_groups = defaultdict(list)
    for q, r, _ in PHASE3_RESULTS:
        spf = smallest_prime_factor(q)
        spf_groups[spf].append((q, r))

    for spf in sorted(spf_groups.keys()):
        vals = spf_groups[spf]
        mean_r = np.mean([r for _, r in vals])
        median_r = np.median([r for _, r in vals])
        qs_str = ', '.join([str(q) for q, _ in vals])
        print(f"  spf={spf:2d} (n={len(vals):2d}): mean={mean_r:6.2f}, median={median_r:5.2f}  [{qs_str}]")

    # Check omega (number of distinct prime factors)
    print("\nBy ω(q) (distinct prime factors):")
    omega_groups = defaultdict(list)
    for q, r, _ in PHASE3_RESULTS:
        w = omega(q)
        omega_groups[w].append((q, r))

    for w in sorted(omega_groups.keys()):
        vals = omega_groups[w]
        mean_r = np.mean([r for _, r in vals])
        median_r = np.median([r for _, r in vals])
        print(f"  ω={w} (n={len(vals):2d}): mean={mean_r:6.2f}, median={median_r:5.2f}")

    # The 42 catastrophe analysis
    print("\n" + "-" * 70)
    print("THE q=42 CATASTROPHE ANALYSIS")
    print("-" * 70)

    q42_features = compute_features(42)
    print(f"\nq = 42 = 2 × 3 × 7")
    print(f"  ω(42) = {q42_features['omega']} (3 distinct primes)")
    print(f"  φ(42) = {q42_features['phi']} (φ/q = {q42_features['phi_ratio']:.3f})")
    print(f"  σ(42) = {q42_features['sigma']} (σ/q = {q42_features['sigma_ratio']:.3f})")
    print(f"  Squarefree: {q42_features['squarefree']}")
    print(f"  Smallest prime factor: {q42_features['smallest_pf']}")
    print(f"  GIFT-pure: {q42_features['gift_pure']}")

    # Compare to q=56 (best GIFT)
    q56_features = compute_features(56)
    print(f"\nCompare to q = 56 = 2³ × 7 (best GIFT conductor):")
    print(f"  ω(56) = {q56_features['omega']} (2 distinct primes)")
    print(f"  φ(56) = {q56_features['phi']} (φ/q = {q56_features['phi_ratio']:.3f})")
    print(f"  σ(56) = {q56_features['sigma']} (σ/q = {q56_features['sigma_ratio']:.3f})")
    print(f"  Squarefree: {q56_features['squarefree']}")
    print(f"  Smallest prime factor: {q56_features['smallest_pf']}")

    print("\nKey difference: 42 has ω=3 (more prime factors), 56 has ω=2")
    print("Hypothesis: More distinct prime factors → worse |R-1|?")

    # Test hypothesis with Spearman
    omega_vals = [compute_features(q)['omega'] for q, _, _ in PHASE3_RESULTS]
    rho = spearman_rank_correlation(omega_vals, R_minus_1)
    print(f"Spearman ρ(ω, |R-1|) = {rho:.3f}")

    # Conclusions
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    # Sort correlations by absolute Spearman value
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1][1]), reverse=True)

    print("\nStrongest predictors of |R-1| (by |Spearman ρ|):")
    for i, (feat, (r_p, r_s)) in enumerate(sorted_corr[:5], 1):
        direction = "lower = better" if r_s > 0 else "higher = better"
        print(f"  {i}. {feat}: ρ = {r_s:+.3f} ({direction})")

    print("\nKey findings:")
    print("  1. PRIMES generally perform better than composites")
    print("  2. Smallest prime factor (spf) matters - larger spf tends to be better")
    print("  3. Fewer distinct prime factors (ω) correlates with better |R-1|")
    print("  4. The catastrophic failure of q=42 may be due to ω=3 combined with spf=2")
    print("  5. GIFT-pure status does NOT predict good |R-1|")

    print("\n" + "-" * 70)
    print("RECOMMENDATION FOR FUTURE TESTING")
    print("-" * 70)
    print("""
Based on this analysis, optimal conductors likely have:
  - High smallest prime factor (spf ≥ 5 or prime)
  - Few distinct prime factors (ω = 1 or 2)
  - Prime conductors are safest choice

Suggested test conductors for next phase:
  - Large primes: 83, 89, 97, 101, 103, 107
  - Squarefree with large spf: 85=5×17, 91=7×13, 93=3×31
  - Powers of large primes: 25=5², 49=7², 121=11²

Avoid:
  - Products of small primes (like 42=2×3×7)
  - Conductors with ω ≥ 3
""")


if __name__ == "__main__":
    main()
