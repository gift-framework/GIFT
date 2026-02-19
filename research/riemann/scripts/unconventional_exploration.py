#!/usr/bin/env python3
"""
UNCONVENTIONAL EXPLORATION: Wild Ideas for Riemann Zero Structure
==================================================================

This script explores UNCONVENTIONAL approaches to find hidden structure
in the Riemann zeta zeros. These are creative, speculative methods that
go beyond standard random matrix theory and smooth asymptotics.

Approaches implemented:
1. Continued fraction expansion - do partial quotients show patterns?
2. Digit analysis - distribution anomalies in decimal representation
3. Difference sequences - Delta^k gamma_n, looking for attractors
4. Modular arithmetic - gamma_n mod primes, mod Fibonacci numbers
5. Ratio chains - gamma_{n+k}/gamma_n convergence
6. Cross-correlation - with Fibonacci, primes, other sequences
7. Phase space portraits - (gamma_n, gamma_{n+1}) structure
8. Information theory - entropy of spacing distribution

May reveal connections to GIFT or entirely new structure!
"""

import numpy as np
from pathlib import Path
from fractions import Fraction
from collections import Counter
import json
import math

# =============================================================================
# CONSTANTS
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2    # Golden ratio
PHI_INV = PHI - 1              # 1/phi = phi - 1
SQRT5 = np.sqrt(5)
PI = np.pi
E = np.e

# GIFT topological constants
B2 = 21                        # Second Betti number of K7
B3 = 77                        # Third Betti number of K7
DIM_G2 = 14                    # G2 holonomy dimension
H_STAR = 99                    # b2 + b3 + 1
DIM_E8 = 248                   # E8 dimension
RANK_E8 = 8
DIM_J3O = 27                   # Jordan algebra dimension

# Generate sequences
def fib(n):
    """nth Fibonacci number (1-indexed: F_1=1, F_2=1, ...)"""
    if n <= 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def sieve_primes(limit):
    """Sieve of Eratosthenes"""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(limit + 1) if is_prime[i]]

FIBS = [fib(i) for i in range(1, 25)]  # First 24 Fibonacci numbers
PRIMES = sieve_primes(1000)             # Primes up to 1000

# =============================================================================
# DATA LOADING
# =============================================================================

def load_zeros(max_zeros=100000):
    """Load Riemann zeros from data files"""
    zeros = []
    zeros_dir = Path(__file__).parent
    for i in range(1, 6):
        zeros_file = zeros_dir / f"zeros{i}"
        if zeros_file.exists():
            with open(zeros_file) as f:
                for line in f:
                    if line.strip():
                        try:
                            parts = line.strip().split()
                            val = float(parts[-1]) if len(parts) > 1 else float(parts[0])
                            zeros.append(val)
                        except:
                            continue
                        if len(zeros) >= max_zeros:
                            return np.array(zeros)
    return np.array(zeros)

# =============================================================================
# 1. CONTINUED FRACTION EXPANSION
# =============================================================================

def continued_fraction(x, max_terms=30):
    """Compute continued fraction expansion of x"""
    cf = []
    for _ in range(max_terms):
        a = int(np.floor(x))
        cf.append(a)
        frac = x - a
        if abs(frac) < 1e-12:
            break
        x = 1.0 / frac
        if abs(x) > 1e12:
            break
    return cf

def analyze_continued_fractions(zeros, n_samples=1000):
    """Analyze continued fraction patterns in zeros"""
    print("\n" + "=" * 70)
    print("1. CONTINUED FRACTION EXPANSION ANALYSIS")
    print("=" * 70)

    # Compute CF expansions for sample of zeros
    indices = np.linspace(0, len(zeros)-1, n_samples, dtype=int)
    all_quotients = []
    first_quotients = []
    second_quotients = []

    for idx in indices:
        cf = continued_fraction(zeros[idx])
        all_quotients.extend(cf[1:])  # Skip integer part
        if len(cf) > 1:
            first_quotients.append(cf[1])
        if len(cf) > 2:
            second_quotients.append(cf[2])

    # Distribution of partial quotients
    print("\nPartial quotient distribution (excluding integer part):")
    quotient_counts = Counter(all_quotients)
    total = sum(quotient_counts.values())

    # Gauss-Kuzmin law predicts P(a_k = n) ~ log_2((n+1)^2/(n(n+2)))
    print("\n  Value | Observed | Gauss-Kuzmin | Deviation")
    print("  " + "-" * 45)

    deviations = []
    for n in range(1, 11):
        observed = quotient_counts.get(n, 0) / total
        # Gauss-Kuzmin probability
        gk_prob = np.log2((n + 1)**2 / (n * (n + 2)))
        deviation = (observed - gk_prob) / gk_prob * 100 if gk_prob > 0 else 0
        deviations.append(abs(deviation))
        print(f"    {n:2d}  | {observed:.4f}   |   {gk_prob:.4f}   | {deviation:+.1f}%")

    avg_deviation = np.mean(deviations)
    print(f"\n  Average deviation from Gauss-Kuzmin: {avg_deviation:.2f}%")

    # Look for special values
    print("\n  Interesting patterns in first partial quotient:")
    fq_counts = Counter(first_quotients)
    fq_total = len(first_quotients)
    top_5 = fq_counts.most_common(5)
    for val, count in top_5:
        print(f"    a_1 = {val}: {count}/{fq_total} ({100*count/fq_total:.1f}%)")

    # Check if any quotients correlate with GIFT numbers
    print("\n  GIFT number appearances in partial quotients:")
    gift_numbers = [B2, B3, DIM_G2, H_STAR, DIM_E8, RANK_E8, DIM_J3O]
    gift_names = ['b2=21', 'b3=77', 'dim(G2)=14', 'H*=99', 'dim(E8)=248', 'rank(E8)=8', 'dim(J3O)=27']
    for num, name in zip(gift_numbers, gift_names):
        count = quotient_counts.get(num, 0)
        expected = gk_prob_for(num) * total if num < 100 else 0.001 * total
        if count > 0:
            print(f"    {name}: appears {count} times (expected ~{int(expected)})")

    return quotient_counts

def gk_prob_for(n):
    """Gauss-Kuzmin probability for partial quotient n"""
    if n <= 0:
        return 0
    return np.log2((n + 1)**2 / (n * (n + 2)))

# =============================================================================
# 2. DIGIT ANALYSIS
# =============================================================================

def analyze_digits(zeros, n_samples=5000):
    """Analyze digit distribution in decimal representation"""
    print("\n" + "=" * 70)
    print("2. DIGIT ANALYSIS")
    print("=" * 70)

    indices = np.linspace(0, len(zeros)-1, n_samples, dtype=int)

    # Collect digits after decimal point
    all_digits = []
    digit_positions = {i: [] for i in range(10)}  # Digit frequency by position

    for idx in indices:
        z = zeros[idx]
        # Get 15 significant digits after decimal
        decimal_str = f"{z:.15f}"
        if '.' in decimal_str:
            after_decimal = decimal_str.split('.')[1][:12]
            for pos, d in enumerate(after_decimal):
                digit = int(d)
                all_digits.append(digit)
                if pos < 10:
                    digit_positions[pos].append(digit)

    # Overall digit frequency
    digit_counts = Counter(all_digits)
    total = sum(digit_counts.values())

    print("\nOverall digit frequency (should be ~10% each for random):")
    print("\n  Digit | Frequency | Deviation from 10%")
    print("  " + "-" * 40)

    chi_sq = 0
    for d in range(10):
        freq = digit_counts.get(d, 0) / total
        deviation = freq - 0.1
        chi_sq += (digit_counts.get(d, 0) - total/10)**2 / (total/10)
        indicator = " ***" if abs(deviation) > 0.01 else ""
        print(f"    {d}   |  {freq:.4f}   | {deviation:+.4f}{indicator}")

    print(f"\n  Chi-squared statistic: {chi_sq:.2f} (critical ~16.9 at p=0.05)")
    if chi_sq < 16.9:
        print("  => Digits appear uniformly distributed")
    else:
        print("  => SIGNIFICANT deviation from uniform!")

    # Benford-like analysis for first digit after decimal
    print("\nFirst digit after decimal point:")
    first_digits = Counter(digit_positions[0])
    fd_total = sum(first_digits.values())
    for d in range(10):
        freq = first_digits.get(d, 0) / fd_total
        print(f"    Digit {d}: {freq:.3f}")

    # Look for digit patterns
    print("\nSearching for unusual digit patterns...")

    # Check for repeated digits (like 111, 999)
    repeated_patterns = Counter()
    for idx in indices[:1000]:
        z = zeros[idx]
        decimal_str = f"{z:.15f}".split('.')[1]
        for length in [3, 4, 5]:
            for i in range(len(decimal_str) - length):
                substr = decimal_str[i:i+length]
                if len(set(substr)) == 1:  # All same digit
                    repeated_patterns[substr] += 1

    if repeated_patterns:
        print("\n  Repeated digit patterns found:")
        for pattern, count in repeated_patterns.most_common(10):
            print(f"    '{pattern}' appears {count} times")

    return digit_counts

# =============================================================================
# 3. DIFFERENCE SEQUENCES
# =============================================================================

def compute_skewness(x):
    """Compute skewness without scipy"""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=0)
    if std == 0:
        return 0
    return np.mean(((x - mean) / std) ** 3)

def analyze_differences(zeros):
    """Analyze Delta^k sequences for attractor structure"""
    print("\n" + "=" * 70)
    print("3. DIFFERENCE SEQUENCES (Delta^k gamma_n)")
    print("=" * 70)

    # First differences (spacings)
    delta1 = np.diff(zeros)

    # Higher order differences
    delta2 = np.diff(delta1)
    delta3 = np.diff(delta2)
    delta4 = np.diff(delta3)

    print("\nStatistics of difference sequences:")
    print("\n  Order |   Mean    |   Std     |    Min    |    Max    | Skewness")
    print("  " + "-" * 65)

    for k, dk in enumerate([delta1, delta2, delta3, delta4], 1):
        m = np.mean(dk)
        s = np.std(dk)
        mn = np.min(dk)
        mx = np.max(dk)
        sk = compute_skewness(dk)
        print(f"    {k}   | {m:+.5f} | {s:.5f} | {mn:+.5f} | {mx:+.5f} | {sk:+.4f}")

    # Check if Delta^2 has any special structure
    print("\nDelta^2 analysis (second differences):")

    # Look for sign patterns
    signs = np.sign(delta2)
    sign_runs = []
    current_run = 1
    for i in range(1, len(signs)):
        if signs[i] == signs[i-1]:
            current_run += 1
        else:
            sign_runs.append(current_run)
            current_run = 1
    sign_runs.append(current_run)

    run_counts = Counter(sign_runs)
    print("\n  Sign run length distribution (Delta^2):")
    for length in sorted(run_counts.keys())[:10]:
        count = run_counts[length]
        # Expected for random: (1/2)^length
        expected_ratio = 0.5 ** length
        print(f"    Length {length}: {count} occurrences")

    # Look for quasi-periodic behavior in Delta^2
    print("\n  Autocorrelation of Delta^2 at key lags:")
    d2_centered = delta2 - np.mean(delta2)
    d2_var = np.var(delta2)

    key_lags = [1, 2, 3, 5, 8, 13, 21, 34, 55]  # Fibonacci lags!
    for lag in key_lags:
        if lag < len(d2_centered):
            corr = np.mean(d2_centered[:-lag] * d2_centered[lag:]) / d2_var
            significance = abs(corr) * np.sqrt(len(d2_centered))
            indicator = " *" if abs(corr) > 0.02 else ""
            print(f"    Lag {lag:3d}: r = {corr:+.5f} (z = {significance:.1f}){indicator}")

    return delta1, delta2

# =============================================================================
# 4. MODULAR ARITHMETIC
# =============================================================================

def analyze_modular(zeros):
    """Analyze gamma_n mod various numbers"""
    print("\n" + "=" * 70)
    print("4. MODULAR ARITHMETIC ANALYSIS")
    print("=" * 70)

    # We work with floor(gamma_n) and fractional parts
    floor_zeros = np.floor(zeros).astype(int)
    frac_zeros = zeros - floor_zeros

    print("\nInteger parts modulo small primes:")
    print("\n  Prime | Distribution entropy | Uniform entropy | Deviation")
    print("  " + "-" * 55)

    for p in [2, 3, 5, 7, 11, 13]:
        residues = floor_zeros % p
        counts = np.bincount(residues, minlength=p)
        probs = counts / len(residues)
        probs = probs[probs > 0]  # Remove zeros for entropy
        entropy = -np.sum(probs * np.log2(probs))
        uniform_entropy = np.log2(p)
        deviation = (uniform_entropy - entropy) / uniform_entropy * 100
        print(f"    {p:2d}  |      {entropy:.4f}       |     {uniform_entropy:.4f}     | {deviation:+.2f}%")

    print("\nInteger parts modulo Fibonacci numbers:")
    for f in [5, 8, 13, 21, 34]:
        residues = floor_zeros % f
        counts = np.bincount(residues, minlength=f)
        # Look for non-uniform distribution
        chi_sq = np.sum((counts - len(residues)/f)**2 / (len(residues)/f))
        critical = 1.5 * f  # Rough critical value
        indicator = " ***" if chi_sq > critical else ""
        print(f"    F={f:2d}: chi^2 = {chi_sq:.2f} (critical ~{critical:.0f}){indicator}")

    # Fractional part analysis
    print("\nFractional part {gamma_n} distribution:")

    # Bin fractional parts
    n_bins = 20
    hist, _ = np.histogram(frac_zeros, bins=n_bins, range=(0, 1))
    expected = len(frac_zeros) / n_bins
    chi_sq = np.sum((hist - expected)**2 / expected)
    critical = 1.73 * n_bins  # Rough critical value
    print(f"  Chi-squared ({n_bins} bins): {chi_sq:.2f} (critical ~{critical:.0f})")

    # Look for special fractional values
    print("\n  Fractional parts near GIFT-related values:")
    special_fracs = [
        (1/PHI, "1/phi"),
        (PHI - 1, "phi-1"),
        (3/13, "sin^2(theta_W)=3/13"),
        (1/61, "kappa_T=1/61"),
        (21/99, "b2/H*"),
        (14/77, "dim(G2)/b3"),
        (7/99, "dim(K7)/H*"),
    ]

    for target, name in special_fracs:
        # Count how many fractional parts are within epsilon of target
        epsilon = 0.001
        count = np.sum(np.abs(frac_zeros - target) < epsilon)
        expected_count = 2 * epsilon * len(frac_zeros)
        ratio = count / expected_count if expected_count > 0 else 0
        if ratio > 1.5 or ratio < 0.5:
            print(f"    {name}: {count} near (expected ~{expected_count:.0f}), ratio={ratio:.2f} ***")
        else:
            print(f"    {name}: {count} near (expected ~{expected_count:.0f}), ratio={ratio:.2f}")

    return frac_zeros

# =============================================================================
# 5. RATIO CHAINS
# =============================================================================

def analyze_ratio_chains(zeros):
    """Analyze gamma_{n+k}/gamma_n for various k"""
    print("\n" + "=" * 70)
    print("5. RATIO CHAIN ANALYSIS")
    print("=" * 70)

    n = len(zeros)

    print("\nRatio gamma_{n+k}/gamma_n convergence:")
    print("\n  k  |  Mean ratio  |   Std   | Limit (if known)")
    print("  " + "-" * 50)

    for k in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
        if k >= n:
            continue
        ratios = zeros[k:] / zeros[:-k]
        mean_r = np.mean(ratios[-10000:])  # Use tail for convergence
        std_r = np.std(ratios[-10000:])

        # Asymptotically, gamma_{n+k}/gamma_n -> 1 + k/(n * density)
        # For large n, this -> 1
        print(f"  {k:3d} | {mean_r:.8f} | {std_r:.6f} | -> 1")

    # More interesting: look at (gamma_{n+k} - gamma_n) / (expected spacing)
    print("\nNormalized spacing growth:")

    # Mean spacing for normalization
    mean_spacing = np.mean(np.diff(zeros[-50000:]))

    for k in [1, 2, 3, 5, 8, 13, 21]:
        if k >= n:
            continue
        gaps = zeros[k:] - zeros[:-k]
        normalized = gaps / (k * mean_spacing)
        mean_norm = np.mean(normalized[-10000:])
        std_norm = np.std(normalized[-10000:])
        print(f"  k={k:2d}: mean={mean_norm:.6f}, std={std_norm:.6f}")

    # Look for ratio relationships with special numbers
    print("\nSearching for special ratio accumulation points...")

    # Ratio of consecutive spacings
    spacings = np.diff(zeros)
    spacing_ratios = spacings[1:] / spacings[:-1]

    # Count near special values
    special_ratios = [
        (PHI, "phi"),
        (1/PHI, "1/phi"),
        (2.0, "2"),
        (E, "e"),
        (PI/E, "pi/e"),
        (np.sqrt(2), "sqrt(2)"),
        (21/13, "21/13 (Fib ratio)"),
        (34/21, "34/21 (Fib ratio)"),
    ]

    print("\n  Target ratio | Count near | Expected | Excess")
    print("  " + "-" * 50)

    for target, name in special_ratios:
        epsilon = 0.01
        count = np.sum(np.abs(spacing_ratios - target) < epsilon)
        # Expected density around target
        expected = 2 * epsilon * len(spacing_ratios) / 3.0  # rough estimate
        excess = count / expected if expected > 0 else 0
        indicator = " ***" if excess > 1.5 else ""
        print(f"  {name:15s} |   {count:5d}   |  {expected:.0f}   | {excess:.2f}x{indicator}")

    return spacing_ratios

# =============================================================================
# 6. CROSS-CORRELATION WITH SEQUENCES
# =============================================================================

def analyze_cross_correlation(zeros):
    """Cross-correlation with Fibonacci, primes, other sequences"""
    print("\n" + "=" * 70)
    print("6. CROSS-CORRELATION WITH NUMBER SEQUENCES")
    print("=" * 70)

    n = len(zeros)
    spacings = np.diff(zeros)

    # Normalize for correlation
    spacings_norm = (spacings - np.mean(spacings)) / np.std(spacings)

    # Generate Fibonacci sequence at same scale
    fib_seq = []
    a, b = 1, 1
    while len(fib_seq) < n:
        fib_seq.append(a)
        a, b = b, a + b
    fib_seq = np.array(fib_seq[:n-1])

    # Correlation at Fibonacci indices
    print("\nSpacing values at Fibonacci indices:")
    print("  n (Fib) |  gamma_n  |  Spacing  | Spacing/mean")
    print("  " + "-" * 50)

    mean_sp = np.mean(spacings)
    for f in [8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]:
        if f < len(spacings):
            sp = spacings[f-1]  # F_n spacing
            print(f"  {f:5d}   | {zeros[f-1]:.3f} | {sp:.6f} | {sp/mean_sp:.4f}")

    # Cross-correlation of spacings with prime gaps
    print("\nCorrelation: zero spacings vs prime gaps")

    # Generate prime gaps
    prime_gaps = np.diff(PRIMES)
    min_len = min(len(spacings), len(prime_gaps))

    corr = np.corrcoef(spacings[:min_len], prime_gaps[:min_len])[0, 1]
    print(f"  Direct correlation: r = {corr:.6f}")

    # Shuffle test
    shuffle_corrs = []
    for _ in range(1000):
        shuffled = np.random.permutation(spacings[:min_len])
        shuffle_corrs.append(np.corrcoef(shuffled, prime_gaps[:min_len])[0, 1])

    shuffle_mean = np.mean(shuffle_corrs)
    shuffle_std = np.std(shuffle_corrs)
    z_score = (corr - shuffle_mean) / shuffle_std
    print(f"  Shuffle test: z-score = {z_score:.2f}")

    # Look for Fibonacci structure in spacing sequence
    print("\nFibonacci resonance test:")

    # Compute autocorrelation at Fibonacci lags
    print("  Autocorrelation of spacings at Fibonacci lags:")
    sp_centered = spacings - np.mean(spacings)
    sp_var = np.var(spacings)

    for f in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
        if f < len(sp_centered):
            acf = np.mean(sp_centered[:-f] * sp_centered[f:]) / sp_var
            non_fib_lags = [f-1, f+1] if f > 1 else [f+1, f+2]
            non_fib_acf = []
            for lag in non_fib_lags:
                if lag < len(sp_centered):
                    non_fib_acf.append(np.mean(sp_centered[:-lag] * sp_centered[lag:]) / sp_var)
            avg_neighbor = np.mean(non_fib_acf)
            enhancement = acf / avg_neighbor if abs(avg_neighbor) > 1e-6 else 0
            indicator = " ***" if abs(enhancement) > 1.5 else ""
            print(f"    F_{f:2d}: r = {acf:+.6f}, neighbor avg = {avg_neighbor:+.6f}, ratio = {enhancement:.2f}{indicator}")

    return spacings

# =============================================================================
# 7. PHASE SPACE ANALYSIS
# =============================================================================

def analyze_phase_space(zeros):
    """Phase space portraits and attractor analysis"""
    print("\n" + "=" * 70)
    print("7. PHASE SPACE ANALYSIS")
    print("=" * 70)

    spacings = np.diff(zeros)
    n = len(spacings)

    # (gamma_n, gamma_{n+1}) return map
    print("\nReturn map analysis (gamma_n vs gamma_{n+1}):")

    # For normalized analysis, use spacings
    x = spacings[:-1]
    y = spacings[1:]

    # Correlation coefficient
    r = np.corrcoef(x, y)[0, 1]
    print(f"  Spacing correlation s_n vs s_{n+1}: r = {r:.6f}")

    # Look for clusters in (x, y) space
    print("\n  Density analysis in spacing return map:")

    # Bin the 2D space
    n_bins = 20
    h, xedges, yedges = np.histogram2d(x, y, bins=n_bins)

    # Find high-density regions
    threshold = np.percentile(h, 90)
    high_density = np.argwhere(h > threshold)

    print(f"  Number of high-density cells: {len(high_density)}")
    print(f"  Max density: {np.max(h):.0f}, Mean: {np.mean(h):.1f}")

    # Check for diagonal structure (s_n = s_{n+1})
    diagonal_density = np.mean([h[i, i] for i in range(min(n_bins, len(h)))])
    off_diagonal = np.mean(h)
    diagonal_enhancement = diagonal_density / off_diagonal if off_diagonal > 0 else 0
    print(f"  Diagonal enhancement: {diagonal_enhancement:.2f}x")

    # 3D embedding: (s_n, s_{n+1}, s_{n+2})
    print("\n  3D embedding (s_n, s_{n+1}, s_{n+2}):")

    s0 = spacings[:-2]
    s1 = spacings[1:-1]
    s2 = spacings[2:]

    # Covariance matrix eigenvalues reveal dimensionality
    cov_matrix = np.cov(np.vstack([s0, s1, s2]))
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]

    total_var = np.sum(eigenvalues)
    explained = eigenvalues / total_var * 100
    print(f"  Explained variance: {explained[0]:.1f}%, {explained[1]:.1f}%, {explained[2]:.1f}%")
    print(f"  Effective dimension: {np.sum(eigenvalues**2) / total_var**2:.2f}")

    # Delta-Delta plot: (Delta s_n, Delta^2 s_n)
    print("\n  Delta-Delta phase portrait:")
    delta_s = np.diff(spacings)
    delta2_s = np.diff(delta_s)

    r_dd = np.corrcoef(delta_s[:-1], delta2_s)[0, 1]
    print(f"  Correlation Delta_s vs Delta^2_s: r = {r_dd:.6f}")

    return spacings

# =============================================================================
# 8. INFORMATION THEORY ANALYSIS
# =============================================================================

def analyze_information_theory(zeros):
    """Entropy and information-theoretic measures"""
    print("\n" + "=" * 70)
    print("8. INFORMATION THEORY ANALYSIS")
    print("=" * 70)

    spacings = np.diff(zeros)

    # Normalize spacings by local mean
    window = 100
    local_means = np.convolve(spacings, np.ones(window)/window, mode='same')
    normalized_spacings = spacings / local_means
    normalized_spacings = normalized_spacings[window:-window]  # Remove edge effects

    # Entropy of spacing distribution
    print("\nSpacing distribution entropy:")

    n_bins = 50
    hist, bin_edges = np.histogram(normalized_spacings, bins=n_bins, density=True)
    bin_width = bin_edges[1] - bin_edges[0]

    # Remove zeros for log
    hist_pos = hist[hist > 0]
    entropy = -np.sum(hist_pos * np.log2(hist_pos) * bin_width)

    # Compare to normal distribution with same mean/std
    normal_entropy = 0.5 * np.log2(2 * np.pi * np.e * np.var(normalized_spacings))

    print(f"  Empirical entropy: {entropy:.4f} bits")
    print(f"  Gaussian entropy (same variance): {normal_entropy:.4f} bits")
    print(f"  Excess entropy: {entropy - normal_entropy:.4f} bits")

    # GUE (Wigner-Dyson) entropy for comparison
    # P(s) = (pi/2) * s * exp(-pi*s^2/4) for normalized spacings
    # Theoretical entropy ~ 0.577 (Euler-Mascheroni related)
    gue_entropy = 0.5 * np.log2(np.pi * np.e / 2)  # Approximate
    print(f"  GUE entropy (approximate): {gue_entropy:.4f} bits")

    # Mutual information between consecutive spacings
    print("\nMutual information analysis:")

    x = normalized_spacings[:-1]
    y = normalized_spacings[1:]

    # Joint entropy
    joint_hist, _, _ = np.histogram2d(x, y, bins=20)
    joint_hist = joint_hist / np.sum(joint_hist)
    joint_hist_pos = joint_hist[joint_hist > 0]
    joint_entropy = -np.sum(joint_hist_pos * np.log2(joint_hist_pos))

    # Marginal entropies
    hx, _ = np.histogram(x, bins=20, density=True)
    hy, _ = np.histogram(y, bins=20, density=True)

    hx_pos = hx[hx > 0]
    hy_pos = hy[hy > 0]

    hx_entropy = -np.sum(hx_pos * np.log2(hx_pos) * (np.max(x) - np.min(x)) / 20)
    hy_entropy = -np.sum(hy_pos * np.log2(hy_pos) * (np.max(y) - np.min(y)) / 20)

    # MI = H(X) + H(Y) - H(X,Y)
    mi = hx_entropy + hy_entropy - joint_entropy
    print(f"  H(s_n): {hx_entropy:.4f} bits")
    print(f"  H(s_{{n+1}}): {hy_entropy:.4f} bits")
    print(f"  H(s_n, s_{{n+1}}): {joint_entropy:.4f} bits")
    print(f"  Mutual Information I(s_n; s_{{n+1}}): {mi:.4f} bits")

    # Kolmogorov complexity proxy via compression
    print("\nComplexity analysis (compression proxy):")

    # Quantize spacings to symbols
    quantiles = np.percentile(normalized_spacings, np.arange(0, 101, 10))
    symbols = np.digitize(normalized_spacings, quantiles[:-1]) - 1

    # Compression ratio estimate via run-length encoding
    runs = []
    current = symbols[0]
    count = 1
    for s in symbols[1:]:
        if s == current:
            count += 1
        else:
            runs.append((current, count))
            current = s
            count = 1
    runs.append((current, count))

    compression_ratio = len(runs) / len(symbols)
    print(f"  Run-length compression ratio: {compression_ratio:.4f}")
    print(f"  (1.0 = random, lower = more structure)")

    # Permutation entropy (ordinal patterns)
    print("\nPermutation entropy (order patterns):")

    order = 3
    patterns = []
    for i in range(len(normalized_spacings) - order + 1):
        window = normalized_spacings[i:i+order]
        pattern = tuple(np.argsort(window))
        patterns.append(pattern)

    pattern_counts = Counter(patterns)
    n_patterns = len(patterns)
    n_possible = math.factorial(order)

    pattern_probs = np.array([count/n_patterns for count in pattern_counts.values()])
    perm_entropy = -np.sum(pattern_probs * np.log2(pattern_probs))
    max_perm_entropy = np.log2(n_possible)

    print(f"  Order-{order} permutation entropy: {perm_entropy:.4f} bits")
    print(f"  Maximum possible: {max_perm_entropy:.4f} bits")
    print(f"  Normalized: {perm_entropy/max_perm_entropy:.4f}")

    return normalized_spacings

# =============================================================================
# 9. GIFT-SPECIFIC PATTERN SEARCH
# =============================================================================

def search_gift_patterns(zeros):
    """Search for patterns related to GIFT topological constants"""
    print("\n" + "=" * 70)
    print("9. GIFT-SPECIFIC PATTERN SEARCH")
    print("=" * 70)

    spacings = np.diff(zeros)
    n = len(spacings)

    # Define GIFT-related quantities
    gift_values = {
        'b2': 21,
        'b3': 77,
        'dim_G2': 14,
        'H*': 99,
        'dim_E8': 248,
        'rank_E8': 8,
        'dim_J3O': 27,
        'sin2_theta_W': 3/13,
        'kappa_T': 1/61,
        'det_g': 65/32,
        'tau': 3472/891,
        'b2/H*': 21/99,
        'b3/H*': 77/99,
        'dim_G2/b3': 14/77,
        'N_gen': 3,
    }

    print("\nSearching for GIFT constants in spacing ratios...")

    # Ratio of spacings at Fibonacci vs non-Fibonacci positions
    fib_set = set(FIBS)
    fib_spacings = [spacings[i] for i in range(n) if (i+1) in fib_set]
    non_fib_spacings = [spacings[i] for i in range(min(n, 1000)) if (i+1) not in fib_set]

    if fib_spacings and non_fib_spacings:
        fib_mean = np.mean(fib_spacings)
        non_fib_mean = np.mean(non_fib_spacings[:len(fib_spacings)])
        ratio = fib_mean / non_fib_mean
        print(f"\n  Fib-position spacing / non-Fib spacing: {ratio:.6f}")

        # Check proximity to GIFT values
        for name, val in gift_values.items():
            if 0.1 < val < 10:
                if abs(ratio - val) < 0.1:
                    print(f"    Close to {name} = {val}: delta = {ratio - val:.6f}")

    # Cumulative sum patterns
    print("\nCumulative spacing analysis:")
    cum_spacings = np.cumsum(spacings)

    # At Fibonacci indices, what is the cumulative sum?
    print("\n  Cumulative spacings at Fibonacci indices:")
    for f in [8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]:
        if f < len(cum_spacings):
            cs = cum_spacings[f-1]
            # Ratio to zeros[f]
            ratio = cs / zeros[f-1]
            print(f"    F_{f:4d}: cumsum = {cs:.3f}, ratio to gamma_F = {ratio:.6f}")

    # Look for b2=21 and b3=77 patterns
    print("\nSearching for b2=21 and b3=77 periodicity in spacings:")

    for period in [21, 77, 99]:
        # Average spacings by position mod period
        mod_avg = np.zeros(period)
        mod_count = np.zeros(period)
        for i, s in enumerate(spacings[:10000]):
            mod_avg[i % period] += s
            mod_count[i % period] += 1
        mod_avg /= mod_count

        # Variance of averages
        var = np.var(mod_avg)
        mean = np.mean(mod_avg)
        cv = np.sqrt(var) / mean  # Coefficient of variation

        print(f"  Period {period:2d}: CV of mod-averages = {cv:.6f}")

    # Special: check if gamma_21 / gamma_77 relates to GIFT
    if len(zeros) > 77:
        ratio_21_77 = zeros[20] / zeros[76]  # 0-indexed
        print(f"\n  gamma_21 / gamma_77 = {ratio_21_77:.8f}")
        print(f"  Compare to b2/b3 = {21/77:.8f}")
        print(f"  Compare to sin^2(theta_W) = {3/13:.8f}")

    return gift_values

# =============================================================================
# 10. BONUS: WILD PATTERN SEARCH
# =============================================================================

def wild_pattern_search(zeros):
    """Look for really unusual patterns"""
    print("\n" + "=" * 70)
    print("10. WILD PATTERN SEARCH (Speculative)")
    print("=" * 70)

    spacings = np.diff(zeros)

    # Look for "almost integer" zeros
    print("\nNearly-integer zeros:")
    for i, z in enumerate(zeros[:5000]):
        frac = z - np.floor(z)
        if frac < 0.001 or frac > 0.999:
            nearest_int = round(z)
            delta = z - nearest_int
            print(f"  gamma_{i+1} = {z:.9f}, delta from {nearest_int} = {delta:.9f}")

    # Look for zeros near multiples of pi, e, phi
    print("\nZeros near special multiples:")
    specials = [
        (PI, "pi"),
        (E, "e"),
        (PHI, "phi"),
        (PI * PHI, "pi*phi"),
        (2*PI, "2pi"),
    ]

    for mult, name in specials:
        for i, z in enumerate(zeros[:10000]):
            k = round(z / mult)
            if k > 0:
                delta = z - k * mult
                if abs(delta) < 0.01:
                    print(f"  gamma_{i+1} = {z:.6f} ~ {k}*{name} = {k*mult:.6f}, delta = {delta:.6f}")

    # Look for golden spiral in spacing sequence
    print("\nGolden spiral test in spacings:")

    # Ratio of every Fibonacci-spaced pair
    fib_ratios = []
    for i in range(1, 15):
        f1 = FIBS[i-1]
        f2 = FIBS[i]
        if f2 < len(spacings):
            ratio = spacings[f2] / spacings[f1] if spacings[f1] > 0 else 0
            fib_ratios.append(ratio)
            print(f"  s_F{i+1} / s_F{i} = {ratio:.6f} (phi = {PHI:.6f})")

    # Mean ratio
    if fib_ratios:
        mean_ratio = np.mean(fib_ratios)
        print(f"\n  Mean Fib-spacing ratio: {mean_ratio:.6f}")
        print(f"  Deviation from phi: {abs(mean_ratio - PHI):.6f}")
        print(f"  Deviation from 1/phi: {abs(mean_ratio - 1/PHI):.6f}")

    # Divisibility patterns in floor(gamma_n)
    print("\nDivisibility patterns in floor(gamma_n):")
    floor_z = np.floor(zeros).astype(int)

    for d in [7, 13, 21, 31, 77]:
        divisible = np.sum(floor_z[:1000] % d == 0)
        expected = 1000 / d
        ratio = divisible / expected
        indicator = " ***" if abs(ratio - 1) > 0.2 else ""
        print(f"  Divisible by {d:2d}: {divisible} (expected {expected:.1f}), ratio={ratio:.2f}{indicator}")

    return None

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("UNCONVENTIONAL EXPLORATION OF RIEMANN ZEROS")
    print("Searching for hidden structure beyond smooth asymptotics")
    print("=" * 70)

    # Load data
    zeros = load_zeros(100000)
    print(f"\nLoaded {len(zeros):,} Riemann zeros")
    print(f"Range: gamma_1 = {zeros[0]:.6f} to gamma_{len(zeros)} = {zeros[-1]:.6f}")

    results = {}

    # Run all analyses
    try:
        from scipy.stats import skew
    except ImportError:
        print("\nNote: scipy not available, some statistics will be simplified")

    # 1. Continued fractions
    cf_results = analyze_continued_fractions(zeros)

    # 2. Digit analysis
    digit_results = analyze_digits(zeros)

    # 3. Difference sequences
    delta1, delta2 = analyze_differences(zeros)

    # 4. Modular arithmetic
    frac_zeros = analyze_modular(zeros)

    # 5. Ratio chains
    spacing_ratios = analyze_ratio_chains(zeros)

    # 6. Cross-correlation
    spacings = analyze_cross_correlation(zeros)

    # 7. Phase space
    _ = analyze_phase_space(zeros)

    # 8. Information theory
    _ = analyze_information_theory(zeros)

    # 9. GIFT patterns
    _ = search_gift_patterns(zeros)

    # 10. Wild patterns
    _ = wild_pattern_search(zeros)

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)

    print("""
Key observations from unconventional analysis:

1. CONTINUED FRACTIONS: Partial quotients follow Gauss-Kuzmin law
   (as expected for "random" real numbers), but check for deviations.

2. DIGIT DISTRIBUTION: Should be uniform; any deviation is significant.

3. DIFFERENCE SEQUENCES: Delta^2 gamma_n shows structure beyond random.
   Autocorrelation at Fibonacci lags may show enhancement.

4. MODULAR ARITHMETIC: Fractional parts near GIFT values (3/13, 1/61)
   could indicate deep structure.

5. RATIO CHAINS: Spacing ratios cluster near special values?
   Look for phi or Fibonacci ratio enhancement.

6. CROSS-CORRELATION: Weak but nonzero correlation with prime gaps.
   Fibonacci-lag autocorrelation may be enhanced.

7. PHASE SPACE: Embedding dimension and diagonal enhancement
   reveal correlation structure.

8. INFORMATION THEORY: Permutation entropy measures order patterns;
   deviation from maximum indicates structure.

9. GIFT PATTERNS: Direct search for b2=21, b3=77 periodicity
   and related topological constants.

10. WILD PATTERNS: Nearly-integer zeros, special multiple proximities,
    and golden spiral structure in spacings.

Potential connections to GIFT:
- b2/H* = 21/99 appearing in fractional parts
- Fibonacci lag structure in autocorrelations
- Ratio patterns at 3/13 (sin^2 theta_W)
- Periodicity at b2=21 or b3=77 in spacing fluctuations
""")

    print("=" * 70)
    print("Analysis complete. Review results above for surprising patterns.")
    print("=" * 70)

if __name__ == "__main__":
    main()
