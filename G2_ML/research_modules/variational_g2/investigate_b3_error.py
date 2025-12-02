#!/usr/bin/env python3
"""
Investigation: Why b3 = 76 instead of 77?

Hypothesis: The S¹ modulation loses one mode because:
- sin(πλ) = 0 at BOTH λ=0 AND λ=1
- This creates a "dead" mode that doesn't contribute variance

Test: Try alternative modulations and check rank.
"""

import numpy as np
import torch

# Simulate the construction
N = 10000
np.random.seed(42)

# Coordinates in [-1, 1]
coords = np.random.rand(N, 7) * 2 - 1

# λ normalized to [0, 1]
lam = (coords[:, 0] + 1) / 2  # Map from [-1,1] to [0,1]

# 35 local modes (random for simulation)
phi = np.random.randn(N, 35) * 0.1
phi[:, [0, 9, 14, 20, 23, 27, 28]] = 1.05  # G2 structure

# 21 metric 2-forms (random for simulation)
metric_2forms = np.random.randn(N, 21) * 0.5

print("="*60)
print("INVESTIGATION: Missing Mode in b3 = 76")
print("="*60)

# === Test 1: Original sin/cos modulation ===
print("\n[1] Original: sin(pi*lam) / cos(pi*lam)")
s1_plus_orig = np.sin(np.pi * lam)[:, None]
s1_minus_orig = np.cos(np.pi * lam)[:, None]

global_plus_orig = metric_2forms * s1_plus_orig
global_minus_orig = metric_2forms * s1_minus_orig

basis_orig = np.hstack([phi, global_plus_orig, global_minus_orig])
print(f"  Basis shape: {basis_orig.shape}")

# Check rank
U, S, Vh = np.linalg.svd(basis_orig - basis_orig.mean(axis=0), full_matrices=False)
rank_orig = np.sum(S > 1e-6 * S[0])
print(f"  Effective rank: {rank_orig}")
print(f"  Smallest 5 singular values: {S[-5:]}")

# Issue: sin(π*0) = 0 and sin(π*1) = 0!
print(f"\n  Problem: sin(pi*0) = {np.sin(0):.4f}, sin(pi*1) = {np.sin(np.pi):.6f}")
print(f"  The sin term vanishes at BOTH endpoints!")

# === Test 2: Shifted sin/cos ===
print("\n[2] Shifted: sin(pi*(lam+0.5)) / cos(pi*(lam+0.5))")
s1_plus_shift = np.sin(np.pi * (lam + 0.5))[:, None]
s1_minus_shift = np.cos(np.pi * (lam + 0.5))[:, None]

global_plus_shift = metric_2forms * s1_plus_shift
global_minus_shift = metric_2forms * s1_minus_shift

basis_shift = np.hstack([phi, global_plus_shift, global_minus_shift])
U, S, Vh = np.linalg.svd(basis_shift - basis_shift.mean(axis=0), full_matrices=False)
rank_shift = np.sum(S > 1e-6 * S[0])
print(f"  Effective rank: {rank_shift}")

# === Test 3: Linear modulation (1-lam) / lam ===
print("\n[3] Linear: (1-lam) for X+, lam for X-")
s1_plus_lin = (1 - lam)[:, None]
s1_minus_lin = lam[:, None]

global_plus_lin = metric_2forms * s1_plus_lin
global_minus_lin = metric_2forms * s1_minus_lin

basis_lin = np.hstack([phi, global_plus_lin, global_minus_lin])
U, S, Vh = np.linalg.svd(basis_lin - basis_lin.mean(axis=0), full_matrices=False)
rank_lin = np.sum(S > 1e-6 * S[0])
print(f"  Effective rank: {rank_lin}")

# === Test 4: Half-period sin/cos ===
print("\n[4] Half-period: sin(pi*lam/2) / cos(pi*lam/2)")
s1_plus_half = np.sin(np.pi * lam / 2)[:, None]
s1_minus_half = np.cos(np.pi * lam / 2)[:, None]

global_plus_half = metric_2forms * s1_plus_half
global_minus_half = metric_2forms * s1_minus_half

basis_half = np.hstack([phi, global_plus_half, global_minus_half])
U, S, Vh = np.linalg.svd(basis_half - basis_half.mean(axis=0), full_matrices=False)
rank_half = np.sum(S > 1e-6 * S[0])
print(f"  Effective rank: {rank_half}")

# === Test 5: Different frequencies for X+ and X- ===
print("\n[5] Mixed frequencies: sin(pi*lam) / sin(2*pi*lam)")
s1_plus_mix = np.sin(np.pi * lam)[:, None]
s1_minus_mix = np.sin(2 * np.pi * lam)[:, None]

global_plus_mix = metric_2forms * s1_plus_mix
global_minus_mix = metric_2forms * s1_minus_mix

basis_mix = np.hstack([phi, global_plus_mix, global_minus_mix])
U, S, Vh = np.linalg.svd(basis_mix - basis_mix.mean(axis=0), full_matrices=False)
rank_mix = np.sum(S > 1e-6 * S[0])
print(f"  Effective rank: {rank_mix}")

# === Summary ===
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
Modulation             | Rank | Target
-----------------------|------|-------
sin(pi*lam)/cos        | {rank_orig:4d} |   77
sin(pi*(lam+0.5))/cos  | {rank_shift:4d} |   77
(1-lam)/lam linear     | {rank_lin:4d} |   77
sin(pi*lam/2)/cos      | {rank_half:4d} |   77
sin(pi*lam)/sin(2pi)   | {rank_mix:4d} |   77
""")

# Check which one gives 77
best_rank = max(rank_orig, rank_shift, rank_lin, rank_half, rank_mix)
print(f"Best rank achieved: {best_rank}")

if best_rank == 77:
    print("SUCCESS: One of the alternatives gives full rank 77!")
else:
    print(f"Gap: {77 - best_rank} mode(s) still missing")
    print("\nPossible causes:")
    print("  - Linear dependency in metric_2forms (21 modes not fully independent)")
    print("  - Overlap between local (35) and global (42) modes")
    print("  - Numerical precision in the PINN output")
