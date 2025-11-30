#!/usr/bin/env python3
"""
Weighted Yukawa Analysis for v2.0 TCS Models

Tests if region-weighted Yukawa integration can reveal more than 35 active modes.
The idea: boost the neck region where left/right modes overlap, preventing
the exact cancellation that kills global modes in uniform integration.

Usage:
    python analyze_weighted_yukawa.py [--sigma 0.3] [--offset 0.2]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# Configuration (must match v2.0 training)
# ============================================================================

class Config:
    dim = 7
    b2_K7 = 21
    b3_K7 = 77
    b3_local = 35
    b3_global = 42
    n_left = 14
    n_right = 14
    n_neck = 14
    hidden_dim = 256
    n_layers = 4
    lambda_L = -1.0
    lambda_R = +1.0
    lambda_neck = 0.0
    sigma_transition = 0.15
    sigma_neck = 0.2

config = Config()

# ============================================================================
# TCS Profile Functions
# ============================================================================

def smooth_step(x, x0=0.0, width=0.1):
    w = max(width, 1e-8)
    t = (x - x0) / w
    return torch.sigmoid(5.0 * t)

def left_plateau(lam, config):
    return 1.0 - smooth_step(lam, x0=config.lambda_neck, width=config.sigma_transition)

def right_plateau(lam, config):
    return smooth_step(lam, x0=config.lambda_neck, width=config.sigma_transition)

def neck_bump(lam, config):
    t = (lam - config.lambda_neck) / max(config.sigma_neck, 1e-8)
    return torch.exp(-t * t)

# ============================================================================
# Networks (must match v2.0)
# ============================================================================

class H2Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        in_dim = config.dim
        for _ in range(config.n_layers):
            layers.extend([nn.Linear(in_dim, config.hidden_dim), nn.SiLU()])
            in_dim = config.hidden_dim
        self.features = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(config.hidden_dim, 21) for _ in range(config.b2_K7)])

    def forward(self, x):
        f = self.features(x)
        return torch.stack([h(f) for h in self.heads], dim=1)


class H3TCSNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        in_dim = config.dim
        for _ in range(config.n_layers):
            layers.extend([nn.Linear(in_dim, config.hidden_dim), nn.SiLU()])
            in_dim = config.hidden_dim
        self.features = nn.Sequential(*layers)
        self.local_heads = nn.ModuleList([nn.Linear(config.hidden_dim, 35) for _ in range(config.b3_local)])
        self.left_heads = nn.ModuleList([nn.Linear(config.hidden_dim, 35) for _ in range(config.n_left)])
        self.right_heads = nn.ModuleList([nn.Linear(config.hidden_dim, 35) for _ in range(config.n_right)])
        self.neck_heads = nn.ModuleList([nn.Linear(config.hidden_dim, 35) for _ in range(config.n_neck)])

    def forward(self, x):
        lam = x[:, 0]
        prof_left = left_plateau(lam, self.config)
        prof_right = right_plateau(lam, self.config)
        prof_neck = neck_bump(lam, self.config)
        f = self.features(x)
        local_forms = torch.stack([h(f) for h in self.local_heads], dim=1)
        left_base = torch.stack([h(f) for h in self.left_heads], dim=1)
        left_forms = left_base * prof_left.unsqueeze(-1).unsqueeze(-1)
        right_base = torch.stack([h(f) for h in self.right_heads], dim=1)
        right_forms = right_base * prof_right.unsqueeze(-1).unsqueeze(-1)
        neck_base = torch.stack([h(f) for h in self.neck_heads], dim=1)
        neck_forms = neck_base * prof_neck.unsqueeze(-1).unsqueeze(-1)
        return torch.cat([local_forms, left_forms, right_forms, neck_forms], dim=1)

# ============================================================================
# Yukawa Coefficients
# ============================================================================

def levi_civita_7(indices):
    if len(set(indices)) != 7:
        return 0
    inv = sum(1 for i in range(7) for j in range(i+1, 7) if indices[i] > indices[j])
    return 1 if inv % 2 == 0 else -1

def build_yukawa_coefficients():
    pairs = list(combinations(range(7), 2))
    triples = list(combinations(range(7), 3))
    coeffs = []
    for i1, p1 in enumerate(pairs):
        for i2, p2 in enumerate(pairs):
            if i2 < i1:
                continue
            for i3, t in enumerate(triples):
                all_idx = p1 + p2 + t
                if len(set(all_idx)) != 7:
                    continue
                sign = levi_civita_7(all_idx)
                if sign != 0:
                    coeffs.append((i1, i2, i3, sign))
    return coeffs

YUKAWA_COEFFS = build_yukawa_coefficients()

# ============================================================================
# Region Weighting Functions
# ============================================================================

def region_weight_neck(lam: torch.Tensor, sigma: float = 0.3, offset: float = 0.2) -> torch.Tensor:
    """
    Weight that boosts the neck region.
    w(lambda) = exp(-lambda^2/sigma^2) + offset
    """
    bump = torch.exp(-(lam / sigma) ** 2)
    return bump + offset

def region_weight_asymmetric(lam: torch.Tensor, sigma: float = 0.3, offset: float = 0.2,
                              asymmetry: float = 0.1) -> torch.Tensor:
    """
    Asymmetric weight to break left/right cancellation.
    w(lambda) = exp(-lambda^2/sigma^2) * (1 + asymmetry * lambda) + offset
    """
    bump = torch.exp(-(lam / sigma) ** 2)
    return bump * (1 + asymmetry * lam) + offset

def region_weight_double_bump(lam: torch.Tensor, sigma: float = 0.2, offset: float = 0.1,
                               separation: float = 0.5) -> torch.Tensor:
    """
    Two bumps at the left/right transition zones.
    """
    bump_left = torch.exp(-((lam + separation) / sigma) ** 2)
    bump_right = torch.exp(-((lam - separation) / sigma) ** 2)
    return bump_left + bump_right + offset

# ============================================================================
# Weighted Yukawa Computation
# ============================================================================

def compute_weighted_yukawa(h2_model, h3_model, coords, metric,
                            weight_fn, weight_params: dict,
                            coeffs, n_pts: int = 5000) -> Dict:
    """Compute Yukawa with region weighting."""

    h2_model.eval()
    h3_model.eval()

    n = min(n_pts, coords.shape[0])
    idx = torch.randperm(coords.shape[0])[:n]
    x, g = coords[idx].to(device), metric[idx].to(device)
    lam = x[:, 0]

    # Volume element
    det_g = torch.det(g)
    vol = torch.sqrt(det_g.abs())

    # Region weight
    w = weight_fn(lam, **weight_params)

    # Combined weight
    combined_weight = vol * w
    total_weight = combined_weight.sum()

    with torch.no_grad():
        omega = h2_model(x)
        Phi = h3_model(x)

    Y = torch.zeros(21, 21, 77, device=device)

    for a in range(21):
        omega_a = omega[:, a, :]
        for b in range(a, 21):
            omega_b = omega[:, b, :]
            for c in range(77):
                Phi_c = Phi[:, c, :]
                integral = torch.zeros(n, device=device)
                for i1, i2, i3, sign in coeffs:
                    integral += sign * omega_a[:, i1] * omega_b[:, i2] * Phi_c[:, i3]
                Y[a, b, c] = (integral * combined_weight).sum() / total_weight
                if a != b:
                    Y[b, a, c] = -Y[a, b, c]

    # Gram matrix
    M = torch.einsum('ijk,ijl->kl', Y, Y)

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    idx_sort = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx_sort]

    return {
        'eigenvalues': eigenvalues.cpu().numpy(),
        'Y': Y.cpu().numpy(),
        'M': M.cpu().numpy(),
    }

# ============================================================================
# Spectral Analysis
# ============================================================================

def analyze_spectrum(eigs: np.ndarray, tau_target: float = 3472/891, label: str = "") -> Dict:
    """Analyze eigenvalue spectrum."""

    print(f"\n{'='*60}")
    print(f"SPECTRAL ANALYSIS: {label}")
    print('='*60)

    # Non-zero count
    nonzero = (np.abs(eigs) > 1e-10).sum()

    # Gaps
    gaps = np.abs(np.diff(eigs))
    mean_gap = gaps.mean() if len(gaps) > 0 else 1.0

    print(f"\n[EIGENVALUES]")
    print(f"  Non-zero (>1e-10): {nonzero}")
    print(f"  Top 5: {eigs[:5].round(4)}")

    # Find largest gaps
    gap_order = np.argsort(gaps)[::-1]
    print(f"\n[TOP 5 GAPS]")
    for i in range(min(5, len(gap_order))):
        idx = gap_order[i]
        ratio = gaps[idx] / mean_gap if mean_gap > 0 else 0
        print(f"  #{i+1}: gap {idx}->{idx+1}: {gaps[idx]:.4f} ({ratio:.1f}x mean)")

    # Key positions
    print(f"\n[KEY POSITIONS]")
    for pos in [34, 35, 42, 43]:
        if pos < len(gaps):
            ratio = gaps[pos] / mean_gap if mean_gap > 0 else 0
            print(f"  Gap {pos}->{pos+1}: {gaps[pos]:.6f} ({ratio:.1f}x mean)")

    # Cumulative
    cumsum = np.cumsum(eigs)
    total = eigs.sum() if eigs.sum() > 0 else 1.0

    print(f"\n[CUMULATIVE]")
    for n in [35, 43, 50]:
        if n <= len(eigs):
            pct = 100 * cumsum[n-1] / total
            print(f"  First {n}: {pct:.1f}%")

    # Tau search
    best_n, best_ratio, best_err = 0, 0, float('inf')
    for n in range(20, 55):
        if n < len(eigs):
            visible = cumsum[n-1]
            hidden = total - visible
            if hidden > 1e-8:
                ratio = visible / hidden
                err = 100 * abs(ratio - tau_target) / tau_target
                if err < best_err:
                    best_n, best_ratio, best_err = n, ratio, err

    print(f"\n[TAU]")
    print(f"  Target: {tau_target:.4f}")
    if best_err < float('inf'):
        print(f"  Best: n={best_n}, tau={best_ratio:.4f}, error={best_err:.1f}%")

    # Verdict
    largest_gap_idx = np.argmax(gaps) if len(gaps) > 0 else 0

    return {
        'nonzero': int(nonzero),
        'largest_gap_idx': int(largest_gap_idx),
        'n_visible': int(largest_gap_idx + 1),
        'tau_best_n': int(best_n),
        'tau_estimate': float(best_ratio),
        'tau_error_pct': float(best_err),
    }

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Weighted Yukawa Analysis')
    parser.add_argument('--sigma', type=float, default=0.3, help='Neck bump sigma')
    parser.add_argument('--offset', type=float, default=0.2, help='Base offset')
    parser.add_argument('--asymmetry', type=float, default=0.0, help='Left/right asymmetry')
    parser.add_argument('--model-path', type=str, default='models.pt', help='Path to models')
    parser.add_argument('--data-path', type=str, default='samples.npz', help='Path to samples')
    args = parser.parse_args()

    print("="*60)
    print("WEIGHTED YUKAWA ANALYSIS v2.0")
    print("="*60)
    print(f"Parameters: sigma={args.sigma}, offset={args.offset}, asymmetry={args.asymmetry}")

    # Load models
    print(f"\nLoading models from {args.model_path}...")
    ckpt = torch.load(args.model_path, map_location=device)

    h2_model = H2Network(config).to(device)
    h3_model = H3TCSNetwork(config).to(device)
    h2_model.load_state_dict(ckpt['h2'])
    h3_model.load_state_dict(ckpt['h3'])
    print("  Models loaded.")

    # Load data
    print(f"Loading data from {args.data_path}...")
    data = np.load(args.data_path)
    coords = torch.from_numpy(data['coords']).float()
    metric = torch.from_numpy(data['metric']).float()
    print(f"  Samples: {coords.shape[0]}")

    # ===== Test 1: Uniform (baseline) =====
    print("\n" + "="*60)
    print("TEST 1: Uniform weighting (baseline)")
    print("="*60)

    result_uniform = compute_weighted_yukawa(
        h2_model, h3_model, coords, metric,
        weight_fn=lambda lam, **kw: torch.ones_like(lam),
        weight_params={},
        coeffs=YUKAWA_COEFFS
    )
    analysis_uniform = analyze_spectrum(result_uniform['eigenvalues'], label="Uniform")

    # ===== Test 2: Neck bump =====
    print("\n" + "="*60)
    print(f"TEST 2: Neck bump (sigma={args.sigma}, offset={args.offset})")
    print("="*60)

    result_neck = compute_weighted_yukawa(
        h2_model, h3_model, coords, metric,
        weight_fn=region_weight_neck,
        weight_params={'sigma': args.sigma, 'offset': args.offset},
        coeffs=YUKAWA_COEFFS
    )
    analysis_neck = analyze_spectrum(result_neck['eigenvalues'], label="Neck bump")

    # ===== Test 3: Asymmetric (if requested) =====
    if args.asymmetry != 0:
        print("\n" + "="*60)
        print(f"TEST 3: Asymmetric (asymmetry={args.asymmetry})")
        print("="*60)

        result_asym = compute_weighted_yukawa(
            h2_model, h3_model, coords, metric,
            weight_fn=region_weight_asymmetric,
            weight_params={'sigma': args.sigma, 'offset': args.offset, 'asymmetry': args.asymmetry},
            coeffs=YUKAWA_COEFFS
        )
        analysis_asym = analyze_spectrum(result_asym['eigenvalues'], label="Asymmetric")

    # ===== Test 4: Double bump =====
    print("\n" + "="*60)
    print("TEST 4: Double bump (transition zones)")
    print("="*60)

    result_double = compute_weighted_yukawa(
        h2_model, h3_model, coords, metric,
        weight_fn=region_weight_double_bump,
        weight_params={'sigma': 0.2, 'offset': 0.1, 'separation': 0.5},
        coeffs=YUKAWA_COEFFS
    )
    analysis_double = analyze_spectrum(result_double['eigenvalues'], label="Double bump")

    # ===== Summary =====
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Method':<20} | {'Non-zero':>8} | {'n_visible':>9} | {'tau_n':>6} | {'tau':>8} | {'error':>6}")
    print("-"*70)

    for name, analysis in [
        ("Uniform", analysis_uniform),
        ("Neck bump", analysis_neck),
        ("Double bump", analysis_double),
    ]:
        print(f"{name:<20} | {analysis['nonzero']:>8} | {analysis['n_visible']:>9} | "
              f"{analysis['tau_best_n']:>6} | {analysis['tau_estimate']:>8.4f} | {analysis['tau_error_pct']:>5.1f}%")

    print("\nDone.")

if __name__ == '__main__':
    main()
