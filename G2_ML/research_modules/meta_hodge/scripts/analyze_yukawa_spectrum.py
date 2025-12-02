#!/usr/bin/env python
"""Analyze Yukawa tensor spectrum to see if 43/77 split emerges naturally.

Key hypothesis: The visible/hidden split (43/34) should emerge from the
spectral structure of the Yukawa coupling matrix, not be imposed by hand.

Method:
1. Compute Y_ijk (21 x 21 x 77) Yukawa tensor
2. Contract: M_kl = sum_ij Y_ijk * Y_ijl  (77x77 Gram matrix)
3. Eigendecompose M to find natural mode structure
4. Check if eigenvalue gap suggests 43/34 split
5. Relate largest eigenvalue gap to tau = 3472/891
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch

# Add parent directories to path
script_dir = Path(__file__).resolve().parent
meta_hodge_dir = script_dir.parent
g2_ml_dir = meta_hodge_dir.parent
if str(g2_ml_dir) not in sys.path:
    sys.path.insert(0, str(g2_ml_dir))


@dataclass
class SpectralAnalysis:
    """Results of Yukawa spectral analysis."""
    eigenvalues: np.ndarray  # (77,) sorted descending
    eigenvectors: np.ndarray  # (77, 77)

    # Gap analysis
    gaps: np.ndarray  # eigenvalue gaps
    largest_gap_idx: int  # index of largest gap
    largest_gap_ratio: float  # ratio of gap to mean

    # Split analysis
    suggested_n_visible: int  # where the gap suggests splitting
    visible_fraction: float  # n_visible / 77

    # Tau connection
    spectral_tau: float  # ratio that might match 3472/891
    tau_target: float  # 3472/891 = 3.896...


def compute_yukawa_gram_matrix(
    phi: torch.Tensor,
    coords: torch.Tensor,
    n_h2: int = 21,
    n_h3: int = 77,
) -> torch.Tensor:
    """Compute Gram matrix M_kl = sum_ij Y_ijk * Y_ijl.

    This is a proxy for the full Yukawa computation - we use
    the outer product structure of phi to estimate Y.
    """
    batch_size = phi.shape[0]
    device = phi.device

    # For each H3 mode k, compute its "coupling strength" to H2 pairs
    # Simplified: use phi components directly as proxy for H3 modes
    # Real implementation would use full harmonic form computation

    # Build 77 H3 mode profiles from phi (35 local) + global (42)
    # Local modes: direct phi components
    local_modes = phi  # (batch, 35)

    # Global modes: position-dependent combinations
    lam = coords[:, 0:1]  # (batch, 1)
    xi = coords[:, 1:]  # (batch, 6)

    global_modes = []
    # Polynomial in lambda
    for p in range(5):
        global_modes.append(lam ** p)
    # Mixed terms
    for i in range(6):
        global_modes.append(xi[:, i:i+1])
        global_modes.append(lam * xi[:, i:i+1])
    # Trigonometric
    for k in [1, 2, 3]:
        global_modes.append(torch.sin(2 * np.pi * k * lam))
        global_modes.append(torch.cos(2 * np.pi * k * lam))
    # Fill remaining to get 42
    while len(global_modes) < 42:
        idx = len(global_modes)
        global_modes.append(torch.sin(np.pi * idx * lam / 10))

    global_modes = torch.cat(global_modes[:42], dim=1)  # (batch, 42)

    # Combine into 77 modes
    h3_modes = torch.cat([local_modes, global_modes], dim=1)  # (batch, 77)

    # Normalize modes
    h3_modes = h3_modes / (torch.norm(h3_modes, dim=0, keepdim=True) + 1e-8)

    # Compute Gram matrix: M_kl = <mode_k, mode_l> weighted by geometry
    # This approximates sum_ij Y_ijk * Y_ijl
    M = (h3_modes.T @ h3_modes) / batch_size  # (77, 77)

    return M


def analyze_spectrum(M: torch.Tensor, tau_target: float = 3472/891) -> SpectralAnalysis:
    """Analyze eigenvalue spectrum of Gram matrix."""
    # Symmetrize
    M = 0.5 * (M + M.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(M)

    # Sort descending
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx].cpu().numpy()
    eigenvectors = eigenvectors[:, idx].cpu().numpy()

    # Compute gaps
    gaps = np.abs(np.diff(eigenvalues))

    # Find largest gap (excluding trivial ones at the end)
    # Look in range [10, 60] to avoid edge effects
    search_range = gaps[10:60]
    if len(search_range) > 0:
        largest_gap_idx = 10 + np.argmax(search_range)
    else:
        largest_gap_idx = np.argmax(gaps)

    largest_gap_ratio = gaps[largest_gap_idx] / (np.mean(gaps) + 1e-10)

    # The gap at index k suggests splitting into (k+1) visible and (77-k-1) hidden
    suggested_n_visible = largest_gap_idx + 1
    visible_fraction = suggested_n_visible / 77

    # Try to find tau in the spectrum
    # Hypothesis: tau = lambda_visible / lambda_hidden or similar ratio
    if suggested_n_visible < 77:
        sum_visible = np.sum(eigenvalues[:suggested_n_visible])
        sum_hidden = np.sum(eigenvalues[suggested_n_visible:])
        if sum_hidden > 1e-10:
            spectral_tau = sum_visible / sum_hidden
        else:
            spectral_tau = float('inf')
    else:
        spectral_tau = float('inf')

    return SpectralAnalysis(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        gaps=gaps,
        largest_gap_idx=largest_gap_idx,
        largest_gap_ratio=largest_gap_ratio,
        suggested_n_visible=suggested_n_visible,
        visible_fraction=visible_fraction,
        spectral_tau=spectral_tau,
        tau_target=tau_target,
    )


def find_43_77_gap(eigenvalues: np.ndarray) -> Dict:
    """Specifically check the gap at position 43."""
    if len(eigenvalues) < 77:
        return {"error": "Not enough eigenvalues"}

    gap_at_43 = abs(eigenvalues[42] - eigenvalues[43])
    mean_gap = np.mean(np.abs(np.diff(eigenvalues)))

    # Compare to nearby gaps
    gaps_around = np.abs(np.diff(eigenvalues[35:50]))
    gap_43_rank = np.sum(gaps_around > gap_at_43) + 1

    return {
        "gap_at_43": float(gap_at_43),
        "mean_gap": float(mean_gap),
        "gap_43_ratio": float(gap_at_43 / (mean_gap + 1e-10)),
        "gap_43_rank_in_35_50": int(gap_43_rank),
        "eigenvalue_42": float(eigenvalues[42]),
        "eigenvalue_43": float(eigenvalues[43]),
        "ratio_42_43": float(eigenvalues[42] / (eigenvalues[43] + 1e-10)),
    }


def analyze_tau_candidates(eigenvalues: np.ndarray, tau_target: float = 3472/891) -> List[Dict]:
    """Search for ratios in the spectrum that match tau."""
    candidates = []

    # Try various ratio definitions
    for split in range(20, 60):
        sum_top = np.sum(eigenvalues[:split])
        sum_bottom = np.sum(eigenvalues[split:])
        if sum_bottom > 1e-10:
            ratio = sum_top / sum_bottom
            error = abs(ratio - tau_target) / tau_target
            candidates.append({
                "split": split,
                "ratio": float(ratio),
                "error_pct": float(error * 100),
                "sum_top": float(sum_top),
                "sum_bottom": float(sum_bottom),
            })

    # Sort by error
    candidates.sort(key=lambda x: x["error_pct"])
    return candidates[:10]  # Top 10


def print_spectrum_analysis(analysis: SpectralAnalysis, gap_43: Dict, tau_candidates: List[Dict]):
    """Print detailed analysis results."""
    print("=" * 70)
    print("YUKAWA SPECTRUM ANALYSIS")
    print("=" * 70)
    print()

    print("[EIGENVALUE DISTRIBUTION]")
    print(f"  Total modes: 77")
    print(f"  Top 5 eigenvalues: {analysis.eigenvalues[:5]}")
    print(f"  Bottom 5 eigenvalues: {analysis.eigenvalues[-5:]}")
    print(f"  Sum of all: {np.sum(analysis.eigenvalues):.4f}")
    print()

    print("[GAP ANALYSIS]")
    print(f"  Largest gap at index: {analysis.largest_gap_idx}")
    print(f"  Largest gap ratio (vs mean): {analysis.largest_gap_ratio:.2f}x")
    print(f"  Suggested visible modes: {analysis.suggested_n_visible}")
    print(f"  Visible fraction: {analysis.visible_fraction:.4f} (target: {43/77:.4f})")
    print()

    print("[43/77 SPECIFIC CHECK]")
    print(f"  Gap at position 43: {gap_43['gap_at_43']:.6f}")
    print(f"  Mean gap: {gap_43['mean_gap']:.6f}")
    print(f"  Gap 43 / mean: {gap_43['gap_43_ratio']:.2f}x")
    print(f"  Rank of gap 43 in [35-50]: {gap_43['gap_43_rank_in_35_50']}/15")
    print(f"  lambda_42 / lambda_43: {gap_43['ratio_42_43']:.4f}")
    print()

    print("[TAU CANDIDATES]")
    print(f"  Target tau = 3472/891 = {3472/891:.6f}")
    print()
    print(f"  {'Split':>6} | {'Ratio':>10} | {'Error %':>10}")
    print(f"  {'-'*6} | {'-'*10} | {'-'*10}")
    for c in tau_candidates[:5]:
        marker = " <-- 43!" if c["split"] == 43 else ""
        print(f"  {c['split']:>6} | {c['ratio']:>10.4f} | {c['error_pct']:>10.2f}%{marker}")
    print()

    # Key finding
    print("[KEY FINDING]")
    if abs(analysis.suggested_n_visible - 43) <= 5:
        print(f"  The spectral gap suggests ~{analysis.suggested_n_visible} visible modes")
        print(f"  This is CLOSE to the predicted 43!")
        print(f"  => 43/77 split may EMERGE from Yukawa structure")
    else:
        print(f"  The spectral gap suggests {analysis.suggested_n_visible} visible modes")
        print(f"  This differs from predicted 43")
        print(f"  => 43/77 split may need different mechanism")


def load_phi_from_samples(version: str = "1_6") -> Tuple[torch.Tensor, torch.Tensor]:
    """Load phi and coords from saved samples."""
    samples_path = g2_ml_dir / version.replace("_", ".") / "samples.npz"
    if not samples_path.exists():
        # Try alternate path
        samples_path = g2_ml_dir / f"1_{version.split('_')[-1]}" / "samples.npz"

    if samples_path.exists():
        data = np.load(samples_path)
        coords = torch.from_numpy(data["coords"]).float()
        phi = torch.from_numpy(data["phi"]).float()
        return phi, coords

    # Fallback: generate synthetic data
    print(f"Samples not found at {samples_path}, generating synthetic data...")
    n_samples = 5000
    coords = torch.rand(n_samples, 7)

    # Generate phi with structure similar to K7_GIFT
    # 35 components for Lambda^3
    phi = torch.randn(n_samples, 35) * 0.5
    # Add position-dependent modulation
    lam = coords[:, 0:1]
    phi = phi * (1 + 0.3 * torch.sin(2 * np.pi * lam))
    # Normalize to ||phi||^2 ~ 7
    phi = phi * np.sqrt(7.0) / (torch.norm(phi, dim=1, keepdim=True) + 1e-8)

    return phi, coords


def run_analysis_on_baseline(version: str = "1_8", n_samples: int = 5000):
    """Run spectral analysis on baseline model."""
    print(f"Loading data (version {version})...")

    # Try to load from samples.npz
    phi, coords = load_phi_from_samples(version)

    if phi.shape[0] > n_samples:
        idx = torch.randperm(phi.shape[0])[:n_samples]
        phi = phi[idx]
        coords = coords[idx]

    print(f"Loaded {phi.shape[0]} samples, phi shape: {phi.shape}")

    # Compute Gram matrix
    print("Computing Yukawa Gram matrix...")
    M = compute_yukawa_gram_matrix(phi, coords)

    # Analyze spectrum
    print("Analyzing spectrum...")
    analysis = analyze_spectrum(M)

    # Check 43 specifically
    gap_43 = find_43_77_gap(analysis.eigenvalues)

    # Find tau candidates
    tau_candidates = analyze_tau_candidates(analysis.eigenvalues)

    # Print results
    print_spectrum_analysis(analysis, gap_43, tau_candidates)

    return analysis, gap_43, tau_candidates


def run_analysis_on_deformed(results_dir: Path, n_samples: int = 2000):
    """Run spectral analysis on multiple deformed points."""
    # Load stable points
    json_path = results_dir / "deformation_results.json"
    if not json_path.exists():
        print(f"Results file not found: {json_path}")
        return []

    with open(json_path) as f:
        results = json.load(f)

    stable = [r for r in results if r["stable"]]
    print(f"Found {len(stable)} stable points")

    if not stable:
        return []

    # Load baseline phi
    phi, coords = load_phi_from_samples("1_8")

    all_analyses = []

    # Sample a few representative points
    sample_indices = [0, len(stable)//2, -1] if len(stable) >= 3 else list(range(len(stable)))

    for idx in sample_indices:
        point = stable[idx]
        sigma, s, alpha = point["sigma"], point["s"], point["alpha"]

        print(f"\n--- Point ({sigma:.2f}, {s:.2f}, {alpha:+.2f}) ---")

        # Simple deformation: scale phi by sigma*s and add alpha asymmetry
        lam = coords[:, 0:1]
        scale = sigma * s
        asymmetry = 1 + alpha * torch.sign(lam - 0.5)
        phi_deformed = phi * scale * asymmetry

        # Compute Gram matrix
        M = compute_yukawa_gram_matrix(phi_deformed, coords)

        # Analyze
        analysis = analyze_spectrum(M)
        gap_43 = find_43_77_gap(analysis.eigenvalues)

        print(f"  Suggested n_visible: {analysis.suggested_n_visible}")
        print(f"  Gap 43 ratio: {gap_43['gap_43_ratio']:.2f}x mean")

        all_analyses.append({
            "point": (sigma, s, alpha),
            "suggested_n_visible": analysis.suggested_n_visible,
            "gap_43_ratio": gap_43["gap_43_ratio"],
            "eigenvalues": analysis.eigenvalues.tolist(),
        })

    return all_analyses


def main():
    parser = argparse.ArgumentParser(description="Analyze Yukawa spectrum for 43/77 emergence")
    parser.add_argument("--baseline", action="store_true", help="Analyze baseline only")
    parser.add_argument("--deformed", action="store_true", help="Analyze deformed points")
    parser.add_argument("--results-dir", type=Path, help="Deformation results directory")
    parser.add_argument("--latest", action="store_true", help="Use latest results")
    parser.add_argument("--samples", type=int, default=3000, help="Number of samples")
    parser.add_argument("--output", "-o", type=Path, help="Output directory")

    args = parser.parse_args()

    # Default to baseline analysis
    if not args.baseline and not args.deformed:
        args.baseline = True

    results = {}

    if args.baseline:
        print("\n" + "=" * 70)
        print("BASELINE ANALYSIS")
        print("=" * 70)
        analysis, gap_43, tau_candidates = run_analysis_on_baseline(n_samples=args.samples)
        results["baseline"] = {
            "suggested_n_visible": int(analysis.suggested_n_visible),
            "gap_43": gap_43,
            "tau_candidates": [(int(s), float(r), float(e)) for s, r, e in tau_candidates[:5]],
            "eigenvalues": analysis.eigenvalues.tolist(),
        }

    if args.deformed:
        # Find results directory
        if args.latest or args.results_dir is None:
            base_dir = meta_hodge_dir / "artifacts" / "deformation_atlas"
            if base_dir.exists():
                dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
                if dirs:
                    args.results_dir = dirs[-1]

        if args.results_dir and args.results_dir.exists():
            print("\n" + "=" * 70)
            print("DEFORMED POINTS ANALYSIS")
            print("=" * 70)
            deformed_results = run_analysis_on_deformed(args.results_dir, n_samples=args.samples)
            results["deformed"] = deformed_results

    # Save results
    if args.output:
        output_dir = args.output
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = meta_hodge_dir / "artifacts" / "spectral_analysis" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "spectral_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
