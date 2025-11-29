#!/usr/bin/env python3
"""
TCS/Joyce v2.1 - Improved Global Mode Coupling

Key insight from v2.0 results:
- Local modes (phi) dominate the Yukawa spectrum
- TCS global modes couple too weakly
- Need to boost global mode contribution

Strategy:
1. Use the ACTUAL phi structure to derive global modes
2. Build global modes that couple through phi, not independently
3. Match the global mode amplitude to local mode amplitude
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data(path: Path) -> tuple:
    """Load phi and coords from samples.npz."""
    print(f"Loading: {path}")
    data = np.load(path)
    print(f"  Keys: {list(data.keys())}")

    phi = torch.from_numpy(data['phi']).float()
    coords = torch.from_numpy(data['coords']).float()

    # Also load omega (H2) and Phi (H3) if available
    omega = torch.from_numpy(data['omega']).float() if 'omega' in data else None
    Phi = torch.from_numpy(data['Phi']).float() if 'Phi' in data else None

    print(f"  phi: {phi.shape}")
    print(f"  coords: {coords.shape}")
    if omega is not None:
        print(f"  omega (H2): {omega.shape}")
    if Phi is not None:
        print(f"  Phi (H3): {Phi.shape}")

    return phi, coords, omega, Phi


def build_tcs_global_modes_v2(
    phi: torch.Tensor,
    coords: torch.Tensor,
    coupling_strength: float = 1.0,
) -> torch.Tensor:
    """Build 42 TCS global modes that couple through phi.

    Key improvement: Global modes are built as phi-weighted profiles,
    not independent of phi. This ensures they participate in Yukawa.
    """
    N = phi.shape[0]
    lam = coords[:, 0]
    xi = coords[:, 1:]

    # Profile functions
    def left_profile(x):
        return torch.exp(-((x - 0.2) / 0.15)**2)

    def right_profile(x):
        return torch.exp(-((x - 0.8) / 0.15)**2)

    def neck_profile(x):
        return torch.exp(-((x - 0.5) / 0.1)**2)

    # Compute phi statistics
    phi_mean = phi.mean(dim=0, keepdim=True)  # (1, 35)
    phi_std = phi.std(dim=0, keepdim=True) + 1e-8
    phi_normalized = (phi - phi_mean) / phi_std

    # Key insight: global modes should be CORRELATED with phi variation
    # This way they contribute to the same Yukawa structure

    global_modes = []

    # Left modes (14): phi variation weighted by left profile
    f_L = left_profile(lam)  # (N,)
    for j in range(14):
        # Mix phi components based on left-region structure
        phi_component = phi[:, j % 35]
        xi_mod = 1 + 0.3 * xi[:, j % 6]
        mode = f_L * phi_component * xi_mod * coupling_strength
        global_modes.append(mode)

    # Right modes (14): phi variation weighted by right profile
    f_R = right_profile(lam)
    for j in range(14):
        phi_component = phi[:, (j + 14) % 35]
        xi_mod = 1 + 0.3 * xi[:, (5 - j) % 6]
        mode = f_R * phi_component * xi_mod * coupling_strength
        global_modes.append(mode)

    # Neck modes (14): phi variation weighted by neck profile
    f_N = neck_profile(lam)
    for j in range(14):
        # Use phi gradient as proxy for neck coupling
        phi_component = phi[:, (j + 7) % 35]
        xi_coupling = 1 + 0.2 * (xi**2).mean(dim=1)
        mode = f_N * phi_component * xi_coupling * coupling_strength
        global_modes.append(mode)

    return torch.stack(global_modes, dim=1)


def build_hybrid_global_modes(
    phi: torch.Tensor,
    coords: torch.Tensor,
    omega: torch.Tensor = None,
    Phi: torch.Tensor = None,
) -> torch.Tensor:
    """Build global modes using all available H2/H3 data.

    If omega (21 H2 forms) and Phi (77 H3 forms) are available,
    use them to construct better global modes.
    """
    N = phi.shape[0]
    lam = coords[:, 0]

    # If we have the full H3 (77 modes), use modes 35-76 directly
    if Phi is not None and Phi.shape[1] >= 77:
        print("  Using pre-computed H3 global modes (35-76)")
        return Phi[:, 35:77]

    # If we have H2 (21 modes), use them to inform global mode structure
    if omega is not None:
        print("  Building global modes informed by H2 (omega)")

        global_modes = []

        # Use H2 modes to create H3-like combinations
        # This is a proxy for wedge products
        for j in range(42):
            i1 = j % 21
            i2 = (j + 7) % 21
            # Approximate omega_i ^ omega_j contribution
            h2_combo = omega[:, i1] * omega[:, i2]

            # Weight by TCS profile
            if j < 14:
                profile = torch.exp(-((lam - 0.2) / 0.15)**2)
            elif j < 28:
                profile = torch.exp(-((lam - 0.8) / 0.15)**2)
            else:
                profile = torch.exp(-((lam - 0.5) / 0.1)**2)

            mode = h2_combo * profile
            global_modes.append(mode)

        return torch.stack(global_modes, dim=1)

    # Fallback to phi-based construction
    print("  Using phi-based global mode construction")
    return build_tcs_global_modes_v2(phi, coords)


def analyze_spectrum_v2(
    phi: torch.Tensor,
    coords: torch.Tensor,
    omega: torch.Tensor = None,
    Phi: torch.Tensor = None,
    coupling_strength: float = 1.0,
) -> dict:
    """Run spectral analysis with improved global mode coupling."""
    N = phi.shape[0]

    print("\n" + "=" * 70)
    print("TCS/JOYCE v2.1 SPECTRAL ANALYSIS")
    print("=" * 70)

    # Build H3 modes
    print("\n[1] Building 77 H3 modes...")
    local_modes = phi  # (N, 35)

    # Try hybrid construction first
    global_modes = build_hybrid_global_modes(phi, coords, omega, Phi)

    # Match amplitude scales
    local_scale = local_modes.abs().mean()
    global_scale = global_modes.abs().mean()

    print(f"    Local mode scale: {local_scale:.6f}")
    print(f"    Global mode scale: {global_scale:.6f}")

    # Boost global modes to match local scale
    if global_scale > 0:
        scale_ratio = local_scale / global_scale
        global_modes = global_modes * scale_ratio * coupling_strength
        print(f"    Applied scale boost: {scale_ratio * coupling_strength:.2f}x")

    h3_modes = torch.cat([local_modes, global_modes], dim=1)

    # Normalize each mode
    norms = torch.norm(h3_modes, dim=0, keepdim=True) + 1e-10
    h3_modes = h3_modes / norms

    print(f"    H3 total: {h3_modes.shape}")

    # Compute Gram matrix
    print("\n[2] Computing Yukawa Gram matrix...")
    M = (h3_modes.T @ h3_modes) / N
    M = 0.5 * (M + M.T)

    # Eigendecomposition
    print("\n[3] Eigendecomposition...")
    eigenvalues, _ = torch.linalg.eigh(M)
    eigenvalues = eigenvalues.flip(0)
    eigenvalues = torch.clamp(eigenvalues, min=0)

    print(f"    Top 5: {[f'{e:.6f}' for e in eigenvalues[:5].tolist()]}")
    print(f"    Around 43: {[f'{e:.6f}' for e in eigenvalues[40:46].tolist()]}")

    # Count significant eigenvalues
    threshold = eigenvalues.max() * 0.01
    n_significant = (eigenvalues > threshold).sum().item()
    print(f"    Significant (>1% max): {n_significant}")

    # Gap analysis
    print("\n[4] Gap analysis...")
    gaps = torch.abs(eigenvalues[:-1] - eigenvalues[1:])
    mean_gap = gaps[20:55].mean()

    print(f"    Mean gap (20-55): {mean_gap:.6f}")
    print()
    print("    Pos | Eigenvalue | Gap      | Ratio")
    print("    " + "-" * 42)

    for pos in range(38, 48):
        ev = eigenvalues[pos].item()
        g = gaps[pos].item()
        r = g / (mean_gap + 1e-10)
        mark = " *** 43! ***" if pos == 42 else ""
        print(f"    {pos:>3} | {ev:>10.6f} | {g:.6f} | {r:>5.2f}x{mark}")

    # Find largest gap in range
    idx = gaps[35:50].argmax().item() + 35
    print(f"\n    Largest gap [35-50]: position {idx}")
    print(f"    Gap ratio: {gaps[idx]/mean_gap:.2f}x")

    # Tau analysis
    print("\n[5] Tau analysis...")
    tau_target = 3472.0 / 891.0

    # Find split that gives closest tau
    best_split = 43
    best_error = float('inf')

    for split in range(30, 50):
        s1 = eigenvalues[:split].sum().item()
        s2 = eigenvalues[split:].sum().item()
        if s2 > 1e-10:
            tau = s1 / s2
            err = abs(tau - tau_target) / tau_target
            if err < best_error:
                best_error = err
                best_split = split

    s43 = eigenvalues[:43].sum().item()
    s34 = eigenvalues[43:].sum().item()
    tau_43 = s43 / (s34 + 1e-10)

    print(f"    At split 43:")
    print(f"      Sum(1:43) = {s43:.6f}")
    print(f"      Sum(44:77) = {s34:.6f}")
    print(f"      Tau = {tau_43:.4f}")
    print(f"      Error = {abs(tau_43 - tau_target)/tau_target*100:.2f}%")
    print(f"    Best split for tau: {best_split} (error: {best_error*100:.2f}%)")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Significant modes: {n_significant}")
    print(f"  Gap at 42-43: {gaps[42]/mean_gap:.2f}x mean")
    print(f"  Largest gap: position {idx}")
    print(f"  Tau at 43: {tau_43:.4f} (target: {tau_target:.4f})")

    if n_significant > 40:
        print("\n  Global modes contributing!")
    if idx == 42:
        print("  *** Gap at 43 detected! ***")

    print("=" * 70)

    return {
        'eigenvalues': eigenvalues,
        'n_significant': n_significant,
        'largest_gap': idx,
        'tau_43': tau_43,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', '-s', type=Path, required=True)
    parser.add_argument('--coupling', '-c', type=float, default=1.0,
                        help='Global mode coupling strength (default: 1.0)')
    args = parser.parse_args()

    phi, coords, omega, Phi = load_data(args.samples)

    # Try different coupling strengths
    for coupling in [1.0, 2.0, 5.0]:
        print(f"\n{'#' * 70}")
        print(f"# COUPLING STRENGTH = {coupling}")
        print(f"{'#' * 70}")
        results = analyze_spectrum_v2(phi, coords, omega, Phi, coupling)


if __name__ == '__main__':
    main()
