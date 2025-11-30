"""Correct wedge product implementation for Yukawa computation.

The key insight: Y_ijk = integral(omega_i ^ omega_j ^ Phi_k)

where:
- omega_i, omega_j are 2-forms (21 components each, indexed by pairs (ab) with a<b)
- Phi_k is a 3-form (35 components, indexed by triples (cde) with c<d<e)
- The wedge gives a 7-form (volume form) when indices partition {0,1,2,3,4,5,6}
"""

import torch
import numpy as np
from itertools import combinations


def build_index_maps():
    """Build index mappings for forms."""
    # 2-form indices: pairs (a,b) with a < b
    pairs = list(combinations(range(7), 2))  # 21 pairs
    pair_to_idx = {p: i for i, p in enumerate(pairs)}

    # 3-form indices: triples (c,d,e) with c < d < e
    triples = list(combinations(range(7), 3))  # 35 triples
    triple_to_idx = {t: i for i, t in enumerate(triples)}

    # 4-form indices: quads (a,b,c,d) with a < b < c < d
    quads = list(combinations(range(7), 4))  # 35 quads
    quad_to_idx = {q: i for i, q in enumerate(quads)}

    return pairs, triples, quads, pair_to_idx, triple_to_idx, quad_to_idx


def levi_civita_7(indices):
    """Compute Levi-Civita symbol for 7 indices."""
    if len(set(indices)) != 7:
        return 0
    # Count inversions
    n = len(indices)
    inversions = sum(1 for i in range(n) for j in range(i+1, n) if indices[i] > indices[j])
    return 1 if inversions % 2 == 0 else -1


def compute_wedge_coefficient(pair1, pair2, triple):
    """Compute coefficient of omega[pair1] ^ omega[pair2] ^ Phi[triple].

    Returns sign if the 7 indices partition {0,...,6}, else 0.
    """
    # Check that all 7 indices are distinct
    all_indices = pair1 + pair2 + triple
    if len(set(all_indices)) != 7:
        return 0

    # The coefficient is the Levi-Civita symbol
    return levi_civita_7(all_indices)


def build_yukawa_coefficients():
    """Precompute all non-zero Yukawa coefficients.

    Returns list of (pair1_idx, pair2_idx, triple_idx, sign) tuples.
    """
    pairs, triples, _, pair_to_idx, triple_to_idx, _ = build_index_maps()

    coeffs = []
    for i1, p1 in enumerate(pairs):
        for i2, p2 in enumerate(pairs):
            if i2 < i1:  # Will handle antisymmetry separately
                continue
            for i3, t in enumerate(triples):
                sign = compute_wedge_coefficient(p1, p2, t)
                if sign != 0:
                    coeffs.append((i1, i2, i3, sign))

    print(f"Found {len(coeffs)} non-zero Yukawa coefficients")
    return coeffs


def compute_yukawa_proper(omega, Phi, metric, coeffs=None):
    """Compute Yukawa tensor with proper wedge product.

    Args:
        omega: (batch, 21 modes, 21 components) - H2 forms
        Phi: (batch, 77 modes, 35 components) - H3 forms
        metric: (batch, 7, 7)
        coeffs: precomputed coefficients (optional)

    Returns:
        Y: (21, 21, 77) Yukawa tensor
    """
    if coeffs is None:
        coeffs = build_yukawa_coefficients()

    batch = omega.shape[0]
    n_h2 = omega.shape[1]  # 21
    n_h3 = Phi.shape[1]    # 77
    device = omega.device

    # Volume element
    det_g = torch.det(metric)
    vol = torch.sqrt(det_g.abs())
    total_vol = vol.sum()

    # Initialize Yukawa tensor
    Y = torch.zeros(n_h2, n_h2, n_h3, device=device)

    # For each H2 mode pair (a, b) and H3 mode c
    for a in range(n_h2):
        for b in range(a, n_h2):
            omega_a = omega[:, a, :]  # (batch, 21)
            omega_b = omega[:, b, :]  # (batch, 21)

            for c in range(n_h3):
                Phi_c = Phi[:, c, :]  # (batch, 35)

                # Sum over all valid index combinations
                integral = torch.zeros(batch, device=device)

                for i1, i2, i3, sign in coeffs:
                    # omega_a[i1] * omega_b[i2] * Phi_c[i3]
                    contrib = sign * omega_a[:, i1] * omega_b[:, i2] * Phi_c[:, i3]
                    integral += contrib

                # Weight by volume and integrate
                Y[a, b, c] = (integral * vol).sum() / total_vol
                if a != b:
                    Y[b, a, c] = -Y[a, b, c]  # Antisymmetry

    return Y


if __name__ == "__main__":
    # Test
    print("Building Yukawa coefficients...")
    coeffs = build_yukawa_coefficients()

    print(f"\nFirst 10 coefficients:")
    for c in coeffs[:10]:
        pairs, triples, _, _, _, _ = build_index_maps()
        p1, p2, t, sign = c
        print(f"  omega[{pairs[p1]}] ^ omega[{pairs[p2]}] ^ Phi[{triples[t]}] : {sign:+d}")
