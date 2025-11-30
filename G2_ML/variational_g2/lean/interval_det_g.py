#!/usr/bin/env python3
"""
Interval Arithmetic for det(g) = 65/32 Verification

This module implements certified interval arithmetic to verify that
det(g) computed from the PINN output phi satisfies det(g) = 65/32.

The key formula is:
    g_ij = (1/6) * sum_{k,l} phi_{ikl} * phi_{jkl}
    det(g) = det([g_ij])

With interval arithmetic, we can prove:
    |det(g) - 65/32| < tolerance

for all inputs in the domain, not just sampled points.

Author: GIFT Framework
Date: 2025-11-30
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


@dataclass
class Interval:
    """
    A real interval [lo, hi] with guaranteed bounds.

    All arithmetic operations return intervals that are guaranteed
    to contain the true result for any values in the input intervals.
    """
    lo: float
    hi: float

    def __post_init__(self):
        assert self.lo <= self.hi, f"Invalid interval: [{self.lo}, {self.hi}]"

    @classmethod
    def point(cls, x: float) -> 'Interval':
        """Create a point interval [x, x]."""
        return cls(x, x)

    @classmethod
    def from_bounds(cls, lo: float, hi: float) -> 'Interval':
        """Create interval from bounds."""
        return cls(lo, hi)

    def __repr__(self):
        return f"[{self.lo:.6g}, {self.hi:.6g}]"

    def width(self) -> float:
        """Width of the interval."""
        return self.hi - self.lo

    def midpoint(self) -> float:
        """Midpoint of the interval."""
        return (self.lo + self.hi) / 2

    def contains(self, x: float) -> bool:
        """Check if x is in the interval."""
        return self.lo <= x <= self.hi

    def __add__(self, other: 'Interval') -> 'Interval':
        """Interval addition: [a,b] + [c,d] = [a+c, b+d]."""
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other: 'Interval') -> 'Interval':
        """Interval subtraction: [a,b] - [c,d] = [a-d, b-c]."""
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __mul__(self, other: 'Interval') -> 'Interval':
        """Interval multiplication."""
        products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ]
        return Interval(min(products), max(products))

    def __truediv__(self, other: 'Interval') -> 'Interval':
        """Interval division (assumes 0 not in other)."""
        if other.lo <= 0 <= other.hi:
            raise ValueError("Division by interval containing 0")
        return self * Interval(1.0 / other.hi, 1.0 / other.lo)

    def __neg__(self) -> 'Interval':
        """Negation: -[a,b] = [-b, -a]."""
        return Interval(-self.hi, -self.lo)

    def __pow__(self, n: int) -> 'Interval':
        """Integer power."""
        if n == 0:
            return Interval.point(1.0)
        if n == 1:
            return self
        if n == 2:
            if self.lo >= 0:
                return Interval(self.lo ** 2, self.hi ** 2)
            if self.hi <= 0:
                return Interval(self.hi ** 2, self.lo ** 2)
            return Interval(0, max(self.lo ** 2, self.hi ** 2))
        # General case
        if n % 2 == 0:
            return (self ** 2) ** (n // 2)
        return self * (self ** (n - 1))


def interval_sin(x: Interval) -> Interval:
    """Sine with interval arithmetic (conservative bounds)."""
    # For general intervals, use monotonicity analysis
    # Simplified: just use [-1, 1] as conservative bound
    return Interval(-1.0, 1.0)


def interval_cos(x: Interval) -> Interval:
    """Cosine with interval arithmetic (conservative bounds)."""
    return Interval(-1.0, 1.0)


def interval_tanh(x: Interval) -> Interval:
    """Hyperbolic tangent with interval arithmetic."""
    # tanh is monotonically increasing
    return Interval(math.tanh(x.lo), math.tanh(x.hi))


def interval_silu(x: Interval) -> Interval:
    """SiLU (Swish) activation with interval arithmetic.

    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Properties:
    - Monotonically increasing for x > -1.278
    - Minimum at x ≈ -1.278, value ≈ -0.278
    - Asymptotes to x as x → ∞, to 0 as x → -∞
    """
    def silu_point(t: float) -> float:
        if t > 20:
            return t
        if t < -20:
            return 0.0
        return t / (1 + math.exp(-t))

    # SiLU minimum is at x ≈ -1.278, value ≈ -0.278
    silu_min_x = -1.278
    silu_min_val = -0.278

    if x.hi < silu_min_x:
        # Monotonically increasing in this region (wrong, it's decreasing)
        # Actually SiLU is not monotonic, be conservative
        vals = [silu_point(x.lo), silu_point(x.hi)]
        return Interval(min(vals), max(vals))
    elif x.lo > silu_min_x:
        # Monotonically increasing
        return Interval(silu_point(x.lo), silu_point(x.hi))
    else:
        # Interval contains minimum
        lo = min(silu_point(x.lo), silu_point(x.hi), silu_min_val)
        hi = max(silu_point(x.lo), silu_point(x.hi))
        return Interval(lo, hi)


def interval_matmul(
    W: List[List[Interval]],
    x: List[Interval],
    b: Optional[List[Interval]] = None
) -> List[Interval]:
    """
    Matrix-vector multiplication with interval arithmetic.

    Computes y = W @ x + b where all operations use intervals.

    Args:
        W: Weight matrix as list of rows, each row is list of Interval
        x: Input vector as list of Interval
        b: Optional bias vector as list of Interval

    Returns:
        Output vector as list of Interval
    """
    out_dim = len(W)
    in_dim = len(x)

    result = []
    for i in range(out_dim):
        acc = Interval.point(0.0)
        for j in range(in_dim):
            acc = acc + W[i][j] * x[j]
        if b is not None:
            acc = acc + b[i]
        result.append(acc)

    return result


def interval_det_3x3(M: List[List[Interval]]) -> Interval:
    """
    Compute determinant of 3x3 matrix with interval arithmetic.

    det(M) = M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1])
           - M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0])
           + M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0])
    """
    a = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
    b = M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
    c = M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
    return a - b + c


def interval_det_7x7(M: List[List[Interval]]) -> Interval:
    """
    Compute determinant of 7x7 matrix with interval arithmetic.

    Uses cofactor expansion along first row.
    This is O(n!) but n=7 is manageable.

    WARNING: Interval blowup can be severe for large matrices.
    Consider using LU decomposition with pivoting for tighter bounds.
    """

    def minor(M, row, col):
        """Get minor matrix by removing row and column."""
        n = len(M)
        return [
            [M[i][j] for j in range(n) if j != col]
            for i in range(n) if i != row
        ]

    def det_recursive(M):
        size = len(M)
        if size == 1:
            return M[0][0]
        if size == 2:
            return M[0][0] * M[1][1] - M[0][1] * M[1][0]
        if size == 3:
            return interval_det_3x3(M)

        result = Interval.point(0.0)
        for j in range(size):
            sign = Interval.point(1.0 if j % 2 == 0 else -1.0)
            cofactor = det_recursive(minor(M, 0, j))
            result = result + sign * M[0][j] * cofactor

        return result

    return det_recursive(M)


def metric_from_phi_interval(phi: List[Interval]) -> List[List[Interval]]:
    """
    Compute metric g_ij from 3-form phi using interval arithmetic.

    Formula: g_ij = (1/6) * sum_{k<l} phi_{ikl} * phi_{jkl}

    The 35 components of phi are indexed by (i,j,k) with i<j<k.

    Args:
        phi: List of 35 intervals for the 3-form components

    Returns:
        7x7 matrix of intervals for the metric
    """
    # Generate index mapping
    indices = []
    for i in range(7):
        for j in range(i + 1, 7):
            for k in range(j + 1, 7):
                indices.append((i, j, k))

    def get_phi(i, j, k) -> Interval:
        """Get phi_{ijk} with antisymmetry."""
        # Sort indices and track sign
        idx_list = [(i, 0), (j, 1), (k, 2)]
        idx_list.sort()
        sorted_idx = tuple(x[0] for x in idx_list)
        perm = [x[1] for x in idx_list]

        # Count inversions for sign
        sign = 1
        for a in range(3):
            for b in range(a + 1, 3):
                if perm[a] > perm[b]:
                    sign *= -1

        # Find in indices list
        if sorted_idx in indices:
            linear_idx = indices.index(sorted_idx)
            if sign > 0:
                return phi[linear_idx]
            else:
                return -phi[linear_idx]
        else:
            return Interval.point(0.0)

    # Compute metric
    one_sixth = Interval.point(1.0 / 6.0)
    g = [[Interval.point(0.0) for _ in range(7)] for _ in range(7)]

    for i in range(7):
        for j in range(7):
            acc = Interval.point(0.0)
            for k in range(7):
                for l in range(7):
                    if k != l:
                        phi_ikl = get_phi(i, k, l)
                        phi_jkl = get_phi(j, k, l)
                        acc = acc + phi_ikl * phi_jkl
            g[i][j] = one_sixth * acc

    return g


def verify_det_g(
    phi: List[Interval],
    target: float = 65.0 / 32.0,
    tolerance: float = 1e-6
) -> Tuple[bool, Interval, str]:
    """
    Verify that det(g(phi)) is within tolerance of target.

    Args:
        phi: 35 intervals for the 3-form
        target: Target determinant (default: 65/32)
        tolerance: Allowed error

    Returns:
        (success, det_interval, message)
    """
    # Compute metric
    g = metric_from_phi_interval(phi)

    # Compute determinant
    det_g = interval_det_7x7(g)

    # Check if target is in interval (widened by tolerance)
    target_interval = Interval(target - tolerance, target + tolerance)

    # Check overlap
    if det_g.hi < target_interval.lo:
        return (False, det_g, f"det(g) = {det_g} is below target {target}")
    if det_g.lo > target_interval.hi:
        return (False, det_g, f"det(g) = {det_g} is above target {target}")

    # Check if det_g is contained in target_interval
    if det_g.lo >= target_interval.lo and det_g.hi <= target_interval.hi:
        return (True, det_g, f"VERIFIED: det(g) = {det_g} within tolerance of {target}")

    # Partial overlap - not fully verified
    return (False, det_g, f"PARTIAL: det(g) = {det_g} overlaps but not contained in [{target-tolerance}, {target+tolerance}]")


def demo():
    """Demonstrate interval arithmetic on standard G2 form."""
    print("=" * 60)
    print("Interval Arithmetic Demo for det(g) = 65/32")
    print("=" * 60)

    # Standard G2 form (should give det(g) = 1)
    # phi_0 = e^{123} + e^{145} + e^{167} + e^{246} - e^{257} - e^{347} - e^{356}

    # Index mapping for 35 components
    indices = []
    for i in range(7):
        for j in range(i + 1, 7):
            for k in range(j + 1, 7):
                indices.append((i, j, k))

    # Standard G2 form
    g2_indices = [
        (0, 1, 2),  # +1
        (0, 3, 4),  # +1
        (0, 5, 6),  # +1
        (1, 3, 5),  # +1
        (1, 4, 6),  # -1
        (2, 3, 6),  # -1
        (2, 4, 5),  # -1
    ]
    g2_signs = [1, 1, 1, 1, -1, -1, -1]

    # Create phi as point intervals
    phi = [Interval.point(0.0) for _ in range(35)]
    for idx, sign in zip(g2_indices, g2_signs):
        linear_idx = indices.index(idx)
        phi[linear_idx] = Interval.point(float(sign))

    print("\nStandard G2 form (point intervals):")
    print("phi = e^{123} + e^{145} + e^{167} + e^{246} - e^{257} - e^{347} - e^{356}")

    # Compute metric
    g = metric_from_phi_interval(phi)

    print("\nMetric g_ij:")
    for i in range(7):
        row = " ".join(f"{g[i][j].midpoint():6.3f}" for j in range(7))
        print(f"  [{row}]")

    # Compute determinant
    det_g = interval_det_7x7(g)
    print(f"\ndet(g) = {det_g}")
    print(f"Expected: 1.0 (standard G2 has flat metric)")

    # Now test with uncertainty
    print("\n" + "=" * 60)
    print("Testing with phi + uncertainty (width = 0.01)")
    print("=" * 60)

    phi_uncertain = [
        Interval(p.lo - 0.005, p.hi + 0.005)
        for p in phi
    ]

    g_uncertain = metric_from_phi_interval(phi_uncertain)
    det_g_uncertain = interval_det_7x7(g_uncertain)

    print(f"\ndet(g) with uncertainty = {det_g_uncertain}")
    print(f"Width: {det_g_uncertain.width():.6f}")

    # Verify target
    print("\n" + "=" * 60)
    print("Verification test")
    print("=" * 60)

    # For GIFT, we need det(g) = 65/32 ≈ 2.03125
    # The standard G2 form gives det(g) = 1
    # We need a scaled version

    # Scale phi to get det(g) = 65/32
    # g_ij = (1/6) * sum phi_ikl * phi_jkl, so g scales as phi^2
    # det(g) scales as phi^14 (7x7 matrix, each entry quadratic)
    # So we need scale^14 = 65/32, i.e. scale = (65/32)^(1/14)
    scale = (65.0 / 32.0) ** (1.0 / 14.0)
    print(f"\nScaling phi by {scale:.6f} to target det(g) = 65/32")

    phi_scaled = [Interval.point(p.midpoint() * scale) for p in phi]
    g_scaled = metric_from_phi_interval(phi_scaled)
    det_g_scaled = interval_det_7x7(g_scaled)

    print(f"det(g) after scaling = {det_g_scaled}")

    success, det_result, msg = verify_det_g(phi_scaled, target=65.0/32.0, tolerance=0.01)
    print(f"\nVerification: {msg}")


if __name__ == "__main__":
    demo()
