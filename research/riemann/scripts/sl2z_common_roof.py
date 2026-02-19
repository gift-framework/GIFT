#!/usr/bin/env python3
"""
SL(2,Z) as the Common Roof: Computational Exploration
=====================================================

This script explores the algebraic relationships between:
1. The Fibonacci matrix M
2. The G2 Cartan matrix C(G2)
3. Their connection via SL(2,Z)

Key questions:
- Does M^6 (h_G2 power) have special properties?
- How do trace(M^n) and trace(C(G2)^n) relate?
- Can we derive 31/21 from the matrix structure?
"""

import numpy as np
from fractions import Fraction
from sympy import Matrix, sqrt, simplify, symbols, expand, Rational, latex

print("=" * 70)
print("SL(2,Z) AS THE COMMON ROOF")
print("=" * 70)

# =============================================================================
# BASIC DEFINITIONS
# =============================================================================

print("\n" + "-" * 70)
print("1. BASIC DEFINITIONS")
print("-" * 70)

# Fibonacci matrix
M = np.array([[1, 1],
              [1, 0]])

# G2 Cartan matrix
C_G2 = np.array([[2, -1],
                 [-3, 2]])

# SL(2,Z) generators
S = np.array([[0, -1],
              [1, 0]])

T = np.array([[1, 1],
              [0, 1]])

print(f"\nFibonacci matrix M:")
print(f"  M = {M.tolist()}")
print(f"  det(M) = {int(np.linalg.det(M))}")

print(f"\nG2 Cartan matrix C(G2):")
print(f"  C(G2) = {C_G2.tolist()}")
print(f"  det(C(G2)) = {int(np.linalg.det(C_G2))}")

print(f"\nSL(2,Z) generators:")
print(f"  S = {S.tolist()}  (rotation)")
print(f"  T = {T.tolist()}  (shear)")

# =============================================================================
# FIBONACCI NUMBERS
# =============================================================================

print("\n" + "-" * 70)
print("2. FIBONACCI NUMBERS FROM MATRIX POWERS")
print("-" * 70)

def fibonacci(n):
    """Compute F_n using matrix power"""
    if n == 0:
        return 0
    Mn = np.linalg.matrix_power(M, n)
    return int(Mn[0, 1])

print("\nFibonacci sequence from M^n:")
F = [fibonacci(n) for n in range(15)]
print(f"F_0 to F_14: {F}")

# Lucas numbers from trace
def lucas(n):
    """L_n = trace(M^n) for n >= 1"""
    if n == 0:
        return 2
    Mn = np.linalg.matrix_power(M, n)
    return int(np.trace(Mn))

print("\nLucas numbers from trace(M^n):")
L = [lucas(n) for n in range(15)]
print(f"L_0 to L_14: {L}")

# =============================================================================
# M^6 ANALYSIS (h_G2 = 6)
# =============================================================================

print("\n" + "-" * 70)
print("3. M^6 ANALYSIS (h_G2 = 6)")
print("-" * 70)

h_G2 = 6

M6 = np.linalg.matrix_power(M, h_G2)
print(f"\nM^{h_G2} = M^6 =")
print(f"  [[{int(M6[0,0])}, {int(M6[0,1])}],")
print(f"   [{int(M6[1,0])}, {int(M6[1,1])}]]")
print(f"  = [[F_7, F_6], [F_6, F_5]]")
print(f"  = [[{F[7]}, {F[6]}], [{F[6]}, {F[5]}]]")

print(f"\nProperties of M^6:")
print(f"  det(M^6) = {int(np.linalg.det(M6))} (in SL(2,Z))")
print(f"  trace(M^6) = {int(np.trace(M6))} = L_6")

# Eigenvalues
phi = (1 + np.sqrt(5)) / 2
psi = (1 - np.sqrt(5)) / 2
print(f"\nEigenvalues of M:")
print(f"  phi = (1 + sqrt(5))/2 = {phi:.6f}")
print(f"  psi = (1 - sqrt(5))/2 = {psi:.6f}")
print(f"\nEigenvalues of M^6:")
print(f"  phi^6 = {phi**6:.6f}")
print(f"  psi^6 = {psi**6:.6f}")
print(f"  phi^6 + psi^6 = {phi**6 + psi**6:.6f} = L_6 = 18")

# =============================================================================
# C(G2) ANALYSIS
# =============================================================================

print("\n" + "-" * 70)
print("4. C(G2) CARTAN MATRIX ANALYSIS")
print("-" * 70)

# Eigenvalues
eigs_C = np.linalg.eigvals(C_G2)
print(f"\nEigenvalues of C(G2):")
print(f"  lambda_1 = 2 + sqrt(3) = {2 + np.sqrt(3):.6f}")
print(f"  lambda_2 = 2 - sqrt(3) = {2 - np.sqrt(3):.6f}")

print(f"\nPowers of C(G2):")
for n in range(1, 7):
    Cn = np.linalg.matrix_power(C_G2, n)
    print(f"  trace(C^{n}) = {int(np.trace(Cn))}")

print(f"\nCRITICAL: trace(C^2) = {int(np.trace(np.linalg.matrix_power(C_G2, 2)))} = dim(G2)!")

# =============================================================================
# THE 31/21 FORMULA
# =============================================================================

print("\n" + "-" * 70)
print("5. THE 31/21 FORMULA")
print("-" * 70)

k = h_G2

a_num = F[k+3] - F[k-2]  # F_9 - F_4 = 34 - 3 = 31
a_den = F[k+2]           # F_8 = 21

b_num = -F[k-1]          # -F_5 = -5... wait, let me recalculate
# Actually: a + b = 1, so b = 1 - a = 1 - 31/21 = -10/21

a = Fraction(a_num, a_den)
b = Fraction(-10, 21)

print(f"\nFor k = h_G2 = {k}:")
print(f"  F_{{k+3}} = F_9 = {F[9]}")
print(f"  F_{{k-2}} = F_4 = {F[4]}")
print(f"  F_{{k+2}} = F_8 = {F[8]}")
print(f"\n  a = (F_9 - F_4) / F_8 = ({F[9]} - {F[4]}) / {F[8]} = {a} = {float(a):.6f}")
print(f"  b = 1 - a = {b} = {float(b):.6f}")
print(f"\nVerification: a + b = {a + b}")

# =============================================================================
# SEARCHING FOR CONNECTIONS
# =============================================================================

print("\n" + "-" * 70)
print("6. SEARCHING FOR CONNECTIONS")
print("-" * 70)

print("\nLooking for 31 and 21 in various structures...")

# From Fibonacci
print(f"\nFibonacci:")
print(f"  F_8 = {F[8]} = 21 ✓")
print(f"  F_9 - F_4 = {F[9]} - {F[4]} = {F[9] - F[4]} = 31 ✓")

# From Lucas
print(f"\nLucas:")
print(f"  L_5 = {L[5]} = 11")
print(f"  L_6 = {L[6]} = 18")
print(f"  L_6 + 3 = 21? Yes: {L[6] + 3} ✓")

# From G2 dimensions
dim_G2 = 14
b2 = 21
b3 = 77

print(f"\nG2/K7 topology:")
print(f"  dim(G2) = {dim_G2}")
print(f"  b_2 = {b2} = F_8 ✓")
print(f"  dim(G2) + b_2 - 4 = {dim_G2 + b2 - 4} = 31 ✓")

# =============================================================================
# COMMUTATOR AND SUBGROUP STRUCTURE
# =============================================================================

print("\n" + "-" * 70)
print("7. SUBGROUP STRUCTURE")
print("-" * 70)

# Check if M^2 and C(G2) commute
M2 = np.linalg.matrix_power(M, 2)
comm = M2 @ C_G2 - C_G2 @ M2

print(f"\nM^2 = {M2.tolist()}")
print(f"\nM^2 @ C(G2) = {(M2 @ C_G2).tolist()}")
print(f"C(G2) @ M^2 = {(C_G2 @ M2).tolist()}")
print(f"\n[M^2, C(G2)] = M^2 @ C(G2) - C(G2) @ M^2 = {comm.tolist()}")
print(f"\nM^2 and C(G2) do NOT commute.")

# =============================================================================
# CHEBYSHEV POLYNOMIALS
# =============================================================================

print("\n" + "-" * 70)
print("8. CHEBYSHEV POLYNOMIAL CONNECTION")
print("-" * 70)

print("""
For A in SL(2,Z) with trace t:
  A^n = U_{n-1}(t/2) * A - U_{n-2}(t/2) * I

where U_n(x) is the Chebyshev polynomial of the second kind.
""")

def chebyshev_U(n, x):
    """Chebyshev polynomial of second kind"""
    if n == 0:
        return 1
    elif n == 1:
        return 2 * x
    else:
        # Recurrence: U_{n+1}(x) = 2x*U_n(x) - U_{n-1}(x)
        U_prev, U_curr = 1, 2*x
        for _ in range(2, n+1):
            U_prev, U_curr = U_curr, 2*x*U_curr - U_prev
        return U_curr

print(f"\nFor M^2 (trace = 3, t/2 = 3/2):")
for n in range(8):
    U_val = chebyshev_U(n, 1.5)
    print(f"  U_{n}(3/2) = {U_val:.1f}")

print(f"\nCompare to Fibonacci: F_{{2n+2}} for n=0,1,2,3,4,5,6,7:")
for n in range(8):
    print(f"  F_{2*n+2} = {F[2*n+2] if 2*n+2 < len(F) else 'N/A'}")

print(f"\nFor C(G2) (trace = 4, t/2 = 2):")
for n in range(8):
    U_val = chebyshev_U(n, 2)
    print(f"  U_{n}(2) = {U_val:.0f}")

# =============================================================================
# THE KEY INSIGHT: trace(M^6) and dim(G2)
# =============================================================================

print("\n" + "-" * 70)
print("9. THE KEY NUMERICAL COINCIDENCES")
print("-" * 70)

print(f"""
Key numerical coincidences:

1. trace(M^{h_G2}) = trace(M^6) = {int(np.trace(M6))} = L_6
2. trace(C(G2)^2) = {int(np.trace(C_G2 @ C_G2))} = dim(G2)
3. F_{h_G2+2} = F_8 = {F[8]} = b_2 (second Betti number of K7)
4. The cluster algebra period for G2 is h+2 = {h_G2+2} = F_6 = 8

The formula:
  a = (F_{{k+3}} - F_{{k-2}}) / F_{{k+2}}

with k = h_G2 = 6 gives:
  a = (F_9 - F_4) / F_8 = (34 - 3) / 21 = 31/21

The lags are:
  Lag 1 = F_6 = 8 = h_G2 + 2 = cluster period
  Lag 2 = F_8 = 21 = b_2

This suggests the Fibonacci structure emerges from G2 cluster algebra periodicity!
""")

# =============================================================================
# PRODUCT OF MATRICES: EXPLORING RELATIONS
# =============================================================================

print("\n" + "-" * 70)
print("10. EXPLORING MATRIX PRODUCTS")
print("-" * 70)

# Various products
print("\nProducts to explore:")

print(f"\nM^2 @ C(G2) @ M^2 =")
prod1 = M2 @ C_G2 @ M2
print(f"  {prod1.tolist()}")
print(f"  trace = {int(np.trace(prod1))}")

print(f"\nC(G2) @ M^6 @ C(G2)^(-1) (conjugation) =")
C_G2_inv = np.linalg.inv(C_G2).astype(int)
prod2 = C_G2 @ M6 @ C_G2_inv
print(f"  {prod2.tolist()}")
print(f"  trace = {int(np.trace(prod2))} (preserved under conjugation)")

# =============================================================================
# THE 31/21 FROM MATRIX PERSPECTIVE
# =============================================================================

print("\n" + "-" * 70)
print("11. CAN WE GET 31/21 FROM MATRIX ENTRIES?")
print("-" * 70)

print(f"""
Looking for 31 and 21 in matrix entries...

M^6 = [[13, 8], [8, 5]]
  - 13 + 8 = 21 = F_8 ✓
  - No 31 directly

M^7 = [[F_8, F_7], [F_7, F_6]] = [[21, 13], [13, 8]]
  - 21 + 13 - 3 = 31 ✓

M^8 = [[F_9, F_8], [F_8, F_7]] = [[34, 21], [21, 13]]
  - 34 - 3 = 31 ✓
  - This is F_9 - F_4

Key observation:
  31 = F_9 - F_4 = M^8[0,0] - F_4
  21 = F_8 = M^8[0,1] = M^8[1,0]

So 31/21 = (M^8[0,0] - F_4) / M^8[0,1]
""")

M8 = np.linalg.matrix_power(M, 8)
print(f"M^8 = {M8.tolist()}")
print(f"\n31/21 = (M^8[0,0] - F_4) / M^8[0,1]")
print(f"      = ({int(M8[0,0])} - {F[4]}) / {int(M8[0,1])}")
print(f"      = {int(M8[0,0]) - F[4]} / {int(M8[0,1])}")
print(f"      = {Fraction(int(M8[0,0]) - F[4], int(M8[0,1]))}")

# =============================================================================
# CONCLUSION
# =============================================================================

print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)

print("""
ESTABLISHED FACTS:
------------------
1. M^2 and C(G2) are both in SL(2,Z)
2. trace(M^6) = L_6 = 18 (Lucas number at Coxeter index)
3. trace(C(G2)^2) = 14 = dim(G2)
4. The formula a = 31/21 comes from Fibonacci at index k = h_G2 = 6
5. The lags 8 = F_6 and 21 = F_8 are Fibonacci numbers

PROPOSED MECHANISM:
------------------
The G2 cluster algebra has period h+2 = 8, which equals F_6.
This sets the first lag at 8.
The second lag 21 = F_8 maintains the Fibonacci structure.
The coefficient 31/21 = (F_9 - F_4)/F_8 emerges from the Fibonacci
closure property when k = h_G2.

WHAT'S MISSING:
---------------
- Direct algebraic derivation of 31/21 from C(G2)
- Connection to Hecke operators
- Proof that zeta zeros must satisfy this recurrence

NEXT STEPS:
-----------
1. Study Weng's zeta_G2 construction
2. Look for modular forms with Fibonacci Hecke eigenvalues
3. Investigate cluster algebras with Hecke actions
""")

print("\n" + "=" * 70)
print("Analysis complete.")
print("=" * 70)
