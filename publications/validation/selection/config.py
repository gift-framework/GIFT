"""
GIFT Selection Principle — Configuration & Constants.

All topological invariants of the K7 manifold and E8xE8 gauge structure.
These are the building blocks (atoms) of the formula grammar.
"""

import sympy

# === Primary topological invariants (cost 1) ===
INVARIANTS_PRIMARY = {
    "b0": 1,          # Zeroth Betti number
    "b2": 21,         # Second Betti number of K7
    "b3": 77,         # Third Betti number of K7
    "dim_G2": 14,     # Dimension of G2 holonomy group
    "dim_K7": 7,      # Dimension of K7 manifold
    "dim_E8": 248,    # Dimension of E8 Lie algebra
    "rank_E8": 8,     # Rank of E8
    "N_gen": 3,       # Number of generations
    "H_star": 99,     # b2 + b3 + 1
}

# === Derived invariants (cost 2) ===
INVARIANTS_DERIVED = {
    "p2": 2,              # Pontryagin class
    "kappa_T_inv": 61,    # b3 - dim_G2 - p2 = 77 - 14 - 2
    "det_g_num": 65,      # numerator of det(g) = 65/32
    "det_g_den": 32,      # denominator
    "Weyl": 5,            # From W(E8) factorization
    "tau_num": 3472,      # numerator of tau = 496*21/(27*99)
    "tau_den": 891,       # denominator = 27*99
    "D_bulk": 11,         # M-theory bulk dimension
    "dim_J3O": 27,        # Exceptional Jordan algebra dimension
    "dim_F4": 52,         # F4 Lie algebra dimension
    "dim_E6": 78,         # E6 Lie algebra dimension
    "dim_E7": 133,        # E7 Lie algebra dimension
    "fund_E7": 56,        # E7 fundamental representation
    "PSL_2_7": 168,       # |PSL(2,7)| order
    "alpha_sum": 13,      # rank_E8 + Weyl = 8 + 5
    "dim_E8xE8": 496,     # 2 * dim_E8
    "two_b2": 42,         # 2 * b2 structural invariant
}

# === All invariants combined ===
INVARIANTS = {**INVARIANTS_PRIMARY, **INVARIANTS_DERIVED}

# === Transcendental constants (higher cost) ===
TRANSCENDENTALS = {
    "pi": sympy.pi,
    "sqrt2": sympy.sqrt(2),
    "phi": (1 + sympy.sqrt(5)) / 2,   # Golden ratio
    "ln2": sympy.log(2),
    "zeta3": sympy.zeta(3),
    "zeta5": sympy.zeta(5),
    "zeta11": sympy.zeta(11),
}

# === Scoring weights (tunable) ===
SCORE_WEIGHTS = {
    "alpha": 1.0,    # error weight
    "beta": 0.5,     # complexity weight
    "gamma": 2.0,    # naturalness violation (heavy penalty)
    "delta": 0.3,    # fragility weight
    "eta": 0.4,      # redundancy bonus
}

# === Complexity budgets per observable class ===
COMPLEXITY_BUDGETS = {
    "A": 8,    # Integer observables
    "B": 12,   # Ratio in [0,1]
    "C": 15,   # Ratio > 0
    "D": 15,   # Angles
    "E": 20,   # Transcendental
}

# === Max enumeration depth ===
MAX_DEPTH = 3
MAX_COEFF = 5  # Integer leaves in [-5, 5] for v0.1

# === Pilot observables for v0.1 ===
PILOT_OBSERVABLES = [
    "Q_Koide",        # Class B
    "sin2_theta_W",   # Class B
    "N_gen",          # Class A
    "delta_CP",       # Class D
    "n_s",            # Class E
]
