"""Atom definitions with complexity costs."""

ATOM_COSTS = {
    # Primary invariants: cost 1
    "b0": 1, "b2": 1, "b3": 1, "dim_G2": 1, "dim_K7": 1,
    "dim_E8": 1, "rank_E8": 1, "N_gen": 1, "H_star": 1,
    # Derived invariants: cost 2
    "p2": 2, "kappa_T_inv": 2, "Weyl": 2, "D_bulk": 2,
    "dim_J3O": 2, "dim_F4": 2, "dim_E6": 2, "dim_E7": 2,
    "fund_E7": 2, "PSL_2_7": 2, "alpha_sum": 2,
    "det_g_num": 2, "det_g_den": 2, "dim_E8xE8": 2, "two_b2": 2,
    "tau_num": 2, "tau_den": 2,
    # Transcendentals: cost 4-7
    "pi": 4, "sqrt2": 4, "phi": 5, "ln2": 4,
    "zeta3": 6, "zeta5": 6, "zeta11": 7,
}

CLASS_ALLOWED_ATOMS = {
    "A": {"primary", "derived"},
    "B": {"primary", "derived"},
    "C": {"primary", "derived"},
    "D": {"primary", "derived", "transcendental_trig"},
    "E": {"primary", "derived", "transcendental"},
}

def get_allowed_atoms(obs_class: str) -> list[str]:
    allowed = CLASS_ALLOWED_ATOMS.get(obs_class, set())
    atoms = []
    for name, cost in ATOM_COSTS.items():
        if cost <= 1 and "primary" in allowed:
            atoms.append(name)
        elif cost == 2 and "derived" in allowed:
            atoms.append(name)
        elif cost >= 4 and name == "pi" and "transcendental_trig" in allowed:
            atoms.append(name)
        elif cost >= 4 and "transcendental" in allowed:
            atoms.append(name)
    return atoms
