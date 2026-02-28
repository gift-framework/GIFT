#!/usr/bin/env python3
"""
GIFT v3.3 -- Extended Statistical Analysis (Section 7.5 complement)
===================================================================
Four independent analyses to strengthen the statistical case:

1. Joint null model      -- eliminates Fisher independence assumption
2. Permutation test      -- tests formula-observable mapping significance
3. Leave-one-out (LOO)   -- robustness of (b2, b3) = (21, 77) optimality
4. Extended grammar enum -- all dimensionless observables ranked

Output: extended_analysis_results.json + console report
Runtime: ~5-10 min on CPU (no GPU needed)
"""

import math, random, time, json, sys, os
import numpy as np
from pathlib import Path

# ============================================================
# 1. OBSERVABLE DATABASE (26 testable dimensionless observables)
# ============================================================
# Each entry: (name, gift_value, exp_value, obs_class, formula_as_func_of_b2_b3)
# Constants (fixed, independent of b2/b3 choice):
DIM_G2 = 14; DIM_K7 = 7; DIM_E8 = 248; RANK_E8 = 8
DIM_J3O = 27; DIM_F4 = 52; DIM_E6 = 78; DIM_E7 = 133
FUND_E7 = 56; PSL_2_7 = 168; WEYL = 5; P2 = 2
SQRT2 = math.sqrt(2); PHI = (1 + math.sqrt(5)) / 2
LN2 = math.log(2)

def _zeta(s):
    """Approximate Riemann zeta for integer s >= 2."""
    return sum(1.0 / n**s for n in range(1, 100000))

ZETA5 = _zeta(5)
ZETA11 = _zeta(11)

# Formulas as functions of (b2, b3) -- returns predicted value
def f_N_gen(b2, b3):
    if b3 == b2: return float('nan')
    return RANK_E8 * b2 / (b3 - b2)

def f_sin2_theta_W(b2, b3): return b2 / (b3 + DIM_G2)
def f_alpha_s(b2, b3): return SQRT2 / (DIM_G2 - P2)  # fixed
def f_Q_Koide(b2, b3): return DIM_G2 / b2 if b2 != 0 else float('nan')
def f_m_tau_over_m_e(b2, b3): return DIM_K7 + 10 * DIM_E8 + 10 * (b2 + b3 + 1)
def f_m_mu_over_m_e(b2, b3): return DIM_J3O ** PHI  # fixed
def f_m_s_over_m_d(b2, b3): return P2**2 * WEYL  # fixed
def f_delta_CP(b2, b3): return DIM_K7 * DIM_G2 + (b2 + b3 + 1)
def f_theta_12(b2, b3):
    H = b2 + b3 + 1
    delta = 2 * math.pi / WEYL**2
    gamma = (2 * RANK_E8 + 5 * H) / (10 * DIM_G2 + 3 * DIM_E8)
    if gamma <= 0: return float('nan')
    return math.degrees(math.atan(math.sqrt(delta / gamma)))
def f_theta_13(b2, b3): return 180.0 / b2 if b2 != 0 else float('nan')
def f_theta_23(b2, b3):
    H = b2 + b3 + 1
    if H == 0: return float('nan')
    arg = (b3 - P2) / H
    if abs(arg) >= 1: return float('nan')
    return math.degrees(math.asin(arg))
def f_lambda_H(b2, b3):
    N = f_N_gen(b2, b3)
    if math.isnan(N): return float('nan')
    v = DIM_G2 + N
    if v < 0: return float('nan')
    return math.sqrt(v) / 2**WEYL
def f_m_b_over_m_t(b2, b3): return 1.0 / (2 * b2) if b2 != 0 else float('nan')
def f_Omega_DE(b2, b3):
    H = b2 + b3 + 1
    if H == 0: return float('nan')
    return LN2 * (b2 + b3) / H
def f_n_s(b2, b3): return ZETA11 / ZETA5  # fixed
def f_m_H_over_m_W(b2, b3):
    N = f_N_gen(b2, b3)
    if math.isnan(N): return float('nan')
    return (N + DIM_E6) / DIM_F4
def f_m_W_over_m_Z(b2, b3): return (2*b2 - WEYL) / (2*b2) if b2 != 0 else float('nan')
def f_m_H_over_m_t(b2, b3): return FUND_E7 / b3 if b3 != 0 else float('nan')
def f_sin2_theta_12_CKM(b2, b3): return FUND_E7 / DIM_E8  # fixed
def f_A_Wolfenstein(b2, b3):
    H = b2 + b3 + 1
    if H == 0: return float('nan')
    return (WEYL + DIM_E6) / H
def f_sin2_theta_23_CKM(b2, b3): return DIM_K7 / PSL_2_7  # fixed
def f_Omega_DM_over_Omega_b(b2, b3): return (1 + 2*b2) / RANK_E8
def f_h_Hubble(b2, b3): return (PSL_2_7 - 1) / DIM_E8  # fixed
def f_Omega_b_over_Omega_m(b2, b3):
    N = f_N_gen(b2, b3)
    if math.isnan(N): return float('nan')
    denom = b2 + DIM_G2 - N
    if denom == 0: return float('nan')
    return WEYL / denom
def f_sigma_8(b2, b3):
    N = f_N_gen(b2, b3)
    if math.isnan(N): return float('nan')
    denom = 2 * b2
    if denom == 0: return float('nan')
    return (P2 + b2 + DIM_G2 - N) / denom
def f_Y_p(b2, b3):
    denom = b3 - DIM_G2 - P2
    if denom == 0: return float('nan')
    return (1 + DIM_G2) / denom

def f_alpha_inv(b2, b3):
    H = b2 + b3 + 1
    N = f_N_gen(b2, b3)
    if math.isnan(N): return float('nan')
    denom = b2 + DIM_G2 - N
    if denom == 0: return float('nan')
    return (DIM_E8 + RANK_E8) / 2 + H / 11 + 65 / (32 * (b3 - DIM_G2 - P2)) if (b3 - DIM_G2 - P2) != 0 else float('nan')

def f_m_c_over_m_s(b2, b3):
    if b2 == 0: return float('nan')
    return (DIM_E8 - P2) / b2

# Full observable list
OBSERVABLES = [
    ("N_gen",                3.0,       3.0,      "A", f_N_gen),
    ("sin2_theta_W",         3/13,      0.23122,  "B", f_sin2_theta_W),
    ("alpha_s",              SQRT2/12,  0.1179,   "B", f_alpha_s),
    ("Q_Koide",              2/3,       0.666661, "B", f_Q_Koide),
    ("m_tau_over_m_e",       3477.0,    3477.15,  "A", f_m_tau_over_m_e),
    ("m_mu_over_m_e",        DIM_J3O**PHI, 206.768, "C", f_m_mu_over_m_e),
    ("m_s_over_m_d",         20.0,      20.0,     "A", f_m_s_over_m_d),
    ("delta_CP",             197.0,     197.0,    "D", f_delta_CP),
    ("theta_12",             33.40,     33.41,    "D", f_theta_12),
    ("theta_13",             8.571,     8.54,     "D", f_theta_13),
    ("theta_23",             49.25,     49.3,     "D", f_theta_23),
    ("lambda_H",             math.sqrt(17)/32, 0.129, "B", f_lambda_H),
    ("m_b_over_m_t",         1/42,      0.024,    "B", f_m_b_over_m_t),
    ("Omega_DE",             LN2*98/99, 0.6847,   "B", f_Omega_DE),
    ("n_s",                  ZETA11/ZETA5, 0.9649, "E", f_n_s),
    ("m_H_over_m_W",         81/52,     1.558,    "C", f_m_H_over_m_W),
    ("m_W_over_m_Z",         37/42,     0.8815,   "B", f_m_W_over_m_Z),
    ("m_H_over_m_t",         56/77,     0.725,    "C", f_m_H_over_m_t),
    ("sin2_theta_12_CKM",   56/248,    0.2250,   "B", f_sin2_theta_12_CKM),
    ("A_Wolfenstein",        83/99,     0.836,    "B", f_A_Wolfenstein),
    ("sin2_theta_23_CKM",   7/168,     0.0412,   "B", f_sin2_theta_23_CKM),
    ("Omega_DM_over_Omega_b", 43/8,    5.375,    "C", f_Omega_DM_over_Omega_b),
    ("h_Hubble",             167/248,   0.674,    "B", f_h_Hubble),
    ("Omega_b_over_Omega_m", 5/32,     0.157,    "B", f_Omega_b_over_Omega_m),
    ("sigma_8",              34/42,     0.811,    "B", f_sigma_8),
    ("Y_p",                  15/61,     0.245,    "B", f_Y_p),
    ("alpha_inv",            137.0333,  137.036,  "C", f_alpha_inv),
    ("m_c_over_m_s",         246/21,    11.7,     "C", f_m_c_over_m_s),
]

def rel_dev(pred, exp):
    """Relative deviation in percent."""
    if abs(exp) < 1e-15: return 0.0 if abs(pred) < 1e-15 else 100.0
    return abs(pred - exp) / abs(exp) * 100.0

def mean_dev_pct(preds, exps):
    """Mean relative deviation in percent."""
    devs = [rel_dev(p, e) for p, e in zip(preds, exps)]
    return np.mean(devs)

# ============================================================
# Compute GIFT baseline
# ============================================================
GIFT_PREDS = np.array([o[1] for o in OBSERVABLES])
EXP_VALS = np.array([o[2] for o in OBSERVABLES])
OBS_CLASSES = [o[3] for o in OBSERVABLES]
OBS_NAMES = [o[0] for o in OBSERVABLES]
N_OBS = len(OBSERVABLES)

GIFT_DEVS = np.array([rel_dev(p, e) for p, e in zip(GIFT_PREDS, EXP_VALS)])
GIFT_MEAN_DEV = float(np.mean(GIFT_DEVS))

print(f"GIFT baseline: {N_OBS} observables, mean deviation = {GIFT_MEAN_DEV:.4f}%")
print(f"  Max deviation: {np.max(GIFT_DEVS):.3f}% ({OBS_NAMES[np.argmax(GIFT_DEVS)]})")
print(f"  Median deviation: {np.median(GIFT_DEVS):.4f}%")
print()

# ============================================================
# ANALYSIS 1: Joint Null Model
# ============================================================
def run_joint_null_model(n_trials=100_000, seed=42):
    """Generate random formula value sets, compute mean deviation.

    For each trial, we draw a random value for each observable from a
    distribution representing 'what the grammar could produce'. We use
    a simple model: for each observable class, sample uniformly from
    the range of values formulas in that class typically produce.
    """
    print("=" * 60)
    print("ANALYSIS 1: Joint Null Model")
    print("=" * 60)

    rng = np.random.default_rng(seed)

    # Class-specific value ranges (based on grammar output distributions)
    # These are conservative ranges for each class
    class_ranges = {
        "A": (1, 10000),       # Integers
        "B": (0.001, 0.999),   # Ratios in [0,1]
        "C": (0.1, 10000),     # Positive ratios
        "D": (0, 360),         # Angles in degrees
        "E": (0.001, 10),      # Transcendental ratios
    }

    # Group observables by class
    class_indices = {}
    for i, cls in enumerate(OBS_CLASSES):
        class_indices.setdefault(cls, []).append(i)

    n_beat = 0
    null_mean_devs = np.zeros(n_trials)

    t0 = time.time()
    for trial in range(n_trials):
        random_preds = np.zeros(N_OBS)
        for cls, indices in class_indices.items():
            lo, hi = class_ranges[cls]
            if cls == "A":
                # Integer class: sample integers
                random_preds[indices] = rng.integers(int(lo), int(hi) + 1, size=len(indices)).astype(float)
            elif cls == "D":
                # Angles: uniform in [0, 360]
                random_preds[indices] = rng.uniform(lo, hi, size=len(indices))
            else:
                # Log-uniform for ratios (more realistic for algebraic formulas)
                log_lo, log_hi = math.log(lo), math.log(hi)
                random_preds[indices] = np.exp(rng.uniform(log_lo, log_hi, size=len(indices)))

        devs = np.abs(random_preds - EXP_VALS) / np.maximum(np.abs(EXP_VALS), 1e-15) * 100
        null_mean_devs[trial] = np.mean(devs)
        if null_mean_devs[trial] <= GIFT_MEAN_DEV:
            n_beat += 1

    elapsed = time.time() - t0
    p_joint = n_beat / n_trials

    # Upper bound via Wilson score interval
    if n_beat == 0:
        p_upper = 3.0 / n_trials  # 95% CL upper bound for 0 events
    else:
        p_upper = p_joint + 1.96 * math.sqrt(p_joint * (1 - p_joint) / n_trials)

    print(f"  Trials: {n_trials:,}")
    print(f"  GIFT mean deviation: {GIFT_MEAN_DEV:.4f}%")
    print(f"  Null mean deviation (median): {np.median(null_mean_devs):.1f}%")
    print(f"  Null trials beating GIFT: {n_beat}")
    print(f"  Joint p-value: {'< ' + f'{p_upper:.2e}' if n_beat == 0 else f'{p_joint:.2e}'}")
    print(f"  Time: {elapsed:.1f}s")
    print()

    return {
        "n_trials": n_trials,
        "gift_mean_dev": GIFT_MEAN_DEV,
        "null_median_dev": float(np.median(null_mean_devs)),
        "null_mean_dev": float(np.mean(null_mean_devs)),
        "n_beat": n_beat,
        "p_joint": p_joint,
        "p_upper_95CL": p_upper,
        "elapsed_s": round(elapsed, 1),
    }


# ============================================================
# ANALYSIS 2: Permutation Test
# ============================================================
def run_permutation_test(n_trials=1_000_000, seed=123):
    """Permute GIFT values among observables, compute mean deviation."""
    print("=" * 60)
    print("ANALYSIS 2: Permutation Test")
    print("=" * 60)

    rng = np.random.default_rng(seed)

    # --- 2a: Global permutation ---
    n_beat_global = 0
    t0 = time.time()
    for _ in range(n_trials):
        perm = rng.permutation(N_OBS)
        shuffled_preds = GIFT_PREDS[perm]
        devs = np.abs(shuffled_preds - EXP_VALS) / np.maximum(np.abs(EXP_VALS), 1e-15) * 100
        if np.mean(devs) <= GIFT_MEAN_DEV:
            n_beat_global += 1
    t_global = time.time() - t0
    p_global = n_beat_global / n_trials

    print(f"  Global permutation ({n_trials:,} trials):")
    print(f"    Trials beating GIFT: {n_beat_global}")
    p_str = f"< {3.0/n_trials:.2e}" if n_beat_global == 0 else f"{p_global:.2e}"
    print(f"    p-value: {p_str}")
    print(f"    Time: {t_global:.1f}s")

    # --- 2b: Within-class permutation ---
    n_beat_class = 0
    t0 = time.time()
    class_groups = {}
    for i, cls in enumerate(OBS_CLASSES):
        class_groups.setdefault(cls, []).append(i)

    for _ in range(n_trials):
        shuffled_preds = GIFT_PREDS.copy()
        for cls, indices in class_groups.items():
            if len(indices) > 1:
                idx_arr = np.array(indices)
                perm = rng.permutation(len(indices))
                shuffled_preds[idx_arr] = GIFT_PREDS[idx_arr[perm]]
        devs = np.abs(shuffled_preds - EXP_VALS) / np.maximum(np.abs(EXP_VALS), 1e-15) * 100
        if np.mean(devs) <= GIFT_MEAN_DEV:
            n_beat_class += 1
    t_class = time.time() - t0
    p_class = n_beat_class / n_trials

    print(f"  Within-class permutation ({n_trials:,} trials):")
    print(f"    Trials beating GIFT: {n_beat_class}")
    p_str2 = f"< {3.0/n_trials:.2e}" if n_beat_class == 0 else f"{p_class:.2e}"
    print(f"    p-value: {p_str2}")
    print(f"    Time: {t_class:.1f}s")
    print()

    return {
        "n_trials": n_trials,
        "global": {
            "n_beat": n_beat_global,
            "p_value": p_global,
            "p_upper_95CL": 3.0/n_trials if n_beat_global == 0 else p_global + 1.96*math.sqrt(p_global*(1-p_global)/n_trials),
        },
        "within_class": {
            "n_beat": n_beat_class,
            "p_value": p_class,
            "p_upper_95CL": 3.0/n_trials if n_beat_class == 0 else p_class + 1.96*math.sqrt(p_class*(1-p_class)/n_trials),
        },
        "elapsed_s": round(t_global + t_class, 1),
    }


# ============================================================
# ANALYSIS 3: Leave-One-Out Cross-Validation
# ============================================================
def run_leave_one_out(b2_range=(1, 100), b3_range=(1, 200)):
    """For each observable, remove it and find optimal (b2, b3)."""
    print("=" * 60)
    print("ANALYSIS 3: Leave-One-Out Cross-Validation")
    print("=" * 60)

    formulas = [o[4] for o in OBSERVABLES]

    # Build grid
    b2_grid = np.arange(b2_range[0], b2_range[1] + 1)
    b3_grid = np.arange(b3_range[0], b3_range[1] + 1)

    t0 = time.time()

    # Precompute all predictions for all (b2, b3) pairs
    n_b2 = len(b2_grid)
    n_b3 = len(b3_grid)

    # For each observable and each (b2, b3), compute predicted value
    # Shape: (N_OBS, n_b2, n_b3)
    all_preds = np.full((N_OBS, n_b2, n_b3), np.nan)
    for k in range(N_OBS):
        f = formulas[k]
        for i, b2 in enumerate(b2_grid):
            for j, b3 in enumerate(b3_grid):
                try:
                    val = f(int(b2), int(b3))
                    if math.isfinite(val):
                        all_preds[k, i, j] = val
                except (ValueError, ZeroDivisionError, OverflowError):
                    pass

    # Compute deviations: (N_OBS, n_b2, n_b3)
    exp_vals_3d = EXP_VALS.reshape(-1, 1, 1)
    all_devs = np.abs(all_preds - exp_vals_3d) / np.maximum(np.abs(exp_vals_3d), 1e-15) * 100

    # Full mean deviation landscape
    full_mean = np.nanmean(all_devs, axis=0)  # (n_b2, n_b3)

    # Find global optimum
    valid_mask = np.isfinite(full_mean)
    full_mean_masked = np.where(valid_mask, full_mean, np.inf)
    best_idx = np.unravel_index(np.argmin(full_mean_masked), full_mean_masked.shape)
    best_b2_full = int(b2_grid[best_idx[0]])
    best_b3_full = int(b3_grid[best_idx[1]])
    best_dev_full = float(full_mean_masked[best_idx])

    # GIFT's position
    gift_i = int(np.searchsorted(b2_grid, 21))
    gift_j = int(np.searchsorted(b3_grid, 77))
    gift_dev = float(full_mean[gift_i, gift_j]) if gift_i < n_b2 and gift_j < n_b3 else float('nan')

    print(f"  Full set: optimal (b2, b3) = ({best_b2_full}, {best_b3_full}), "
          f"mean dev = {best_dev_full:.4f}%")
    print(f"  GIFT (21, 77): mean dev = {gift_dev:.4f}%")
    is_optimal = (best_b2_full == 21 and best_b3_full == 77)
    print(f"  GIFT is global optimum: {is_optimal}")
    print()

    # LOO analysis
    loo_results = []
    n_still_optimal = 0

    for k in range(N_OBS):
        # Remove observable k
        remaining = list(range(N_OBS))
        remaining.remove(k)

        loo_mean = np.nanmean(all_devs[remaining], axis=0)
        loo_masked = np.where(np.isfinite(loo_mean), loo_mean, np.inf)
        loo_best_idx = np.unravel_index(np.argmin(loo_masked), loo_masked.shape)
        loo_best_b2 = int(b2_grid[loo_best_idx[0]])
        loo_best_b3 = int(b3_grid[loo_best_idx[1]])
        loo_best_dev = float(loo_masked[loo_best_idx])

        optimal = (loo_best_b2 == 21 and loo_best_b3 == 77)
        if optimal:
            n_still_optimal += 1

        loo_results.append({
            "removed": OBS_NAMES[k],
            "optimal_b2": loo_best_b2,
            "optimal_b3": loo_best_b3,
            "optimal_dev": round(loo_best_dev, 4),
            "gift_still_optimal": optimal,
        })

    elapsed = time.time() - t0

    print(f"  LOO results: (21, 77) optimal in {n_still_optimal}/{N_OBS} cases")
    for r in loo_results:
        status = "OK" if r["gift_still_optimal"] else f"SHIFTED to ({r['optimal_b2']}, {r['optimal_b3']})"
        print(f"    Remove {r['removed']:30s} -> {status} (dev={r['optimal_dev']:.4f}%)")

    print(f"  Time: {elapsed:.1f}s")
    print()

    return {
        "full_optimal": {"b2": best_b2_full, "b3": best_b3_full, "dev": best_dev_full},
        "gift_dev": gift_dev,
        "gift_is_global_optimum": is_optimal,
        "n_still_optimal": n_still_optimal,
        "n_total": N_OBS,
        "stability_fraction": n_still_optimal / N_OBS,
        "loo_details": loo_results,
        "elapsed_s": round(elapsed, 1),
    }


# ============================================================
# ANALYSIS 4: Extended Grammar Enumeration
# ============================================================
def run_extended_grammar():
    """Enumerate formulas for all dimensionless observables and rank GIFT."""
    print("=" * 60)
    print("ANALYSIS 4: Extended Grammar Enumeration")
    print("=" * 60)

    # Try to import the selection module
    sel_dir = Path(__file__).resolve().parent / "selection"
    sys.path.insert(0, str(sel_dir.parent))

    try:
        from selection.config import INVARIANTS, TRANSCENDENTALS, COMPLEXITY_BUDGETS, MAX_DEPTH
        from selection.search.enumerator import enumerate_formulas
        from selection.scoring.error import relative_error
        from selection.grammar.ast_node import ASTNode
    except ImportError as e:
        print(f"  WARNING: Could not import selection module: {e}")
        print(f"  Skipping extended grammar enumeration.")
        return {"error": str(e)}

    # Load GIFT formula ASTs
    data_dir = sel_dir / "data"
    gift_formulas_path = data_dir / "gift_formulas.json"
    if not gift_formulas_path.exists():
        print(f"  WARNING: {gift_formulas_path} not found. Skipping.")
        return {"error": "gift_formulas.json not found"}

    with open(gift_formulas_path) as f:
        gift_data = json.load(f)
    gift_formulas = gift_data["formulas"]

    # Load observables
    obs_path = data_dir / "observables.json"
    with open(obs_path) as f:
        obs_list = json.load(f)
    obs_dict = {o["name"]: o for o in obs_list}

    results = []
    total_formulas = 0
    n_rank1 = 0
    n_top3 = 0
    n_tested = 0

    t0 = time.time()

    for obs_name, gift_f in gift_formulas.items():
        if obs_name not in obs_dict:
            continue
        obs = obs_dict[obs_name]
        obs_class = obs.get("obs_class")
        if not obs_class:
            continue

        y_exp = obs["experimental"]
        gift_ast = ASTNode.from_json(gift_f["ast"])
        gift_val = gift_ast.evaluate_float(INVARIANTS, TRANSCENDENTALS)
        if not math.isfinite(gift_val):
            gift_val = gift_f.get("predicted", float('nan'))
        gift_err = relative_error(gift_val, y_exp)

        max_comp = COMPLEXITY_BUDGETS.get(obs_class, 12)

        print(f"  {obs_name} (class {obs_class})...", end=" ", flush=True)

        # Enumerate with ALL invariants (not just primary)
        try:
            formulas = enumerate_formulas(
                obs_class=obs_class,
                target_value=y_exp,
                max_complexity=max_comp,
                max_depth=MAX_DEPTH,
                invariant_set="all",
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        n_formulas = len(formulas)
        total_formulas += n_formulas

        # Rank GIFT by error
        n_better = sum(1 for val, _ in formulas
                       if relative_error(val, y_exp) < gift_err - 1e-12)
        rank = n_better + 1

        n_tested += 1
        if rank == 1: n_rank1 += 1
        if rank <= 3: n_top3 += 1

        results.append({
            "observable": obs_name,
            "obs_class": obs_class,
            "gift_value": round(gift_val, 6),
            "exp_value": y_exp,
            "gift_error_pct": round(gift_err * 100, 4),
            "search_space": n_formulas,
            "gift_rank": rank,
        })

        print(f"{n_formulas} formulas, GIFT rank #{rank}")

    elapsed = time.time() - t0

    print()
    print(f"  Summary: {n_tested} observables tested")
    print(f"  Total unique formulas: {total_formulas:,}")
    print(f"  Rank #1: {n_rank1}/{n_tested} ({100*n_rank1/n_tested:.0f}%)")
    print(f"  Top 3: {n_top3}/{n_tested} ({100*n_top3/n_tested:.0f}%)")
    print(f"  Time: {elapsed:.1f}s")
    print()

    return {
        "n_tested": n_tested,
        "total_formulas": total_formulas,
        "n_rank1": n_rank1,
        "n_top3": n_top3,
        "pct_rank1": round(100 * n_rank1 / max(n_tested, 1), 1),
        "pct_top3": round(100 * n_top3 / max(n_tested, 1), 1),
        "details": results,
        "elapsed_s": round(elapsed, 1),
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("GIFT v3.3 Extended Statistical Analysis")
    print("=" * 60)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Observables: {N_OBS}")
    print()

    results = {}

    # Analysis 1: Joint Null Model
    results["joint_null_model"] = run_joint_null_model(n_trials=200_000)

    # Analysis 2: Permutation Test
    results["permutation_test"] = run_permutation_test(n_trials=500_000)

    # Analysis 3: Leave-One-Out
    results["leave_one_out"] = run_leave_one_out()

    # Analysis 4: Extended Grammar
    results["extended_grammar"] = run_extended_grammar()

    # Save results
    output_path = Path(__file__).resolve().parent / "extended_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    jnm = results["joint_null_model"]
    perm = results["permutation_test"]
    loo = results["leave_one_out"]
    ext = results["extended_grammar"]

    print(f"1. Joint null model:     p < {jnm['p_upper_95CL']:.2e}")
    p_g = perm['global']
    p_wc = perm['within_class']
    print(f"2. Permutation (global): p < {p_g['p_upper_95CL']:.2e}")
    print(f"   Permutation (class):  p < {p_wc['p_upper_95CL']:.2e}")
    print(f"3. LOO stability:        {loo['n_still_optimal']}/{loo['n_total']} "
          f"({loo['stability_fraction']*100:.0f}%)")
    if not isinstance(ext, dict) or "error" not in ext:
        print(f"4. Extended grammar:     {ext.get('n_rank1', '?')}/{ext.get('n_tested', '?')} rank #1, "
              f"{ext.get('n_top3', '?')}/{ext.get('n_tested', '?')} top-3")

    print(f"\nResults saved to: {output_path}")
