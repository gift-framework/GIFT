"""
End-to-end benchmark runner for the Selection Principle.

For each pilot observable:
  1. Enumerate admissible formulas
  2. Score all formulas + the GIFT formula
  3. Compute Pareto frontier and ranking
  4. Run null models
  5. Generate plots
  6. Produce report
"""

from __future__ import annotations
import json
import os
import sys
import time
import math
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from selection.config import (
    INVARIANTS, TRANSCENDENTALS, COMPLEXITY_BUDGETS,
    PILOT_OBSERVABLES, MAX_DEPTH,
)
from selection.grammar.ast_node import ASTNode
from selection.grammar.complexity import complexity
from selection.search.enumerator import enumerate_formulas
from selection.scoring.composite import composite_score
from selection.scoring.error import error_score
from selection.analysis.pareto import pareto_frontier
from selection.analysis.ranking import rank_gift_formula
from selection.analysis.null_model import (
    null_distribution_random, null_distribution_shuffled, compute_p_value,
)
from selection.analysis.visualize import (
    plot_pareto, plot_null_distribution, plot_rank_histogram,
)


def load_observables(data_dir: str) -> dict:
    """Load observables.json, keyed by name."""
    path = os.path.join(data_dir, "observables.json")
    with open(path) as f:
        obs_list = json.load(f)
    return {o["name"]: o for o in obs_list}


def load_gift_formulas(data_dir: str) -> dict:
    """Load gift_formulas.json."""
    path = os.path.join(data_dir, "gift_formulas.json")
    with open(path) as f:
        data = json.load(f)
    return data["formulas"]


def run_single_observable(
    obs_name: str,
    obs_data: dict,
    gift_formula: dict,
    output_dir: str,
    n_null_random: int = 5000,
    n_null_shuffled: int = 5000,
) -> dict:
    """Run the full selection analysis for one observable."""
    obs_class = obs_data["obs_class"]
    y_exp = obs_data["experimental"]
    sigma_y = obs_data["uncertainty"]
    max_comp = COMPLEXITY_BUDGETS.get(obs_class, 12)

    result = {
        "observable": obs_name,
        "obs_class": obs_class,
        "experimental": y_exp,
        "uncertainty": sigma_y,
    }

    # --- Step 1: Enumerate ---
    t0 = time.time()
    formulas = enumerate_formulas(
        obs_class=obs_class,
        target_value=y_exp,
        max_complexity=max_comp,
        max_depth=MAX_DEPTH,
        invariant_set="primary",
    )
    t_enum = time.time() - t0
    result["n_enumerated"] = len(formulas)
    result["t_enum_s"] = round(t_enum, 2)
    print(f"  Enumerated {len(formulas)} formulas in {t_enum:.1f}s")

    # --- Step 2: Score all formulas ---
    t0 = time.time()
    all_scores = []
    for val, node in formulas:
        scores = composite_score(
            node=node, y_exp=y_exp, sigma_y=sigma_y,
            obs_class=obs_class,
            invariants=INVARIANTS, transcendentals=TRANSCENDENTALS,
        )
        scores["formula_str"] = node.to_str()
        all_scores.append(scores)
    t_score = time.time() - t0
    result["t_score_s"] = round(t_score, 2)

    # --- Step 3: Score the GIFT formula ---
    gift_ast = ASTNode.from_json(gift_formula["ast"])
    gift_scores = composite_score(
        node=gift_ast, y_exp=y_exp, sigma_y=sigma_y,
        obs_class=obs_class,
        invariants=INVARIANTS, transcendentals=TRANSCENDENTALS,
    )
    gift_scores["formula_str"] = gift_ast.to_str()
    result["gift_scores"] = gift_scores
    print(f"  GIFT formula: {gift_ast.to_str()}")
    print(f"    predicted={gift_scores['predicted']:.6f}, error={gift_scores['err']:.4f}, "
          f"complexity={gift_scores['comp']:.1f}, total={gift_scores['total']:.4f}")

    # --- Step 4: Pareto frontier & ranking ---
    frontier = pareto_frontier(all_scores)
    ranking = rank_gift_formula(gift_scores, all_scores)
    result["frontier_size"] = len(frontier)
    result["ranking"] = ranking
    print(f"  Pareto frontier: {len(frontier)} formulas")
    print(f"  GIFT rank: #{ranking['rank_by_error']}/{ranking['total_formulas']} "
          f"(error), #{ranking['rank_by_composite']}/{ranking['total_formulas']} (composite)")
    print(f"  On Pareto frontier: {ranking['on_pareto']}")

    # --- Step 5: Null models ---
    obs_dir = os.path.join(output_dir, obs_name)
    os.makedirs(obs_dir, exist_ok=True)

    # Random null
    t0 = time.time()
    null_rand = null_distribution_random(
        obs_class=obs_class, y_exp=y_exp, sigma_y=sigma_y,
        n_random=n_null_random, max_complexity=max_comp,
    )
    t_null = time.time() - t0
    gift_err = gift_scores["err"]
    p_random = compute_p_value(gift_err, null_rand["errors"])
    result["null_random"] = {
        "n_valid": null_rand["n_valid"],
        "n_total": null_rand["n_total"],
        "mean_error": null_rand["mean_error"],
        "p_value": p_random,
    }
    result["t_null_random_s"] = round(t_null, 2)
    print(f"  Random null: p={p_random:.4f} ({null_rand['n_valid']}/{null_rand['n_total']} valid)")

    # Shuffled null
    t0 = time.time()
    null_shuf = null_distribution_shuffled(
        gift_ast=gift_ast, obs_class=obs_class,
        y_exp=y_exp, sigma_y=sigma_y,
        n_shuffles=n_null_shuffled,
    )
    t_null_shuf = time.time() - t0
    p_shuffled = compute_p_value(gift_err, null_shuf["errors"])
    result["null_shuffled"] = {
        "n_valid": null_shuf["n_valid"],
        "n_total": null_shuf["n_total"],
        "mean_error": null_shuf["mean_error"],
        "p_value": p_shuffled,
    }
    result["t_null_shuffled_s"] = round(t_null_shuf, 2)
    print(f"  Shuffled null: p={p_shuffled:.4f} ({null_shuf['n_valid']}/{null_shuf['n_total']} valid)")

    # --- Step 6: Plots ---
    plot_pareto(all_scores, frontier, gift_scores, obs_name, obs_dir)
    plot_null_distribution(null_rand["errors"], gift_err, p_random, obs_name, obs_dir, "random")
    plot_null_distribution(null_shuf["errors"], gift_err, p_shuffled, obs_name, obs_dir, "shuffled")
    plot_rank_histogram(all_scores, ranking["rank_by_error"], obs_name, obs_dir, "err")
    plot_rank_histogram(all_scores, ranking["rank_by_composite"], obs_name, obs_dir, "total")

    return result


def run_benchmark(
    pilot_only: bool = True,
    output_dir: str | None = None,
    n_null_random: int = 5000,
    n_null_shuffled: int = 5000,
) -> dict:
    """Run the full benchmark suite."""
    data_dir = str(Path(__file__).resolve().parent.parent / "data")
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "results")
    os.makedirs(output_dir, exist_ok=True)

    observables = load_observables(data_dir)
    gift_formulas = load_gift_formulas(data_dir)

    # Filter to pilot observables if requested
    if pilot_only:
        target_names = PILOT_OBSERVABLES
    else:
        target_names = [name for name, gf in gift_formulas.items()]

    results = []
    t_total = time.time()

    for obs_name in target_names:
        if obs_name not in gift_formulas:
            print(f"SKIP {obs_name}: no GIFT formula in gift_formulas.json")
            continue
        if obs_name not in observables:
            print(f"SKIP {obs_name}: not found in observables.json")
            continue

        print(f"\n{'='*60}")
        print(f"Observable: {obs_name}")
        print(f"{'='*60}")

        r = run_single_observable(
            obs_name=obs_name,
            obs_data=observables[obs_name],
            gift_formula=gift_formulas[obs_name],
            output_dir=output_dir,
            n_null_random=n_null_random,
            n_null_shuffled=n_null_shuffled,
        )
        results.append(r)

    t_total = time.time() - t_total

    summary = {
        "n_observables": len(results),
        "total_time_s": round(t_total, 1),
        "results": results,
    }

    # Save JSON results
    results_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GIFT Selection Principle Benchmark")
    parser.add_argument("--all", action="store_true", help="Run all observables (not just pilot)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--n-null", type=int, default=5000, help="Number of null samples")
    args = parser.parse_args()

    run_benchmark(
        pilot_only=not args.all,
        output_dir=args.output,
        n_null_random=args.n_null,
        n_null_shuffled=args.n_null,
    )
