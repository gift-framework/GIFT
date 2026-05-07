"""Ranking analysis: where does the GIFT formula sit among all candidates?"""

from __future__ import annotations
from .pareto import pareto_frontier, is_on_frontier


def rank_gift_formula(gift_scores: dict, all_scores: list[dict],
                      metric: str = "total") -> dict:
    total = len(all_scores)
    if total == 0:
        return {"rank_by_error": 0, "rank_by_composite": 0, "rank_by_complexity": 0,
                "total_formulas": 0, "percentile_error": 0.0,
                "percentile_composite": 0.0, "on_pareto": False}

    rank_err = sum(1 for s in all_scores if s["err"] < gift_scores["err"]) + 1
    rank_comp = sum(1 for s in all_scores if s["total"] < gift_scores["total"]) + 1
    rank_cx = sum(1 for s in all_scores if s["comp"] < gift_scores["comp"]) + 1

    frontier = pareto_frontier(all_scores)
    on_pareto = is_on_frontier(gift_scores, frontier)

    return {"rank_by_error": rank_err, "rank_by_composite": rank_comp,
            "rank_by_complexity": rank_cx, "total_formulas": total,
            "percentile_error": rank_err / total,
            "percentile_composite": rank_comp / total, "on_pareto": on_pareto}
