"""Composite scoring: combine all scoring axes."""

from ..grammar.ast_node import ASTNode
from ..grammar.complexity import complexity_with_depth_penalty
from .error import error_score
from .naturalness import naturalness_score
from .fragility import fragility_score
from .redundancy import redundancy_score

DEFAULT_WEIGHTS = {
    "alpha": 1.0, "beta": 0.5, "gamma": 2.0, "delta": 0.3, "eta": 0.4,
}

def composite_score(node: ASTNode, y_exp: float, sigma_y: float,
                    obs_class: str, invariants: dict, transcendentals: dict,
                    formula_values: dict[float, int] | None = None,
                    weights: dict | None = None) -> dict:
    w = weights or DEFAULT_WEIGHTS
    predicted = node.evaluate_float(invariants, transcendentals)
    err = error_score(predicted, y_exp, sigma_y)
    comp = complexity_with_depth_penalty(node)
    unnat = naturalness_score(node, obs_class, predicted)
    frag = fragility_score(node, y_exp, invariants, transcendentals)
    redund = redundancy_score(predicted, formula_values) if formula_values else 0.0
    total = (w["alpha"] * err + w["beta"] * comp +
             w["gamma"] * unnat + w["delta"] * frag - w["eta"] * redund)
    return {"predicted": predicted, "err": err, "comp": comp,
            "unnat": unnat, "frag": frag, "redund": redund, "total": total}
