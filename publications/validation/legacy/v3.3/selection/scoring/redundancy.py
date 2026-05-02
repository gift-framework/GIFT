"""Redundancy scoring: how many distinct formulas give the same value?"""

import math

def redundancy_score(target_value: float, formula_values: dict[float, int],
                     tolerance: float = 1e-10) -> float:
    count = sum(n for val, n in formula_values.items()
                if abs(val - target_value) < tolerance * max(abs(val), 1.0))
    return math.log(1 + count)
