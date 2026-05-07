"""Precision scoring."""

import math

def error_score(predicted: float, experimental: float, uncertainty: float = 0.0) -> float:
    if math.isnan(predicted) or math.isinf(predicted):
        return float('inf')
    diff = abs(predicted - experimental)
    if uncertainty > 0:
        return diff / uncertainty
    if abs(experimental) > 1e-15:
        return diff / abs(experimental)
    return diff

def relative_error(predicted: float, experimental: float) -> float:
    if abs(experimental) < 1e-15:
        return abs(predicted)
    return abs(predicted - experimental) / abs(experimental)
