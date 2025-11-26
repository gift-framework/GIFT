"""
Figure registry for the professional visualization suite.
"""
from __future__ import annotations

from typing import Callable, Dict

from . import dimensional_flow, e8_root, precision_matrix

FigureFn = Callable[..., dict]


def registry() -> Dict[str, FigureFn]:
    """
    Map figure keys to their render functions.
    """
    return {
        "e8-root-system": e8_root.render,
        "dimensional-flow": dimensional_flow.render,
        "precision-matrix": precision_matrix.render,
    }







