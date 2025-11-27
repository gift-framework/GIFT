"""
GIFTpy - Geometric Information Field Theory

A Python package for computing Standard Model predictions from topological geometry.

Examples
--------
>>> import giftpy
>>> gift = giftpy.GIFT()
>>> alpha_s = gift.gauge.alpha_s()
>>> print(f"α_s(M_Z) = {alpha_s:.6f}")
α_s(M_Z) = 0.117900
"""

__version__ = "0.1.0"
__author__ = "GIFT Framework Collaboration"
__license__ = "MIT"

from .core.framework import GIFT
from .core.constants import CONSTANTS, TopologicalConstants

__all__ = [
    "GIFT",
    "CONSTANTS",
    "TopologicalConstants",
    "__version__",
]
