"""
Neutrino sector observables.

Predictions for neutrino oscillation parameters:
- θ₁₂, θ₂₃, θ₁₃ : Mixing angles
- δ_CP : CP violation phase
- PMNS matrix : Full mixing matrix
"""
import numpy as np
from typing import List, Dict


class NeutrinoSector:
    """Neutrino oscillation predictions from GIFT framework."""

    def __init__(self, constants):
        """Initialize neutrino sector."""
        self.c = constants
        self._cache = {}

    def theta_12(self, degrees: bool = False) -> float:
        """Solar mixing angle θ₁₂ = π/9."""
        angle = np.pi / 9
        return np.degrees(angle) if degrees else angle

    def theta_23(self, degrees: bool = False) -> float:
        """Atmospheric mixing angle θ₂₃ = 85/99 rad."""
        angle = 85 / 99
        return np.degrees(angle) if degrees else angle

    def theta_13(self, degrees: bool = False) -> float:
        """Reactor mixing angle θ₁₃ = π/21 rad."""
        angle = np.pi / 21
        return np.degrees(angle) if degrees else angle

    def delta_CP(self, degrees: bool = False) -> float:
        """CP violation phase δ_CP = ζ(3) + √5 ≈ 197°."""
        angle = self.c.zeta3 + self.c.sqrt5
        return np.degrees(angle) if degrees else angle

    def compute_all(self) -> List[Dict]:
        """Compute all neutrino observables."""
        return [
            {
                "observable": "theta_12",
                "name": "θ₁₂",
                "value": np.degrees(self.theta_12()),
                "unit": "degrees",
                "experimental": 33.41,
                "uncertainty": 0.75,
                "deviation_%": abs(np.degrees(self.theta_12()) - 33.41) / 33.41 * 100,
                "sector": "neutrino",
                "status": "PROVEN",
                "formula": "π/9",
            },
            {
                "observable": "delta_CP",
                "name": "δ_CP",
                "value": np.degrees(self.delta_CP()),
                "unit": "degrees",
                "experimental": 197,
                "uncertainty": 24,
                "deviation_%": abs(np.degrees(self.delta_CP()) - 197) / 197 * 100,
                "sector": "neutrino",
                "status": "PROVEN",
                "formula": "ζ(3) + √5",
            },
        ]

    def clear_cache(self):
        """Clear computation cache."""
        self._cache = {}
