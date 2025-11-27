"""
Cosmology sector observables.

Predictions for cosmological parameters:
- Ω_DE : Dark energy density parameter
- n_s : Scalar spectral index
"""
import numpy as np
from typing import List, Dict


class CosmologySector:
    """Cosmological predictions from GIFT framework."""

    def __init__(self, constants):
        """Initialize cosmology sector."""
        self.c = constants
        self._cache = {}

    def Omega_DE(self) -> float:
        """Dark energy density parameter Ω_DE = ln(2) ≈ 0.693."""
        return self.c.ln2

    def n_s(self) -> float:
        """Scalar spectral index n_s = ξ² where ξ = 5β₀/2."""
        return self.c.xi**2

    def compute_all(self) -> List[Dict]:
        """Compute all cosmology observables."""
        return [
            {
                "observable": "Omega_DE",
                "name": "Ω_DE",
                "value": self.Omega_DE(),
                "unit": "dimensionless",
                "experimental": 0.6847,
                "uncertainty": 0.0073,
                "deviation_%": abs(self.Omega_DE() - 0.6847) / 0.6847 * 100,
                "sector": "cosmology",
                "status": "PROVEN",
                "formula": "ln(2)",
            },
            {
                "observable": "n_s",
                "name": "n_s",
                "value": self.n_s(),
                "unit": "dimensionless",
                "experimental": 0.9649,
                "uncertainty": 0.0042,
                "deviation_%": abs(self.n_s() - 0.9649) / 0.9649 * 100,
                "sector": "cosmology",
                "status": "DERIVED",
                "formula": "ξ² = (5β₀/2)²",
            },
        ]

    def clear_cache(self):
        """Clear computation cache."""
        self._cache = {}
