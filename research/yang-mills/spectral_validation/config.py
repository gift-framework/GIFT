"""
Configuration centralisée pour validation spectrale.

Ce fichier définit tous les paramètres de référence, variétés de test,
grilles de robustesse et critères PASS/FAIL pour la validation de:

    λ₁ × H* = 13 = dim(G₂) - 1

Référence: ROADMAP_SPECTRAL_VALIDATION.md
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

# =============================================================================
# CHEMINS
# =============================================================================

BASE_DIR = Path(__file__).parent
CALIBRATION_DIR = BASE_DIR / "calibration"
ROBUSTNESS_DIR = BASE_DIR / "robustness"
ANALYSIS_DIR = BASE_DIR / "analysis"

# =============================================================================
# PARAMÈTRES DE RÉFÉRENCE (V11)
# =============================================================================

@dataclass
class ReferenceParams:
    """Paramètres de référence validés dans V11."""
    N: int = 5000              # Nombre de points
    k_neighbors: int = 25      # k pour k-NN
    sigma_method: str = "auto" # ratio * sqrt(dim/k)
    laplacian_type: str = "symmetric"  # I - D^{-1/2} W D^{-1/2}

REFERENCE = ReferenceParams()

# =============================================================================
# VARIÉTÉS DE TEST
# =============================================================================

MANIFOLDS = {
    # Calibration: sphères avec λ₁ analytique connu
    "S3": {
        "dim": 3,
        "lambda1_exact": 3.0,  # λ₁(S³) = n(n+2)/R² avec n=1, R=1
        "multiplicity": 4,      # Multiplicité de λ₁
        "description": "3-sphère (calibration)",
    },
    "S7": {
        "dim": 7,
        "lambda1_exact": 7.0,  # λ₁(S⁷) = n(n+6)/R² avec n=1, R=1
        "multiplicity": 8,
        "description": "7-sphère (calibration, même dim que K₇)",
    },

    # Variété plate: tore (mode zéro)
    "T7": {
        "dim": 7,
        "lambda1_exact": 0.0,  # Tore plat → pas de gap
        "description": "7-tore (vérification mode zéro)",
    },

    # Variété cible: K₇ avec holonomie G₂
    "K7_GIFT": {
        "dim": 7,
        "H_star": 99,           # b₂ + b₃ + 1
        "b2": 21,               # 2ème nombre de Betti
        "b3": 77,               # 3ème nombre de Betti
        "lambda1_predicted": 13 / 99,  # λ₁ × H* = 13
        "description": "K₇ GIFT (variété G₂-holonomie)",
    },
}

# =============================================================================
# CONSTANTES TOPOLOGIQUES GIFT
# =============================================================================

TOPOLOGY = {
    "dim_E8": 248,
    "dim_G2": 14,
    "dim_K7": 7,
    "b2": 21,
    "b3": 77,
    "H_star": 99,              # b₂ + b₃ + 1
    "p2": 2,                   # Contribution Pontryagin

    # Relations remarquables
    "H_star_factorization": "dim(G₂) × dim(K₇) + 1 = 14 × 7 + 1 = 99",
    "universal_constant": 13,  # λ₁ × H* = dim(G₂) - 1
}

# =============================================================================
# GRILLE DE ROBUSTESSE
# =============================================================================

ROBUSTNESS_GRID = {
    "N": [1000, 2000, 5000, 10000, 20000],
    "k": [15, 25, 40, 60],
    "laplacian": ["unnormalized", "random_walk", "symmetric"],
}

# Sous-grille rapide pour tests
ROBUSTNESS_GRID_QUICK = {
    "N": [1000, 5000],
    "k": [25],
    "laplacian": ["symmetric"],
}

# =============================================================================
# PARTITIONS BETTI POUR TEST D'INDÉPENDANCE
# =============================================================================

BETTI_PARTITIONS = [
    {"b2": 21, "b3": 77, "name": "K7_GIFT", "description": "Valeurs physiques K₇"},
    {"b2": 0, "b3": 98, "name": "extreme_b3", "description": "Tout dans b₃"},
    {"b2": 49, "b3": 49, "name": "symmetric", "description": "Partition symétrique"},
    {"b2": 98, "b3": 0, "name": "extreme_b2", "description": "Tout dans b₂"},
    {"b2": 14, "b3": 84, "name": "dim_G2_b2", "description": "b₂ = dim(G₂)"},
    {"b2": 7, "b3": 91, "name": "dim_K7_b2", "description": "b₂ = dim(K₇)"},
]

# Vérification: toutes les partitions donnent H* = 99
for p in BETTI_PARTITIONS:
    assert p["b2"] + p["b3"] + 1 == TOPOLOGY["H_star"], f"Invalid partition: {p}"

# =============================================================================
# CRITÈRES PASS/FAIL
# =============================================================================

TOLERANCE = {
    # Phase 1: Calibration
    "calibration_S3": 0.05,      # 5% max deviation sur S³
    "calibration_S7": 0.10,      # 10% max deviation sur S⁷ (plus dur)
    "calibration_S3_range": (2.85, 3.15),  # λ₁ ∈ [2.85, 3.15]
    "calibration_S7_range": (6.3, 7.7),    # λ₁ ∈ [6.3, 7.7]

    # Phase 2: Robustesse
    "convergence_plateau": 0.02,  # 2% variation max dans plateau
    "plateau_N_min": 5000,        # N minimum pour définir plateau
    "plateau_range_13": (12.5, 13.5),  # Plateau proche de 13
    "plateau_range_14": (13.5, 14.5),  # Plateau proche de 14

    # Phase 3: Indépendance Betti
    "betti_independence": 1e-8,   # spread max entre partitions (très strict)

    # Phase 4: Analyse biais
    "bias_significance": 0.05,    # 5% pour considérer biais significatif
}

# =============================================================================
# CRITÈRES BLOQUANTS
# =============================================================================

BLOCKING_CRITERIA = {
    "calibration": "S³ calibration doit donner λ₁ ∈ [2.5, 3.5]",
    "plateau": "Variation < 2% pour N ≥ 5000",
    "reproducibility": "pytest 100% pass",
}

# =============================================================================
# SEEDS POUR REPRODUCTIBILITÉ
# =============================================================================

SEEDS = {
    "calibration": 42,
    "robustness": 2024,
    "betti_test": 99,
    "bias_analysis": 13,
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_output_path(phase: str, filename: str) -> Path:
    """Retourne le chemin de sortie pour une phase donnée."""
    phase_dirs = {
        "calibration": CALIBRATION_DIR / "outputs",
        "robustness": ROBUSTNESS_DIR / "outputs",
        "analysis": ANALYSIS_DIR / "outputs",
    }
    return phase_dirs.get(phase, BASE_DIR) / filename


def check_pass_fail(value: float, criterion: str) -> dict:
    """Vérifie si une valeur passe un critère."""
    if criterion.endswith("_range"):
        low, high = TOLERANCE[criterion]
        passed = low <= value <= high
        return {
            "value": value,
            "criterion": criterion,
            "range": (low, high),
            "passed": passed,
            "status": "PASS" if passed else "FAIL",
        }
    else:
        threshold = TOLERANCE[criterion]
        passed = value <= threshold
        return {
            "value": value,
            "criterion": criterion,
            "threshold": threshold,
            "passed": passed,
            "status": "PASS" if passed else "FAIL",
        }


# =============================================================================
# VERSION INFO
# =============================================================================

VERSION = "1.0.0"
ROADMAP_VERSION = "1.0 - 2026-01-22"

if __name__ == "__main__":
    print(f"Spectral Validation Config v{VERSION}")
    print(f"Base directory: {BASE_DIR}")
    print(f"\nTopology constants:")
    for k, v in TOPOLOGY.items():
        print(f"  {k}: {v}")
    print(f"\nManifolds: {list(MANIFOLDS.keys())}")
    print(f"Robustness grid size: {len(ROBUSTNESS_GRID['N']) * len(ROBUSTNESS_GRID['k']) * len(ROBUSTNESS_GRID['laplacian'])} configurations")
