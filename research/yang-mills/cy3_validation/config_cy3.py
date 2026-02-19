"""
Configuration centralisée pour validation spectrale CY₃.

CONJECTURE UNIVERSELLE:
    λ₁ × H* = dim(Hol) - h

Pour CY₃ (holonomie SU(3)):
    dim(SU(3)) = 8
    h = 2 (spineurs parallèles: chiral + anti-chiral)
    Target = 6

Référence: ROADMAP_CY3_SIDEQUEST.md
"""

from dataclasses import dataclass
from typing import Dict, Callable
from pathlib import Path

# =============================================================================
# CHEMINS
# =============================================================================

BASE_DIR = Path(__file__).parent
SAMPLING_DIR = BASE_DIR / "sampling"
SPECTRAL_DIR = BASE_DIR / "spectral"
ANALYSIS_DIR = BASE_DIR / "analysis"
OUTPUTS_DIR = BASE_DIR / "outputs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# =============================================================================
# CONSTANTES UNIVERSELLES
# =============================================================================

UNIVERSAL_LAW = {
    "formula": "λ₁ × H* = dim(Hol) - h",
    "description": "Spectral gap scales inversely with effective cohomology",
}

# Holonomies spéciales et leurs prédictions
HOLONOMY_PREDICTIONS = {
    "G2": {
        "dim_Hol": 14,
        "h_spinors": 1,
        "target": 13,
        "status": "VALIDATED (N=50000, exact)",
    },
    "SU3": {
        "dim_Hol": 8,
        "h_spinors": 2,
        "target": 6,
        "status": "TESTING",
    },
    "SU2": {  # K3
        "dim_Hol": 3,
        "h_spinors": 2,
        "target": 1,
        "status": "FUTURE",
    },
    "Spin7": {
        "dim_Hol": 21,
        "h_spinors": 1,
        "target": 20,
        "status": "FUTURE",
    },
}

# =============================================================================
# DÉFINITIONS H* POUR CY₃
# =============================================================================

def Hstar_A(h11: int, h21: int) -> int:
    """
    Option A: h¹¹ + h²¹ + h (nombres de Hodge + spineurs)

    Analogie avec G₂ où H* = b₂ + b₃ + 1.
    Pour CY₃: h = 2 spineurs parallèles.
    """
    return h11 + h21 + 2


def Hstar_B(h11: int, h21: int) -> int:
    """
    Option B: b₂ + b₃ + h = h¹¹ + 2h²¹ + 4

    Utilise les nombres de Betti:
    - b₂ = h¹¹
    - b₃ = 2(h²¹ + 1) = 2h²¹ + 2
    """
    return h11 + 2 * h21 + 4


def Hstar_C(h11: int, h21: int) -> int:
    """
    Option C: |χ|/2 + h (basé sur caractéristique d'Euler)

    Pour CY₃: χ = 2(h¹¹ - h²¹)
    Donc |χ|/2 = |h¹¹ - h²¹|
    """
    return abs(h11 - h21) + 2


HSTAR_DEFINITIONS = {
    "A": {"fn": Hstar_A, "formula": "h¹¹ + h²¹ + 2", "description": "Hodge + spineurs"},
    "B": {"fn": Hstar_B, "formula": "h¹¹ + 2h²¹ + 4", "description": "Betti + spineurs"},
    "C": {"fn": Hstar_C, "formula": "|h¹¹ - h²¹| + 2", "description": "Euler + spineurs"},
}

# =============================================================================
# VARIÉTÉS CY₃ DE TEST
# =============================================================================

CY3_MANIFOLDS = {
    # =========================================================================
    # TIER 1: Métriques explicites (calibration)
    # =========================================================================
    "T6_flat": {
        "h11": 9,
        "h21": 9,
        "chi": 0,
        "metric_type": "flat",
        "holonomy": "trivial",  # PAS SU(3)! Juste calibration
        "tier": 1,
        "description": "6-torus (limite plate, calibration pipeline)",
        "lambda1_exact": 1.0,  # Pour R=1 sur chaque cercle
        "notes": "Holonomie triviale, PAS un test de la conjecture",
    },

    # =========================================================================
    # TIER 2: Orbifolds (vrai holonomie SU(3))
    # =========================================================================
    "T6_Z3": {
        "h11": 9,
        "h21": 0,
        "chi": 18,
        "metric_type": "orbifold",
        "holonomy": "SU(3)",
        "tier": 2,
        "description": "T⁶/ℤ₃ orbifold - premier vrai CY₃",
        "notes": "27 points fixes résolus en P² exceptionnels",
    },
    "T6_Z3xZ3": {
        "h11": 3,
        "h21": 0,
        "chi": 6,
        "metric_type": "orbifold",
        "holonomy": "SU(3)",
        "tier": 2,
        "description": "T⁶/(ℤ₃×ℤ₃) orbifold",
    },

    # =========================================================================
    # TIER 3: CICY (métriques numériques)
    # =========================================================================
    "Quintic": {
        "h11": 1,
        "h21": 101,
        "chi": -200,
        "metric_type": "numerical",
        "holonomy": "SU(3)",
        "tier": 3,
        "description": "Quintic hypersurface P⁴[5]",
        "embedding": "CP4",
        "degree": 5,
        "notes": "Le plus célèbre CY₃, métrique via Donaldson",
    },
    "Bicubic": {
        "h11": 19,
        "h21": 19,
        "chi": 0,
        "metric_type": "numerical",
        "holonomy": "SU(3)",
        "tier": 3,
        "description": "Bicubic (3,3) in P²×P²",
        "notes": "χ=0 intéressant pour comparaison",
    },
    "Mirror_Quintic": {
        "h11": 101,
        "h21": 1,
        "chi": 200,
        "metric_type": "numerical",
        "holonomy": "SU(3)",
        "tier": 3,
        "description": "Mirror du Quintic",
        "notes": "Test symétrie miroir",
    },

    # =========================================================================
    # TIER 4: Sweep paramétrique
    # =========================================================================
    "CY_h11_1_h21_1": {"h11": 1, "h21": 1, "chi": 0, "tier": 4, "metric_type": "synthetic"},
    "CY_h11_5_h21_5": {"h11": 5, "h21": 5, "chi": 0, "tier": 4, "metric_type": "synthetic"},
    "CY_h11_20_h21_20": {"h11": 20, "h21": 20, "chi": 0, "tier": 4, "metric_type": "synthetic"},
}

# Pré-calculer H* pour chaque variété
for name, data in CY3_MANIFOLDS.items():
    h11, h21 = data["h11"], data["h21"]
    data["Hstar_A"] = Hstar_A(h11, h21)
    data["Hstar_B"] = Hstar_B(h11, h21)
    data["Hstar_C"] = Hstar_C(h11, h21)

# =============================================================================
# PARAMÈTRES NUMÉRIQUES
# =============================================================================

@dataclass
class NumericalParams:
    """Paramètres numériques par défaut."""
    N_calibration: int = 5000
    N_validation: int = 10000
    N_high_res: int = 50000
    k_default: int = 25
    k_high_res: int = 165  # Optimal pour N=50000 (de la validation G₂)
    laplacian_type: str = "symmetric"
    sigma_method: str = "auto"


PARAMS = NumericalParams()

# Grille de robustesse
ROBUSTNESS_GRID = {
    "N": [1000, 2000, 5000, 10000],
    "k": [15, 25, 40, 60],
    "laplacian": ["symmetric"],
}

# Seeds pour reproductibilité
SEEDS = {
    "calibration": 42,
    "validation": 2024,
    "sweep": [42, 123, 456],
}

# =============================================================================
# CRITÈRES PASS/FAIL
# =============================================================================

TOLERANCE = {
    # Phase 1: Calibration T⁶
    "T6_lambda1_range": (0.8, 1.2),  # ±20% de λ₁=1
    "T6_convergence": 0.05,  # 5% variation max

    # Phase 2-3: Validation CY₃
    "target_6_tolerance": 0.10,  # 10% → [5.4, 6.6]
    "target_6_range": (5.4, 6.6),

    # Cohérence entre définitions H*
    "Hstar_consistency": 0.15,  # 15% spread max acceptable

    # Cross-validation avec G₂
    "G2_CY3_compatibility": 0.20,  # 20% de différence max
}

# =============================================================================
# CRITÈRES BLOQUANTS
# =============================================================================

BLOCKING_CRITERIA = {
    "Phase1": "T⁶ calibration: λ₁ ∈ [0.8, 1.2] pour R=1",
    "Phase2": "Au moins 1 CY₃ avec holonomie SU(3) donne λ₁×H* ∈ [5, 7]",
    "Phase4": "Une définition H* est consistante (spread < 15%)",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_target_lambda1(h11: int, h21: int, Hstar_def: str = "A") -> float:
    """Retourne λ₁ cible pour la conjecture λ₁ × H* = 6."""
    Hstar_fn = HSTAR_DEFINITIONS[Hstar_def]["fn"]
    Hstar = Hstar_fn(h11, h21)
    return 6.0 / Hstar


def check_pass_fail(product: float, target: float = 6.0) -> dict:
    """Vérifie si λ₁ × H* est proche de la cible."""
    deviation_pct = abs(product - target) / target * 100
    passed = TOLERANCE["target_6_range"][0] <= product <= TOLERANCE["target_6_range"][1]

    return {
        "product": product,
        "target": target,
        "deviation_pct": deviation_pct,
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
    }


def summary_table() -> str:
    """Génère un tableau récapitulatif des variétés."""
    lines = [
        "| Variété | h¹¹ | h²¹ | χ | H*_A | H*_B | H*_C | λ₁_target (A) |",
        "|---------|-----|-----|---|------|------|------|---------------|",
    ]
    for name, data in CY3_MANIFOLDS.items():
        if data.get("tier", 99) <= 3:
            target = get_target_lambda1(data["h11"], data["h21"], "A")
            lines.append(
                f"| {name} | {data['h11']} | {data['h21']} | {data['chi']} | "
                f"{data['Hstar_A']} | {data['Hstar_B']} | {data['Hstar_C']} | {target:.4f} |"
            )
    return "\n".join(lines)


# =============================================================================
# VERSION INFO
# =============================================================================

VERSION = "1.0.0"
ROADMAP_VERSION = "CY3 Sidequest v1.0 - 2026-01-22"

if __name__ == "__main__":
    print(f"CY₃ Spectral Validation Config v{VERSION}")
    print(f"Base directory: {BASE_DIR}")
    print(f"\nTarget: λ₁ × H* = {HOLONOMY_PREDICTIONS['SU3']['target']} (SU(3) holonomy)")
    print(f"\nManifolds ({len(CY3_MANIFOLDS)} total):")
    for name, data in CY3_MANIFOLDS.items():
        if data.get("tier", 99) <= 3:
            print(f"  [{data.get('tier', '?')}] {name}: h¹¹={data['h11']}, h²¹={data['h21']}")
    print(f"\nH* definitions summary:")
    print(summary_table())
