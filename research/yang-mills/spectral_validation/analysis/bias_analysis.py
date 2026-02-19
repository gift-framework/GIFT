"""
Analyse du Biais "-1"

OBJECTIF:
Déterminer si le -1 observé (13 = dim(G₂) - 1 au lieu de 14 = dim(G₂))
est un artifact du pipeline ou une propriété genuine de G₂.

HYPOTHÈSES:
A) ARTIFACT: Le graph Laplacian a un biais systématique de -1
   → Si on corrige, K₇ devrait donner 14

B) GENUINE: Le -1 est une propriété de la géométrie G₂
   → 13 est la vraie constante universelle

MÉTHODE:
1. Comparer les ratios observés sur S³, S⁷ vs théoriques
2. Si biais systématique détecté, l'appliquer à K₇
3. Déterminer si K₇ corrigé est plus proche de 13 ou 14

Reference: ROADMAP_SPECTRAL_VALIDATION.md Phase 4
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
import sys
from datetime import datetime

# Ajouter le parent pour importer config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TOPOLOGY, get_output_path


# =============================================================================
# ANALYSE DU BIAIS
# =============================================================================

def load_calibration_results() -> Dict:
    """Charge les résultats de calibration S³ et S⁷."""
    calibration_dir = Path(__file__).parent.parent / "calibration" / "outputs"

    results = {}

    # S³
    s3_path = calibration_dir / "S3_calibration_results.json"
    if s3_path.exists():
        with open(s3_path) as f:
            results["S3"] = json.load(f)
    else:
        results["S3"] = None

    # S⁷
    s7_path = calibration_dir / "S7_calibration_results.json"
    if s7_path.exists():
        with open(s7_path) as f:
            results["S7"] = json.load(f)
    else:
        results["S7"] = None

    # Comparaison
    comp_path = calibration_dir / "S3_S7_comparison.json"
    if comp_path.exists():
        with open(comp_path) as f:
            results["comparison"] = json.load(f)
    else:
        results["comparison"] = None

    return results


def load_robustness_results() -> Optional[Dict]:
    """Charge les résultats de robustesse K₇."""
    robustness_path = (Path(__file__).parent.parent / "robustness" /
                       "outputs" / "K7_robustness_matrix.json")

    if robustness_path.exists():
        with open(robustness_path) as f:
            return json.load(f)
    return None


def analyze_sphere_bias(calibration: Dict) -> Dict:
    """
    Analyse le biais sur les sphères.

    Le graph Laplacian normalisé ne donne pas λ₁(LB) directement.
    On compare les RATIOS pour détecter un biais systématique.
    """
    s3 = calibration.get("S3")
    s7 = calibration.get("S7")
    comparison = calibration.get("comparison")

    analysis = {
        "expected_ratio_S7_S3": 7.0 / 3.0,  # λ₁(S⁷)/λ₁(S³) théorique
    }

    if comparison and comparison.get("mean_ratio"):
        analysis["measured_ratio_S7_S3"] = comparison["mean_ratio"]
        analysis["ratio_deviation_pct"] = comparison.get("deviation_from_expected_pct", 0)

        # Le ratio mesuré est-il systématiquement biaisé ?
        expected = 7.0 / 3.0
        measured = comparison["mean_ratio"]

        if measured > expected * 1.5:
            analysis["ratio_bias"] = "HIGH"
            analysis["interpretation"] = (
                "Le graph Laplacian surestime le ratio S⁷/S³. "
                "Biais positif détecté."
            )
        elif measured < expected * 0.5:
            analysis["ratio_bias"] = "LOW"
            analysis["interpretation"] = (
                "Le graph Laplacian sous-estime le ratio S⁷/S³. "
                "Biais négatif détecté."
            )
        else:
            analysis["ratio_bias"] = "MODERATE"
            analysis["interpretation"] = (
                "Le ratio S⁷/S³ est biaisé mais pas de façon extrême. "
                "Le biais n'est pas simplement ±1."
            )

    # Valeurs brutes
    if s3 and "analysis" in s3:
        sym_stats = s3["analysis"]["by_laplacian_type"].get("symmetric", {})
        analysis["S3_mean_lambda1"] = sym_stats.get("mean_lambda1", None)

    if s7 and "analysis" in s7:
        sym_stats = s7["analysis"]["by_laplacian_type"].get("symmetric", {})
        analysis["S7_mean_lambda1"] = sym_stats.get("mean_lambda1", None)

    return analysis


def analyze_k7_bias(robustness: Dict, sphere_bias: Dict) -> Dict:
    """
    Analyse le biais potentiel sur K₇.

    Question clé: Le 13 observé est-il un 14 biaisé ou un vrai 13 ?
    """
    analysis = {}

    if not robustness:
        analysis["error"] = "No robustness data"
        return analysis

    plateau = robustness.get("analysis", {}).get("plateau_analysis", {})

    if "mean" not in plateau:
        analysis["error"] = "No plateau data"
        return analysis

    k7_product = plateau["mean"]
    analysis["k7_raw_product"] = k7_product
    analysis["k7_deviation_from_13"] = plateau.get("deviation_from_13_pct", 0)
    analysis["k7_deviation_from_14"] = plateau.get("deviation_from_14_pct", 0)

    # Hypothèse A: Si on a un biais de -1, K₇ "vrai" serait 14
    analysis["if_bias_minus_1"] = {
        "corrected_product": k7_product + 1,
        "closer_to": "14" if abs(k7_product + 1 - 14) < abs(k7_product + 1 - 13) else "13",
    }

    # Hypothèse B: Le 13 est genuine
    analysis["if_genuine_13"] = {
        "interpretation": "13 = dim(G₂) - 1 est la constante universelle",
        "deviation_pct": abs(k7_product - 13) / 13 * 100,
    }

    # Décision
    # Le graph Laplacian normalisé a spectre [0,2], pas [0,∞)
    # Donc on ne peut pas simplement ajouter +1
    # Le biais n'est pas linéaire en général

    # Critère: K₇ est-il naturellement plus proche de 13 ou 14 ?
    if analysis["k7_deviation_from_13"] < analysis["k7_deviation_from_14"]:
        analysis["verdict"] = "GENUINE_13"
        analysis["confidence"] = "HIGH" if analysis["k7_deviation_from_13"] < 5 else "MODERATE"
        analysis["conclusion"] = (
            f"K₇ donne λ₁×H* = {k7_product:.2f}, plus proche de 13 que de 14. "
            f"Le -1 semble être une propriété genuine de G₂, pas un artifact."
        )
    else:
        analysis["verdict"] = "BIAS_ARTIFACT"
        analysis["confidence"] = "MODERATE"
        analysis["conclusion"] = (
            f"K₇ donne λ₁×H* = {k7_product:.2f}, plus proche de 14 que de 13. "
            f"Le graph Laplacian pourrait avoir un biais."
        )

    # Argument dimensionnel
    analysis["dimensional_argument"] = {
        "dim_G2": TOPOLOGY["dim_G2"],
        "dim_G2_minus_1": TOPOLOGY["dim_G2"] - 1,
        "interpretation": (
            "13 = dim(G₂) - 1 pourrait représenter les degrés de liberté "
            "de G₂ moins la direction 'nulle' (ou le mode zéro du Laplacien)."
        ),
    }

    return analysis


def run_bias_analysis() -> Dict:
    """Exécute l'analyse complète du biais."""
    # Charger données
    calibration = load_calibration_results()
    robustness = load_robustness_results()

    # Analyser biais sphères
    sphere_bias = analyze_sphere_bias(calibration)

    # Analyser biais K₇
    k7_bias = analyze_k7_bias(robustness, sphere_bias)

    # Synthèse
    synthesis = {
        "question": "Le -1 dans λ₁×H* = 13 = dim(G₂) - 1 est-il genuine ou artifact ?",
    }

    if k7_bias.get("verdict") == "GENUINE_13":
        synthesis["answer"] = "GENUINE"
        synthesis["conclusion"] = (
            "Le -1 semble être une propriété genuine de la géométrie G₂. "
            "La constante universelle est 13 = dim(G₂) - 1, pas 14."
        )
        synthesis["physical_interpretation"] = (
            "Possible interprétation: le gap spectral est réduit de 1 par rapport "
            "à dim(G₂) à cause du mode zéro du Laplacien qui 'consomme' un degré de liberté."
        )
    else:
        synthesis["answer"] = "ARTIFACT"
        synthesis["conclusion"] = (
            "Le -1 pourrait être un artifact du graph Laplacian. "
            "Investigation supplémentaire requise."
        )

    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "analysis": "Bias -1 Investigation",
        },
        "sphere_calibration": sphere_bias,
        "k7_analysis": k7_bias,
        "synthesis": synthesis,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  BIAS ANALYSIS - Spectral Validation Phase 4")
    print("  Question: Is λ₁×H* = 13 genuine or artifact?")
    print("=" * 70)
    print()

    # Run analysis
    output = run_bias_analysis()

    # Save
    output_path = get_output_path("analysis", "bias_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary
    print("SPHERE CALIBRATION ANALYSIS")
    print("-" * 40)
    sphere = output["sphere_calibration"]
    print(f"  Expected ratio S⁷/S³: {sphere.get('expected_ratio_S7_S3', 'N/A'):.4f}")
    if sphere.get("measured_ratio_S7_S3"):
        print(f"  Measured ratio S⁷/S³: {sphere['measured_ratio_S7_S3']:.4f}")
        print(f"  Deviation: {sphere.get('ratio_deviation_pct', 0):.1f}%")
        print(f"  Interpretation: {sphere.get('interpretation', 'N/A')}")

    print()
    print("K₇ ANALYSIS")
    print("-" * 40)
    k7 = output["k7_analysis"]
    if "error" not in k7:
        print(f"  Raw λ₁×H*: {k7.get('k7_raw_product', 'N/A'):.4f}")
        print(f"  Deviation from 13: {k7.get('k7_deviation_from_13', 'N/A'):.2f}%")
        print(f"  Deviation from 14: {k7.get('k7_deviation_from_14', 'N/A'):.2f}%")
        print(f"  Verdict: {k7.get('verdict', 'N/A')}")
        print(f"  Confidence: {k7.get('confidence', 'N/A')}")
    else:
        print(f"  Error: {k7['error']}")

    print()
    print("=" * 70)
    print("SYNTHESIS")
    print("=" * 70)
    synth = output["synthesis"]
    print(f"\nQuestion: {synth['question']}")
    print(f"Answer: {synth.get('answer', 'N/A')}")
    print(f"\nConclusion: {synth.get('conclusion', 'N/A')}")

    if "physical_interpretation" in synth:
        print(f"\nPhysical interpretation: {synth['physical_interpretation']}")

    print(f"\nSaved: {output_path}")
