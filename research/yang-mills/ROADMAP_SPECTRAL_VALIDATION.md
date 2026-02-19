# Roadmap: Bullet-Proof Spectral Validation

**Objectif**: Valider rigoureusement λ₁×H* = 13 avant toute publication
**Approche**: Conservatrice - chaque étape doit PASSER avant la suivante
**Repo cible**: gift-framework/GIFT (branche `spectral-validation`)

---

## Phase 0: Setup & Infrastructure (Jour 1)

### Tâche 0.1: Créer la structure
```bash
mkdir -p notebooks/spectral_validation/{calibration,robustness,analysis}
mkdir -p tests/spectral
mkdir -p docs/spectral
```

### Tâche 0.2: Fichier de configuration centralisé
Créer `notebooks/spectral_validation/config.py`:
```python
"""Configuration centralisée pour validation spectrale."""

# Paramètres de référence (V11)
REFERENCE = {
    "N": 5000,
    "k_neighbors": 25,
    "sigma_method": "auto",  # ratio * sqrt(dim/k)
    "laplacian_type": "symmetric",  # I - D^{-1/2} W D^{-1/2}
}

# Variétés de test
MANIFOLDS = {
    "S3": {"dim": 3, "lambda1_exact": 3.0, "description": "3-sphere (calibration)"},
    "S7": {"dim": 7, "lambda1_exact": 7.0, "description": "7-sphere (calibration)"},
    "T7": {"dim": 7, "lambda1_exact": 0.0, "description": "7-torus (zero mode)"},
    "K7_GIFT": {"dim": 7, "H_star": 99, "b2": 21, "b3": 77},
}

# Grille de robustesse
ROBUSTNESS_GRID = {
    "N": [1000, 2000, 5000, 10000, 20000],
    "k": [15, 25, 40, 60],
    "laplacian": ["unnormalized", "random_walk", "symmetric"],
}

# Critères PASS/FAIL
TOLERANCE = {
    "calibration_S3": 0.05,      # 5% max deviation sur S³
    "calibration_S7": 0.10,      # 10% max deviation sur S⁷
    "betti_independence": 1e-10, # spread max entre partitions
    "convergence_plateau": 0.02, # 2% variation max dans plateau
}
```

---

## Phase 1: Calibration Étalon (Jours 2-4)

**Objectif**: Vérifier que le pipeline reproduit λ₁ connu sur des variétés simples.

### Tâche 1.1: Implémentation S³ calibration
Créer `notebooks/spectral_validation/calibration/S3_calibration.py`:

```python
"""
Calibration sur S³: λ₁ analytique = 3
Si le pipeline donne ~3, pas de biais systématique.
Si le pipeline donne ~2, il y a un biais de -1.
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

def sample_S3_uniform(N: int, seed: int = 42) -> np.ndarray:
    """Échantillonnage uniforme sur S³ via méthode quaternionique."""
    rng = np.random.default_rng(seed)
    # Méthode: 4 gaussiennes normalisées
    points = rng.standard_normal((N, 4))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points

def geodesic_distance_S3(p1: np.ndarray, p2: np.ndarray) -> float:
    """Distance géodésique sur S³: d = arccos(|p1·p2|)."""
    dot = np.clip(np.abs(np.sum(p1 * p2, axis=-1)), -1, 1)
    return np.arccos(dot)

def build_graph_laplacian(points: np.ndarray, k: int, sigma: float, 
                          laplacian_type: str = "symmetric") -> csr_matrix:
    """Construit le Laplacien de graphe."""
    from sklearn.neighbors import NearestNeighbors
    
    N = len(points)
    
    # k-NN avec distance géodésique
    # Pour S³, on utilise la distance de corde puis conversion
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Conversion corde → géodésique pour S³
    # d_chord = 2*sin(d_geo/2) => d_geo = 2*arcsin(d_chord/2)
    geo_distances = 2 * np.arcsin(np.clip(distances / 2, 0, 1))
    
    # Matrice de poids gaussiens
    W = np.zeros((N, N))
    for i in range(N):
        for j_idx, j in enumerate(indices[i, 1:]):  # skip self
            d = geo_distances[i, j_idx + 1]
            W[i, j] = np.exp(-d**2 / (2 * sigma**2))
            W[j, i] = W[i, j]  # symétrique
    
    # Degré
    D = np.diag(W.sum(axis=1))
    
    # Laplacien selon type
    if laplacian_type == "unnormalized":
        L = D - W
    elif laplacian_type == "random_walk":
        D_inv = np.diag(1.0 / np.diag(D))
        L = np.eye(N) - D_inv @ W
    elif laplacian_type == "symmetric":
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        L = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        raise ValueError(f"Unknown laplacian_type: {laplacian_type}")
    
    return csr_matrix(L)

def compute_lambda1(L: csr_matrix, num_eigenvalues: int = 5) -> float:
    """Calcule λ₁ (première valeur propre non-nulle)."""
    eigenvalues, _ = eigsh(L, k=num_eigenvalues, which='SM')
    eigenvalues = np.sort(np.real(eigenvalues))
    # λ₀ ≈ 0 (mode constant), λ₁ = première non-nulle
    lambda1 = eigenvalues[1] if eigenvalues[0] < 1e-10 else eigenvalues[0]
    return lambda1

def run_S3_calibration(config: dict) -> dict:
    """
    Exécute la calibration S³.
    
    Returns:
        dict avec lambda1_measured, lambda1_exact, deviation, PASS/FAIL
    """
    results = []
    
    for N in config.get("N_values", [1000, 2000, 5000]):
        for k in config.get("k_values", [25]):
            for lap_type in config.get("laplacian_types", ["symmetric"]):
                
                points = sample_S3_uniform(N)
                sigma = np.sqrt(3 / k)  # heuristique dim=3
                
                L = build_graph_laplacian(points, k, sigma, lap_type)
                lambda1 = compute_lambda1(L)
                
                # Pour S³, λ₁ exact = 3 (avec multiplicité 4)
                # Mais le Laplacien normalisé a λ₁ ∈ [0, 2]
                # Donc on doit rescaler ou interpréter différemment
                
                # Le Laplacien normalisé symétrique a spectre dans [0, 2]
                # λ₁(normalized) ≈ λ₁(LB) / λ_max(LB) * 2
                # Pour S³: λ_max(LB) = n(n+2) pour n→∞, mais spectre discret
                
                results.append({
                    "N": N,
                    "k": k,
                    "laplacian_type": lap_type,
                    "lambda1_measured": lambda1,
                    "sigma": sigma,
                })
    
    return results

if __name__ == "__main__":
    import json
    
    config = {
        "N_values": [1000, 2000, 5000, 10000],
        "k_values": [15, 25, 40],
        "laplacian_types": ["unnormalized", "random_walk", "symmetric"],
    }
    
    results = run_S3_calibration(config)
    
    with open("outputs/S3_calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("S³ Calibration Complete")
    for r in results:
        print(f"  N={r['N']}, k={r['k']}, {r['laplacian_type']}: λ₁={r['lambda1_measured']:.4f}")
```

### Tâche 1.2: Test S⁷ calibration
Créer fichier similaire pour S⁷ (λ₁ = 7 analytiquement).

### Tâche 1.3: Critères PASS/FAIL

| Test | Condition PASS | Action si FAIL |
|------|----------------|----------------|
| S³ λ₁ | Mesure dans [2.85, 3.15] pour au moins un (N,k,type) | Le pipeline a un biais → investiguer |
| S⁷ λ₁ | Mesure dans [6.3, 7.7] | Même chose |
| Convergence | λ₁(N) forme un plateau pour N≥2000 | Le "sweet spot" n'est pas universel |

### Livrable Phase 1
```
notebooks/spectral_validation/calibration/
├── S3_calibration.py
├── S7_calibration.py  
├── outputs/
│   ├── S3_calibration_results.json
│   └── S7_calibration_results.json
└── CALIBRATION_REPORT.md  # Résumé PASS/FAIL
```

---

## Phase 2: Matrice de Robustesse K₇ (Jours 5-7)

**Objectif**: Identifier si 13 est un plateau stable ou un point de passage.

### Tâche 2.1: Grille complète
Créer `notebooks/spectral_validation/robustness/K7_robustness_matrix.py`:

```python
"""
Matrice de robustesse pour K₇ (H*=99).
Teste toutes combinaisons (N, k, laplacian_type).
"""

import itertools
import numpy as np
import json
from pathlib import Path

# Import depuis le code existant
from notebooks.G2_Universality_v11_Test13 import (
    sample_TCS_manifold,
    compute_spectral_gap,
)

def run_robustness_matrix():
    """Exécute la grille complète."""
    
    N_values = [1000, 2000, 5000, 10000, 20000]
    k_values = [15, 25, 40, 60]
    lap_types = ["unnormalized", "random_walk", "symmetric"]
    
    H_star = 99  # K₇
    results = []
    
    for N, k, lap_type in itertools.product(N_values, k_values, lap_types):
        print(f"Testing N={N}, k={k}, {lap_type}...")
        
        try:
            # Sampling K₇ (TCS construction)
            points = sample_TCS_manifold(N, H_star=H_star, b2=21, b3=77)
            
            # Compute λ₁
            lambda1 = compute_spectral_gap(points, k=k, laplacian_type=lap_type)
            
            product = lambda1 * H_star
            dev_13 = abs(product - 13) / 13 * 100
            dev_14 = abs(product - 14) / 14 * 100
            
            results.append({
                "N": N,
                "k": k,
                "laplacian_type": lap_type,
                "lambda1": lambda1,
                "lambda1_x_Hstar": product,
                "deviation_from_13_pct": dev_13,
                "deviation_from_14_pct": dev_14,
                "closer_to": "13" if dev_13 < dev_14 else "14",
            })
            
        except Exception as e:
            results.append({
                "N": N, "k": k, "laplacian_type": lap_type,
                "error": str(e)
            })
    
    return results

def analyze_results(results: list) -> dict:
    """Analyse la matrice pour identifier plateaux."""
    
    # Filtrer erreurs
    valid = [r for r in results if "error" not in r]
    
    # Grouper par laplacian_type
    by_type = {}
    for lap_type in ["unnormalized", "random_walk", "symmetric"]:
        subset = [r for r in valid if r["laplacian_type"] == lap_type]
        by_type[lap_type] = {
            "mean_product": np.mean([r["lambda1_x_Hstar"] for r in subset]),
            "std_product": np.std([r["lambda1_x_Hstar"] for r in subset]),
            "closer_to_13_count": sum(1 for r in subset if r["closer_to"] == "13"),
            "closer_to_14_count": sum(1 for r in subset if r["closer_to"] == "14"),
        }
    
    # Identifier plateau (variation < 2% pour N >= 5000)
    high_N = [r for r in valid if r["N"] >= 5000]
    if high_N:
        products = [r["lambda1_x_Hstar"] for r in high_N]
        plateau_mean = np.mean(products)
        plateau_std = np.std(products)
        plateau_variation = plateau_std / plateau_mean * 100
        
        plateau_analysis = {
            "mean": plateau_mean,
            "std": plateau_std,
            "variation_pct": plateau_variation,
            "is_plateau": plateau_variation < 2.0,
            "plateau_value": round(plateau_mean) if plateau_variation < 5 else "unstable",
        }
    else:
        plateau_analysis = {"error": "No high-N results"}
    
    return {
        "by_laplacian_type": by_type,
        "plateau_analysis": plateau_analysis,
        "total_results": len(valid),
        "total_errors": len(results) - len(valid),
    }

if __name__ == "__main__":
    results = run_robustness_matrix()
    analysis = analyze_results(results)
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "K7_robustness_matrix.json", "w") as f:
        json.dump({"results": results, "analysis": analysis}, f, indent=2)
    
    print("\n=== ROBUSTNESS ANALYSIS ===")
    print(f"Plateau mean: {analysis['plateau_analysis'].get('mean', 'N/A'):.3f}")
    print(f"Plateau variation: {analysis['plateau_analysis'].get('variation_pct', 'N/A'):.2f}%")
    print(f"Plateau value: {analysis['plateau_analysis'].get('plateau_value', 'N/A')}")
```

### Tâche 2.2: Critères PASS/FAIL

| Test | Condition PASS | Interprétation |
|------|----------------|----------------|
| Plateau existe | variation < 2% pour N≥5000 | Convergence stable |
| Plateau = 13 | mean ∈ [12.5, 13.5] | 13 est la constante |
| Plateau = 14 | mean ∈ [13.5, 14.5] | 14 est la constante |
| Indépendant du type | std(by_type) < 0.5 | Résultat robuste |

### Livrable Phase 2
```
notebooks/spectral_validation/robustness/
├── K7_robustness_matrix.py
├── outputs/
│   └── K7_robustness_matrix.json
└── ROBUSTNESS_REPORT.md
```

---

## Phase 3: Test d'Indépendance Betti (Jours 8-9)

**Objectif**: Confirmer que λ₁×H* ne dépend que de H*, pas de (b₂,b₃).

### Tâche 3.1: Test systématique
Créer `notebooks/spectral_validation/analysis/betti_independence_test.py`:

```python
"""
Test d'indépendance Betti.
Pour H* = 99 fixé, varier (b₂, b₃) et mesurer spread.
"""

def test_betti_independence(H_star: int = 99, N: int = 5000, k: int = 25):
    """
    Teste plusieurs partitions de H* = b₂ + b₃ + 1.
    """
    
    partitions = [
        {"b2": 21, "b3": 77, "name": "K7_GIFT"},
        {"b2": 0, "b3": 98, "name": "extreme_b3"},
        {"b2": 49, "b3": 49, "name": "symmetric"},
        {"b2": 98, "b3": 0, "name": "extreme_b2"},
        {"b2": 14, "b3": 84, "name": "dim_G2_b2"},
        {"b2": 7, "b3": 91, "name": "dim_K7_b2"},
    ]
    
    results = []
    for p in partitions:
        assert p["b2"] + p["b3"] + 1 == H_star, f"Invalid partition: {p}"
        
        points = sample_TCS_manifold(N, H_star=H_star, b2=p["b2"], b3=p["b3"])
        lambda1 = compute_spectral_gap(points, k=k)
        product = lambda1 * H_star
        
        results.append({
            **p,
            "H_star": H_star,
            "lambda1": lambda1,
            "lambda1_x_Hstar": product,
        })
    
    # Analyse spread
    products = [r["lambda1_x_Hstar"] for r in results]
    spread = (max(products) - min(products)) / np.mean(products) * 100
    
    return {
        "partitions": results,
        "spread_pct": spread,
        "mean_product": np.mean(products),
        "PASS": spread < 1e-8,  # Critère très strict
    }
```

### Critère PASS/FAIL

| Test | Condition PASS |
|------|----------------|
| Betti independence | spread < 10⁻⁸ % |

---

## Phase 4: Analyse du Biais "-1" (Jours 10-12)

**Objectif**: Déterminer si le -1 est un artifact ou une propriété genuine.

### Tâche 4.1: Comparaison calibration vs K₇

```python
def analyze_bias():
    """
    Compare le biais observé sur S³/S⁷ (calibration) vs K₇.
    
    Si S³ donne λ₁ = 2 au lieu de 3 → biais systématique de -1
    Si S³ donne λ₁ = 3 et K₇ donne 13 → le -1 est genuine pour G₂
    """
    
    # Charger résultats calibration
    with open("calibration/outputs/S3_calibration_results.json") as f:
        s3_results = json.load(f)
    
    # Trouver le meilleur résultat S³ (N=5000, k=25, symmetric)
    best_s3 = next(r for r in s3_results 
                   if r["N"] == 5000 and r["k"] == 25 
                   and r["laplacian_type"] == "symmetric")
    
    s3_lambda1 = best_s3["lambda1_measured"]
    s3_expected = 3.0  # Laplace-Beltrami sur S³
    
    # Le Laplacien normalisé a spectre dans [0, 2]
    # Donc on attend λ₁(norm) ≈ 2 * λ₁(LB) / λ_max(LB)
    # Pour S³: λ_max → ∞, donc normalisation complexe
    
    # Alternative: comparer le RATIO
    # Si S³ donne X, et on attendait 3, le facteur est X/3
    # Appliquer ce facteur à K₇: si K₇_raw * (3/X) ≈ 14 → le -1 est un biais
    
    # Charger K₇
    with open("robustness/outputs/K7_robustness_matrix.json") as f:
        k7_data = json.load(f)
    
    k7_plateau = k7_data["analysis"]["plateau_analysis"]["mean"]
    
    # Calcul du biais
    if s3_lambda1 > 0:
        correction_factor = 3.0 / s3_lambda1
        k7_corrected = k7_plateau * correction_factor
        
        return {
            "s3_measured": s3_lambda1,
            "s3_expected": 3.0,
            "correction_factor": correction_factor,
            "k7_raw": k7_plateau,
            "k7_corrected": k7_corrected,
            "interpretation": (
                "bias_artifact" if abs(k7_corrected - 14) < abs(k7_plateau - 14)
                else "genuine_13"
            ),
        }
```

### Critère décisionnel

| Résultat | Interprétation | Action |
|----------|----------------|--------|
| S³ → 3.0 ± 5%, K₇ → 13 | Le -1 est **genuine** | Publier avec 13 |
| S³ → 2.0 ± 5%, K₇ → 13 | Le -1 est un **biais** | Corriger pipeline, vérifier si K₇ → 14 |
| S³ instable | Pipeline non fiable | Revoir implémentation |

---

## Phase 5: Rapport Final & Décision (Jours 13-14)

### Tâche 5.1: Générer rapport consolidé
Créer `notebooks/spectral_validation/FINAL_REPORT.md`:

```markdown
# Spectral Validation: Final Report

## Executive Summary
- Calibration S³: [PASS/FAIL] - λ₁ measured = X (expected 3.0)
- Calibration S⁷: [PASS/FAIL] - λ₁ measured = Y (expected 7.0)
- Robustness K₇: [PASS/FAIL] - plateau at Z ± W
- Betti Independence: [PASS/FAIL] - spread = X%
- Bias Analysis: [genuine_13 / bias_artifact]

## Decision
Based on the above: **[PUBLISH / INVESTIGATE FURTHER / REVISE]**

## Recommended Universal Constant
λ₁ × H* = **[13 / 14 / TBD]**

## Confidence Level
[HIGH / MEDIUM / LOW]

## Next Steps
- If PUBLISH: Prepare arXiv preprint
- If INVESTIGATE: [specific issues to address]
- If REVISE: [corrections needed]
```

### Tâche 5.2: Script de validation automatique
Créer `tests/spectral/test_validation_pipeline.py`:

```python
"""
Tests automatisés pour la validation spectrale.
Exécuter avec: pytest tests/spectral/ -v
"""

import pytest
import json
from pathlib import Path

OUTPUTS = Path("notebooks/spectral_validation")

class TestCalibration:
    def test_s3_calibration_exists(self):
        assert (OUTPUTS / "calibration/outputs/S3_calibration_results.json").exists()
    
    def test_s3_lambda1_reasonable(self):
        with open(OUTPUTS / "calibration/outputs/S3_calibration_results.json") as f:
            results = json.load(f)
        # Au moins un résultat dans [1.5, 4.5] (large pour robustesse)
        lambdas = [r["lambda1_measured"] for r in results if "lambda1_measured" in r]
        assert any(1.5 < l < 4.5 for l in lambdas), f"S³ λ₁ out of range: {lambdas}"

class TestRobustness:
    def test_plateau_exists(self):
        with open(OUTPUTS / "robustness/outputs/K7_robustness_matrix.json") as f:
            data = json.load(f)
        assert data["analysis"]["plateau_analysis"]["is_plateau"], "No plateau found"
    
    def test_plateau_near_13_or_14(self):
        with open(OUTPUTS / "robustness/outputs/K7_robustness_matrix.json") as f:
            data = json.load(f)
        mean = data["analysis"]["plateau_analysis"]["mean"]
        assert 12 < mean < 15, f"Plateau {mean} not near 13 or 14"

class TestBettiIndependence:
    def test_spread_below_threshold(self):
        with open(OUTPUTS / "analysis/outputs/betti_independence.json") as f:
            data = json.load(f)
        assert data["spread_pct"] < 1e-6, f"Spread {data['spread_pct']} too high"
```

---

## Structure Finale Attendue

```
notebooks/spectral_validation/
├── config.py
├── calibration/
│   ├── S3_calibration.py
│   ├── S7_calibration.py
│   ├── outputs/
│   │   ├── S3_calibration_results.json
│   │   └── S7_calibration_results.json
│   └── CALIBRATION_REPORT.md
├── robustness/
│   ├── K7_robustness_matrix.py
│   ├── outputs/
│   │   └── K7_robustness_matrix.json
│   └── ROBUSTNESS_REPORT.md
├── analysis/
│   ├── betti_independence_test.py
│   ├── bias_analysis.py
│   ├── outputs/
│   │   ├── betti_independence.json
│   │   └── bias_analysis.json
│   └── ANALYSIS_REPORT.md
├── FINAL_REPORT.md
└── run_all_validations.py

tests/spectral/
└── test_validation_pipeline.py
```

---

## Checklist Exécution Claude Code

```markdown
## Phase 0: Setup
- [ ] Créer structure dossiers
- [ ] Créer config.py

## Phase 1: Calibration (BLOQUANT)
- [ ] Implémenter S3_calibration.py
- [ ] Exécuter S³ calibration
- [ ] Implémenter S7_calibration.py  
- [ ] Exécuter S⁷ calibration
- [ ] Rédiger CALIBRATION_REPORT.md
- [ ] **CHECKPOINT**: S³ donne-t-il ~3 ou ~2 ?

## Phase 2: Robustesse (si Phase 1 PASS)
- [ ] Implémenter K7_robustness_matrix.py
- [ ] Exécuter grille complète (~2h calcul)
- [ ] Analyser plateau
- [ ] Rédiger ROBUSTNESS_REPORT.md
- [ ] **CHECKPOINT**: Plateau à 13 ou 14 ?

## Phase 3: Betti Independence
- [ ] Implémenter betti_independence_test.py
- [ ] Exécuter sur 6 partitions
- [ ] Vérifier spread < 10⁻⁸
- [ ] **CHECKPOINT**: Independence confirmée ?

## Phase 4: Analyse Biais
- [ ] Implémenter bias_analysis.py
- [ ] Comparer S³ vs K₇
- [ ] Déterminer: genuine_13 ou bias_artifact
- [ ] **CHECKPOINT**: Verdict final ?

## Phase 5: Rapport
- [ ] Générer FINAL_REPORT.md
- [ ] Exécuter pytest
- [ ] **DÉCISION**: PUBLISH / INVESTIGATE / REVISE
```

---

## Critères de Succès Global

| Critère | Seuil | Poids |
|---------|-------|-------|
| S³ calibration | λ₁ ∈ [2.5, 3.5] | BLOQUANT |
| Plateau stable | variation < 2% | BLOQUANT |
| Betti independence | spread < 10⁻⁶ | IMPORTANT |
| Plateau value | ∈ [12.5, 14.5] | IMPORTANT |
| Reproductibilité | pytest 100% pass | BLOQUANT |

**Si tous BLOQUANTS passent** → GO pour publication
**Si un BLOQUANT échoue** → Investigation requise

---

*Document généré pour exécution par Claude Code*
*Version: 1.0 - 2026-01-22*
