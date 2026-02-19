# Plan pour Notebook Colab : Spectres sur Variétés à Topologie Non-Triviale

## Objectif

Créer un notebook Colab avec CuPy (GPU CUDA) pour calculer le spectre du Laplacien sur des variétés de complexité croissante, afin de comprendre comment la topologie affecte λ₁ et dériver le facteur 6.39 entre T⁷ et K₇.

---

## Constantes GIFT à Utiliser

```python
# Constantes topologiques K₇
DIM_K7 = 7
DIM_G2 = 14
B2_K7 = 21      # Second Betti number
B3_K7 = 77      # Third Betti number
H_STAR = 99     # = b2 + b3 + 1

# Métrique G₂
DET_G = 65/32   # = 2.03125
G_II = DET_G**(1/7)  # ≈ 1.1065

# Nombres de Coxeter
H_G2 = 6
H_E6 = 12
H_E7 = 18
H_E8 = 30

# Résultats connus sur T⁷
LAMBDA1_T7 = 1/G_II  # ≈ 0.9037
PRODUCT_T7 = LAMBDA1_T7 * H_STAR  # ≈ 89.47

# Cible conjecturée pour K₇
TARGET_K7 = DIM_G2  # = 14
FACTOR_T7_TO_K7 = PRODUCT_T7 / TARGET_K7  # ≈ 6.39
```

---

## Structure du Notebook

### Section 1 : Setup et Imports

```python
# Titre: "GIFT Spectral Analysis - Topological Effects on Eigenvalues"
# 
# Imports requis:
# - cupy as cp (GPU arrays)
# - cupyx.scipy.sparse as cpx_sparse
# - cupyx.scipy.sparse.linalg.eigsh
# - numpy as np (pour comparaisons CPU)
# - scipy.sparse pour construction matrices
# - matplotlib pour visualisation
# - json pour export résultats
#
# Vérifier GPU disponible: cp.cuda.runtime.getDeviceCount()
```

### Section 2 : Fonctions Utilitaires

```python
def build_laplacian_sphere(N, dim, radius=1.0):
    """
    Construit le Laplacien discret sur S^dim
    Utilise coordonnées sphériques ou embedding dans R^{dim+1}
    """
    pass

def build_laplacian_torus(N, dim, metric_diag=None):
    """
    Laplacien sur T^dim avec métrique diagonale optionnelle
    Déjà implémenté dans notebooks précédents
    """
    pass

def build_laplacian_product(N, manifold1_params, manifold2_params):
    """
    Laplacien sur M1 × M2 via somme de Kronecker
    Δ_{M1×M2} = Δ_{M1} ⊗ I + I ⊗ Δ_{M2}
    """
    pass

def compute_spectrum(laplacian, k=10):
    """
    Calcule les k plus petites valeurs propres (non nulles)
    Utilise eigsh avec shift-invert pour précision
    """
    pass

def calibrate_eigenvalue(lambda_numerical, lambda_analytical):
    """
    Facteur de calibration pour corriger erreur de discrétisation
    """
    return lambda_analytical / lambda_numerical
```

### Section 3 : S³ - Validation (Spectre Connu)

**Spectre analytique de S³ (rayon R=1) :**
```
λ_k = k(k+2) / R²  pour k = 1, 2, 3, ...
```
Donc λ₁ = 3, λ₂ = 8, λ₃ = 15, ...

**Multiplicités :** m_k = (k+1)²

**À faire :**
1. Implémenter Laplacien sur S³ via embedding dans R⁴
2. Grille : N = 10, 15, 20, 30 points par dimension
3. Calculer λ₁ à λ₅ numériquement
4. Comparer aux valeurs analytiques
5. Déterminer facteur de calibration

**Output attendu :**
```json
{
  "manifold": "S3",
  "dimension": 3,
  "grid_sizes": [10, 15, 20, 30],
  "lambda1_numerical": [...],
  "lambda1_analytical": 3.0,
  "calibration_factors": [...],
  "betti_numbers": {"b0": 1, "b1": 0, "b2": 0, "b3": 1}
}
```

### Section 4 : S⁴ - Extension

**Spectre analytique de S⁴ :**
```
λ_k = k(k+3) / R²  pour k = 1, 2, 3, ...
```
Donc λ₁ = 4, λ₂ = 10, λ₃ = 18, ...

**À faire :** Même procédure que S³

### Section 5 : Produits S³ × Sⁿ

**Spectre de M₁ × M₂ :**
```
λ_{i,j}(M₁ × M₂) = λ_i(M₁) + λ_j(M₂)
```

**Cas S³ × S¹ (dim = 4) :**
- λ₁(S³) = 3, λ₁(S¹) = (2π)² ≈ 39.48 (pour S¹ de période 1)
- Premier λ non nul = min(3, 39.48) = 3

**Cas S³ × S³ (dim = 6) :**
- λ₁ = 3 + 0 = 3 (mode (1,0) ou (0,1))
- Betti: b₀=1, b₃=2, b₆=1

**Cas S³ × S⁴ (dim = 7) :**
- λ₁ = min(3, 4) = 3
- Betti: b₀=1, b₃=1, b₄=1, b₇=1

**À faire :**
1. Construire Laplacien produit via Kronecker
2. Calculer spectre pour différentes tailles
3. Vérifier formule λ_{i,j} = λ_i + λ_j
4. Noter les nombres de Betti

### Section 6 : T⁷ avec Métrique G₂ (Référence)

**Rappel des résultats existants :**
- λ₁ × H* = 89.47 sur T⁷ plat avec g_ii = (65/32)^{1/7}

**À faire :**
1. Recalculer pour validation avec nouveau code
2. Calculer λ₁ à λ₁₀ (pas seulement λ₁)
3. Chercher structure dans les ratios λ_n/λ₁

### Section 7 : Modèle TCS Simplifié

**Construction Twisted Connected Sum :**
```
K₇ ≈ (S¹ × CY₃)_L ∪_φ (S¹ × CY₃)_R
```

**Approximation accessible : S¹ × (S² × S²) × S²**
- Dimension : 1 + 2 + 2 + 2 = 7 ✓
- Betti non triviaux : b₂ = 3 (trois S²)

**Ou : S³ × S² × S²**
- Dimension : 3 + 2 + 2 = 7 ✓
- Betti : b₂ = 2, b₃ = 1

**À faire :**
1. Construire Laplacien pour ces produits
2. Comparer λ₁ × (b₂ + b₃ + 1) aux prédictions
3. Chercher tendance quand b₂, b₃ augmentent

### Section 8 : Analyse du Facteur Topologique

**Hypothèse à tester :**
```
λ₁(M) × H*(M) ≈ f(dim(M), holonomy)
```

où f pourrait être :
- dim(G₂) = 14 pour holonomie G₂
- dim(SU(3)) = 8 pour holonomie SU(3) (Calabi-Yau)
- etc.

**Collecter les données :**
| Manifold | dim | b₂ | b₃ | H* | λ₁ | λ₁×H* |
|----------|-----|----|----|----|----|-------|
| T⁷       | 7   | 0  | 0  | 1  | ?  | ?     |
| S⁷       | 7   | 0  | 0  | 1  | 6  | 6     |
| S³×S⁴    | 7   | 0  | 1  | 2  | 3  | 6     |
| S³×S²×S² | 7   | 2  | 1  | 4  | ?  | ?     |
| ...      |     |    |    |    |    |       |

### Section 9 : Export des Résultats

```python
results = {
    "metadata": {
        "date": "2026-01-XX",
        "gpu": "T4/A100/...",
        "cupy_version": "...",
    },
    "gift_constants": {
        "dim_K7": 7,
        "dim_G2": 14,
        "b2": 21,
        "b3": 77,
        "H_star": 99,
        "det_g": 65/32,
    },
    "S3_results": {...},
    "S4_results": {...},
    "product_results": {...},
    "T7_results": {...},
    "tcs_approx_results": {...},
    "factor_analysis": {
        "T7_product": 89.47,
        "target_K7": 14,
        "factor": 6.39,
        "interpretation": "..."
    }
}

# Sauvegarder en JSON
with open('spectral_topology_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## Spécifications Techniques

### GPU/CuPy

```python
# Configuration mémoire
mempool = cp.get_default_memory_pool()
mempool.set_limit(size=4 * 1024**3)  # 4 GB max

# Pour matrices creuses
# Utiliser format COO pour construction, CSR pour calculs
# eigsh avec which='SA' (smallest algebraic), pas 'SM'

# Nettoyage mémoire entre gros calculs
mempool.free_all_blocks()
```

### Tailles de Grille Recommandées

| Manifold | dim | N_max (T4 GPU) | N_max (A100) |
|----------|-----|----------------|--------------|
| S³       | 3   | 50             | 100          |
| S⁴       | 4   | 30             | 50           |
| T⁷       | 7   | 12             | 20           |
| S³×S⁴    | 7   | 10             | 15           |

### Validation

Pour chaque calcul :
1. Comparer à analytique si disponible
2. Vérifier convergence en N
3. Checker que λ₀ ≈ 0 (mode constant)
4. Vérifier multiplicités si connues

---

## Outputs Attendus pour Upload

1. **spectral_topology_results.json** - Toutes les données numériques
2. **plots/** - Graphiques de convergence et comparaisons
3. **summary.txt** - Résumé textuel des découvertes

---

## Questions Clés à Répondre

1. **Comment λ₁ × H* varie-t-il avec b₂, b₃ ?**
   - Augmente ? Diminue ? Reste constant ?

2. **Le facteur 6.39 peut-il s'exprimer comme H*/dim(G₂)/g_ii ?**
   - Vérifier algébriquement

3. **Y a-t-il une formule universelle λ₁ × H* = f(holonomie) ?**
   - Comparer différentes holonomies

4. **Le passage T⁷ → K₇ est-il "continu" en topologie ?**
   - Interpoler via produits intermédiaires

---

## Priorité d'Exécution

1. ⭐⭐⭐ S³ (validation du code)
2. ⭐⭐⭐ T⁷ avec métrique G₂ (référence GIFT)
3. ⭐⭐ S³ × S⁴ (dim 7, topologie simple)
4. ⭐⭐ Produits avec b₂, b₃ croissants
5. ⭐ Modèles TCS si temps permet

---

*Plan préparé pour Claude Code - Janvier 2026*
