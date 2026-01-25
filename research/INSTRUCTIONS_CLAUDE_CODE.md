# Instructions pour Claude Code

## Mission
Créer un notebook Colab (`GIFT_Spectral_Topology.ipynb`) qui calcule le spectre du Laplacien sur différentes variétés pour comprendre l'effet de la topologie sur λ₁.

## Contexte GIFT
- On a calculé λ₁ × H* = 89.47 sur T⁷ plat
- On conjecture λ₁ × H* = 14 sur K₇ (vraie variété G₂)
- Le facteur mystère est 89.47/14 ≈ 6.39
- But : comprendre d'où vient ce facteur 6.39

## Constantes Clés
```python
H_STAR = 99          # b2 + b3 + 1 pour K7
DIM_G2 = 14          # dimension du groupe G2
DET_G = 65/32        # déterminant métrique G2
G_II = DET_G**(1/7)  # ≈ 1.1065
```

## Variétés à Calculer (par priorité)

### 1. S³ [VALIDATION]
- Spectre analytique : λ_k = k(k+2), donc λ₁ = 3
- Sert à valider le code et calibrer

### 2. T⁷ avec métrique G₂ [RÉFÉRENCE]
- Recalculer λ₁ à λ₁₀
- Confirmer λ₁ × H* ≈ 89.47

### 3. S³ × S⁴ [DIM 7]
- Premier modèle de dimension 7
- λ₁ = min(λ₁(S³), λ₁(S⁴)) = min(3, 4) = 3
- Betti : b₃=1, b₄=1, donc H* = 3

### 4. Produits avec Betti croissants
- S³ × S² × S² (dim 7, b₂=2)
- S² × S² × S³ × ... 
- Observer tendance λ₁ × H* vs H*

## Technique CuPy

```python
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import eigsh

# Laplacien produit M1 × M2
# Δ = Δ₁ ⊗ I₂ + I₁ ⊗ Δ₂  (somme de Kronecker)

# Eigenvalues : utiliser which='SA' pas 'SM'
eigenvalues = eigsh(laplacian, k=10, which='SA', return_eigenvectors=False)
```

## Output Requis

Fichier JSON `spectral_topology_results.json` avec :
```json
{
  "S3": {"lambda1": ..., "analytical": 3.0, "calibration": ...},
  "T7": {"lambda1": ..., "product_H_star": 89.47},
  "S3xS4": {"lambda1": ..., "H_star": 3, "product": ...},
  ...
  "factor_analysis": {
    "observed_factor": 6.39,
    "formula_test": "H_star / (dim_G2 * g_ii)"
  }
}
```

## Question Principale à Répondre

**Est-ce que λ₁ × H* décroît quand H* augmente ?**

Si oui, on pourrait avoir :
- T⁷ (H*=1) : λ₁ × H* ~ 89
- K₇ (H*=99) : λ₁ × H* ~ 14

Avec une loi du type λ₁ × H* ∝ 1/H* ou log(H*) ?

---

Voir `PLAN_SPECTRAL_NOTEBOOK.md` pour les détails complets.
