# Side Quest: Calabi-Yau Spectral Validation 🦋

**Objectif**: Tester la conjecture universelle λ₁ × H* = dim(Hol) - h sur CY₃
**Prédiction**: λ₁ × H* = 8 - 2 = **6** (si la conjecture tient)
**Statut**: Papillon pour la potion (Skyrim style)

---

## La Conjecture à Tester

Pour une variété à holonomie spéciale:

$$\lambda_1 \times H^* = \dim(\text{Hol}) - h$$

| Variété | Holonomie | dim(Hol) | h (spineurs //) | Prédiction |
|---------|-----------|----------|-----------------|------------|
| K₇ (G₂) | G₂ | 14 | 1 | **13** ✓ |
| **CY₃** | **SU(3)** | **8** | **2** | **6** ? |
| K3 | SU(2) | 3 | 2 | **1** ? |
| Spin(7) | Spin(7) | 21 | 1 | **20** ? |

**Si CY₃ donne 6, la conjecture devient très crédible.**

---

## Contexte Mathématique: CY₃ vs G₂

### Calabi-Yau 3-folds (CY₃)

| Propriété | Valeur |
|-----------|--------|
| Dimension réelle | 6 |
| Holonomie | SU(3) ⊂ SO(6) |
| dim(SU(3)) | 8 |
| Spineurs parallèles | h = 2 (chiral + anti-chiral) |
| SUSY préservée | N = 2 en 4D |

### Nombres de Hodge CY₃

```
        h⁰⁰ = 1
       h¹⁰  h⁰¹ = 0  0
      h²⁰  h¹¹  h⁰² = 0  h¹¹  0
     h³⁰  h²¹  h¹²  h⁰³ = 1  h²¹  h²¹  1
      h³¹  h²²  h¹³ = 0  h¹¹  0
       h³²  h²³ = 0  0
        h³³ = 1
```

**Deux nombres indépendants**: h¹¹ et h²¹

**Nombres de Betti**:
- b₀ = b₆ = 1
- b₁ = b₅ = 0
- b₂ = b₄ = h¹¹
- b₃ = 2(h²¹ + 1) = 2h²¹ + 2

### Définition de H* pour CY₃

Par analogie avec G₂ où H* = b₂ + b₃ + 1:

**Option A** (middle Betti + spineurs):
```
H*_CY = h¹¹ + h²¹ + 2
```

**Option B** (somme Betti paires):
```
H*_CY = b₂ + b₃ + 2 = h¹¹ + 2h²¹ + 4
```

**Option C** (Euler / 2 + correction):
```
H*_CY = |χ|/2 + 2 = |h¹¹ - h²¹| + 2
```

**On testera les trois pour voir laquelle donne λ₁ × H* = 6.**

---

## Variétés CY₃ Candidates

### Tier 1: Métriques (semi-)explicites

| Variété | h¹¹ | h²¹ | χ | Métrique | Difficulté |
|---------|-----|-----|---|----------|------------|
| **T⁶** (limite plate) | 9 | 9 | 0 | Plate! | ⭐ |
| **T⁶/ℤ₃** orbifold | 9 | 0 | 18 | Plate + singularités | ⭐⭐ |
| **Produit K3 × T²** | 21 | 21 | 0 | Ricci-flat connue | ⭐⭐ |

### Tier 2: Constructions standard

| Variété | h¹¹ | h²¹ | χ | Notes |
|---------|-----|-----|---|-------|
| **Quintic** P⁴[5] | 1 | 101 | -200 | Le plus célèbre |
| **Bicubic** P²×P²[3,3] | 1 | 73 | -144 | CICY simple |
| **Mirror Quintic** | 101 | 1 | 200 | Miroir du quintic |

### Tier 3: Pour comparaison multi-H*

| Variété | h¹¹ | h²¹ | H*_A | H*_B |
|---------|-----|-----|------|------|
| CY_small | 1 | 1 | 4 | 7 |
| CY_medium | 10 | 10 | 22 | 34 |
| CY_large | 50 | 50 | 102 | 154 |

---

## Phase 0: Setup Infrastructure (Jour 1)

### Tâche 0.1: Structure
```bash
mkdir -p notebooks/cy3_validation/{sampling,spectral,analysis}
mkdir -p tests/cy3
```

### Tâche 0.2: Configuration CY₃
Créer `notebooks/cy3_validation/config_cy3.py`:

```python
"""Configuration pour validation spectrale CY₃."""

# Prédiction conjecture
PREDICTION = {
    "dim_SU3": 8,
    "h_spinors": 2,
    "lambda1_x_Hstar_target": 6,  # = 8 - 2
}

# Définitions H* à tester
def Hstar_A(h11, h21):
    """Option A: h¹¹ + h²¹ + h"""
    return h11 + h21 + 2

def Hstar_B(h11, h21):
    """Option B: b₂ + b₃ + h = h¹¹ + 2h²¹ + 4"""
    return h11 + 2 * h21 + 4

def Hstar_C(h11, h21):
    """Option C: |χ|/2 + h"""
    return abs(h11 - h21) + 2

# Variétés de test
CY3_MANIFOLDS = {
    # Tier 1: Métriques explicites
    "T6_flat": {
        "h11": 9, "h21": 9, "chi": 0,
        "metric": "flat",
        "description": "6-torus (trivial holonomy limit)",
    },
    "T6_Z3": {
        "h11": 9, "h21": 0, "chi": 18,
        "metric": "orbifold",
        "description": "T⁶/ℤ₃ orbifold",
    },
    # Tier 2: Constructions CICY
    "Quintic": {
        "h11": 1, "h21": 101, "chi": -200,
        "metric": "numerical",
        "description": "Quintic hypersurface in P⁴",
    },
    "Bicubic": {
        "h11": 1, "h21": 73, "chi": -144,
        "metric": "numerical",
        "description": "Bicubic in P²×P²",
    },
    # Tier 3: Sweep H*
    "CY_sweep_small": {"h11": 2, "h21": 2, "chi": 0},
    "CY_sweep_medium": {"h11": 11, "h21": 11, "chi": 0},
    "CY_sweep_large": {"h11": 51, "h21": 51, "chi": 0},
}

# Paramètres numériques
NUMERICAL_PARAMS = {
    "N_samples": [1000, 2000, 5000, 10000],
    "k_neighbors": [15, 25, 40],
    "laplacian_type": "symmetric",
}

# Critères PASS/FAIL
TOLERANCES = {
    "target_match": 0.10,  # 10% de λ₁×H* = 6
    "Hstar_consistency": 0.05,  # Les 3 définitions donnent même résultat ±5%
}
```

---

## Phase 1: T⁶ Plat - Calibration (Jours 2-3)

**Pourquoi T⁶ ?** Métrique plate connue exactement, spectre analytique connu.

### Spectre du Laplacien sur T⁶

Pour T⁶ = (S¹)⁶ avec rayons R₁,...,R₆:

$$\lambda_{n_1,...,n_6} = \sum_{i=1}^{6} \frac{n_i^2}{R_i^2}$$

**λ₁ (premier non-nul)** = 1/R²_max (si tous rayons égaux R)

### Tâche 1.1: Échantillonnage T⁶
Créer `notebooks/cy3_validation/sampling/T6_sampling.py`:

```python
"""
Échantillonnage uniforme sur T⁶ = [0, 2π)⁶
"""

import numpy as np

def sample_T6_uniform(N: int, radii: list = None, seed: int = 42) -> np.ndarray:
    """
    Échantillonne N points uniformément sur T⁶.
    
    Args:
        N: nombre de points
        radii: [R₁,...,R₆] rayons (défaut: tous égaux à 1)
        seed: graine aléatoire
    
    Returns:
        points: (N, 6) array de coordonnées angulaires
    """
    rng = np.random.default_rng(seed)
    
    if radii is None:
        radii = [1.0] * 6
    
    # Coordonnées angulaires uniformes
    angles = rng.uniform(0, 2 * np.pi, size=(N, 6))
    
    # Pondération par rayons pour métrique
    points = angles * np.array(radii)
    
    return points

def geodesic_distance_T6(p1: np.ndarray, p2: np.ndarray, radii: list = None) -> np.ndarray:
    """
    Distance géodésique sur T⁶ (distance torique).
    
    La distance sur chaque S¹ est min(|θ₁-θ₂|, 2π - |θ₁-θ₂|).
    """
    if radii is None:
        radii = [1.0] * 6
    
    radii = np.array(radii)
    
    # Différence angulaire
    diff = np.abs(p1 - p2)
    
    # Distance torique sur chaque cercle
    diff_toric = np.minimum(diff, 2 * np.pi * radii - diff)
    
    # Distance euclidienne dans l'espace produit
    return np.sqrt(np.sum(diff_toric**2, axis=-1))

def lambda1_T6_exact(radii: list = None) -> float:
    """
    λ₁ exact sur T⁶ = min_{n≠0} Σᵢ nᵢ²/Rᵢ²
    """
    if radii is None:
        radii = [1.0] * 6
    
    # λ₁ = 1/R²_max (mode le plus bas non-constant)
    return 1.0 / max(radii)**2

# H* pour T⁶ (h¹¹ = h²¹ = 9 pour holonomie triviale "CY-like")
def Hstar_T6():
    """
    T⁶ a h¹¹ = h²¹ = 9 (comme limite de CY avec χ=0).
    Mais holonomie = {1}, pas SU(3).
    
    Pour calibration, on utilise juste les nombres.
    """
    h11, h21 = 9, 9
    return {
        "A": h11 + h21 + 2,      # = 20
        "B": h11 + 2*h21 + 4,    # = 31
        "C": abs(h11 - h21) + 2,  # = 2
    }
```

### Tâche 1.2: Test Laplacien sur T⁶
```python
def test_T6_spectral():
    """
    Vérifie que le pipeline reproduit λ₁(T⁶).
    
    ATTENTION: T⁶ n'a PAS holonomie SU(3), c'est juste une calibration.
    """
    from spectral_utils import build_graph_laplacian, compute_lambda1
    
    results = []
    
    for N in [1000, 2000, 5000]:
        points = sample_T6_uniform(N, radii=[1.0]*6)
        
        # Distance torique
        # Note: on doit adapter build_graph_laplacian pour utiliser geodesic_distance_T6
        
        L = build_graph_laplacian_custom(points, k=25, 
                                          distance_fn=geodesic_distance_T6)
        lambda1_measured = compute_lambda1(L)
        lambda1_exact = lambda1_T6_exact()  # = 1.0
        
        results.append({
            "N": N,
            "lambda1_measured": lambda1_measured,
            "lambda1_exact": lambda1_exact,
            "deviation_pct": abs(lambda1_measured - lambda1_exact) / lambda1_exact * 100,
        })
    
    return results
```

### Critère PASS Phase 1

| Test | Condition PASS |
|------|----------------|
| T⁶ λ₁ | Mesure dans ±20% de λ₁_exact |
| Convergence | λ₁(N) → λ₁_exact quand N ↑ |

**Note**: T⁶ est une calibration du pipeline, PAS un test de la conjecture (holonomie triviale).

---

## Phase 2: Orbifold T⁶/ℤ₃ (Jours 4-5)

**Pourquoi T⁶/ℤ₃ ?** Premier vrai CY₃ avec métrique (presque) explicite.

### Structure de T⁶/ℤ₃

L'action ℤ₃ sur T⁶ = ℂ³/Λ:
```
g: (z₁, z₂, z₃) → (ωz₁, ωz₂, ωz₃)    où ω = e^{2πi/3}
```

**Nombres de Hodge**: h¹¹ = 9, h²¹ = 0, χ = 18

**Singularités**: 27 points fixes (résolus en P² exceptionnels)

### Tâche 2.1: Échantillonnage T⁶/ℤ₃
```python
"""
Échantillonnage sur l'orbifold T⁶/ℤ₃.
"""

def sample_T6_Z3_orbifold(N: int, seed: int = 42) -> np.ndarray:
    """
    Échantillonne sur T⁶/ℤ₃.
    
    Stratégie: échantillonner T⁶, puis projeter sur domaine fondamental.
    """
    rng = np.random.default_rng(seed)
    
    # Échantillonner T⁶
    points_T6 = sample_T6_uniform(N * 3, seed=seed)  # oversample
    
    # Projeter sur domaine fondamental de ℤ₃
    # Action: (θ₁, θ₂, θ₃, θ₄, θ₅, θ₆) → (θ₁+2π/3, θ₂+2π/3, θ₃+2π/3, ...)
    
    # Domaine fondamental: θ₁ ∈ [0, 2π/3)
    mask = points_T6[:, 0] < 2 * np.pi / 3
    points_fund = points_T6[mask][:N]
    
    return points_fund

def geodesic_distance_T6_Z3(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Distance sur T⁶/ℤ₃ = min sur les 3 copies ℤ₃.
    """
    omega = 2 * np.pi / 3
    
    distances = []
    for k in range(3):
        # Rotation par ω^k
        p2_rotated = p2.copy()
        p2_rotated[:, :3] += k * omega  # rotation sur les 3 premiers angles
        p2_rotated = np.mod(p2_rotated, 2 * np.pi)
        
        d = geodesic_distance_T6(p1, p2_rotated)
        distances.append(d)
    
    return np.minimum.reduce(distances)
```

### Tâche 2.2: Test spectral T⁶/ℤ₃

```python
def test_T6_Z3_spectral():
    """
    Test de la conjecture sur T⁶/ℤ₃.
    
    h¹¹ = 9, h²¹ = 0
    H*_A = 9 + 0 + 2 = 11
    H*_B = 9 + 0 + 4 = 13
    
    Prédiction: λ₁ × H* = 6
    Donc: λ₁ = 6/11 ≈ 0.545 (option A)
          λ₁ = 6/13 ≈ 0.462 (option B)
    """
    results = []
    
    for N in [2000, 5000, 10000]:
        points = sample_T6_Z3_orbifold(N)
        L = build_graph_laplacian_custom(points, k=25,
                                          distance_fn=geodesic_distance_T6_Z3)
        lambda1 = compute_lambda1(L)
        
        # Tester les différentes définitions H*
        h11, h21 = 9, 0
        
        for name, Hstar_fn in [("A", Hstar_A), ("B", Hstar_B), ("C", Hstar_C)]:
            Hstar = Hstar_fn(h11, h21)
            product = lambda1 * Hstar
            deviation = abs(product - 6) / 6 * 100
            
            results.append({
                "N": N,
                "Hstar_def": name,
                "Hstar": Hstar,
                "lambda1": lambda1,
                "lambda1_x_Hstar": product,
                "deviation_from_6_pct": deviation,
            })
    
    return results
```

---

## Phase 3: CICY Numériques (Jours 6-8)

**Complete Intersection Calabi-Yau** - métriques approximées par méthodes numériques.

### Approche: Donaldson Algorithm / ML

Pour les CICY sans métrique explicite, on utilise:
1. **Donaldson's algorithm**: itération pour approximer métrique Ricci-flat
2. **Neural network**: PINN pour apprendre la métrique

### Tâche 3.1: Interface Quintic

```python
"""
Interface pour le Quintic P⁴[5].

Le Quintic est défini par {z ∈ P⁴ | p(z) = 0} où p est polynôme degré 5.
Exemple: p = z₀⁵ + z₁⁵ + z₂⁵ + z₃⁵ + z₄⁵ - 5ψ z₀z₁z₂z₃z₄

h¹¹ = 1, h²¹ = 101
"""

def sample_quintic_hypersurface(N: int, psi: float = 1.0, seed: int = 42):
    """
    Échantillonne N points sur le Quintic.
    
    Méthode: rejection sampling sur P⁴, accepter si proche de l'hypersurface.
    """
    rng = np.random.default_rng(seed)
    
    points = []
    attempts = 0
    max_attempts = N * 100
    
    while len(points) < N and attempts < max_attempts:
        # Point aléatoire dans P⁴ (5 coordonnées complexes, normalisées)
        z = rng.standard_normal((5,)) + 1j * rng.standard_normal((5,))
        z = z / np.linalg.norm(z)
        
        # Évaluer le polynôme
        p = sum(z[i]**5 for i in range(5)) - 5 * psi * np.prod(z)
        
        # Accepter si |p| < ε (proche de l'hypersurface)
        if np.abs(p) < 0.01:
            # Projeter exactement sur l'hypersurface (Newton)
            z_proj = newton_project_quintic(z, psi)
            points.append(z_proj)
        
        attempts += 1
    
    return np.array(points)

def Hstar_quintic():
    """H* pour le Quintic."""
    h11, h21 = 1, 101
    return {
        "A": h11 + h21 + 2,      # = 104
        "B": h11 + 2*h21 + 4,    # = 207
        "C": abs(h11 - h21) + 2,  # = 102
    }
```

### Tâche 3.2: Test multi-CICY

```python
def test_cicy_sweep():
    """
    Teste plusieurs CICY avec différents (h¹¹, h²¹).
    
    Objectif: voir si λ₁ × H* ≈ 6 est universel.
    """
    cicys = [
        {"name": "Quintic", "h11": 1, "h21": 101},
        {"name": "Bicubic", "h11": 1, "h21": 73},
        {"name": "Sextic_P5", "h11": 1, "h21": 103},
        {"name": "CICY_7862", "h11": 19, "h21": 19},  # χ = 0 example
    ]
    
    results = []
    
    for cicy in cicys:
        print(f"Testing {cicy['name']}...")
        
        # Sampling (utilise méthode appropriée)
        points = sample_cicy(cicy["name"], N=5000)
        
        # Spectral
        L = build_graph_laplacian(points, k=25)
        lambda1 = compute_lambda1(L)
        
        # Test toutes définitions H*
        for Hstar_name, Hstar_fn in [("A", Hstar_A), ("B", Hstar_B)]:
            Hstar = Hstar_fn(cicy["h11"], cicy["h21"])
            product = lambda1 * Hstar
            
            results.append({
                **cicy,
                "Hstar_def": Hstar_name,
                "Hstar": Hstar,
                "lambda1": lambda1,
                "product": product,
                "dev_from_6": abs(product - 6) / 6 * 100,
            })
    
    return results
```

---

## Phase 4: Analyse & Décision (Jours 9-10)

### Tâche 4.1: Identifier la bonne définition H*

```python
def analyze_Hstar_definitions(results: list) -> dict:
    """
    Quelle définition de H* donne λ₁ × H* ≈ 6 de manière consistante ?
    """
    by_definition = {}
    
    for Hstar_def in ["A", "B", "C"]:
        subset = [r for r in results if r["Hstar_def"] == Hstar_def]
        products = [r["product"] for r in subset]
        
        by_definition[Hstar_def] = {
            "mean_product": np.mean(products),
            "std_product": np.std(products),
            "deviation_from_6": abs(np.mean(products) - 6) / 6 * 100,
            "is_consistent": np.std(products) / np.mean(products) < 0.1,
        }
    
    # Identifier le gagnant
    winner = min(by_definition.keys(), 
                 key=lambda k: by_definition[k]["deviation_from_6"])
    
    return {
        "by_definition": by_definition,
        "winner": winner,
        "winner_deviation": by_definition[winner]["deviation_from_6"],
    }
```

### Tâche 4.2: Comparaison G₂ vs CY₃

```python
def compare_G2_CY3():
    """
    Tableau comparatif final.
    """
    return """
    | Holonomie | dim(Hol) | h | Target | Measured | Deviation |
    |-----------|----------|---|--------|----------|-----------|
    | G₂        | 14       | 1 | 13     | 13.45    | 3.5%      |
    | SU(3)     | 8        | 2 | 6      | ???      | ???       |
    
    Si CY₃ donne λ₁×H* ≈ 6 → CONJECTURE VALIDÉE
    Si CY₃ donne autre chose → Conjecture fausse ou définition H* incorrecte
    """
```

---

## Critères de Succès

### PASS Global (Conjecture validée)

| Critère | Seuil |
|---------|-------|
| Au moins 1 CY₃ donne λ₁×H* ∈ [5.5, 6.5] | REQUIS |
| Définition H* consistante entre CY₃s | variation < 15% |
| Même définition H* marche pour G₂ | cross-check |

### FAIL (Conjecture réfutée)

| Critère | Interprétation |
|---------|----------------|
| Tous CY₃ donnent λ₁×H* >> 6 ou << 6 | Conjecture fausse |
| Aucune définition H* n'est consistante | Formule incorrecte |
| G₂ et CY₃ incompatibles | Pas de loi universelle |

---

## Structure Finale

```
notebooks/cy3_validation/
├── config_cy3.py
├── sampling/
│   ├── T6_sampling.py
│   ├── T6_Z3_sampling.py
│   ├── quintic_sampling.py
│   └── cicy_sampling.py
├── spectral/
│   ├── laplacian_custom.py
│   └── eigenvalue_analysis.py
├── analysis/
│   ├── Hstar_comparison.py
│   └── G2_vs_CY3.py
├── outputs/
│   ├── T6_calibration.json
│   ├── T6_Z3_results.json
│   ├── cicy_sweep.json
│   └── final_comparison.json
├── CY3_VALIDATION_REPORT.md
└── run_cy3_validation.py

tests/cy3/
├── test_T6_calibration.py
└── test_conjecture.py
```

---

## Timeline

| Phase | Jours | Objectif | Bloquant? |
|-------|-------|----------|-----------|
| 0. Setup | 1 | Infrastructure | Non |
| 1. T⁶ calibration | 2-3 | Pipeline fonctionne | OUI |
| 2. T⁶/ℤ₃ | 4-5 | Premier vrai CY₃ | OUI |
| 3. CICY sweep | 6-8 | Multi-variétés | Non |
| 4. Analyse | 9-10 | Verdict | - |

**Total**: ~10 jours (side quest raisonnable)

---

## Résultat Espéré

```
╔═══════════════════════════════════════════════════════════════╗
║  CONJECTURE UNIVERSELLE: λ₁ × H* = dim(Hol) - h              ║
╠═══════════════════════════════════════════════════════════════╣
║  G₂ (K₇):   14 - 1 = 13  ✓  (mesuré: 13.45)                 ║
║  SU(3) (CY₃): 8 - 2 = 6   ?  (mesuré: ???)                   ║
╠═══════════════════════════════════════════════════════════════╣
║  Si CY₃ ≈ 6 → Publier "Universal Spectral Law"               ║
║  Si CY₃ ≠ 6 → Réviser ou abandonner conjecture               ║
╚═══════════════════════════════════════════════════════════════╝
```

---

*"Le papillon de Calabi-Yau danse dans l'espace des modules..."* 🦋

**Document Status**: SIDE QUEST READY
**Prerequisite**: G₂ validation (roadmap principale) en parallèle ou avant
