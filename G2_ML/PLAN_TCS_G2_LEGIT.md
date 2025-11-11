# Plan Minimal : Notebook v0.8 → Legit TCS G₂

**Base**: `/home/user/GIFT/G2_ML/Complete_G2_Metric_Training_v0_8.ipynb`
**Objectif**: Implémenter une variété TCS G₂ rigoureuse avec validations publiables

---

## 🎯 Architecture Cible

```
M = M₁ ⊔ (Neck) ⊔ M₂

M₁, M₂ : ACylCY3 (Asymptotically Cylindrical CY3)
Neck   : TCSGlue avec fenêtres lisses
```

**vs. Current**: `[-T,T] × (S¹)² × T⁴` avec ACyl "approximatif"

---

## 📋 Plan en 6 Phases

### **Phase 1 : Refactor Géométrie** 🏗️
**Priorité: CRITIQUE** | **Effort: 2-3 jours**

#### 1.1 Implémenter `ACylCY3`
**Fichier**: Nouveau module `geometry/acyl_cy3.py`

```python
class ACylCY3:
    """
    Asymptotically Cylindrical CY3: R⁺ × (S¹)² × T⁴/Z₂

    Zones:
    - Core: r ∈ [0, r_neck]   → metric compact
    - Neck: r ∈ [r_neck, ∞)  → cyl(S¹×S¹) × eguchi-hanson
    """
    def __init__(self, r_neck=3.0, neck_params=...):
        self.r_neck = r_neck
        self.cy3_core = ...  # Compact CY3 proxy
        self.neck_metric = ...  # Cylindrical part

    def metric(self, coords):
        """Returns g_ij(x) with C² matching at r=r_neck"""
        pass

    def transition_function(self, r):
        """Smooth cutoff: 0 (core) → 1 (neck)"""
        # Use χ(r) = tanh((r - r_neck)/δ) or similar
        pass
```

**Actions concrètes**:
- [ ] Créer `geometry/acyl_cy3.py`
- [ ] Implémenter fenêtre lisse `χ(r)` avec contrôle C²
- [ ] Ajouter paramètres: `r_neck`, `δ_transition`, `neck_radius`
- [ ] Test unitaire: `test_acyl_metric_continuity()`

#### 1.2 Implémenter `TCSGlue`
**Fichier**: `geometry/tcs_glue.py`

```python
class TCSGlue:
    """
    Glue M₁ + M₂ via:
    - Identify necks: (r₁ → ∞) ↔ (r₂ → ∞)
    - Match: (S¹)₁² = (S¹)₂²
    - Smooth interpolation sur [t_min, t_max]
    """
    def __init__(self, acyl1: ACylCY3, acyl2: ACylCY3):
        self.acyl1 = acyl1
        self.acyl2 = acyl2
        self.t_range = (-12, 12)  # Neck length

    def combined_metric(self, x):
        """g_total = χ₁·g₁ + χ₂·g₂ with partition of unity"""
        pass
```

**Actions concrètes**:
- [ ] Créer `geometry/tcs_glue.py`
- [ ] Partition of unity: `χ₁ + χ₂ = 1` sur le cou
- [ ] Vérifier: `||g₁ - g₂|| → 0` à l'interface (tolérance 1e-6)
- [ ] Ajouter cartes locales: `chart_M1`, `chart_neck`, `chart_M2`

#### 1.3 Proxy K3 Rigoureux
**Options**:
1. **Option A** (facile, honnête): Garder T⁴ mais documenter clairement
   ```python
   # Dans le notebook:
   # NOTE: Using T⁴ as K3 proxy (b₂=22 → 21 après quotienting)
   # This is an APPROXIMATION. For rigorous K3, see Option B.
   ```

2. **Option B** (rigoureux, +1 semaine):
   - Implémenter `T⁴/Z₂` lissé (Kummer surface)
   - Utiliser résolution de singularités explicite
   - Référence: Joyce 1996, §7.2

**Décision recommandée**: **Option A** pour le plan minimal, documenter l'approximation.

**Actions**:
- [ ] Ajouter section "## Geometric Approximations" dans le notebook
- [ ] Déclarer: `K3 ≈ T⁴ with b₂=21` + référence académique
- [ ] Limiter les claims: "TCS-like structure" au lieu de "rigorous TCS"

---

### **Phase 2 : Opérateurs Différentiels Rigoureux** 📐
**Priorité: HAUTE** | **Effort: 2 jours**

#### 2.1 Exterior Derivative (d) Antisymétrique
**Fichier**: `operators/differential.py`

**Actuel**: Approx via `∇φ` (pas antisymétrique)
**Cible**: Vrai `dω` avec antisymétrie

```python
def exterior_derivative_pform(omega, g, mesh):
    """
    Compute dω for p-form ω

    Args:
        omega: (batch, n_cells, p-form_components)
        g: metric tensor
        mesh: DifferentialMesh with adjacency

    Returns:
        d_omega: (batch, n_cells, (p+1)-form_components)

    Formula:
        (dω)_{i₀...iₚ} = Σⱼ ∂[iⱼ ω_{i₁...iₚ]}  (antisym bracket)
    """
    p = infer_form_degree(omega)
    d_omega = torch.zeros(..., form_components(p+1))

    for face in mesh.cells:
        # Compute ∂ᵢ ω via finite diff on face
        partial_omega = compute_partial(omega, face, mesh)
        # Antisymmetrize indices
        d_omega[face] = antisymmetrize(partial_omega)

    return d_omega
```

**Actions**:
- [ ] Implémenter `antisymmetrize()` avec signes de permutation
- [ ] Test: `d(dω) = 0` (Poincaré, tolérance 1e-8)
- [ ] Benchmark: comparer avec `torch.autograd` sur formes simples

#### 2.2 Codifferentiel (δ)
**Fichier**: `operators/differential.py`

```python
def codifferential(omega, g, mesh):
    """
    δ = (-1)^{np+n+1} ⋆ d ⋆

    For G₂: n=7
    For 3-forms (p=3): δφ = (-1)^{7·3+7+1} ⋆d⋆φ = -⋆d⋆φ
    """
    star_omega = hodge_star(omega, g, mesh)
    d_star_omega = exterior_derivative_pform(star_omega, g, mesh)
    star_d_star_omega = hodge_star(d_star_omega, g, mesh)

    p = infer_form_degree(omega)
    n = 7
    sign = (-1)**(n*p + n + 1)

    return sign * star_d_star_omega
```

**Actions**:
- [ ] Implémenter `hodge_star(omega, g)` exact (pas approximation)
- [ ] Test: `δδ = 0` (tolérance 1e-8)
- [ ] Profiler: optimiser calculs ⋆ (c'est coûteux)

#### 2.3 Laplacien (Δ)
```python
def laplacian_pform(omega, g, mesh):
    """Δ = dδ + δd (Hodge-de Rham Laplacian)"""
    d_omega = exterior_derivative_pform(omega, g, mesh)
    delta_omega = codifferential(omega, g, mesh)

    delta_d_omega = codifferential(d_omega, g, mesh)
    d_delta_omega = exterior_derivative_pform(delta_omega, g, mesh)

    return delta_d_omega + d_delta_omega
```

**Actions**:
- [ ] Implémenter `laplacian_pform()`
- [ ] Test: Sur formes harmoniques, `Δφ ≈ 0` (tolérance 1e-6)

---

### **Phase 3 : Curvature Monitoring** 📊
**Priorité: HAUTE** | **Effort: 1 jour**

#### 3.1 Christoffel Symbols (amélioration)
**Actuel**: `finite_diff_christoffel()` avec ε=1e-4
**Améliorations**:
- [ ] Stencil adaptatif: `ε = max(1e-5, ||∇g|| · scale)`
- [ ] Vérifier symétrie: `Γⁱⱼₖ = Γⁱₖⱼ` (tolérance 1e-10)

#### 3.2 Ricci Curvature Full Mesh
**Actuel**: Calculé en fin d'entraînement (section "Ricci Flatness")
**Cible**: Monitoring durant training

```python
class RicciMonitor:
    def __init__(self, sample_rate=0.05):
        """Monitor Ricci on 5% of mesh during training"""
        self.sample_indices = ...

    def __call__(self, g, step):
        if step % 100 == 0:
            # Compute on sub-sample
            ricci = compute_ricci_sample(g, self.sample_indices)
            log_metric("ricci/L2_norm", ricci.norm())
            log_metric("ricci/max_abs", ricci.abs().max())
```

**Actions**:
- [ ] Ajouter `RicciMonitor(sample_rate=0.05)` au training loop
- [ ] Full pass: Calculer Ricci sur tout le maillage à steps [1k, 5k, 10k, final]
- [ ] Afficher: `||Ric||_L²`, `max|Ric_ij|`, `mean|Ric_ii - Ric_jj|`

---

### **Phase 4 : Cohomologie Discrète** 🔢
**Priorité: MOYENNE** | **Effort: 2 jours**

#### 4.1 Laplacien Spectral
**Fichier**: `cohomology/spectral.py`

```python
def discrete_laplacian_spectrum(omega, g, mesh, p=3):
    """
    Solve: Δφ = λφ

    Returns:
        eigenvalues: (n_harmonics,)
        eigenvectors: (n_harmonics, n_cells, form_components)
    """
    # Build sparse matrix for Δ
    L = build_laplacian_matrix(g, mesh, p)  # Sparse (N_cells × N_cells)

    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = torch.lobpcg(L, k=50, largest=False)

    # Find harmonic forms: |λ| < tol
    harmonic_mask = (eigenvalues.abs() < 1e-6)

    return eigenvalues[harmonic_mask], eigenvectors[:, harmonic_mask]
```

**Actions**:
- [ ] Implémenter `build_laplacian_matrix()` pour 2-formes et 3-formes
- [ ] Résoudre spectre sur maillage 12⁷ (actuel) et 16⁷ (convergence)
- [ ] Compter: `b₂ = dim(ker Δ₂)`, `b₃ = dim(ker Δ₃)`

#### 4.2 Validation Topologique
```python
def validate_cohomology(b2, b3):
    """
    TCS topology check:
    - b₂ = 21 (from 2 × K3-like)
    - b₃ = 77 (rigorous for TCS)
    - χ = Σ(-1)ⁱ bᵢ should match topology
    """
    chi_computed = -b2 + b3  # For G₂ manifold
    chi_expected = 0  # TCS is null-homologous

    assert abs(chi_computed - chi_expected) < 2, \
        f"Euler char mismatch: {chi_computed} ≠ {chi_expected}"

    print(f"✓ b₂={b2}, b₃={b3}, χ={chi_computed}")
```

**Actions**:
- [ ] Ajouter section "## 6. Cohomology Validation" dans notebook
- [ ] Afficher tableau: `b₀, b₁, b₂, b₃, b₄` avec tolérance d'extraction
- [ ] Comparer avec référence Joyce (TCS théorique)

---

### **Phase 5 : Tableau de Validation Publiable** ✅
**Priorité: CRITIQUE** | **Effort: 1 jour**

#### 5.1 Métriques Globales

| Métrique | Formule | Tolérance Cible | Actuel v0.8 |
|----------|---------|-----------------|-------------|
| **Closedness** | `\\|dφ\\|_L²` | < 1e-6 | ~3e-4 (approx) |
| **Co-closedness** | `\\|δφ\\|_L²` | < 1e-6 | Non mesuré |
| **Harmonicity** | `\\|Δφ\\|_L²` | < 1e-8 | Non mesuré |
| **Ricci Flatness** | `\\|Ric\\|_L²` | < 1e-4 | ~1e-3 (sparse) |
| **Torsion-free** | `\\|dφ - ⋆(φ∧φ)\\|_L²` | < 1e-5 | ~7e-6 ✓ |
| **Volume** | `Vol(M)` | (2π)⁷ ± 0.1% | ✓ |
| **Metric Positivity** | `min eig(g)` | > 0.1 | ✓ |
| **Conditioning** | `cond(g)` | < 100 | ? |
| **Determinant** | `max\|det(g)-1\|` | < 0.01 | ✓ |

**Actions**:
- [ ] Créer fonction `compute_validation_table(phi, g, mesh)`
- [ ] Ajouter au notebook: section "## 7. Final Validation Table"
- [ ] Sauvegarder CSV: `results/validation_v0_9.csv`

#### 5.2 Cartes de Résidus par Région

```python
def residual_heatmap(phi, g, mesh):
    """
    Plot spatial distribution of residuals:
    - Region M₁: t < -6
    - Neck:      |t| < 6
    - Region M₂: t > 6
    """
    d_phi = exterior_derivative_pform(phi, g, mesh)
    delta_phi = codifferential(phi, g, mesh)

    residuals = {
        'M1': d_phi.norm(dim=-1)[mesh.t < -6].mean(),
        'Neck': d_phi.norm(dim=-1)[mesh.t.abs() < 6].mean(),
        'M2': d_phi.norm(dim=-1)[mesh.t > 6].mean(),
    }

    # Plot heatmap (t vs θ₁)
    plot_heatmap(residuals, 't', 'theta1')
```

**Actions**:
- [ ] Implémenter `residual_heatmap()` pour `||dφ||`, `||δφ||`, `||Ric||`
- [ ] Générer 3 figures: M₁, Neck, M₂
- [ ] Vérifier: résidus < tol dans chaque région

---

### **Phase 6 : Documentation & Tests** 📝
**Priorité: HAUTE** | **Effort: 1 jour**

#### 6.1 Disclaimers Académiques
Ajouter en haut du notebook:

```markdown
## ⚠️ Geometric Approximations

1. **K3 Surface**: Using T⁴ as proxy (b₂=22 → 21). Rigorous K3 requires
   Kummer surface resolution (Joyce 1996, §7.2). Claims limited to
   "TCS-like structure."

2. **Asymptotic Cylindrical**: ACyl zones use C² matching with finite
   decay. True ACyl requires |∂ʳg| = O(r⁻ᵏ) for all k.

3. **Mesh Resolution**: 12⁷ ≈ 35M cells. Publish-quality requires
   convergence test with 16⁷ or 20⁷ mesh.

**Status**: Research prototype → Mathematical rigor: ⭐⭐⭐☆☆
```

#### 6.2 Tests Unitaires
**Fichier**: `tests/test_operators.py`

```python
def test_exterior_derivative_poincare():
    """Test: d(dω) = 0"""
    omega = random_2form()
    d_omega = exterior_derivative_pform(omega, g, mesh)
    dd_omega = exterior_derivative_pform(d_omega, g, mesh)
    assert dd_omega.norm() < 1e-8

def test_codifferential_nilpotent():
    """Test: δδ = 0"""
    omega = random_3form()
    delta_omega = codifferential(omega, g, mesh)
    delta_delta_omega = codifferential(delta_omega, g, mesh)
    assert delta_delta_omega.norm() < 1e-8

def test_laplacian_harmonic():
    """Test: Δφ = 0 for harmonic form"""
    phi = harmonic_3form()  # From cohomology
    laplacian_phi = laplacian_pform(phi, g, mesh)
    assert laplacian_phi.norm() < 1e-6
```

**Actions**:
- [ ] Créer `tests/test_operators.py` avec 10+ tests
- [ ] CI: Exécuter tests sur chaque commit
- [ ] Coverage: Viser >80% sur `operators/` et `geometry/`

---

## 🚀 Ordre d'Exécution Recommandé

```
┌─────────────────────────────────────────────┐
│ SEMAINE 1: Geometry + Operators             │
├─────────────────────────────────────────────┤
│ J1-2: Phase 1.1-1.2 (ACylCY3 + TCSGlue)     │
│ J3:   Phase 1.3 (K3 disclaimer)             │
│ J4-5: Phase 2 (Differential operators)      │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ SEMAINE 2: Curvature + Validation           │
├─────────────────────────────────────────────┤
│ J6:   Phase 3 (Ricci monitoring)            │
│ J7-8: Phase 4 (Cohomology)                  │
│ J9:   Phase 5 (Validation table)            │
│ J10:  Phase 6 (Docs + Tests)                │
└─────────────────────────────────────────────┘
```

**Milestone final**: Notebook v0.9 avec validation publiable.

---

## 📊 Success Criteria

✅ **Minimum Viable Product (MVP)**:
- [ ] Geometry: ACyl zones avec C² continuity (`||g₁-g₂|| < 1e-6`)
- [ ] Operators: `d`, `δ`, `Δ` implémentés + tests Poincaré passés
- [ ] Validation: `||dφ||_L² < 1e-6`, `||δφ||_L² < 1e-6`
- [ ] Cohomology: `b₂=21 ± 1`, `b₃=77 ± 2` (avec tolérance spectrale)
- [ ] Ricci: `||Ric||_L² < 1e-4` sur maillage complet
- [ ] Docs: Disclaimers + références académiques

✅ **Publication-Ready** (optionnel, +1 mois):
- [ ] K3 rigoureux (T⁴/Z₂ lissé)
- [ ] Convergence mesh: 12⁷ → 16⁷ → 20⁷
- [ ] Adaptive curvature solver (pas seulement loss)
- [ ] Benchmark contre Joyce's examples

---

## 🔗 Références

1. Joyce, D. (1996). *Compact Riemannian 7-manifolds with holonomy G₂*. I, II.
2. Kovalev, A. (2003). *Twisted connected sums and special Riemannian holonomy*.
3. Corti, Haskins, Nordström, Pacini (2015). *G₂-manifolds and associative submanifolds*.

---

**Next Steps**:
1. Review ce plan avec équipe
2. Fork notebook → `Complete_G2_Metric_Training_v0_9.ipynb`
3. Start avec Phase 1.1 (ACylCY3 scaffold)

**Questions?** Ping avant de démarrer Phase 1!
