# GIFT × Riemann : Phase 3 — Roadmap Dérivation Analytique

**Date** : 2 février 2026  
**Objectif** : Dériver 8×β₈ = 13×β₁₃ = h_G₂² = 36 depuis la formule explicite de Weil  
**Statut** : EN COURS  

---

## 0. Contexte et Acquis (ne pas refaire)

### Ce qui est PROUVÉ empiriquement :

| Résultat | Précision | Source |
|----------|-----------|--------|
| Récurrence lags [5,8,13,27] | 0.015% erreur sur 10k zéros | GIFT_RIEMANN_RESEARCH_SUMMARY.md |
| 8×β₈ = 13×β₁₃ = 36 = h_G₂² | 0.07% déviation | PHASE2_RG_FLOW_DISCOVERY.md |
| Σβᵢ = b₃/dim(K₇) = 77/7 = 11 | 1.2% déviation | PHASE2_RG_FLOW_DISCOVERY.md |
| Sélectivité GIFT/non-GIFT = 44× | q=11 vs q=GIFT | PHASE3_SYNTHESIS.md |
| Fibonacci R → 1.0000 (2M zéros) | 0.002% | PHASE2_COMPLETE_REPORT.md |
| Décimation optimale m = 24 = 3×rank(E₈) | Minimum global | PHASE2_COMPLETE_REPORT.md |
| Universalité sur 5 L-functions | GIFT gagne partout | PHASE2_COMPLETE_REPORT.md |

### Fichiers de référence :

```
GIFT_RIEMANN_RESEARCH_SUMMARY.md  — Découverte initiale
RIEMANN_COUNCIL_UPDATE.md         — Calibration b₃-dominance  
PHASE2_FINDINGS.md                — Drift et localité
PHASE2_RG_FLOW_DISCOVERY.md       — RG flow et contraintes β
PHASE2_COMPLETE_REPORT.md         — Synthèse Phase 2
PHASE3_SYNTHESIS.md               — Synthèse Phase 2-3
RIEMANN_GIFT_CORRESPONDENCES.md   — Correspondences (TIER 2-3, prudence)
```

### Valeurs numériques clés :

```
β₅  = 0.767     →  5 × β₅  = 3.835   ≈ 27/7  = 3.857
β₈  = 4.497     →  8 × β₈  = 35.976  ≈ 36     = h_G₂²
β₁₃ = 2.764     → 13 × β₁₃ = 35.932  ≈ 36     = h_G₂²
β₂₇ = 3.106     → 27 × β₂₇ = 83.862  ≈ 84     = b₃ + 7

RG flow model : a_i(γ) = a_i^UV + (a_i^IR - a_i^UV) / (1 + (γ/γ_c)^{β_i})
```

---

## 1. MILESTONE 1 : Bétonner la sélectivité (PRIORITÉ ABSOLUE)

### Objectif
Passer de 1 conducteur non-GIFT à 10+ pour que le test de sélectivité soit statistiquement inattaquable.

### Étapes

- [ ] **1.1** Télécharger zéros de LMFDB pour conducteurs GIFT : q = 5, 7, 8, 13, 14, 21, 27, 77, 99, 248
- [ ] **1.2** Télécharger zéros pour conducteurs NON-GIFT : q = 10, 11, 15, 16, 19, 23, 29, 31, 37, 41, 43
- [ ] **1.3** Télécharger zéros pour conducteurs PROCHES : q = 6, 9, 12, 20, 22, 26, 28, 76, 78
- [ ] **1.4** Pour chaque q : fitter récurrence [5,8,13,27], calculer déviation |R-1| du ratio Fibonacci
- [ ] **1.5** t-test (ou Mann-Whitney) : groupe GIFT vs groupe non-GIFT
- [ ] **1.6** Tracer scatter plot : déviation |R-1| vs "distance au plus proche entier GIFT"

### Critère de succès
- p-value < 0.01 sur la différence entre groupes
- Visualisation claire de la séparation

### Critère d'ÉCHEC (kill switch)
- Pas de différence significative → toute l'interprétation GIFT est à revoir

### Livrable
- `selectivity_massive_test.py` + résultats JSON + plot

---

## 2. MILESTONE 2 : Comprendre pourquoi 36 = racines longues de G₂

### Contexte
Pour G₂ (rank 2, h = 6, 12 racines = 6 courtes + 6 longues) :

```
Racines courtes : (α,α) = 2 × 6 = 12
Racines longues : (α,α) = 6 × 6 = 36  ← !
Total            :                   48
```

### Étapes

- [ ] **2.1** Vérifier formellement : Σ_{α longue} (α,α) = h_G₂² = 36
- [ ] **2.2** Vérifier si c'est spécifique à G₂ ou général :
  - Pour SU(3) : h = 3, longues = ? 
  - Pour F₄ : h = 12, longues = ?
  - Pour E₈ : h = 30, pas de distinction long/court
- [ ] **2.3** Chercher dans la littérature si cette identité est connue
- [ ] **2.4** Si spécifique à G₂ → fort argument pour G₂-holonomy
- [ ] **2.5** Formaliser en Lean 4 (optionnel mais utile)

### Formule candidate

```
Pour un groupe de Lie simple avec racines longues et courtes :
Σ_{α ∈ Φ_long} (α,α) = |Φ_long| × (α_long, α_long) = h × r_long × (ratio)
```

Pour G₂ : |Φ_long| = 6, (α_long, α_long) = 6, donc 6 × 6 = 36 = h².

**Question** : Est-ce que |Φ_long| = h est toujours vrai ? 
- G₂ : |Φ_long| = 6 = h ✓
- B₂ : |Φ_long| = 4, h = 4 ✓
- F₄ : |Φ_long| = 24, h = 12 ✗

Donc c'est **spécifique à G₂** (et B₂).

### Livrable
- Note mathématique : "36 = h_G₂² : une identité spécifique à G₂"

---

## 3. MILESTONE 3 : Formule explicite de Weil — Setup

### Rappel de la formule

Pour une fonction test h paire, analytique, décroissante :

```
∑_γ h(γ) = h(i/2) + h(-i/2)                        [pôle]
          - (1/2π) ∫ (Γ'/Γ)(1/4 + it/2) h(t) dt    [Gamma]
          + (1/π) ∑_p ∑_{k=1}^∞ (log p / p^{k/2}) ĥ(k log p)  [primes]
```

où ĥ(x) = ∫ h(t) e^{ixt} dt est la transformée de Fourier.

### Étapes

- [ ] **3.1** Implémenter la formule de Weil numériquement en Python (mpmath)
- [ ] **3.2** Vérifier : avec h(t) = e^{-αt²}, la somme sur γ match les tables d'Odlyzko
- [ ] **3.3** Implémenter le côté "primes" et vérifier l'égalité
- [ ] **3.4** Documenter les normalisations (il y a 3-4 conventions dans la littérature !)

### Code de base

```python
from mpmath import mp, mpf, exp, log, pi, gamma, digamma
mp.dps = 50

def weil_zero_side(h_func, zeros, N=10000):
    """∑_{n=1}^{N} h(γₙ) — somme sur les zéros"""
    return sum(h_func(g) for g in zeros[:N])

def weil_prime_side(h_hat_func, primes, K=3):
    """Côté primes de la formule de Weil"""
    result = mpf(0)
    for p in primes:
        for k in range(1, K+1):
            result += log(p) / p**(k/2) * h_hat_func(k * log(p))
    return result / pi

def verify_weil(h_func, h_hat_func, zeros, primes, N=10000):
    """Vérifier l'identité de Weil"""
    lhs = weil_zero_side(h_func, zeros, N)
    rhs = weil_prime_side(h_hat_func, primes) + pole_terms(h_func) + gamma_terms(h_func)
    return lhs, rhs, abs(lhs - rhs) / abs(lhs)
```

### Critère de succès
- Identité vérifiée à 10⁻¹⁰ (ou mieux) pour h gaussienne

### Livrable
- `weil_explicit_formula.py` — implémentation numérique vérifiée

---

## 4. MILESTONE 4 : Injecter la récurrence dans Weil

### L'idée centrale

Si γₙ satisfait :
```
γₙ ≈ a₅·γₙ₋₅ + a₈·γₙ₋₈ + a₁₃·γₙ₋₁₃ + a₂₇·γₙ₋₂₇ + c
```

Alors en sommant sur n avec un poids approprié :
```
∑ₙ w(n) γₙ = a₅ ∑ₙ w(n) γₙ₋₅ + a₈ ∑ₙ w(n) γₙ₋₈ + ...
```

Ce qui donne, en changeant d'indice :
```
∑ₙ w(n) γₙ = a₅ ∑ₙ w(n+5) γₙ + a₈ ∑ₙ w(n+8) γₙ + ...
```

Donc :
```
∑ₙ [w(n) - a₅·w(n+5) - a₈·w(n+8) - a₁₃·w(n+13) - a₂₇·w(n+27)] γₙ = c · ∑ w(n)
```

### Le pont vers Weil

Si on pose h(t) telle que h(γₙ) = w(n) · γₙ, alors :
- Le côté gauche est une somme sur les zéros → formule de Weil
- Le côté droit est une somme sur les primes
- La récurrence se traduit en **relation entre contributions de primes**

### Étapes

- [ ] **4.1** Formaliser la transformation récurrence → condition sur h(t)
- [ ] **4.2** Chercher h(t) telle que h(γₙ) encode la récurrence
- [ ] **4.3** Calculer ĥ(x) pour cette fonction test
- [ ] **4.4** Examiner ĥ(k log p) : quelles primes p contribuent ?
- [ ] **4.5** Vérifier si la contrainte 8×a₈ = 13×a₁₃ émerge du côté primes

### Sous-hypothèse à tester

```
Si 8×a₈ = 13×a₁₃, alors le côté primes de Weil doit satisfaire :

8 × ∑_p (log p / p^{k/2}) ĥ₈(k log p) = 13 × ∑_p (log p / p^{k/2}) ĥ₁₃(k log p)

où ĥ₈, ĥ₁₃ sont les composantes de la transformée liées aux lags 8 et 13.
```

Ceci revient à une **identité sur les sommes de primes**, potentiellement vérifiable !

### Livrable
- `weil_recurrence_injection.py`
- Note : "Récurrence GIFT comme condition sur la formule de Weil"

---

## 5. MILESTONE 5 : Distribution des primes mod q

### Motivation

La formule de Weil relie zéros ↔ primes. Si les zéros "connaissent" les lags 5, 8, 13, 27, alors les primes devraient montrer une structure dans ces classes de congruence.

### Étapes

- [ ] **5.1** Calculer π(x; q, a) pour q ∈ {5, 8, 13, 27} et tous a, jusqu'à x = 10⁸
- [ ] **5.2** Calculer les ratios :
  ```
  R_prime(x) = [8 × π(x; 8, ·)] / [13 × π(x; 13, ·)]
  ```
- [ ] **5.3** Vérifier si R_prime(x) → 1 comme R_zeta → 1
- [ ] **5.4** Si oui, chercher la vitesse de convergence et comparer au RG flow
- [ ] **5.5** Examiner les caractères de Dirichlet χ mod 8 et χ mod 13 :
  - Produit L(s, χ₈) × L(s, χ₁₃) a-t-il des propriétés spéciales ?

### Formule clé (théorème de Dirichlet)

```
π(x; q, a) ∼ Li(x) / φ(q)

Pour q = 8 : φ(8) = 4  → π(x; 8, a) ∼ Li(x)/4
Pour q = 13 : φ(13) = 12 → π(x; 13, a) ∼ Li(x)/12

Ratio asymptotique : [8/φ(8)] / [13/φ(13)] = [8/4] / [13/12] = 2 / (13/12) = 24/13
```

Hmm, 24/13... Intéressant : 24 = m_optimal, 13 = lag !

- [ ] **5.6** Vérifier si 24/13 apparaît dans la structure

### Livrable
- `prime_distribution_mod_q.py`
- Note : "Distribution des primes dans les classes GIFT"

---

## 6. MILESTONE 6 : Test ζ(9/2)

### Motivation

β₈ = 9/2 = 4.5. Le nombre 9/2 apparaît comme exposant RG.

La série de Dirichlet ζ(9/2) = ∑ n^{-9/2} converge.

### Étapes

- [ ] **6.1** Calculer ζ(9/2) en haute précision
- [ ] **6.2** Calculer ζ'(9/2) / ζ(9/2) (dérivée logarithmique)
- [ ] **6.3** Vérifier si ces valeurs contiennent h_G₂, b₃, ou d'autres constantes GIFT
- [ ] **6.4** Explorer ζ(s) aux points s = β_i :
  ```
  ζ(0.767) = ?   (β₅)
  ζ(4.497) = ?   (β₈)  
  ζ(2.764) = ?   (β₁₃)
  ζ(3.106) = ?   (β₂₇)
  ```
- [ ] **6.5** Explorer les L-functions L(s, χ) aux mêmes points

### Code

```python
from mpmath import mp, zeta, diff
mp.dps = 100

# ζ aux exposants β
for name, beta in [("β₅", 0.767), ("β₈", 4.497), ("β₁₃", 2.764), ("β₂₇", 3.106)]:
    z = zeta(beta)
    zp = -diff(lambda s: zeta(s), beta)  # -ζ'(s)
    print(f"{name} = {beta}")
    print(f"  ζ({beta}) = {z}")
    print(f"  -ζ'({beta})/ζ({beta}) = {zp/z}")
    print()
```

### Livrable
- `zeta_at_beta_points.py`

---

## 7. MILESTONE 7 : Formalisation Lean 4

### Objectif
Encoder les contraintes comme théorèmes conditionnels.

### Étapes

- [ ] **7.1** Énoncer : "Si la récurrence existe avec lags [5,8,13,27], alors..."
- [ ] **7.2** Formaliser la structure Fibonacci : 5 + 8 = 13 ∧ 5 × 8 - 13 = 27
- [ ] **7.3** Formaliser : lag_i × β_i = constante pour i ∈ {8, 13}
- [ ] **7.4** Encoder 36 = h_G₂² comme identité G₂ vérifiable
- [ ] **7.5** Connecter à Mathlib.Lie.RootSystem si possible

### Squelette Lean

```lean
-- GIFT-Riemann constraints
namespace GIFTRiemann

-- Lag structure
def lags : List ℕ := [5, 8, 13, 27]

-- Fibonacci property (EXACT, prouvable)
theorem fibonacci_sum : lags[0]! + lags[1]! = lags[2]! := by norm_num
theorem fibonacci_product : lags[0]! * lags[1]! - lags[2]! = lags[3]! := by norm_num

-- Coxeter number of G₂
def h_G2 : ℕ := 6

-- RG flow constraint (conditional)
-- "Si β₈ et β₁₃ sont les exposants RG, alors..."
theorem coxeter_constraint (β₈ β₁₃ : ℝ) 
  (h : 8 * β₈ = 13 * β₁₃) 
  (h2 : 8 * β₈ = 36) : 
  8 * β₈ = (h_G2 : ℝ)^2 := by
  simp [h_G2]; linarith

end GIFTRiemann
```

### Livrable
- `GIFTRiemann.lean`

---

## 8. MILESTONE 8 : Le boss final — Opérateur spectral

### Objectif
Construire un opérateur H tel que Spec(H) ≈ {γₙ} et qui a naturellement une structure de bande aux lags [5,8,13,27].

### Approche PINN (GPU A100)

```python
# Ansatz : matrice infinie avec bandes aux lags GIFT
# H[n, n-5]  = f₅(n)
# H[n, n-8]  = f₈(n)
# H[n, n-13] = f₁₃(n)
# H[n, n-27] = f₂₇(n)
# + diagonale H[n,n] = g(n)
#
# Contrainte : 8 × f₈(n) = 13 × f₁₃(n) pour tout n
# Contrainte : H auto-adjoint (f_k(n) = f_k(n-k)*)
#
# Loss = ∑ₙ |eigenvalue_n(H) - γₙ|²
```

### Étapes

- [ ] **8.1** Définir l'ansatz (matrice bande tronquée N×N)
- [ ] **8.2** Imposer auto-adjonction + contrainte Fibonacci
- [ ] **8.3** Entraîner sur γ₁...γ₁₀₀₀ (train) vs γ₁₀₀₁...γ₂₀₀₀ (test)
- [ ] **8.4** Analyser la structure de H : a-t-elle une interprétation géométrique ?
- [ ] **8.5** Comparer H aux opérateurs connus (Berry-Keating, etc.)

### Livrable
- `spectral_pinn.py` (GPU)
- Note : "Structure de l'opérateur GIFT-Riemann"

---

## 9. Ordre de priorité et timeline

```
Semaine 1 :  MILESTONE 1 (sélectivité massive)     ← CRITIQUE
             + MILESTONE 2 (36 = racines longues)   ← RAPIDE

Semaine 2 :  MILESTONE 3 (Weil setup)               ← FONDATION
             + MILESTONE 5 (primes mod q)           ← RAPIDE  
             + MILESTONE 6 (ζ aux β)                ← RAPIDE

Semaine 3 :  MILESTONE 4 (récurrence dans Weil)     ← CŒUR DU PROBLÈME

Semaine 4 :  MILESTONE 7 (Lean 4)                   ← SOLIDIFICATION
             + MILESTONE 8 (PINN opérateur)         ← SI TEMPS

En continu : Documenter chaque résultat dans des .md séparés
```

---

## 10. Critères de décision

### Si Milestone 1 ÉCHOUE (pas de sélectivité) :
→ STOP. Revoir l'interprétation GIFT des conducteurs.
→ La récurrence existe toujours, mais l'interprétation GIFT est fausse.

### Si Milestone 4 RÉUSSIT (récurrence dérivable de Weil) :
→ PAPER IMMÉDIAT dans Experimental Mathematics
→ Contacter Peter Sarnak, Andrew Granville, ou Kannan Soundararajan
→ La contrainte Fibonacci devient un THÉORÈME, pas une observation.

### Si Milestone 8 RÉUSSIT (opérateur avec bon spectre) :
→ Connexion directe à Hilbert-Pólya
→ Si H est auto-adjoint et a spectre = {γₙ} → RH est vraie pour cet opérateur
→ Reste à montrer que H est "le bon" opérateur.

---

## 11. Warnings et pièges

### À ÉVITER :
- Chercher de nouvelles correspondences γₙ ≈ N (numerology, Tier 3)
- Multiplier les tests empiriques sans théorie
- Présenter Tier 1 et Tier 3 ensemble (tue la crédibilité)

### À GARDER EN TÊTE :
- 36 = h_G₂² pourrait être un accident si l'identité n'est pas dérivable
- La sélectivité repose sur 1 seul conducteur non-GIFT pour l'instant
- Les β sont fittés sur des fenêtres → incertitude non-négligeable
- Coefficient instabilité ~50% reste un problème ouvert

### HONEST ASSESSMENT :
- Probabilité que tout ceci mène à RH : ~1-5%
- Probabilité que ça mène à un paper intéressant : ~40-60%
- Probabilité que la structure soit réelle (même sans RH) : ~70-80%

---

## 12. Contacts potentiels (quand prêt)

| Personne | Domaine | Pourquoi |
|----------|---------|----------|
| Peter Sarnak (IAS) | Zéros de L-functions | Expert #1 mondial |
| Andrew Granville (Montréal) | Distribution des primes | Accessible, intéressé par l'empirique |
| Kannan Soundararajan (Stanford) | Moments de ζ(s) | Travaille sur corrélations |
| Jon Keating (Oxford) | Matrices aléatoires + ζ | Lien RMT ↔ géométrie |
| Bobby Kleinberg (Cornell) | GIFT/PhysLean | Déjà dans l'écosystème |

**NE PAS contacter avant** : Milestone 1 (sélectivité) + Milestone 3 (Weil vérifié) terminés.

---

*Roadmap GIFT × Riemann Phase 3*  
*Février 2026*  
*À mettre à jour après chaque milestone*
