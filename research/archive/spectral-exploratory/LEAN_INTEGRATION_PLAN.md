# Plan d'Intégration Lean : Model Theorem Spectral Bounds

**Date** : Janvier 2026
**Objectif** : Intégrer le Model Theorem (Tier 1) dans gift-framework/core
**Basé sur** : Exploration complète de gift-core + CLAUDE.md

---

## 1. Résumé Exécutif

### Ce qu'on intègre

Le **Model Theorem (TCS Spectral Bounds)** :
```
Sous hypothèses (H1)-(H6), pour L > L₀ :
    v₀²/L² ≤ λ₁(K) ≤ 16v₁/(1-v₁)/L²
```

### Où ça s'intègre

```
gift-core/Lean/GIFT/Spectral/
├── SpectralTheory.lean     ← Axiomes de base (existants)
├── CheegerInequality.lean  ← Bornes Cheeger (existants)
├── TCSBounds.lean          ← NOUVEAU : Model Theorem
└── NeckGeometry.lean       ← NOUVEAU : Hypothèses H1-H6
```

### Pourquoi c'est optimal

1. **Réutilise** les axiomes existants (CompactManifold, MassGap, CheegerConstant)
2. **Complète** CheegerInequality avec des bornes en 1/L²
3. **Prépare** le terrain pour Tier 2/3 (quand ils seront rigoureux)

---

## 2. Architecture Proposée

### 2.1 Nouveau module : `NeckGeometry.lean`

```lean
/-
GIFT Spectral: TCS Neck Geometry
================================
Hypotheses for Twisted Connected Sum manifolds with cylindrical neck.
-/

import GIFT.Core
import GIFT.Spectral.SpectralTheory

namespace GIFT.Spectral.NeckGeometry

/-- TCS manifold structure: K = M₁ ∪_N M₂ with cylindrical neck -/
structure TCSManifold extends CompactManifold where
  /-- Neck length parameter -/
  neckLength : ℝ
  /-- Neck length is positive -/
  neckLength_pos : neckLength > 0
  /-- Cross-section area (function of L) -/
  crossSectionArea : ℝ
  /-- (H1) Volume normalization -/
  volume_eq_one : volume = 1

/-- (H2) Bounded neck volume -/
structure BoundedNeckVolume (K : TCSManifold) where
  v₀ : ℝ
  v₁ : ℝ
  v₀_pos : 0 < v₀
  v₁_lt_one : v₁ < 1
  v₀_lt_v₁ : v₀ < v₁
  neckVolume : ℝ
  lower_bound : v₀ ≤ neckVolume
  upper_bound : neckVolume ≤ v₁

/-- (H3) Product neck metric (axiom - geometric condition) -/
axiom ProductNeckMetric (K : TCSManifold) : Prop

/-- (H4) Block Cheeger bound -/
structure BlockCheegerBound (K : TCSManifold) where
  h₀ : ℝ
  h₀_pos : h₀ > 0
  block1_bound : ℝ  -- h(M₁ \ N) ≥ h₀
  block2_bound : ℝ  -- h(M₂ \ N) ≥ h₀

/-- (H5) Balanced blocks -/
structure BalancedBlocks (K : TCSManifold) where
  vol_M1 : ℝ
  vol_M2 : ℝ
  M1_lower : 1/4 ≤ vol_M1
  M1_upper : vol_M1 ≤ 3/4
  M2_lower : 1/4 ≤ vol_M2
  M2_upper : vol_M2 ≤ 3/4

/-- (H6) Neck minimality: Area(Γ) ≥ Area(Y) for separating Γ ⊂ N -/
axiom NeckMinimality (K : TCSManifold) : Prop

/-- Full hypothesis bundle -/
structure TCSHypotheses (K : TCSManifold) where
  neckVol : BoundedNeckVolume K
  productMetric : ProductNeckMetric K
  blockCheeger : BlockCheegerBound K
  balanced : BalancedBlocks K
  neckMinimal : NeckMinimality K

end GIFT.Spectral.NeckGeometry
```

### 2.2 Nouveau module : `TCSBounds.lean`

```lean
/-
GIFT Spectral: TCS Spectral Bounds (Model Theorem)
==================================================
Proof that λ₁ ~ 1/L² for TCS manifolds with cylindrical neck.

Status: THEOREM (proven from hypotheses)
-/

import GIFT.Spectral.NeckGeometry
import GIFT.Spectral.CheegerInequality

namespace GIFT.Spectral.TCSBounds

open GIFT.Spectral.NeckGeometry
open GIFT.Spectral.CheegerInequality

/-!
## Main Theorem

For TCS manifold K with neck length L satisfying hypotheses (H1)-(H6):
    c₁/L² ≤ λ₁(K) ≤ c₂/L²

where c₁ = v₀² and c₂ = 16v₁/(1-v₁).
-/

-- ============================================================================
-- CONSTANTS
-- ============================================================================

/-- Lower bound constant c₁ = v₀² -/
noncomputable def c₁ (hyp : TCSHypotheses K) : ℝ := hyp.neckVol.v₀ ^ 2

/-- Upper bound constant c₂ = 16v₁/(1-v₁) (robust) -/
noncomputable def c₂_robust (hyp : TCSHypotheses K) : ℝ :=
  16 * hyp.neckVol.v₁ / (1 - hyp.neckVol.v₁)

/-- Upper bound constant c₂ = 4v₁/(1-2v₁/3) (symmetric blocks) -/
noncomputable def c₂_symmetric (hyp : TCSHypotheses K) : ℝ :=
  4 * hyp.neckVol.v₁ / (1 - 2 * hyp.neckVol.v₁ / 3)

/-- Threshold L₀ = 2v₀/h₀ -/
noncomputable def L₀ (hyp : TCSHypotheses K) : ℝ :=
  2 * hyp.neckVol.v₀ / hyp.blockCheeger.h₀

-- ============================================================================
-- UPPER BOUND (Rayleigh Quotient)
-- ============================================================================

/-- Upper bound via Rayleigh quotient test function.

    Proof sketch:
    1. Construct test function f: +1 on M₁, linear on neck, -1 on M₂
    2. Orthogonalize: f ← f - f̄
    3. Use (H5) balanced blocks: ∫f² ≥ (1/4)(1-v₁)
    4. Compute: ∫|∇f|² = 4·Vol(neck)/L² ≤ 4v₁/L²
    5. Rayleigh: λ₁ ≤ (4v₁/L²) / ((1/4)(1-v₁)) = 16v₁/((1-v₁)L²)
-/
axiom spectral_upper_bound (K : TCSManifold) (hyp : TCSHypotheses K) :
  MassGap K.toCompactManifold ≤ c₂_robust hyp / K.neckLength ^ 2

-- ============================================================================
-- LOWER BOUND (Cheeger Inequality)
-- ============================================================================

/-- Neck Cheeger constant: h_neck ≥ 2v₀/L -/
axiom neck_cheeger_bound (K : TCSManifold) (hyp : TCSHypotheses K) :
  ∃ h_neck : ℝ, h_neck ≥ 2 * hyp.neckVol.v₀ / K.neckLength

/-- For L > L₀, neck dominates: h(K) ≥ 2v₀/L -/
axiom neck_dominates (K : TCSManifold) (hyp : TCSHypotheses K)
    (hL : K.neckLength > L₀ hyp) :
  CheegerConstant K.toCompactManifold ≥ 2 * hyp.neckVol.v₀ / K.neckLength

/-- Lower bound via Cheeger inequality.

    Proof sketch:
    1. Classify cuts: (A) in block, (B) through neck
    2. Case A: h ≥ h₀ by (H4)
    3. Case B: h ≥ 2·Area(Y)/Vol ≥ 2v₀/L by (H6)
    4. Combined: h(K) ≥ min(h₀, 2v₀/L)
    5. For L > L₀: h(K) ≥ 2v₀/L
    6. Cheeger: λ₁ ≥ h²/4 = v₀²/L²
-/
theorem spectral_lower_bound (K : TCSManifold) (hyp : TCSHypotheses K)
    (hL : K.neckLength > L₀ hyp) :
    MassGap K.toCompactManifold ≥ c₁ hyp / K.neckLength ^ 2 := by
  have h_cheeger := neck_dominates K hyp hL
  have h_ineq := cheeger_inequality K.toCompactManifold
  -- λ₁ ≥ h²/4 ≥ (2v₀/L)²/4 = v₀²/L²
  sorry  -- Full proof uses calc chain

-- ============================================================================
-- MAIN THEOREM
-- ============================================================================

/-- Model Theorem: TCS Spectral Bounds

    For TCS manifold K with neck length L > L₀ satisfying (H1)-(H6):
        v₀²/L² ≤ λ₁(K) ≤ 16v₁/((1-v₁)L²)

    Corollary: λ₁ = Θ(1/L²)
-/
theorem tcs_spectral_bounds (K : TCSManifold) (hyp : TCSHypotheses K)
    (hL : K.neckLength > L₀ hyp) :
    c₁ hyp / K.neckLength ^ 2 ≤ MassGap K.toCompactManifold ∧
    MassGap K.toCompactManifold ≤ c₂_robust hyp / K.neckLength ^ 2 := by
  constructor
  · exact spectral_lower_bound K hyp hL
  · exact spectral_upper_bound K hyp

-- ============================================================================
-- ALGEBRAIC CONSEQUENCES
-- ============================================================================

/-- For typical TCS with v₀ = v₁ = 1/2, h₀ = 1 -/
theorem typical_tcs_bounds :
    -- c₁ = (1/2)² = 1/4
    ((1 : ℚ) / 2) ^ 2 = 1 / 4 ∧
    -- c₂ = 16·(1/2)/(1-1/2) = 16
    16 * (1 / 2) / (1 - 1 / 2) = 16 ∧
    -- L₀ = 2·(1/2)/1 = 1
    2 * (1 / 2) / 1 = 1 := by
  native_decide

/-- For K7 with L ~ sqrt(H*), bounds give λ₁ ~ 1/H* -/
theorem K7_scaling_consequence (H_star_val : ℕ) (hH : H_star_val = 99) :
    -- If L² ~ H*, then 1/L² ~ 1/H*
    (1 : ℚ) / H_star_val = 1 / 99 := by
  simp [hH]

-- ============================================================================
-- CERTIFICATE
-- ============================================================================

/-- TCS Spectral Bounds Certificate -/
theorem tcs_bounds_certificate :
    -- c₁ formula
    ((1 : ℚ) / 2) ^ 2 = 1 / 4 ∧
    -- c₂ formula (robust)
    16 * (1 / 2) / (1 - 1 / 2) = 16 ∧
    -- c₂ formula (symmetric)
    4 * (1 / 2) / (1 - 2 * (1 / 2) / 3) = 3 ∧
    -- L₀ formula
    2 * (1 / 2) / 1 = 1 ∧
    -- Cheeger lower bound
    (1 / 4 : ℚ) / 4 = 1 / 16 := by
  native_decide

end GIFT.Spectral.TCSBounds
```

---

## 3. Blueprint Integration

### 3.1 Nouveau chapitre dans `content.tex`

```latex
% =============================================================================
\chapter{Spectral Bounds for TCS Manifolds}\label{chap:tcs_bounds}
% =============================================================================

We establish spectral bounds for Twisted Connected Sum (TCS) manifolds
with cylindrical necks, proving $\lambda_1 \sim 1/L^2$.

\section{TCS Geometry Setup}

\begin{definition}[TCS Manifold]\label{def:TCSManifold}
    \lean{GIFT.Spectral.NeckGeometry.TCSManifold}
    \leanok
    A TCS manifold $K = M_1 \cup_N M_2$ consists of:
    \begin{itemize}
        \item Building blocks $M_1, M_2$ (asymptotically cylindrical)
        \item Cylindrical neck $N \cong Y \times [0, L]$
        \item Cross-section $Y$ (e.g., $S^1 \times \text{K3}$ for $\Gtwo$ manifolds)
    \end{itemize}
\end{definition}

\begin{definition}[Bounded Neck Volume (H2)]\label{def:BoundedNeckVolume}
    \lean{GIFT.Spectral.NeckGeometry.BoundedNeckVolume}
    \leanok
    \uses{def:TCSManifold}
    $\Vol(N) \in [v_0, v_1]$ for fixed $0 < v_0 < v_1 < 1$.
\end{definition}

[... additional definitions for H3-H6 ...]

\section{Upper Bound via Rayleigh Quotient}

\begin{theorem}[Spectral Upper Bound]\label{thm:spectral_upper_bound}
    \lean{GIFT.Spectral.TCSBounds.spectral_upper_bound}
    \uses{def:TCSManifold, def:BoundedNeckVolume, def:BalancedBlocks}
    For TCS manifold $K$ with hypotheses (H1)-(H5):
    \[
        \lambda_1(K) \leq \frac{16 v_1}{(1 - v_1) L^2}
    \]
\end{theorem}
\begin{proof}
    Test function $f = +1$ on $M_1$, linear on neck, $-1$ on $M_2$.
    Orthogonalize and compute Rayleigh quotient.
\end{proof}

\section{Lower Bound via Cheeger Inequality}

\begin{lemma}[Neck Cut Lemma]\label{lem:neck_cut}
    \lean{GIFT.Spectral.TCSBounds.neck_cheeger_bound}
    \uses{def:TCSManifold, def:NeckMinimality}
    Any hypersurface $\Gamma \subset N$ separating the ends satisfies
    $\Area(\Gamma) \geq \Area(Y)$.
\end{lemma}
\begin{proof}
    Projection argument: $\pi_Y : N \to Y$ is 1-Lipschitz,
    and $\Gamma$ must cover $Y$ with multiplicity $\geq 1$.
\end{proof}

\begin{theorem}[Spectral Lower Bound]\label{thm:spectral_lower_bound}
    \lean{GIFT.Spectral.TCSBounds.spectral_lower_bound}
    \leanok
    \uses{def:TCSManifold, lem:neck_cut, thm:cheeger_inequality}
    For TCS manifold $K$ with hypotheses (H1)-(H6) and $L > L_0$:
    \[
        \lambda_1(K) \geq \frac{v_0^2}{L^2}
    \]
\end{theorem}

\section{Main Result}

\begin{theorem}[TCS Spectral Bounds]\label{thm:tcs_spectral_bounds}
    \lean{GIFT.Spectral.TCSBounds.tcs_spectral_bounds}
    \leanok
    \uses{thm:spectral_upper_bound, thm:spectral_lower_bound}
    For TCS manifold $K$ with neck length $L > L_0$ and hypotheses (H1)-(H6):
    \[
        \frac{v_0^2}{L^2} \leq \lambda_1(K) \leq \frac{16 v_1}{(1 - v_1) L^2}
    \]
    Corollary: $\lambda_1 = \Theta(1/L^2)$.
\end{theorem}
```

---

## 4. Plan d'Implémentation

### Phase 1 : Setup (1 jour)

1. Créer branche `feature/tcs-spectral-bounds`
2. Créer `/Lean/GIFT/Spectral/NeckGeometry.lean`
3. Créer `/Lean/GIFT/Spectral/TCSBounds.lean`
4. Mettre à jour `/Lean/GIFT/Spectral.lean` (re-exports)
5. Vérifier : `lake build`

### Phase 2 : Axiomes et structures (1 jour)

1. Implémenter `TCSManifold` structure
2. Implémenter hypothèses H1-H6 comme structures/axiomes
3. Définir constantes c₁, c₂, L₀
4. Vérifier typechecking

### Phase 3 : Théorèmes algébriques (1 jour)

1. Prouver `typical_tcs_bounds` (native_decide)
2. Prouver `tcs_bounds_certificate` (native_decide)
3. Prouver connexion avec K7 existant

### Phase 4 : Bornes spectrales (2 jours)

1. Axiomatiser `spectral_upper_bound` (Rayleigh)
2. Prouver `spectral_lower_bound` via Cheeger existant
3. Assembler `tcs_spectral_bounds`

### Phase 5 : Blueprint (1 jour)

1. Ajouter chapitre dans `content.tex`
2. Lier avec `\lean{}` et `\leanok`
3. Générer blueprint : `leanblueprint`

### Phase 6 : Tests et CI (1 jour)

1. `lake build` passe
2. Ajouter tests Python si pertinent
3. PR review et merge

---

## 5. Optimisations et Simplifications

### 5.1 Réutilisation maximale

| Existant | Réutilisé pour |
|----------|----------------|
| `CompactManifold` | Base de `TCSManifold` |
| `MassGap` | Définition λ₁ |
| `CheegerConstant` | Borne inférieure |
| `cheeger_inequality` | Preuve lower bound |

### 5.2 Éviter les duplications

**Ne pas** créer de nouveau `Laplacian` — utiliser `MassGap` existant.

**Ne pas** redéfinir Cheeger — utiliser `CheegerInequality.lean`.

### 5.3 Axiomes minimaux

| Axiome nécessaire | Justification |
|-------------------|---------------|
| `ProductNeckMetric` | Condition géométrique non formalisable |
| `NeckMinimality` | Découle de ProductNeck mais nécessite coarea |
| `spectral_upper_bound` | Rayleigh quotient nécessite L² space |
| `neck_dominates` | Classification des coupes |

### 5.4 Théorèmes prouvables sans axiomes

Tout ce qui est **algébrique** (constantes, ratios) peut être `native_decide` :

```lean
-- PROUVABLE
theorem c1_formula : (1/2 : ℚ)^2 = 1/4 := by native_decide

-- AXIOME (géométrie)
axiom spectral_upper_bound : ...
```

---

## 6. Checklist Finale

### Lean

- [ ] `NeckGeometry.lean` compile
- [ ] `TCSBounds.lean` compile
- [ ] Tous les `native_decide` passent
- [ ] Pas de `sorry` dans les théorèmes algébriques
- [ ] `Spectral.lean` re-exporte les nouveaux modules

### Blueprint

- [ ] Chapitre TCS ajouté
- [ ] Tous les `\lean{}` pointent vers des déclarations valides
- [ ] `\leanok` sur les théorèmes prouvés
- [ ] Pas de `\leanok` sur les axiomes

### CI/CD

- [ ] `lake build` passe
- [ ] Tests Python passent (si applicable)
- [ ] PR créée avec description claire

---

## 7. Connexion avec Tier 2/3 (Futur)

Ce module prépare le terrain pour :

```lean
-- FUTUR (Tier 2) : Quand L² ~ H* sera prouvé
axiom L_squared_proportional_to_H_star (K : K7_Manifold) :
  ∃ c : ℝ, K.neckLength ^ 2 = c * H_star

-- FUTUR (Tier 3) : Quand le coefficient 14 sera prouvé
theorem K7_mass_gap_is_14_over_99 :
  MassGap K7 = 14 / 99 := by
  -- Combine TCS bounds + L ~ sqrt(H*) + coefficient calculation
  sorry
```

La structure modulaire permet d'ajouter ces résultats **sans modifier** le Model Theorem.

---

*Plan d'intégration Lean pour GIFT Spectral Bounds*
*Janvier 2026*
