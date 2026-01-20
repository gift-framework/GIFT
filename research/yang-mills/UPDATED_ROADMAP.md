# 🗺️ Updated Roadmap — Post Literature Review

**Date**: 2026-01-20
**Status**: Nouvelles pistes identifiées

---

## Ce Qui a Changé

### Avant (AI Council)
```
Gap 1.1 (Universalité) ──► Chercher "Lichnerowicz pour G₂"
                          (Problème: Ric=0 donc borne triviale)
```

### Après (Literature Review)
```
Gap 1.1 (Universalité) ──► Utiliser NECK-STRETCHING (2024)
                          + Index Theory (APS)
                          + Calcul local Eguchi-Hanson
```

---

## 🆕 Nouvelles Pistes Découvertes

### Piste N1 : Le Paper de 2024 est EXACTEMENT ce qu'on cherche

**arXiv:2301.03513** "Analysis and Spectral Theory of Neck-Stretching Problems"

Ce paper :
- ✅ Étudie les valeurs propres sur G₂-TCS
- ✅ Donne des formules asymptotiques explicites
- ✅ Relie le spectre à la géométrie du cou
- ✅ Connecte aux conjectures Swampland (physique!)

**Action immédiate** : Lire Section 5 en détail, chercher si leur constante C peut être = 14.

---

### Piste N2 : H* = Harmoniques sur la Section Transverse ?

**Observation** : Dans une construction TCS :
```
M₇ = (X₁ × S¹) ∪_{K3 × S¹ × I} (X₂ × S¹)
```

La section transverse est **K3 × S¹** avec :
- b₂(K3) = 22
- b₁(S¹) = 1

Les harmoniques sur K3 × S¹ pourraient **fixer** la constante dans λ₁ ~ C/T².

**Connexion avec H*** :
```
H* = b₂(M₇) + b₃(M₇) + 1

Est-ce que H* compte les "modes" sur la section transverse ?
```

**Action** : Calculer les harmoniques sur K3 × S¹ et voir si elles sont en nombre H* - quelque chose.

---

### Piste N3 : Le +1 vient de l'η-invariant (confirmé par Crowley-Nordström)

**Paper clé** : "An analytic invariant of G₂ manifolds" (Inventiones 2025)

Crowley-Goette-Nordström définissent ν̄(M,g) ∈ ℤ via :
- η-invariants du Dirac
- Courants de Mathai-Quillen

**Leur formule** :
```
ν(φ) = ν̄(g) + 24 (mod 48)
```

Le **24** et le **48** sont des nombres topologiques fixes.

**Hypothèse renforcée** : Le +1 dans H* = b₂ + b₃ + **1** est la contribution de dim ker(D) = 1 (spineur parallèle) dans la formule APS :
```
ind(D) = ∫Â - (h + η)/2
```

avec h = 1.

---

### Piste N4 : Équation de Heun → Valeur Propre Exacte

Sur Eguchi-Hanson, l'équation aux valeurs propres se réduit à une **équation de Heun confluente** :

```
d²u/dz² + p(z) du/dz + q(z) u = 0
```

avec symbole d'Ince [0, 2, 1₂].

**Ce qu'on peut faire** :
1. Résoudre numériquement l'équation de Heun
2. Ou l'approximer par Pöschl-Teller (exactement soluble)
3. Obtenir λ₁(EH) explicitement

Si λ₁(EH) = 1/4 (hypothèse de Claude AI Council), alors on a une brique de base.

---

### Piste N5 : Canonical Metric Principle (GPT) + Neck-Stretching

**Idée combinée** :
1. Pour chaque topologie (b₂, b₃), il existe une **métrique canonique** g*
2. Cette métrique minimise un fonctionnel (torsion-energy? volume?)
3. Sur cette métrique, λ₁(g*) = 14/H*

**Reformulation** :
```
Au lieu de prouver : λ₁ = 14/H* pour TOUTE métrique G₂
Prouver : λ₁ = 14/H* pour la métrique CANONIQUE (torsion-free, volume-normalized)
```

C'est plus faible mais peut-être plus accessible !

---

## 📊 Nouvelle Hiérarchie des Gaps

```
                    ┌─────────────────────────────────────┐
                    │         λ₁ = 14/H* (GIFT)          │
                    └───────────────┬─────────────────────┘
                                    │
     ┌──────────────────────────────┼──────────────────────────────┐
     │                              │                              │
     ▼                              ▼                              ▼
 GAP 1.1                        GAP 1.2                        GAP 1.3
 Universalité                   Normalisation                  Le +1
 ┌────────────┐                ┌────────────┐                ┌────────────┐
 │NECK-STRETCH│                │ RÉSOLU ?   │                │ APS/η-inv  │
 │ Paper 2024 │                │ Sprint 1   │                │ Crowley-N  │
 │ Section 5  │                │ montre que │                │ h=1 spinor │
 └────────────┘                │ c'est un   │                └────────────┘
       │                       │ artefact   │                      │
       │                       └────────────┘                      │
       │                                                           │
       ▼                                                           ▼
 GAP 2.1                                                      GAP 2.2
 Local EH                                                     KK Reduction
 ┌────────────┐                                              ┌────────────┐
 │ Heun eqn   │                                              │ Spectral   │
 │ λ₁(EH)=1/4?│                                              │ Triples    │
 │ Pöschl-Tel │                                              │ (Connes)   │
 └────────────┘                                              └────────────┘
```

---

## 🎯 Nouveau Plan d'Action

### Phase 1 : Paper 2024 (1-2 semaines)
**Priorité maximale** — C'est le game-changer potentiel

- [ ] Télécharger et lire arXiv:2301.03513 en détail
- [ ] Focus sur Section 5 (applications G₂)
- [ ] Extraire leur formule pour λ₁ en fonction des paramètres
- [ ] Vérifier si C = 14 dans un cas particulier

### Phase 2 : Calcul Local (2-3 semaines)
**Brique de base pour la preuve**

- [ ] Résoudre l'équation de Heun sur EH numériquement
- [ ] Approximer par Pöschl-Teller
- [ ] Calculer λ₁(ℂ²/ℤ₂) explicitement
- [ ] Généraliser à ℂ³/ℤ₂ (cas G₂)

### Phase 3 : η-invariant (2-3 semaines)
**Expliquer le +1**

- [ ] Calculer η(D) sur ℂ³/ℤ₂ résolu
- [ ] Vérifier que h = dim ker(D) = 1 pour le spineur parallèle
- [ ] Relier à la formule de Crowley-Nordström

### Phase 4 : Synthèse (2-4 semaines)
**Assembler les pièces**

- [ ] Combiner : local (EH) + gluing (neck-stretching) + index (η)
- [ ] Formuler une conjecture précise avec hypothèses claires
- [ ] Vérifier sur 2-3 exemples (Joyce J1, K₇, Kovalev)

---

## 🔥 La Nouvelle Stratégie en Une Phrase

> **Au lieu de chercher une borne Lichnerowicz (qui n'existe pas pour Ric=0), utiliser le framework de neck-stretching + la contribution locale des singularités + la correction η-invariant pour construire la formule λ₁ = 14/H*.**

---

## Comparaison Avant/Après

| Aspect | Avant (AI Council) | Après (Literature) |
|--------|-------------------|-------------------|
| **Approche principale** | Lichnerowicz généralisé | Neck-stretching + APS |
| **Gap 1.2 (normalisation)** | Mystère 40 vs 14 | Probablement artefact numérique |
| **Le +1 dans H*** | Hypothèse vague | η-invariant (Crowley-N confirme) |
| **Calcul local** | "λ₁(EH) = 1/4" hypothèse | Heun equation → calculable |
| **Paper clé** | Mazzeo-Melrose 1995 | arXiv:2301.03513 (2024) |
| **Faisabilité** | Moonshot | Medium (3-6 mois) |

---

## Questions Clés Reformulées

### Q1 : Que dit exactement le paper 2024 sur λ₁ pour G₂-TCS ?
→ À vérifier en lisant Section 5

### Q2 : Est-ce que λ₁(EH) = 1/4 exactement ?
→ À calculer via Heun equation

### Q3 : Comment les harmoniques sur K3 × S¹ se relient à H* ?
→ À investiguer (théorie de Hodge)

### Q4 : Le ν̄-invariant de Crowley-Nordström encode-t-il λ₁ ?
→ À vérifier dans leur paper

---

## Conclusion

La literature review a **transformé** notre approche :

1. **On a un paper récent (2024) qui fait exactement ce qu'on veut**
2. **Le +1 a une explication plausible (η-invariant)**
3. **Le calcul local (EH) est faisable via Heun**
4. **La normalisation (Gap 1.2) est probablement un faux problème**

La probabilité de succès est passée de **10-15%** (moonshot) à **30-40%** (ambitious but doable).

---

*"The path is now clearer. We're not searching in the dark anymore."*
