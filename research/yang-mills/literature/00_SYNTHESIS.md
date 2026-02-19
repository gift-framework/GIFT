# Synthèse de la Littérature — Yang-Mills via G₂

**Objectif**: Prouver λ₁ = 14/H* pour le Laplacien sur les variétés G₂.

---

## Vue d'Ensemble des Pistes

```
                    ┌─────────────────────────────────────┐
                    │         λ₁ = 14/H* (GIFT)          │
                    └───────────────┬─────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
   PISTE A                     PISTE B                     PISTE C
   Index Theory                Analytic Surgery            Hodge Theory
   (APS, η-inv)               (Mazzeo-Melrose)            (G₂ forms)
```

---

## Piste A : Théorie d'Indice (APS)

### Idée Centrale
Le **+1 dans H* = b₂ + b₃ + 1** pourrait venir de l'η-invariant.

### Papers Clés
| Paper | Auteurs | Contribution |
|-------|---------|--------------|
| Spectral asymmetry I-III | Atiyah-Patodi-Singer | Théorie fondamentale |
| An analytic invariant of G₂ | Crowley-Goette-Nordström | ν-invariant pour G₂ |
| Eta invariants and manifolds | Müller | Calculs explicites |

### À Faire
1. Calculer η pour ℂ³/ℤ₂ résolu par Eguchi-Hanson
2. Vérifier si η_total = 1 (mod 2) pour T⁷/ℤ₂³

### Probabilité de Succès : 40%

---

## Piste B : Chirurgie Analytique

### Idée Centrale
Les valeurs propres des variétés TCS sont gouvernées par la **géométrie du cou**.

### Papers Clés
| Paper | Auteurs | Contribution |
|-------|---------|--------------|
| Analytic surgery and eigenvalues | Hassell-Mazzeo-Melrose | Théorie fondamentale |
| Neck-stretching problems | (2024) | Application aux G₂-TCS |
| Gluing spectral | Mazzeo-Pacini | Techniques de recollement |

### À Faire
1. Lire Section 5 de arXiv:2301.03513 (applications G₂)
2. Calculer la constante C dans λ₁ ~ C/T²
3. Voir si C dépend de H*

### Probabilité de Succès : 30%

---

## Piste C : Théorie de Hodge G₂

### Idée Centrale
Les nombres de Betti contraignent le spectre via la décomposition de Hodge.

### Papers Clés
| Paper | Auteurs | Contribution |
|-------|---------|--------------|
| Compact G₂ manifolds I-II | Joyce | Construction + Betti |
| Laplacian flow for G₂ | Wei | Évolution du spectre |
| G₂ structures | Bryant | Structure des formes |

### À Faire
1. Chercher des bornes λ₁ vs bₖ dans la littérature
2. Étudier la constante de Cheeger pour G₂
3. Relier Vol et H*

### Probabilité de Succès : 25%

---

## Piste D : Calcul Local (Eguchi-Hanson)

### Idée Centrale
Le spectre global est construit à partir des contributions locales des singularités résolues.

### Papers Clés
| Paper | Auteurs | Contribution |
|-------|---------|--------------|
| Eigenvalues on Eguchi-Hanson | (ResearchGate) | Équation de Heun |
| Pöschl-Teller potential | Standard | Solutions exactes |
| ALE spaces | Kronheimer | Classification |

### À Faire
1. Résoudre l'équation de Heun confluente pour EH
2. Trouver λ₁(ℂ³/ℤ₂)
3. Comprendre la synchronisation des 16 singularités

### Probabilité de Succès : 35%

---

## Connexions Entre Pistes

```
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │   A (Index)  ←─── η-invariant ───→  B (Surgery)    │
    │       │                                  │          │
    │       │                                  │          │
    │   +1 dans H*                    λ₁ ~ C/T²         │
    │       │                                  │          │
    │       ▼                                  ▼          │
    │   D (Local)  ←─── singularités ───→  C (Hodge)    │
    │                                                     │
    │   λ₁(EH) = 1/4 ?           b₂, b₃ harmoniques    │
    │                                                     │
    └─────────────────────────────────────────────────────┘
```

---

## Roadmap Recommandée

### Phase 1 : Préparation (1-2 semaines)
- [ ] Lire Hassell-Mazzeo-Melrose 1995 (Piste B)
- [ ] Lire Crowley-Nordström 2015/2025 (Piste A)
- [ ] Comprendre l'équation de Heun sur EH (Piste D)

### Phase 2 : Calculs (2-4 semaines)
- [ ] Calculer η(ℂ³/ℤ₂) explicitement
- [ ] Calculer λ₁(EH) par Pöschl-Teller
- [ ] Vérifier si C = 14 dans la formule de neck-stretching

### Phase 3 : Synthèse (2-4 semaines)
- [ ] Assembler les résultats locaux → globaux
- [ ] Rédiger une conjecture précise avec hypothèses
- [ ] Identifier les gaps restants

### Phase 4 : Preuve (? mois)
- [ ] Attaquer la preuve complète ou
- [ ] Publier la conjecture + evidence

---

## Questions Ouvertes Clés

1. **Pourquoi 14 ?**
   - dim(G₂) = 14, mais pourquoi apparaît-il au numérateur ?

2. **Pourquoi b₂ + b₃ + 1 ?**
   - Le +1 vient-il de η ou de h (noyau) ?
   - Pourquoi b₂ + b₃ et pas b₃ seul ?

3. **Universalité ?**
   - Est-ce vrai pour TOUTES les variétés G₂ ou seulement certaines constructions ?

4. **Lien physique ?**
   - Comment λ₁ se traduit-il en mass gap de Yang-Mills via KK ?

---

## Ressources Compilées

### PDFs à Télécharger
1. [Hassell-Mazzeo-Melrose 1995](https://intlpress.com/site/pub/files/_fulltext/journals/cag/1995/0003/0001/CAG-1995-0003-0001-a004.pdf)
2. [Neck-stretching 2024](https://arxiv.org/abs/2301.03513)
3. [Crowley-Nordström ν-invariant](https://arxiv.org/abs/1505.02734)
4. [Joyce G₂ handout](https://people.maths.ox.ac.uk/joyce/G2Handout.pdf)

### Contacts Potentiels
- **Jason Lotay** (Oxford) — G₂ instantons, spectral curves
- **Johannes Nordström** (Bath) — ν-invariant, TCS
- **Rafe Mazzeo** (Stanford) — Analytic surgery

---

## Conclusion

La preuve de λ₁ = 14/H* nécessite de combiner :
1. **Théorie d'indice** (pour le +1)
2. **Chirurgie analytique** (pour le comportement global)
3. **Calculs locaux** (pour les singularités)
4. **Théorie de Hodge G₂** (pour les Betti numbers)

Aucune de ces pistes seule ne suffit. C'est leur **intersection** qui donnera la preuve.

---

*"The spectral gap is not a number we fit — it's a number the topology dictates."*

Maintenant on sait **où chercher** pourquoi c'est 14/H*.
