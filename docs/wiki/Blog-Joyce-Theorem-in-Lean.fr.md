---
title: "Blog : Le théorème de Joyce, désormais en Lean"
layout: default
---

> Publié à l'origine sur [arithmon.substack.com](https://arithmon.substack.com/p/joyces-theorem-now-in-lean)

> **Statut 2026-07-09.** Billet historique. La branche analytique courante ne
> considère pas le théorème compact sur `K_7` comme fermé : le verrou restant est
> le théorème de perturbation anisotrope `(J)`, suivi dans
> `docs/analytic_status.md`.

# Le théorème de Joyce, désormais en Lean
Vers une existence vérifiée par machine des variétés G₂

10 décembre 2025

« K₇ existe-t-il vraiment ? »

C'est une question raisonnable. GIFT prétend que la physique émerge d'une variété 7D à holonomie G₂. Mais affirmer qu'une variété existe et prouver qu'elle existe sont deux exercices différents.

En 1996, Dominic Joyce a prouvé que les variétés G₂ compactes existent. La preuve utilise de l'analyse difficile : espaces de Sobolev, théorèmes des fonctions implicites, applications contractantes. Elle s'étale sur quelque 200 pages de géométrie différentielle.

Nous avons maintenant formalisé des parties clés de cet argument en Lean 4. Voici ce que cela signifie, et ce que cela ne signifie pas.

Le défi des théorèmes d'existence

Les preuves d'existence en géométrie sont notoirement difficiles à vérifier. Elles impliquent typiquement :

- Des espaces de fonctions de dimension infinie
- Des estimations qui « découlent de la théorie elliptique standard »
- Des constantes qui sont « suffisamment petites »
- Des itérations qui « convergent par le théorème de Banach »

Chaque étape peut être correcte. Mais la chaîne est longue, et la vérifier exige une expertise qui n'est pas largement partagée. Résultat : les théorèmes d'existence sont cités, mais rarement redérivés à partir de zéro.

La vérification formelle offre une approche différente. Et si une machine suivait chaque estimation ?

Ce que dit le théorème de Joyce

Soit M une variété compacte de dimension 7 avec une G₂-structure φ₀. La structure a une torsion T(φ₀), qui mesure à quel point φ₀ est éloignée d'être « sans torsion », le cas désirable, qui implique d'être Ricci-plate.

**Théorème de Joyce (en gros)** : si ‖T(φ₀)‖ < ε₀ pour un certain seuil ε₀, alors il existe une G₂-structure lisse sans torsion φ sur M, avec ‖φ − φ₀‖ ≤ C·‖T(φ₀)‖.

Autrement dit : si vous trouvez une G₂-structure approximative avec une torsion suffisamment petite, une structure exacte existe à proximité.

La subtilité : ε₀ dépend des constantes de Sobolev de M. Le calculer pour une variété spécifique demande une analyse soignée.

Ce que la formalisation Lean couvre

La formalisation a trois couches :

**Couche 1 : Cadre abstrait**

Les espaces de Sobolev, les formes différentielles et le théorème des fonctions implicites sont mis en place comme des structures Lean :

```lean
-- Plongement de Sobolev : H^4 se plonge continûment dans C^0
theorem sobolev_embedding_H4_C0 (M : Manifold) [Compact M] :
    ContinuousEmbedding (H 4 M) (C 0 M)

-- Le laplacien de Hodge est auto-adjoint
theorem hodge_laplacian_self_adjoint :
    IsSelfAdjoint (hodge_laplacian : Ω^k M → Ω^k M)
```

**Couche 2 : L'itération de Joyce comme contraction**

La preuve de Joyce fonctionne en itérant une application de correction. La formalisation capture la propriété clé : cette application est une contraction de constante K < 1.

```lean
noncomputable def joyce_K : NNReal := ⟨9/10, by norm_num⟩

theorem joyce_K_lt_one : joyce_K < 1 := by simp [joyce_K]; norm_num

theorem joyce_is_contraction : ContractingWith joyce_K JoyceFlow :=
  ⟨joyce_K_lt_one, joyce_lipschitz⟩
```

**Couche 3 : Point fixe de Banach**

Mathlib fournit le théorème du point fixe de Banach, prouvé à partir des premiers principes dans la bibliothèque :

```lean
-- Existence via ContractingWith.fixedPoint de Mathlib
noncomputable def torsion_free_structure : G2Structures :=
  joyce_is_contraction.fixedPoint JoyceFlow

theorem k7_admits_torsion_free_g2 :
    ∃ φ : G2Structures, IsTorsionFree φ :=
  ⟨torsion_free_structure, fixed_point_is_torsion_free⟩
```

Quand cela compile, Lean a vérifié la chaîne logique allant de la contraction à l'existence.

Le côté numérique

L'existence abstraite a besoin d'un ancrage. Nous devons aussi vérifier que K₇ satisfait spécifiquement aux hypothèses de Joyce.

Un réseau de neurones informé par la physique (PINN) construit une G₂-structure approximative φ₀ sur K₇. Le réseau a environ 200 000 paramètres et s'entraîne en 5 à 10 minutes sur des plateformes gratuites comme Colab.

La borne sur la torsion est la borne critique. Notre estimation pour le seuil de Joyce sur K₇ est ε₀ ≈ 0,0288. Le PINN atteint ‖T‖ = 0,00140.

Marge de sécurité : approximativement 20×

Cette marge donne une certaine confiance que la borne n'est pas marginale. Bien sûr, la vraie valeur de ε₀ pour K₇ dépend de constantes de Sobolev que nous n'avons pas calculées exactement, nous y reviendrons plus bas.

Audit des axiomes

Que suppose la preuve ? Lean le rend explicite :

```
#print axioms k7_admits_torsion_free_g2
```

**Axiomes standards** (de Lean/Mathlib) :

- `propext` : extensionnalité propositionnelle
- `Quot.sound` : robustesse des quotients

Ce sont des fondations standards, rien d'exotique.

**Axiomes de domaine** (interface vers la géométrie) :

| Axiome | Sens | Source |
|---|---|---|
| K7 | K₇ existe comme type topologique | interface abstraite |
| JoyceFlow | l'application d'itération existe | construction de Joyce |
| joyce_lipschitz | JoyceFlow a une constante de Lipschitz < 1 | propriété de contraction |
| fixed_point_torsion_zero | les points fixes sont sans torsion | conclusion du théorème de Joyce |

Notamment, le théorème du point fixe de Banach n'est pas axiomatisé : il vient de `ContractingWith.fixedPoint` de Mathlib, qui est prouvé à l'intérieur de la bibliothèque.

Ce que cela ne prouve PAS

Il vaut la peine d'être clair sur les limites :

1. **Le théorème de Joyce n'est pas formalisé à partir des premiers principes.** Le cœur analytique (estimations de Sobolev, régularité elliptique, théorie de Schauder) est axiomatisé plutôt que prouvé en Lean. Une formalisation complète serait un projet substantiel, qui demanderait probablement des années de travail.

2. **Le seuil ε₀ est estimé, pas calculé exactement.** Nous utilisons une estimation conservative basée sur des constantes de Sobolev typiques. La vraie valeur pour K₇ requerrait une analyse plus détaillée.

3. **L'interprétation physique reste conjecturale.** L'affirmation que notre univers implique une compactification sur K₇ est empirique, pas mathématique. Cette formalisation ne touche pas à cette question.

4. **L'unicité est inconnue.** D'autres G₂-structures avec ces invariants peuvent exister. La structure de l'espace des modules n'a pas été caractérisée.

Ce que cela accomplit

Malgré ces réserves, la formalisation accomplit quelque chose d'utile :

- **Arithmétique vérifiée par machine.** Les bornes (marge de sécurité 20×, accord sur det(g), etc.) sont vérifiées par Lean, pas à la main.
- **Hypothèses transparentes.** Chaque axiome est listé explicitement. Pas d'étape « croyez-moi sur parole » cachée.
- **Reproductibilité.** N'importe qui peut vérifier le calcul :

```
pip install giftpy
python -c "from gift_core.analysis import JoyceCertificate; print(JoyceCertificate.verify())"
```

- **Structure modulaire.** L'argument est décomposé en lemmes autonomes. Chaque pièce peut être examinée ou améliorée indépendamment.

Le travail restant

La formalisation relie deux extrémités :

- **En haut** : existence abstraite via le point fixe de Banach (formalisé)
- **En bas** : bornes numériques issues de l'entraînement du PINN (certifiées)

La couche du milieu (la machinerie analytique de Joyce) est actuellement axiomatisée. Compléter le tableau impliquerait de formaliser :

- La régularité elliptique sur les variétés compactes
- Les théorèmes de multiplication et de plongement de Sobolev
- Les estimations de Schauder pour les EDP non linéaires

C'est des mathématiques substantielles. Des efforts similaires (comme la formalisation de parties du dernier théorème de Fermat) ont pris des années. Nous ne prétendons pas l'avoir fait.

Mais l'échafaudage existe. L'énoncé du théorème compile. Les bornes numériques sont vérifiées. Ce qu'il reste à faire, c'est de remplir le milieu analytique, une tâche bien définie, si exigeante.

Conclusion

```lean
theorem k7_admits_torsion_free_g2 : ∃ φ : G2Structures, IsTorsionFree φ
```

Cet énoncé compile en Lean 4.14.0 avec Mathlib. Sous réserve des axiomes listés ci-dessus, l'existence d'une G₂-structure sans torsion sur K₇ est vérifiée par machine.

Les axiomes sont explicites. Les manques sont reconnus. Les preuves numériques sont reproductibles.

Si cela se connecte à la physique, c'est une question distincte, une question à laquelle ce sont les expériences, et non les assistants de preuve, qui finiront par répondre.

Dépôt : github.com/Arithmon/K7-Lean

Notebook : github.com/Arithmon/K7/notebooks
