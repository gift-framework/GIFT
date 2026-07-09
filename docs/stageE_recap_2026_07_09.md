# Récap Stage E — Level Q coefficient package — 2026-07-09

## Résumé court

Stage E a transformé la couche Phase 4.1 Donaldson coefficients en package
Level Q citable :

- les valeurs exactes restent produites rationnellement;
- le checker Stage C/E recompte les formules et fabrique les intervalles
  outward-rounded;
- un package compact extrait les champs citables;
- un checker vérifie que ce package est une projection lossless;
- une table paper-facing est gardée synchronisée par un checker dédié.

## Artefacts principaux

Producteur / checker complet :

- `scripts/phase4_donaldson_coefficients_values.py`
- `scripts/phase4_donaldson_coefficients_checker.py`
- `certificates/phase4_donaldson_coefficients_values.json`
- `certificates/phase4_donaldson_coefficients_check.json`

Package Level Q compact :

- `scripts/phase4_level_Q_coefficients.py`
- `scripts/phase4_level_Q_coefficients_checker.py`
- `certificates/phase4_level_Q_coefficients.json`
- `certificates/phase4_level_Q_coefficients_check.json`

Table paper-facing :

- `paper/theorem_Q_certified.md`
- `scripts/phase4_level_Q_table_checker.py`
- `certificates/phase4_level_Q_table_check.json`

Documentation :

- `docs/stageE_start_2026_07_09.md`
- `docs/phase4_progress_2026_07_03.md`

## Résultats de checks

```text
phase4_donaldson_coefficients_check: 172/172 pass
phase4_level_Q_coefficients_check:   185/185 pass
phase4_level_Q_table_check:          43/43 pass
```

## R_threshold

Le package Level Q expose maintenant les deux formes de citation :

```text
machine bracket: [3664.065985330004, 3664.065985330005]
machine endpoint: 3664.065985330005
human citable upper: 3664.066
```

Champ plat ajouté :

```json
"citable_upper": "3664.066"
```

Texte recommandé :

```text
R_threshold <= 3664.066, with machine bracket [3664.065985330004, 3664.065985330005]
```

Le checker vérifie que :

- le endpoint machine est l'upper endpoint du bracket;
- `3664.066` est conservateur;
- le texte recommandé contient les deux formes.

## Table citable

`paper/theorem_Q_certified.md` contient maintenant une table compacte des 12
quantités :

- `source_P1`
- `source_P2`
- `DP1_norm`
- `DP2_norm`
- `D2P1_norm`
- `D3m_norm`
- `raw_P3_scale`
- `xi1_bound`
- `xi2_forcing`
- `xi2_bound`
- `remainder_R3`
- `tail_contraction_denominator`

La ligne `R_threshold` est ajoutée séparément avec le bracket cube-root et la
forme `<= 3664.066`.

Le checker `phase4_level_Q_table_checker.py` vérifie que cette table reste
alignée avec `phase4_level_Q_coefficients.json`.

## datum_D0

`certificates/datum_D0.json` référence maintenant :

- `phase4_level_Q_coefficients`
- `phase4_level_Q_table_check`

Les champs volontairement ouverts restent inchangés :

- `inj_base`
- `curv_base`
- intervalle exact bilatéral pour `A_bulk(alpha_1, alpha_1)`

## Frontière de claim

Stage E ferme le packaging intervalle outward-rounded de la couche coefficient
Level Q à `D0`.

Stage E ne prétend pas prouver :

- une uniformité hors `D0`;
- l'hypothèse anisotropic Joyce `(J)`;
- de nouveaux estimates L2, déjà importés séparément par Stage D.

## Prochaines suites naturelles

1. Ajouter une entrée changelog/release note Stage E.
2. Décider si le package Level Q doit être cité directement dans les textes
   publics ou rester derrière `theorem_Q_certified.md`.
3. Passer au chantier suivant hors packaging : les champs encore pending de
   `datum_D0.json` ou une passe de consolidation avant commit.
