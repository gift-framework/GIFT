# Réorganisation du dossier G2_ML - 1er décembre 2025

## Objectif

Nettoyer et réorganiser le dossier G2_ML pour éliminer les doublons et regrouper les modules de recherche de manière cohérente.

## Actions réalisées

### 1. Suppression des doublons de versions 0.x

Les versions 0.1, 0.2, 0.7, 0.8, 0.9a, 0.9b ont été supprimées de la racine car elles étaient déjà présentes dans `archived/early_development/`.

### 2. Conservation des versions récentes

Les versions actives/récentes ont été conservées à la racine:
- `1_9b/` - Version 1.9b (stable)
- `2_0/` - Version 2.0 (stable)
- `2_1/` - Version 2.1 (développement actuel)

### 3. Création du dossier research_modules

Un nouveau dossier `research_modules/` a été créé pour regrouper les modules de recherche spécialisés:
- `meta_hodge/` - Implémentations avancées de la théorie de Hodge
- `tcs_joyce/` - Méthodes de construction de Joyce (Torelli-type theorems)
- `variational_g2/` - Approches variationnelles pour la géométrie G2

Chacun de ces modules a été déplacé depuis la racine vers ce nouveau dossier.

### 4. Mise à jour de la documentation

Le fichier README.md principal a été mis à jour pour refléter la nouvelle structure:
- Mise à jour de la section "Quick Links"
- Révision de la section "Directory Structure" avec la nouvelle organisation
- Mise à jour de la section "Versions" avec les emplacements corrects
- Ajout de références au nouveau dossier research_modules

Un nouveau fichier `research_modules/README.md` a été créé pour documenter le contenu et l'objectif de chaque module de recherche.

## Structure finale

```
G2_ML/
├── README.md
├── STATUS.md
├── FUTURE_WORK.md
├── VERSIONS.md
├── REORGANIZATION_2025-12-01.md (ce fichier)
│
├── 1_9b/                      # Version 1.9b
├── 2_0/                       # Version 2.0
├── 2_1/                       # Version 2.1 (actuelle)
│
├── archived/
│   ├── early_development/    # Versions 0.1 à 0.9
│   └── v1_iterations/        # Versions 1.0 à 1.x
│
├── research_modules/
│   ├── README.md
│   ├── meta_hodge/
│   ├── tcs_joyce/
│   └── variational_g2/
│
├── G2_Lean/                   # Vérification formelle (lié sur X.com)
└── tests/
```

## Bénéfices

Cette réorganisation apporte plusieurs avantages:
- Elimination des doublons pour réduire l'encombrement
- Meilleure séparation entre versions actives et archives
- Regroupement logique des modules de recherche
- Documentation claire de la structure
- Plus facile de naviguer et de comprendre l'organisation du projet

## Notes

Le dossier `G2_Lean/` n'a pas été modifié car il est lié sur X.com et doit rester accessible à son emplacement actuel.

Toutes les anciennes versions restent accessibles dans le dossier `archived/` pour référence historique et reproductibilité.

