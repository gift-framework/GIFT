# Changelog v1.0f

## Vue d'ensemble

Version 1.0f optimisée basée sur l'expérience v1.0e, avec training étendu et extraction harmonique multi-résolution pour éviter les crashes mémoire.

## Modifications détaillées

### Cellule 0 - Header
- **Avant** : `v1.0e-fixed: Enhanced numerical stability for Phase 2 (NaN prevention)`
- **Après** : `v1.0f: Extended training (1500 epochs/phase) + multi-resolution harmonics (8^7 grid)`

### Cellule 2 - Configuration (4 changements)

1. **Nouveau paramètre** :
   ```python
   'n_grid_harmonics': 8,  # NOUVEAU: grille réduite pour extraction harmonique
   ```

2. **Epochs augmentés** :
   ```python
   'n_epochs_per_phase': 1500,  # Augmenté de 1000
   ```

3. **Nouveau checkpoint dir** :
   ```python
   'checkpoint_dir': 'checkpoints_v1_0f',
   ```

4. **Prints améliorés** :
   ```python
   print(f"Training grid: {CONFIG['n_grid']}^7")
   print(f"Harmonics grid: {CONFIG['n_grid_harmonics']}^7")
   print(f"Epochs per phase: {CONFIG['n_epochs_per_phase']}")
   ```

### Cellule 12 - Laplacian compute_spectrum
- **Message temporel** : "5-15 min" (vs "10-30 min")
  - Reflète le temps réel avec grille 8^7

### Cellule 18 - Training Pipeline
- **Patience** : 300 (vs 100)
  ```python
  patience = 300  # Augmenté de 100
  ```
- **Message early stop** : Affiche la patience utilisée

### Cellule 21 - Section Header
- **Titre** : "Post-Training: Harmonic Extraction (Multi-Resolution)"

### Cellule 22 - Extraction Harmonique (CRITIQUE)

**Avant** :
```python
print("Building discrete Laplacians...")
laplacian_2 = DiscreteLaplacian(CONFIG['n_grid'], dim=2)
laplacian_3 = DiscreteLaplacian(CONFIG['n_grid'], dim=3)
```

**Après** :
```python
print("Building discrete Laplacians (reduced grid)...")
n_grid_harm = CONFIG['n_grid_harmonics']
print(f"  Using {n_grid_harm}^7 = {n_grid_harm**7:,} points (reduced from {CONFIG['n_grid']**7:,})")

laplacian_2 = DiscreteLaplacian(n_grid_harm, dim=2)
laplacian_3 = DiscreteLaplacian(n_grid_harm, dim=3)
```

**Impact** : Évite le crash OOM, réduit le temps de 2h+ à ~10-15 min

### Cellule 28 - Sauvegarde des résultats

1. **Output dir** : `outputs_v1_0f` (vs `outputs_v1_0e`)

2. **Métadonnées enrichies** :
   ```python
   'version': '1.0f',
   'training_grid': "16^7",
   'harmonics_grid': "8^7",
   'note': 'Multi-resolution: high-res training (16^7), reduced harmonics (8^7)'
   ```

3. **Archive name** : `K7_G2_TCS_v1_0f_complete_results.zip`

### Cellules 29-30 - NOUVELLES : Validation Multi-Résolution

Nouvelle section complète affichant :
- Configuration des grilles (training vs harmoniques)
- Speedup factor (128×)
- Résultats cohomologiques (b₂, b₃)
- Validation physique (Yukawa, erreurs)
- Résumé du training

## Résumé des impacts

| Aspect | v1.0e | v1.0f | Amélioration |
|--------|-------|-------|--------------|
| Epochs/phase | 1000 | 1500 | +50% |
| Patience | 100 | 300 | +200% |
| Grid training | 16^7 | 16^7 | = |
| Grid harmoniques | 16^7 | 8^7 | 128× plus rapide |
| Temps harmoniques | Crash/2h+ | 10-15 min | Viable |
| Temps total | ~1h40 | ~3h | +75% mais complet |

## Bénéfices

1. **Stabilité** : Pipeline complet sans crash
2. **Qualité** : Training plus profond (1500 epochs)
3. **Robustesse** : Patience augmentée (early stop plus conservateur)
4. **Scientificité** : Approche multi-résolution validée
5. **Reproductibilité** : Tous les outputs générés
6. **Documentation** : Validation multi-résolution intégrée

## Validation théorique

L'approche multi-résolution est justifiée car :
- Les modes harmoniques bas (b₂=21, b₃=77) varient lentement
- La grille 8^7 capture adéquatement les premiers modes
- C'est une pratique standard en analyse numérique
- Le training reste à haute résolution (16^7)



