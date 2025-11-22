# K₇ G₂ TCS Explicit Metric v1.0f

Version optimisée basée sur l'expérience v1.0e avec training étendu et extraction harmonique multi-résolution.

## Changements vs v1.0e

### 1. Training étendu
- **Epochs par phase** : 1500 (vs 1000 en v1.0e)
- **Durée estimée** : ~2h30 pour le training complet (vs 1h40)
- **Patience early stop** : 300 (vs 100) pour Phase 5

### 2. Extraction harmonique multi-résolution
- **Training grid** : 16^7 = 268,435,456 points (haute résolution)
- **Harmonics grid** : 8^7 = 2,097,152 points (grille réduite)
- **Speedup** : 128× plus rapide pour l'extraction harmonique
- **Durée estimée** : 10-15 min (vs crash/2h+ avec 16^7)

### 3. Configuration

```python
CONFIG = {
    'n_grid': 16,                    # Training haute résolution
    'n_grid_harmonics': 8,           # NOUVEAU: grille réduite pour Laplacien
    'n_epochs_per_phase': 1500,      # Augmenté de 1000
    'checkpoint_dir': 'checkpoints_v1_0f',
    # ... reste identique à v1.0e
}
```

## Justification scientifique

### Training étendu (1500 epochs)
- v1.0e a montré convergence rapide mais 1500 epochs permettent :
  - Exploration plus complète de l'espace des solutions
  - Consolidation de la géométrie G₂
  - Réduction supplémentaire de la torsion résiduelle

### Multi-résolution pour harmoniques
- Les modes harmoniques bas (b₂=21, b₃=77) ne nécessitent pas 16^7 points
- Théoriquement, 8^7 suffit pour capturer les premiers modes
- Évite les problèmes de mémoire tout en conservant la validité physique
- Approche standard en analyse numérique (coarse-to-fine)

## Estimation temporelle totale

- **Phase 1-5 training** : ~2h30
- **Extraction harmonique 2-forms** : ~5-10 min
- **Extraction harmonique 3-forms** : ~5-10 min
- **Yukawa tensor (21×21×77)** : ~5 min
- **Total** : ~3h (gérable sur Colab Pro+ A100)

## Outputs

Le pipeline génère :
- `checkpoints_v1_0f/` : Checkpoints du training
- `outputs_v1_0f/` : Résultats finaux
  - `harmonic_2forms.npy` : 21 modes harmoniques (2-formes)
  - `harmonic_3forms.npy` : 77 modes harmoniques (3-formes)
  - `yukawa_tensor.npy` : Tenseur de Yukawa (21×21×77)
  - `phi_samples.npy` : Échantillons de la 3-forme φ
  - `metric_samples.npy` : Échantillons de métrique G₂
  - `loss_history.npy` : Historique complet du training
  - `metadata.json` : Configuration et statistiques
- `K7_G2_TCS_v1_0f_complete_results.zip` : Archive complète

## Validation multi-résolution

La cellule 15 du notebook fournit une validation complète :
- Configuration des grilles
- Résultats cohomologiques (b₂, b₃)
- Validation physique (Yukawa, det(g), torsion)
- Résumé du training

## Utilisation

1. Ouvrir le notebook dans Google Colab
2. Runtime → Change runtime type → A100 GPU
3. Exécuter toutes les cellules
4. Le training s'arrêtera automatiquement si convergence (patience=300)
5. Les résultats seront automatiquement téléchargés

## Héritage de v1.0e

Conserve toutes les améliorations de stabilité numérique :
- Régularisation phase-adaptative de la métrique
- Guards NaN multiples
- Clipping préventif pour dφ/dψ
- Fallback robuste pour eigenvalues
- Torsion floor pour éviter sur-convergence



