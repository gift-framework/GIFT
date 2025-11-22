# Comparaison v1.0e vs v1.0f

## Tableau récapitulatif

| Paramètre | v1.0e | v1.0f | Justification |
|-----------|-------|-------|---------------|
| **Training Grid** | 16^7 (268M pts) | 16^7 (268M pts) | Maintenu - haute résolution nécessaire |
| **Harmonics Grid** | 16^7 (268M pts) | 8^7 (2M pts) | **CRASH FIX** - 128× plus rapide |
| **Epochs/Phase** | 1000 | 1500 | Exploration plus complète |
| **Patience (Phase 5)** | 100 | 300 | Early stop plus conservateur |
| **Checkpoint Dir** | `checkpoints_v1_0e` | `checkpoints_v1_0f` | Versioning propre |
| **Output Dir** | `outputs_v1_0e` | `outputs_v1_0f` | Versioning propre |

## Temps d'exécution estimés

### v1.0e
- Training (5 phases × 1000 epochs) : ~1h40
- Extraction harmoniques : **CRASH** (OOM après 20+ min)
- **TOTAL** : Training seulement, pas d'harmoniques

### v1.0f
- Training (5 phases × 1500 epochs) : ~2h30
- Extraction harmoniques 2-forms (8^7) : ~5-10 min
- Extraction harmoniques 3-forms (8^7) : ~5-10 min
- Yukawa tensor : ~5 min
- **TOTAL** : ~3h ✓ Pipeline complet

## Résultats obtenus

### v1.0e (incomplet)
- ✓ Training terminé (early stop epoch 100, phase 5)
- ✗ Extraction harmoniques plantée
- ✗ Pas de tenseur de Yukawa
- ✓ Modèle sauvegardé
- Score : **Modèle entraîné seulement**

### v1.0f (attendu complet)
- ✓ Training étendu (1500 epochs ou early stop)
- ✓ Extraction harmoniques viable
- ✓ Tenseur de Yukawa (21×21×77)
- ✓ Validation multi-résolution
- ✓ Tous les outputs
- Score : **Pipeline end-to-end**

## Modifications du code

### Cellules modifiées
1. **Cellule 0** : Header (v1.0f)
2. **Cellule 2** : Config (4 changements)
3. **Cellule 12** : Message temps Laplacian
4. **Cellule 18** : Patience 300
5. **Cellule 21** : Titre section
6. **Cellule 22** : **CRITIQUE** - Grille réduite
7. **Cellule 28** : Métadonnées + outputs

### Cellules ajoutées
- **Cellule 29** : Header validation (markdown)
- **Cellule 30** : Code validation multi-résolution (python)

**Total** : 7 cellules modifiées, 2 cellules ajoutées

## Justification scientifique

### Pourquoi 8^7 pour les harmoniques ?

1. **Nyquist-Shannon** : Pour capturer les k premiers modes, besoin de ~2k points/dimension
   - b₂ = 21 modes → besoin de ~5-6 points/dim minimum
   - b₃ = 77 modes → besoin de ~10 points/dim
   - 8 points/dim : marge confortable

2. **Décroissance spectrale** : Les modes harmoniques bas varient lentement
   - Haute fréquence = petits eigenvalues → négligeables
   - Basse fréquence = capturée par grille coarse

3. **Pratique standard** :
   - Multigrid methods en CFD
   - Coarse-to-fine en vision
   - Hierarchical basis en FEM

### Pourquoi 1500 epochs ?

v1.0e early stop à epoch 100 (phase 5) suggère :
- Convergence très rapide
- Mais peut-être locale
- 1500 epochs : exploration plus robuste
- Patience 300 : confirmation de convergence

## Recommandation

**Utiliser v1.0f** pour :
- Production : Pipeline complet garanti
- Publication : Tous les résultats nécessaires
- Validation : Multi-résolution documentée

**Utiliser v1.0e** pour :
- Debug rapide : Training seulement
- Tests : Validation de convergence
- Expérimentations : Itération rapide

## Notes de migration

Pour passer de v1.0e à v1.0f :
1. Les checkpoints v1.0e sont **compatibles**
2. Peut reprendre training v1.0e en changeant `checkpoint_dir` dans CONFIG
3. Ou repartir de zéro avec paramètres v1.0f

Code de reprise :
```python
# Dans cellule 2, modifier :
CONFIG = {
    # ... autres params v1.0f
    'checkpoint_dir': 'checkpoints_v1_0e',  # Utiliser anciens checkpoints
}
```

Le training reprendra depuis le dernier checkpoint v1.0e et continuera avec les nouveaux paramètres (1500 epochs/phase, patience 300).



