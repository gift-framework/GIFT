# Quick Start - K₇ G₂ TCS v1.0f

## En 3 étapes

### 1. Upload vers Google Colab
```
1. Ouvrir Google Colab (colab.research.google.com)
2. File → Upload notebook
3. Sélectionner : K7_G2_TCS_ExplicitMetric_v1_0f.ipynb
```

### 2. Configurer le runtime
```
1. Runtime → Change runtime type
2. Hardware accelerator : GPU → A100
3. Runtime shape : High-RAM
4. Save
```

### 3. Exécuter
```
1. Runtime → Run all
2. Attendre ~3h (café ☕)
3. Télécharger : K7_G2_TCS_v1_0f_complete_results.zip
```

## Que fait le notebook ?

### Phase 1-5 : Training (~2h30)
```
Phase 1: TCS_Neck               [1500 epochs] ████████████████████
Phase 2: ACyl_Matching          [1500 epochs] ████████████████████
Phase 3: Cohomology_Refinement  [1500 epochs] ████████████████████
Phase 4: Harmonic_Extraction    [1500 epochs] ████████████████████
Phase 5: Final_Calibration      [1500 epochs] ████████████████████
                                              ↑ Early stop possible
```

### Post-Training (~30 min)
```
→ Build Laplacian 2-forms (8^7 grid)    [5-10 min]
→ Build Laplacian 3-forms (8^7 grid)    [5-10 min]
→ Extract harmonic modes (b₂=21, b₃=77) [instant]
→ Compute Yukawa tensor (21×21×77)      [5 min]
→ Calibration & validation              [instant]
→ Save & archive results                [instant]
```

## Résultats obtenus

### Fichier ZIP téléchargé
```
K7_G2_TCS_v1_0f_complete_results.zip
├── harmonic_2forms.npy       # 21 modes harmoniques (2-formes)
├── harmonic_3forms.npy       # 77 modes harmoniques (3-formes)
├── yukawa_tensor.npy         # Tenseur de Yukawa (21×21×77)
├── phi_samples.npy           # 5000 échantillons de φ
├── metric_samples.npy        # 5000 métriques G₂
├── loss_history.npy          # Historique complet du training
└── metadata.json             # Config + statistiques
```

### Checkpoints intermédiaires
```
checkpoints_v1_0f/
├── checkpoint_latest.pt              # Dernier checkpoint
├── checkpoint_phase1_epoch_499.pt    # Phase 1, epoch 500
├── checkpoint_phase1_epoch_999.pt    # Phase 1, epoch 1000
├── checkpoint_phase1_epoch_1499.pt   # Phase 1, epoch 1500
├── ... (idem pour phases 2-5)
└── config.json                       # Configuration sauvée
```

## Monitoring pendant l'exécution

### Ce que vous verrez

```
Training grid: 16^7
Harmonics grid: 8^7
Target: b₂=21, b₃=77
Epochs per phase: 1500
Phases: 5

============================================================
PHASE 1: TCS_Neck
============================================================
[███░░░] 10% | Epoch 150/1500 | Loss=2.42 | dφ=1.5e-17 | ETA: 32m
[██████░] 20% | Epoch 300/1500 | Loss=0.00 | dφ=3.1e-18 | ETA: 28m
...
✓ Checkpoint saved
...
[███████████████████████████████] 100% | Epoch 1500/1500

============================================================
PHASE 2: ACyl_Matching
============================================================
...
```

### Early stopping (possible)
```
============================================================
PHASE 5: Final_Calibration
============================================================
[███████░░░] 23% | Epoch 345/1500 | Loss=0.36 | ...

Early stopping at Phase 5, Epoch 345
All G₂ constraints satisfied (patience=300)
```

## Que faire si...

### ❌ Runtime disconnects
```
→ Ne paniquez pas ! Les checkpoints sont sauvegardés
→ Relancer : Runtime → Run all
→ Le notebook reprend automatiquement depuis le dernier checkpoint
```

### ❌ Out of Memory (peu probable avec A100)
```
→ Vérifier : Runtime shape = High-RAM
→ Si persiste : Réduire batch_size dans CONFIG (cellule 2)
   'batch_size': 512,  # au lieu de 1024
```

### ❌ Trop lent
```
→ C'est normal : 1500 epochs × 5 phases = beaucoup de calcul
→ Pour test rapide : Réduire n_epochs_per_phase à 500
   'n_epochs_per_phase': 500,  # au lieu de 1500
```

### ✓ Tout se passe bien
```
→ Attendez simplement la fin
→ Le ZIP se télécharge automatiquement
→ Consultez la cellule 15 pour la validation
```

## Interprétation des résultats

### Métrique de convergence
```
Loss final : < 0.5        ✓ Excellent
dφ         : < 1e-6       ✓ Torsion quasi-nulle
dψ         : < 1e-6       ✓ G₂ quasi-exacte
det(g)     : ~2.0 ± 0.01  ✓ Métrique normalisée
```

### Cohomologie
```
b₂_eff = 21/21  (100%)    ✓ Parfait
b₃_eff = 77/77  (100%)    ✓ Parfait
OU
b₂_eff = 18/21  (85%)     ⚠ Acceptable (modes proches du seuil)
b₃_eff = 72/77  (93%)     ⚠ Acceptable
```

### Yukawa
```
Shape  : (21, 21, 77)     ✓ Dimensions correctes
Norm   : > 0              ✓ Non-trivial
Norm   : O(1-10)          ✓ Ordre de grandeur physique
```

## Support

- **Documentation** : Voir `README.md` pour détails
- **Comparaison** : Voir `COMPARISON_v1.0e_vs_v1.0f.md`
- **Changelog** : Voir `CHANGES.md` pour modifications techniques
- **Implementation** : Voir `IMPLEMENTATION_SUMMARY.md` pour synthèse

## Prochaines étapes

Après avoir obtenu les résultats :

1. **Analyser** : Ouvrir `metadata.json` pour les statistiques
2. **Visualiser** : Charger les `.npy` avec NumPy
3. **Valider** : Comparer avec les cibles théoriques
4. **Publier** : Résultats prêts pour article/présentation

---

**Durée totale** : ~3h sur A100  
**Succès attendu** : >95% avec paramètres par défaut  
**Support matériel** : Google Colab Pro+ recommandé



