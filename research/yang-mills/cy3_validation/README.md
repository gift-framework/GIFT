# CY₃ Spectral Validation

**Sidequest dans la sidequest**: Tester la loi spectrale universelle sur les Calabi-Yau 3-folds.

---

## Conjecture Universelle

$$\lambda_1 \times H^* = \dim(\text{Hol}) - h$$

| Variété | Holonomie | dim(Hol) | h (spineurs) | Target |
|---------|-----------|----------|--------------|--------|
| K₇ (G₂) | G₂ | 14 | 1 | **13** ✓ |
| **CY₃** | **SU(3)** | **8** | **2** | **6** ? |
| K3 | SU(2) | 3 | 2 | 1 ? |
| Spin(7) | Spin(7) | 21 | 1 | 20 ? |

**Statut G₂**: Validé exactement à N=50,000 (λ₁ × H* = 13.0)

**Objectif CY₃**: Valider λ₁ × H* = 6

---

## Structure

```
cy3_validation/
├── config_cy3.py                    # Configuration centralisée
├── notebooks/
│   ├── CY3_Phase1_T6_Calibration.ipynb    # T⁶ flat (calibration pipeline)
│   ├── CY3_Phase2_T6_Z3_Orbifold.ipynb    # T⁶/ℤ₃ (premier vrai CY₃)
│   └── CY3_Phase3_CICY_Quintic.ipynb      # Quintic P⁴[5]
├── outputs/                         # Résultats JSON + plots
└── README.md
```

---

## Phases de Validation

### Phase 1: T⁶ Calibration ⭐

**Objectif**: Vérifier que le pipeline reproduit λ₁ analytique connu.

- **Variété**: Tore plat T⁶ = (S¹)⁶
- **λ₁ exact**: 1.0 (pour R = 1)
- **Holonomie**: Triviale (PAS SU(3))
- **Critère PASS**: Déviation < 20%

**Note**: T⁶ n'est PAS un test de la conjecture, juste une calibration.

### Phase 2: T⁶/ℤ₃ Orbifold ⭐⭐

**Objectif**: Premier test réel sur CY₃ avec holonomie SU(3).

- **Variété**: T⁶/ℤ₃ orbifold
- **Hodge**: h¹¹ = 9, h²¹ = 0, χ = 18
- **H* candidates**: A=11, B=13, C=11
- **Target λ₁**: ~0.55 (si H*=11 et λ₁×H*=6)
- **Critère PASS**: Au moins une définition H* donne λ₁×H* ∈ [5, 7]

### Phase 3: Quintic CICY ⭐⭐⭐

**Objectif**: Tester sur le CY₃ le plus célèbre.

- **Variété**: Quintic hypersurface P⁴[5]
- **Hodge**: h¹¹ = 1, h²¹ = 101, χ = -200
- **H* candidates**: A=104, B=207, C=102
- **Target λ₁**: ~0.058 (si H*≈100)
- **Métrique**: Fubini-Study (approximation ambiante)
- **Critère PASS**: Déviation < 20%

**Limitation**: Pas de métrique Ricci-flat explicite. Utilise FS comme approximation.

---

## Définitions H* à Tester

| Option | Formule | Interprétation |
|--------|---------|----------------|
| **A** | h¹¹ + h²¹ + 2 | Hodge middle + spineurs |
| **B** | h¹¹ + 2h²¹ + 4 | Betti total + spineurs |
| **C** | \|h¹¹ - h²¹\| + 2 | Euler-based |

**Comparaison avec G₂**: Pour K₇, H* = b₂ + b₃ + 1 = 99.
Le "+1" identifié comme le spineur parallèle unique de G₂.

---

## Utilisation des Notebooks

### Prérequis

```python
numpy
scipy
matplotlib
# Optionnel pour GPU:
cupy  # Pour A100 acceleration
```

### Exécution

1. **Phase 1** (calibration):
   - Run `CY3_Phase1_T6_Calibration.ipynb`
   - Vérifier que λ₁ ≈ 1.0 pour T⁶
   - Télécharger `T6_calibration_results.json`

2. **Phase 2** (premier test réel):
   - Run `CY3_Phase2_T6_Z3_Orbifold.ipynb`
   - Identifier quelle définition H* fonctionne
   - Télécharger `T6_Z3_validation_results.json`

3. **Phase 3** (Quintic):
   - Run `CY3_Phase3_CICY_Quintic.ipynb`
   - Comparer avec Phase 2
   - Télécharger `Quintic_validation_results.json`

---

## Résultats Attendus

### Scénario PASS (Conjecture validée)

```
╔═══════════════════════════════════════════════════════════════╗
║  UNIVERSAL SPECTRAL LAW: λ₁ × H* = dim(Hol) - h              ║
╠═══════════════════════════════════════════════════════════════╣
║  G₂ (K₇):     14 - 1 = 13  ✓  (mesuré: 13.0)                 ║
║  SU(3) (CY₃): 8 - 2 = 6    ✓  (mesuré: ~6.0)                 ║
╚═══════════════════════════════════════════════════════════════╝
```

### Scénario FAIL (Conjecture réfutée)

- Tous CY₃ donnent λ₁×H* >> 6 ou << 6
- Ou: aucune définition H* n'est consistante entre variétés

---

## Références Clés

### Littérature CY₃

- [arXiv:2305.08901](https://arxiv.org/abs/2305.08901) - Numerical spectra on CY hypersurfaces
- [arXiv:2410.11284](https://arxiv.org/abs/2410.11284) - Grassmannian learning + Donaldson
- [Ashmore 2020](https://www.semanticscholar.org/paper/Eigenvalues-and-eigenforms-on-Calabi-Yau-threefolds-Ashmore/) - Foundational eigenvalue computations

### Validation G₂ (GIFT)

- `/research/yang-mills/spectral_validation/N50000_GPU_VALIDATION.md`
- `/research/yang-mills/UNIVERSALITY_CONJECTURE.md`

---

## Contact

Résultats à partager avec Claude pour analyse et synthèse.

---

*"Le papillon de Calabi-Yau danse dans l'espace des modules..."*
