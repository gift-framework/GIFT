# Training Analysis Results

**Convergence analysis for K₇ TCS v1.0 training (15,000 epochs)**

---

## Contents

### Notebooks
- **[training_analysis.ipynb](training_analysis.ipynb)** - Main analysis notebook
  - Loads `training_history.json` (15k epochs)
  - Generates convergence plots (torsion, cohomology, loss)
  - Computes phase-by-phase statistics
  - Exports summary JSON files

### Generated Outputs (After Running Notebook)
- `convergence_plots_torsion.png` - Torsion closure/coclosure, regional structure
- `convergence_plots_cohomology.png` - Ranks, Gram determinants, harmonic penalties
- `convergence_plots_loss.png` - Total loss and calibration
- `phase_breakdown.json` - Detailed statistics for each of the 5 training phases
- `training_summary.json` - Overall summary metrics

---

## Quick Start

```bash
# Navigate to analysis directory
cd analysis/

# Run notebook
jupyter notebook training_analysis.ipynb

# Or from command line
jupyter nbconvert --to notebook --execute training_analysis.ipynb
```

**Runtime**: ~2-3 minutes

---

## Key Results

### Torsion Convergence
- **Initial** (epoch 0): 2.68×10⁻⁴
- **Final** (epoch 14999): **1.78×10⁻¹¹**
- **Improvement**: 15 million-fold (1.5×10⁷)

### Harmonic Basis Emergence
- **Ranks**: 21/21 (H²), 77/77 (H³) stable throughout
- **Det(Gram)**: Grew from ~10⁻²⁶ to **~1.0** by epoch 10k
- **Critical phase**: Phase 3 (5k-8k epochs) for Det transition

### Five Training Phases
1. **Neck Stability** (0-2k): Torsion 10⁻⁴ → 10⁻⁸
2. **ACyl Matching** (2k-5k): Det(Gram) 10⁻⁸ → 0.4
3. **Cohomology Refinement** (5k-8k): Det(Gram) 0.4 → 1.0
4. **Harmonic Extraction** (8k-10k): Torsion 10⁻¹¹, Det stable
5. **Calibration Fine-tune** (10k-15k): Calibration → 99.99%

---

## Plots Generated

### convergence_plots_torsion.png (4 panels)
1. Torsion closure (dφ) vs epoch (log scale)
2. Torsion coclosure (d★φ) vs epoch (log scale)
3. Regional torsion (M₁/Neck/M₂) vs epoch
4. Neck smoothness penalty vs epoch

### convergence_plots_cohomology.png (4 panels)
1. Harmonic basis ranks (H², H³) vs epoch
2. Gram determinants vs epoch (log scale)
3. Gram losses vs epoch (log scale)
4. Harmonic penalty vs epoch

### convergence_plots_loss.png (2 panels)
1. Total loss vs epoch (showing explosion at epoch ~10k)
2. Calibration accuracy vs epoch

**Note**: Loss exploded to ~10⁸ in Phase 5 due to adaptive scheduler, but **geometric metrics remained excellent**.

---

## Statistics Files

### phase_breakdown.json
Contains for each phase:
- Torsion closure/coclosure start/end values
- Reduction factors
- Mean and std over phase
- Rank stability
- Gram determinant evolution
- Calibration progression

### training_summary.json
Contains:
- Final results (epoch 14999)
- Improvement factors (initial → final)
- Notable events (when ranks stabilized, loss explosion epoch)
- Targets achieved (all ✓)

---

## Understanding the Plots

### Phase Backgrounds
All plots show colored backgrounds indicating the 5 training phases:
- Blue: Phase 1 (Neck Stability)
- Orange: Phase 2 (ACyl Matching)
- Green: Phase 3 (Cohomology Refinement)
- Red: Phase 4 (Harmonic Extraction)
- Purple: Phase 5 (Calibration Fine-tune)

### Key Transitions
- **Epoch 5000**: Grid resolution 8 → 10, Det(Gram) starts growing rapidly
- **Epoch 8000**: Grid resolution 10 → 10, Det(Gram) approaches 1.0
- **Epoch 10000**: Grid resolution 10 → 12, loss explosion begins (metrics OK)

---

## Interpretation

### Excellent Convergence
- Torsion reduced to **machine precision** (10⁻¹¹)
- Harmonic bases **orthonormal** (Det ≈ 1.0)
- Calibration **99.99%** accurate
- TCS structure **smooth** (neck torsion ~10⁻¹⁷)

### Loss Explosion (Phase 5)
- Adaptive scheduler boosted weights exponentially
- Total loss grew to ~10⁸-10⁹
- **Geometric metrics unaffected**
- Lesson: Cap adaptive boosting or disable when targets met

### Phase 3 Critical
- Det(Gram) transition from <0.5 to ~1.0
- Established orthonormal harmonic bases
- Most important phase for cohomology structure

---

## For Publications

Use these plots in papers:
1. **convergence_plots_torsion.png** - Shows torsion-free achievement
2. **convergence_plots_cohomology.png** - Shows harmonic basis extraction
3. **phase_breakdown.json** - For tables of phase statistics

**All plots**: 300 DPI, publication-ready

---

## Dependencies

```python
numpy
pandas
matplotlib
seaborn
json
```

All standard scientific Python libraries.

---

## See Also

- **Parent**: `../RESULTS_REPORT.md` Section 2 (Convergence Analysis)
- **Methodology**: `../METHODOLOGY.md` Section 4 (Curriculum Learning)
- **Raw data**: `../training_history.json` (5.5 MB, 15k epochs)

---

*Generated as part of K₇ TCS v1.0 results package*
*Version: 1.0 | Date: 2025-11-19*
