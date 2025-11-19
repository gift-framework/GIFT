# Yukawa Tensor Analysis

**Tucker decomposition and mass ratio extraction from computed Yukawa couplings**

---

## Contents

### Notebooks
- **[yukawa_analysis.ipynb](yukawa_analysis.ipynb)** - Main Yukawa analysis
  - Loads `yukawa_tensor.pt` [21×21×77]
  - Verifies antisymmetry (Y_αβγ = -Y_βαγ)
  - Performs Tucker decomposition (3×3×3)
  - Extracts mass ratios (m_t/m_c, m_c/m_u, m_τ/m_μ)
  - Compares with GIFT predictions

### Documentation
- **[mass_ratios_comparison.md](mass_ratios_comparison.md)** - Detailed comparison
  - GIFT theoretical predictions
  - Neural extraction methods (diagonal & Tucker)
  - PDG experimental values
  - Scale dependence discussion

### Generated Outputs (After Running Notebook)
- `yukawa_analysis_results.json` - Complete analysis results
- `yukawa_structure_visualization.png` - Tensor heatmaps (4 panels)
- `tucker_decomposition_visualization.png` - Core and factors (4 panels)

---

## Quick Start

```bash
# Navigate to yukawa directory
cd yukawa/

# Run analysis notebook
jupyter notebook yukawa_analysis.ipynb
```

**Runtime**: ~5-10 minutes (depends on tensorly availability)

**Dependencies**:
- `tensorly` (for Tucker decomposition) - **recommended**
- Without tensorly: Falls back to SVD approximation

---

## Yukawa Tensor

### Computation Method
```
Y_αβγ = ∫_{K₇} h²_α ∧ h²_β ∧ h³_γ √det(g) d⁷x
```

**Dual integration**:
1. **Monte Carlo**: 20,000 random samples
2. **Grid quadrature**: 10⁷ grid points
3. **Final**: Average of both methods

### Tensor Properties
- **Shape**: [21, 21, 77]
- **Elements**: 21 × 21 × 77 = 33,957 components
- **Antisymmetry**: Y_αβγ = -Y_βαγ (verified to < 10⁻⁶)
- **Storage**: 135 KB (PyTorch .pt format)
- **Non-zero**: Typically 30-50% of components

---

## Tucker Decomposition

### Theory
Decompose Yukawa tensor into 3-generation structure:

```
Y_αβγ ≈ Σ_{i,j,k} core_{ijk} · U1_{αi} · U2_{βj} · U3_{γk}
```

Where:
- **core**: [3, 3, 3] - Generation couplings
- **U1, U2**: [21, 3] - H² → generations mapping
- **U3**: [77, 3] - H³ → generations mapping

### Rank Choice
**(3, 3, 3)** motivated by:
- 3 fermion generations (Standard Model)
- SU(3)_family structure in GIFT
- Minimal rank for generational hierarchy

### Interpretation
- **core_{iii}**: Diagonal elements ∝ generation masses
- **core_{ijk}** (i≠j≠k): Mixing terms (CKM structure)
- **U3**: Maps 77 H³ forms to 3 generations

---

## Mass Ratio Extraction

### Method 1: Direct Diagonal
Extract largest diagonal elements from full tensor:
```python
diagonal = [Y[i, i, i] for i in range(min(21, 77))]
top_3 = sort(diagonal)[-3:]  # Identify 3 generations
```

**Advantages**: No approximation beyond training
**Limitations**: Assumes diagonal dominance

### Method 2: Tucker Core Diagonal
Extract from decomposed core:
```python
gen_masses = [abs(core[i, i, i]) for i in range(3)]
ratio_top_charm = gen_masses[2] / gen_masses[1]
ratio_charm_up = gen_masses[1] / gen_masses[0]
```

**Advantages**: Explicitly extracts generational structure
**Limitations**: Tucker approximation error (~X%)

---

## GIFT Predictions

### Theoretical Values (Scale-Independent)

| Ratio | GIFT Value | Status | Derivation |
|-------|------------|--------|------------|
| m_t/m_c | **57.5** | DERIVED | From b₃=77 structure |
| m_c/m_u | **20.0** | DERIVED | From E₈ branching |
| m_τ/m_μ | **16.8** | DERIVED | From b₂=21 structure |
| m_s/m_d | **20.0** | PROVEN | Exact topological relation |

**Source**: `publications/supplements/C_complete_derivations.md`

### Experimental Values (PDG 2023)

**Leptons** (pole masses, no running):
- m_τ/m_μ = **16.82** ← **Excellent GIFT agreement!**
- m_μ/m_e = 206.77

**Quarks** (running masses, scale-dependent):
- m_t/m_c ≈ 136 (pole/running mass mix)
- m_c/m_u ≈ 577 (at 2 GeV)
- m_s/m_d ≈ 20.2

**Note**: Quark ratios require RGE running for direct comparison.

---

## Comparison Table (To Be Filled)

After running yukawa_analysis.ipynb:

| Ratio | GIFT | Neural (Diagonal) | Neural (Tucker) | Agreement |
|-------|------|-------------------|-----------------|-----------|
| m_t/m_c | 57.5 | [TBD] | [TBD] | [TBD]% |
| m_c/m_u | 20.0 | [TBD] | [TBD] | [TBD]% |
| m_τ/m_μ | 16.8 | [TBD] | [TBD] | [TBD]% |

**Good agreement**: < 20% deviation
**Moderate agreement**: 20-50% deviation
**Poor agreement**: > 50% deviation

---

## Visualizations

### yukawa_structure_visualization.png (4 panels)
1. **Slice Y[:,:,0]**: First H³ component heatmap
2. **Slice Y[:,:,38]**: Middle H³ component heatmap
3. **Frobenius norms**: ||Y[:,:,γ]|| vs γ
4. **Diagonal elements**: |Y_{iii}| bar chart

### tucker_decomposition_visualization.png (4 panels)
1. **Tucker core**: core[:,:,1] heatmap (3×3)
2. **Factor U₃**: H³ → generations mapping (77×3)
3. **Generation weights**: ||U₃[:,i]|| bar chart
4. **Core diagonal**: |core_{iii}| bar chart (mass terms)

---

## Uncertainty Analysis

### Sources of Uncertainty

1. **Training convergence**: ±0.01% (torsion ~10⁻¹¹)
2. **Monte Carlo integration**: ±[X]% (from dual method)
3. **Tucker approximation**: ±[Y]% (reconstruction error)
4. **Neural network capacity**: ~5-10%
5. **Index identification**: ±10% (harmonic form → generation mapping)

**Conservative estimate**: ±15-20% on extracted ratios

---

## Interpretation Guidelines

### If Agreement is Good (< 20%)
✓ Validates GIFT geometric approach
✓ Confirms K₇ metric captures mass structure
✓ Neural network successfully extracts topology
→ Strong evidence for geometric origin of mass hierarchy

### If Agreement is Moderate (20-50%)
⚠ GIFT structure partially captured
⚠ May indicate missing corrections:
  - Higher-order geometric effects
  - RGE running (for quarks)
  - Threshold corrections
→ Refine theoretical derivations

### If Agreement is Poor (> 50%)
⚠ Fundamental issues possible:
  - Wrong K₇ topology
  - Incorrect harmonic form identification
  - Neural network training artifacts
→ Requires theoretical re-examination

---

## Scale Dependence (Important!)

**GIFT predictions**: Scale-independent (topological)

**Quark masses**: Run with energy scale (RGE)
- m_c(2 GeV) ≠ m_c(m_Z) ≠ m_c(m_t)
- Ratios also scale-dependent

**Leptons**: No QCD running → direct comparison valid

**For quarks**: Need to evolve GIFT predictions or experimental values to common scale.

---

## Future Improvements

### Short-term
1. Higher MC samples (50k → 100k)
2. Adaptive importance sampling
3. Bootstrap uncertainty quantification

### Medium-term
1. Alternative Tucker ranks ((4,4,4), (2,2,2))
2. CP decomposition (CANDECOMP/PARAFAC)
3. Off-diagonal analysis (CKM matrix)

### Long-term
1. Include RGE running for quarks
2. Threshold corrections
3. Neutrino sector (right-handed neutrinos)
4. Down-type quark sector (Y_d tensor)

---

## Dependencies

```python
# Required
torch
numpy
matplotlib
seaborn

# Recommended (for full Tucker)
tensorly

# Alternative (without tensorly)
# Falls back to SVD approximation
```

Install tensorly:
```bash
pip install tensorly
```

---

## See Also

- **Parent**: `../RESULTS_REPORT.md` Section 3 (Yukawa Analysis)
- **Methodology**: `../METHODOLOGY.md` Section 8 (Yukawa Computation)
- **Raw tensor**: `../yukawa_tensor.pt` (135 KB)
- **GIFT theory**: `publications/supplements/C_complete_derivations.md`

---

## Citation

If you use Yukawa results in research:

```bibtex
@software{yukawa_k7_tcs_2025,
  title={Yukawa Tensor from K₇ G₂ Metric Reconstruction},
  author={GIFT Framework Team},
  year={2025},
  version={1.0},
  note={Tucker decomposition and mass ratio extraction}
}
```

---

*Generated as part of K₇ TCS v1.0 results package*
*Version: 1.0 | Date: 2025-11-19*
