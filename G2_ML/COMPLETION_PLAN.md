# G2_ML Framework Completion Plan

**Budget: ~$300 | Timeline: 2-3 days**

## Current Status

GIFT has G2 metric learning across versions 0.1 to 0.7:
- **v0.7** (latest): Fully trained, b₂=21 validated
- **Gap 1**: b₃=77 extraction incomplete
- **Gap 2**: Yukawa couplings pending
- **Gap 3**: Architecture not fully optimized

## Completion Objectives

### 1. b₃=77 Harmonic 3-Forms Extraction

**Current**: Mentioned as "pending" in v0.5 README
**Goal**: Extract all 77 linearly independent harmonic 3-forms

**Technical approach:**
- Extend harmonic network from 21 to 77 outputs
- Network architecture: `HarmonicB3Network(input_dim=7, output_dim=77×35)`
  - 35 components per 3-form in 7D
  - ~30M additional parameters (3× increase)
- Loss function:
  ```python
  L_b3 = L_orthonormality + L_closedness + L_coclosedness
  ```
- Training time: 3× longer (~20h per run on A100)

**Execution:**
1. Copy v0.7 notebook structure
2. Add `harmonic_3forms_network` module
3. Implement b₃ Gram matrix validation
4. Train with curriculum (similar to b₂)
5. Validate: `det(Gram_b3) ≈ 1`, eigenvalues ∈ [0.9, 1.1]

**Budget**: 5-10 runs × $30/run = **$150-300**

**Deliverable**:
- `G2_ML/0.8/Complete_G2_Metric_Training_v0_8_B3_77.ipynb`
- `b3_extraction_results.json` with 77×77 Gram matrix
- `b3_forms.npy` (77 3-forms saved)

### 2. Yukawa Couplings Computation

**Current**: "Pending (run Section 8)" in v0.5
**Goal**: Compute full Yukawa tensor Y_αβγ (21×21×21)

**Physical interpretation:**
- Y_αβγ = ∫_{K₇} ω_α ∧ ω_β ∧ ω_γ (triple wedge product)
- Determines fermion mass hierarchies in GIFT framework
- Connects to SM Yukawa matrices via dimensional reduction

**Technical approach:**
```python
def compute_yukawa_tensor(harmonic_2forms, manifold):
    Y = np.zeros((21, 21, 21))
    for alpha in range(21):
        for beta in range(21):
            for gamma in range(21):
                # Wedge product
                wedge_product = omega[alpha] ^ omega[beta] ^ omega[gamma]
                # Integrate over K7
                Y[alpha, beta, gamma] = manifold.integrate(wedge_product)
    return Y
```

**Computational cost:**
- 21³ = 9,261 triple wedge products
- Each requires 7D integration over K₇
- Monte Carlo with 100k samples per integral
- ~6-12 hours on GPU

**Budget**: 2-3 runs × $20/run = **$40-60**

**Deliverable**:
- `yukawa_tensor.npy` (21×21×21 array)
- `yukawa_analysis.json` (structure, eigenvalues, symmetries)
- Visualization of mass hierarchy predictions

### 3. Architecture Optimization

**Current**: v0.7 uses [384, 384, 256] for φ network
**Goal**: Find optimal architecture via grid search

**Hyperparameter space:**
- φ network depth: [2, 3, 4] layers
- φ network width: [256, 384, 512, 768]
- Harmonic network hidden: [64, 128, 256]
- Fourier modes: [24, 32, 48, 64]
- Learning rate: [5e-5, 1e-4, 5e-4]
- Batch size: [1024, 2048, 4096]

**Grid search strategy:**
1. Coarse grid (48 configurations): 2h each = 96h total
2. Fine grid around top-5 (20 configurations): 6h each = 120h
3. Final validation (3 seeds × 3 configs): 9 × 10h = 90h

**Budget estimate:**
- Coarse: 96h on A100 @ $2.50/h = $240
- Fine: 120h @ $2.50/h = $300
- **Total: ~$540** (exceeds single-axis budget)

**Recommendation**:
- **Quick version** ($100): Sample 20 configs, 2h each = 40h
- **Medium version** ($200): 40 configs + top-10 refinement
- **Full version** ($500): Complete grid search as above

**Deliverable**:
- `architecture_search_results.csv` (all configs + metrics)
- `best_architecture.json` (optimal hyperparameters)
- Trained model with best architecture

## Execution Plans

### Plan A: Essential Completion ($150)
- b₃=77 extraction only (5 runs × $30)
- **ROI**: Framework 90% → 95% complete
- **Publications**: 1 paper on b₃ extraction

### Plan B: Core Completion ($250)
- b₃=77 extraction ($150)
- Yukawa couplings ($50)
- Quick architecture search ($50)
- **ROI**: Framework 95% → 98% complete
- **Publications**: 2 papers (b₃ + Yukawa)

### Plan C: Full Completion ($300)
- b₃=77 extraction ($150)
- Yukawa couplings ($60)
- Medium architecture search ($90)
- **ROI**: Framework 98% → 100% complete
- **Publications**: 3 papers (b₃ + Yukawa + optimization)

## Recommended: Plan C ($300)

Provides complete G2_ML framework with all features functional.

## Implementation Commands

### b₃=77 Extraction
```bash
# Create v0.8 directory
mkdir -p G2_ML/0.8
cp G2_ML/0.7/Complete_G2_Metric_Training_v0_7.ipynb G2_ML/0.8/Complete_G2_Metric_Training_v0_8_B3_77.ipynb

# Modify notebook to add harmonic_3forms_network
# Train on cloud GPU

python G2_ML/0.8/train_b3_extraction.py \
    --epochs 10000 \
    --batch-size 2048 \
    --output-dir G2_ML/0.8/b3_results
```

### Yukawa Couplings
```bash
python G2_ML/compute_yukawa.py \
    --harmonic-forms G2_ML/0.7/harmonic_network_final.pt \
    --n-samples 100000 \
    --output yukawa_tensor.npy
```

### Architecture Search
```bash
python G2_ML/architecture_search.py \
    --mode quick \
    --n-configs 20 \
    --max-hours-per-config 2 \
    --output architecture_results.csv
```

## Timeline

| Day | Task | Budget | Deliverable |
|-----|------|--------|-------------|
| 1 | b₃ extraction (3 runs) | $90 | b3_forms.npy |
| 1-2 | b₃ refinement (2 runs) | $60 | Validated b₃=77 |
| 2 | Yukawa computation | $60 | yukawa_tensor.npy |
| 2-3 | Architecture search | $90 | best_architecture.json |

**Total**: 2-3 days, **$300**

## Success Criteria

- [  ] b₃=77 forms extracted with det(Gram) ∈ [0.9, 1.1]
- [  ] All 77 eigenvalues positive (λᵢ > 0.5)
- [  ] Yukawa tensor computed with < 1% MC uncertainty
- [  ] Architecture search identifies >10% improvement in torsion
- [  ] All results published in G2_ML/0.8/

## Next Steps After Completion

1. **Phenomenology**: Connect Yukawa couplings to SM fermion masses
2. **Publication**: "Complete G2 Metric Learning on K₇ with b₂=21, b₃=77"
3. **Extensions**: Time-dependent metrics, temperature evolution
4. **Applications**: Use trained metrics for GIFT predictions refinement

---

**Created**: 2025-11-13
**Budget**: $300
**Status**: READY TO EXECUTE
**Priority**: HIGH (completes core G2_ML framework)
