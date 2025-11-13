# G2 Metric Learning Framework Completion Plan

**Allocated budget**: $300 | **Estimated timeline**: 2-3 days

## Current Status

The G2 metric learning framework includes versions 0.1 through 0.7. Version 0.7 represents the most recent implementation with b2=21 validation completed. Three components require completion:

1. Harmonic 3-forms extraction (b3=77)
2. Yukawa coupling tensor computation
3. Architecture hyperparameter optimization

## Completion Objectives

### 1. Harmonic 3-Forms Extraction (b3=77)

**Current state**: Mentioned as pending in version 0.5
**Objective**: Extract 77 linearly independent harmonic 3-forms

**Technical approach**:
- Extend harmonic network from 21 to 77 outputs
- Network architecture: HarmonicB3Network with input dimension 7, output dimension 77×35
  - 35 components per 3-form in 7 dimensions
  - Approximately 30 million additional parameters
- Loss function components:
  ```python
  L_b3 = L_orthonormality + L_closedness + L_coclosedness
  ```
- Training duration: Estimated 3× longer than b2 training (approximately 20 hours per run on A100)

**Implementation steps**:
1. Adapt version 0.7 notebook structure
2. Implement harmonic_3forms_network module
3. Add b3 Gram matrix validation
4. Train with curriculum schedule similar to b2
5. Validate: det(Gram_b3) approximately 1, eigenvalues in range [0.9, 1.1]

**Estimated cost**: 5-10 training runs at $30 per run = $150-300

**Deliverables**:
- `G2_ML/0.8/Complete_G2_Metric_Training_v0_8_B3_77.ipynb`
- `b3_extraction_results.json` containing 77×77 Gram matrix
- `b3_forms.npy` (77 3-forms stored as numpy array)

### 2. Yukawa Coupling Computation

**Current state**: Marked as pending in version 0.5
**Objective**: Compute Yukawa tensor Y_αβγ with dimensions 21×21×21

**Physical interpretation**:
- Y_αβγ = ∫_{K7} ω_α ∧ ω_β ∧ ω_γ (triple wedge product integral)
- Relates to fermion mass hierarchies in GIFT framework
- Connection to Standard Model Yukawa matrices via dimensional reduction

**Technical approach**:
```python
def compute_yukawa_tensor(harmonic_2forms, manifold):
    Y = np.zeros((21, 21, 21))
    for alpha in range(21):
        for beta in range(21):
            for gamma in range(21):
                wedge_product = omega[alpha] ^ omega[beta] ^ omega[gamma]
                Y[alpha, beta, gamma] = manifold.integrate(wedge_product)
    return Y
```

**Computational requirements**:
- 21³ = 9,261 triple wedge products
- Each requires 7-dimensional integration over K7
- Monte Carlo method with 100,000 samples per integral
- Estimated runtime: 6-12 hours on GPU

**Estimated cost**: 2-3 runs at $20 per run = $40-60

**Deliverables**:
- `yukawa_tensor.npy` (21×21×21 array)
- `yukawa_analysis.json` (structural properties, eigenvalues, symmetries)
- Visualization of predicted mass hierarchy

### 3. Architecture Optimization

**Current configuration**: Version 0.7 uses [384, 384, 256] for phi network
**Objective**: Determine optimal architecture via systematic search

**Hyperparameter search space**:
- Phi network depth: [2, 3, 4] layers
- Phi network width: [256, 384, 512, 768]
- Harmonic network hidden dimension: [64, 128, 256]
- Fourier modes: [24, 32, 48, 64]
- Learning rate: [5×10⁻⁵, 1×10⁻⁴, 5×10⁻⁴]
- Batch size: [1024, 2048, 4096]

**Search strategy**:
1. Coarse grid: 48 configurations, 2 hours each = 96 hours total
2. Fine grid: Top 5 configurations with 20 variations, 6 hours each = 120 hours
3. Final validation: 3 random seeds × 3 configurations, 10 hours each = 90 hours

**Budget estimates**:
- Coarse search: 96 hours on A100 at $2.50/hour = $240
- Fine search: 120 hours at $2.50/hour = $300
- Complete search: Approximately $540

**Recommended variants**:
- Quick version ($100): 20 configurations, 2 hours each = 40 hours
- Standard version ($200): 40 configurations plus top-10 refinement
- Complete version ($500): Full grid search as outlined above

**Deliverables**:
- `architecture_search_results.csv` (all configurations with metrics)
- `best_architecture.json` (optimal hyperparameters)
- Trained model using best architecture

## Implementation Plans

### Plan A: Essential Completion ($150)
- b3=77 extraction only (5 runs at $30)
- Framework completion: 90% to 95%
- Publications: 1 paper on b3 extraction methodology

### Plan B: Core Completion ($250)
- b3=77 extraction ($150)
- Yukawa coupling computation ($50)
- Quick architecture search ($50)
- Framework completion: 95% to 98%
- Publications: 2 papers (b3 extraction, Yukawa couplings)

### Plan C: Complete Implementation ($300)
- b3=77 extraction ($150)
- Yukawa coupling computation ($60)
- Standard architecture search ($90)
- Framework completion: 98% to 100%
- Publications: 3 papers (b3, Yukawa, optimization)

**Recommendation**: Plan C provides complete framework functionality.

## Implementation Commands

### b3=77 Extraction
```bash
mkdir -p G2_ML/0.8
cp G2_ML/0.7/Complete_G2_Metric_Training_v0_7.ipynb \
   G2_ML/0.8/Complete_G2_Metric_Training_v0_8_B3_77.ipynb

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
| 1 | b3 extraction (3 runs) | $90 | b3_forms.npy |
| 1-2 | b3 refinement (2 runs) | $60 | Validated b3=77 |
| 2 | Yukawa computation | $60 | yukawa_tensor.npy |
| 2-3 | Architecture search | $90 | best_architecture.json |

**Total**: 2-3 days, $300

## Success Criteria

- b3=77 forms extracted with det(Gram) in range [0.9, 1.1]
- All 77 eigenvalues positive (λi > 0.5)
- Yukawa tensor computed with Monte Carlo uncertainty below 1%
- Architecture search identifies configuration with torsion improvement
- All results documented in G2_ML/0.8/

## Future Directions

Following completion:

1. Phenomenology: Connect Yukawa couplings to Standard Model fermion masses
2. Publication: Complete G2 metric learning on K7 with b2=21, b3=77
3. Extensions: Time-dependent metrics, temperature evolution
4. Applications: Use trained metrics for GIFT prediction refinement

---

**Prepared**: 2025-11-13
**Allocated budget**: $300
**Status**: Ready for execution
**Framework version**: GIFT v2.0
