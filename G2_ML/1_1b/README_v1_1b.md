# G2_ML v1.1b: Complete GIFT 2.1 RG Flow Implementation

## Overview

Version 1.1b implements the complete GIFT 2.1 Renormalization Group flow formula to address the 99.16% RG flow error observed in v1.1a, while preserving the excellent torsion (1.68% error) and geometry (0.00007% error) results.

### Version History

- **v1.0f**: Torsion minimization → 94.9% torsion error, Yukawa norm ~10⁻¹⁰
- **v1.1a**: Torsion targeting → 1.68% torsion error ✓, but 99.16% RG flow error ✗
- **v1.1b**: Complete GIFT 2.1 RG flow → Target < 20% RG flow error while maintaining torsion and geometry

## GIFT 2.1 RG Flow Formula

### Complete Formula

The GIFT 2.1 framework specifies the RG flow as:

```
ℱ_RG = A·(∇·T) + B·|T|² + C·(∂ε g) + D·fractality(T)

Δα = ∫₀^λₘₐₓ ℱ_RG dλ
```

where:
- **A·(∇·T)**: Torsion divergence term (geometric flow)
- **B·|T|²**: Torsion norm term (already in v1.1a)
- **C·(∂ε g)**: Metric scale variation (dimensional running)
- **D·fractality(T)**: Multi-scale structure (fractal geometry)

### Physical Interpretation

Each component represents a different aspect of the geometric RG flow:

1. **Divergence (∇·T)**: Measures how torsion "spreads" through the manifold. Non-zero divergence indicates sources/sinks of torsion that affect coupling constants.

2. **Norm (|T|²)**: Direct contribution from torsion magnitude. Strong torsion leads to stronger running.

3. **Epsilon Variation (∂ε g)**: Captures how metric changes with RG scale ε. Related to dimensional transmutation and symmetry breaking scale ε₀ = 1/8.

4. **Fractality**: Multi-scale structure of torsion field. Fractal patterns enhance RG flow through scale-invariant contributions.

## Implementation Details

### 1. Torsion Divergence

**File Location**: Notebook section "NEW v1.1b: GIFT 2.1 RG Flow Components"

```python
def compute_torsion_divergence(torsion, phi_net, coords, dx=1.0/16):
    """
    Compute ∇·T = ∂_i T^i_jk using centered finite differences.
    """
```

**Method**: Centered finite differences on 7D grid with periodic boundary conditions.

**Numerical Considerations**:
- Grid spacing: dx = 1/16 (matches training grid)
- Approximation: Uses adjacent batch elements for derivatives
- Normalization: Divided by 7×7 = 49 for proper scaling

**Validation**: Should be ~0 for constant torsion fields (test included).

### 2. Epsilon Derivative

```python
def compute_epsilon_derivative(phi_net, coords, geometry, epsilon_0=0.125):
    """
    Compute ∂ε g measuring metric scale variation.
    Returns [trace_var, det_var, norm_var].
    """
```

**Method**: Numerical derivative via coordinate rescaling.

**GIFT Scale**: ε₀ = 1/8 corresponds to the U(1) symmetry breaking scale in GIFT.

**Output**: Three components capturing different metric variations:
- Trace variation: Volume element change
- Determinant variation: Overall metric rescaling
- Norm variation: Shape distortion

### 3. Fractality Index

```python
def compute_fractality_fourier(torsion):
    """
    Fourier power spectrum slope P(k) ~ k^(-α).
    Returns normalized α ∈ [0, 1].
    """
```

**Method**: Power spectrum analysis in Fourier space.

**Physical Meaning**:
- α = 0: White noise (no fractality)
- α ∈ [0.3, 0.7]: Moderate fractal structure
- α > 0.8: Strong scale-invariant pattern

**Validation**: White noise should give ~0, Brownian motion ~0.33.

### 4. RGFlowGIFT Class

Main calculator integrating all components:

```python
class RGFlowGIFT:
    def __init__(self, config):
        self.A = -4.68   # Divergence coefficient
        self.B = 15.17   # Norm coefficient (from v1.1a)
        self.C = [10.0, 5.0, 1.0]  # Epsilon derivative weights
        self.D = 2.5     # Fractality coefficient
    
    def compute_delta_alpha(self, phi_net, geometry, coords, torsion, epoch):
        """Compute Δα with all four GIFT 2.1 components."""
```

**Calibration**: Coefficients A, B, C, D are calibrated at epoch 5000 (advanced from 6000 in v1.1a).

**Component Toggles**: Each term can be enabled/disabled via config for ablation studies.

### 5. Adaptive Geodesic Frequency

```python
if config['rg_flow']['adaptive_frequency']:
    T_magnitude = torch.norm(dphi)
    adaptive_factor = 1.0 + 0.5 * torch.tanh(T_magnitude / 0.01)
    freq = torch.clamp(base_freq * adaptive_factor, 0.1, 0.8)
```

**Purpose**: Sample RG flow more frequently where torsion is large (important regions).

**Base Frequency**: 0.3 (increased from 0.1 in v1.1a)

**Adaptive Range**: [0.1, 0.8] to balance exploration and computational cost.

### 6. Smart Early Stopping

```python
class SmartEarlyStopping:
    def check(self, epoch, losses, metrics, config):
        # NaN detection
        # Minimum epoch enforcement
        # Criteria satisfaction check
        # Patience mechanism
```

**Key Features**:
- NaN/Inf detection with immediate stop
- Minimum epochs per phase (prevents premature stopping on RG flow)
- Phase-specific criteria (torsion, geometry, RG flow)
- Patience counter (requires sustained convergence)

**Phase 5 Criteria**:
- `min_epochs`: 1000 (ensures RG flow has time to converge)
- `rg_flow_delta`: 0.05 (5% tolerance on Δα)
- `torsion_target_reached`: Within 20% of target
- `patience`: 500 epochs

### 7. RG Flow Monitoring

```python
class RGFlowMonitor:
    def log(self, epoch, rg_components, metrics):
        """Log A, B, C, D components separately."""
```

**Output**: `rg_flow_log.csv` with columns:
- `epoch`, `delta_alpha`, `A_div`, `B_norm`, `C_eps`, `D_frac`
- `div_T`, `frac_idx`, `torsion_norm`, `det_g`

**Usage**: Analyze which components contribute most to RG flow.

## Configuration Changes

### Updated Parameters

```python
CONFIG = {
    'n_epochs_per_phase': 2000,  # Extended from 1500
    'min_total_epochs': 7500,    # NEW: Minimum total training
    'checkpoint_dir': 'checkpoints_v1_1b',
    
    'phases': {
        4: {'rg_flow': 0.5},  # Earlier introduction (was 0.1)
        5: {'rg_flow': 3.0},  # Stronger weight (was 1.0)
    },
    
    'rg_flow': {
        'lambda_max': 39.44,
        'target_delta_alpha': -0.9,
        'n_integration_steps': 100,
        'geodesic_batch_freq_base': 0.3,  # Increased from 0.1
        'calibration_epoch': 5000,  # Advanced from 6000
        'adaptive_frequency': True,  # NEW
        'monitor_components': True,  # NEW
        'enable_divergence': True,   # NEW
        'enable_epsilon_var': True,  # NEW
        'enable_fractality': True,   # NEW
    }
}
```

## Expected Results

### Success Criteria

Based on v1.1a baseline:

| Metric | v1.1a Result | v1.1b Target | Status |
|--------|--------------|--------------|--------|
| Torsion error | 1.68% | < 5% | Maintain ✓ |
| Geometry (det g) | 0.00007% | < 0.001% | Maintain ✓ |
| RG Flow error | 99.16% | < 20% | **IMPROVE** |
| Yukawa norm | 5.87×10⁻¹⁰ | > 10⁻⁵ | Optional |

### Component Balance

Expected contribution of each GIFT 2.1 term:

- **B (norm)**: Dominant (~60-70% of total)
- **A (divergence)**: Moderate (~15-25%)
- **C (epsilon)**: Moderate (~10-20%)
- **D (fractality)**: Small (~5-15%)

No single component should be < 1% of total (indicates it's not contributing).

### Validation Output Example

```
=== RG FLOW VALIDATION (GIFT 2.1 Components) ===
Target Δα: -0.9000
Predicted Δα: -0.7200  (20% error ✓)

Component Breakdown:
  A (∇·T):       -0.1234
  B (|T|²):      -0.5123
  C (∂ε g):      -0.0678
  D (fractality): -0.0165
  Total Δα:      -0.7200
```

## Training Procedure

### Phase Progression

1. **Phase 1-3** (Epochs 0-6000): Establish geometry and torsion
   - RG flow weight = 0
   - Focus on TCS construction, ACyl matching, cohomology

2. **Phase 4** (Epochs 6000-8000): Introduce RG flow
   - RG flow weight = 0.5
   - Adaptive frequency begins
   - Monitor component contributions

3. **Epoch 5000**: Calibration Event
   - Update RGFlowGIFT coefficients
   - Freeze α⁻¹ functional if needed
   - Log calibration results

4. **Phase 5** (Epochs 8000+): Full RG flow optimization
   - RG flow weight = 3.0
   - All GIFT 2.1 components active
   - Smart early stopping monitors convergence

### Estimated Training Time

- On GPU (NVIDIA A100): ~8-12 hours for 7500 epochs
- On GPU (NVIDIA V100): ~12-18 hours
- On CPU: Not recommended (>100 hours)

### Checkpointing

Automatic checkpoints saved:
- Every 500 epochs
- At phase boundaries
- On early stop
- On NaN detection (before crash)

Resume from checkpoint:
```python
checkpoint = torch.load('checkpoints_v1_1b/checkpoint_latest.pt')
phi_net.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
```

## Ablation Studies

To test individual components, disable others in config:

```python
CONFIG['rg_flow'] = {
    'enable_divergence': True,   # Test A term alone
    'enable_epsilon_var': False,
    'enable_fractality': False,
}
```

Recommended ablations:
1. B only (baseline, should match v1.1a)
2. A+B (add divergence)
3. A+B+C (add epsilon)
4. A+B+C+D (full GIFT 2.1)

## Troubleshooting

### Issue: NaN in training

**Symptoms**: Training stops with "NaN detected"

**Solutions**:
1. Reduce learning rate: `'learning_rate': 1e-4`
2. Increase Phase 4 weight gradually: `'rg_flow': 0.2` → `0.5`
3. Disable fractality initially: `'enable_fractality': False`

### Issue: RG flow not converging

**Symptoms**: RG flow error stays > 50% after 10k epochs

**Solutions**:
1. Check component balance (look at `rg_flow_log.csv`)
2. Increase RG flow weight in Phase 5: `'rg_flow': 5.0`
3. Extend training: `'min_total_epochs': 10000`
4. Adjust calibration epoch: `'calibration_epoch': 4000`

### Issue: Torsion or geometry degraded

**Symptoms**: Torsion error > 10% or det(g) error > 0.01%

**Solutions**:
1. Reduce RG flow weight: `'rg_flow': 2.0` in Phase 5
2. Start RG flow later: Phase 4 weight = 0, Phase 5 weight = 1.0
3. Increase torsion/det weights in Phase 5
4. Check if early stopping is too aggressive

## Files Generated

### Training Outputs

- `checkpoints_v1_1b/checkpoint_latest.pt`: Latest model state
- `checkpoints_v1_1b/checkpoint_phase{N}_epoch_{M}.pt`: Phase checkpoints
- `training_history.csv`: Complete loss history (all epochs)
- `rg_flow_log.csv`: RG flow component breakdown (Phases 4-5)

### Validation Outputs

- `metadata.json`: Configuration and validation results
- `harmonic_2forms.npy`: H² basis (21 modes)
- `harmonic_3forms.npy`: H³ basis (77 modes)
- `yukawa_tensor.npy`: Coupling tensor (21×21×77)
- `phi_samples.npy`: φ field samples (5000 points)
- `metric_samples.npy`: Metric samples (5000 points)

### Monitoring

- Console output: Real-time progress bars and metrics
- Epoch 100, 200, ... : Periodic status updates
- Early stop events: Logged with criteria details
- Calibration events: Coefficient updates logged

## Comparison with v1.1a

### What Changed

1. **RG Flow Formula**: B·|T|² → A·(∇·T) + B·|T|² + C·(∂ε g) + D·frac(T)
2. **Adaptive Sampling**: Fixed 10% → Adaptive 10-80% based on torsion
3. **Early Stopping**: Simple patience → Smart with NaN detection and min epochs
4. **Monitoring**: Basic loss logging → Detailed component breakdown
5. **Training Length**: 7500 epochs → 7500+ with min_total_epochs check
6. **Calibration**: Epoch 6000 → Epoch 5000 (earlier)

### What Stayed the Same

- TCS geometry (extended neck, σ = 5.0)
- φ-network architecture (866k parameters)
- Torsion targeting (not minimization)
- Phase curriculum (5 phases)
- Geodesic integrator (RK4 with Christoffel symbols)
- AlphaInverseFunctional (A, B coefficients)

## Future Work (v1.1c+)

Potential improvements for future versions:

1. **Wavelet Fractality**: Replace Fourier spectrum with wavelet analysis for better multi-scale characterization

2. **Learned Coefficients**: Make A, B, C, D fully learnable instead of calibrated at single epoch

3. **Higher-Order Divergence**: Include ∇²T and mixed derivatives

4. **Gauge Field Corrections**: Add Wilson loop corrections to RG flow

5. **Stochastic Geodesics**: Multiple geodesic paths instead of single trajectory

6. **Phase Transition Detection**: Automatically detect and handle phase transitions in RG flow

## References

- GIFT Framework v2.1 Specification
- v1.1a Results: `G2_ML/1_1a/metadata.json`
- Original TCS Construction: v1.0f documentation
- RG Flow Theory: GIFT whitepaper Section 4.2

## Citation

If you use this code, please cite:

```
G2_ML v1.1b: Complete GIFT 2.1 RG Flow Implementation
K₇ manifold with holonomy reduction
https://github.com/gift-framework/GIFT/tree/main/G2_ML/1_1b
```

## License

Part of the GIFT (Geometric Information Field Theory) framework.

---

**Last Updated**: 2024  
**Author**: GIFT Development Team  
**Status**: Complete implementation, ready for training

