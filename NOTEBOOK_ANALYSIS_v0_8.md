# Complete_G2_Metric_Training_v0_8.ipynb - Comprehensive Analysis

## 1. CURRENT ARCHITECTURE

### Geometric Construction: TCS Neck Manifold
The notebook implements a **TCS-inspired Neck Geometry**:
```
Manifold:  [−T, T] × (S¹)² × T⁴  (7D)
```

**Components:**
- **t-direction**: Non-periodic interval [−T, T]
  - T_neck = τ × 2π ≈ 24.48
  - T_boundary = T_neck / 2 ≈ 12.24
  
- **Fiber (S¹ × S¹)**: Two periodic circles
  - Radius: R_fiber = 2π
  - Represents gluing transition

- **K3-like (T⁴)**: Four-torus base
  - Radii: {2π, 2π, 2π/φ, 2π/φ}
  - φ = 1.618... (golden ratio) creates hierarchy
  - Hints at K3 complex structure

**GIFT Parameters Embedded:**
- τ = 10416/2673 ≈ 3.897 (neck modulus)
- ξ = 5π/16 ≈ 0.982 rad (HyperKähler rotation)
- γ = 511/884 ≈ 0.578 (exponential decay rate)
- φ = golden ratio

**ACyl Boundary Structure (NEW v0.7):**
- Left ACyl zone: t ∈ [−12.24, −9.24]
- Central region: t ∈ [−9.24, +9.24]
- Right ACyl zone: t ∈ [+9.24, +12.24]
- ACyl width: 3.0 units
- Transition smoothness: exp(−γ|t|/T) with sigmoid smoothing

---

## 2. EXISTING DIFFERENTIAL OPERATORS

### Implemented Operators:

#### a) **Exterior Derivative (d)**
- **Method**: Finite differences via automatic differentiation
- **Implementation**: `compute_torsion_simplified()`, `compute_torsion_loss_enhanced()`
- **Approximation**: dφ ≈ ||∇φ|| via gradient norms
- **Scope**: Samples ~10 components of 35-component 3-form for efficiency
- **Status**: Simplified approximation, not rigorous wedge product

#### b) **Hodge Star (*)**
- **Method**: Metric-weighted scaling
- **Function**: `hodge_star_rigorous()`
- **Formula**: 
  ```
  *φ = φ × √det(g) × (Tr(g) / 7) / ||·||
  ```
- **Notes**: 
  - For 3-form → 4-form on 7D
  - Uses metric tensor for proper weighting
  - Normalized to preserve G₂ structure

#### c) **Codifferential (δ)**
- **Status**: NOT explicitly implemented
- **Workaround**: Torsion-free condition (dφ = 0) enforced directly via loss
- **Missing**: δφ computation and cohomology with respect to (d, δ)

#### d) **Laplacian (Δ = dδ + δd)**
- **Status**: NOT implemented
- **Critical Gap**: Would be needed for rigorous Hodge decomposition

#### e) **Ricci Curvature (NEW v0.8)**
- **Method**: Finite differences
- **Steps**:
  1. Christoffel symbols via finite differences: Γ^k_ij = 0.5 g^kl (∂_i g_jl + ...)
  2. Riemann tensor from Christoffel: R^l_ijk = ∂_j Γ^l_ik - ∂_k Γ^l_ij + Γ·Γ
  3. Ricci tensor by contraction: R_ij = R^k_ikj
  4. Ricci scalar: R = g^ij R_ij
- **Validation**: Checks for Ricci-flatness (||R_ij|| < 1e-3)

#### f) **Christoffel Symbols**
- **Formula**: Γ^k_ij = 0.5 g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
- **Method**: Finite difference step ε = 1e-4
- **Purpose**: Foundation for curvature computation

### Critical Limitations:
- **No proper wedge products**: Uses norms instead of actual ∧
- **No explicit codifferential**: Only approximates via gradient decay
- **No harmonic analysis**: Laplacian not computed
- **Finite difference methods**: Numerically unstable for higher derivatives
- **Single epsilon**: Uses ε = 1e-4 without adaptive refinement

---

## 3. METRIC CONSTRUCTION

### From 35 Components to 7×7 SPD Metric
**Two approaches implemented:**

#### Approach A: `metric_from_phi_robust()`
```python
1. Initialize metric from φ's 35 components
   g[i,j] = φ[idx] × 0.1 + δ_ij
   
2. Add regularization: g → g + 0.15 × I

3. Symmetrize: g → 0.5(g + g^T)

4. SPD projection via eigenvalue clamping:
   - Compute eigh(g)
   - Clamp eigenvalues: λ_min = 0.3 (CRITICAL)
   - If condition_number > 100, clamp to 0.5
   
5. Reconstruct: g = V Λ V^T

6. Volume normalization: g → g / vol^(2/7)
```

**Robustness measures:**
- Min eigenvalue floor: 0.3 (prevents singularity)
- Condition number monitoring (ratio > 100 triggers alert)
- Fallback: If all fails, g + 0.5I

**Issues addressed in v0.6c:**
- Increased regularization from 0.1 → 0.15
- Raised minimum eigenvalue from 0.1 → 0.3
- Added condition number check

#### Approach B: `MetricNetwork` (NEW v0.7)
- Direct 28-coefficient prediction from coordinates
- Diagonal entries: exp(diag_raw) + 0.1 (ensures positivity)
- Upper triangular: 21 free parameters (symmetric)
- Boundary modulation via sigmoid
- SPD enforcement: Add 0.01 × I

---

## 4. COHOMOLOGY COMPUTATIONS

### b₂ = 21 (Harmonic 2-Forms)

**Method:**
1. Network generates 21 harmonic 2-forms via `Harmonic2FormsNetwork_TCS`
   - 21 distinct networks with different initializations
   - Each outputs 21-dimensional form
   - Hidden dims: [128, 128]

2. **Gram Matrix**: G_αβ = ∫ h_α ∧ *h_β
   - Inner product via metric
   - Volume normalization: vol = √|det(g)|
   - Normalization: diagonal elements → 1

3. **Validation**:
   - det(G_{21}) ≈ 1 (target: 0.995)
   - ||G - I|| < 0.2 (orthonormality error)
   - Eigenvalues in [0.85, 1.15]: minimum 18/21
   - det error < 0.3, gram error < 0.2 → PASS

4. **Output**: Eigenvalue spectrum, Gram matrix heatmap, eigenvector basis

---

### b₃ = 77 (Harmonic 3-Forms)

**Method: Spectral FFT Extraction (v0.7 - Grid=12)**

**STEP 1-2: Grid Construction & φ Computation**
- Grid resolution: 12⁷ = 35,831,808 points
- Memory optimization: Process t-slices sequentially
- Batch size: 8,192 points/batch
- φ network: 35 components per point
- Output shape: [12, 12, 12, 12, 12, 12, 12, 35]

**STEP 3: FFT (Component-wise Streaming)**
- Apply FFT in all 7 directions
- Process each component separately (memory efficient)
- Output: 35 complex coefficient arrays, shape [12⁷]

**STEP 4: Importance Scoring via GIFT Hierarchy**
- Compute mode energies: E[k] = Σ_comp |φ̂_comp[k]|²
- Select top 250 candidates by energy
- Score range monitoring

**STEP 5: Coefficient Extraction**
- For each top mode: extract coefficients from all 35 components
- Result: 250 × 35 coefficient matrix

**STEP 6: Sequential Gram-Schmidt Orthogonalization**
```
for each candidate in [1..250]:
    if no selected forms yet:
        add candidate
    else:
        orthogonalize: c_orth = c - Σ <c, s_i> s_i
        if ||c_orth|| > 1e-3 (linear independence threshold):
            normalize and add
        stop when 77 forms extracted
```

**Output**:
- 77 orthonormal harmonic 3-forms
- Gram matrix (should be identity)
- Eigenvalue spectrum

**Validation:**
- Number of forms extracted: 75-77 (target: 77)
- Eigenvalues: [0.05, ∞) threshold, expecting ≈ 77 non-zero

---

## 5. VALIDATION METRICS

### Training Validation

| Metric | Computation | Target | v0.8 Status |
|--------|-------------|--------|-------------|
| **Torsion Loss** | ||∇φ||, gradient components | → 0 | Ramped 0.1 → 10 |
| **det(Gram_{21})** | det(G) | ≈ 1 | Target: 0.995 |
| **Orthogonality** | ||G - I|| | < 0.2 | Monitored continuously |
| **Volume** | ∫√det(g) dV | (2π)⁷ ≈ 9488.53 | NEW: explicit loss |
| **ACyl Decay** | exp(-γ\|t\|/T) | Matches learned φ | NEW: C² continuity check |
| **Boundary** | torsion at t ≈ ±T | → 0 | Smooth transition |

### Post-Training Validation (NEW v0.8)

#### a) **Ricci Curvature (NEW)**
```
Metric:           ||R_ij||_L²  < 1e-3
Ricci-flat check: Is R_ij ≈ 0?
Frobenius norm:   Σ_ij R_ij²
Ricci scalar:     R = g^ij R_ij
```
- Threshold for "Ricci-flat": 1e-3 (relaxed for numeric computation)
- Expected: Near-zero for G₂ manifold

#### b) **Cohomology Structure (NEW)**
- **b₂ rank check**: eigenvalues > 0.1, expect 21
- **b₃ rank check**: eigenvalues > 0.05, expect 75-77
- **Orthonormality**: ||G - I||_F < 0.5 (b₂), < 1.0 (b₃)
- **Euler characteristic**: χ = 1 - 0 + 21 - 77 + 77 - 21 + 0 - 1 = 0 (correct)

#### c) **Mesh Convergence (NEW)**
- Compare torsion across multiple resolutions
- Measure coefficient of variation (CV) in torsion

#### d) **Reproducibility Test (NEW)**
- Multiple training runs with different seeds
- Measure consistency of torsion loss
- CV should be < 5%

#### e) **Exponential Decay Validation**
- Expected: φ_amplitude ∝ exp(-γ|t|/T)
- Check: Match between learned φ and theoretical decay
- Plot: Overlay of actual vs expected

#### f) **Yukawa Coupling (v0.7)**
- Compute Y_αβγ = ∫ h_α ∧ h_β ∧ h_γ
- Matrix size: b₃³ = 77³ (huge!)
- Non-zero entries: Should be sparse per G₂ geometry

---

## 6. NEURAL NETWORK ARCHITECTURE

### Network Components:

#### a) **G2PhiNetwork_TCS** (φ Network)
```
Input:    Coordinates (batch, 7)
          ↓
Fourier Encoding  → High-dimensional features
          ↓
MLP:      Dense[256] → SiLU → LayerNorm
          Dense[256] → SiLU → LayerNorm
          Dense[128] → SiLU → LayerNorm
          ↓
Output:   35 components (3-form φ)
          ↓
Normalization: φ → φ × √7 / ||φ||
Boundary decay: φ → φ × (1 - decay × 0.5)
```

**Parameters**: ~180k

**Initialization**: Small output weights (0.01) to start from identity

#### b) **MetricNetwork** (NEW v0.7)
```
Input:    Coordinates (batch, 7)
          ↓
Fourier Encoding
          ↓
Deep MLP: Dense[512] → SiLU → LayerNorm
          Dense[512] → SiLU → LayerNorm
          Dense[256] → SiLU → LayerNorm
          Dense[256] → SiLU → LayerNorm
          Dense[128] → SiLU → LayerNorm
          ↓
Output:   28 coefficients
          - 7 diagonal (→ exp for positivity)
          - 21 upper triangular (symmetric)
          ↓
Boundary modulation: coeffs × sigmoid(10(1 - decay))
          ↓
SPD Construction: Diag + Symm + regularization
```

**Parameters**: ~800k

**Key innovation**: Directly learns metric structure instead of deriving from φ

#### c) **BoundaryNetwork** (NEW v0.8 - FIXED v0.7 U-shape bug)
```
Input:    Coordinates (batch, 7)
          ↓
Extract:  t = coords[:, 0]  (neck parameter)
          ↓
Compute:  t_norm = |t| / T_neck
          decay = exp(-γ_eff × t_norm)  [FIXED: from CENTER not boundaries]
          ↓
Smooth:   smooth = sigmoid(5.0 × (0.5 - t_norm))
          ↓
Combine:  boundary_factor = smooth + (1-smooth) × decay
          result = 1 - boundary_factor  [0 at center, 1 at boundaries]
          ↓
Output:   Boundary transition factors (batch,)
```

**Parameters**: 2 (gamma_offset, amplitude)

**Critical v0.8 fix**: 
- v0.7 bug: exp(-γ × dist_from_boundary) → U-shape artifact
- v0.8 fix: exp(-γ × |t|/T) from center → proper exponential decay

#### d) **Harmonic2FormsNetwork_TCS**
```
21 independent networks:
Each:  Fourier Encode → Dense[128] → SiLU → Dropout(0.1)
                     → Dense[128] → SiLU → Dropout(0.1)
                     → Dense[21] → form

Initialization: DISTINCT per network
  - Seed: 47 + form_idx × 100
  - Xavier gain: 0.5 + form_idx × 0.05
  - Bias: 0.01 × form_idx
  - Per-form noise: 0.01 × (form_idx + 1) / 21

Output: (batch, 21, 21) forms
```

**Parameters**: 21 × (~50k each) ≈ ~1.05M

**Purpose**: Extract linearly independent harmonic 2-forms

---

### Fourier Encoding

```python
Input: coords (batch, 7)

For each mode (n, m₁, ..., m₆):
    freq_0 = n × π / T_neck  (t-direction)
    freq_1,2 = n / R_fiber   (fiber circles)
    freq_3-6 = n / K3_radius (K3 torus)
    
Features = [sin(freq × coords), cos(freq × coords), ...]

Output: High-dim feature vector (typically 200-400 dims)
```

---

## 7. LOSS FUNCTION (Multi-Component, 4-Phase Curriculum)

### Overall Loss Structure
```
L_total = λ_torsion × L_torsion
        + λ_volume × L_volume
        + λ_ortho × L_ortho
        + λ_det × L_det
        + λ_sep × L_sep
        + λ_boundary × L_boundary
        + λ_decay × L_decay
        + λ_acyl × L_acyl
```

### Individual Loss Components

#### 1. **Torsion Loss** (dφ = 0)
```python
L_torsion = mean(||∇φ_i|| for i in [1..10])

Method: Automatic differentiation on sampled φ components
Phase 1: 0.1  → Phase 2: 2.0  → Phase 3: 5.0  → Phase 4: 10.0
```
- Gradient-aware computation via `SafeMetrics.compute_torsion_safe()`
- Uses exact computational graph (no cloning)
- CRITICAL v0.8 fix: Don't clone coords in gradient computation

#### 2. **Volume Normalization Loss** (NEW v0.7)
```python
L_volume = |V_est - V_target| / V_target

where:  V_est ≈ mean(√det(g)) × vol_factor
        V_target = (2π)⁷ ≈ 9488.53

Phase 1: 0.6  → Phase 2: 0.4  → Phase 3: 0.2  → Phase 4: 0.15
```
- Enforces proper scaling of manifold
- Monte Carlo integration approximation

#### 3. **Harmonic Orthogonality Loss**
```python
L_ortho = ||Gram - I||_F / 21

where:  Gram_αβ = (1/vol) ∫ h_α · h_β √det(g) dV

Phase 1: 6.0  → Phase 2: 3.0  → Phase 3: 1.5  → Phase 4: 1.0
```
- Per-element normalized
- Encourages Gram matrix → Identity

#### 4. **Harmonic Determinant Loss**
```python
L_det = ReLU(det - 0.995) + 0.1 × (det - 0.995)²

Phase 1: 3.0  → Phase 2: 1.5  → Phase 3: 0.8  → Phase 4: 0.5
```
- Encourages det(Gram) ≈ 1 (volume element)
- Asymmetric: allows det > 1 more easily

#### 5. **Separation Loss** (NEW v0.7)
```python
L_sep = ReLU(0.5 - (diag_mean - off_diag_mean))

where:  diag_mean = mean(diagonal elements of Gram)
        off_diag_mean = mean(absolute off-diagonal elements)

Phase 1: 2.0  → Phase 2: 1.0  → Phase 3: 0.5  → Phase 4: 0.3
```
- Diagonal should dominate off-diagonal
- Helps numerical stability

#### 6. **Boundary Loss**
```python
L_boundary = ||∇φ||_boundary + ||φ||_boundary × 0.5

where:  points near t = ±T (within 15% of T_neck)
        ∇φ computed via autodiff

Phase 1: 0.05  → Phase 2: 0.5  → Phase 3: 1.5  → Phase 4: 2.0
```
- φ should decay to ~Fano-like at boundaries
- Penalizes both large values and steep gradients

#### 7. **Exponential Decay Loss**
```python
L_decay = mean(|φ_amplitude - exp(-γ|t|/T)|)

where:  φ_amplitude = ||φ||
        γ = 0.578... from GIFT parameters

Phase 1: 0.05  → Phase 2: 0.3  → Phase 3: 0.5  → Phase 4: 0.5
```
- Enforces theoretical TCS asymptotic form
- Matches learned structure to GIFT predictions

#### 8. **ACyl Matching Loss** (NEW v0.7 - FIXED v0.8)
```python
L_acyl = L_gradient + L_continuity

where:  L_gradient = mean(||∇φ_i||) at transition points
        L_continuity = ||φ_amplitude - (1 - boundary_factor)||

Transition zone: boundary_factor ∈ [0.3, 0.7]

Phase 1: 0.0  → Phase 2: 0.1  → Phase 3: 0.6  → Phase 4: 0.8
```
- Ensures C² continuity at ACyl boundary
- Smooth transition between regions

---

### 4-Phase Training Curriculum

```
PHASE 1 (Epochs 0-2000): ESTABLISH STRUCTURE
├─ Focus: Build basic G₂ structure + harmonic forms
├─ Weight profile: 
│  ├─ torsion: 0.1 (minimal)
│  ├─ volume: 0.6 (2×)
│  ├─ harmonic: 6.0 + 3.0 + 2.0 (strong)
│  └─ acyl: 0.0
├─ Mixed precision: NO
└─ Purpose: Initial structure formation

PHASE 2 (Epochs 2000-5000): IMPOSE TORSION
├─ Focus: Ramp torsion loss 0.1 → 2.0 (20×)
├─ Weight profile:
│  ├─ torsion: 2.0 (20× increase from phase 1)
│  ├─ harmonic: Reduced (3.0, 1.5, 1.0)
│  ├─ boundary: 0.5
│  └─ acyl: 0.1 (start)
├─ Mixed precision: ENABLED (epoch 2000+)
├─ Warmup: 500 epochs
└─ Purpose: Torsion-free structure + mixed precision stability

PHASE 3 (Epochs 5000-8000): REFINE b₃ + ACyl
├─ Focus: ACyl boundaries + b₃ extraction
├─ Weight profile:
│  ├─ torsion: 5.0 (continue increase)
│  ├─ acyl: 0.6 (ramp to 2× from phase 2)
│  └─ harmonic: Further reduced
├─ Purpose: Boundary geometry + cohomology structure
└─ Grid resolution: Full 12⁷ for b₃ extraction

PHASE 4 (Epochs 8000-10000): POLISH FINAL
├─ Focus: Final balancing and convergence
├─ Weight profile:
│  ├─ torsion: 10.0 (maximum)
│  ├─ boundary: 2.0 (peak)
│  └─ volume: 0.15 (minimal)
├─ LR schedule: Cosine annealing
└─ Purpose: Convergence to optimal manifold
```

---

### Training Configuration

```python
CONFIG = {
    'epochs': 10000,
    'batch_size': 1536,
    'grad_accumulation_steps': 2,
    'effective_batch': 3072,
    
    'optimizer': 'AdamW',
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    
    'scheduler': 'cosine',
    'warmup_epochs': 500,
    
    'mixed_precision': True,
    'mixed_precision_start_epoch': 2000,
    
    'seed': 47,
    'deterministic': True,
}
```

---

## 8. KNOWN ISSUES & LIMITATIONS

### Architectural Gaps (For "Legit TCS G₂" Refactor)

| Issue | Current | Needed |
|-------|---------|--------|
| **Wedge product** | Approximate (norms only) | Rigorous ∧ product |
| **Codifferential** | None | δ operator on forms |
| **Laplacian** | None | Δ = dδ + δd |
| **Harmonic decomposition** | Implicit | Hodge: V = H ⊕ dV ⊕ δV |
| **Curvature** | Christoffel + Ricci | Riemann, sectional, scalar |
| **Torsion-free enforcement** | Loss function | Direct PDE solver |
| **Metric positivity** | Eigenvalue clamping | Manifold-constrained optimization |

### Numerical Issues

1. **Finite differences**: ε = 1e-4 fixed, no adaptivity
2. **No grid refinement**: Uses same ε for all scales
3. **Memory**: 12⁷ grid for b₃ requires careful chunking
4. **Condition numbers**: Metric may become ill-conditioned

### v0.8 Fixes Applied

- ✓ Test gradient enabled (ricci computation)
- ✓ Boundary decay: Fixed U-shape (exp from center, not boundaries)
- ✓ PDE residuals: Logged explicitly
- ✓ Grid b₃=12: Verified for 77 forms extraction
- ✓ Torsion computation: No clone() in validation (preserves gradients)

---

## 9. SUMMARY FOR REFACTOR PLANNING

### Strengths
- Complete curriculum-based training
- Dual network approach (φ + direct metric)
- Novel boundary transition modeling
- Publication-quality b₂ extraction
- Spectral b₃ via FFT (grid=12)
- Ricci flatness validation
- Multiple reproducibility tests

### Critical Gaps for "Legit TCS G₂"
1. **No rigorous differential geometry**: Operators are approximations
2. **No Hodge decomposition**: Only explicit harmonic forms, no projection
3. **Loss-based torsion**: Should use PDE solver instead
4. **Metric parameterization**: φ→metric mapping is ad-hoc
5. **No homological algebra**: Missing differential complex structure
6. **Validation only post-hoc**: Should enforce geometry during training

### Priority Refactors
1. Implement true ∧, d, δ, Δ operators
2. Add Hodge decomposition machinery
3. Use constrained optimization on metric manifold
4. Implement rigorous Ricci-flat solver
5. Add homology class tracking
6. Better numerical methods (not just finite differences)

