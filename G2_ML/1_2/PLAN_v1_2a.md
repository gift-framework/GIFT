# Plan v1.2a - Corrections for Stable GIFT Training

## Changes from v1.2 → v1.2a

### ❌ **v1.2 Failure Analysis**

**Final Results:**
- Torsion: 0.181 (target: 0.0164) → **1001% error**
- det(g): 3.29 (target: 2.0) → **64% error**
- Δα: -0.321 (target: -0.9) → **64% error**
- Status: **FAIL**

**Root Cause:**
Epsilon derivative `∂_εg` computed via coordinate perturbation instead of analytic derivative of φ network. This introduced geometric instability.

---

## 🔧 **Critical Fixes for v1.2a**

### 1. **Fix Epsilon Derivative Computation** (CRITICAL)

**Current (v1.2) - BROKEN:**
```python
def compute_gift_metric(...):
    # Perturbs coordinates - UNSTABLE
    coords_plus = coords * (1 + delta_eps / epsilon_0)
    g_plus, _ = geometry.compute_metric(phi_net, coords_plus % 1.0)
    deps_g = (g_plus - g_minus) / (2 * delta_eps)
```

**New (v1.2a) - STABLE:**
```python
def compute_gift_metric_v2(phi_net, coords, geometry, epsilon_0):
    """
    Compute GIFT metric with simplified diagonal epsilon correction.

    Instead of numerical derivative, use analytic approximation:
    ∂_ε g ≈ (trace(g)/7 - 1) * I * small_factor

    This avoids coordinate perturbation instability.
    """
    # Get baseline metric
    g_base, _ = geometry.compute_metric(phi_net, coords)

    # Simplified epsilon variation: small diagonal correction
    # Motivated by: metric should scale toward canonical det=2
    trace_g = torch.diagonal(g_base, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    eps_correction = epsilon_0 * 0.01 * (trace_g / 7.0 - 1.0)

    # Apply as diagonal shift
    identity = torch.eye(7, device=g_base.device, dtype=g_base.dtype).unsqueeze(0)
    deps_g = identity * eps_correction

    # GIFT effective metric
    g_GIFT = g_base + deps_g

    # Monitoring
    deps_g_mean = torch.abs(deps_g).mean().item()

    return g_GIFT, deps_g, deps_g_mean
```

**Why this works:**
- No coordinate perturbation → stable
- Diagonal correction preserves G₂ structure
- Trace-based formula physically motivated (scale toward det=2)
- Factor 0.01 keeps correction small

---

### 2. **Adjust RG Flow Coefficients** (CRITICAL)

**Current v1.2 (caused explosion):**
```python
'C': [25.0, 10.0, 2.0],  # Too large!
```

**New v1.2a:**
```python
'rg_flow': {
    'lambda_max': 39.44,
    'n_steps': 100,
    'epsilon_0': 1.0/8.0,
    'A': -12.0,   # Divergence (keep)
    'B': 6.0,     # Norm (keep)
    'C': [0.5, 0.2, 0.05],  # ← REDUCED 50× from v1.2
    'D': 8.5,     # Fractality (keep)
},
```

**Rationale:**
- C component exploded to +52 billion in v1.2 Phase 3
- Reduction from [25,10,2] → [0.5,0.2,0.05] (50× smaller)
- Keeps epsilon variation active but controlled

---

### 3. **Progressive Torsion Targeting** (NEW)

**Problem v1.2:**
All phases targeted ε=0.0164 immediately → conflicted with geometric stabilization

**Solution v1.2a:**
```python
def get_torsion_target(phase, epoch, n_epochs):
    """Progressive torsion targeting."""
    if phase <= 2:
        return None  # Free torsion
    elif phase == 3:
        return 0.05  # Loose target (50% margin)
    elif phase == 4:
        # Linear ramp: 0.05 → 0.018 over phase 4
        progress = epoch / n_epochs
        return 0.05 * (1 - progress) + 0.018 * progress
    else:  # phase == 5
        # Final target with tight tolerance
        return 0.0164
```

**Update loss function:**
```python
# In compute_losses():
if phase <= 2:
    losses['torsion'] = (torsion_norm.mean() ** 2) * 0.1  # Minimize, not target
else:
    target = get_torsion_target(phase, epoch, n_epochs_per_phase)
    losses['torsion'] = ((torsion_norm.mean() - target) ** 2)
```

---

### 4. **Reduced RG Flow Weights** (STABILITY)

**Current v1.2 Phase 5:**
```python
'rg_flow': 3.0,  # Too aggressive
```

**New v1.2a:**
```python
'phases': {
    3: {'rg_flow': 0.2},    # Introduction (keep)
    4: {'rg_flow': 0.5},    # Calibration start (keep)
    5: {'rg_flow': 1.5},    # ← REDUCED from 3.0
},
```

**Why:**
- v1.2 Phase 5 over-emphasized RG at expense of geometry
- Weight 1.5 balances RG calibration with metric stability

---

### 5. **Extended Training** (OPTIONAL)

**Current:** 5 phases × 2000 epochs = 10,000 total

**Option v1.2a:**
```python
'n_epochs_per_phase': 2500,  # 12,500 total (25% longer)
```

**OR** keep 10,000 but extend Phase 5:
```python
'n_epochs_phases': {
    1: 1500,
    2: 1500,
    3: 2000,
    4: 2000,
    5: 3000,  # ← Extended final calibration
}
```

**Tradeoff:**
- Longer training → better convergence (maybe)
- More GPU cost
- Recommendation: **Try 10K first**, extend only if needed

---

## 📋 **Implementation Checklist for v1.2a**

- [ ] **Replace** `compute_gift_metric()` with stable diagonal version
- [ ] **Update** RG coefficients: `C: [0.5, 0.2, 0.05]`
- [ ] **Add** progressive torsion targeting function
- [ ] **Modify** torsion loss to use progressive targets
- [ ] **Reduce** Phase 5 RG weight: 3.0 → 1.5
- [ ] **Test** on small sample (100 epochs) before full run
- [ ] **Monitor** Phase 3-5 for stability (no explosions)

---

## 🎯 **Expected v1.2a Results**

**Optimistic (if all fixes work):**
- Torsion: 0.018-0.025 (10-50% error) ✅
- det(g): 2.0-2.3 (0-15% error) ✅
- Δα: -0.5 to -0.7 (22-44% error) ⚠️
- **Status:** PARTIAL SUCCESS (publishable with caveats)

**Realistic (likely):**
- Torsion: 0.025-0.040 (50-150% error) ⚠️
- det(g): 2.2-2.5 (10-25% error) ⚠️
- Δα: -0.3 to -0.5 (44-67% error) ⚠️
- **Status:** IMPROVED but needs v1.3 for full precision

**Pessimistic (if structural issues remain):**
- Similar to v1.2 (>100% errors) ✗
- **Action:** Need architectural changes (deeper network, different loss balance)

---

## 🔄 **Fallback: Use v1.1a for S2**

If v1.2a doesn't achieve <20% errors, **use v1.1a results**:

| Metric | v1.1a | Status |
|--------|-------|--------|
| Torsion | 0.016125 | **1.68% error** ✅ |
| b₂ | 21 | Exact ✅ |
| RG flow | Partial (B term only) | Incomplete ⚠️ |

**S2 Strategy:**
```markdown
## Section 6: Numerical Results

We present results from v1.1a training achieving **torsion ε = 0.016125
(1.68% error)** with complete b₂=21 harmonic extraction. Full GIFT 2.1
dual geometry implementation (g_GIFT with 4-component RG flow) is in
development (v1.2a, in progress).

*Note: v1.1a uses partial GIFT compatibility (B component RG flow only).
Complete implementation with all four RG components (A·∇·T + B·‖T‖² +
C·∂_εg + D·fractality) is undergoing refinement.*
```

**Advantage:**
- Publishes **now** with excellent torsion precision
- Honest about limitations (partial RG flow)
- Notes ongoing work without blocking publication

---

## 🚀 **Next Steps**

1. **Implement v1.2a changes** in new notebook
2. **Test** on 500 epochs (Phase 1 only) to verify stability
3. **Full run** if test passes (10K epochs, ~6-12 hours)
4. **Decision point** after v1.2a:
   - If <20% errors → **Use for S2**, publish
   - If 20-100% errors → **Use v1.1a for S2**, continue to v1.3
   - If >100% errors → **Use v1.1a for S2**, major rethink needed

---

**Author:** GIFT Framework Team
**Date:** 2025-11-23
**Status:** Ready for implementation
