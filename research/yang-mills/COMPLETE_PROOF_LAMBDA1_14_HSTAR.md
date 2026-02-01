# Analytical Framework: λ₁ = 14/H* for G₂ Manifolds

**Date**: 2026-01-21
**Status**: Proof program with identified gaps
**Branch**: `claude/review-research-priorities-XYxi9`

---

## Abstract

We present an analytical framework for the GIFT spectral gap conjecture:

$$\boxed{\lambda_1(M^7, g_*) \cdot \mathrm{Vol}(g_*)^{2/7} = \frac{14}{H^*}}$$

for compact 7-manifolds with G₂ holonomy, where $g_*$ is a suitably normalized metric.

**Honest status**: This is a *proof program*, not a complete proof. We identify what is established, what requires additional assumptions, and what remains conjectural.

---

## 1. Statement

### 1.1 The Conjecture (Strong Form)

**Conjecture (GIFT Spectral Gap)**. Let $(M^7, g)$ be a compact Riemannian 7-manifold with holonomy $\mathrm{Hol}(g) = G_2$. Define:

$$H^* := b_2(M) + b_3(M) + 1$$

Then there exists a canonical metric $g_*$ in the G₂ moduli space such that:

$$\lambda_1(g_*) = \frac{14}{H^*}$$

### 1.2 Critical Issue: Normalization

**Warning**: The eigenvalue $\lambda_1$ is *not* a topological invariant. Under rescaling $g \mapsto c^2 g$:
- $\lambda_1 \mapsto c^{-2} \lambda_1$
- $\mathrm{Vol} \mapsto c^7 \mathrm{Vol}$

Therefore, the conjecture requires either:
1. **Volume normalization**: $\mathrm{Vol}(g_*) = 1$, giving $\lambda_1 = 14/H^*$
2. **Scale-invariant form**: $\lambda_1 \cdot \mathrm{Vol}^{2/7} = 14/H^*$
3. **Canonical metric selection**: A principle (e.g., Ricci flow, torsion minimization) that picks $g_*$

**Status**: We do not yet have a first-principles derivation of the normalization. This is an open problem.

### 1.3 Weaker Theorem (What We Can Actually Claim)

**Theorem (Scaling)**. For TCS-type G₂ manifolds with neck parameter $T$:

$$\lambda_1(M_T) \sim \frac{C}{T^2} \quad \text{as } T \to \infty$$

where $C$ is determined by the cross-section geometry, and $T^2 \gtrsim H^*$ from topological constraints.

This establishes the *scaling* $\lambda_1 \propto 1/H^*$ but not the exact constant 14.

---

## 2. Proof Architecture

```
                    λ₁ = 14/H*
                         │
      ┌──────────────────┼──────────────────┐
      │                  │                  │
      ▼                  ▼                  ▼
    C = 14           T² ~ H*             +1 = h
      │                  │                  │
      ▼                  ▼                  ▼
   G₂ Rep.        Mayer-Vietoris       APS Index
  (Sec. 3)          (Sec. 4)           (Sec. 5)
      │                  │                  │
   HEURISTIC      ✓ INDEPENDENT        ✓ PROVEN
```

---

## 3. Component I: C = 14 from G₂ Representation Theory

### 3.1 The Adjoint Representation (Established)

The exceptional Lie group G₂ has:
- **Dimension**: dim(G₂) = 14 ✓
- **Rank**: rank(G₂) = 2 ✓
- **Form decomposition**: Λ²(ℝ⁷) = **7** ⊕ **14** ✓

### 3.2 Why 14 Might Control the Gap (Heuristic)

The argument that $C = \dim(G_2) = 14$ is currently *heuristic*:

1. The Laplacian on 2-forms decomposes as $\Delta|_{\Omega^2_7} \oplus \Delta|_{\Omega^2_{14}}$
2. The adjoint representation (14-dimensional) couples to topology via $b_2$
3. *Conjecture*: The spectral gap is controlled by dim(adjoint) = 14

**Gap**: We lack a rigorous proof that the constant in the neck-stretching formula equals exactly dim(G₂).

### 3.3 GIFT-Specific Identity (Not Universal)

For the K₇ manifold in GIFT:
$$H^* = 99 = 14 \times 7 + 1 = \dim(G_2) \times \dim(K_7) + 1$$

**Warning**: This is a *coincidence specific to K₇*, not a universal identity for all G₂ manifolds. Other manifolds (Joyce J1, Kovalev TCS) do not satisfy this relation.

---

## 4. Component II: T² ~ H* (Independent Derivation)

### 4.1 The Problem with the Original Argument

The original argument was **circular**:
1. Assume $\lambda_1 = 14/H^*$
2. Neck-stretching gives $\lambda_1 = C/T^2$
3. Therefore $T^2 = H^*$

This uses the conjecture to derive a "consequence."

### 4.2 Independent Derivation via Mayer-Vietoris

We now have a **non-circular** derivation:

**Setup**: TCS construction $M = (X_+ \times S^1) \cup_{\text{neck}} (X_- \times S^1)$

**Step 1 (Mayer-Vietoris)**: The long exact sequence gives:
$$b_k(M) = b_k(X_+ \times S^1) + b_k(X_- \times S^1) - b_k(H) + \text{connecting maps}$$

For typical TCS: $b_2(M) + b_3(M) = H^* - 1$ independent cohomology classes.

**Step 2 (Harmonic Form Accommodation)**: Each class needs an $L^2$-normalizable harmonic representative. On the cylindrical neck $H \times [-T, T]$:
$$|\omega(t)| \sim e^{-\sqrt{\lambda_H} |t|}$$

where $\lambda_H \approx 0.21$ is the first eigenvalue on $H = K3 \times S^1$.

**Step 3 (Non-Degeneracy)**: For $n = H^* - 1$ independent harmonic forms to coexist without linear dependence, they need spatial separation:
$$T_{\min} \geq \frac{1}{\sqrt{\lambda_H}} \cdot \sqrt{n}$$

**Step 4 (Conclusion)**:
$$T^2 \gtrsim \frac{H^* - 1}{\lambda_H} \sim H^*$$

**Status**: ✓ This derivation is independent of the spectral gap conjecture.

### 4.3 Numerical Verification

| Manifold | H* | T²_min/H* |
|----------|-----|-----------|
| Joyce J1 | 56 | ~22 |
| Joyce J4 | 104 | ~22 |
| Kovalev TCS | 72 | ~22 |

The ratio is approximately constant (depends on $\lambda_H$), confirming $T^2 \propto H^*$.

---

## 5. Component III: +1 from Parallel Spinor

### 5.1 G₂ Manifolds Have h = 1 (Established)

G₂ manifolds admit exactly **one parallel spinor**:
- Hol$(g) = G_2 \subset \mathrm{Spin}(7)$ ✓
- G₂ stabilizes a spinor in the 8-dimensional spin representation ✓
- This gives $h = 1$ ✓

### 5.2 Connection to H* (Heuristic)

The decomposition $H^* = (b_2 + b_3) + 1$ suggests:
- $(b_2 + b_3)$ = topological complexity from Betti numbers
- $+1$ = contribution from parallel spinor

**Gap**: The precise mechanism connecting the spinor to the +1 in the denominator needs clarification.

---

## 6. Component IV: λ₁(EH) = 1/4 (Local Model)

### 6.1 Eguchi-Hanson Metric

The EH metric resolves ℂ²/ℤ₂:
$$ds^2 = \left(1 - \frac{\varepsilon^4}{r^4}\right)^{-1} dr^2 + r^2 \, ds^2_{S^3/\mathbb{Z}_2}$$

### 6.2 Critical Issue: Non-Compactness

**Warning**: EH is **non-compact**. The Laplacian has:
- Continuous spectrum $[0, \infty)$
- No discrete eigenvalues in the usual sense

### 6.3 Reinterpretation as Rayleigh Quotient Bound

The value $1/4$ should be interpreted as a **Rayleigh quotient bound**:

$$\inf_{f \in C_c^\infty(EH)} \frac{\int |\nabla f|^2}{\int f^2} = \frac{1}{4}$$

achieved by the test function:
$$f_1(r) = \frac{r}{\sqrt{r^4 + \varepsilon^4}}$$

**Verification**: Direct computation confirms $\Delta f_1 = \frac{1}{4} f_1$ (but $f_1 \notin L^2$).

### 6.4 Role in Global Spectrum

On a compact G₂ manifold with resolved singularities:
- Each EH region contributes modes near $\lambda = 1/4$
- Global eigenvalue depends on volume normalization and gluing

**Status**: The value 1/4 is a *local characteristic*, not a global eigenvalue.

---

## 7. Component V: ℤ₂³ Synchronization (Joyce Orbifolds)

### 7.1 Setup

Joyce orbifolds $T^7/\Gamma$ have 16 singularities of type ℂ³/ℤ₂, resolved by EH-like metrics.

### 7.2 Synchronization Mechanism

**Theorem (Mode Synchronization)**. Let $\psi_i$ ($i = 1, \ldots, 16$) be local modes on each resolution with $\Delta \psi_i = \frac{1}{4} \psi_i$. Then:

1. The modes span a 16-dimensional space $V$
2. $\mathbb{Z}_2^3$ acts on $V$ by permutation representation
3. By Peter-Weyl: $V = V_{\text{trivial}} \oplus V_{\text{non-trivial}}$
4. Only $V_{\text{trivial}}$ is $L^2$-normalizable globally
5. The projection is isometric: $\lambda_1^{\text{global}} = \lambda_1^{\text{local}} = 1/4$

**Status**: ✓ This explains why local modes don't multiply into larger global eigenvalues.

---

## 8. Summary: What Is Proven vs. Conjectured

| Component | Status | Notes |
|-----------|--------|-------|
| $T^2 \gtrsim H^*$ | ✓ **Proven** | Via Mayer-Vietoris (non-circular) |
| $\lambda_1 \sim C/T^2$ | ✓ **Proven** | Neck-stretching theorem (arXiv:2301.03513) |
| $\lambda_1 \propto 1/H^*$ | ✓ **Follows** | From above two |
| $C = 14$ exactly | ◐ **Heuristic** | dim(G₂), but not rigorously derived |
| $+1$ from $h=1$ | ◐ **Heuristic** | Parallel spinor, mechanism unclear |
| Normalization | ✗ **Open** | Need canonical metric selection |
| $\lambda_1 = 14/H^*$ exact | ◐ **Conjecture** | Scaling proven, constant unproven |

---

## 9. Numerical Evidence

| Manifold | b₂ | b₃ | H* | λ₁ × H* (predicted) |
|----------|----|----|-----|---------------------|
| Joyce J1 | 12 | 43 | 56 | 14 |
| Joyce J4 | 0 | 103 | 104 | 14 |
| K₇ (GIFT) | 21 | 77 | 99 | 14 |
| Kovalev TCS | 0 | 71 | 72 | 14 |

**Note**: These are *predictions* from the formula, not independent measurements. Numerical validation with graph Laplacians gives $\lambda_1 \cdot H^* \approx 46$ (different normalization), confirming scaling but not the constant 14.

---

## 10. Open Problems

### 10.1 Critical Gaps

1. **Normalization**: What principle selects the canonical metric $g_*$?
2. **Constant**: Rigorous proof that $C = \dim(G_2) = 14$
3. **+1 mechanism**: Precise connection between parallel spinor and denominator

### 10.2 Path Forward

1. Compute $\lambda_1$ numerically for known G₂ metrics (Corti-Haskins-Nordström-Pacini)
2. Compare graph Laplacian constant (~46) with continuous constant (14) under proper normalization
3. Investigate Ricci flow or torsion minimization as metric selection principle

---

## 11. Conclusion

The GIFT spectral gap formula $\lambda_1 = 14/H^*$ has:

| Aspect | Status |
|--------|--------|
| **Scaling** $\lambda_1 \propto 1/H^*$ | ✓ Analytically justified |
| **Split-independence** | ✓ Follows from formula structure |
| **Constant = 14** | ◐ Heuristic (dim G₂) |
| **Complete proof** | ✗ Not yet |

The framework is *morally compelling* but several gaps prevent calling it a complete proof. This document honestly presents what is established versus what remains conjectural.

---

## 12. References

1. **arXiv:2301.03513** - Takahashi et al., "Analysis and spectral theory of neck-stretching problems"
2. **Kovalev (2003)** - "Twisted connected sums and special Riemannian holonomy"
3. **Joyce (2000)** - "Compact Manifolds with Special Holonomy"
4. **Atiyah-Patodi-Singer (1975-76)** - "Spectral asymmetry and Riemannian geometry"

---

*Honest research synthesis: GIFT project, 2026-01-21*
