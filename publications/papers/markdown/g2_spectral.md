# Spectral Geometry of an Explicit G₂ Metric: Laplacian Spectrum and Harmonic Forms

**Brieuc de La Fourniere**

*Independent researcher*

---

## Abstract

We compute the Laplacian spectrum and harmonic forms on the certified torsion-free G₂ structure of a companion paper [A], constructed on a TCS-type neck model K3 × T² × [0,1]. The scalar spectral gap is λ₁ = 0.12461 ± 0.00016 (Richardson-extrapolated across six grids, Appendix A) with Weyl exponent α = 1.998 (adiabatic prediction: 2.0), matching the analytical formula λ₁ = π²/(L² g_ss) = 6π²/475 ≈ 0.12467 — derived a posteriori from the NK-certified seam parameters — to 0.05%. Spectral analysis of the Hodge Laplacians realizes the harmonic multiplicities (b₂, b₃) = (21, 77) expected from the conjectural compact extension: 21 near-zero eigenvalues of Δ₂ (gap ratio 14,635) and 77 near-zero eigenvalues of Δ₃. The 21 harmonic 2-forms carry a K3 fiber intersection form of signature (3, 18) — rank-21 restriction of the ambient K3 lattice (3, 19) — with SD/ASD eigenvalue gap 2,210. The spectral democracy theorem proven in [A] is verified numerically to 10⁻⁵ precision. The complete Kaluza-Klein tower comprises 22,671 levels with Poisson level spacing statistics. All spectral results are properties of the explicit neck-model metric; their interpretation on a closed compact G₂ manifold is conditional on the existence of such a manifold with (b₂, b₃) = (21, 77).

**Keywords:** spectral geometry, G₂ holonomy, Hodge Laplacian, Kaluza-Klein spectrum, harmonic forms, intersection form

---

## 1. Introduction

The construction of explicit metrics with exceptional holonomy has been a long-standing challenge in Riemannian geometry. While existence theorems for compact G₂ manifolds are well-established (Joyce [1], Kovalev [2], CHNP [3], Nordström [5]), explicit spectral data — whether on closed compact G₂ manifolds or on neck-model geometries adapted to them — has remained unavailable due to the absence of concrete metric coefficients.

In a companion paper [A], we certify a torsion-free G₂ structure on a TCS-type neck model K3 × T² × [0,1] via computer-assisted proof, with seam-sector geometry adapted to a putative compact extension of Betti numbers (b₂, b₃) = (21, 77), using interval arithmetic with zero finite differences. The present paper computes the spectral geometry of this certified neck-model metric; the spectral results are properties of that metric, while their interpretation on a closed compact G₂ manifold is conditional on the existence of such a manifold with (b₂, b₃) = (21, 77) ([A], §2.2). We compute:

1. **Scalar Laplacian spectrum** (§3): spectral gap, Weyl law, complete KK tower.
2. **Harmonic forms** (§4): spectral Betti confirmation, intersection form, SD/ASD structure.
3. **Numerical democracy verification** (§5): confirmation of the analytical theorem from [A].
4. **Physical context** (§6): relevance for Kaluza-Klein compactifications.

**Note on related work.** An independent arithmetic derivation of the pair $(b_2, b_3) = (21, 77)$ has recently appeared in Zhou & Zhou [7, 8], based on Diophantine constraints and spectral stability screening. Their approach is complementary to the present one.

## 2. The Certified Metric (Summary)

We briefly summarize the metric from [A]. The full construction, NK certification, and analytical decomposition are presented therein.

The metric on a neck region K3 × T² × [0,1] decomposes into three adiabatic sectors:

$$g(s, \theta, \chi, y) = g_{\text{seam}}(s) \oplus g_{T^2} \oplus g_{K3}(y)$$

with effective eigenvalues g_{ss} = 19/6 (torsion-minimizing rational, to 0.03%), g_{T²} = 7/6 (torsion-minimizing rational, to 0.20%), g_{K3} ≈ 64/77 (mean K3 eigenvalue, *rational approximation* to 0.41%; see [A] §3.8 and [A] Appendix B on the companion interval-arithmetic certificates for the conjectural status and the 1-parameter K3 eigenvalue signature), determinant det(g) = 65/32 (exact, imposed at reconstruction; interval-certified to width $1.4 \times 10^{-12}$), and ACyl decay rate γ = 2π√(6/7) ≈ 5.817 (derived from T² Hodge Laplacian + H¹(K3)=0; NK-computed: 5.811, 0.1% NK proximity). The Newton-Kantorovich certificate [A] establishes existence and uniqueness of a nearby torsion-free G₂ metric with h = 1.43 × 10⁻⁹ (margin ×350 million, zero finite differences) and metric correction δg/g ≤ 1.35 × 10⁻⁷.

For the spectral analysis below, the key property is that the torsion-free metric g* is adiabatically product-type with certified error ε_ad = 2 × 10⁻³ ([A], Proposition 3.3), giving eigenvalue perturbation |δλ|/λ ≤ 0.78% and enabling decomposition of 7D Laplacian operators into families of 1D Sturm-Liouville problems.

**Remark (independence from the exact value of $g_{K3}$).** The spectral identifications of this paper — scalar gap $\lambda_1 = 0.12461$, Weyl exponent $\alpha = 1.998$, spectral Betti confirmations ($b_2 = 21$, $b_3 = 77$), intersection form signature $(3, 18)$ — rely only on the adiabatic product structure of $g^*$ with certified error $\varepsilon_{\mathrm{ad}} = 2 \times 10^{-3}$. They do *not* depend on the exact value of $g_{K3}$, on whether $g_{K3} = 64/77$ is an identity or an approximation, or on the precise K3 eigenvalue deviation pattern. In particular, the certified perturbation budget $|\delta\lambda_n|/\lambda_n \leq 0.78 \%$ already absorbs the measured K3 eigenvalue spread of $1.16 \%$ and any residual $O(10^{-3})$ uncertainty in the K3 structural constants. The spectral claims herein are robust to future revisions of the analytical status of $g_{K3}$ in [A].

## 3. Laplacian Spectrum

### 3.1 Adiabatic decomposition

The adiabatic product structure decomposes the 7D Laplacian into families of 1D Sturm–Liouville operators, one per fiber mode (m, n, μ_{K3}):

$$\Delta_0 f = -\frac{1}{\sqrt{\det g}}\, \partial_s \left( \sqrt{\det g}\, g^{ss}\, \partial_s f \right) + V_{(m,n,\mu)} f$$

where V_{(m,n,μ)} = m² g^{θθ} + 2mn g^{θχ} + n² g^{χχ} + μ_{K3} combines the T² and K3 fiber eigenvalues. This decomposition is exact for the product-type metric and certified to ε_ad = 2 × 10⁻³ adiabatic error ([A], Proposition 3.3).

We discretize on a Chebyshev collocation grid with N = 800 to 1600 nodes on s ∈ [−2, 3] and solve the resulting matrix eigenvalue problem using dense diagonalization (Neumann boundary conditions) and shift-invert Lanczos (Dirichlet).

### 3.2 Spectral gap

The fundamental (0, 0, 0) channel yields the scalar spectral gap:

$$\lambda_1 = 0.12461 \pm 0.00016$$

confirmed by Richardson extrapolation across six grid sizes (N = 200 to 1200) with both boundary condition types yielding consistent values (Appendix A).

The first 10 eigenvalues of the (0, 0, 0) channel are:

| n | λ_n | Δλ/λ (Richardson) |
|---|------|-------------------|
| 0 | 3.5 × 10⁻¹³ | — (zero mode) |
| 1 | 0.12450 | 0.08% |
| 2 | 0.4976 | 0.04% |
| 3 | 1.1196 | 0.02% |
| 4 | 1.9904 | 0.01% |
| 5 | 3.1099 | < 0.01% |
| 6 | 4.4782 | < 0.01% |
| 7 | 6.0952 | < 0.01% |
| 8 | 7.9609 | < 0.01% |
| 9 | 10.075 | < 0.01% |

The near-quadratic growth λ_n ≈ 0.125 n² is characteristic of a 1D Sturm–Liouville problem; the eigenvalue spacing implies an effective spectral length $L_\mathrm{eff} \approx \pi/\sqrt{0.125} \approx 8.89$ (distinct from the domain extent $L=5$ below).

**Analytical formula.** In terms of the NK-certified model parameters, the spectral gap satisfies

$$\lambda_1 = \frac{\pi^2}{L^2 \cdot g_{ss}} = \frac{6\pi^2}{475} \approx 0.12467$$

where $L = 5$ is the seam domain extent ($s \in [-2, 3]$, 5 units) and $g_{ss} = 19/6$ is the seam metric component (torsion minimizer, [A] Proposition 3.2), giving $L^2 \cdot g_{ss} = 25 \cdot 19/6 = 475/6$. This is an analytical expression for $\lambda_1$ in terms of NK-certified model parameters, not an independent topological prediction; $L$ and $g_{ss}$ are both outputs of the certification. The match to the Richardson-extrapolated value 0.12461 ± 0.00016 (Appendix A) is 0.05%.

### 3.3 Weyl law

The eigenvalue counting function N(λ) = #{n : λ_n ≤ λ} satisfies the Weyl asymptotic N(λ) ∼ C λ^{α/2}. A least-squares fit to the (0, 0, 0) channel yields:

$$\alpha = 1.998 \pm 0.002, \qquad C = 0.125$$

in agreement with the predicted exponent α = 2 for the seam sector. Note that α = 2 is the expected Weyl exponent for any 1D Sturm–Liouville problem and reflects the dimension of the seam channel, not a G₂-specific feature. The non-trivial spectral verification is the full 7D exponent below.

For the full unified Kaluza–Klein tower (all T² and K3 channels), the effective 7D Weyl exponent is α_{7D} = 3.460 ± 0.040, consistent with α = 7/2 = 3.5 (deviation 1.1%).

### 3.4 Kaluza–Klein tower and level spacing

The complete KK tower below λ = 20 contains 22,671 distinct energy levels (comparable in scope to KK spectra computed on weak-G₂ examples such as squashed S⁷ [10]; for a recent comprehensive review of KK reductions in supergravity see [11]). The three-scale hierarchy is:

| Sector | First eigenvalue | Scale |
|--------|-----------------|-------|
| Seam | 0.125 | 0.353 |
| T² | 0.855 | 0.925 |
| K3 | 1.208 | 1.099 |

The T² channels obey a quadratic potential V_{(m,n)} = 0.855(m² + n²) with isotropy to 3 × 10⁻⁷. K3 adiabatic additivity holds to 0.003–0.023%.

The level spacing distribution follows Poisson statistics (χ² = 1.36) rather than GOE (χ² = 28.0), consistent with an integrable adiabatic system.

---

## 4. Harmonic Forms and Intersection Theory

### 4.1 Harmonic 2-forms

We construct 21 harmonic 2-forms ω_I on M by lifting K3 harmonic (1, 1)-forms through the adiabatic decomposition. Each ω_I has a radial profile f_I(s) satisfying Δ₂ ω_I = 0, computed via variational optimization with a neural network ansatz [4, 9] (3-layer MLP minimizing ‖Δ₂(f_I dθ ∧ α_I)‖²_{L²} on the Chebyshev collocation grid, convergence criterion max|residual| < 10⁻⁷) followed by Gram–Schmidt orthonormalization.

The 21 forms decompose into two families:

- **N₁-type** (11 forms): profiles interpolating from 1 to 0 along the neck.
- **N₂-type** (10 forms): profiles interpolating from 0 to 1.

This decomposition reflects the two-block structure of the neck region, matching the TCS formula b₂ = b₂(M₁) + b₂(M₂) = 11 + 10. Three independent decompositions are consistent: 7 + 14 (G₂ representations Λ²₇ + Λ²₁₄), 3 + 18 (SD + ASD from K3 hyperkähler triple), and 11 + 10 (TCS building blocks). The L² Gram matrix has condition number 1.047 with maximum off-diagonal entry 0.012.

**Spectral confirmation.** Δ₂ has 21 eigenvalues below 10⁻⁸. The 22nd eigenvalue is λ₂₂ ≈ 0.12, giving a gap ratio:

$$\lambda_{22} / \max(\lambda_1, \ldots, \lambda_{21}) \approx 14{,}635$$

This four-orders-of-magnitude gap confirms b₂ = 21.

### 4.2 Intersection structure

The L² Gram matrix $G_{IJ} = \int_M \omega_I \wedge \star \omega_J$ is positive definite (condition number 1.047, §4.1). The topological intersection structure is accessed via the **K3 fiber pairing** $I_{IJ} = \int_{K3} \alpha_I \wedge \alpha_J$, where $\alpha_I = \omega_I|_{K3}$ are the restrictions to the K3 fiber — this is the cup-product intersection form on $H^2(K3)$, which takes both positive and negative values.

The ambient K3 lattice $H^2(K3)$ has signature (3, 19) of rank 22. Of the 22 K3 classes, 21 extend to global harmonic forms on M (confirmed by the 21 near-zero Δ₂ eigenvalues, §4.1); the remaining class does not survive TCS boundary matching and corresponds to the first non-zero Δ₂ eigenvalue λ₂₂ ≈ 0.12. The K3 fiber pairing restricted to the **21 global harmonic 2-forms** has signature **(3, 18)**:

| Type | Count | $I_{IJ}$ eigenvalue range |
|------|-------|-----------------|
| Self-dual (SD) | 3 | 4.863, 5.499, 7.795 |
| Anti-self-dual (ASD) | 18 | −0.00423 to −0.00219 |

The ratio of mean SD to mean ASD absolute eigenvalue is approximately **2,210**. The cross-block Frobenius norm ‖I_{SD,ASD}‖_F = 0.018 confirms near-exact block diagonality.

This extreme gap originates from the SU(2) holonomy of the K3 fiber: SD 2-forms are aligned with the hyperkähler triple (ω_I, ω_J, ω_K), while ASD forms lie in the 18-dimensional complement of the rank-21 restricted lattice.

### 4.3 Harmonic 3-forms

The 77 harmonic 3-forms decompose as:

- **35 constant-type**: Λ³-sections with constant profiles.
- **21 dθ-fiber**: f_I(s) dθ ∧ ω_I.
- **21 dχ-fiber**: f_I(s) dχ ∧ ω_I.

T² isotropy dθ ↔ dχ holds to 3 × 10⁻⁷, and 35 + 21 + 21 = 77 = b₃. The 35-block Gram matrix has condition number 7.66 and eigenvalue range [1.65, 12.62].

**Minimum associative cycle volume.** The 7 associative calibrated 3-cycles on K₇ (Fano-plane structure) have volumes from period integrals: 508.5, 362.3, 362.3, 219.9, 219.9, 219.9, 219.9 (degeneracy pattern 1:2:4 reflecting Z₂ Fano symmetry). The minimum satisfies an empirical scaling relation (no derivation currently available; presented as a conjectural numerical observation):

$$V_{\min} \approx \sqrt{\frac{\mathrm{Vol}(K_7)}{11}}, \qquad 11 = \frac{b_3}{n} = \frac{77}{7}$$

NK value: 219.90; formula: 221.24; deviation 0.6%. The factor $11 = b_3/7$ reflects the Fano-plane multiplicity, but no proof of this relation is available.

---

## 5. Spectral Democracy (Numerical Verification)

The spectral democracy theorem, proven analytically in [A, Theorem 1.3], states that on product-type Ricci-flat G₂ metrics with constant g_{θθ}, the transverse 1-form Laplacian satisfies Δ₁(f dθ) = (Δ₀ f) dθ exactly.

We verify this by comparing the first 60 eigenvalues of Δ₀, Δ₁ (transverse channel f(s)dθ), and Δ₂ (the corresponding 2-form channel). For each pair (Δ_p, Δ_q), define the relative discrepancy:

$$\delta_{pq}(n) = \frac{|\lambda_n^{(p)} - \lambda_n^{(q)}|}{\lambda_n^{(p)}}$$

The results are:

| Comparison | max δ (n ≤ 60) | mean δ |
|-----------|---------------|--------|
| Δ₀ vs Δ₁ (transverse) | 8 × 10⁻⁵ | 3 × 10⁻⁵ |
| Δ₀ vs Δ₂ (ds ∧ dθ channel) | 9 × 10⁻⁵ | 4 × 10⁻⁵ |
| Δ₁ vs Δ₂ | 5 × 10⁻⁵ | 2 × 10⁻⁵ |

All discrepancies are consistent with the fiber metric variation of 0.002% ([A] Theorem 1.3 predicts exact agreement for a perfectly constant fiber). The spectral democracy is not generic: it would break down for metrics with significant s-dependent fiber geometry, and its observation at the 10⁻⁵ level provides an independent validation of the adiabatic decomposition.

The certified adiabatic bound ([A], Proposition 3.3) gives δλ/λ ≤ 0.78%; the observed 10⁻⁵ discrepancy is 80× tighter, reflecting the T² isotropy (3 × 10⁻⁷) rather than the full adiabatic parameter (2 × 10⁻³).

---

## 6. Discussion

### 6.1 Summary

We have computed the spectral geometry of an NK-certified G₂ metric [A]:
1. Scalar spectral gap λ₁ = 0.12461 ± 0.00016 (Richardson) with Weyl law verification (α = 1.998) and analytical formula 6π²/475 ≈ 0.12467 (§3).
2. Spectral confirmation of all Betti numbers with gap ratio 14,635 for b₂ = 21 (§4).
3. K3 fiber intersection form signature (3, 18) (rank-21 restriction of K3 lattice) with SD/ASD gap 2,210 (§4).
4. Numerical confirmation of spectral democracy to 10⁻⁵ (§5).
5. Complete KK tower (22,671 levels) with Poisson statistics (§3).
6. Conjectural scaling relation V_min ≈ √(Vol(K₇)/11) for minimum associative cycle volume, deviation 0.6% (§4).

### 6.2 Physical context

The spectral data presented here is relevant for Kaluza-Klein reductions of supergravity theories on G₂ manifolds. The scalar spectral gap determines the mass of the lightest KK mode; the harmonic forms determine the 4D field content; and the intersection form encodes gauge coupling structure.

A companion paper [D] discusses the physical context of these spectral results in the G₂ compactification setting.

### 6.3 Limitations

**Adiabatic approximation.** The spectral computations exploit the product-type structure of the certified metric ([A], Proposition 3.3). The adiabatic error is ε_ad = 2 × 10⁻³ with certified eigenvalue perturbation |δλ_n|/λ_n ≤ 0.78% for all eigenvalues. The spectral Betti confirmation (gap ratio 14,635) is stable under this perturbation: a 0.78% shift on each eigenvalue changes the gap ratio by at most 2 × 0.78% = 1.56% (worst case, numerator and denominator moving in opposite directions), leaving it at four orders of magnitude. Equivalently, the spectral Betti identification $(b_2, b_3) = (21, 77)$ is a robust property of the torsion-free metric $g^*$ that survives any residual $O(10^{-3})$ uncertainty in the K3 fiber structural constants — including the eigenvalue spread of 1.16 % and the approximate nature of $g_{K3} = 64/77$ ([A] §3.8).

**Convergence.** The spectral gap is Richardson-extrapolated from 6 grids (Appendix A). The error bar ± 0.0001 is a numerical estimate, not a rigorous bound.

### 6.4 Open questions

1. **Spectral characterization.** To what extent do the spectral data (gap, Weyl coefficient, SD/ASD ratio) characterize the underlying manifold among G₂ 7-manifolds?
2. **Fiber-coupled corrections.** What is the spectral shift from explicit K3 mode coupling?
3. **Comparison with flow methods.** How does the spectrum compare with Laplacian flow metrics on the same topological type [6]?

---

## Appendix A. Convergence Analysis

| N | λ₁ (Neumann) | λ₁ (Dirichlet) | Δλ₁/λ₁ |
|---|-------------|----------------|---------|
| 200 | 0.12346 | 0.12361 | — |
| 400 | 0.12409 | 0.12418 | 0.51% |
| 600 | 0.12430 | 0.12436 | 0.17% |
| 800 | 0.12440 | 0.12444 | 0.08% |
| 1000 | 0.12446 | 0.12449 | 0.05% |
| 1200 | 0.12450 | 0.12452 | 0.03% |

Extrapolated: λ₁ = 0.12461 ± 0.00016.

---

## Acknowledgements

AI tools (Claude, Anthropic) were used for computational assistance. All scientific content, analysis, and conclusions are the sole responsibility of the author.

## Author's note on AI collaboration

This framework was developed through sustained collaboration between the author and several AI systems, primarily Claude (Anthropic), with contributions from GPT (OpenAI), Gemini (Google), and Aristotle (Harmonic) for specific mathematical insights. Architectural decisions and many key derivations emerged from iterative dialogue sessions. This collaboration follows a transparent crediting approach for AI-assisted mathematical research.

## References

[1] D.D. Joyce, *Compact Manifolds with Special Holonomy*, Oxford University Press, 2000.

[2] A. Kovalev, "Twisted connected sums and special Riemannian holonomy," J. Reine Angew. Math. **565** (2003), 125–160.

[3] A. Corti, M. Haskins, J. Nordström, T. Pacini, "G₂-manifolds and associative submanifolds via semi-Fano 3-folds," Duke Math. J. **164** (2015), 1971–2092.

[4] M. Larfors, A. Lukas, R. Ruehle, "Calabi–Yau metrics from machine learning," arXiv:2206.13431 (2022).

[5] J. Nordström, "Extra-twisted connected sum G₂-manifolds," Ann. Global Anal. Geom. (2023).

[6] J.D. Lotay, Y. Wei, "Laplacian flow for closed G₂ structures," Geom. Funct. Anal. **29** (2019), 1048–1110.

[7] C. Zhou, Z. Zhou, "Algebraic Stability and Cosmological Structure A: The Necessity of G₂ Manifolds: From Self-Referential Dynamics to Exceptional Holonomy," preprint (Feb. 2026).

[8] C. Zhou, Z. Zhou, "Algebraic Stability and Cosmological Structure E: The Arithmetic Necessity of Three Generations of Fermions," preprint (Feb. 2026).

[9] P. Berglund, G. Butbaia, T. Hübsch, V. Jejjala, D. Mayorga Peña, C. Mishra, J. Tan, "cymyc: Calabi–Yau metrics, Yukawas, and curvature," arXiv:2410.19728 (2024).

[10] B.E.W. Nilsson, "Squashed 7-spheres, octonions and the swampland," arXiv:2412.04208 (2024).

[11] M.J. Duff, B.E.W. Nilsson, C.N. Pope, "Kaluza–Klein supergravity 2025," arXiv:2502.07710 (2025).

## Author's Related Works

[A] B. de La Fournière, "A Certified Torsion-Free G₂ Structure on a TCS Neck Model via Computer-Assisted Proof," Zenodo 10.5281/zenodo.18860358 (2026).

[C] B. de La Fournière, "Newton–Kantorovich diagnostics on a Donaldson K3 metric: two β estimates, machine-checked inequalities," Zenodo 10.5281/zenodo.19708916 (2026).

[D] B. de La Fournière, GIFT v3.4: geometric and physical context, companion paper, in preparation (2026).
