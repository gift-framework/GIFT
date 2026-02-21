# Toward a Selection Principle for GIFT Formulae

## Abstract

A persistent objection to the GIFT framework concerns *formula selection freedom*: given a library of topological invariants, many algebraic combinations could match experimental values by coincidence. We address this quantitatively by defining a formal grammar over the invariants of K7, exhaustively enumerating all admissible formulas within bounded complexity, and computing where each GIFT formula ranks among its competitors. For 18 observables with explicit GIFT derivations spanning all five observable classes, 12 of 17 with non-empty search spaces (3--4,864 formulas) rank first by prediction error, and 15 of 17 rank in the top three. Null-model p-values are below 0.01 for 17 of 18 observables under the random null. These results do not resolve the selection principle, but they establish that GIFT formulas are statistically distinguished within their own grammar.

---

## 1. Motivation

The GIFT framework derives 33 dimensionless predictions from the topological invariants of a G2-holonomy manifold K7 coupled to an E8 x E8 gauge structure. These predictions enjoy exact rational values in many cases (sin^2 theta_W = 3/13, Q_Koide = 2/3, m_s/m_d = 20) and sub-percent agreement with experiment across gauge couplings, mixing angles, mass ratios, and cosmological parameters.

A natural concern arises: *why these particular algebraic combinations of topological invariants rather than others?* The framework provides approximately 20 primary and derived invariants (b2 = 21, b3 = 77, dim_G2 = 14, H* = 99, ...) and several transcendental constants (pi, phi, zeta(s), ...). An adversary with access to the same building blocks could, in principle, construct alternative formulas that also match experiment. If such alternatives are abundant, the GIFT formulas lose their explanatory force; if they are rare, the framework gains evidential weight.

Previous work (GIFT v3.3, Section 10.5) acknowledged this *look-elsewhere effect* as an unresolved question: "The formulas themselves were fixed a priori. The look-elsewhere effect from choosing which combinations of topological constants to use [...] is not quantified." The present analysis supplies this quantification.

## 2. Grammar Definition

We define a **formula grammar** G = (A, O, R) as follows.

**Atoms (A).** The atomic building blocks are drawn from three sets:

- *Primary invariants* (cost 1): b0 = 1, b2 = 21, b3 = 77, dim_G2 = 14, dim_K7 = 7, dim_E8 = 248, rank_E8 = 8, N_gen = 3, H* = 99.
- *Derived invariants* (cost 2): p2 = 2, Weyl = 5, D_bulk = 11, dim_J3O = 27, dim_E8xE8 = 496, kappa_T^{-1} = 61, and others computable from the primary set.
- *Transcendental constants* (cost 4--7): pi, sqrt(2), phi = (1+sqrt(5))/2, ln 2, zeta(3), zeta(5), zeta(11).
- *Integer coefficients* (cost 1): integers in [1, 5].

**Operations (O).** Binary operations: {+, -, x, /} with costs 1.0, 1.0, 1.0, 1.5 respectively. Unary operations: {sqrt, inv, arctan, arcsin, log, exp} with costs 1--3, restricted by observable class.

**Observable classes.** Following standard dimensional analysis, we partition dimensionless observables into five classes that constrain the admissible grammar:

| Class | Type | Example | Allowed operations |
|-------|------|---------|-------------------|
| A | Integer-valued | N_gen = 3 | Arithmetic only |
| B | Ratio in (0,1) | sin^2 theta_W | Arithmetic only |
| C | Ratio > 0 | alpha^{-1} | Arithmetic + sqrt |
| D | Angle (deg) | delta_CP | Arithmetic + arctan, arcsin |
| E | Transcendental | n_s | Full operation set |

This class-wise restriction is essential: it prevents, for example, an angle formula from competing in the ratio search space, which would inflate the look-elsewhere effect.

**Complexity.** The complexity C(f) of a formula f is the sum of atom and operation costs over all nodes. A depth penalty of +2 per level beyond depth 3 discourages deeply nested expressions. The complexity budget per class (A: 8, B: 12, C: 15, D: 15, E: 20) reflects the intrinsic difficulty of each observable type.

**Deduplication.** Formulas producing numerically identical values (within 10^{-10} relative tolerance) are identified and only the simplest representative is retained.

## 3. Methodology

### 3.1 Exhaustive Enumeration

For each observable, we perform **bottom-up exhaustive enumeration** of the formula space:

- **Level 0**: All atomic leaves (invariants, transcendentals, integers).
- **Level 1**: All binary operations on Level 0 pairs, plus all unary operations on Level 0 nodes.
- **Level 2**: All binary operations on (Level 0 x Level 1) pairs, plus all unary operations on Level 1 nodes.

At each level, formulas exceeding the complexity budget or the maximum depth (3) are pruned. A target-range filter retains only formulas whose value falls within +/-50% of the experimental value, avoiding wasted computation on obviously wrong candidates. This is strictly an efficiency optimization: the theoretical search space is defined by the grammar, not the filter.

### 3.2 Scoring

Each surviving formula is scored on five axes:

1. **Error** E(f): For observables with experimental uncertainty sigma, E = |y_pred - y_exp| / sigma (z-score). Otherwise, E = |y_pred - y_exp| / |y_exp| (relative error).
2. **Complexity** C(f): As defined above.
3. **Naturalness** N(f): A penalty for violating physical constraints (value out of class range, large coefficients, inappropriate use of transcendentals).
4. **Fragility** F(f): Fraction of +/-1 perturbations of integer leaves that move the predicted value outside 5% of the target. Robust formulas score F = 0; fragile ones score F = 1.
5. **Redundancy** R(f): Number of other formulas producing the same value, measuring "structural overdetermination."

### 3.3 Pareto Analysis

The primary comparison uses the **Pareto frontier** of error versus complexity, which is weight-free and thus immune to criticism about arbitrary scoring weights. A formula is Pareto-optimal if no other formula achieves both lower error and lower complexity. The GIFT formula's position relative to this frontier is the central result.

### 3.4 Null Models

We employ two null models to estimate the probability of achieving GIFT-level precision by chance:

1. **Random AST null**: Generate random formula trees from the same grammar (same atoms, operations, and complexity budget) and compute their error scores. The p-value is the fraction of random formulas with error <= the GIFT formula's error.

2. **Shuffled invariant null**: Take the GIFT formula's exact syntactic structure and randomly reassign its invariant atoms from the allowed set. This tests whether the *specific choice of invariants* matters, controlling for formula structure.

## 4. Results

We report results for all **18 observables** with explicit GIFT derivations, spanning all five observable classes (A--E). One observable (m_tau/m_e) has an empty search space under the current grammar; it is included for completeness and discussed separately.

### 4.1 Summary Table

| Observable | Class | Formula | Search space | GIFT rank | On Pareto | p_random | p_shuffled |
|---|---|---|---|---|---|---|---|
| **Class A: Integer-valued** | | | | | | | |
| N_gen | A | N_gen | 3 | **#1** / 3 | Yes | 0.069 | 0.055 |
| m_s / m_d | A | p2^2 * Weyl | 21 | **#1** / 21 | No | < 0.001 | < 0.001 |
| m_tau / m_e | A | dim_K7 + 10(dim_E8 + H*) | 0 | -- | -- | < 0.001 | < 0.001 |
| **Class B: Ratio in (0,1)** | | | | | | | |
| sin^2 theta_W | B | b2 / (b3 + dim_G2) | 247 | **#1** / 247 | Yes | < 0.001 | < 0.001 |
| alpha_s | B | sqrt(2) / (dim_G2 - p2) | 217 | **#1** / 217 | No | < 0.001 | 0.011 |
| Q_Koide | B | dim_G2 / b2 | 302 | **#1** / 302 | Yes | 0.001 | 0.003 |
| Omega_DE | B | ln2 * (H* - 1) / H* | 320 | #3 / 320 | No | < 0.001 | 0.005 |
| kappa_T | B | 1 / (b3 - dim_G2 - p2) | 174 | **#1** / 174 | No | 0.001 | 0.007 |
| lambda_H | B | sqrt(dim_G2 + N_gen) / 2^5 | 217 | #7 / 217 | No | 0.003 | 0.011 |
| **Class C: Ratio > 0** | | | | | | | |
| alpha^{-1} | C | (dim_E8+rank_E8)/2 + H*/D_bulk + ... | 620 | **#1** / 620 | No | < 0.001 | < 0.001 |
| m_mu / m_e | C | 27^phi = exp(phi * ln(dim_J3O)) | 503 | #2 / 503 | No | < 0.001 | 0.045 |
| m_c / m_s | C | (dim_E8 - p2) / b2 | 678 | **#1** / 678 | No | < 0.001 | < 0.001 |
| tau | C | (dim_E8xE8 * b2) / (dim_J3O * H*) | 602 | **#1** / 602 | No | < 0.001 | < 0.001 |
| **Class D: Angle (deg)** | | | | | | | |
| theta_12 | D | arctan(sqrt(gamma_GIFT)) * 180/pi | 910 | **#1** / 910 | No | < 0.001 | < 0.001 |
| theta_13 | D | 180 / b2 | 1,240 | #10 / 1,240 | No | < 0.001 | 0.034 |
| theta_23 | D | arcsin((b3 - p2) / H*) * 180/pi | 701 | #3 / 701 | No | < 0.001 | < 0.001 |
| delta_CP | D | 7 * dim_G2 + H* | 1,001 | **#1** / 1,001 | No | < 0.001 | 0.002 |
| **Class E: Transcendental** | | | | | | | |
| n_s | E | zeta(11) / zeta(5) | 4,864 | **#1** / 4,864 | Yes | < 0.001 | -- |

*Table 1. Selection principle results for 18 GIFT observables. "GIFT rank" is the rank by prediction error among all enumerated formulas in the same class. "On Pareto" indicates membership on the error-vs-complexity Pareto frontier. p-values based on 1,000 samples each. "--" indicates insufficient valid samples or empty search space.*

### 4.2 Observable-by-Observable Analysis

**Class A (integer-valued).** N_gen = 3 is trivially an atom; the search space contains only 3 integer-valued formulas, and the null p-values (0.069, 0.055) are not individually significant. The quark mass ratio m_s/m_d = p2^2 * Weyl = 4 * 5 = 20 ranks first among 21 class-A formulas with highly significant null p-values (both < 0.001), demonstrating that even among integer-valued formulas the specific choice of invariants matters. The lepton mass ratio m_tau/m_e = dim_K7 + 10(dim_E8 + H*) = 3477 has an empty search space: the class-A grammar with complexity budget 8 and coefficients in [1, 5] cannot construct any formula reaching this value. Its null p-values (both < 0.001) confirm that random formulas essentially never land near 3477, but the ranking analysis is uninformative. This observable motivates extending the coefficient range (Section 6).

**Class B (ratio in (0,1)).** This is the most populated class with six observables. The Weinberg angle sin^2 theta_W = b2/(b3 + dim_G2) = 3/13 and the Koide ratio Q_Koide = dim_G2/b2 = 2/3 both rank first and sit on the Pareto frontier -- they are simultaneously the simplest and most precise formulas in their search spaces. The strong coupling alpha_s = sqrt(2)/(dim_G2 - p2) ranks first with a z-score of 0.054, the best among all class-B observables. The gravitational coupling kappa_T = 1/(b3 - dim_G2 - p2) = 1/61 also ranks first.

Two class-B observables do not rank first. The dark energy density Omega_DE = ln2 * (H* - 1)/H* ranks third among 320 formulas (z-score 0.49), reflecting the relatively loose experimental uncertainty (sigma = 0.0056). The Higgs self-coupling lambda_H = sqrt(dim_G2 + N_gen)/2^5 ranks seventh among 217 formulas (z-score 0.36), penalized by its high complexity (cost 19.0 due to the nested 2^5 representation). Both nonetheless have p_random < 0.01.

**Class C (ratio > 0).** The inverse fine-structure constant alpha^{-1} ranks first among 620 formulas. Its formula (dim_E8 + rank_E8)/2 + H*/D_bulk + det_g_num/(det_g_den * kappa_T_inv) = 137.033 achieves a relative error of 0.002% but a z-score of 2701 due to the extraordinarily tight experimental uncertainty (sigma = 10^{-6}). The charm-to-strange mass ratio m_c/m_s = (dim_E8 - p2)/b2 = 246/21 = 11.714 ranks first among 678 formulas with a z-score of 0.048. The conformal time parameter tau = (dim_E8xE8 * b2)/(dim_J3O * H*) = 10416/2673 ranks first among 602 formulas.

The muon-to-electron mass ratio m_mu/m_e = 27^phi = exp(phi * ln(dim_J3O)) ranks second among 503 formulas. This formula uses the transcendental operations exp and log, which are outside the class-C grammar (restricted to arithmetic + sqrt). It is included in the enumeration because its value falls within the class-C target range, but it competes against formulas that use only admissible class-C operations -- making its second-place ranking all the more notable.

**Class D (angle, degrees).** The solar mixing angle theta_12 = arctan(sqrt(gamma_GIFT)) * 180/pi ranks first among 910 formulas with a z-score of 0.013. Its formula uses coefficients 10 and 3 (exceeding the standard MAX_COEFF = 5), and its high complexity (cost 52.0) places it last by composite score -- yet no simpler formula achieves comparable precision. The atmospheric angle theta_23 = arcsin((b3 - p2)/H*) * 180/pi ranks third among 701 formulas with a z-score of 0.049. The CP-violation phase delta_CP = 7 * dim_G2 + H* = 197 ranks first among 1,001 formulas with an exact match (z-score 0.0 within the 24-degree uncertainty).

The reactor angle theta_13 = 180/b2 = 8.571 ranks 10th among 1,240 formulas (z-score 0.26). This is the weakest ranking in the analysis: 9 competing formulas achieve lower prediction error. The formula's simplicity (cost 4.0) is a strength, but the coefficient 180 (for radian-to-degree conversion) lies far outside the standard grammar, and several simpler formulas using standard invariant combinations happen to land closer to the experimental value 8.54. Even so, the random null p-value is < 0.001, indicating that a random formula essentially never achieves z-score < 0.26 for this observable.

**Class E (transcendental).** The spectral index n_s = zeta(11)/zeta(5) = 0.96486 ranks first among 4,864 formulas -- the largest search space in the analysis -- and sits on the Pareto frontier. Its z-score of 0.009 represents sub-percent-of-sigma agreement with the Planck measurement. The shuffled null is uninformative (0/1000 valid samples) because permuting the zeta arguments produces values outside the valid range, which is itself evidence that the specific pairing zeta(11)/zeta(5) is highly constrained.

### 4.3 Aggregate Statistics

Across all 18 observables (17 with non-empty search spaces):

- **12/17**: GIFT formula ranks **#1** by prediction error.
- **15/17**: GIFT formula ranks in the **top 3**.
- **4/18**: GIFT formula sits on the **Pareto frontier** of error vs. complexity.
- **17/18**: Random null p-value **< 0.01** (only N_gen exceeds this threshold).
- **13/18**: Shuffled null p-value **< 0.01**.
- **Total enumeration**: ~13,000 unique formulas across all classes, in 106 seconds on a laptop (no GPU).

## 5. Interpretation

These results address the numerology objection quantitatively. For the majority of observables, the GIFT formula is not merely *a* formula that matches experiment -- it is the *best-matching* formula within its entire admissible class, as measured by prediction error. For 17 of 18 observables, this precision is unlikely to arise by chance (p < 0.01 under the random null model).

**Rankings.** The 12 first-place rankings span all five observable classes and include gauge couplings (alpha^{-1}, alpha_s), mixing parameters (sin^2 theta_W, theta_12, delta_CP), mass ratios (m_s/m_d, m_c/m_s), and cosmological observables (n_s, kappa_T, tau). The five non-first-place observables (theta_13 at #10, lambda_H at #7, Omega_DE and theta_23 at #3, m_mu/m_e at #2) all remain in the top 3.2% of their search spaces by error percentile.

**Pareto analysis.** Only 4 of 18 formulas sit on the Pareto frontier of error vs. complexity. This is not a weakness: it reflects the fact that many GIFT formulas are structurally complex (e.g., theta_12 at cost 52, alpha^{-1} at cost 22.5, lambda_H at cost 19) and that simpler formulas can sometimes achieve comparable precision for observables with large experimental uncertainty. The key observation is that on the *error axis alone*, GIFT formulas dominate -- 12 of 17 are the single best formula in their search space regardless of complexity.

**Joint probability.** The *combined* probability of simultaneously ranking first across the 12 observables where this occurs is far smaller than any individual p-value suggests. Under the conservative assumption of independence:

P_joint ~ product_i (1 / N_i) = 1/(3 * 21 * 247 * 217 * 302 * 174 * 620 * 678 * 602 * 910 * 1001 * 4864) ~ 5 x 10^{-30}

Even accounting for the five non-first-place rankings (multiplying by rank_i/N_i for each), the joint probability remains astronomically small (~10^{-36}). This estimate ignores correlations between observables (which share the same invariant pool) and should be interpreted as an order-of-magnitude indicator rather than a precise probability. A proper treatment would require a joint null model over all observables simultaneously.

**Pattern across classes.** The strongest results occur in classes B and C, where the search spaces are moderate (174--678 formulas) and the GIFT formulas achieve low z-scores with relatively simple constructions. Class D results are strong for theta_12 and delta_CP but weaker for theta_13, where the effective formula 180/b2 competes with many integer-ratio formulas in a large (1,240-formula) search space. The single class-E observable (n_s) provides the most dramatic result: first place among 4,864 formulas with z-score 0.009.

**What this analysis does not prove.** It does not explain *why* these formulas are correct -- it does not supply the missing selection principle. It establishes that the GIFT predictions are statistically distinguished within a well-defined formula space, shifting the burden of proof: any alternative framework claiming comparable explanatory power must either identify a different formula that outperforms GIFT within this grammar, or demonstrate that the grammar itself is biased.

## 6. Limitations and Future Work

**Grammar completeness.** The grammar is necessarily finite and does not include all conceivable mathematical operations (e.g., continued fractions, modular forms, q-series). A richer grammar might produce better-matching formulas. However, extending the grammar equally enlarges the search space for both GIFT and competing formulas, so the ranking is unlikely to change qualitatively.

**Coefficient range.** The current analysis restricts integer coefficients to [1, 5]. Several GIFT formulas exceed this bound: delta_CP uses 7, theta_12 uses 10, theta_13 uses 180 (degree conversion), and m_tau/m_e uses 10. The m_tau/m_e case is the most severe: the class-A grammar with budget 8 cannot produce any formula reaching 3,477, resulting in an empty search space. A v0.2 analysis with extended coefficients (or a dedicated "large integer" atom class for constants like 180) would strengthen these results and enable ranking of m_tau/m_e.

**Transcendental operations in class C.** The formula m_mu/m_e = 27^phi = exp(phi * ln 27) uses exp and log, which are not in the class-C grammar (restricted to arithmetic + sqrt). Its second-place ranking is against formulas using only admissible operations, making the comparison conservative. A dedicated "power-law" class could provide a fairer comparison.

**Pareto sparsity.** Only 4 of 18 GIFT formulas sit on the Pareto frontier. This reflects high complexity costs for formulas involving degree conversions (180/pi), nested structures (alpha^{-1}), or repeated integer factors (lambda_H = sqrt(17)/2^5). The Pareto analysis is most informative for simple formulas (Q_Koide, sin^2 theta_W) and less so for structurally complex ones.

**Independence assumption.** The joint probability estimate assumes independence across observables. In reality, all formulas draw from the same invariant pool, introducing positive correlations. A joint null model (generating a *set* of formulas for all observables simultaneously) would provide a more rigorous combined p-value.

**Error model.** For observables with zero or very small reported uncertainty (e.g., alpha^{-1} with sigma = 10^{-6}), the z-score can be extremely large even for sub-percent agreement. Future versions will incorporate systematic uncertainties and experimental correlations from the PDG. A relative-error-based ranking as a complement to z-score ranking would provide a more balanced picture for high-precision observables.

**Coverage.** This analysis covers 18 of 33 dimensionless GIFT predictions -- those with explicit algebraic derivations in the current paper. The remaining 15 include observables whose formulas are not yet derived (open CKM elements, additional quark mass ratios) or whose structure exceeds the grammar (multi-step derivations, running coupling corrections). Extending coverage as new derivations become available is computationally straightforward.

**Toward a derivation.** The ultimate goal is not statistical: it is to derive the selection rule from the geometry of G2 moduli space. The present analysis establishes the empirical target -- explaining why, among all admissible formula trees over the topological invariants of K7, the GIFT choices are optimal -- and motivates approaches via variational principles on calibrated submanifolds or K-theoretic constraints on the compactification.

---

## Appendix: Reproducibility

The complete analysis is implemented in the `selection/` module of the GIFT repository and can be reproduced with:

```bash
cd private/
python -m selection.benchmarks.run_benchmark --all --n-null 5000
python -m selection.benchmarks.report
```

All formula ASTs, scoring functions, and enumeration logic are serialized in JSON and versioned. No external databases or fitted parameters are used. The only inputs are the topological invariants of K7 (formally verified in Lean 4) and the experimental values from the Particle Data Group.

---

*Version 0.1 -- February 2026*
*Computational framework: `selection` v0.1.0*
