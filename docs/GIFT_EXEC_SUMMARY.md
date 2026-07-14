---
title: K₇: Executive Summary
---

# K₇: Reading topology instead of fitting it

*A technical summary for curious physics readers*

---

## The problem, stated plainly

The Standard Model of particle physics operates with a precision no other scientific theory has ever matched. It describes every known fundamental interaction apart from gravity, with experimental agreement sometimes reaching twelve decimal places.

Yet it contains nineteen free parameters, nineteen numbers the theory does not compute, and which must be measured experimentally before the theory can be used. The masses of the three lepton generations. The masses of the six quarks. The four parameters of the CKM matrix describing quark mixing. The four analogous parameters of the PMNS matrix for neutrinos. The three coupling constants of the electromagnetic, weak, and strong forces. The Higgs mass and its vacuum expectation value.

Nineteen numbers. No explanation. The Standard Model is an extraordinarily precise measuring instrument that requires calibration via nineteen tuning dials, with no account of why those dials exist or why they point where they point.

To this are added a handful of long-standing puzzles that have crossed the discipline for decades. Why exactly three generations of matter rather than two, or five, or seventeen? Why do the charged lepton masses obey the Koide relation *Q* = 2/3 to a precision of 10⁻⁴, a relation first noticed in 1981 and never explained? Why does the ratio of dark matter to baryonic matter sit close to a simple rational?

K₇ (*formerly Geometric Information Field Theory*) is an attempt to answer these questions by changing the nature of the question itself. Rather than asking "what is the value of these parameters," K₇ asks: "what if these parameters were not values to be measured, but consequences of a geometric structure to be identified?"

## The hypothesis

In its most compact form, the idea fits in a single sentence: the dimensionless parameters of the Standard Model emerge as algebraic combinations of topological invariants of a compact seven-dimensional manifold carrying an exceptional holonomy, coupled to an E₈ × E₈ gauge structure.

That sentence deserves to be unpacked.

A **seven-dimensional manifold** is a curved space with seven dimensions; one can picture it, very loosely, as an abstract surface, only with six extra dimensions. **Holonomy** measures how a vector changes when parallel-transported around a closed loop in that space. For most manifolds this change can be any rotation: the holonomy group is then the full orthogonal group. Certain privileged manifolds, however, have holonomy *restricted* to a particular subgroup. In seven dimensions there is an exceptional such subgroup, denoted **G₂**, of dimension 14, which arises as the automorphism group of the octonion algebra. Manifolds whose holonomy is reduced to G₂ are rare, difficult to construct, and carry remarkable geometric properties.

The **topological invariants** of such a manifold are integers that characterize its shape independently of its precise metric. For G₂ manifolds, the most important are the **Betti numbers** *b₂* and *b₃*, which count the harmonic 2-forms and 3-forms the space can carry. These are discrete global properties: they cannot vary continuously. They are whole numbers, full stop.

K₇ makes the specific hypothesis that there exists such a manifold *K₇* with (*b₂*, *b₃*) = (21, 77), coupled to the E₈ × E₈ gauge structure of dimension 496. From these ingredients alone, it attempts to derive the observed parameters algebraically.

**E₈** is the largest of the exceptional Lie groups, of dimension 248, with a singularly rich combinatorial structure. The product E₈ × E₈ arises naturally in heterotic string theory. K₇ does not attempt to embed the Standard Model particles directly in E₈: an approach whose mathematical impossibility was proven by Distler and Garibaldi in 2010. E₈ × E₈ plays a different role in K₇: it supplies the gauge architecture, while matter content and parameter values emerge from the geometry of K₇.

## How the framework was built: a story in three acts

This is where the argument becomes interesting, and where the history matters.

The initial version of K₇, in 2025, carried **four free geometric parameters**: quantities *ξ*, *τ*, *β₀*, *δ* encoding various aspects of the geometric architecture, tuned to reproduce about 22 observables with a mean precision of 0.38%. This was already more constrained than the Standard Model itself (four inputs for twenty-two outputs), but it retained some slack: there was enough latitude in the choice of those four parameters to leave the framework vulnerable to the charge of fitting.

A second version, at the end of 2025, reduced that count to **three topological parameters**: *p₂* = 2 (a binary duality emerging from dim(G₂)/dim(K₇)), *β₀* = π/8 (an angular quantization tied to the rank of E₈), and a pentagonal factor linked to the Weyl group structure. These three parameters were no longer really free: each was a ratio of topological integers. The number of observables covered had risen to 34, with mean precision 0.13%.

The current version, K₇ v3.4, took a further step. **There are no continuously adjustable physical parameters.** The structure is entirely fixed by the choice (*b₂*, *b₃*) = (21, 77), the algebraic properties of E₈ × E₈, and a metric normalization target det(g) = 65/32. Ninety-five observables follow, organized in four types: 33 direct algebraic Type I predictions (mean deviation 0.99%), 19 one-step physical extractions (Type II, 0.17%), 21 multi-step dynamical chains (Type III, 3.44%), and 22 structural diagnostics (Type IV). Of the 95 observables, 55 are formally verified in Lean 4 (140 certificate conjuncts, 15 axioms (4 main-chain + 11 interval-arithmetic), 0 sorry).

This trajectory, viewed from the outside, deserves a moment's pause.

In general, when a theoretical framework is tuned to data (what statisticians call an *overfit*) it has to *increase* its degrees of freedom in order to maintain or improve its fit quality as it is asked to cover more observations. This is almost tautological. Add observations without adding parameters and fit quality degrades.

K₇ did the opposite. At each iteration, the number of adjustable parameters went down, the number of observables covered went up, and precision held or improved. This trajectory (4 → 3 → 0 parameters) is an unusual epistemic signature. It suggests that the topological constraints, as they tightened, were genuinely absorbing the degrees of freedom that had previously seemed necessary. In other words: what looked like tunable slack in the early versions turned out to be a consequence of the structure.

This is not proof that K₇ is correct. But it is proof that *what is happening during iteration is not fitting*. It is compression. And compression that holds, historically in physics, is what precedes moments of new understanding.

## The predictions that no longer move

At the end of this trajectory, certain relations have revealed themselves to be **exact topological identities**: integer ratios that fall directly out of the structure, with no numerical coefficient to tune. These are the strongest predictions of the framework, because they leave no room for arbitrariness.

**The number of generations**: *N*_gen = rank(E₈) − pentagonal factor = 8 − 5 = 3. A subtraction of integers. Exactly three, not two or four. The observed value is 3.

**The strange-to-down mass ratio**: *m*_s / *m*_d = 2² × 5 = 20. A product of integers. The value obtained from lattice QCD is 20.0 ± 1.0.

**The tau-to-electron mass ratio**: *m*_τ / *m*_e = dim(*K₇*) + 10 × dim(E₈) + 10 × *H*\* = 7 + 2480 + 990 = 3477. A sum of integer dimensions. The experimental value is 3477.0 ± 0.5.

**The Koide parameter**: *Q* = dim(G₂) / *b*₂(*K₇*) = 14 / 21 = 2/3. An exact rational ratio. The experimental value is 0.6667 ± 0.0001. This relation, observed for more than forty years, had no theoretical explanation.

**The neutrino CP-violating phase**: *δ*_CP = 7 × dim(G₂) + *H*\* = 7 × 14 + 99 = 197°. An additive combination of topological invariants. Under NuFIT 6.1 the best fit has moved to about 207° (no-SK) / 212° (SK), so the predicted 197° sits within about 1σ. This is the framework's most important falsifiable prediction: the DUNE experiment (2028–2040) will measure this phase with a precision that settles it decisively.

Alongside these exact identities, the framework predicts the ten elements of the CKM matrix with mean deviation 0.11%, the three neutrino mixing angles to better than 0.5%, the inverse fine-structure constant to 0.002%, nine quark mass ratios to 0.09% on average, and several cosmological observables (including the dark-to-baryonic matter ratio Ω_DM/Ω_b = 43/8) with comparable precision.

## Why the coincidence hypothesis can be ruled out

Faced with a constellation of formulas this precise, the question is unavoidable: isn't this simply the result of an extended fishing expedition in an ocean of possible algebraic combinations? With enough tries, one eventually hits formulas that match.

The honest answer, in two parts.

First, yes, some formulas were indeed found by automated search over restricted algebraic grammars. This is not the only mode of discovery; a substantial part of the framework emerges from direct structural constraints. But it would be dishonest to pretend that every relation was divined by pure geometric intuition. Brute exploration is part of the method, as it has always been in mathematical physics.

Second, and this is the decisive point: during that exploration, formulas were encountered that produced *better* numerical deviations than those ultimately retained, but which had no structural meaning within the framework. They were set aside. That qualitative criterion (a formula must fit the geometric architecture, not merely match the data) is what distinguishes theoretical compression from numerical fitting. A numerological framework keeps whatever fits best; a structural framework keeps whatever sits inside the structure, even at the cost of slightly lower precision.

Beyond this methodological discipline, several statistical analyses documented in the v3.4 main paper test the coincidence hypothesis directly.

**Combinatorial test**: the configuration (*b*₂, *b*₃) = (21, 77) was compared against 3,070,396 alternatives tested within a bounded space, including thirty G₂ manifolds explicitly known in the mathematical literature. It remains optimal with *p* < 2 × 10⁻⁵, corresponding to significance above 4.2σ. It is also Pareto-optimal: no other tested configuration simultaneously improves on multiple criteria.

**Multiple comparisons correction**: the Westfall–Young maxT procedure, which controls the family-wise error rate, confirms that 11 of the 33 predictions remain individually significant after correction, with global significance *p* = 0.008.

**Leave-one-out validation**: in 28 independent tests where one observable is withheld and the others used to reconstruct the framework, the (21, 77) configuration remains the unique optimum every time. The withheld prediction is then compared against observation: agreement holds without exception.

**Bayesian comparison**: Bayes factors between K₇ and reasonable null models range from 288 to 4,567, placing the result in the "decisive evidence" category by Jeffreys's conventions.

None of these tests constitutes proof that K₇ is correct. They constitute proof that the observed values are not a combinatorial accident. That distinction matters.

## Falsifiability

The framework is, by construction, without tunable parameters. This means it cannot be "saved" by retouching a constant to absorb an experimental disagreement. Any significant deviation between a K₇ prediction and a sufficiently precise measurement refutes it.

The decisive upcoming tests are clearly identified.

**DUNE** will measure *δ*_CP with an expected precision of a few degrees by 2035. A measured value outside the [182°, 212°] window (more than 15° from the predicted 197°) falsifies the framework.

**Hyper-Kamiokande** will measure *θ*_23 with sub-degree precision. The framework predicts the rational value 85/99 rad = 49.19°. A significant disagreement falsifies.

**The discovery of a fourth generation of matter** at any scale accessible to future colliders falsifies the framework immediately: *N*_gen = 3 is an exact topological consequence, not an empirical observation.

**Future cosmological measurements** (Euclid, CMB-S4) will test predictions for the scalar spectral index, the dark energy density, and other large-scale observables.

The timelines run over 2027–2040. Until then, the framework remains a proposal to be examined, not a validated theory.

## What K₇ is not

A new theoretical framework in a mature discipline benefits from defining its limits as sharply as its content.

K₇ is not a theory of everything. It does not claim to explain quantum gravity, resolve the measurement problem in quantum mechanics, or unify all forces in a single framework. It confines itself to a specific problem: deriving the dimensionless parameters of the Standard Model from a fixed geometric structure. The rest is outside its current scope.

K₇ is not a cosmogony. It proposes no mechanism for the origin of the universe, for the emergence of spacetime, or for the selection of the manifold K₇ among all mathematically possible manifolds. The question "why *this* topology rather than another" remains open, and the framework explicitly acknowledges this as a limit. The only defense available at this stage is statistical: this configuration is optimal among those tested. *Why* it is so remains unknown.

K₇ does not reduce to any single reading of the word *information*. The name *Geometric Information Field Theory* deserves to be unfolded across three distinct registers.

**At the first register, the empirical predictions depend on no ontological thesis.** The thirty-three derived relations, the numerical deviations, the planned falsification tests, all of these stand independently of what "information" ultimately is. An instrumentalist physicist can work with K₇ by treating its predictions as algebraic consequences of topological invariants, without ever committing further.

**At the second register, the framework's architecture sits within a well-established physics lineage.** The choice of a G₂ manifold coupled to E₈ × E₈, of Betti numbers as primitives, of the total cohomological dimension *H\** as algebraic organizer, belongs to a precise program. Wheeler proposed as early as 1989 that "it from bit" (that matter, energy, and spacetime emerge from information-theoretic foundations. Bekenstein established since the 1970s that the informational capacity of a region is bounded by its area, not its volume. Jacobson derived in 1995 Einstein's equations as a thermodynamic equation of state for observer horizons. Van Raamsdonk showed in 2010 how spacetime connectivity emerges from quantum entanglement. Thirty years of mainstream physics) the holographic principle, black hole thermodynamics, entanglement entropy, have made the idea that geometry, information, and energy are three readings of a single structural organization far less exotic than it sounds. The Betti numbers are not convenient labels in K₇: they are dimensions of harmonic form spaces, hence exact counts of degrees of freedom: the very notion of informational capacity applied to a geometric structure.

**At the third register, the ontological thesis proper** (that geometry, information, and energy are not three correlated aspects but three views of a single underlying configuration) K₇ does not settle. It is compatible with the framework, it motivates the architecture, but it is neither required nor demonstrated by the predictions. Its elaboration is the subject of a separate text (see the companion post [*Gift from Bit*](https://arithmon.substack.com/p/gift-from-bit)). A reader who finds Wheeler prophetic and the holographic program persuasive will see K₇ as a natural piece of that puzzle. A reader who prefers to stay empirical will see a falsifiable predictive framework. Both readings are defensible; the framework requires neither.

Finally, K₇ does not belong to the tradition of unification attempts through direct embedding of Standard Model particles into E₈. That approach, associated with Lisi (2007), meets a mathematical impossibility proven by Distler and Garibaldi in 2010. K₇ uses E₈ × E₈ differently, as gauge architecture, with particles emerging from the cohomology of K₇ rather than from a direct E₈ representation.

## The open questions

The framework has weak points its authors have no interest in hiding.

**The selection principle remains open.** No first-principles argument currently derives (*b*₂, *b*₃) = (21, 77) from deeper constraints. The configuration is *selected* by its agreement with experiment, not *deduced* from first principles. Statistical tests show it is optimal, not that it is necessary.

**The explicit construction of K₇ is a program in progress.** The Betti numbers (21, 77) are plausible within the *twisted connected sum* landscape, which produces compact G₂ manifolds. But producing an explicit example with precisely these invariants, and computing directly the volume integrals and Yukawa couplings that would follow, remains a goal to be reached. A first numerical prototype of a local G₂ metric has been produced and verified by the Newton–Kantorovich method, but the full link to the spectrum of observed parameters is yet to be established formally.

**Formulas involving transcendental constants** (where *π*, *ζ*(3), *γ*, or the golden ratio *φ* appear) have a more speculative status than the exact topological identities. The framework treats them as *phenomenological* pending a rigorous derivation from geometry. That floor of the building still needs work.

These limits are acknowledged, documented, and do not obscure what the framework does well.

## What remains to be seen

K₇ is a proposal. No more, no less. A mathematically precise proposal, empirically precise at present, and falsifiable through experiments already on the books.

Its value, if any, will be judged in the years to come. If DUNE measures *δ*_CP at 197° to within a few degrees, the framework passes a hard test. If it measures outside the prediction window, the framework is set aside. If a fourth generation of matter is discovered, the framework is set aside. If the predictions hold, further digging is called for.

What makes this proposal worth examining is not that it would be elegant, many elegant proposals have turned out to be wrong. It is that it was built through a process in which the degrees of freedom *decreased* at each step while *more* observables were covered, and that it has reached a state where it is no longer adjustable. It is either correct or wrong, and experiment will decide.

In a discipline where many recent frameworks struggle to produce near-term testable predictions, that is already an unusual position.

---

*For primary sources, formal proofs in Lean 4, complete derivations of the 33 predictions, and detailed statistical analyses, see the main manuscript on Zenodo (DOI [10.5281/zenodo.18837071](https://doi.org/10.5281/zenodo.18837071)) and the code repository on [GitHub](https://github.com/Arithmon/K7).*
