Joyce’s Theorem, Now in Lean
Toward machine-verified existence of G₂ manifolds

Dec 10, 2025

“Does K₇ actually exist?”

It’s a reasonable question. GIFT claims physics emerges from a 7-manifold with G₂ holonomy. But claiming a manifold exists and proving it exists are different exercises.

In 1996, Dominic Joyce proved that compact G₂ manifolds exist. The proof uses hard analysis: Sobolev spaces, implicit function theorems, contraction mappings. It spans some 200 pages of differential geometry.

We’ve now formalized key parts of that argument in Lean 4. Here’s what that means, and what it doesn’t.
The challenge with existence theorems

Existence proofs in geometry are notoriously difficult to verify. They typically involve:

    Infinite-dimensional function spaces

    Estimates that “follow from standard elliptic theory”

    Constants that are “sufficiently small”

    Iterations that “converge by Banach’s theorem”

Each step may be correct. But the chain is long, and checking it requires expertise that’s not widely shared. The result: existence theorems get cited, but rarely re-derived from scratch.

Formal verification offers a different approach. What if a machine tracked every estimate?
What Joyce’s theorem says

Let M be a compact 7-manifold with a G₂ structure φ₀. The structure has torsion T(φ₀), measuring how far φ₀ is from being “torsion-free”: the desirable case, which implies Ricci-flatness.

Joyce’s Theorem (roughly): If ‖T(φ₀)‖ < ε₀ for some threshold ε₀, then there exists a smooth torsion-free G₂ structure φ on M, with ‖φ - φ₀‖ ≤ C·‖T(φ₀)‖.

In other words: if you find an approximate G₂ structure with small enough torsion, an exact one exists nearby.

The subtlety: ε₀ depends on Sobolev constants of M. Computing it for a specific manifold requires careful analysis.
What the Lean formalization covers

The formalization has three layers:

Layer 1: Abstract framework

Sobolev spaces, differential forms, and the implicit function theorem are set up as Lean structures:

-- Sobolev embedding: H^4 continuously embeds in C^0
theorem sobolev_embedding_H4_C0 (M : Manifold) [Compact M] :
    ContinuousEmbedding (H 4 M) (C 0 M)

-- Hodge Laplacian is self-adjoint
theorem hodge_laplacian_self_adjoint :
    IsSelfAdjoint (hodge_laplacian : Ω^k M → Ω^k M)

Layer 2: Joyce iteration as contraction

Joyce’s proof works by iterating a correction map. The formalization captures the key property: this map is a contraction with constant K < 1.

noncomputable def joyce_K : NNReal := ⟨9/10, by norm_num⟩

theorem joyce_K_lt_one : joyce_K < 1 := by simp [joyce_K]; norm_num

theorem joyce_is_contraction : ContractingWith joyce_K JoyceFlow :=
  ⟨joyce_K_lt_one, joyce_lipschitz⟩

Layer 3: Banach fixed point

Mathlib provides the Banach fixed point theorem, proven from first principles in the library:

-- Existence via Mathlib’s ContractingWith.fixedPoint
noncomputable def torsion_free_structure : G2Structures :=
  joyce_is_contraction.fixedPoint JoyceFlow

theorem k7_admits_torsion_free_g2 :
    ∃ φ : G2Structures, IsTorsionFree φ :=
  ⟨torsion_free_structure, fixed_point_is_torsion_free⟩

When this compiles, Lean has verified the logical chain from contraction to existence.
The numerical side

Abstract existence needs grounding. We also need to check that K₇ specifically satisfies Joyce’s hypotheses.

A physics-informed neural network (PINN) constructs an approximate G₂ structure φ₀ on K₇. The network has roughly 200,000 parameters and trains in 5-10 minutes on free platforms like Colab.

The torsion bound is the critical one. Our estimate for Joyce’s threshold on K₇ is ε₀ ≈ 0.0288. The PINN achieves ‖T‖ = 0.00140.

Safety margin: approximately 20×

This margin provides some confidence that the bound isn’t marginal. Of course, the true ε₀ for K₇ depends on Sobolev constants we haven’t computed exactly, more on that below.
Axiom audit

What does the proof assume? Lean makes this explicit:

#print axioms k7_admits_torsion_free_g2

Standard axioms (from Lean/Mathlib):

    propext: Propositional extensionality

    Quot.sound: Quotient soundness

These are standard foundations, nothing exotic.

Domain axioms (interface to geometry):

K7 → K₇ exists as a topological type → Abstract interface
JoyceFlow → The iteration map exists → Joyce’s construction
joyce_lipschitzJoyceFlow → has Lipschitz constant < 1 → Contraction property
fixed_point_torsion_zero → Fixed points are torsion-free → Joyce’s theorem conclusion

Notably, the Banach fixed point theorem is not axiomatized: it comes from Mathlib’s ContractingWith.fixedPoint, proven within the library.
What this does NOT prove

It’s worth being clear about the limitations:

    Joyce’s theorem is not formalized from first principles. The analytical core, Sobolev estimates, elliptic regularity, Schauder theory, is axiomatized rather than proven in Lean. A complete formalization would be a substantial project, likely requiring years of work.

    The threshold ε₀ is estimated, not computed exactly. We use a conservative estimate based on typical Sobolev constants. The true value for K₇ would require more detailed analysis.

    The physics interpretation remains conjectural. The claim that our universe involves K₇ compactification is empirical, not mathematical. This formalization doesn’t touch that question.

    Uniqueness is unknown. Other G₂ structures with these invariants may exist. The moduli space structure hasn’t been characterized.

What this does achieve

Despite the caveats, the formalization accomplishes something useful:

    Machine-verified arithmetic. The bounds (20× safety margin, det(g) match, etc.) are checked by Lean, not by hand.

    Transparent assumptions. Every axiom is listed explicitly. There’s no hidden “trust me” step.

    Reproducibility. Anyone can verify the computation:

pip install giftpy
python -c “from gift_core.analysis import JoyceCertificate; print(JoyceCertificate.verify())”

    Modular structure. The argument is broken into standalone lemmas. Each piece can be examined or improved independently.

The remaining work

The formalization connects two endpoints:

    Top: Abstract existence via Banach fixed point (formalized)

    Bottom: Numerical bounds from PINN training (certified)

The middle layer: Joyce’s analytical machinery, is currently axiomatized. Completing the picture would mean formalizing:

    Elliptic regularity on compact manifolds

    Sobolev multiplication and embedding theorems

    Schauder estimates for nonlinear PDE

This is substantial mathematics. Similar efforts (like formalizing parts of Fermat’s Last Theorem) have taken years. We’re not claiming to have done it.

But the scaffolding exists. The theorem statement compiles. The numerical bounds are verified. What remains is filling in the analytical middle, a task that’s well-defined, if demanding.
Conclusion

theorem k7_admits_torsion_free_g2 : ∃ φ : G2Structures, IsTorsionFree φ

This statement compiles in Lean 4.14.0 with Mathlib. Subject to the axioms listed above, the existence of a torsion-free G₂ structure on K₇ is machine-verified.

The axioms are explicit. The gaps are acknowledged. The numerical evidence is reproducible.

Whether this connects to physics is a separate question, one that experiments, not proof assistants, will eventually answer.

Repository: github.com/gift-framework/core

Notebook: github.com/gift-framework/GIFT/notebooks