---
title: "Blog 13 Theorems Zero Trust"
layout: default
---

> Originally published on [giftheory.substack.com](https://giftheory.substack.com/p/13-theorems-zero-trust-required)

# 13 Theorems, Zero Trust Required
GIFT meets Lean 4

Dec 03, 2025

“Prove it.”

Fair enough.

GIFT is now Lean 4 verified. 13 exact relations derived from topology, checked by a proof assistant. Zero sorry. Only standard axioms. The arithmetic compiles.

You don’t have to trust the math. Lean checked it.
The problem with pen and paper

Theoretical physics has a quiet reproducibility issue. Not fraud, just complexity. Derivations span pages. Indices multiply. Sign errors hide. A factor of 2 slips in somewhere around equation 47.

Peer review catches some mistakes. But reviewers are human, busy, and rarely re-derive everything from scratch. The result: papers get published, errors persist, and years later someone finds a minus sign that changes everything.

What if a machine checked every step?
Enter Lean

Lean is a proof assistant, software that verifies mathematical proofs down to foundational axioms. If a theorem compiles, it’s correct. No ambiguity. No “trust me.”

The Lean community has used it to formalize serious mathematics: the Mathlib library covers thousands of theorems, contributors are working toward Fermat’s Last Theorem, and Fields medalists take it seriously.

The question was simple: can GIFT’s core relations be formalized? Not the physics interpretation, just the arithmetic. Do the numbers actually work out?
What the formalization proves

GIFT claims that physical constants emerge from topological invariants. The Lean formalization verifies a precise statement:

IF the following integers are fixed:

    dim(E₈) = 248

    b₂(K₇) = 21

    b₃(K₇) = 77

    dim(G₂) = 14

    dim(J₃(𝕆)) = 27

THEN these exact relations hold:

Relation Formula Value Proof sin²θ_W b₂ / (b₃ + dim G₂) 3/13 norm_num τ (496 × 21) / (27 × 99) 3472/891 norm_num det(g) (5 × 13) / 32 65/32 norm_num κ_T 1 / (77 − 14 − 2) 1/61 norm_num δ_CP 7 × 14 + 99 197° rfl m_τ/m_e 7 + 10×248 + 10×99 3477 rfl m_s/m_d 4 × 5 20 rfl Q_Koide 14 / 21 2/3 norm_num λ_H numerator 14 + 3 17 rfl H* 21 + 77 + 1 99 rfl p₂ 14 / 7 2 rfl N_gen topological 3 rfl dim(E₈×E₈) 2 × 248 496 rfl

Each relation has a standalone theorem. Each theorem compiles. The main theorem bundles all 13:

theorem GIFT_framework_certified (G : GIFTStructure) 
    (h : is_zero_parameter G) :
    (G.b2 : ℚ) / (G.b3 + G.dim_G2) = 3 / 13 ∧
    (G.dim_E8xE8 * G.b2 : ℚ) / (G.dim_J3O * G.H_star) = 3472 / 891 ∧
    -- ... 11 more conjuncts
    G.dim_E8xE8 = 496 := by
  obtain ⟨he, hr, hw, hk, hb2, hb3, hg, hj⟩ := h
  refine ⟨?_, ?_, ?_, ...⟩ <;> simp_all <;> norm_num

Axiom audit

A natural question: what axioms does the proof depend on? Hidden assumptions could undermine everything.

#print axioms GIFT_framework_certified
-- [propext, Quot.sound]

Two axioms. Both are standard Lean foundations:

Axiom Description Status propext Propositional extensionality Standard Quot.sound Quotient soundness Standard

No physics axioms. No domain-specific assumptions. The proof is pure arithmetic from fixed integers.
What the formalization does NOT prove

Clarity matters. The Lean code proves:

IF these topological integers THEN these ratios.

It does not prove the IF. The claim that b₂(K₇) = 21 and b₃(K₇) = 77 are the correct values for our universe, that’s physics, not mathematics. That claim is empirical and falsifiable.

Experiments will test it:

Prediction Value Experiment Timeline δ_CP 197° DUNE, Hyper-K 2027-2030 sin²θ_W 3/13 = 0.23077 FCC-ee 2040s κ_T 1/61 DESI 2025-2027

If DUNE measures δ_CP = 250° ± 10°, the framework is falsified. No reinterpretation. No parameter adjustment. Dead.

The Lean formalization doesn’t protect against that. It only guarantees: if the topology is right, the arithmetic is right.
Repository structure

The formalization is modular:

Lean/
├── GIFT/
│   ├── Algebra/           # E₈ root system, Weyl group, representations
│   ├── Geometry/          # G₂ holonomy, TCS construction
│   ├── Topology/          # Betti numbers, cohomology
│   ├── Relations/         # Gauge, neutrino, quark, lepton, Higgs, cosmology
│   └── Certificate/       # Main theorems, axiom audit

17 modules. ~2000 lines of Lean. Each sector (gauge, neutrino, quark, etc.) has its own file with standalone theorems.

Build it yourself:

git clone https://github.com/gift-framework/GIFT.git
cd GIFT/Lean
lake update
lake exe cache get   # Download Mathlib cache (~2GB)
lake build           # ~5 min with cache

Or just read the proofs. They’re short. Most are one line.
Why this matters

Formal verification isn’t common in theoretical physics. Most papers rely on peer review and reputation. But proofs are proofs, they shouldn’t depend on who wrote them.

The Lean formalization sets a precedent:

    Reproducibility: Anyone can verify the arithmetic. Clone, build, check.

    Transparency: No hidden assumptions. Axioms are listed.

    Precision: 3/13 means exactly 3/13, not “approximately 0.231.”

This won’t convince anyone that GIFT is correct physics. That’s what experiments are for. But it removes one class of objections entirely: the arithmetic is not wrong. Lean checked it.
Conclusion

13 theorems. Zero sorry. Standard axioms only.

The relations between topological invariants and physical constants are now machine-verified. The derivations are not approximations or fits: they’re exact rational arithmetic, proven from fixed integers.

The physics remains to be tested. The math is settled.

Repository: https://github.com/gift-framework/core/

Feedback, corrections, and brutal criticism welcome via GitHub issues.