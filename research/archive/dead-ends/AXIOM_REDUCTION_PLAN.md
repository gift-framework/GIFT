# Axiom Reduction Plan

**Created**: 2026-01-29
**Goal**: Reduce ~47 axioms to irreducible geometric definitions only

---

## Current Axiom Inventory

| Category | Count | Priority | Effort |
|----------|-------|----------|--------|
| π bounds | 3 | 🔴 HIGH | ~1 day |
| Spectral theory | 15 | 🟡 MEDIUM | ~2 weeks |
| Literature (cited) | 8 | 🟢 LOW | 2-4 weeks/paper |
| Zeta numerical | 18 | 🟢 LOW | Keep as data |
| Yang-Mills structure | ~10 | 🟡 MEDIUM | ~3 weeks |
| Geometric/definitional | 6 | ⚪ KEEP | Irreducible |

---

## Priority 1: π Bounds (3 axioms) 🔴

**Axioms**:
- `pi_gt_three` : π > 3
- `pi_lt_four` : π < 4
- `pi_lt_sqrt_ten` : π < √10

**Problem**: Mathlib 4.27.0 has incomplete `Real.pi` API.

**Solutions** (pick one):

### Option A: Upgrade Mathlib (recommended)
```bash
# In lakefile.toml, update to Mathlib ≥ 4.28
lake update
```
Check if `Real.pi_lt_four` exists in newer Mathlib.

### Option B: Import interval arithmetic library
```lean
import Mathlib.Analysis.SpecialFunctions.Pow.Real
-- Use Real.pi bounds from Mathlib.Analysis.SpecialFunctions.Pi.Bounds
```

### Option C: Prove manually via series
```lean
-- π = 4 * (1 - 1/3 + 1/5 - 1/7 + ...)
-- Truncate series to get bounds
theorem pi_lt_four : Real.pi < 4 := by
  have h := Real.pi_lt_315 / 100  -- if available
  linarith
```

**Estimated effort**: 1 day

**Files to modify**:
- `Lean/GIFT/Spectral/SelectionPrinciple.lean`

---

## Priority 2: Spectral Theory (15 axioms) 🟡

### Tier 2a: Abstract definitions (keep as structure)

These define WHAT we're talking about:
- `CompactManifold : Type`
- `LaplaceBeltrami.canonical`
- `MassGap`

**Strategy**: Keep as typeclass/structure. Not claims, just definitions.

### Tier 2b: Standard theorems (prove or cite)

| Axiom | Standard Result | Source |
|-------|-----------------|--------|
| `spectral_theorem_discrete` | Compact self-adjoint has discrete spectrum | Functional analysis |
| `weyl_law` | Eigenvalue asymptotics | Weyl 1911 |
| `rayleigh_quotient_characterization` | Min-max principle | Courant-Hilbert |

**Strategy**:
1. Check if Mathlib has these (likely not for manifolds)
2. If not, keep as axioms but add `-- Standard result: [citation]`
3. Long-term: Contribute to Mathlib manifold library

### Tier 2c: TCS-specific (literature axioms)

| Axiom | Paper | Status |
|-------|-------|--------|
| `mass_gap_decay_rate` | Mazzeo-Melrose | Cited |
| `cheeger_lower_bound` | Cheeger 1970 | Cited |
| `cgn_no_small_eigenvalues` | Crowley-Goette-Nordström 2024 | Cited |

**Strategy**: Keep as literature axioms. Add detailed citations.

**Estimated effort**: 2 weeks to clean up and properly cite.

**Files to modify**:
- `Lean/GIFT/Spectral/SpectralTheory.lean`
- `Lean/GIFT/Spectral/RefinedSpectralBounds.lean`
- `Lean/GIFT/Spectral/LiteratureAxioms.lean`

---

## Priority 3: Yang-Mills Structure (~10 axioms) 🟡

These define gauge theory setup:
- `CompactSimpleGroup`, `SU : ℕ → CompactSimpleGroup`
- `Connection`, `Curvature`, `YangMillsAction`
- `yang_mills_nonneg`, `flat_connection_minimizes`
- `YangMillsHamiltonian`, `vacuum_energy`, `first_excited_energy`

**Strategy**:
1. These are DEFINITIONS, not theorems
2. Rename to make clear they are structure, not claims
3. The GIFT claim is: "On G₂ manifolds, mass_gap = 14/99"
4. The gauge setup is just context

**Estimated effort**: 3 weeks to build proper gauge theory foundation.

**Alternative**: Keep as axioms, document as "gauge theory setup assumed".

---

## Priority 4: Zeta Numerical (18 axioms) 🟢

**Axioms**: `gamma_1_approx`, `gamma_2_approx`, ..., `gamma_107_approx`

**Strategy**: KEEP. These are data, not claims.
- Document as: "Zeta zeros from Odlyzko tables"
- Could add verification against LMFDB API
- Not a proof bottleneck

---

## Priority 5: Literature Axioms (8 axioms) 🟢

| Axiom | Paper | Action |
|-------|-------|--------|
| `langlais_spectral_density` | Langlais 2024 | Read & formalize |
| `cgn_cheeger_lower_bound` | CGN 2024 | Read & formalize |
| `torsion_free_correction` | Joyce 1996 | Already have IFT |
| `canonical_neck_length_conjecture` | TCS literature | Cite multiple sources |
| `selection_principle_holds` | GIFT | This IS the claim |

**Strategy**:
- `selection_principle_holds` should stay as the CLAIM (not axiom)
- Others: formalize if important, otherwise keep with citations

---

## Irreducible (6 axioms) ⚪ KEEP

| Axiom | Why Irreducible |
|-------|-----------------|
| `K7 : Type` | Definition of the manifold |
| `K7_hodge_data` | Hodge structure (topological) |
| `hodge_theorem_K7` | Classical result (cite Hodge) |
| `CompactManifold` | Type definition |
| `Riemannian` | Structure definition |
| `G2Holonomy` | Structure definition |

These are not claims - they're the mathematical objects we reason about.

---

## Execution Plan

### Week 1: π bounds
- [ ] Check Mathlib 4.28+ for `pi_lt_four`
- [ ] If not, implement interval arithmetic proof
- [ ] Update SelectionPrinciple.lean
- [ ] Run full build

### Week 2-3: Spectral cleanup
- [ ] Add proper citations to all spectral axioms
- [ ] Rename definitional axioms to `structure` or `class`
- [ ] Separate "standard results" from "GIFT claims"
- [ ] Document which are truly irreducible

### Week 4+: Optional refinements
- [ ] Formalize Langlais 2024 (if tractable)
- [ ] Build gauge theory foundation (long-term)
- [ ] Contribute manifold spectral theory to Mathlib

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Total axioms | ~47 | ~25 |
| π axioms | 3 | 0 |
| Undocumented axioms | ~10 | 0 |
| Axioms clearly labeled | ~60% | 100% |

---

## Non-Goals

- **Not trying to prove RH** - zeta axioms are data
- **Not building full gauge theory** - too big for this scope
- **Not formalizing all literature** - diminishing returns

---

*The goal is clarity, not minimalism. A well-documented axiom is better than a poorly-understood proof.*
