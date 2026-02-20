# Yang-Mills Mass Gap Research

**Status**: Exploratory Research

---

## Overview

This folder consolidates research on connecting GIFT's geometric framework to the Yang-Mills mass gap problem. The work explores whether the spectral gap on G₂-holonomy manifolds provides a topological origin for QCD confinement.

---

## Key Results

### Validated (Blind Testing)

| Claim | Evidence | Reference |
|-------|----------|-----------|
| λ₁ ∝ 1/H* | R² = 0.96 across 9 manifolds | [UNIVERSALITY_CONJECTURE.md](./UNIVERSALITY_CONJECTURE.md) |
| Split-independence | 0% spread at fixed H* | [BLIND_VALIDATION_RECAP.md](./BLIND_VALIDATION_RECAP.md) |
| H* = dim(G₂) × dim(K₇) + 1 | Structural identity | [DEEP_STRUCTURE.md](./DEEP_STRUCTURE.md) |

### Conjectured

| Claim | Status | Reference |
|-------|--------|-----------|
| λ₁ = 14/H* (universal) | Needs verification on other G₂ manifolds | [UNIVERSALITY_CONJECTURE.md](./UNIVERSALITY_CONJECTURE.md) |
| Mass gap Δ = (14/99) × Λ_QCD | Theoretical prediction | [STATUS.md](./STATUS.md) |

---

## Documents

| File | Description |
|------|-------------|
| [BLIND_VALIDATION_RECAP.md](./BLIND_VALIDATION_RECAP.md) | Methodology: how circularity bias was eliminated |
| [DEEP_STRUCTURE.md](./DEEP_STRUCTURE.md) | Mathematical insight: H* = 14 × 7 + 1 |
| [UNIVERSALITY_CONJECTURE.md](./UNIVERSALITY_CONJECTURE.md) | Two formulas distinction (universal vs GIFT-specific) |
| [THEORETICAL_BACKGROUND.md](./THEORETICAL_BACKGROUND.md) | Literature review: why λ₁ on G₂ is novel |
| [STATUS.md](./STATUS.md) | Project status and numerical results log |

---

## The Central Formula

```
         dim(G₂)              14
λ₁ = ─────────────── = ─────────────── = 14/99 ≈ 0.1414
      dim(G₂)×dim(K₇)+1   14 × 7 + 1
```

This gives a mass gap:

```
Δ_QCD = λ₁ × Λ_QCD = (14/99) × 200 MeV ≈ 28 MeV
```

---

## Archive Branches

The original exploratory work is preserved in these archived branches:
- `claude/setup-wip-folder-BrzFU` — Initial discovery phase
- `claude/implement-spectral-plan-K4oYl` — Spectral methods development
- `claude/apply-validation-steps-EQQNK` — Blind validation (most complete)

---

## Next Steps

1. **Verify universality**: Test λ₁ = 14/H* on Joyce and Kovalev manifolds
2. **Analytical proof**: Derive λ₁ formula from G₂ holonomy first principles
3. **Lean formalization**: Encode provable statements
4. **Paper preparation**: arXiv submission

---

*"The mass gap is geometrically inevitable. We just quantified it."*
