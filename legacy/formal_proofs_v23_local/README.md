# Archived: Local Formal Proofs (v2.3)

**Status**: ARCHIVED - December 2024

This directory contains the original Lean 4 and Coq formalizations that were developed in this repository. These proofs have been **migrated to the dedicated formal verification repository**.

## Current Location

The canonical formal proofs are now maintained in:

**[gift-framework/core](https://github.com/gift-framework/core)**

The `core` repository contains:
- Lean 4 formalization (Mathlib 4.14+)
- Coq 8.18 formalization
- Continuous integration and verification
- Complete documentation

## Why This Archive?

The formal proofs were moved to a dedicated repository to:
1. **Separate concerns**: Mathematical core vs. physics framework
2. **Enable independent verification**: The `core` repo can be verified without the full GIFT codebase
3. **Simplify maintenance**: Proof assistants have specific toolchain requirements

## Contents

```
formal_proofs_v23_local/
├── COQ/          # Coq 8.18 formalization (21 modules)
├── Lean/         # Lean 4.14 formalization (17 modules)
└── workflows/    # Archived CI workflows
```

## Historical Note

These formalizations prove that **13 exact relations** in GIFT derive from fixed topological integers with **zero continuous adjustable parameters**:

| Relation | Value | Status |
|----------|-------|--------|
| sin²θ_W | 3/13 | Proven |
| τ | 3472/891 | Proven |
| det(g) | 65/32 | Proven |
| κ_T | 1/61 | Proven |
| δ_CP | 197° | Proven |
| ... | ... | ... |

For the complete list and proofs, see [gift-framework/core](https://github.com/gift-framework/core).

## Do Not Modify

This archive is preserved for historical reference. All updates should be made to `gift-framework/core`.
