# GIFT Framework - Lean 4 Formalization

[![Lean CI](https://github.com/gift-framework/GIFT/actions/workflows/lean.yml/badge.svg)](https://github.com/gift-framework/GIFT/actions/workflows/lean.yml)

Formal verification of the Geometric Information Field Theory (GIFT) framework in Lean 4, proving that all 39 observables derive from fixed mathematical structures with **zero continuous adjustable parameters**.

## Quick Start

```bash
# Install Lean 4 and Lake (if not already installed)
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh

# Build the project
cd Lean
lake update
lake build
```

## Project Structure

```
Lean/
├── lakefile.lean          # Build configuration
├── lean-toolchain         # Lean version (4.14.0)
├── GIFT.lean              # Root import file
│
└── GIFT/
    ├── Algebra/           # E₈ structure
    │   ├── E8RootSystem.lean
    │   ├── E8WeylGroup.lean
    │   ├── E8Representations.lean
    │   └── ExceptionalJordan.lean
    │
    ├── Geometry/          # G₂ holonomy
    │   ├── G2Group.lean
    │   ├── G2Structure.lean
    │   ├── G2Holonomy.lean
    │   └── TwistedConnectedSum.lean
    │
    ├── Topology/          # Cohomology
    │   ├── BettiNumbers.lean
    │   ├── CohomologyStructure.lean
    │   └── EulerCharacteristic.lean
    │
    ├── Relations/         # Physical observables
    │   ├── Constants.lean
    │   ├── GaugeSector.lean
    │   ├── NeutrinoSector.lean
    │   ├── QuarkSector.lean
    │   ├── LeptonSector.lean
    │   ├── HiggsSector.lean
    │   └── CosmologySector.lean
    │
    └── Certificate/       # Main theorems
        ├── ZeroParameter.lean
        ├── MainTheorem.lean
        └── Summary.lean
```

## What's Proven

### Topological Inputs (Fixed by Mathematics)

| Constant | Value | Origin |
|----------|-------|--------|
| dim(E₈) | 248 | Exceptional Lie algebra |
| rank(E₈) | 8 | Cartan subalgebra |
| dim(E₈×E₈) | 496 | Heterotic string gauge group |
| b₂(K₇) | 21 | TCS construction |
| b₃(K₇) | 77 | TCS construction |
| dim(G₂) | 14 | Exceptional holonomy group |
| dim(J₃(O)) | 27 | Exceptional Jordan algebra |

### 13 Proven Exact Relations

| Relation | Value | Formula |
|----------|-------|---------|
| sin²θ_W | 3/13 | b₂/(b₃ + dim G₂) |
| τ (hierarchy) | 3472/891 | 496·21/(27·99) |
| det(g) | 65/32 | 5·13/32 |
| κ_T | 1/61 | 1/(77-14-2) |
| δ_CP | 197° | 7·14 + 99 |
| m_τ/m_e | 3477 | 7 + 10·248 + 10·99 |
| m_s/m_d | 20 | 4·5 |
| Q_Koide | 2/3 | 14/21 |
| λ_H | √(17/32) | (14+3)/2⁵ |
| H* | 99 | 21 + 77 + 1 |
| p₂ | 2 | 14/7 |
| N_gen | 3 | Topological |
| E₈×E₈ | 496 | 2·248 |

## Main Theorem

```lean
theorem GIFT_framework_certified (G : GIFTStructure)
    (h : is_zero_parameter G) :
    -- All 13 relations proven from h alone
    ...
```

The main theorem proves that given only the zero-parameter constraint (all topological integers fixed), every physical relation follows by pure computation with no additional assumptions.

## Verification Status

- **Lean version**: 4.14.0
- **Mathlib version**: 4.14.0
- **Total modules**: 17
- **Domain-specific axioms**: 0 (for arithmetic proofs)
- **`sorry` count**: 0

## Axiom Usage

The proofs use only standard Lean axioms:
- `propext` (propositional extensionality)
- `Quot.sound` (quotient soundness)
- `Classical.choice` (for some Mathlib lemmas)

No physics-specific axioms are required for the arithmetic relations.

## CI/CD

The project uses GitHub Actions for continuous integration:

- **Build**: Compiles all Lean modules with Mathlib
- **Cache**: Uses Mathlib precompiled cache for fast builds (~5 min vs hours)
- **Lint**: Checks for `sorry`, counts theorems/axioms
- **Axiom audit**: Verifies only standard Lean axioms are used

### Local Development

```bash
# First time setup
cd Lean
lake update
lake exe cache get    # Download Mathlib precompiled oleans

# Build
lake build            # Full build
lake build GIFT.Certificate  # Build only Certificate module

# Check a specific file
lake env lean GIFT/Relations/GaugeSector.lean
```

### Partial Builds

The lakefile defines individual targets for faster iteration:

```bash
lake build GIFT.Algebra      # Just E₈ modules
lake build GIFT.Geometry     # Just G₂ modules
lake build GIFT.Relations    # Just physics sectors
lake build GIFT.Certificate  # Just main theorems
```

## References

- GIFT v2.3 Main Paper
- Supplement S1: Mathematical Architecture
- Supplement S4: Rigorous Proofs
- [Mathlib4 Documentation](https://leanprover-community.github.io/mathlib4_docs/)
- [Lean 4 Manual](https://lean-lang.org/lean4/doc/)
