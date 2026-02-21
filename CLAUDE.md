# CLAUDE.md - Development Guide for GIFT Documentation

> **Persistent context**: Read `../.claude-persistent-context.md` at session start for cross-session memory (key insights, ongoing experiments, decisions).

This file provides development conventions for the GIFT theoretical documentation repository.

## Repository Purpose

This repository contains the **theoretical documentation** for GIFT (Geometric Information Field Theory). For formal proofs and computational code, see [gift-framework/core](https://github.com/gift-framework/core).

## Project Structure

```
GIFT/
├── publications/
│   ├── papers/
│   │   ├── markdown/       # Core documents (v3.3)
│   │   │   ├── GIFT_v3.3_main.md        # Main paper (accessible, quasi-autonomous)
│   │   │   ├── GIFT_v3.3_S1_foundations.md  # E₈, G₂, K₇ foundations
│   │   │   ├── GIFT_v3.3_S2_derivations.md  # All 33 dimensionless derivations
│   │   │   └── Numerical_G2_Metric.md       # PINN-based G₂ metric construction
│   │   ├── tex/            # LaTeX sources
│   │   ├── pdf/            # Generated PDFs
│   │   └── FoP/            # Foundations of Physics submission
│   ├── references/         # Extended topics (number theory, speculative physics)
│   ├── outreach/           # Vulgarization & blog posts
│   └── validation/         # Monte Carlo validation code
│
├── research/               # Exploratory research (WIP)
│
├── docs/
│   ├── FAQ.md             # Common questions
│   ├── GLOSSARY.md        # Technical terms
│   ├── GIFT_FOR_EVERYONE.md  # Complete guide with everyday analogies
│   ├── GIFTPY_FOR_GEOMETERS.md
│   ├── INFO_GEO_FOR_PHYSICISTS.md
│   ├── LEAN_FOR_PHYSICS.md
│   ├── figures/           # Lean blueprints, diagrams
│   ├── media/             # Logos, images
│   └── legacy/            # Archived v2.3/v3.0 supplements
│
├── notebooks/              # Curated demo notebooks
│
├── README.md              # Repository overview
├── STRUCTURE.md           # Directory layout
├── CHANGELOG.md           # Version history
└── CITATION.md            # How to cite
```

---

## Document Hierarchy

### Main Paper
- **Accessible** to physicists and mathematicians
- **Quasi-autonomous**: readable without supplements
- Contains: framework overview, representative derivations, predictions catalog, falsification tests
- References supplements for technical details

### Supplements
- **S1 Foundations**: Complete mathematical construction (E₈ lattice, G₂ holonomy, K₇ topology)
- **S2 Derivations**: All 33 dimensionless predictions with full derivations
- **Numerical G₂ Metric**: PINN-based G₂ metric construction (companion numerical paper)

### Extended References
- Exploratory topics (Monster, sequences, Yukawa)
- Not part of core claims

---

## Terminology Standards

### Use Standard Academic Terms

| Internal Code | Standard Academic Term |
|---------------|------------------------|
| ε_ijk | G₂ structure constants / associative 3-form coefficients |
| φ₀ | Standard associative 3-form |
| ψ_ijkl | Coassociative 4-form |
| "Lagrange identity" | ‖u × v‖² = ‖u‖²‖v‖² − ⟨u,v⟩² for G₂-invariant cross product |
| "Fano structure" | Octonion multiplication from Fano plane |

### Avoid Internal Jargon

```markdown
# Incorrect (internal reference)
"The B4 axiom is now proven via epsilon contraction"

# Correct (academic standard)
"The Lagrange identity ‖u × v‖² = ‖u‖²‖v‖² − ⟨u,v⟩² for the
G₂-invariant cross product in ℝ⁷ is formally verified in Lean 4"
```

### State Current Results Only

```markdown
# Incorrect (evolutionary language)
"In v3.1, we improved the axiom resolution from 6/10 to 9/10"

# Correct (current state)
"The E₈ root system properties are fully verified (12 theorems).
The G₂ cross product satisfies the Lagrange identity (verified)
and antisymmetry properties (verified)."
```

---

## Mathematical Notation

### Topological Constants

| Symbol | Value | Definition |
|--------|-------|------------|
| dim(E₈) | 248 | E₈ Lie algebra dimension |
| rank(E₈) | 8 | E₈ Cartan subalgebra dimension |
| dim(G₂) | 14 | G₂ holonomy group dimension |
| b₂ | 21 | Second Betti number of K₇ |
| b₃ | 77 | Third Betti number of K₇ |
| H* | 99 | b₂ + b₃ + 1 |
| p₂ | 2 | Pontryagin class contribution |
| dim(J₃(O)) | 27 | Exceptional Jordan algebra dimension |

### Physical Predictions

| Symbol | Value | Topological Origin |
|--------|-------|-------------------|
| sin²θ_W | 3/13 | b₂/(b₃ + dim(G₂)) |
| κ_T | 1/61 | 1/(b₃ − dim(G₂) − p₂) |
| det(g) | 65/32 | G₂ metric determinant |
| τ | 3472/891 | (dim(E₈×E₈) × b₂)/(dim(J₃(O)) × H*) |
| N_gen | 3 | rank(E₈) − Weyl = b₂/dim(K₇) |

---

## Blueprint Workflow

The Lean formalization uses **blueprint documentation** linking mathematical statements to code.

### Blueprint Structure (in gift-core)

```
blueprint/
├── src/
│   └── content.tex     # LaTeX with \lean{} references
├── home_page/
└── lakefile.toml
```

### Key Commands

```latex
\begin{theorem}[Theorem Name]\label{thm:name}
    \lean{GIFT.Module.theorem_name}
    \leanok                           % Marks as proven
    \uses{def:dependency}             % Declares dependencies
    Mathematical statement here.
\end{theorem}
```

### Status Indicators

- `\leanok` — Fully proven in Lean
- No marker — Statement only (axiom or pending)
- `\uses{}` — Dependency tracking

### Viewing Blueprint

Generated at: https://gift-framework.github.io/core/

---

## Writing Guidelines

### Tone

- **Humble and scientific**: "The framework proposes..." not "We prove..."
- **Precise claims**: "Formally verified in Lean 4" vs "proven"
- **Acknowledge limitations**: Distinguish topological derivation from physical truth

### Recommended Formulations

```markdown
# Avoid
"This proves the framework is correct"
"Revolutionary discovery"
"Zero torsion solves everything"

# Prefer
"This elevates the framework from numerical agreement to algebraic derivation"
"The G₂ metric admits an exact analytical form"
"The classical solution has T = 0; physical interactions require mechanisms
for effective torsion that remain under investigation"
```

### Status Classifications

| Status | Meaning |
|--------|---------|
| **PROVEN** | Formally verified in Lean 4 |
| **TOPOLOGICAL** | Derived from topology, not fitted |
| **THEORETICAL** | Proposed mechanism, not yet verified |
| **SPECULATIVE** | Exploratory extension |

---

## Cross-Repository Coordination

### This Repository (GIFT)
- Theoretical documentation
- Publications (markdown, LaTeX, PDF)
- Statistical validation
- Guides for different audiences

### gift-framework/core
- Formal proofs (Lean 4)
- Python package `giftpy`
- Blueprint documentation
- CI/CD workflows

### Synchronization

When updating:
1. Verify claim matches Lean theorem in core
2. Use theorem name from blueprint, not internal identifiers
3. Update version references consistently

---

## Editing Checklist

Before committing documentation changes:

- [ ] Academic terminology (no internal codes)
- [ ] Current state only (no evolutionary history)
- [ ] Lean theorem references match blueprint
- [ ] Cross-references between Main and Supplements work
- [ ] Version numbers consistent
- [ ] Claims match formal verification status

---

## Links

| Resource | URL |
|----------|-----|
| Core Repository | https://github.com/gift-framework/core |
| Blueprint | https://gift-framework.github.io/core/ |
| PyPI Package | https://pypi.org/project/gift-core/ |

---

## GPU Computing Tips (CuPy/CUDA)

When writing notebooks for GPU execution (Colab A100), be aware of CuPy limitations:

### CuPy Sparse Matrix Gotchas

| SciPy | CuPy | Workaround |
|-------|------|------------|
| `M.tolil()` | ❌ Not implemented | Build COO/CSR directly |
| `eigsh(..., which='SM')` | ❌ Only 'LM', 'LA', 'SA' | Use `which='SA'` for smallest eigenvalues |
| `M[i,j] = x` on CSR | ❌ Inefficient/broken | Build from COO format |

### Correct Pattern for Periodic BC Sparse Matrix

```python
# ❌ Wrong (CuPy incompatible)
D2 = cp_diags([...], format='csr')
D2 = D2.tolil()           # NotImplementedError!
D2[0, N-1] = 1/h²
D2 = D2.tocsr()

# ✓ Correct (build COO directly)
row, col, data = [], [], []
for i in range(N):
    row.append(i); col.append(i); data.append(-2/h²)
    row.append(i); col.append((i+1) % N); data.append(1/h²)  # periodic
    row.append(i); col.append((i-1) % N); data.append(1/h²)  # periodic
D2 = cp_csr((cp.array(data), (cp.array(row), cp.array(col))), shape=(N, N))
```

### Eigenvalue Computation

```python
# ❌ Wrong for CuPy
eigs = cp_eigsh(L, k=10, which='SM')  # ValueError!

# ✓ Correct for CuPy (positive semi-definite matrices)
eigs = cp_eigsh(L, k=10, which='SA')  # Smallest Algebraic
```

### Memory Management

```python
# Clear GPU memory between large computations
cp.get_default_memory_pool().free_all_blocks()
```

### JSON Serialization with NumPy Types

NumPy types (`numpy.bool_`, `numpy.int64`, `numpy.float64`) are not JSON-serializable:

```python
# ❌ Wrong (TypeError: Object of type bool_ is not JSON serializable)
results = {'passed': deviation < 5}  # numpy.bool_
json.dump(results, f)

# ✓ Correct (explicit conversion to Python types)
results = {'passed': bool(deviation < 5)}
json.dump(results, f)
```

Common conversions:
| NumPy Type | Python Conversion |
|------------|-------------------|
| `numpy.bool_` | `bool(x)` |
| `numpy.int64` | `int(x)` |
| `numpy.float64` | `float(x)` |
| `numpy.ndarray` | `x.tolist()` |

---

*GIFT Documentation Repository*
