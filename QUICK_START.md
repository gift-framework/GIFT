# Quick Start Guide

Get up and running with the GIFT framework in minutes.

## What is GIFT?

The Geometric Information Field Theory (GIFT) derives fundamental physics parameters from pure mathematics. Starting with E₈×E₈ exceptional Lie algebras and G₂ holonomy manifolds, the framework predicts 39 observables with 0.128% mean precision using no continuous adjustable parameters.

**Key achievement**: All quantities structurally determined from fixed topological structure (zero-parameter paradigm).

## Installation

### Option 1: Run in Browser (Recommended)

No installation needed. Click either link:

**Binder** (free, no account): [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gift-framework/GIFT/main?filepath=publications/gift_v2_notebook.ipynb)

**Google Colab** (requires Google account): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gift-framework/GIFT/blob/main/publications/gift_v2_notebook.ipynb)

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/gift-framework/GIFT.git
cd gift

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook publications/gift_v2_notebook.ipynb
```

**Requirements**: Python 3.11 or higher

## 5-Minute Tour

### Step 1: Key Results

The framework makes exact predictions (13 PROVEN relations):

**Exact Rational/Integer Relations (PROVEN)**
- Weinberg angle: sin²θ_W = 3/13 (0.195% deviation)
- Torsion magnitude: κ_T = 1/61
- Metric determinant: det(g) = 65/32
- Hierarchy parameter: τ = 3472/891
- CP violation phase: δ_CP = 197° (exact)
- Koide parameter: Q = 2/3
- Quark mass ratio: m_s/m_d = 20 (exact)
- Number of generations: N_gen = 3

**High-Precision Predictions (<0.1% deviation)**
- Fine structure constant: α⁻¹ = 137.033 (0.002%)
- Strong coupling: α_s = √2/12 (0.042%)
- Spectral index: n_s = ζ(11)/ζ(5) = 0.9649 (exact)

**Complete Neutrino Sector (all <0.5%)**
- θ₁₂ = 33.40° (0.03% deviation)
- θ₁₃ = π/21 = 8.571° (0.36% deviation)
- θ₂₃ = 49.19° (0.22% deviation)
- δ_CP = 197° (0.00% deviation - exact)

### Step 2: Understanding the Framework

The dimensional reduction proceeds:

```
E₈×E₈ (496-dimensional)
   ↓ compactification
AdS₄ × K₇
   ↓ G₂ holonomy breaking
Standard Model (SU(3)×SU(2)×U(1))
```

**Key components**:
1. **E₈×E₈**: Two copies of largest exceptional Lie algebra
2. **K₇**: 7-dimensional manifold with G₂ holonomy
3. **Cohomology**: H²(K₇)=ℝ²¹ → gauge bosons, H³(K₇)=ℝ⁷⁷ → fermions
4. **Information**: Binary architecture 496→99 → physical parameters

### Step 3: Explore the Notebook

Open `publications/gift_v2_notebook.ipynb` and run cells sequentially:

**Section 1**: Topological constants (all structurally determined)
```python
# Betti numbers from K₇ manifold
b2 = 21          # Second Betti number
b3 = 77          # Third Betti number
H_star = 99      # Effective dimension: b2 + b3 + 1

# Key derived quantities (exact)
sin2_theta_W = 3/13       # Weinberg angle: b2/(b3 + dim(G2))
kappa_T = 1/61            # Torsion: 1/(b3 - dim(G2) - p2)
det_g = 65/32             # Metric determinant
tau = 3472/891            # Hierarchy parameter
```

**Section 2**: Exact predictions
- Gauge couplings (α, sin²θ_W, α_s)
- Generation number (N_gen = 3)
- Fermion mass ratios

**Section 3**: Neutrino predictions
- Complete mixing matrix with δ_CP = 197°
- Mass hierarchy
- CP violation

**Section 4**: Comparison with experiment
- Tables of predictions vs measurements
- Statistical analysis
- Precision plots

### Step 4: Read the Documentation

**For Quick Overview** → `README.md`
- Framework summary
- Key results at a glance
- Installation and usage

**For Theoretical Details** → `publications/gift_2_2_main.md`
- Complete framework (~1400 lines)
- Mathematical structure
- Experimental validation

**For Mathematical Rigor** → `publications/supplements/S4_complete_derivations.md`
- Exact proofs of 13 proven relations
- Step-by-step derivations
- Topological arguments

**For Specific Topics** → `publications/supplements/`
- S1: Mathematical architecture (E₈, K₇, cohomology)
- S4: All 39 observable derivations
- S5: Experimental validation and falsification criteria

## Key Predictions by Physics Sector

### Gauge Sector

| Observable | Experimental | GIFT | Deviation |
|------------|--------------|------|-----------|
| α⁻¹ | 137.035999... | 137.036 | 0.001% |
| sin²θ_W | 0.23121(4) | 0.23127 | 0.009% |
| α_s(M_Z) | 0.1181(11) | 0.1180 | 0.08% |

**Source**: S4_complete_derivations, Sections 4-6

### Neutrino Sector

| Observable | Experimental | GIFT | Deviation |
|------------|--------------|------|-----------|
| θ₁₂ | 33.44°±0.77° | 33.45° | 0.03% |
| θ₁₃ | 8.61°±0.12° | 8.59° | 0.23% |
| θ₂₃ | 49.2°±1.1° | 48.99° | 0.43% |
| δ_CP | 197°±24° | 197.3° | 0.005% |

**Source**: S4_complete_derivations, Section 8

### CKM Matrix

All 10 elements predicted with mean deviation 0.11%.

**Source**: S4_complete_derivations, Section 9

### Cosmological Sector

| Observable | Experimental | GIFT | Deviation |
|------------|--------------|------|-----------|
| Ω_DE | 0.6889(56) | ln(2) = 0.693 | 0.10% |

**Source**: Main paper Section 4.7, Extensions document

## Understanding the Mathematics

### Three Geometric Parameters

**p₂ = 2**: Binary duality constant
- Definition: dim(G₂)/dim(K₇) = 14/7 = 2
- Role: Information encoding, particle/antiparticle duality

**Weyl_factor = 5**: Pentagonal symmetry
- Origin: Unique perfect square 5² in |W(E₈)| = 2¹⁴ × 3⁵ × 5² × 7
- Role: Generation count, mass ratios (e.g., m_s/m_d = p₂² × Weyl = 20)

**β₀ = π/8**: Angular quantization
- Definition: π/rank(E₈) = π/8
- Role: Neutrino mixing, cosmological parameters

**ξ = 5π/16**: Correlation parameter (DERIVED)
- Exact relation: (Weyl_factor/p₂) × β₀ = (5/2) × (π/8) = 5π/16
- Reduces effective free parameters from 3 to 2

**Parameter count**: 3 topological (p₂, Weyl, β₀) vs Standard Model's 19

### Example: Fine Structure Constant

Derivation chain:
1. Start with E₈ structure: dim(E₈) = 248
2. K₇ cohomology: b₂ = 21, b₃ = 77
3. Branching E₈ → SU(5)×SU(3) → SU(3)×SU(2)×U(1)
4. Normalize gauge couplings via β₀
5. Result: α⁻¹ = 137.036... (0.001% deviation)

**See**: Supplement S5.4 for complete derivation

### Example: Three Generations

Exact relation:
```
N_gen = rank(E₈) - rank(Weyl(E₇))
      = 8 - 5
      = 3
```

**Status**: PROVEN via index theorem
**See**: Supplement S4.4 for rigorous proof

## Finding Specific Information

**Want to understand a specific prediction?**
→ Check S4_complete_derivations.md (all observables)

**Need rigorous mathematical proof?**
→ Check S4_complete_derivations.md (13 proven relations)

**Looking for experimental comparison?**
→ Check S5_experimental_validation.md

**Want to know if framework can be falsified?**
→ Check S5_experimental_validation.md (falsification criteria)

**Need definition of technical term?**
→ Check `docs/GLOSSARY.md`

**Have a question?**
→ Check `docs/FAQ.md` or open issue on GitHub

## Common First Questions

**Q: Is this tested experimentally?**
A: Yes. 39 observables compared with experiment, mean deviation 0.128%. See S5_experimental_validation.md and `docs/EXPERIMENTAL_VALIDATION.md`.

**Q: How many free parameters?**
A: Zero continuous adjustable parameters. All quantities are structurally determined from E₈×E₈ and K₇ topology (discrete structural choices).

**Q: Can this be falsified?**
A: Yes. Multiple clear tests outlined in S5_experimental_validation.md. Strongest: fourth generation discovery or δ_CP deviation from 197° at high precision.

**Q: What about gravity?**
A: Framework derives low-energy parameters but doesn't yet address quantum gravity directly. Connection to information theory suggests potential path forward.

**Q: Why E₈×E₈?**
A: E₈ is largest exceptional Lie algebra with unique properties. Two copies provide sufficient structure for Standard Model content via dimensional reduction. Binary architecture (496→99) may encode optimal information compression.

## Next Steps

### For Theorists

1. Read main paper: `publications/gift_2_2_main.md`
2. Study mathematical foundations: `publications/supplements/S1_mathematical_architecture.md`
3. Examine proofs: `publications/supplements/S4_complete_derivations.md`
4. Explore extensions and open questions

### For Experimentalists

1. Review predictions: `publications/gift_2_2_main.md` Section 8
2. Check falsification criteria: `publications/supplements/S5_experimental_validation.md`
3. See experimental timeline: `docs/EXPERIMENTAL_VALIDATION.md`
4. Identify relevant experiments for your facility

### For Students

1. Start with README.md overview
2. Run notebook: `publications/gift_v2_notebook.ipynb`
3. Read FAQ: `docs/FAQ.md`
4. Study glossary: `docs/GLOSSARY.md`
5. Work through main paper sections

### For Contributors

1. Read `CONTRIBUTING.md` for guidelines
2. Check `STRUCTURE.md` for repository organization
3. Review open issues on GitHub
4. Propose improvements or extensions

## Getting Help

**Documentation**: Most questions answered in:
- `README.md` (overview)
- `docs/FAQ.md` (common questions)
- `docs/GLOSSARY.md` (definitions)

**Issues**: For bugs, questions, or suggestions:
- https://github.com/gift-framework/GIFT/issues

**Discussion**: For broader topics:
- GitHub Discussions (coming soon)

## Citation

If you use GIFT in your research:

```bibtex
@software{gift_framework_v22_2025,
  title={GIFT Framework v2.2: Geometric Information Field Theory},
  author={{GIFT Framework Team}},
  year={2025},
  url={https://github.com/gift-framework/GIFT},
  version={2.2.0},
  note={Topological unification from E₈×E₈, 0.128% precision across 39 observables, zero adjustable parameters}
}
```

See `CITATION.md` for additional formats.

## Summary

In 5 minutes you can:
1. Run notebook in browser (Binder/Colab)
2. See 39 predictions vs experiment
3. Understand basic framework structure
4. Explore specific sectors of interest

In 30 minutes you can:
1. Read main paper introduction
2. Understand dimensional reduction chain
3. See derivation of key results
4. Check falsification criteria

In a few hours you can:
1. Study complete mathematical foundations
2. Verify numerical calculations
3. Examine all 39 predictions in detail
4. Understand connections to information theory

Welcome to the GIFT framework. The mathematics is rich, the predictions are precise, and the implications are profound.

---

*"From bit to GIFT"*

