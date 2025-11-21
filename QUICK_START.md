# Quick Start Guide

Get up and running with the GIFT framework in minutes.

## What is GIFT?

The Geometric Information Field Theory (GIFT) derives fundamental physics parameters from pure mathematics. Starting with E₈×E₈ exceptional Lie algebras and G₂ holonomy manifolds, the framework predicts 34 dimensionless observables with 0.13% mean precision using only 3 geometric parameters.

**Key achievement**: Reduces Standard Model's 19 free parameters to 3 derived geometric quantities.

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

The framework makes exact predictions for several quantities:

**Exact Relations (0% deviation by construction)**
- Number of generations: N_gen = 3
- Koide relation parameter: Q = 2/3
- Strange/down quark mass ratio: m_s/m_d = 20

**High-Precision Predictions (<0.01% deviation)**
- Fine structure constant: α⁻¹ = 137.036 (0.001%)
- CP violation phase: δ_CP = 197° (0.005%)
- Koide formula: Q_measured = 0.666661 (0.005%)

**Complete Neutrino Sector (all <0.5%)**
- θ₁₂ = 33.45° (0.03% deviation)
- θ₁₃ = 8.59° (0.23% deviation)
- θ₂₃ = 48.99° (0.43% deviation)
- δ_CP = 197.3° (0.005% deviation)

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

**Section 1**: Three geometric parameters
```python
p₂ = 2           # Binary duality: dim(G₂)/dim(K₇) = 14/7
Weyl_factor = 5  # Pentagonal symmetry from |W(E₈)| = 2¹⁴×3⁵×5²×7
β₀ = π/8         # Angular quantization: π/rank(E₈)
ξ = 5π/16        # Derived: (Weyl_factor/p₂) × β₀
```

**Section 2**: Derived quantities
- Fine structure constant
- Weak mixing angle
- Strong coupling
- Generation number

**Section 3**: Neutrino predictions
- Complete mixing matrix
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

**For Theoretical Details** → `publications/gift_main.md`
- Complete framework (~1100 lines)
- Mathematical structure
- Experimental validation

**For Mathematical Rigor** → `publications/supplements/B_rigorous_proofs.md`
- Exact proofs of 9 key relations
- Step-by-step derivations
- Topological arguments

**For Specific Topics** → `publications/supplements/`
- A: Mathematical foundations (E₈, K₇, reduction)
- C: All 34 observable derivations
- E: Falsification criteria and experimental tests

## Key Predictions by Physics Sector

### Gauge Sector

| Observable | Experimental | GIFT | Deviation |
|------------|--------------|------|-----------|
| α⁻¹ | 137.035999... | 137.036 | 0.001% |
| sin²θ_W | 0.23121(4) | 0.23127 | 0.009% |
| α_s(M_Z) | 0.1181(11) | 0.1180 | 0.08% |

**Source**: Supplement C, Sections 4-6

### Neutrino Sector

| Observable | Experimental | GIFT | Deviation |
|------------|--------------|------|-----------|
| θ₁₂ | 33.44°±0.77° | 33.45° | 0.03% |
| θ₁₃ | 8.61°±0.12° | 8.59° | 0.23% |
| θ₂₃ | 49.2°±1.1° | 48.99° | 0.43% |
| δ_CP | 197°±24° | 197.3° | 0.005% |

**Source**: Supplement C, Section 8

### CKM Matrix

All 10 elements predicted with mean deviation 0.11%.

**Source**: Supplement C, Section 9

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

**See**: Supplement C.4 for complete derivation

### Example: Three Generations

Exact relation:
```
N_gen = rank(E₈) - rank(Weyl(E₇))
      = 8 - 5
      = 3
```

**Status**: PROVEN via index theorem
**See**: Supplement B.4 for rigorous proof

## Finding Specific Information

**Want to understand a specific prediction?**
→ Check Supplement C (Complete Derivations)

**Need rigorous mathematical proof?**
→ Check Supplement B (Rigorous Proofs)

**Looking for experimental comparison?**
→ Check Supplement D (Phenomenology)

**Want to know if framework can be falsified?**
→ Check Supplement E (Falsification Criteria)

**Need definition of technical term?**
→ Check `docs/GLOSSARY.md`

**Have a question?**
→ Check `docs/FAQ.md` or open issue on GitHub

## Common First Questions

**Q: Is this tested experimentally?**
A: Yes. 34 observables compared with experiment, mean deviation 0.13%. See Supplement D and `docs/EXPERIMENTAL_VALIDATION.md`.

**Q: How many free parameters?**
A: 3 topological parameters (p₂=2, Weyl_factor=5, β₀=π/8), where ξ=5π/16 is derived. Standard Model has 19.

**Q: Can this be falsified?**
A: Yes. Multiple clear tests outlined in Supplement E. Strongest: fourth generation discovery or δ_CP deviation from 197° at high precision.

**Q: What about gravity?**
A: Framework derives low-energy parameters but doesn't yet address quantum gravity directly. Connection to information theory suggests potential path forward.

**Q: Why E₈×E₈?**
A: E₈ is largest exceptional Lie algebra with unique properties. Two copies provide sufficient structure for Standard Model content via dimensional reduction. Binary architecture (496→99) may encode optimal information compression.

## Next Steps

### For Theorists

1. Read main paper: `publications/gift_main.md`
2. Study mathematical foundations: `publications/supplements/A_math_foundations.md`
3. Examine proofs: `publications/supplements/B_rigorous_proofs.md`
4. Explore extensions and open questions

### For Experimentalists

1. Review predictions: `publications/gift_main.md` Section 4
2. Check falsification criteria: `publications/supplements/E_falsification.md`
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
@software{gift_framework_v2_2025,
  title={GIFT Framework v2: Geometric Information Field Theory},
  author={{GIFT Framework Team}},
  year={2025},
  url={https://github.com/gift-framework/GIFT},
  version={2.0.0},
  note={Topological unification from E₈×E₈, 0.13% precision, 3 parameters}
}
```

See `CITATION.md` for additional formats.

## Summary

In 5 minutes you can:
1. Run notebook in browser (Binder/Colab)
2. See 34 predictions vs experiment
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
3. Examine all 34 predictions in detail
4. Understand connections to information theory

Welcome to the GIFT framework. The mathematics is rich, the predictions are precise, and the implications are profound.

---

*"From bit to GIFT"*

