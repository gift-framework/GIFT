# GIFT Framework Interactive Visualizations

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gift-framework/GIFT/main?filepath=assets/visualizations/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gift-framework/GIFT/blob/main/assets/visualizations/)

Interactive Jupyter notebooks visualizing the geometric and mathematical structures underlying the GIFT framework.

## Overview

These visualizations provide interactive exploration of the dimensional reduction process, precision analysis, and mathematical structures that enable topological unification of Standard Model parameters.

## Notebooks

### 1. E₈ Root System 3D (`e8_root_system_3d.ipynb`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gift-framework/GIFT/blob/main/assets/visualizations/e8_root_system_3d.ipynb)

**Description**: Interactive 3D visualization of the complete E₈ root system (240 roots) projected from 8D to 3D using Principal Component Analysis.

**Features**:
- Complete 240-root structure with two root lengths (√2 and 2)
- Color-coded by root type (short vs long)
- Interactive rotation, zoom, and pan
- Statistical analysis of root geometry
- Connection to GIFT framework (Weyl group, N_gen = 3)

**Key insights**:
- E₈ provides the fundamental 248-dimensional structure
- Weyl group W(E₈) = 2¹⁴ × 3⁵ × 5² × 7 determines Weyl factor = 5
- N_gen = rank(E₈) - Weyl = 8 - 5 = 3 (three generations)

**Runtime**: ~5-10 seconds

### 2. Precision Dashboard (`precision_dashboard.ipynb`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gift-framework/GIFT/blob/main/assets/visualizations/precision_dashboard.ipynb)

**Description**: Comprehensive comparison of all validated dimensionless observables against experimental measurements across all physics sectors.

**Features**:
- Sector-wise precision gauges (Gauge, Neutrino, Lepton, Quark, Higgs, Cosmology)
- Interactive heatmap of all 16 observables sorted by precision
- Statistical distributions (histogram and box plots)
- Detailed results table with deviations
- Mean deviation: ~0.15% across all observables

**Sectors covered**:
- Gauge couplings (3): α⁻¹(M_Z), sin²θ_W, α_s(M_Z)
- Neutrino mixing (4): θ₁₂, θ₁₃, θ₂₃, δ_CP
- Lepton masses (4): Q_Koide, m_μ/m_e, m_τ/m_e, m_τ/m_μ
- Quark ratios (1): m_s/m_d
- Higgs sector (1): λ_H
- Cosmology (3): Ω_DE, n_s, H₀

**Key results**:
- 4 exact predictions (0.000% deviation): δ_CP, m_s/m_d, m_τ/m_e, N_gen
- All observables < 0.5% deviation
- Most observables < 0.2% deviation

**Runtime**: ~2-3 seconds

### 3. Dimensional Reduction Flow (`dimensional_reduction_flow.ipynb`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gift-framework/GIFT/blob/main/assets/visualizations/dimensional_reduction_flow.ipynb)

**Description**: Animated visualization of the dimensional reduction process from E₈×E₈ (496D) through K₇ (99D) to the Standard Model (4D).

**Features**:
- Animated bar chart showing progressive reduction
- Sankey diagram illustrating information flow
- Detailed breakdown of cohomology structure
- Information compression analysis (496 → 99 = 5:1 ratio)
- Connection to quantum error-correcting codes [[496, 99, 31]]

**Reduction stages**:
1. **E₈×E₈ (496D)**: Two copies of exceptional Lie algebra
2. **K₇ (99D)**: G₂ holonomy manifold with H*(K₇) = 1 + 21 + 77
3. **SM (4D)**: Standard Model emerges from cohomological structure

**Key insights**:
- H²(K₇) = 21 provides gauge boson structure
- H³(K₇) = 77 provides chiral matter content
- Information preserved through cohomological mapping
- Physical parameters as topological invariants

**Runtime**: ~3-5 seconds

## Installation

### Local Setup

1. Ensure Python 3.11+ is installed
2. Install dependencies:
```bash
pip install -r ../requirements.txt
```

3. Launch Jupyter:
```bash
jupyter notebook
```

### Cloud Execution (Binder)

Click the badge to run all notebooks in your browser without installation:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gift-framework/GIFT/main?filepath=visualizations/)

## Dependencies

Core visualization libraries:
- `plotly>=5.14.0` - Interactive 3D plots and animations
- `numpy>=1.21.0` - Numerical computations
- `pandas>=1.3.0` - Data handling
- `scikit-learn>=1.0.0` - PCA projection
- `matplotlib>=3.5.0` - Static plots
- `seaborn>=0.11.0` - Statistical visualizations
- `kaleido>=0.2.1` - Static image export (optional)
- `networkx>=3.0` - Graph structures (optional)

## Usage Tips

### Interactive Features

All visualizations support:
- **Hover**: Detailed information on data points
- **Zoom**: Scroll or pinch to zoom in/out
- **Pan**: Click and drag to move view
- **Rotate** (3D plots): Click and drag to rotate camera
- **Export**: Download as PNG/HTML using Plotly controls

### Performance

For best performance:
- Run notebooks in order (some may reuse computed data)
- Use JupyterLab for enhanced interactivity
- Export HTML for offline viewing with full interactivity

### Customization

All notebooks are fully editable. Key parameters to modify:
- `e8_root_system_3d.ipynb`: PCA components, color schemes, plot angles
- `precision_dashboard.ipynb`: Deviation thresholds, sector groupings
- `dimensional_reduction_flow.ipynb`: Animation speed, color transitions

## Export Options

Each notebook can export visualizations:
- **HTML**: Fully interactive, shareable (`.write_html()`)
- **PNG**: Static images for papers (`.write_image()`)
- **CSV**: Data tables for further analysis

Example:
```python
fig.write_html('my_visualization.html')
fig.write_image('my_visualization.png', width=1200, height=900)
```

## Scientific References

These visualizations implement mathematical structures from:
- **Supplement A**: E₈ Lie algebra, K₇ manifold, dimensional reduction
- **Supplement B**: Rigorous proofs of exact relations
- **Supplement C**: Complete observable derivations
- **Main Paper**: Framework overview and key results

## Technical Notes

### Color Schemes

All visualizations use colorblind-friendly palettes:
- E₈ roots: Red (short) vs Blue (long)
- Precision: Green (excellent) → Yellow (good) → Red (tension)
- Reduction stages: Purple (E₈×E₈) → Blue (K₇) → Green (SM)

### Data Validation

All numerical values are validated against:
- `publications/gift_v2_notebook.ipynb` (computational validation)
- Experimental values from PDG, NuFIT, Planck collaborations
- Mathematical formulas from Supplements B and C

### Reproducibility

To ensure reproducibility:
- Fixed random seeds where applicable
- Explicit formulas in code comments
- Data provenance documented in each cell
- Versions pinned in `requirements.txt`

## Contributing

Improvements welcome! Potential enhancements:
- Additional physical sectors (CKM matrix details, quark masses)
- 3D animations of K₇ twisted connected sum construction
- Interactive parameter exploration widgets
- Comparative visualizations with other unification approaches

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

These visualizations are part of the GIFT Framework and are licensed under the MIT License - see [LICENSE](../../LICENSE).

## Citation

If you use these visualizations in your research, please cite:

```bibtex
@software{gift_visualizations_2025,
  title={GIFT Framework Interactive Visualizations},
  author={{GIFT Framework Team}},
  year={2025},
  url={https://github.com/gift-framework/GIFT/tree/main/visualizations},
  note={Interactive exploration of E₈×E₈ dimensional reduction}
}
```

## Questions and Support

- **Documentation**: See main [README.md](../README.md)
- **FAQ**: [docs/FAQ.md](../../docs/FAQ.md)
- **Issues**: https://github.com/gift-framework/GIFT/issues
- **Discussions**: Use "visualization" tag for visualization-specific questions

---

**Gift from bit**: Physical law as information geometry.

