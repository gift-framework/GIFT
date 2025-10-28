# GIFT Framework - Visualization Figures

This directory contains exported HTML visualizations for the dashboard.

## Contents

When `generate_all_figures.py` is run, the following files are copied here:

- `e8_roots.html` - E₈ root system 3D visualization
- `precision_dashboard.html` - Precision analysis dashboard
- `dimensional_reduction.html` - Dimensional reduction flow

## Generation

To populate this directory with figures:

```bash
# From repository root
python generate_all_figures.py
```

This will:
1. Execute all visualization notebooks
2. Generate figures in `visualizations/outputs/`
3. Copy HTML files to this directory

## Usage

These HTML files are:
- Referenced by `../index.html` (dashboard)
- Embedded in iframes for display
- Self-contained with embedded Plotly.js
- Fully interactive

## Note

This directory may be empty in the repository. Generated figures are created on-demand and can be large (1-5MB each). They are typically generated locally or via CI/CD for deployment.

For GitHub Pages deployment, ensure figures are present before pushing or set up automated generation in GitHub Actions.

---

Part of GIFT Framework v2.0 Visualization Dashboard

