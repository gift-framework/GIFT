# GIFT Framework v2.3 - Visualization Dashboard

Terminal-style interactive dashboard showcasing GIFT framework visualizations.

## Overview

This directory contains a Pip-Boy themed visualization dashboard for the GIFT Framework v2.3. The dashboard provides an interactive, terminal-aesthetic interface for exploring computational results.

## Dashboard

**Main Interface**: `index.html`

Features:
- Three interactive visualization panels
- Tab navigation (keyboard shortcuts: 1, 2, 3)
- Terminal aesthetic with amber monochrome theme
- Scanline effects for retro CRT appearance

## Visualizations

### 1. E8 Root System (Tab 1)
- Complete 240-root structure of exceptional Lie algebra E8
- 3D interactive projection from 8D using PCA
- Foundation for 496-dimensional information architecture
- Source: `visualizations/e8_root_system_3d.ipynb`

### 2. Precision Analysis (Tab 2)
- Statistical comparison across all physics sectors
- 39 validated observables with 13 proven exact relations
- Sector-wise performance: Gauge, Neutrino, Lepton, Quark, Higgs, Cosmology
- Mean deviation: 0.128%
- Source: `visualizations/precision_dashboard.ipynb`

### 3. Dimensional Reduction Flow (Tab 3)
- Animated visualization of dimensional reduction
- E8xE8 (496D) -> K7 (99D) -> Standard Model (4D)
- Cohomology structure: H2(K7)=21, H3(K7)=77
- Source: `visualizations/dimensional_reduction_flow.ipynb`

## Directory Structure

```
docs/
├── index.html              # Main dashboard page
├── assets/
│   ├── css/
│   │   └── pipboy.css      # Pip-Boy terminal theme
│   └── js/
│       └── dashboard.js    # Tab navigation and interactivity
├── figures/                # Exported visualizations (HTML)
│   ├── e8_roots.html
│   ├── precision_dashboard.html
│   └── dimensional_reduction.html
└── README.md               # This file
```

## Local Usage

### Option 1: Direct File Open
Simply open `index.html` in a web browser.

### Option 2: Local Server (Recommended)
For best results, serve with a local HTTP server:

```bash
# Python 3
cd docs
python -m http.server 8000

# Then visit: http://localhost:8000
```

## GitHub Pages Deployment

To deploy on GitHub Pages:

1. Enable GitHub Pages in repository settings
2. Set source to `main` branch, `/docs` folder
3. Dashboard will be available at: `https://[username].github.io/[repo]/`

## Updating Figures

Figures are generated from source notebooks. To update:

```bash
# From repository root
python generate_all_figures.py
```

This will:
1. Execute all visualization notebooks
2. Generate figures in multiple formats
3. Copy HTML exports to `docs/figures/`

## Design Theme

**Inspiration**: Fallout Pip-Boy terminal aesthetic

**Visual Elements**:
- Pure black background (#000000)
- Amber monochrome color scheme (#FFB000)
- Monospace fonts (Courier New, Consolas)
- CRT scanline effects
- Terminal-style borders and layouts
- Minimal, functional design

**Navigation**:
- Click tabs or use keyboard shortcuts (1, 2, 3)
- Fully responsive for mobile and desktop
- Accessibility features (focus states, semantic HTML)

## Browser Compatibility

Tested and working on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

Requires JavaScript enabled for tab navigation.

## Customization

### Colors
Edit `assets/css/pipboy.css` CSS variables:
```css
:root {
    --color-primary: #FFB000;      /* Amber primary */
    --color-secondary: #CC8800;    /* Darker amber */
    --color-bg: #000000;           /* Pure black */
}
```

### Content
Edit `index.html` to modify:
- Header text and statistics
- Tab labels and descriptions
- Footer links

### Visualizations
Replace HTML files in `figures/` directory or modify source notebooks.

## Technical Details

- Pure HTML/CSS/JavaScript (no frameworks)
- Semantic HTML5 structure
- CSS Grid and Flexbox layouts
- Vanilla JavaScript (no dependencies)
- Plotly.js embedded in figure HTML files

## Performance

- Initial load: <1s (with cached figures)
- Tab switching: Instant
- Figure rendering: Depends on Plotly complexity
- Tested with figures up to 10MB

## Accessibility

- Keyboard navigation support
- Focus indicators for interactive elements
- Semantic HTML structure
- ARIA labels where appropriate
- High contrast color scheme

## Contributing

To add new visualizations:

1. Create notebook in `visualizations/`
2. Add entry to `generate_all_figures.py` NOTEBOOKS list
3. Add new tab section to `index.html`
4. Update CSS if needed for styling

## License

Part of GIFT Framework v2.3 - MIT License

## Links

- Main Repository: https://github.com/gift-framework/GIFT
- Main Paper: `../publications/markdown/gift_2_3_main.md`
- Supplements: `../publications/markdown/`
- Visualizations: `../assets/visualizations/`
- Formal Proofs (Lean 4 + Coq): https://github.com/gift-framework/core

---

GIFT FRAMEWORK v2.3 | VISUALIZATION TERMINAL | TOPOLOGICAL UNIFICATION
