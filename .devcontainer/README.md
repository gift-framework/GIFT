# GIFT Development Container

Complete development environment for the GIFT Framework with Lean 4, Coq, and Python scientific stack.

## What's Included

### Proof Assistants
- **Lean 4.14.0** with elan toolchain manager
- **Coq 8.18** via opam

### Python Stack
| Category | Packages |
|----------|----------|
| Core | NumPy, SciPy, SymPy, Pandas, Polars |
| ML/DL | PyTorch, scikit-learn, XGBoost, LightGBM |
| Visualization | Matplotlib, Seaborn, Plotly, Altair, Bokeh, HoloViews |
| Jupyter | JupyterLab 4, ipywidgets, Voila |
| Math | mpmath, gmpy2, Numba |
| Quality | pytest, mypy, ruff, black |

### Other Tools
- LaTeX (XeLaTeX, full scientific packages)
- Pandoc for document conversion
- Git LFS for large files
- GitHub CLI

## Usage

### GitHub Codespaces (Recommended)

1. Go to the repository on GitHub
2. Click **Code** → **Codespaces** → **Create codespace on main**
3. Wait for the container to build (~5-10 minutes first time)
4. Start working!

### Local VS Code + Docker

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Install [VS Code](https://code.visualstudio.com/) with [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
3. Clone the repository
4. Open in VS Code → "Reopen in Container"

## Quick Start

After the container starts:

```bash
# Start Jupyter Lab
jupyter lab --no-browser

# Run verification pipeline
./verify.sh status     # Check status
./verify.sh all        # Full verification

# Build Lean proofs
cd Lean && lake build

# Build Coq proofs
cd COQ && make

# Run tests
pytest tests/
```

## VS Code Extensions

The container comes with pre-installed extensions for:
- Python & Jupyter (Pylance, debugger, notebooks)
- Lean 4 (official extension)
- Coq (VsCoq)
- Git (GitLens, GitHub PR)
- Markdown & LaTeX
- Data visualization

## Port Forwarding

| Port | Service |
|------|---------|
| 8888 | Jupyter Lab |
| 8889 | Jupyter Notebook |
| 8890 | Voila Dashboard |

## Customization

### Adding Python packages
```bash
pip install your-package
# Or add to requirements.txt
```

### Updating Lean toolchain
```bash
cd Lean
echo "leanprover/lean4:v4.x.x" > lean-toolchain
lake update
```

### Using a different Coq version
```bash
opam switch create coq-8.19 ocaml-base-compiler.4.14.1
opam pin add coq 8.19.0 -y
```

## Troubleshooting

### Lean not found
```bash
source ~/.bashrc
# or
export PATH="$HOME/.elan/bin:$PATH"
```

### Coq not found
```bash
eval $(opam env --switch=coq-8.18)
```

### Slow Lean build
The first build downloads Mathlib (~2GB). Subsequent builds use cache:
```bash
cd Lean
lake exe cache get  # Download prebuilt .olean files
```

### Out of memory
Request a larger Codespace machine (4-core, 8GB recommended).

## Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| Storage | 16 GB | 32 GB |
