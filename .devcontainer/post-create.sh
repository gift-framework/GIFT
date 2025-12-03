#!/usr/bin/env bash
# =============================================================================
# GIFT Framework - Post-Create Setup Script
# =============================================================================
# This script runs after the devcontainer is created.
# It initializes all components and fetches dependencies.
# =============================================================================

set -e

echo "=============================================="
echo "  GIFT Framework Environment Setup"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

success() { echo -e "${GREEN}✓${NC} $1"; }
info() { echo -e "${BLUE}→${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }

# =============================================================================
# PYTHON SETUP
# =============================================================================
echo ""
echo -e "${BLUE}[1/5] Python Environment${NC}"
echo "----------------------------------------------"

if [ -f "requirements.txt" ]; then
    info "Installing project requirements..."
    pip install -q -r requirements.txt
    success "Python requirements installed"
else
    warn "No requirements.txt found, skipping"
fi

# Install any additional dev requirements
if [ -f "requirements-dev.txt" ]; then
    info "Installing dev requirements..."
    pip install -q -r requirements-dev.txt
    success "Dev requirements installed"
fi

# =============================================================================
# LEAN 4 SETUP
# =============================================================================
echo ""
echo -e "${BLUE}[2/5] Lean 4 Environment${NC}"
echo "----------------------------------------------"

if [ -d "Lean" ]; then
    info "Setting up Lean 4 environment..."
    cd Lean

    # Ensure correct toolchain
    if [ -f "lean-toolchain" ]; then
        info "Using toolchain from lean-toolchain file"
    else
        info "Setting default toolchain to v4.14.0"
        echo "leanprover/lean4:v4.14.0" > lean-toolchain
    fi

    # Update lake dependencies
    info "Running lake update (this may take a moment)..."
    lake update 2>/dev/null || warn "Lake update had some warnings"

    # Try to get Mathlib cache (speeds up builds significantly)
    info "Fetching Mathlib cache..."
    lake exe cache get 2>/dev/null || warn "Mathlib cache not available, will build from source"

    cd ..
    success "Lean 4 environment ready"
else
    warn "No Lean directory found, skipping Lean setup"
fi

# =============================================================================
# COQ SETUP
# =============================================================================
echo ""
echo -e "${BLUE}[3/5] Coq Environment${NC}"
echo "----------------------------------------------"

if [ -d "COQ" ]; then
    info "Setting up Coq environment..."

    # Ensure opam environment is loaded
    eval $(opam env --switch=coq-8.18 2>/dev/null) || true

    # Verify Coq is available
    if command -v coqc &> /dev/null; then
        COQ_VERSION=$(coqc --version 2>/dev/null | head -1 || echo "unknown")
        success "Coq available: $COQ_VERSION"

        # Generate dependencies
        cd COQ
        info "Generating Coq dependencies..."
        coqdep -Q . GIFT $(cat _CoqProject | grep '\.v$') > .depend 2>/dev/null || true
        cd ..
    else
        warn "Coq not found in PATH. Run: eval \$(opam env --switch=coq-8.18)"
    fi
else
    warn "No COQ directory found, skipping Coq setup"
fi

# =============================================================================
# GIT CONFIGURATION
# =============================================================================
echo ""
echo -e "${BLUE}[4/5] Git Configuration${NC}"
echo "----------------------------------------------"

# Initialize Git LFS if needed
if command -v git-lfs &> /dev/null; then
    git lfs install --skip-repo 2>/dev/null || true
    success "Git LFS initialized"
fi

# Set up useful git aliases
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.lg "log --oneline --graph --decorate -10"
success "Git aliases configured"

# =============================================================================
# JUPYTER SETUP
# =============================================================================
echo ""
echo -e "${BLUE}[5/5] Jupyter Configuration${NC}"
echo "----------------------------------------------"

# Ensure Jupyter kernel is registered
python -m ipykernel install --user --name=gift --display-name="GIFT Python 3.11" 2>/dev/null || true
success "Jupyter kernel 'gift' registered"

# Build JupyterLab extensions
info "Building JupyterLab extensions..."
jupyter lab build --minimize=False 2>/dev/null || warn "JupyterLab build skipped"

# =============================================================================
# VERIFICATION
# =============================================================================
echo ""
echo "=============================================="
echo -e "${GREEN}  Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "Available tools:"
echo ""

# Python
PYTHON_VERSION=$(python --version 2>&1)
echo -e "  ${GREEN}✓${NC} Python:  $PYTHON_VERSION"

# Lean
if command -v lean &> /dev/null; then
    LEAN_VERSION=$(lean --version 2>&1 | head -1)
    echo -e "  ${GREEN}✓${NC} Lean:    $LEAN_VERSION"
else
    echo -e "  ${YELLOW}○${NC} Lean:    Run 'source ~/.bashrc' to activate"
fi

# Coq
if command -v coqc &> /dev/null; then
    COQ_VER=$(coqc --version 2>&1 | head -1)
    echo -e "  ${GREEN}✓${NC} Coq:     $COQ_VER"
else
    echo -e "  ${YELLOW}○${NC} Coq:     Run 'eval \$(opam env --switch=coq-8.18)'"
fi

# Jupyter
if command -v jupyter &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Jupyter: $(jupyter --version 2>&1 | head -1)"
fi

echo ""
echo "Quick start commands:"
echo ""
echo "  jupyter lab --no-browser     # Start Jupyter Lab"
echo "  ./verify.sh status           # Check verification status"
echo "  ./verify.sh g2               # Run G2 metric validation"
echo "  cd Lean && lake build        # Build Lean proofs"
echo "  cd COQ && make               # Build Coq proofs"
echo ""
echo "=============================================="
