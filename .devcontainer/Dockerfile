# =============================================================================
# GIFT Framework Development Container
# =============================================================================
# Complete environment for GIFT v2.3 development with:
# - Lean 4.14.0 + Mathlib
# - Coq 8.18
# - Python 3.11 + Scientific Stack
# - LaTeX for publication-quality exports
# - Visualization tools
# =============================================================================

FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

LABEL maintainer="GIFT Framework Team"
LABEL description="Complete GIFT development environment"
LABEL version="1.0"

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# =============================================================================
# SYSTEM PACKAGES
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    pkg-config \
    # Python
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # Git and utilities
    git \
    git-lfs \
    curl \
    wget \
    unzip \
    jq \
    htop \
    tree \
    # For Coq/opam
    opam \
    m4 \
    # For visualizations and LaTeX
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-plain-generic \
    texlive-science \
    latexmk \
    dvipng \
    ghostscript \
    # For matplotlib backends
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    # For interactive plots
    libcairo2-dev \
    libgirepository1.0-dev \
    # For PDF generation
    pandoc \
    wkhtmltopdf \
    # Fonts for visualizations
    fonts-liberation \
    fonts-dejavu \
    fonts-noto \
    fonts-firacode \
    # Misc
    locales \
    sudo \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_US.UTF-8

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# =============================================================================
# NON-ROOT USER SETUP
# =============================================================================
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# User should already exist in base image, but ensure correct setup
RUN if ! id -u $USERNAME > /dev/null 2>&1; then \
        groupadd --gid $USER_GID $USERNAME \
        && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# =============================================================================
# LEAN 4 INSTALLATION
# =============================================================================
USER $USERNAME
WORKDIR /home/$USERNAME

# Install elan (Lean version manager)
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none

# Add elan to PATH
ENV PATH="/home/${USERNAME}/.elan/bin:${PATH}"

# Install Lean 4.14.0
RUN /home/$USERNAME/.elan/bin/elan default leanprover/lean4:v4.14.0

# =============================================================================
# COQ INSTALLATION
# =============================================================================
# Initialize opam
RUN opam init --disable-sandboxing -y --bare \
    && opam switch create coq-8.18 ocaml-base-compiler.4.14.1 \
    && eval $(opam env --switch=coq-8.18) \
    && opam pin add coq 8.18.0 -y \
    && opam clean -a -c -s --logs

# Add opam to shell
RUN echo 'eval $(opam env --switch=coq-8.18)' >> ~/.bashrc

# =============================================================================
# PYTHON SCIENTIFIC STACK
# =============================================================================
USER root

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Core scientific computing
RUN pip install --no-cache-dir \
    numpy>=1.24 \
    scipy>=1.11 \
    sympy>=1.12 \
    pandas>=2.0 \
    polars>=0.19

# Machine learning & deep learning
RUN pip install --no-cache-dir \
    torch>=2.0 \
    torchvision \
    scikit-learn>=1.3 \
    xgboost \
    lightgbm

# Visualization - comprehensive stack
RUN pip install --no-cache-dir \
    matplotlib>=3.8 \
    seaborn>=0.13 \
    plotly>=5.18 \
    altair>=5.0 \
    bokeh>=3.3 \
    holoviews \
    hvplot \
    datashader \
    colorcet \
    cmocean \
    palettable

# Jupyter ecosystem
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab>=4.0 \
    notebook>=7.0 \
    ipywidgets>=8.0 \
    ipympl \
    jupyterlab-git \
    jupyterlab-lsp \
    jupyter-collaboration \
    nbconvert \
    nbformat \
    voila

# Math & symbolic computation
RUN pip install --no-cache-dir \
    mpmath \
    gmpy2 \
    numba \
    cython

# Data formats & I/O
RUN pip install --no-cache-dir \
    h5py \
    netcdf4 \
    pyarrow \
    fastparquet \
    openpyxl \
    xlrd

# Testing & quality
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-xdist \
    hypothesis \
    mypy \
    ruff \
    black \
    isort

# Documentation
RUN pip install --no-cache-dir \
    sphinx \
    sphinx-rtd-theme \
    myst-parser \
    sphinx-autodoc-typehints

# Utilities
RUN pip install --no-cache-dir \
    tqdm \
    rich \
    typer \
    pyyaml \
    toml \
    python-dotenv \
    requests \
    httpx

# Interactive & REPL enhancements
RUN pip install --no-cache-dir \
    ipython \
    ptpython \
    bpython

# =============================================================================
# JUPYTER CONFIGURATION
# =============================================================================
USER $USERNAME

# Configure Jupyter
RUN mkdir -p /home/$USERNAME/.jupyter \
    && echo "c.ServerApp.ip = '0.0.0.0'" >> /home/$USERNAME/.jupyter/jupyter_server_config.py \
    && echo "c.ServerApp.open_browser = False" >> /home/$USERNAME/.jupyter/jupyter_server_config.py \
    && echo "c.ServerApp.allow_origin = '*'" >> /home/$USERNAME/.jupyter/jupyter_server_config.py

# Configure matplotlib for better defaults
RUN mkdir -p /home/$USERNAME/.config/matplotlib \
    && echo "backend: Agg" > /home/$USERNAME/.config/matplotlib/matplotlibrc \
    && echo "figure.figsize: 12, 8" >> /home/$USERNAME/.config/matplotlib/matplotlibrc \
    && echo "figure.dpi: 150" >> /home/$USERNAME/.config/matplotlib/matplotlibrc \
    && echo "savefig.dpi: 300" >> /home/$USERNAME/.config/matplotlib/matplotlibrc \
    && echo "font.size: 12" >> /home/$USERNAME/.config/matplotlib/matplotlibrc \
    && echo "axes.titlesize: 14" >> /home/$USERNAME/.config/matplotlib/matplotlibrc \
    && echo "axes.labelsize: 12" >> /home/$USERNAME/.config/matplotlib/matplotlibrc

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# =============================================================================
# FINAL SETUP
# =============================================================================
USER $USERNAME
WORKDIR /workspaces/GIFT

# Default command
CMD ["bash"]
