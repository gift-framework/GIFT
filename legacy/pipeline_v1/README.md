# GIFT Framework Verification Pipeline (ARCHIVED)

> **⚠️ ARCHIVED**: This pipeline has been superseded by [giftpy](https://pypi.org/project/giftpy/) from [gift-framework/core](https://github.com/gift-framework/core).
>
> **Use instead:**
> ```bash
> pip install giftpy
> ```
>
> The `giftpy` package (v1.4.0) includes:
> - 39 certified relations (Lean 4 + Coq verified)
> - K₇ metric pipeline with G₂ geometry
> - Monte Carlo validation
> - All formal proofs

---

**Version**: 1.0 (archived)
**GIFT Version**: 2.3
**Last Updated**: 2025-12-03

---

## Overview (Historical)

This directory contains the archived verification pipeline. It was superseded by giftpy.

Original capabilities:

1. **Lean 4 Formal Verification**: 25 exact relations (now 39 in giftpy v1.4.0)
2. **Coq Formal Verification**: Independent verification
3. **G2 Metric Validation**: PINN-based reconstruction

---

## Quick Start

### Setup (First Time)

The formal proofs (Lean 4 + Coq) are maintained in the **[gift-framework/core](https://github.com/gift-framework/core)** repository. Before running verification, clone it locally:

```bash
# Clone core repository (required for Lean/Coq verification)
./pipeline/scripts/setup_core.sh

# Or install the Python package
pip install giftpy
```

### Full Verification

```bash
# From repository root
./verify.sh all
```

### Individual Components

```bash
./verify.sh lean      # Lean 4 verification only
./verify.sh coq       # Coq verification only
./verify.sh g2        # G2 metric validation only
./verify.sh checksums # Compute source checksums
./verify.sh report    # Generate report from existing results
./verify.sh status    # Show current verification status
./verify.sh clean     # Remove all generated outputs
```

### Using Make

```bash
cd pipeline
make all        # Full verification
make lean       # Lean only
make coq        # Coq only
make g2         # G2 only
make report     # Generate report
make checksums  # Compute checksums
make clean      # Clean outputs
make help       # Show available targets
```

---

## Directory Structure

```
pipeline/
├── README.md                 # This file
├── Makefile                  # Make-based orchestration
├── config.env                # Configuration parameters
│
├── scripts/                  # Verification scripts
│   ├── setup_core.sh         # Clone/update gift-framework/core
│   ├── verify_lean.sh        # Lean 4 build and verification
│   ├── verify_coq.sh         # Coq build and verification
│   ├── verify_g2.sh          # G2 metric validation
│   ├── compute_checksums.sh  # Source checksum computation
│   └── generate_report.sh    # Report generation
│
├── notebooks/                # Canonical Jupyter notebooks
│   ├── 01_G2_Metric_Validation.ipynb
│   ├── 02_Lean_Verification.ipynb
│   ├── 03_Coq_Verification.ipynb
│   └── 04_Framework_Report.ipynb
│
├── outputs/                  # Generated outputs
│   ├── lean/                 # Lean verification results
│   ├── coq/                  # Coq verification results
│   ├── g2/                   # G2 validation results
│   ├── reports/              # Generated reports
│   └── checksums/            # Checksum manifests
│
└── templates/                # Report templates
    └── verification_report.md
```

---

## Prerequisites

### Required Tools

| Tool | Version | Purpose |
|------|---------|---------|
| Lean 4 | 4.14.0 | Formal verification |
| Mathlib | 4.14.0 | Mathematical library for Lean |
| Coq | 8.18+ | Formal verification |
| Python | 3.10+ | G2 validation, notebooks |
| jq | 1.6+ | JSON processing |
| sha256sum | any | Checksum computation |

### Optional Tools

| Tool | Purpose |
|------|---------|
| Jupyter | Running notebooks |
| PyTorch | G2 PINN validation |

### Installation

**Lean 4**:
```bash
# Using elan (recommended)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
elan default leanprover/lean4:v4.14.0
```

**Coq**:
```bash
# Using opam
opam install coq.8.18.0
```

**Python dependencies**:
```bash
pip install giftpy  # Official GIFT package from gift-framework/core
pip install torch numpy jupyter
```

**Setup Core Repository** (for local Lean/Coq verification):
```bash
./pipeline/scripts/setup_core.sh
```

---

## Output Formats

### Lean Verification Output

File: `outputs/lean/verification.json`

```json
{
  "timestamp": "2025-12-03T14:30:00Z",
  "component": "lean",
  "lean_version": "4.14.0",
  "status": "PASS",
  "theorems": {
    "total": 13,
    "verified": 13
  },
  "sorry_count": 0,
  "source_checksum": "sha256:..."
}
```

### Coq Verification Output

File: `outputs/coq/verification.json`

```json
{
  "timestamp": "2025-12-03T14:35:00Z",
  "component": "coq",
  "coq_version": "8.18.0",
  "status": "PASS",
  "theorems": {
    "total": 13,
    "verified": 13
  },
  "admitted_count": 0,
  "source_checksum": "sha256:..."
}
```

### G2 Validation Output

File: `outputs/g2/validation.json`

```json
{
  "timestamp": "2025-12-03T14:40:00Z",
  "component": "g2_metric",
  "status": "PASS",
  "metric_validation": {
    "det_g_computed": 2.0312490,
    "det_g_exact": 2.03125,
    "deviation_percent": 0.000025
  },
  "banach_certificate": {
    "contraction_constant_K": 0.9,
    "safety_margin": 35
  }
}
```

---

## Verified Relations

The pipeline verifies 25 exact relations (13 original + 12 topological extension), all derived from zero adjustable parameters.

### Original 13 Relations

| # | Observable | Formula | Value |
|---|------------|---------|-------|
| 1 | sin²θ_W | b₂/(b₃ + dim(G₂)) | 3/13 = 0.23077 |
| 2 | τ | (496 × 21)/(27 × 99) | 3472/891 |
| 3 | det(g) | (5 × 13)/32 | 65/32 = 2.03125 |
| 4 | κ_T | 1/(b₃ - dim(G₂) - p₂) | 1/61 |
| 5 | δ_CP | 7 × dim(G₂) + H* | 197° |
| 6 | m_τ/m_e | 7 + 10×248 + 10×99 | 3477 |
| 7 | m_s/m_d | 4 × Weyl_factor | 20 |
| 8 | Q_Koide | dim(G₂)/b₂ | 2/3 |
| 9 | λ_H | (dim(G₂) + N_gen)/32 | 17/32 |
| 10 | H* | b₂ + b₃ + 1 | 99 |
| 11 | p₂ | dim(G₂)/dim(K₇) | 2 |
| 12 | N_gen | Topological | 3 |
| 13 | dim(E₈×E₈) | 2 × 248 | 496 |

### Topological Extension (12 New Relations)

| # | Observable | Formula | Value |
|---|------------|---------|-------|
| 14 | α_s denom | dim(G₂) - p₂ | 12 |
| 15 | γ_GIFT | (2·rank(E₈) + 5·H*) / (10·dim(G₂) + 3·dim(E₈)) | 511/884 |
| 16 | δ_penta | Weyl² | 25 |
| 17 | θ₂₃ | (rank(E₈) + b₃) / H* | 85/99 |
| 18 | θ₁₃ denom | b₂ | 21 |
| 19 | α_s² denom | (dim(G₂) - p₂)² | 144 |
| 20 | λ_H² | (dim(G₂) + N_gen) / 32² | 17/1024 |
| 21 | θ₁₂ factor | Weyl² × γ_num | 12775 |
| 22 | m_μ/m_e base | dim(J₃(O)) | 27 |
| 23 | n_s indices | D_bulk, Weyl | 11, 5 |
| 24 | Ω_DE frac | (H* - 1) / H* | 98/99 |
| 25 | α⁻¹ base | (dim(E₈) + rank(E₈))/2 + H*/11 | 137 |

---

## Checksum Verification

The pipeline computes SHA-256 checksums for all source files to ensure integrity.

### Verify Checksums

```bash
./verify.sh checksums
cat pipeline/outputs/checksums/manifest.txt
```

### Checksum Manifest Format

```
# GIFT Framework Checksum Manifest
# Generated: 2025-12-03T14:30:00Z
# Algorithm: SHA-256

## Lean Sources
sha256:abc123...  Lean/GIFT.lean
sha256:def456...  Lean/GIFT/Algebra/E8RootSystem.lean
...

## Coq Sources
sha256:789ghi...  COQ/Algebra/E8RootSystem.v
...

## Aggregate
sha256:xyz789...  GIFT_AGGREGATE
```

---

## Troubleshooting

### Core repository not found

```bash
# Clone gift-framework/core
./pipeline/scripts/setup_core.sh

# Or force fresh clone
./pipeline/scripts/setup_core.sh --force
```

### Lean build fails

```bash
# Check Lean version
lean --version

# Update dependencies (in core)
cd .cache/gift-core/Lean
lake update
lake build
```

### Coq build fails

```bash
# Check Coq version
coqc --version

# Rebuild (in core)
cd .cache/gift-core/Coq
make clean
make depend
make
```

### Missing jq

```bash
# Ubuntu/Debian
sudo apt install jq

# macOS
brew install jq
```

---

## Reports

Generated reports are stored in `outputs/reports/`:

- `GIFT_Verification_Report_YYYY-MM-DD.md`: Dated report
- `GIFT_Verification_Report_latest.md`: Symlink to most recent

### Report Contents

1. Executive Summary
2. Lean 4 Verification Details
3. Coq Verification Details
4. G2 Metric Validation Details
5. Cross-Verification Matrix
6. Checksum Manifest
7. Reproducibility Instructions

---

## Configuration

Edit `config.env` to modify pipeline parameters:

```bash
# Framework versions
GIFT_VERSION="2.3"
PIPELINE_VERSION="1.0"

# Expected values
EXPECTED_LEAN_THEOREMS=13
EXPECTED_COQ_THEOREMS=13
DET_G_EXACT_DECIMAL="2.03125"
DET_G_TOLERANCE="0.0001"
```

---

## Contributing

When modifying verification components:

1. Update the corresponding script in `scripts/`
2. Update `config.env` if parameters change
3. Run `./verify.sh all` to ensure pipeline passes
4. Update this README if behavior changes

---

## License

This pipeline is part of the GIFT framework. See repository root for license information.
