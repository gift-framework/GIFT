# G2 Machine Learning Module

Neural network validation of G₂ metrics on compact K₇ manifolds for the GIFT framework.

## Structure

```
G2_ML/
├── G2_Lean/                    # Lean 4 certificates & validation
│   ├── G2CertificateV2_3_Portable_trained.ipynb   # Final G2 certificate
│   ├── Banach_FP_Verification_Colab_trained.ipynb # Banach fixed point
│   ├── numerical/
│   │   └── Spectral_Eigenvalue_Pipeline.ipynb     # λ₁ bounds
│   ├── *.lean                  # Lean certificates
│   └── *.json                  # Validation results
├── VERSIONS.md                 # Version history
└── archived.zip                # Historical development (v0.1-v2.1)
```

## Key Results

| Certificate | Value | Status |
|-------------|-------|--------|
| det(g) | 65/32 = 2.03125 | **Verified** |
| Banach K | < 0.9 | **Verified** |
| Joyce threshold | 0.1 | **35× safety margin** |
| λ₁ lower bound | > 0 | **Verified** |

## Usage

The notebooks can be run on Google Colab (click badge in notebook) or locally:

```bash
pip install torch numpy jupyter
jupyter notebook G2_Lean/G2CertificateV2_3_Portable_trained.ipynb
```

## Pipeline Integration

These certificates are used by the verification pipeline:

```bash
./verify.sh g2  # Validates G2 metrics using G2_ML/G2_Lean/
```

See [pipeline/README.md](../pipeline/README.md) for details.

## Historical Development

All previous implementations (v0.1 through v2.1) are preserved in `archived.zip`.
See [VERSIONS.md](VERSIONS.md) for the complete version history.

## Scientific Context

The G₂ metric on K₇ determines:
- **Dimensional reduction**: E₈×E₈ → Standard Model
- **Betti numbers**: b₂=21, b₃=77 (topological invariants)
- **Harmonic forms**: Basis for cohomology H*(K₇)

## License

MIT License (same as GIFT framework)
