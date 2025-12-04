# Statistical Validation and Uncertainty Quantification

**GIFT Framework v2.3**

## Overview

This directory contains tools for statistical validation of GIFT predictions, including Monte Carlo uncertainty propagation, Sobol sensitivity analysis, and bootstrap validation on experimental data.

## Core Implementations

| File | Version | Description |
|------|---------|-------------|
| `gift_v23_core.py` | **v2.3** | Current - 13 PROVEN relations (8 Lean 4 verified) |
| `gift_v22_core.py` | v2.2 | Zero-parameter paradigm |
| `gift_v21_core.py` | v2.1 | Torsional dynamics |
| `run_validation_v21.py` | v2.1 | Validation runner |

## Contents

- `gift_v23_core.py`: Core implementation v2.3 with Lean 4 verification
- `gift_v22_core.py`: Core implementation v2.2
- `gift_v21_core.py`: Core implementation v2.1
- `run_validation_v21.py`: Validation runner script
- `requirements.txt`: Python dependencies
- `full_results/`: Output directory for validation results

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- tqdm >= 4.62.0
- SALib >= 1.4.0

## Usage

### Quick Test

```bash
python run_validation.py --quick
```

Executes reduced analysis:
- Monte Carlo: 100,000 samples
- Bootstrap: 1,000 samples
- Sobol: 1,000 samples
- Runtime: approximately 2 minutes

### Standard Analysis

```bash
python run_validation.py --full
```

Default configuration:
- Monte Carlo: 1,000,000 samples
- Bootstrap: 10,000 samples
- Sobol: 10,000 samples
- Runtime: approximately 15-20 minutes

### Custom Configuration

```bash
python run_validation.py \
    --mc-samples 10000000 \
    --bootstrap 100000 \
    --sobol 50000 \
    --output-dir custom_results \
    --seed 42
```

## Methodology

### Monte Carlo Uncertainty Propagation

Propagates parameter uncertainties through GIFT formulas using sampling:

Parameter uncertainties (assumed):
- p2: 2.0 ± 0.001 (0.05%)
- Weyl_factor: 5 ± 0.1 (2%)
- tau: 3.8967 ± 0.01 (0.25%)

For each sample:
1. Draw parameter values from normal distributions
2. Compute all observables
3. Record results

Output statistics:
- Mean and standard deviation
- Median and percentiles (2.5%, 16%, 84%, 97.5%)
- Confidence intervals

### Sobol Sensitivity Analysis

Global sensitivity analysis using Saltelli sampling:

1. Generate parameter combinations using Sobol sequences
2. Evaluate model for all combinations
3. Compute sensitivity indices:
   - S1: First-order effects
   - ST: Total effects (including interactions)
   - S2: Second-order interactions

Interpretation:
- S1 indicates direct parameter influence
- ST - S1 indicates interaction effects
- Sum of S1 across parameters indicates additive nature

### Bootstrap Validation

Resamples experimental data within uncertainties:

1. For each bootstrap iteration:
   - Sample experimental values from normal distributions
   - Compute deviations from GIFT predictions
2. Aggregate results across iterations
3. Compute confidence intervals for deviations

## Output Format

### JSON Results

`validation_results.json` contains:

```json
{
  "metadata": {
    "timestamp": "ISO-8601 format",
    "mc_samples": 1000000,
    "bootstrap_samples": 10000,
    "sobol_samples": 10000
  },
  "monte_carlo_statistics": {
    "observable_name": {
      "mean": float,
      "std": float,
      "q025": float,
      "q975": float,
      ...
    }
  },
  "bootstrap_statistics": {...},
  "sobol_indices": {...}
}
```

### CSV Summary

`gift_statistical_validation_summary.csv` includes:
- Observable name
- GIFT prediction (MC mean)
- MC standard deviation
- MC 95% confidence interval
- Experimental value and uncertainty
- Bootstrap deviation statistics
- Sobol sensitivity indices

## Computational Requirements

### Runtime Estimates

| Configuration | Laptop (CPU) | Cloud (8-core) | GPU |
|---------------|--------------|----------------|-----|
| Quick | 3 min | 1 min | 30 sec |
| Standard | 20 min | 5 min | 2 min |
| High precision (10M) | 3 hours | 40 min | 15 min |

### Memory Requirements

- Quick: ~1 GB
- Standard: ~4 GB
- High precision: ~16 GB

## Cloud Execution

### AWS Example

```bash
# Launch c6i.8xlarge instance (32 vCPU)
aws ec2 run-instances --instance-type c6i.8xlarge ...

# Copy files and execute
scp -r statistical_validation/ ubuntu@instance:/home/ubuntu/
ssh ubuntu@instance
cd statistical_validation
python run_validation.py --full
```

Cost estimate: c6i.8xlarge at $1.36/hour
- Standard run (5 min): $0.11
- High precision (40 min): $0.91

### Google Cloud Example

```bash
# Launch n2-standard-8 instance
gcloud compute instances create validation-instance \
    --machine-type=n2-standard-8

# Execute validation
gcloud compute ssh validation-instance
cd statistical_validation
python run_validation.py --full
```

Cost estimate: n2-standard-8 at $0.39/hour
- Standard run (5 min): $0.03
- High precision (40 min): $0.26

## Interpreting Results

### Monte Carlo Statistics

Standard deviation indicates theoretical uncertainty from parameter variations:
- Small std: Prediction insensitive to parameter uncertainties
- Large std: Observable depends critically on parameter values

### Sobol Indices

First-order index (S1):
- S1 > 0.5: Parameter dominates observable
- S1 < 0.1: Parameter has minor influence

Total index (ST):
- ST ≈ S1: Parameter acts independently
- ST >> S1: Strong interactions with other parameters

### Bootstrap Deviations

Confidence intervals on deviations indicate robustness:
- Narrow CI: Stable prediction across experimental uncertainties
- Wide CI: Prediction sensitive to experimental values

## Troubleshooting

### SALib Not Found

```bash
pip install SALib
```

If installation fails, Sobol analysis will be skipped with warning message.

### Memory Error

Reduce sample sizes:
```bash
python run_validation.py --mc-samples 100000 --bootstrap 1000
```

### Slow Execution

Use quick mode for testing:
```bash
python run_validation.py --quick
```

Or increase parallelization (requires modification of script).

## Citation

If using these validation results in publications:

```
GIFT Framework v2.0 statistical validation with Monte Carlo
uncertainty propagation (N=1M), Sobol sensitivity analysis
(N=10k), and bootstrap validation (N=10k).
Repository: https://github.com/gift-framework/GIFT
```

## Files

```
statistical_validation/
├── README.md                    # This file
├── run_validation.py           # Standalone script
├── requirements.txt            # Dependencies
├── quick_test/                 # Quick test results
│   └── validation_results.json
└── full_results/               # Full validation results
    └── validation_results.json
```

## Version History

- v2.3 (2024-12): Added gift_v23_core.py with Lean 4 verification
  - 13 PROVEN relations (8 Lean 4 verified)
  - 39 observables with mean deviation 0.128%
- v2.2 (2024-11): Zero-parameter paradigm
  - sin^2(theta_W) = 3/13, tau = 3472/891, det(g) = 65/32
- v2.1 (2024-11): Torsional dynamics integration
- v1.0 (2024-11-13): Initial implementation
  - Monte Carlo propagation
  - Sobol sensitivity analysis
  - Bootstrap validation
  - Test run completed (100k samples)
  - Full run completed (1M samples)

## Contact

For questions or issues:
- Repository: https://github.com/gift-framework/GIFT
- Issues: https://github.com/gift-framework/GIFT/issues

---

**Last updated**: 2024-12-04
**GIFT version**: 2.3
**Analysis type**: Statistical validation
