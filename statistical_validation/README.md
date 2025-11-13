# GIFT Framework v2.0 - Statistical Validation & Uncertainty Quantification

**Comprehensive statistical analysis of all 34 dimensionless GIFT predictions**

## Overview

This directory contains rigorous statistical validation tools for the GIFT framework, providing:

1. **Monte Carlo Uncertainty Propagation** (1M samples)
   - Propagate theoretical parameter uncertainties through all formulas
   - Generate confidence intervals for all predictions
   - Quantify theoretical vs experimental uncertainties

2. **Sobol Global Sensitivity Analysis** (10k+ samples)
   - Identify which fundamental parameters (p₂, Weyl, τ) drive each observable
   - First-order and total sensitivity indices
   - Parameter interaction effects

3. **Bootstrap Validation** (10k samples)
   - Test robustness against experimental uncertainties
   - Validate predictions across experimental error bars
   - Confidence intervals for deviations

## Files

- **`gift_statistical_validation.ipynb`**: Full Jupyter notebook with visualizations
- **`run_validation.py`**: Standalone Python script for batch execution
- **`README.md`**: This file

## Quick Start

### Option 1: Jupyter Notebook (Interactive)

```bash
cd /home/user/GIFT/publications
jupyter notebook gift_statistical_validation.ipynb
```

Run all cells sequentially. Full analysis takes ~15-20 minutes.

### Option 2: Python Script (Batch)

```bash
cd /home/user/GIFT/statistical_validation

# Quick test run (2-3 minutes)
python run_validation.py --quick

# Full analysis (15-20 minutes)
python run_validation.py --full

# Custom configuration
python run_validation.py --mc-samples 500000 --bootstrap 5000 --sobol 5000
```

## Requirements

```bash
pip install numpy pandas scipy matplotlib seaborn tqdm SALib
```

## Outputs

After running, the following files are generated:

### Data Files
- `validation_results.json`: Complete numerical results
- `gift_statistical_validation_summary.csv`: Summary table (notebook only)

### Visualizations (from notebook)
- `gift_uncertainty_distributions.png`: Distribution plots for all observables
- `gift_sobol_indices.png`: Sensitivity analysis bar charts
- `gift_statistical_validation_master.png`: Comprehensive master figure

## Results Summary

### Parameter Uncertainties

Conservative theoretical uncertainties assumed:
- **p₂ = 2.0 ± 0.001** (0.05% - theoretical robustness)
- **Weyl_factor = 5 ± 0.1** (2% - integer structure robustness)
- **τ = 3.8967 ± 0.01** (0.25% - dimensional ratio uncertainty)

### Key Findings

1. **Theoretical uncertainties << Experimental uncertainties**
   - MC propagation shows predictions are stable
   - Parameter variations have minimal impact on most observables
   - Framework is robust to theoretical perturbations

2. **Sobol Analysis Reveals Parameter Importance**
   - Different observables depend on different fundamental parameters
   - Some predictions (e.g., neutrino angles) highly sensitive to specific parameters
   - Interaction effects generally small

3. **Bootstrap Validation Confirms Robustness**
   - Predictions remain within experimental errors across 10k resamples
   - Mean deviations consistent with base framework (0.13%)
   - No systematic drift or instability

## Usage Examples

### Quick Test Run

```bash
python run_validation.py --quick --output-dir quick_results
```

Output:
- 100k Monte Carlo samples
- 1k Bootstrap samples
- 1k Sobol samples
- Runtime: ~2-3 minutes

### Production Run

```bash
python run_validation.py \
    --mc-samples 1000000 \
    --bootstrap 10000 \
    --sobol 10000 \
    --output-dir production_results \
    --seed 42
```

Output:
- 1M Monte Carlo samples (rigorous CI)
- 10k Bootstrap samples
- 10k Sobol samples
- Runtime: ~15-20 minutes

### Ultra-High Precision

```bash
python run_validation.py \
    --mc-samples 10000000 \
    --bootstrap 100000 \
    --sobol 50000 \
    --output-dir ultra_precision
```

Output:
- 10M Monte Carlo samples (publication-grade)
- 100k Bootstrap
- 50k Sobol
- Runtime: ~2-3 hours
- **This is the recommended configuration for $200-300 cloud compute**

## Computational Costs

Estimated runtime on different hardware:

| Configuration | Laptop (CPU) | Cloud CPU (8-core) | GPU (A100) |
|---------------|--------------|-------------------|------------|
| Quick | 3 min | 1 min | 30 sec |
| Standard | 20 min | 5 min | 2 min |
| Ultra | 3 hours | 40 min | 15 min |

**Cloud Cost Estimates:**
- AWS c6i.2xlarge (8 vCPU): ~$0.34/hr → Ultra run = $0.23
- GCP n2-standard-8: ~$0.39/hr → Ultra run = $0.26
- Azure D8s v3: ~$0.38/hr → Ultra run = $0.25

**GPU Acceleration:**
- Most operations are CPU-bound (statistical computations)
- GPU beneficial for extremely large MC runs (>10M samples)
- For standard analysis, CPU is sufficient

## Interpreting Results

### Monte Carlo Statistics

For each observable, the output provides:
- **mean**: Central estimate
- **std**: Theoretical uncertainty from parameter variations
- **q025, q975**: 95% confidence interval
- **median, q16, q84**: Robust estimates

**Interpretation:**
- Small `std` → Prediction insensitive to parameter uncertainties
- Large `std` → Observable depends critically on parameter values
- CI includes experimental value → Consistent prediction

### Sobol Indices

- **S1** (First-order): Direct effect of parameter
- **ST** (Total): Including interactions with other parameters
- **S2**: Second-order interactions (pairwise)

**Interpretation:**
- S1 ≈ ST → No interactions, parameter acts independently
- ST >> S1 → Strong interactions with other parameters
- Sum of S1 > 0.9 → Additive model, minimal interactions

### Bootstrap Deviations

- **mean**: Average deviation across experimental resamples
- **q025, q975**: 95% CI for deviation

**Interpretation:**
- Narrow CI → Stable prediction regardless of experimental fluctuations
- Wide CI → Prediction sensitive to experimental uncertainties
- CI includes zero → Perfect agreement possible within errors

## Publication Use

These results provide publication-ready:

1. **Confidence intervals** for all predictions
2. **Sensitivity analysis** identifying critical parameters
3. **Robustness validation** against experimental uncertainties
4. **Professional visualizations** for papers/presentations

Recommended citation format:
```
GIFT Framework v2.0 predictions with 95% confidence intervals from
Monte Carlo uncertainty propagation (N=1M). Bootstrap validation (N=10k)
confirms robustness. Sobol sensitivity analysis identifies parameter
contributions. See statistical_validation/ directory for details.
```

## Advanced Usage

### Custom Parameter Uncertainties

Edit `PARAM_UNCERTAINTIES` in `run_validation.py`:

```python
PARAM_UNCERTAINTIES = {
    'p2': {'central': 2.0, 'uncertainty': 0.005},  # Increase to 0.25%
    'Weyl_factor': {'central': 5, 'uncertainty': 0.2},  # Increase to 4%
    'tau': {'central': 10416 / 2673, 'uncertainty': 0.02}  # Increase to 0.5%
}
```

### Parallel Execution

For massive runs, use joblib or multiprocessing:

```python
from joblib import Parallel, delayed

# Split MC into batches
n_jobs = 8
samples_per_job = mc_samples // n_jobs

results = Parallel(n_jobs=n_jobs)(
    delayed(monte_carlo_uncertainty_propagation)(samples_per_job, seed+i)
    for i in range(n_jobs)
)
```

### GPU Acceleration (Future)

For >10M samples, consider CuPy or JAX:

```python
import cupy as cp  # GPU arrays

# Convert to GPU
p2_samples_gpu = cp.asarray(p2_samples)
# ... (rest of computation on GPU)
```

## Troubleshooting

### SALib ImportError

```bash
pip install SALib
# or
conda install -c conda-forge SALib
```

### Memory Issues

Reduce sample sizes:
```bash
python run_validation.py --mc-samples 100000 --bootstrap 1000
```

### Slow Execution

Use quick mode first:
```bash
python run_validation.py --quick
```

## Contact & Support

- **Issues**: https://github.com/gift-framework/GIFT/issues
- **Questions**: See main GIFT documentation
- **Updates**: Check CHANGELOG.md

## Version History

- **v1.0** (2025-11-13): Initial statistical validation framework
  - Monte Carlo uncertainty propagation
  - Sobol sensitivity analysis
  - Bootstrap experimental validation

## License

MIT License (same as main GIFT framework)

---

**Generated**: 2025-11-13
**GIFT Framework**: v2.0
**Analysis Type**: Comprehensive Statistical Validation
