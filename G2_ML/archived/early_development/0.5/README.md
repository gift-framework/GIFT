# G2 Metric Training v0.5 - GIFT Geometry Results

## Overview

This directory contains results from training a G2 metric on K₇ using GIFT-parametrized geometry.

**Key Innovation**: Hierarchical T⁷ radii and twist angles determined by GIFT framework constants (τ, ξ, β₀, δ, γ, φ).

## Files

- `config_v05.json`: Training configuration
- `training_history.csv`: Complete training history
- `test_history.json`: Test set metrics (recorded every 1000 epochs)
- `validation_results.json`: Final validation metrics
- `phi_network_final.pt`: Trained φ network weights
- `harmonic_network_final.pt`: Trained harmonic 2-forms network
- `metric_samples.npz`: Validation metric samples (12k points)
- `comparison_with_v04.json`: Comparison with v0.4
- `training_dashboard_v05.png`: Comprehensive visualization

- `b3_extraction_results.json`: b₃=77 extraction results
- `b3_forms.npy`: Extracted 77 harmonic 3-forms



## Results Summary

**Training**:
- Total time: 6.79 hours
- Final torsion (test): 7.60e-10
- Final det(Gram) b₂ (test): 1.268

**New Features**:
- b₃=77 extraction: 77/77 forms (det=-0.000)
- Yukawa couplings: Pending (run Section 8)

**Hypothesis Status**: HYPOTHESIS PARTIALLY VALIDATED

## GIFT Geometry Parameters

- τ (hierarchical scaling): 3.896745
- ξ (primary twist): 0.981748
- β₀ (phase parameter): 0.392699
- δ (secondary twist): 0.251327
- γ (asymptotic scaling): 0.578054
- φ (golden ratio): 1.618034

## Reference

GIFT Framework: https://doi.org/10.5281/zenodo.17434034

Generated: 2025-11-09 15:41:10


