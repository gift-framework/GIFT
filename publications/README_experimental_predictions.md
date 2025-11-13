# GIFT Framework v2.0 - Experimental Predictions

**Reference datasets for DUNE, Belle II, LHCb, and collider searches**

## Overview

This directory contains high-precision experimental predictions from the GIFT framework, specifically designed for upcoming experimental programs:

1. **DUNE Neutrino Oscillations** (2028-2032)
   - Complete oscillation spectra (νμ → νe, νμ → νμ)
   - CP violation predictions (δCP = 197° EXACT)
   - Energy range: 0.5 - 5.0 GeV
   - Baseline: 1300 km

2. **New Particle Searches**
   - 3.897 GeV scalar (H³(K₇) cohomology)
   - 20.4 GeV gauge boson (E₈×E₈ structure)
   - 4.77 GeV dark matter candidate (K₇ geometry)

3. **Precision Flavor Physics** (Belle II, LHCb)
   - CKM matrix correlations
   - CP violation in B/D decays
   - Rare decay predictions

## Files

- **`gift_experimental_predictions.ipynb`**: Complete Jupyter notebook
- **Generated outputs**:
  - `dune_gift_predictions.csv`: DUNE oscillation data (500 energy points)
  - `experimental_predictions.json`: Complete predictions database
  - `dune_oscillation_predictions.png`: DUNE visualizations
  - `new_particles_cross_sections.png`: Collider predictions

## Quick Start

```bash
cd /home/user/GIFT/publications
jupyter notebook gift_experimental_predictions.ipynb
```

Run all cells to generate complete prediction datasets.

## Key Predictions

### DUNE (Deep Underground Neutrino Experiment)

**GIFT Exact Prediction: δCP = 197°**

This is one of the most striking predictions:
- Formula: δCP = 7×dim(G₂) + H*(K₇) = 7×14 + 99 = 197°
- Status: **EXACT** (proven in Supplement B)
- Current measurement: 197° ± 24° (NuFIT 5.3)
- DUNE precision target: ±2-5° by 2030

**Falsification criterion**: If DUNE measures δCP significantly different from 197° with <5° uncertainty, GIFT is falsified.

### New Particles

| Particle | Mass | Origin | Search Venue |
|----------|------|--------|--------------|
| Scalar | 3.897 GeV | H³(K₇), b₃=77 | Belle II, LHCb, BaBar |
| Gauge Boson | 20.4 GeV | E₈×E₈, b₂=21 | LHC, Future colliders |
| Dark Matter | 4.77 GeV | K₇ geometry | XENON, LZ, Fermi |

## Usage for Experimentalists

### For DUNE Collaboration

1. Load oscillation predictions:
```python
import pandas as pd
dune_data = pd.read_csv('dune_gift_predictions.csv')
```

2. Compare with your Monte Carlo:
```python
# GIFT prediction at specific energy
E_nu = 2.5  # GeV
P_numu_nue_GIFT = dune_data[dune_data['Energy_GeV'].between(2.49, 2.51)]['P_numu_nue'].mean()
```

3. Test δCP = 197° hypothesis in your fits

### For Belle II / LHCb

Search for 3.897 GeV scalar in:
- B → K + X (missing energy signature)
- Υ → γ + X (radiative decays)
- D decays (if couples to charm)

Expected signatures:
- Narrow resonance (~10 MeV width)
- Flavor-changing neutral currents
- Possible invisible decays

### For Direct Dark Matter Searches

4.77 GeV candidate:
- Nuclear recoil searches (XENON, LZ)
- Spin-independent cross-section predictions
- Indirect detection (Fermi-LAT photon lines)

## Computational Details

### Neutrino Oscillations

Three-flavor oscillation probabilities computed using:
- PMNS matrix from GIFT parameters
- Standard vacuum oscillation formula
- Mass-squared differences from PDG 2024

Includes:
- Matter effects (Earth density profile)
- CP asymmetry calculations
- Energy-dependent spectra

### Production Cross-Sections

Rough estimates using:
- Threshold behavior: σ ∝ (1 - m²/s)^(3/2)
- Gluon fusion mechanism
- Normalized to typical BSM scales

**Note**: Detailed cross-sections require full model implementation with couplings.

## Integration with Experiments

### DUNE

Contact DUNE physics coordination with:
- `dune_gift_predictions.csv` - oscillation spectra
- Reference: GIFT Framework v2.0, δCP = 197° prediction

Suggest including GIFT hypothesis in:
- CP violation analysis
- Systematics studies
- Publication comparisons

### Belle II

Contact heavy flavor physics group with:
- 3.897 GeV scalar search proposal
- Expected decay channels
- Background estimates

### LHC Experiments

Contact ATLAS/CMS exotics groups with:
- 20.4 GeV gauge boson prediction
- Drell-Yan production estimates
- Mono-X signatures for DM

## Publication Use

Recommended citation:
```
GIFT Framework v2.0 predicts δCP = 197° (exact) for DUNE,
and three new particles at 3.9 GeV (scalar), 20.4 GeV (gauge boson),
and 4.77 GeV (dark matter). See gift_experimental_predictions.ipynb
for complete datasets and methodology.
```

## Updates

This dataset will be updated:
- When experimental precision improves
- If new colliders come online
- As GIFT framework evolves

Check repository for latest version.

## Contact

- **Repository**: https://github.com/gift-framework/GIFT
- **Issues**: https://github.com/gift-framework/GIFT/issues
- **Experimental collaborations**: Contact via GitHub issues

---

**Generated**: 2025-11-13
**GIFT Version**: 2.0
**Purpose**: Experimental collaboration reference
