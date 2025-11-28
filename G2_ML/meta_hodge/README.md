# Meta-Hodge extraction pipeline

This directory contains a *new* analysis layer that mines all historical
K7/G2 PINN runs (v0.1 → v1.8) without modifying them. The goal is to reuse
the learned metrics and G2 3-forms as a database of approximate geometries,
assemble improved Hodge operators, and extract approximate harmonic forms and
Yukawa couplings offline.

## Layout

- `config.py` – discovery utilities and lightweight registry of historical
  notebooks/checkpoints plus a helper to summarize discovered assets.
- `geometry_loader.py` – unified API to sample coordinates and load metric/phi
  predictors as `ModelBundle` objects.
- `candidate_library.py` – aggregates candidate 2- and 3-form features from all
  versions to form a broad search basis.
- `hodge_operators.py` – numerical Hodge star, basic Laplacian assembly, and
  utilities for canonical bases.
- `harmonic_solver.py` – eigensolvers that return approximate `H^2` and `H^3`
  modes in the candidate subspaces.
- `yukawa_extractor.py` – Monte Carlo evaluator for Yukawa couplings using the
  recovered harmonic bases.
- `K7_Meta_Hodge_v1_0.ipynb` – orchestration notebook demonstrating the full
  pipeline end-to-end.

## Usage sketch

```python
import torch
from pathlib import Path
from G2_ML.meta_hodge import (
    CandidateLibrary,
    HarmonicSolver,
    HodgeOperator,
    load_version_model,
    sample_coords,
    YukawaExtractor,
)

x = sample_coords(4000)
bundle = load_version_model("1_8")
metric = bundle.metric_fn(x)

library = CandidateLibrary()
candidates = library.collect(x)

op = HodgeOperator(metric)
solver = HarmonicSolver(op)
results = solver.solve(candidates.c2, candidates.c3)

extractor = YukawaExtractor(x, metric)
yukawas = extractor.compute(results["H2"].eigenvectors, results["H3"].eigenvectors)
extractor.save(yukawas, Path("artifacts/meta_hodge"))
```

To quickly **explore** what assets are available without running the full
pipeline, use the CLI helper:

```bash
python scripts/run_meta_hodge.py --explore --export-registry artifacts/meta_hodge/registry.json
```

This prints a short summary and optionally writes the registry to disk for
later reference.

The provided implementation emphasizes clarity and modularity; individual
components can be swapped for more precise physics-aware approximations without
changing the public interfaces.
