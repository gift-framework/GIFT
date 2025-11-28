#!/usr/bin/env python
"""CLI entry point for the meta-hodge pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from G2_ML.meta_hodge import (
    CandidateLibrary,
    HarmonicSolver,
    HodgeOperator,
    locate_historical_assets,
    summarize_registry,
    load_version_model,
    sample_coords,
    YukawaExtractor,
)


def main():
    parser = argparse.ArgumentParser(description="Run the meta-hodge extractor headlessly.")
    parser.add_argument("--version", default="1_8", help="Reference metric version to use.")
    parser.add_argument("--samples", type=int, default=2000, help="Number of Monte Carlo samples.")
    parser.add_argument("--out", type=Path, default=Path("artifacts/meta_hodge"), help="Output directory.")
    parser.add_argument("--explore", action="store_true", help="Only list discovered assets and exit.")
    parser.add_argument("--export-registry", type=Path, help="Optional JSON file to store the discovered registry.")
    args = parser.parse_args()

    registry = locate_historical_assets()
    if args.explore:
        summary = summarize_registry(registry)
        print(summary)
        if args.export_registry:
            args.export_registry.parent.mkdir(parents=True, exist_ok=True)
            serialized = {k: v.to_json() for k, v in registry.items()}
            args.export_registry.write_text("\n".join(serialized.values()))
            print(f"Exported registry to {args.export_registry}")
        return

    bundle = load_version_model(args.version, registry)
    x = sample_coords(args.samples)
    g = bundle.metric_fn(x)

    library = CandidateLibrary(versions=list(registry))
    candidates = library.collect(x)

    op = HodgeOperator(g)
    solver = HarmonicSolver(op)
    harmonic = solver.solve(candidates.c2, candidates.c3)

    extractor = YukawaExtractor(x, g)
    yukawa = extractor.compute(harmonic["H2"].eigenvectors, harmonic["H3"].eigenvectors)
    paths = extractor.save(yukawa, args.out)
    print(f"Saved Yukawa tensors to {paths}")


if __name__ == "__main__":
    main()
