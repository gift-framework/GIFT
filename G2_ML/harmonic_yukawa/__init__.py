"""Harmonic Forms and Yukawa Tensor Pipeline for GIFT v2.2.

This module implements the complete pipeline:
    PINN (phi) -> Metric (g) -> Harmonic Forms (H^2, H^3) -> Yukawa Tensor -> Mass Spectrum

The key mathematical objects:
- phi: Associative 3-form on K7, output of trained PINN
- g: Induced metric from phi
- H^2(K7): 21-dimensional space of harmonic 2-forms
- H^3(K7): 77-dimensional space of harmonic 3-forms
- Y_ijk: Yukawa couplings from triple products

Reference: Joyce (2000), "Compact Manifolds with Special Holonomy"
"""
from __future__ import annotations

from .config import HarmonicConfig, default_harmonic_config
from .hodge_laplacian import HodgeLaplacian, LaplacianResult
from .harmonic_extraction import HarmonicExtractor, HarmonicBasis
from .wedge_product import WedgeProduct, wedge_2_2_3, wedge_3_3_to_6
from .yukawa import YukawaTensor, YukawaResult
from .mass_spectrum import MassSpectrum, FermionMasses
from .pipeline import HarmonicYukawaPipeline, PipelineResult
from .lean_export import LeanExporter, NumericalBound, export_pipeline_to_lean

__version__ = "0.1.0"
__all__ = [
    # Config
    "HarmonicConfig",
    "default_harmonic_config",
    # Hodge theory
    "HodgeLaplacian",
    "LaplacianResult",
    # Harmonic forms
    "HarmonicExtractor",
    "HarmonicBasis",
    # Wedge products
    "WedgeProduct",
    "wedge_2_2_3",
    "wedge_3_3_to_6",
    # Yukawa
    "YukawaTensor",
    "YukawaResult",
    # Masses
    "MassSpectrum",
    "FermionMasses",
    # Pipeline
    "HarmonicYukawaPipeline",
    "PipelineResult",
    # Lean export
    "LeanExporter",
    "NumericalBound",
    "export_pipeline_to_lean",
]
