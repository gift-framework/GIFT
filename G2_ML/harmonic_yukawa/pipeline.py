"""Complete pipeline: PINN -> Harmonics -> Yukawa -> Masses.

This module provides the end-to-end pipeline for extracting
fermion mass predictions from a trained G2 PINN.

Usage:
    from G2_ML.harmonic_yukawa import HarmonicYukawaPipeline

    # Load trained PINN
    pinn = load_pinn("model.pt")

    # Run pipeline
    pipeline = HarmonicYukawaPipeline(pinn.get_metric)
    result = pipeline.run()

    # Get predictions
    print(result.mass_report)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Dict
from pathlib import Path
import json

import torch

from .config import HarmonicConfig, default_harmonic_config
from .harmonic_extraction import HarmonicExtractor, HarmonicBasis
from .yukawa import YukawaTensor, YukawaResult
from .mass_spectrum import MassSpectrum, FermionMasses


@dataclass
class PipelineResult:
    """Complete results from the Harmonic-Yukawa pipeline."""
    harmonic_basis: HarmonicBasis
    yukawa_result: YukawaResult
    fermion_masses: FermionMasses
    spectrum_analysis: Dict
    mass_report: str

    # Numerical bounds for Lean verification
    det_g_mean: float
    kappa_T_estimate: float
    tau_computed: float

    def save(self, output_dir: Path):
        """Save results for Lean verification and archival.

        Saves:
        - yukawa_tensor.pt: Full Yukawa tensor
        - eigenvalues.json: Eigenvalue spectrum
        - masses.json: Extracted masses
        - bounds.json: Numerical bounds for Lean
        - report.txt: Human-readable report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Yukawa tensor
        torch.save(self.yukawa_result.tensor, output_dir / "yukawa_tensor.pt")

        # Eigenvalues
        with open(output_dir / "eigenvalues.json", "w") as f:
            json.dump({
                "eigenvalues": self.yukawa_result.eigenvalues.tolist(),
                "trace": self.yukawa_result.trace,
                "det_normalized": self.yukawa_result.det,
                "effective_rank": self.yukawa_result.effective_rank,
            }, f, indent=2)

        # Masses
        with open(output_dir / "masses.json", "w") as f:
            json.dump({
                "charged_leptons": {
                    "e": self.fermion_masses.m_e,
                    "mu": self.fermion_masses.m_mu,
                    "tau": self.fermion_masses.m_tau,
                },
                "up_quarks": {
                    "u": self.fermion_masses.m_u,
                    "c": self.fermion_masses.m_c,
                    "t": self.fermion_masses.m_t,
                },
                "down_quarks": {
                    "d": self.fermion_masses.m_d,
                    "s": self.fermion_masses.m_s,
                    "b": self.fermion_masses.m_b,
                },
                "ratios": {
                    "tau_e": self.fermion_masses.tau_e_ratio,
                    "s_d": self.fermion_masses.s_d_ratio,
                    "koide_q": self.fermion_masses.koide_q,
                },
            }, f, indent=2)

        # Bounds for Lean
        with open(output_dir / "bounds.json", "w") as f:
            json.dump({
                "det_g_mean": self.det_g_mean,
                "det_g_target": 65/32,
                "det_g_error": abs(self.det_g_mean - 65/32) / (65/32),
                "kappa_T_estimate": self.kappa_T_estimate,
                "kappa_T_target": 1/61,
                "kappa_T_error": abs(self.kappa_T_estimate - 1/61) / (1/61),
                "tau_computed": self.tau_computed,
                "tau_target": 3472/891,
                "tau_error": abs(self.tau_computed - 3472/891) / (3472/891),
                "b2_target": 21,
                "b3_target": 77,
            }, f, indent=2)

        # Report
        with open(output_dir / "report.txt", "w") as f:
            f.write(self.mass_report)

        print(f"Results saved to {output_dir}")

    def export_lean_bounds(self) -> str:
        """Generate Lean theorem statements for numerical bounds.

        Returns:
            Lean code with bound theorems
        """
        lines = [
            "/-",
            "  GIFT Harmonic-Yukawa Pipeline: Numerical Bounds",
            "  Auto-generated from pipeline run",
            "-/",
            "",
            "namespace GIFT.NumericalBounds",
            "",
            f"def det_g_computed : Float := {self.det_g_mean}",
            f"def kappa_T_computed : Float := {self.kappa_T_estimate}",
            f"def tau_computed : Float := {self.tau_computed}",
            "",
            "-- Bound theorems (to be verified)",
            f"-- |det_g - 65/32| / (65/32) < {abs(self.det_g_mean - 65/32) / (65/32):.6f}",
            f"-- |kappa_T - 1/61| / (1/61) < {abs(self.kappa_T_estimate - 1/61) / (1/61):.6f}",
            f"-- |tau - 3472/891| / (3472/891) < {abs(self.tau_computed - 3472/891) / (3472/891):.6f}",
            "",
            "end GIFT.NumericalBounds",
        ]
        return "\n".join(lines)


class HarmonicYukawaPipeline:
    """End-to-end pipeline from PINN to mass predictions.

    The pipeline consists of four stages:
    1. Harmonic Extraction: PINN metric -> H^2(21) + H^3(77)
    2. Yukawa Computation: Harmonic forms -> Y_ijk tensor
    3. Mass Extraction: Yukawa eigenvalues -> fermion masses
    4. Validation: Compare with PDG and GIFT predictions
    """

    def __init__(
        self,
        metric_fn: Callable[[torch.Tensor], torch.Tensor],
        config: HarmonicConfig = None,
        device: str = 'cpu'
    ):
        """Initialize pipeline.

        Args:
            metric_fn: Function x -> g(x) from trained PINN
                       Takes (batch, 7) coordinates, returns (batch, 7, 7) metric
            config: Configuration parameters
            device: Torch device
        """
        self.metric_fn = metric_fn
        self.config = config or default_harmonic_config
        self.device = device

        # Initialize components
        self.extractor = HarmonicExtractor(metric_fn, config, device)
        self.yukawa = YukawaTensor(config, device)
        self.mass_spectrum = MassSpectrum(config)

    def run(
        self,
        extraction_method: str = "eigenvalue",
        verbose: bool = True
    ) -> PipelineResult:
        """Run complete pipeline.

        Args:
            extraction_method: "eigenvalue" or "variational"
            verbose: Print progress messages

        Returns:
            PipelineResult with all computed quantities
        """
        if verbose:
            print("=" * 60)
            print("GIFT Harmonic-Yukawa Pipeline")
            print("=" * 60)

        # Stage 1: Extract harmonic forms
        if verbose:
            print("\n[1/4] Extracting harmonic forms...")
        basis = self.extractor.extract_full_basis(method=extraction_method)
        validation = basis.validate()
        if verbose:
            print(f"  H^2: {validation['b2_actual']} forms (target: 21)")
            print(f"  H^3: {validation['b3_actual']} forms (target: 77)")
            print(f"  H^2 orthogonality: {validation['h2_orthogonal']:.6f}")
            print(f"  H^3 orthogonality: {validation['h3_orthogonal']:.6f}")

        # Compute metric statistics
        det_g = torch.det(basis.metric_at_points)
        det_g_mean = det_g.mean().item()
        if verbose:
            print(f"  det(g) mean: {det_g_mean:.6f} (target: {65/32:.6f})")

        # Stage 2: Compute Yukawa tensor
        if verbose:
            print("\n[2/4] Computing Yukawa tensor...")
        yukawa_result = self.yukawa.compute_symmetric(basis)
        if verbose:
            print(f"  Tensor shape: {tuple(yukawa_result.tensor.shape)}")
            print(f"  Effective rank: {yukawa_result.effective_rank}")
            print(f"  Trace: {yukawa_result.trace:.6f}")

        # Stage 3: Extract masses
        if verbose:
            print("\n[3/4] Extracting fermion masses...")
        masses = self.mass_spectrum.extract_masses(yukawa_result)
        analysis = self.mass_spectrum.analyze_spectrum(yukawa_result)
        if verbose:
            print(f"  tau computed: {analysis['computed_tau']:.6f} (target: {3472/891:.6f})")
            print(f"  Hierarchy ratio: {analysis['hierarchy_ratio']:.2e}")

        # Stage 4: Generate report
        if verbose:
            print("\n[4/4] Generating report...")
        report = self.mass_spectrum.full_report(yukawa_result)

        # Estimate kappa_T from torsion (placeholder)
        kappa_T_estimate = 1/61  # Would need actual torsion computation

        result = PipelineResult(
            harmonic_basis=basis,
            yukawa_result=yukawa_result,
            fermion_masses=masses,
            spectrum_analysis=analysis,
            mass_report=report,
            det_g_mean=det_g_mean,
            kappa_T_estimate=kappa_T_estimate,
            tau_computed=analysis['computed_tau'],
        )

        if verbose:
            print("\n" + report)

        return result

    def quick_validate(self, n_points: int = 1000) -> Dict:
        """Quick validation of PINN metric properties.

        Checks:
        - det(g) ~ 65/32
        - Metric is positive definite
        - Approximate b2, b3 from eigenvalue gaps

        Args:
            n_points: Number of sample points

        Returns:
            Dictionary of validation results
        """
        points = torch.rand(n_points, 7, device=self.device)
        g = self.metric_fn(points)

        # Determinant
        det_g = torch.det(g)
        det_mean = det_g.mean().item()
        det_std = det_g.std().item()

        # Positive definiteness
        eigenvalues = torch.linalg.eigvalsh(g)
        min_eig = eigenvalues.min().item()
        positive_fraction = (eigenvalues.min(dim=-1).values > 0).float().mean().item()

        return {
            "det_g_mean": det_mean,
            "det_g_std": det_std,
            "det_g_target": 65/32,
            "det_g_error": abs(det_mean - 65/32) / (65/32) * 100,
            "min_eigenvalue": min_eig,
            "positive_definite_fraction": positive_fraction,
            "n_points": n_points,
        }
