"""Validation and metric computation for GIFT v2.2 G2 extraction.

After training, validates:
- det(g) = 65/32 within tolerance
- kappa_T = 1/61 within tolerance
- phi is positive (g positive definite)
- (b2, b3) = (21, 77) via harmonic form analysis

Computes and reports all validation metrics with uncertainties.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path
import json

from .config import GIFTConfig, default_config
from .model import G2VariationalNet, HarmonicFormsNet
from .g2_geometry import MetricFromPhi, G2Positivity
from .training import sample_coordinates, sample_grid


# =============================================================================
# Validation result containers
# =============================================================================

@dataclass
class ValidationResult:
    """Container for single validation metric."""
    name: str
    value: float
    target: float
    tolerance: float
    error: float
    passed: bool
    unit: str = ""

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return (f"{self.name}: {self.value:.6f} "
                f"(target: {self.target:.6f}, error: {self.error:.2%}) [{status}]")


@dataclass
class FullValidationReport:
    """Complete validation report."""
    det_result: ValidationResult
    kappa_result: ValidationResult
    positivity_result: ValidationResult
    b2_result: Optional[ValidationResult] = None
    b3_result: Optional[ValidationResult] = None
    phi_norm_result: Optional[ValidationResult] = None
    overall_passed: bool = False

    def __repr__(self):
        lines = [
            "="*60,
            "GIFT v2.2 VALIDATION REPORT",
            "="*60,
            "",
            "Core Constraints:",
            f"  {self.det_result}",
            f"  {self.kappa_result}",
            f"  {self.positivity_result}",
        ]

        if self.b2_result:
            lines.append(f"  {self.b2_result}")
        if self.b3_result:
            lines.append(f"  {self.b3_result}")
        if self.phi_norm_result:
            lines.append(f"  {self.phi_norm_result}")

        lines.extend([
            "",
            "="*60,
            f"Overall: {'PASSED' if self.overall_passed else 'FAILED'}",
            "="*60,
        ])

        return "\n".join(lines)


# =============================================================================
# Core validation functions
# =============================================================================

class Validator:
    """Validates trained GIFT G2 model."""

    def __init__(self, model: G2VariationalNet,
                 config: GIFTConfig = None,
                 device: str = 'cpu',
                 n_samples: int = 10000):
        self.model = model.to(device)
        self.config = config or default_config
        self.device = device
        self.n_samples = n_samples

        self.metric_extractor = MetricFromPhi()
        self.positivity = G2Positivity()

    def validate_determinant(self) -> ValidationResult:
        """Validate det(g) = 65/32.

        Returns:
            ValidationResult for determinant constraint
        """
        self.model.eval()
        with torch.no_grad():
            x = sample_coordinates(self.n_samples, self.device)
            phi = self.model(x, project_positive=True)
            g = self.metric_extractor(phi)
            det_g = torch.det(g)

        mean_det = det_g.mean().item()
        std_det = det_g.std().item()
        target = self.config.det_g_target
        error = abs(mean_det - target) / target

        return ValidationResult(
            name="det(g)",
            value=mean_det,
            target=target,
            tolerance=self.config.det_tolerance,
            error=error,
            passed=error < self.config.det_tolerance,
            unit=""
        )

    def validate_torsion(self) -> ValidationResult:
        """Validate kappa_T = 1/61.

        Returns:
            ValidationResult for torsion constraint
        """
        self.model.eval()

        def phi_fn(x):
            return self.model(x, project_positive=True)

        with torch.no_grad():
            x = sample_coordinates(self.n_samples, self.device)
            phi = self.model(x, project_positive=True)
            g = self.metric_extractor(phi)

            # Compute torsion via finite differences
            eps = 1e-4
            torsion_sq = torch.zeros(self.n_samples, device=self.device)

            for i in range(7):
                x_plus = x.clone()
                x_plus[:, i] += eps
                x_minus = x.clone()
                x_minus[:, i] -= eps

                phi_plus = phi_fn(x_plus)
                phi_minus = phi_fn(x_minus)

                grad = (phi_plus - phi_minus) / (2 * eps)
                torsion_sq += (grad ** 2).sum(dim=-1)

            torsion = torch.sqrt(torsion_sq / 7)

        mean_torsion = torsion.mean().item()
        target = self.config.kappa_T
        error = abs(mean_torsion - target) / target

        return ValidationResult(
            name="kappa_T",
            value=mean_torsion,
            target=target,
            tolerance=self.config.kappa_tolerance,
            error=error,
            passed=error < self.config.kappa_tolerance,
            unit=""
        )

    def validate_positivity(self) -> ValidationResult:
        """Validate phi is positive (g positive definite).

        Returns:
            ValidationResult for positivity constraint
        """
        self.model.eval()
        with torch.no_grad():
            x = sample_coordinates(self.n_samples, self.device)
            phi = self.model(x, project_positive=True)
            g = self.metric_extractor(phi)
            eigenvalues = torch.linalg.eigvalsh(g)
            min_eig = eigenvalues.min(dim=-1).values

        mean_min_eig = min_eig.mean().item()
        fraction_positive = (min_eig > 0).float().mean().item()

        return ValidationResult(
            name="Positivity",
            value=fraction_positive,
            target=1.0,
            tolerance=0.001,
            error=1.0 - fraction_positive,
            passed=fraction_positive > 0.999,
            unit="(fraction)"
        )

    def validate_phi_norm(self) -> ValidationResult:
        """Validate ||phi||^2 = 7 (G2 identity).

        Returns:
            ValidationResult for phi norm
        """
        self.model.eval()
        with torch.no_grad():
            x = sample_coordinates(self.n_samples, self.device)
            phi = self.model(x, project_positive=True)

        norm_sq = (phi ** 2).sum(dim=-1)
        mean_norm_sq = norm_sq.mean().item()
        target = 7.0
        error = abs(mean_norm_sq - target) / target

        return ValidationResult(
            name="||phi||^2",
            value=mean_norm_sq,
            target=target,
            tolerance=0.01,
            error=error,
            passed=error < 0.01,
            unit=""
        )

    def full_validation(self) -> FullValidationReport:
        """Run full validation suite.

        Returns:
            FullValidationReport with all results
        """
        det_result = self.validate_determinant()
        kappa_result = self.validate_torsion()
        positivity_result = self.validate_positivity()
        phi_norm_result = self.validate_phi_norm()

        # Overall pass requires core constraints
        overall = (det_result.passed and
                   kappa_result.passed and
                   positivity_result.passed)

        return FullValidationReport(
            det_result=det_result,
            kappa_result=kappa_result,
            positivity_result=positivity_result,
            phi_norm_result=phi_norm_result,
            overall_passed=overall
        )


# =============================================================================
# Cohomology validation
# =============================================================================

class CohomologyValidator:
    """Validates cohomology (b2, b3) via harmonic forms."""

    def __init__(self, g2_model: G2VariationalNet,
                 harmonic_model: HarmonicFormsNet,
                 config: GIFTConfig = None,
                 device: str = 'cpu'):
        self.g2_model = g2_model.to(device)
        self.harmonic_model = harmonic_model.to(device)
        self.config = config or default_config
        self.device = device
        self.metric_extractor = MetricFromPhi()

    def compute_gram_matrices(self, n_samples: int = 5000
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Gram matrices for H2 and H3 forms.

        Args:
            n_samples: number of sample points

        Returns:
            G_h2: (21, 21) Gram matrix for H2
            G_h3: (77, 77) Gram matrix for H3
        """
        self.g2_model.eval()
        self.harmonic_model.eval()

        with torch.no_grad():
            x = sample_coordinates(n_samples, self.device)
            phi = self.g2_model(x, project_positive=True)
            metric = self.metric_extractor(phi)

            omega = self.harmonic_model.forward_h2(x)
            Phi = self.harmonic_model.forward_h3(x)

            # Volume element
            det_g = torch.det(metric)
            vol = torch.sqrt(det_g.abs())

            # H2 Gram matrix
            weighted_omega = omega * vol.unsqueeze(-1).unsqueeze(-1)
            G_h2 = torch.einsum('bic,bjc->ij', weighted_omega, omega) / n_samples

            # H3 Gram matrix
            weighted_Phi = Phi * vol.unsqueeze(-1).unsqueeze(-1)
            G_h3 = torch.einsum('bic,bjc->ij', weighted_Phi, Phi) / n_samples

        return G_h2, G_h3

    def effective_betti_numbers(self, threshold: float = 0.01
                                ) -> Tuple[int, int]:
        """Compute effective Betti numbers from Gram matrices.

        Args:
            threshold: eigenvalue threshold (relative to max)

        Returns:
            b2_eff: effective b2
            b3_eff: effective b3
        """
        G_h2, G_h3 = self.compute_gram_matrices()

        # H2 effective dimension
        eigs_h2 = torch.linalg.eigvalsh(G_h2)
        b2_eff = (eigs_h2 > threshold * eigs_h2.max()).sum().item()

        # H3 effective dimension
        eigs_h3 = torch.linalg.eigvalsh(G_h3)
        b3_eff = (eigs_h3 > threshold * eigs_h3.max()).sum().item()

        return int(b2_eff), int(b3_eff)

    def validate_cohomology(self) -> Tuple[ValidationResult, ValidationResult]:
        """Validate (b2, b3) = (21, 77).

        Returns:
            b2_result, b3_result
        """
        b2_eff, b3_eff = self.effective_betti_numbers()

        b2_result = ValidationResult(
            name="b2",
            value=b2_eff,
            target=self.config.b2_K7,
            tolerance=0,
            error=abs(b2_eff - self.config.b2_K7) / self.config.b2_K7,
            passed=(b2_eff == self.config.b2_K7),
            unit="(integer)"
        )

        b3_result = ValidationResult(
            name="b3",
            value=b3_eff,
            target=self.config.b3_K7,
            tolerance=0,
            error=abs(b3_eff - self.config.b3_K7) / self.config.b3_K7,
            passed=(b3_eff == self.config.b3_K7),
            unit="(integer)"
        )

        return b2_result, b3_result


# =============================================================================
# Stability analysis
# =============================================================================

class StabilityAnalyzer:
    """Analyze stability of the solution under perturbations."""

    def __init__(self, model: G2VariationalNet,
                 config: GIFTConfig = None,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.config = config or default_config
        self.device = device
        self.metric_extractor = MetricFromPhi()

    def perturbation_stability(self, epsilon: float = 0.01,
                               n_perturbations: int = 10) -> Dict:
        """Test stability under random perturbations.

        Args:
            epsilon: perturbation magnitude
            n_perturbations: number of perturbation samples

        Returns:
            stability metrics
        """
        self.model.eval()

        x_base = sample_coordinates(1000, self.device)

        results = {
            'det_variations': [],
            'torsion_variations': [],
            'eigenvalue_variations': []
        }

        with torch.no_grad():
            # Baseline
            phi_base = self.model(x_base, project_positive=True)
            g_base = self.metric_extractor(phi_base)
            det_base = torch.det(g_base).mean()
            eig_base = torch.linalg.eigvalsh(g_base).min(dim=-1).values.mean()

            for _ in range(n_perturbations):
                # Perturbed coordinates
                x_pert = x_base + epsilon * torch.randn_like(x_base)
                phi_pert = self.model(x_pert, project_positive=True)
                g_pert = self.metric_extractor(phi_pert)

                det_pert = torch.det(g_pert).mean()
                eig_pert = torch.linalg.eigvalsh(g_pert).min(dim=-1).values.mean()

                results['det_variations'].append(
                    (det_pert / det_base - 1).abs().item()
                )
                results['eigenvalue_variations'].append(
                    (eig_pert / eig_base - 1).abs().item() if eig_base > 0 else 0
                )

        # Summarize
        return {
            'det_stability': np.mean(results['det_variations']),
            'det_max_variation': np.max(results['det_variations']),
            'eig_stability': np.mean(results['eigenvalue_variations']),
            'eig_max_variation': np.max(results['eigenvalue_variations']),
            'epsilon': epsilon
        }


# =============================================================================
# Report generation
# =============================================================================

def generate_validation_report(model: G2VariationalNet,
                               config: GIFTConfig = None,
                               device: str = 'cpu',
                               save_path: Optional[Path] = None) -> Dict:
    """Generate comprehensive validation report.

    Args:
        model: trained model
        config: GIFT configuration
        device: torch device
        save_path: optional path to save report

    Returns:
        report dictionary
    """
    validator = Validator(model, config, device)
    report = validator.full_validation()

    stability = StabilityAnalyzer(model, config, device)
    stability_results = stability.perturbation_stability()

    print(report)

    print("\nStability Analysis:")
    print(f"  det(g) variation under perturbation: {stability_results['det_stability']:.2%}")
    print(f"  min(eigenvalue) variation: {stability_results['eig_stability']:.2%}")

    result_dict = {
        'det_g': {
            'value': report.det_result.value,
            'target': report.det_result.target,
            'error': report.det_result.error,
            'passed': report.det_result.passed
        },
        'kappa_T': {
            'value': report.kappa_result.value,
            'target': report.kappa_result.target,
            'error': report.kappa_result.error,
            'passed': report.kappa_result.passed
        },
        'positivity': {
            'value': report.positivity_result.value,
            'passed': report.positivity_result.passed
        },
        'stability': stability_results,
        'overall_passed': report.overall_passed
    }

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f"\nReport saved to {save_path}")

    return result_dict
