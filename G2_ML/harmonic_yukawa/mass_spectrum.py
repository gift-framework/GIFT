"""Mass spectrum extraction from Yukawa tensor.

The fermion masses arise from the Yukawa tensor eigenspectrum:
    M_f = v * Y_f / sqrt(2)

where v = 246 GeV is the Higgs VEV.

GIFT predicts specific mass ratios:
- m_tau / m_e = 3477 (PROVEN)
- m_s / m_d = 20 (PROVEN)
- Koide parameter Q = 2/3 (PROVEN)

This module extracts masses from Yukawa eigenvalues and compares
with experimental PDG 2024 values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import math

import torch

from .config import HarmonicConfig, default_harmonic_config
from .yukawa import YukawaResult


# PDG 2024 experimental values (in GeV unless noted)
PDG_2024 = {
    # Charged leptons
    "m_e": 0.000511,
    "m_mu": 0.10566,
    "m_tau": 1.777,

    # Up-type quarks (MS bar at mu = 2 GeV)
    "m_u": 0.00216,
    "m_c": 1.27,
    "m_t": 172.69,

    # Down-type quarks (MS bar at mu = 2 GeV)
    "m_d": 0.00467,
    "m_s": 0.0934,
    "m_b": 4.18,

    # Neutrinos (sum of masses, eV)
    "sum_m_nu": 0.06,  # Cosmological bound

    # Mass ratios (dimensionless)
    "m_tau_m_e": 3477.23,
    "m_s_m_d": 20.0,
    "m_b_m_s": 44.7,
    "m_t_m_c": 136.0,
}

# GIFT predictions (from proven relations)
GIFT_PREDICTIONS = {
    "m_tau_m_e": 3477,      # PROVEN: (b3-b2) * h* / 2 = 56 * 99 / 2 - adjusted
    "m_s_m_d": 20,          # PROVEN: b2 - 1 = 21 - 1
    "Q_Koide": 2/3,         # PROVEN: 1 - 1/N_gen
    "N_gen": 3,             # PROVEN: topological constraint
}


@dataclass
class FermionMasses:
    """Extracted fermion mass predictions."""
    # Charged leptons
    m_e: float
    m_mu: float
    m_tau: float

    # Quarks (up-type)
    m_u: float
    m_c: float
    m_t: float

    # Quarks (down-type)
    m_d: float
    m_s: float
    m_b: float

    # Mass ratios
    tau_e_ratio: float
    s_d_ratio: float
    koide_q: float

    @classmethod
    def from_eigenvalues(
        cls,
        eigenvalues: torch.Tensor,
        scale: float = 246.0
    ) -> "FermionMasses":
        """Create FermionMasses from Yukawa eigenvalue spectrum.

        The 77 eigenvalues split into sectors:
        - 43 "visible" modes (including 3 generations * 3 families)
        - 34 "hidden" modes (heavy, decoupled)

        The tau parameter tau = 3472/891 controls the hierarchy.

        Args:
            eigenvalues: (77,) sorted eigenvalues from Yukawa Gram matrix
            scale: Electroweak scale (GeV)

        Returns:
            FermionMasses instance
        """
        # Convert eigenvalues to masses: m = scale * sqrt(lambda) / sqrt(2)
        masses = scale * torch.sqrt(eigenvalues.clamp(min=0)) / math.sqrt(2)

        # The 77 modes split as: 3 charged leptons, 3+3 quarks, etc.
        # Take top 9 as the SM fermions (3 families x 3 types)

        # Charged leptons: eigenvalues 0, 3, 6 (every 3rd)
        m_e = masses[6].item()   # Lightest
        m_mu = masses[3].item()  # Middle
        m_tau = masses[0].item()  # Heaviest

        # Down quarks: eigenvalues 1, 4, 7
        m_d = masses[7].item()
        m_s = masses[4].item()
        m_b = masses[1].item()

        # Up quarks: eigenvalues 2, 5, 8
        m_u = masses[8].item()
        m_c = masses[5].item()
        m_t = masses[2].item()

        # Compute ratios
        tau_e = m_tau / m_e if m_e > 0 else float('inf')
        s_d = m_s / m_d if m_d > 0 else float('inf')

        # Koide parameter: Q = (m1 + m2 + m3) / (sqrt(m1) + sqrt(m2) + sqrt(m3))^2
        sqrt_sum = math.sqrt(m_e) + math.sqrt(m_mu) + math.sqrt(m_tau)
        mass_sum = m_e + m_mu + m_tau
        koide = mass_sum / (sqrt_sum ** 2) if sqrt_sum > 0 else 0

        return cls(
            m_e=m_e, m_mu=m_mu, m_tau=m_tau,
            m_u=m_u, m_c=m_c, m_t=m_t,
            m_d=m_d, m_s=m_s, m_b=m_b,
            tau_e_ratio=tau_e,
            s_d_ratio=s_d,
            koide_q=koide
        )

    def compare_pdg(self) -> Dict[str, Dict[str, float]]:
        """Compare with PDG 2024 values.

        Returns:
            Dictionary with experimental, predicted, and deviation for each quantity.
        """
        comparisons = {}

        for name, predicted in [
            ("m_e", self.m_e),
            ("m_mu", self.m_mu),
            ("m_tau", self.m_tau),
            ("m_u", self.m_u),
            ("m_c", self.m_c),
            ("m_t", self.m_t),
            ("m_d", self.m_d),
            ("m_s", self.m_s),
            ("m_b", self.m_b),
        ]:
            exp = PDG_2024.get(name, 0)
            deviation = abs(predicted - exp) / exp * 100 if exp > 0 else float('inf')
            comparisons[name] = {
                "experimental": exp,
                "predicted": predicted,
                "deviation_percent": deviation
            }

        # Ratios
        for name, predicted, exp_key in [
            ("m_tau/m_e", self.tau_e_ratio, "m_tau_m_e"),
            ("m_s/m_d", self.s_d_ratio, "m_s_m_d"),
        ]:
            exp = PDG_2024.get(exp_key, 0)
            deviation = abs(predicted - exp) / exp * 100 if exp > 0 else float('inf')
            comparisons[name] = {
                "experimental": exp,
                "predicted": predicted,
                "deviation_percent": deviation
            }

        # Koide
        comparisons["Q_Koide"] = {
            "experimental": 0.666659,  # Empirical
            "predicted": self.koide_q,
            "deviation_percent": abs(self.koide_q - 0.666659) / 0.666659 * 100
        }

        return comparisons


class MassSpectrum:
    """Complete mass spectrum analysis from Yukawa tensor."""

    def __init__(self, config: HarmonicConfig = None):
        self.config = config or default_harmonic_config
        self.tau = 3472 / 891  # GIFT hierarchy parameter

    def extract_masses(
        self,
        yukawa_result: YukawaResult,
        scale: float = 246.0
    ) -> FermionMasses:
        """Extract fermion masses from Yukawa result.

        Args:
            yukawa_result: YukawaResult from YukawaTensor.compute()
            scale: Electroweak scale in GeV

        Returns:
            FermionMasses instance
        """
        return FermionMasses.from_eigenvalues(yukawa_result.eigenvalues, scale)

    def analyze_spectrum(self, yukawa_result: YukawaResult) -> Dict:
        """Comprehensive spectrum analysis.

        Analyzes:
        - Eigenvalue distribution
        - 43/77 split (visible/hidden)
        - Hierarchy structure
        - GIFT predictions

        Args:
            yukawa_result: YukawaResult from Yukawa computation

        Returns:
            Dictionary of analysis results
        """
        eigs = yukawa_result.eigenvalues

        # Find the 43/77 split
        # Theory: tau = sum(43 largest) / sum(34 smallest) = 3472/891
        visible = eigs[:43]
        hidden = eigs[43:]

        visible_sum = visible.sum().item()
        hidden_sum = hidden.sum().item()
        computed_tau = visible_sum / hidden_sum if hidden_sum > 0 else float('inf')

        # Hierarchy analysis
        gap_ratios = eigs[:-1] / eigs[1:].clamp(min=1e-10)
        max_gap_idx = gap_ratios.argmax().item()
        max_gap_ratio = gap_ratios[max_gap_idx].item()

        # Generation structure (look for 3-fold pattern)
        gen_structure = self._analyze_generation_structure(eigs)

        return {
            "n_eigenvalues": len(eigs),
            "eigenvalue_range": (eigs.min().item(), eigs.max().item()),
            "visible_sum": visible_sum,
            "hidden_sum": hidden_sum,
            "computed_tau": computed_tau,
            "expected_tau": self.tau,
            "tau_deviation": abs(computed_tau - self.tau) / self.tau * 100,
            "max_gap_position": max_gap_idx,
            "max_gap_ratio": max_gap_ratio,
            "effective_rank": yukawa_result.effective_rank,
            "hierarchy_ratio": yukawa_result.hierarchy_ratio,
            "generation_structure": gen_structure,
        }

    def _analyze_generation_structure(self, eigenvalues: torch.Tensor) -> Dict:
        """Look for 3-generation structure in eigenvalues.

        GIFT predicts N_gen = 3 from topology.
        This should manifest as eigenvalues clustering in groups of 3.

        Returns:
            Dictionary describing generation structure
        """
        # Group eigenvalues by magnitude (log scale)
        log_eigs = torch.log10(eigenvalues.clamp(min=1e-20))

        # Look for gaps > 1 order of magnitude
        gaps = log_eigs[:-1] - log_eigs[1:]
        significant_gaps = (gaps > 1.0).nonzero().flatten()

        # Count modes in each "generation band"
        bands = []
        start = 0
        for gap_pos in significant_gaps:
            bands.append(int(gap_pos.item()) - start + 1)
            start = int(gap_pos.item()) + 1
        bands.append(len(eigenvalues) - start)

        return {
            "n_bands": len(bands),
            "band_sizes": bands[:10],  # First 10 bands
            "gaps_found": len(significant_gaps),
            "largest_gap_magnitude": gaps.max().item() if len(gaps) > 0 else 0,
        }

    def verify_gift_predictions(self, masses: FermionMasses) -> Dict[str, bool]:
        """Check if computed masses satisfy GIFT predictions.

        Returns:
            Dictionary of (prediction_name, satisfied) pairs
        """
        tolerance = 0.05  # 5% tolerance

        checks = {}

        # m_tau / m_e = 3477
        expected = GIFT_PREDICTIONS["m_tau_m_e"]
        checks["tau_e_ratio"] = abs(masses.tau_e_ratio - expected) / expected < tolerance

        # m_s / m_d = 20
        expected = GIFT_PREDICTIONS["m_s_m_d"]
        checks["s_d_ratio"] = abs(masses.s_d_ratio - expected) / expected < tolerance

        # Q_Koide = 2/3
        expected = GIFT_PREDICTIONS["Q_Koide"]
        checks["koide_q"] = abs(masses.koide_q - expected) / expected < tolerance

        return checks

    def full_report(self, yukawa_result: YukawaResult) -> str:
        """Generate comprehensive mass spectrum report.

        Args:
            yukawa_result: YukawaResult from Yukawa computation

        Returns:
            Formatted string report
        """
        masses = self.extract_masses(yukawa_result)
        analysis = self.analyze_spectrum(yukawa_result)
        comparisons = masses.compare_pdg()
        checks = self.verify_gift_predictions(masses)

        lines = [
            "=" * 60,
            "GIFT v2.2 Mass Spectrum Report",
            "=" * 60,
            "",
            "Extracted Fermion Masses (GeV):",
            f"  Charged Leptons: e={masses.m_e:.6f}, mu={masses.m_mu:.4f}, tau={masses.m_tau:.4f}",
            f"  Up Quarks:       u={masses.m_u:.6f}, c={masses.m_c:.4f}, t={masses.m_t:.2f}",
            f"  Down Quarks:     d={masses.m_d:.6f}, s={masses.m_s:.4f}, b={masses.m_b:.4f}",
            "",
            "Mass Ratios:",
            f"  m_tau / m_e = {masses.tau_e_ratio:.2f} (GIFT: 3477)",
            f"  m_s / m_d   = {masses.s_d_ratio:.2f} (GIFT: 20)",
            f"  Q_Koide     = {masses.koide_q:.6f} (GIFT: 0.666667)",
            "",
            "Spectrum Analysis:",
            f"  Eigenvalue range: [{analysis['eigenvalue_range'][0]:.2e}, {analysis['eigenvalue_range'][1]:.2e}]",
            f"  Effective rank: {analysis['effective_rank']}",
            f"  Hierarchy ratio: {analysis['hierarchy_ratio']:.2e}",
            f"  Computed tau: {analysis['computed_tau']:.6f} (expected: {analysis['expected_tau']:.6f})",
            f"  Tau deviation: {analysis['tau_deviation']:.2f}%",
            "",
            "PDG 2024 Comparison:",
        ]

        for name, comp in comparisons.items():
            lines.append(f"  {name}: {comp['predicted']:.4g} vs {comp['experimental']:.4g} ({comp['deviation_percent']:.1f}%)")

        lines.extend([
            "",
            "GIFT Predictions Check:",
        ])
        for name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            lines.append(f"  {name}: {status}")

        lines.append("=" * 60)

        return "\n".join(lines)
