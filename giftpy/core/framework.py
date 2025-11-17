"""
Core GIFT framework implementation.

The GIFT class provides the main interface for computing all Standard Model
predictions from topological geometry.
"""
from typing import Optional, Dict, List
import pandas as pd

from .constants import CONSTANTS, TopologicalConstants


class GIFT:
    """
    Main GIFT framework class.

    Provides unified access to all GIFT predictions across all physics sectors:
    - Gauge sector (α, α_s, sin²θ_W)
    - Neutrino sector (PMNS matrix, oscillations)
    - Quark sector (CKM matrix, mass ratios)
    - Lepton sector (mass ratios, Koide formula)
    - Cosmology (Ω_DE, n_s, H₀)

    The framework derives 43+ dimensionless observables from just 3 topological
    parameters with mean 0.13% precision vs experiments.

    Parameters
    ----------
    constants : TopologicalConstants, optional
        Custom topological constants. If None, uses default CONSTANTS.
    validate_on_init : bool, default False
        If True, validate all predictions against experiments on initialization.

    Attributes
    ----------
    constants : TopologicalConstants
        Topological parameters defining the framework
    gauge : GaugeSector
        Gauge coupling predictions (lazy loaded)
    neutrino : NeutrinoSector
        Neutrino mixing and oscillations (lazy loaded)
    quark : QuarkSector
        Quark masses and CKM matrix (lazy loaded)
    lepton : LeptonSector
        Lepton mass ratios and Koide formula (lazy loaded)
    cosmology : CosmologySector
        Cosmological parameters (lazy loaded)

    Examples
    --------
    Basic usage:

    >>> import giftpy
    >>> gift = giftpy.GIFT()
    >>> alpha_s = gift.gauge.alpha_s()
    >>> print(f"α_s(M_Z) = {alpha_s:.6f}")
    α_s(M_Z) = 0.117900

    Compute all observables:

    >>> results = gift.compute_all()
    >>> print(results[['observable', 'value', 'deviation_%']])

    Validate against experiments:

    >>> validation = gift.validate()
    >>> print(validation.summary())
    Mean deviation: 0.13%
    All <1%: True

    Custom topological constants (for research):

    >>> from giftpy.core.constants import TopologicalConstants
    >>> custom = TopologicalConstants(p2=2, rank_E8=8, Weyl_factor=5)
    >>> gift = giftpy.GIFT(constants=custom)

    Notes
    -----
    All sector modules are lazy-loaded for performance. They are only imported
    and instantiated when first accessed.

    See Also
    --------
    TopologicalConstants : Fundamental geometric parameters
    """

    def __init__(
        self,
        constants: Optional[TopologicalConstants] = None,
        validate_on_init: bool = False,
    ):
        """Initialize GIFT framework."""
        self.constants = constants or CONSTANTS

        # Lazy-loaded sector modules
        self._gauge = None
        self._neutrino = None
        self._quark = None
        self._lepton = None
        self._cosmology = None
        self._topology = None
        self._temporal = None

        # Cache for computed results
        self._cache = {}

        if validate_on_init:
            self.validate()

    # ========== Sector Properties (Lazy Loading) ==========

    @property
    def gauge(self):
        """
        Access gauge sector computations.

        Returns
        -------
        GaugeSector
            Gauge coupling predictions (α, α_s, sin²θ_W)

        Examples
        --------
        >>> gift = GIFT()
        >>> alpha_s = gift.gauge.alpha_s()
        >>> sin2theta_W = gift.gauge.sin2theta_W()
        """
        if self._gauge is None:
            from ..observables.gauge import GaugeSector

            self._gauge = GaugeSector(self.constants)
        return self._gauge

    @property
    def neutrino(self):
        """
        Access neutrino sector computations.

        Returns
        -------
        NeutrinoSector
            Neutrino mixing angles, CP phase, PMNS matrix

        Examples
        --------
        >>> gift = GIFT()
        >>> theta_12 = gift.neutrino.theta_12()  # Solar angle
        >>> delta_CP = gift.neutrino.delta_CP()  # CP violation
        >>> PMNS = gift.neutrino.PMNS_matrix()   # Full mixing matrix
        """
        if self._neutrino is None:
            from ..observables.neutrino import NeutrinoSector

            self._neutrino = NeutrinoSector(self.constants)
        return self._neutrino

    @property
    def quark(self):
        """
        Access quark sector computations.

        Returns
        -------
        QuarkSector
            Quark mass ratios and CKM matrix elements

        Examples
        --------
        >>> gift = GIFT()
        >>> m_s_m_d = gift.quark.m_s_m_d()  # Strange/down mass ratio
        >>> V_us = gift.quark.V_us()        # CKM element
        """
        if self._quark is None:
            from ..observables.quark import QuarkSector

            self._quark = QuarkSector(self.constants)
        return self._quark

    @property
    def lepton(self):
        """
        Access lepton sector computations.

        Returns
        -------
        LeptonSector
            Lepton mass ratios and Koide formula

        Examples
        --------
        >>> gift = GIFT()
        >>> m_mu_m_e = gift.lepton.m_mu_m_e()    # Muon/electron ratio
        >>> m_tau_m_e = gift.lepton.m_tau_m_e()  # Tau/electron ratio
        >>> Q_Koide = gift.lepton.Q_Koide()      # Koide parameter (2/3 exact)
        """
        if self._lepton is None:
            from ..observables.lepton import LeptonSector

            self._lepton = LeptonSector(self.constants)
        return self._lepton

    @property
    def cosmology(self):
        """
        Access cosmology computations.

        Returns
        -------
        CosmologySector
            Cosmological parameters (dark energy, spectral index, etc.)

        Examples
        --------
        >>> gift = GIFT()
        >>> Omega_DE = gift.cosmology.Omega_DE()  # Dark energy density
        >>> n_s = gift.cosmology.n_s()            # Scalar spectral index
        """
        if self._cosmology is None:
            from ..observables.cosmology import CosmologySector

            self._cosmology = CosmologySector(self.constants)
        return self._cosmology

    @property
    def topology(self):
        """
        Access low-level topology computations (advanced).

        Returns
        -------
        TopologyModule
            E₈ algebra, K₇ manifold structure, cohomology

        Examples
        --------
        >>> gift = GIFT()
        >>> e8 = gift.topology.E8()
        >>> roots = e8.root_system()  # 240 roots of E₈
        """
        if self._topology is None:
            from ..topology import TopologyModule

            self._topology = TopologyModule(self.constants)
        return self._topology

    @property
    def temporal(self):
        """
        Access temporal framework (advanced).

        Returns
        -------
        TemporalModule
            τ-temporal framework for dimensional observables

        Examples
        --------
        >>> gift = GIFT()
        >>> tau_framework = gift.temporal.TauFramework()
        >>> clusters = tau_framework.cluster_observables()
        """
        if self._temporal is None:
            from ..temporal import TemporalModule

            self._temporal = TemporalModule(self.constants)
        return self._temporal

    # ========== Main Interface Methods ==========

    def compute_all(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Compute all GIFT observables across all sectors.

        Parameters
        ----------
        use_cache : bool, default True
            If True, use cached results if available

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - observable: Name of observable
            - value: GIFT prediction
            - unit: Physical unit
            - experimental: Experimental value (if available)
            - uncertainty: Experimental uncertainty
            - deviation_%: Percent deviation from experiment
            - sector: Physics sector
            - status: Classification (PROVEN, DERIVED, etc.)

        Examples
        --------
        >>> gift = GIFT()
        >>> results = gift.compute_all()
        >>> print(results)
        >>> print(f"Mean deviation: {results['deviation_%'].mean():.3f}%")
        """
        if use_cache and "all_observables" in self._cache:
            return self._cache["all_observables"].copy()

        results = []

        # Gauge sector
        results.extend(self.gauge.compute_all())

        # Neutrino sector
        results.extend(self.neutrino.compute_all())

        # Quark sector
        results.extend(self.quark.compute_all())

        # Lepton sector
        results.extend(self.lepton.compute_all())

        # Cosmology
        results.extend(self.cosmology.compute_all())

        df = pd.DataFrame(results)

        if use_cache:
            self._cache["all_observables"] = df.copy()

        return df

    def validate(self, verbose: bool = True) -> "ValidationResult":
        """
        Validate GIFT predictions against experimental data.

        Parameters
        ----------
        verbose : bool, default True
            If True, print validation summary

        Returns
        -------
        ValidationResult
            Validation summary with statistics:
            - mean_deviation: Mean percent deviation
            - median_deviation: Median percent deviation
            - max_deviation: Maximum deviation
            - n_exact: Number of exact predictions (<0.01%)
            - all_under_1_percent: Boolean, all deviations <1%

        Examples
        --------
        >>> gift = GIFT()
        >>> validation = gift.validate()
        >>> print(validation.summary())

        >>> if validation.all_under_1_percent:
        ...     print("All predictions within 1%!")
        """
        from .validation import validate_predictions

        result = validate_predictions(self)

        if verbose:
            print(result.summary())

        return result

    def export(self, filename: str, format: str = "csv", **kwargs):
        """
        Export predictions to file.

        Parameters
        ----------
        filename : str
            Output filename
        format : str, default 'csv'
            Export format: 'csv', 'json', 'latex', 'html', 'excel'
        **kwargs
            Additional arguments passed to export function

        Examples
        --------
        >>> gift = GIFT()
        >>> gift.export('predictions.csv', format='csv')
        >>> gift.export('predictions.tex', format='latex')
        >>> gift.export('predictions.xlsx', format='excel')
        """
        from ..tools.export import export_predictions

        export_predictions(self, filename, format, **kwargs)

    def plot(
        self,
        kind: str = "all",
        filename: Optional[str] = None,
        show: bool = True,
        **kwargs,
    ):
        """
        Plot GIFT predictions.

        Parameters
        ----------
        kind : str, default 'all'
            Type of plot:
            - 'all': All predictions vs experiments
            - 'deviations': Histogram of deviations
            - 'by_sector': Grouped by physics sector
            - 'temporal': Temporal clustering analysis
        filename : str, optional
            Save to file (e.g., 'plot.png', 'plot.pdf')
        show : bool, default True
            If True, display plot interactively
        **kwargs
            Additional arguments for plotting

        Examples
        --------
        >>> gift = GIFT()
        >>> gift.plot(kind='all', filename='predictions.png')
        >>> gift.plot(kind='deviations')
        >>> gift.plot(kind='by_sector')
        """
        from ..tools.visualization import plot_predictions

        plot_predictions(self, kind=kind, filename=filename, show=show, **kwargs)

    def compare(self, other: "GIFT") -> pd.DataFrame:
        """
        Compare predictions with another GIFT instance.

        Useful for testing modifications to topological constants.

        Parameters
        ----------
        other : GIFT
            Another GIFT instance to compare against

        Returns
        -------
        pd.DataFrame
            Comparison showing differences

        Examples
        --------
        >>> gift1 = GIFT()
        >>> custom_constants = TopologicalConstants(...)
        >>> gift2 = GIFT(constants=custom_constants)
        >>> diff = gift1.compare(gift2)
        """
        df1 = self.compute_all()
        df2 = other.compute_all()

        comparison = df1[["observable", "value"]].copy()
        comparison["value_other"] = df2["value"]
        comparison["difference"] = comparison["value"] - comparison["value_other"]
        comparison["percent_change"] = (
            comparison["difference"] / comparison["value"] * 100
        )

        return comparison

    def clear_cache(self):
        """
        Clear cached computation results.

        Call this if you modify constants and want to recompute.
        """
        self._cache = {}
        # Also clear sector-level caches
        if self._gauge is not None:
            self._gauge.clear_cache()
        if self._neutrino is not None:
            self._neutrino.clear_cache()
        if self._quark is not None:
            self._quark.clear_cache()
        if self._lepton is not None:
            self._lepton.clear_cache()
        if self._cosmology is not None:
            self._cosmology.clear_cache()

    def __repr__(self) -> str:
        """String representation."""
        return f"GIFT(constants={self.constants})"

    def __str__(self) -> str:
        """Human-readable string."""
        return f"GIFT Framework (v{self.version})"

    @property
    def version(self) -> str:
        """Package version."""
        from .. import __version__

        return __version__

    def info(self) -> str:
        """
        Print framework information.

        Returns
        -------
        str
            Summary of framework parameters and capabilities
        """
        return f"""
GIFT Framework Information
===========================

Version: {self.version}

Topological Parameters:
  b₂(K₇) = {self.constants.b2}
  b₃(K₇) = {self.constants.b3}
  dim(E₈) = {self.constants.dim_E8}

GIFT Parameters:
  β₀ = {self.constants.beta0:.10f}
  ξ = {self.constants.xi:.10f} (DERIVED!)
  N_gen = {self.constants.N_gen} (PROVEN)

Available Sectors:
  ✓ Gauge (α, α_s, sin²θ_W)
  ✓ Neutrino (PMNS matrix, oscillations)
  ✓ Quark (CKM matrix, mass ratios)
  ✓ Lepton (mass ratios, Koide formula)
  ✓ Cosmology (Ω_DE, n_s, H₀)

Quick Start:
  >>> results = gift.compute_all()
  >>> validation = gift.validate()
  >>> gift.plot(kind='all')

Documentation: https://giftpy.readthedocs.io/
        """
