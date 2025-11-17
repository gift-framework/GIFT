"""
Validation system for GIFT predictions.

Compares GIFT predictions against experimental data and provides
statistical analysis.
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class ValidationResult:
    """
    Results from validating GIFT predictions against experiments.

    Attributes
    ----------
    observables : pd.DataFrame
        Full dataset with predictions and experimental values
    mean_deviation : float
        Mean percent deviation across all observables
    median_deviation : float
        Median percent deviation
    max_deviation : float
        Maximum deviation
    n_observables : int
        Total number of observables validated
    n_exact : int
        Number with deviation < 0.01% (essentially exact)
    n_exceptional : int
        Number with deviation < 0.1%
    n_excellent : int
        Number with deviation < 0.5%
    all_under_1_percent : bool
        True if all deviations < 1%
    chi_squared_dof : float
        χ²/dof statistic
    """

    observables: pd.DataFrame
    mean_deviation: float
    median_deviation: float
    max_deviation: float
    n_observables: int
    n_exact: int
    n_exceptional: int
    n_excellent: int
    all_under_1_percent: bool
    chi_squared_dof: float

    def summary(self) -> str:
        """
        Generate human-readable validation summary.

        Returns
        -------
        str
            Formatted summary text
        """
        status = "✓ VALIDATED" if self.all_under_1_percent else "⚠ REVIEW NEEDED"

        return f"""
╔═══════════════════════════════════════════════════════════╗
║          GIFT Framework Validation Summary                ║
╚═══════════════════════════════════════════════════════════╝

Total Observables: {self.n_observables}

Precision Metrics:
  Mean deviation:   {self.mean_deviation:.4f}%
  Median deviation: {self.median_deviation:.4f}%
  Max deviation:    {self.max_deviation:.4f}%

Distribution:
  Exact       (<0.01%): {self.n_exact:3d} ({self.n_exact/self.n_observables*100:5.1f}%)
  Exceptional (<0.1%):  {self.n_exceptional:3d} ({self.n_exceptional/self.n_observables*100:5.1f}%)
  Excellent   (<0.5%):  {self.n_excellent:3d} ({self.n_excellent/self.n_observables*100:5.1f}%)
  All under 1%:         {self.all_under_1_percent}

Statistical Test:
  χ²/dof: {self.chi_squared_dof:.2f}

Status: {status}
        """

    def plot(self, filename: Optional[str] = None):
        """
        Plot validation results.

        Parameters
        ----------
        filename : str, optional
            Save to file if provided
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Histogram of deviations
            axes[0].hist(
                self.observables["deviation_%"], bins=20, color="steelblue", alpha=0.7
            )
            axes[0].axvline(
                self.mean_deviation,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {self.mean_deviation:.3f}%",
            )
            axes[0].axvline(
                self.median_deviation,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"Median: {self.median_deviation:.3f}%",
            )
            axes[0].set_xlabel("Deviation (%)", fontsize=12)
            axes[0].set_ylabel("Count", fontsize=12)
            axes[0].set_title("Distribution of Deviations", fontsize=14, fontweight="bold")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            # Bar chart by sector
            sector_stats = self.observables.groupby("sector")["deviation_%"].agg(
                ["mean", "max", "count"]
            )
            x = range(len(sector_stats))
            axes[1].bar(x, sector_stats["mean"], color="steelblue", alpha=0.7, label="Mean")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(sector_stats.index, rotation=45, ha="right")
            axes[1].set_ylabel("Mean Deviation (%)", fontsize=12)
            axes[1].set_title("Precision by Sector", fontsize=14, fontweight="bold")
            axes[1].grid(alpha=0.3, axis="y")

            plt.tight_layout()

            if filename:
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"Plot saved to {filename}")
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not installed. Install with: pip install matplotlib")


def validate_predictions(gift: "GIFT") -> ValidationResult:
    """
    Validate all GIFT predictions against experimental data.

    Parameters
    ----------
    gift : GIFT
        GIFT framework instance

    Returns
    -------
    ValidationResult
        Validation statistics and results

    Examples
    --------
    >>> gift = GIFT()
    >>> validation = validate_predictions(gift)
    >>> print(validation.summary())
    """
    # Compute all observables
    results = gift.compute_all()

    # Filter out observables without experimental data
    with_exp = results[results["experimental"].notna()].copy()

    if len(with_exp) == 0:
        raise ValueError("No experimental data available for validation")

    # Compute statistics
    deviations = with_exp["deviation_%"].values
    mean_dev = np.mean(deviations)
    median_dev = np.median(deviations)
    max_dev = np.max(deviations)

    # Count by precision level
    n_exact = np.sum(deviations < 0.01)
    n_exceptional = np.sum(deviations < 0.1)
    n_excellent = np.sum(deviations < 0.5)
    all_under_1 = np.all(deviations < 1.0)

    # Compute χ²/dof
    chi2 = 0.0
    for _, row in with_exp.iterrows():
        if pd.notna(row["uncertainty"]) and row["uncertainty"] > 0:
            residual = row["value"] - row["experimental"]
            chi2 += (residual / row["uncertainty"]) ** 2

    dof = len(with_exp)
    chi2_dof = chi2 / dof if dof > 0 else 0.0

    return ValidationResult(
        observables=with_exp,
        mean_deviation=mean_dev,
        median_deviation=median_dev,
        max_deviation=max_dev,
        n_observables=len(with_exp),
        n_exact=int(n_exact),
        n_exceptional=int(n_exceptional),
        n_excellent=int(n_excellent),
        all_under_1_percent=bool(all_under_1),
        chi_squared_dof=chi2_dof,
    )
