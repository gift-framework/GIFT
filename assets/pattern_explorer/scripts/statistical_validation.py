#!/usr/bin/env python3
"""
Statistical Validation of GIFT Framework Pattern Discoveries
Implements rigorous statistical testing for all discovered patterns.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import zeta
import re
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Constants for GIFT framework
DELTA_F = 4.669201609102990671853203820466201942498
ALPHA_F = 2.502907875095892822283902873218215786996
PHI = (1 + np.sqrt(5)) / 2
EULER_GAMMA = 0.5772156649015329
FEIGENBAUM_DELTA = 4.669201609102990671853203820466201942498
FEIGENBAUM_ALPHA = 2.502907875095892822283902873218215786996

# Total number of observables tested
N_OBSERVABLES = 37

class StatisticalValidator:
    """Comprehensive statistical validation for pattern discoveries."""

    def __init__(self, n_observables=37):
        self.n_observables = n_observables
        self.n_sample = n_observables  # Sample size for statistical tests
        self.all_patterns = []

    def compute_complexity(self, formula: str) -> int:
        """
        Compute formula complexity as number of constants + operations.

        Args:
            formula: Mathematical formula string

        Returns:
            Complexity score
        """
        # Count mathematical constants
        constants = ['ζ', 'zeta', 'delta_F', 'alpha_F', 'phi', 'π', 'pi',
                    'gamma', 'ln', 'log', 'sqrt', 'exp', 'sin', 'cos',
                    'tan', 'arctan', 'Weyl', 'rank', 'M2', 'M3', 'M5',
                    'b2', 'b3', 'tau', 'xi', 'H0_CMB']

        # Count operations
        operations = ['+', '-', '*', '/', '^', '×', '÷']

        complexity = 0

        # Count constants
        for const in constants:
            complexity += formula.count(const)

        # Count operations
        for op in operations:
            complexity += formula.count(op)

        # Count numbers (excluding those in zeta functions)
        # Remove zeta functions first
        temp_formula = re.sub(r'ζ\(\d+\)', '', formula)
        temp_formula = re.sub(r'zeta\(\d+\)', '', temp_formula)
        numbers = re.findall(r'\d+\.?\d*', temp_formula)
        complexity += len(numbers)

        # Minimum complexity of 1
        return max(1, complexity)

    def compute_likelihood(self, predicted: float, experimental: float,
                          uncertainty: float = None) -> float:
        """
        Compute likelihood assuming Gaussian error model.

        Args:
            predicted: Predicted value from formula
            experimental: Experimental/observed value
            uncertainty: Experimental uncertainty (if known)

        Returns:
            Log-likelihood value
        """
        # Use uncertainty if provided, otherwise estimate from deviation
        if uncertainty is None or uncertainty == 0:
            # Estimate uncertainty as 1% of experimental value
            uncertainty = abs(experimental) * 0.01 if experimental != 0 else 0.01

        # Gaussian log-likelihood
        residual = predicted - experimental
        log_likelihood = -0.5 * (np.log(2 * np.pi * uncertainty**2) +
                                 (residual / uncertainty)**2)

        return log_likelihood

    def compute_bic(self, log_likelihood: float, n_params: int,
                    n_samples: int) -> float:
        """
        Compute Bayesian Information Criterion.

        BIC = k×ln(n) - 2×ln(L)
        where k = number of parameters, n = sample size, L = likelihood

        Lower BIC indicates better model.
        """
        return n_params * np.log(n_samples) - 2 * log_likelihood

    def compute_aic(self, log_likelihood: float, n_params: int) -> float:
        """
        Compute Akaike Information Criterion.

        AIC = 2k - 2×ln(L)
        where k = number of parameters, L = likelihood

        Lower AIC indicates better model.
        """
        return 2 * n_params - 2 * log_likelihood

    def compute_adjusted_r2(self, r2: float, n_samples: int,
                           n_params: int) -> float:
        """
        Compute adjusted R² accounting for number of parameters.

        R²_adj = 1 - (1-R²)×(n-1)/(n-k-1)
        """
        if n_samples <= n_params + 1:
            return 0.0

        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_params - 1)
        return max(0.0, adj_r2)

    def compute_r2_from_deviation(self, deviation_pct: float) -> float:
        """
        Estimate R² from deviation percentage.

        R² ≈ 1 - (deviation/100)²
        """
        deviation_fraction = deviation_pct / 100.0
        r2 = 1 - deviation_fraction**2
        return max(0.0, min(1.0, r2))

    def bonferroni_correction(self, p_value: float, n_tests: int) -> float:
        """
        Apply Bonferroni correction for multiple hypothesis testing.

        Corrected p-value = min(1, p_value × n_tests)
        """
        return min(1.0, p_value * n_tests)

    def compute_p_value(self, deviation_pct: float,
                       uncertainty_pct: float = 1.0) -> float:
        """
        Compute p-value for pattern match.

        Assumes Gaussian distribution of experimental errors.
        """
        # Convert deviation to standard deviations (sigma)
        sigma = abs(deviation_pct) / uncertainty_pct

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(sigma))

        return p_value

    def compute_quality_score(self, deviation_pct: float, p_value: float,
                             complexity: int, r2_adj: float) -> float:
        """
        Compute overall quality score for pattern.

        Quality = (precision × significance × fit) / complexity
        where:
            precision = 1 / (1 + deviation)
            significance = 1 - p_value
            fit = R²_adj
        """
        precision = 1.0 / (1.0 + abs(deviation_pct))
        significance = 1.0 - p_value
        fit = max(0.0, r2_adj)

        quality = (precision * significance * (1 + fit)) / np.log1p(complexity)

        return quality

    def estimate_n_tests(self, formula: str) -> int:
        """
        Estimate number of independent tests performed to find pattern.

        This accounts for the look-elsewhere effect.
        """
        # Base tests: all observables
        n_tests = self.n_observables

        # If formula contains zeta functions, multiply by number of zeta values tested
        if 'ζ' in formula or 'zeta' in formula:
            # Estimate ~10 different zeta values tested per observable
            n_tests *= 10

        # If formula contains ratios or products, additional combinations
        if '/' in formula:
            n_tests *= 2
        if '×' in formula or '*' in formula:
            n_tests *= 2

        return n_tests

    def validate_pattern(self, observable: str, formula: str,
                        predicted: float, experimental: float,
                        deviation_pct: float,
                        uncertainty: float = None) -> Dict:
        """
        Perform complete statistical validation of a single pattern.

        Returns dictionary with all statistical metrics.
        """
        # Compute complexity
        complexity = self.compute_complexity(formula)

        # Estimate number of parameters (use complexity as proxy)
        n_params = max(1, complexity // 2)

        # Compute likelihood
        if uncertainty is None:
            # Estimate from experimental value
            uncertainty = abs(experimental) * 0.01 if experimental != 0 else 0.01

        log_likelihood = self.compute_likelihood(predicted, experimental, uncertainty)

        # Compute information criteria
        bic = self.compute_bic(log_likelihood, n_params, self.n_sample)
        aic = self.compute_aic(log_likelihood, n_params)

        # Compute R² metrics
        r2 = self.compute_r2_from_deviation(deviation_pct)
        r2_adj = self.compute_adjusted_r2(r2, self.n_sample, n_params)

        # Compute p-value and apply correction
        uncertainty_pct = (uncertainty / abs(experimental) * 100) if experimental != 0 else 1.0
        p_value_raw = self.compute_p_value(deviation_pct, uncertainty_pct)

        # Estimate number of tests for Bonferroni correction
        n_tests = self.estimate_n_tests(formula)
        p_value_corrected = self.bonferroni_correction(p_value_raw, n_tests)

        # Compute overall quality score
        quality = self.compute_quality_score(deviation_pct, p_value_corrected,
                                            complexity, r2_adj)

        # Determine significance level
        if p_value_corrected < 0.001:
            significance = "highly_significant"
        elif p_value_corrected < 0.01:
            significance = "significant"
        elif p_value_corrected < 0.05:
            significance = "marginally_significant"
        else:
            significance = "not_significant"

        return {
            'observable': observable,
            'formula': formula,
            'predicted': predicted,
            'experimental': experimental,
            'deviation_pct': deviation_pct,
            'uncertainty': uncertainty,
            'complexity': complexity,
            'n_params': n_params,
            'log_likelihood': log_likelihood,
            'bic': bic,
            'aic': aic,
            'r2': r2,
            'r2_adj': r2_adj,
            'p_value_raw': p_value_raw,
            'n_tests': n_tests,
            'p_value_corrected': p_value_corrected,
            'quality_score': quality,
            'significance': significance
        }

    def load_pattern_file(self, filepath: Path, source: str) -> List[Dict]:
        """Load and parse a pattern CSV file."""
        patterns = []

        try:
            df = pd.read_csv(filepath)

            # Handle different CSV formats
            for idx, row in df.iterrows():
                try:
                    # Extract fields (different files have different column names)
                    observable = None
                    formula = None
                    predicted = None
                    experimental = None
                    deviation = None
                    uncertainty = None

                    # Try different column name variations
                    if 'observable' in df.columns:
                        observable = row['observable']
                    elif 'Observable' in df.columns:
                        observable = row['Observable']

                    if 'formula' in df.columns:
                        formula = row['formula']
                    elif 'Formula' in df.columns:
                        formula = row['Formula']
                    elif 'best_ratio' in df.columns:
                        formula = row.get('formula', row['best_ratio'])
                    elif 'Pattern' in df.columns:
                        formula = row['Pattern']

                    if 'predicted' in df.columns:
                        predicted = float(row['predicted'])
                    elif 'Predicted' in df.columns:
                        predicted = float(row['Predicted'])
                    elif 'predicted_value' in df.columns:
                        predicted = float(row['predicted_value'])
                    elif 'GIFT_Value' in df.columns:
                        predicted = float(row['GIFT_Value'])

                    if 'experimental' in df.columns:
                        experimental = float(row['experimental'])
                    elif 'Experimental' in df.columns:
                        experimental = float(row['Experimental'])
                    elif 'experimental_value' in df.columns:
                        experimental = float(row['experimental_value'])
                    elif 'Target_Value' in df.columns:
                        experimental = float(row['Target_Value'])

                    if 'deviation_%' in df.columns:
                        deviation = abs(float(row['deviation_%']))
                    elif 'Deviation_%' in df.columns:
                        deviation = abs(float(row['Deviation_%']))
                    elif 'deviation_pct' in df.columns:
                        deviation = abs(float(row['deviation_pct']))
                    elif 'Deviation_Percent' in df.columns:
                        deviation = abs(float(row['Deviation_Percent']))

                    if 'uncertainty' in df.columns:
                        uncertainty = float(row['uncertainty'])
                    elif 'Uncertainty' in df.columns:
                        uncertainty = float(row['Uncertainty'])

                    # Skip if essential fields are missing
                    if observable is None or formula is None:
                        continue
                    if predicted is None or experimental is None:
                        continue

                    # Compute deviation if not provided
                    if deviation is None and predicted is not None and experimental is not None:
                        if experimental != 0:
                            deviation = abs((predicted - experimental) / experimental * 100)
                        else:
                            deviation = abs(predicted - experimental) * 100

                    patterns.append({
                        'observable': str(observable),
                        'formula': str(formula),
                        'predicted': predicted,
                        'experimental': experimental,
                        'deviation_pct': deviation,
                        'uncertainty': uncertainty,
                        'source': source
                    })

                except (ValueError, TypeError, KeyError) as e:
                    # Skip rows with parsing errors
                    continue

        except Exception as e:
            print(f"Error loading {filepath}: {e}")

        return patterns

    def load_all_patterns(self, base_dir: Path) -> pd.DataFrame:
        """Load all pattern discovery CSV files."""
        all_patterns = []

        # Define pattern files to load
        pattern_files = [
            ('odd_zeta_discoveries.csv', 'odd_zeta'),
            ('odd_zeta_discoveries_extended.csv', 'odd_zeta_extended'),
            ('zeta_ratio_matches.csv', 'zeta_ratio'),
            ('feigenbaum_matches.csv', 'feigenbaum'),
            ('refined_zeta_patterns.csv', 'refined_zeta'),
            ('extended_zeta_patterns.csv', 'extended_zeta'),
            ('ODD_ZETA_RANKED_SUMMARY.csv', 'ranked_summary'),
            ('assets/pattern_explorer/validation_results/discovered_patterns.csv',
             'validation_patterns')
        ]

        for filename, source in pattern_files:
            filepath = base_dir / filename
            if filepath.exists():
                patterns = self.load_pattern_file(filepath, source)
                all_patterns.extend(patterns)
                print(f"Loaded {len(patterns)} patterns from {filename}")

        if not all_patterns:
            print("Warning: No patterns loaded!")
            return pd.DataFrame()

        return pd.DataFrame(all_patterns)

    def validate_all_patterns(self, patterns_df: pd.DataFrame) -> pd.DataFrame:
        """Validate all patterns and compute statistical metrics."""
        validated = []

        for idx, pattern in patterns_df.iterrows():
            result = self.validate_pattern(
                observable=pattern['observable'],
                formula=pattern['formula'],
                predicted=pattern['predicted'],
                experimental=pattern['experimental'],
                deviation_pct=pattern['deviation_pct'],
                uncertainty=pattern.get('uncertainty')
            )
            result['source'] = pattern['source']
            validated.append(result)

        return pd.DataFrame(validated)

    def rank_patterns(self, validated_df: pd.DataFrame,
                     top_n: int = 100) -> pd.DataFrame:
        """Rank patterns by quality score and return top N."""
        # Sort by quality score (descending)
        ranked = validated_df.sort_values('quality_score', ascending=False)

        # Add rank column
        ranked = ranked.reset_index(drop=True)
        ranked.insert(0, 'rank', range(1, len(ranked) + 1))

        return ranked.head(top_n)

    def generate_statistics_summary(self, validated_df: pd.DataFrame) -> Dict:
        """Generate summary statistics for validation report."""
        total_patterns = len(validated_df)

        # Count by significance level
        highly_sig = len(validated_df[validated_df['significance'] == 'highly_significant'])
        significant = len(validated_df[validated_df['significance'] == 'significant'])
        marginal = len(validated_df[validated_df['significance'] == 'marginally_significant'])
        not_sig = len(validated_df[validated_df['significance'] == 'not_significant'])

        # Statistical thresholds
        passing_patterns = len(validated_df[validated_df['p_value_corrected'] < 0.05])
        high_quality = len(validated_df[validated_df['quality_score'] > 0.5])

        # Deviation statistics
        mean_deviation = validated_df['deviation_pct'].mean()
        median_deviation = validated_df['deviation_pct'].median()

        # Complexity statistics
        mean_complexity = validated_df['complexity'].mean()
        median_complexity = validated_df['complexity'].median()

        # Quality statistics
        mean_quality = validated_df['quality_score'].mean()
        median_quality = validated_df['quality_score'].median()

        return {
            'total_patterns': total_patterns,
            'highly_significant': highly_sig,
            'significant': significant,
            'marginally_significant': marginal,
            'not_significant': not_sig,
            'passing_threshold': passing_patterns,
            'high_quality': high_quality,
            'mean_deviation': mean_deviation,
            'median_deviation': median_deviation,
            'mean_complexity': mean_complexity,
            'median_complexity': median_complexity,
            'mean_quality': mean_quality,
            'median_quality': median_quality
        }


def main():
    """Main execution function."""
    print("GIFT Framework - Statistical Validation Analysis")
    print("=" * 60)

    # Initialize validator
    validator = StatisticalValidator(n_observables=37)

    # Set base directory relative to this script
    # scripts/ -> pattern_explorer/ -> assets/ -> GIFT/
    base_dir = Path(__file__).resolve().parent.parent.parent.parent

    # Load all patterns
    print("\nLoading pattern discovery files...")
    patterns_df = validator.load_all_patterns(base_dir)

    if patterns_df.empty:
        print("Error: No patterns loaded. Exiting.")
        return

    print(f"Total patterns loaded: {len(patterns_df)}")

    # Remove duplicates (same observable + formula combination)
    print("\nRemoving duplicate patterns...")
    patterns_df = patterns_df.drop_duplicates(subset=['observable', 'formula'],
                                              keep='first')
    print(f"Unique patterns: {len(patterns_df)}")

    # Validate all patterns
    print("\nPerforming statistical validation...")
    validated_df = validator.validate_all_patterns(patterns_df)

    # Generate statistics
    stats_summary = validator.generate_statistics_summary(validated_df)

    print("\nValidation Summary:")
    print(f"  Total patterns validated: {stats_summary['total_patterns']}")
    print(f"  Highly significant (p < 0.001): {stats_summary['highly_significant']}")
    print(f"  Significant (p < 0.01): {stats_summary['significant']}")
    print(f"  Marginally significant (p < 0.05): {stats_summary['marginally_significant']}")
    print(f"  Not significant: {stats_summary['not_significant']}")
    print(f"  Passing statistical threshold: {stats_summary['passing_threshold']}")
    print(f"  High quality (score > 0.5): {stats_summary['high_quality']}")

    # Save validated patterns
    output_file = base_dir / 'validated_patterns_ranked.csv'
    validated_sorted = validated_df.sort_values('quality_score', ascending=False)
    validated_sorted.to_csv(output_file, index=False, float_format='%.10f')
    print(f"\nSaved: {output_file}")

    # Get top 100
    top_100 = validator.rank_patterns(validated_df, top_n=100)

    print(f"\nTop 20 patterns by quality score:")
    for idx, row in top_100.head(20).iterrows():
        print(f"{row['rank']:3d}. {row['observable']:15s} | "
              f"{row['formula']:40s} | "
              f"Quality: {row['quality_score']:.4f} | "
              f"Dev: {row['deviation_pct']:.4f}%")

    return validated_df, top_100, stats_summary


if __name__ == '__main__':
    validated_df, top_100, stats_summary = main()
