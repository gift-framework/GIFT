#!/usr/bin/env python3
"""
GIFT Pattern Explorer - Unified CLI

Consolidated interface for all pattern discovery and analysis tools.
Organizes 20+ specialized scripts into logical categories.
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Get scripts directory
SCRIPTS_DIR = Path(__file__).parent / "scripts"


class PatternExplorerCLI:
    """Unified command-line interface for pattern exploration tools."""

    def __init__(self):
        self.categories = self._organize_scripts()

    def _organize_scripts(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """Organize scripts into logical categories.

        Returns:
            Dict mapping category name to list of (script_file, display_name, description)
        """
        return {
            "Zeta Function Analysis": [
                ("refined_zeta_analysis.py", "Refined Zeta Analysis",
                 "Systematic analysis of zeta function patterns"),
                ("zeta_ratio_discovery.py", "Zeta Ratio Discovery",
                 "Discover ratios of zeta values in observables"),
                ("extended_zeta_analysis.py", "Extended Zeta Analysis",
                 "Extended analysis with higher-order zeta functions"),
                ("odd_zeta_systematic_search.py", "Odd Zeta Search",
                 "Systematic search for odd zeta value patterns"),
                ("visualize_zeta_patterns.py", "Visualize Zeta Patterns",
                 "Generate visualizations of zeta function patterns"),
            ],

            "Mathematical Constants": [
                ("golden_ratio_search.py", "Golden Ratio Search",
                 "Search for golden ratio (φ) patterns"),
                ("feigenbaum_analysis.py", "Feigenbaum Analysis",
                 "Analyze Feigenbaum constant patterns"),
            ],

            "Number Theory": [
                ("integer_factorization_search.py", "Integer Factorization",
                 "Search for integer factorization patterns"),
                ("number_theory_search.py", "Number Theory Search",
                 "Comprehensive number theory pattern search"),
                ("binary_modular_search.py", "Binary Modular Search",
                 "Search for binary/modular arithmetic patterns"),
            ],

            "Systematic Exploration": [
                ("systematic_explorer.py", "Systematic Explorer",
                 "General systematic pattern exploration"),
                ("deep_dive_explorer.py", "Deep Dive Explorer",
                 "In-depth exploration of specific patterns"),
                ("extended_pattern_search.py", "Extended Pattern Search",
                 "Extended search across multiple pattern types"),
                ("higher_order_systematic_search.py", "Higher Order Search",
                 "Search for higher-order mathematical patterns"),
                ("quick_explorer.py", "Quick Explorer",
                 "Fast preliminary pattern exploration"),
            ],

            "Validation & Statistics": [
                ("comprehensive_validator.py", "Comprehensive Validator",
                 "Validate all discovered patterns"),
                ("statistical_validation.py", "Statistical Validation",
                 "Statistical significance testing"),
                ("statistical_significance_analyzer.py", "Significance Analyzer",
                 "Analyze statistical significance of patterns"),
            ],

            "Output Generation": [
                ("experimental_predictions_generator.py", "Predictions Generator",
                 "Generate experimental prediction reports"),
                ("scientific_paper_generator.py", "Paper Generator",
                 "Generate scientific paper drafts from discoveries"),
            ],
        }

    def print_menu(self):
        """Print the main menu."""
        print("\n" + "="*70)
        print("GIFT Pattern Explorer - Unified CLI")
        print("="*70)
        print("\nAvailable Tools (organized by category):\n")

        idx = 1
        self.script_map = {}

        for category, scripts in self.categories.items():
            print(f"\n{category}:")
            print("-" * 50)
            for script_file, display_name, description in scripts:
                print(f"  [{idx:2d}] {display_name}")
                print(f"       {description}")
                self.script_map[idx] = (script_file, display_name)
                idx += 1

        print("\n" + "="*70)
        print("  [0] Exit")
        print("="*70)

    def run_script(self, script_file: str, display_name: str):
        """Run a selected script.

        Args:
            script_file: Name of the Python script to run
            display_name: Display name for the script
        """
        script_path = SCRIPTS_DIR / script_file

        if not script_path.exists():
            print(f"\n❌ Error: Script not found at {script_path}")
            return

        print(f"\n{'='*70}")
        print(f"Running: {display_name}")
        print(f"Script: {script_file}")
        print(f"{'='*70}\n")

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=SCRIPTS_DIR.parent,
                check=False
            )

            print(f"\n{'='*70}")
            if result.returncode == 0:
                print(f"✓ {display_name} completed successfully")
            else:
                print(f"⚠ {display_name} exited with code {result.returncode}")
            print(f"{'='*70}")

        except KeyboardInterrupt:
            print(f"\n\n⚠ Script interrupted by user")
        except Exception as e:
            print(f"\n❌ Error running script: {e}")

    def run(self):
        """Run the interactive CLI."""
        while True:
            self.print_menu()

            try:
                choice = input("\nSelect a tool (number) or 0 to exit: ").strip()

                if not choice:
                    continue

                choice_num = int(choice)

                if choice_num == 0:
                    print("\nExiting Pattern Explorer CLI. Goodbye!\n")
                    break

                if choice_num in self.script_map:
                    script_file, display_name = self.script_map[choice_num]
                    self.run_script(script_file, display_name)

                    input("\nPress Enter to continue...")
                else:
                    print(f"\n⚠ Invalid choice: {choice_num}")
                    input("Press Enter to continue...")

            except ValueError:
                print(f"\n⚠ Please enter a valid number")
                input("Press Enter to continue...")
            except KeyboardInterrupt:
                print("\n\nExiting Pattern Explorer CLI. Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("Press Enter to continue...")


def main():
    """Main entry point."""
    cli = PatternExplorerCLI()
    cli.run()


if __name__ == "__main__":
    main()
