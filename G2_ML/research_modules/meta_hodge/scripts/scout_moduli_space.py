#!/usr/bin/env python
"""Autonomous ML scout for exploring K7 moduli space.

Instead of refining known boundaries, this script:
1. Launches random probes across the full parameter space
2. Uses ML to learn what regions are "interesting" (stable, weird, boundary)
3. Adaptively focuses on surprising discoveries
4. Maps the global topology, not just local refinements

Philosophy: "Minecraft diamond mining" - systematic random exploration
to find unexpected veins, not just polish known ones.
"""
from __future__ import annotations

import argparse
import json
import sys
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import pickle

import numpy as np
import torch

# Add parent directories to path
script_dir = Path(__file__).resolve().parent
meta_hodge_dir = script_dir.parent
g2_ml_dir = meta_hodge_dir.parent
if str(g2_ml_dir) not in sys.path:
    sys.path.insert(0, str(g2_ml_dir))

from meta_hodge.deformation_explorer import (
    DeformationConfig,
    load_baseline_data,
    explore_one_point,
    BaselineData,
)


@dataclass
class ProbeResult:
    """Result of a single probe into moduli space."""
    sigma: float
    s: float
    alpha: float
    u: float
    v: float

    stable: bool
    det_mean: float
    kappa_T: float
    phi_norm_mean: float

    # Classification
    category: str = "unknown"  # stable, unstable, boundary, anomaly
    surprise_score: float = 0.0  # How unexpected was this result?


@dataclass
class ScoutState:
    """State of the exploration campaign."""
    probes: List[ProbeResult] = field(default_factory=list)
    visited: Set[Tuple[float, float, float]] = field(default_factory=set)

    # Statistics
    n_stable: int = 0
    n_unstable: int = 0
    n_anomalies: int = 0

    # Discovered regions
    stable_regions: List[Dict] = field(default_factory=list)
    anomalies: List[Dict] = field(default_factory=list)

    # ML model state
    model_version: int = 0


class ModuliScout:
    """Autonomous explorer for K7 moduli space."""

    def __init__(
        self,
        baseline: BaselineData,
        config: DeformationConfig,
        bounds: Dict[str, Tuple[float, float]] = None,
    ):
        self.baseline = baseline
        self.config = config

        # Default bounds - focus on promising region but also explore wider
        self.bounds = bounds or {
            "sigma": (0.4, 2.0),
            "s": (0.4, 2.0),
            "alpha": (-0.5, 0.5),
        }

        # Known ridge: u ~ 1 - 1.13|alpha| with tolerance ~0.1
        # We'll use this to guide exploration
        self.ridge_center = lambda alpha: 1.0 - 1.13 * abs(alpha)
        self.ridge_width = 0.15

        self.state = ScoutState()
        self.rng = np.random.default_rng(42)

        # Simple ML model for predicting stability
        self.predictor = None
        self.uncertainty_threshold = 0.3

    def _round_coords(self, sigma: float, s: float, alpha: float) -> Tuple[float, float, float]:
        """Round coordinates for deduplication."""
        return (round(sigma, 2), round(s, 2), round(alpha, 2))

    def _generate_random_probe(self) -> Tuple[float, float, float]:
        """Generate a random probe location."""
        sigma = self.rng.uniform(*self.bounds["sigma"])
        s = self.rng.uniform(*self.bounds["s"])
        alpha = self.rng.uniform(*self.bounds["alpha"])
        return sigma, s, alpha

    def _generate_ridge_probe(self) -> Tuple[float, float, float]:
        """Generate a probe near the known ridge but with variation."""
        # Pick an alpha
        alpha = self.rng.uniform(*self.bounds["alpha"])

        # Target u near the ridge with some deviation
        u_target = self.ridge_center(alpha) + self.rng.normal(0, self.ridge_width)
        u_target = max(0.3, min(2.0, u_target))  # Clamp

        # Generate sigma and s such that sigma*s = u_target
        # Use v = sigma/s as a free parameter
        log_v = self.rng.normal(0, 0.5)  # v around 1 with spread
        v = np.exp(log_v)
        v = max(0.3, min(3.0, v))  # Clamp

        # sigma = sqrt(u*v), s = sqrt(u/v)
        sigma = np.sqrt(u_target * v)
        s = np.sqrt(u_target / v)

        # Clamp to bounds
        sigma = np.clip(sigma, *self.bounds["sigma"])
        s = np.clip(s, *self.bounds["s"])

        return sigma, s, alpha

    def _generate_offridge_probe(self) -> Tuple[float, float, float]:
        """Generate a probe deliberately OFF the ridge to find new ridges."""
        alpha = self.rng.uniform(*self.bounds["alpha"])

        # Target u FAR from the known ridge
        u_ridge = self.ridge_center(alpha)
        offset = self.rng.choice([-1, 1]) * self.rng.uniform(0.3, 0.8)
        u_target = u_ridge + offset
        u_target = max(0.2, min(2.5, u_target))

        # Generate sigma and s
        log_v = self.rng.normal(0, 0.7)
        v = np.exp(log_v)
        v = max(0.2, min(4.0, v))

        sigma = np.sqrt(u_target * v)
        s = np.sqrt(u_target / v)

        sigma = np.clip(sigma, *self.bounds["sigma"])
        s = np.clip(s, *self.bounds["s"])

        return sigma, s, alpha

    def _generate_latin_hypercube(self, n: int) -> List[Tuple[float, float, float]]:
        """Generate Latin Hypercube samples for better coverage."""
        # Simple LHS implementation
        sigma_bins = np.linspace(*self.bounds["sigma"], n + 1)
        s_bins = np.linspace(*self.bounds["s"], n + 1)
        alpha_bins = np.linspace(*self.bounds["alpha"], n + 1)

        # Shuffle bin assignments
        sigma_idx = self.rng.permutation(n)
        s_idx = self.rng.permutation(n)
        alpha_idx = self.rng.permutation(n)

        probes = []
        for i in range(n):
            sigma = self.rng.uniform(sigma_bins[sigma_idx[i]], sigma_bins[sigma_idx[i] + 1])
            s = self.rng.uniform(s_bins[s_idx[i]], s_bins[s_idx[i] + 1])
            alpha = self.rng.uniform(alpha_bins[alpha_idx[i]], alpha_bins[alpha_idx[i] + 1])
            probes.append((sigma, s, alpha))

        return probes

    def _generate_near_anomaly(self, anomaly: Dict, radius: float = 0.2) -> Tuple[float, float, float]:
        """Generate a probe near a known anomaly to investigate."""
        sigma = anomaly["sigma"] + self.rng.uniform(-radius, radius)
        s = anomaly["s"] + self.rng.uniform(-radius, radius)
        alpha = anomaly["alpha"] + self.rng.uniform(-radius, radius)

        # Clamp to bounds
        sigma = np.clip(sigma, *self.bounds["sigma"])
        s = np.clip(s, *self.bounds["s"])
        alpha = np.clip(alpha, *self.bounds["alpha"])

        return sigma, s, alpha

    def _predict_stability(self, sigma: float, s: float, alpha: float) -> Tuple[float, float]:
        """Predict stability probability and uncertainty.

        Returns: (probability_stable, uncertainty)
        """
        if self.predictor is None or len(self.state.probes) < 20:
            return 0.5, 1.0  # No information yet

        try:
            X = np.array([[sigma * s, abs(alpha)]])  # Use (u, |alpha|) features
            proba = self.predictor.predict_proba(X)[0, 1]

            # Estimate uncertainty from distance to training data
            train_X = np.array([[p.u, abs(p.alpha)] for p in self.state.probes])
            distances = np.sqrt(((train_X - X) ** 2).sum(axis=1))
            min_dist = distances.min()
            uncertainty = min(1.0, min_dist / 0.3)  # High uncertainty if far from data

            return proba, uncertainty
        except Exception:
            return 0.5, 1.0

    def _update_predictor(self):
        """Update the ML predictor with new data."""
        if len(self.state.probes) < 20:
            return

        try:
            from sklearn.ensemble import RandomForestClassifier

            X = np.array([[p.u, abs(p.alpha)] for p in self.state.probes])
            y = np.array([1 if p.stable else 0 for p in self.state.probes])

            self.predictor = RandomForestClassifier(n_estimators=50, random_state=42)
            self.predictor.fit(X, y)
            self.state.model_version += 1
        except ImportError:
            pass

    def _classify_result(self, result: ProbeResult) -> str:
        """Classify the probe result."""
        # Get prediction before this probe
        pred_prob, uncertainty = self._predict_stability(result.sigma, result.s, result.alpha)

        # Calculate surprise score
        actual = 1.0 if result.stable else 0.0
        surprise = abs(actual - pred_prob) * (1 - uncertainty)
        result.surprise_score = surprise

        # Check for anomalies
        if result.stable:
            # Stable in unexpected region?
            u = result.u
            expected_u_range = (0.5, 1.1)  # Based on known ridge

            if u < expected_u_range[0] - 0.1 or u > expected_u_range[1] + 0.1:
                return "anomaly"

            # Stable with high |alpha|?
            if abs(result.alpha) > 0.45:
                return "anomaly"

            return "stable"
        else:
            # Unstable where we expected stable?
            if surprise > 0.5:
                return "boundary"
            return "unstable"

    def _detect_new_region(self, result: ProbeResult) -> Optional[Dict]:
        """Check if this probe reveals a new stable region."""
        if not result.stable:
            return None

        # Check distance from known stable regions
        if not self.state.stable_regions:
            return {
                "center": (result.sigma, result.s, result.alpha),
                "u_range": (result.u, result.u),
                "alpha_range": (result.alpha, result.alpha),
                "n_points": 1,
            }

        # Find closest region
        min_dist = float("inf")
        closest_region = None
        for region in self.state.stable_regions:
            center = region["center"]
            dist = np.sqrt(
                (result.sigma - center[0])**2 +
                (result.s - center[1])**2 +
                (result.alpha - center[2])**2
            )
            if dist < min_dist:
                min_dist = dist
                closest_region = region

        # If far from any known region, it's a new discovery!
        if min_dist > 0.5:
            return {
                "center": (result.sigma, result.s, result.alpha),
                "u_range": (result.u, result.u),
                "alpha_range": (result.alpha, result.alpha),
                "n_points": 1,
                "discovery_probe": len(self.state.probes),
            }

        # Update closest region
        if closest_region:
            closest_region["n_points"] += 1
            closest_region["u_range"] = (
                min(closest_region["u_range"][0], result.u),
                max(closest_region["u_range"][1], result.u),
            )
            closest_region["alpha_range"] = (
                min(closest_region["alpha_range"][0], result.alpha),
                max(closest_region["alpha_range"][1], result.alpha),
            )

        return None

    def probe(self, sigma: float, s: float, alpha: float) -> ProbeResult:
        """Execute a single probe and record results."""
        coords = self._round_coords(sigma, s, alpha)

        # Skip if already visited
        if coords in self.state.visited:
            return None

        self.state.visited.add(coords)

        # Execute probe
        result = explore_one_point(self.baseline, sigma, s, alpha, self.config)

        probe_result = ProbeResult(
            sigma=sigma,
            s=s,
            alpha=alpha,
            u=sigma * s,
            v=sigma / s if s != 0 else float("inf"),
            stable=result.stable,
            det_mean=result.det_mean,
            kappa_T=result.kappa_T,
            phi_norm_mean=result.phi_norm_mean,
        )

        # Classify
        probe_result.category = self._classify_result(probe_result)

        # Update statistics
        self.state.probes.append(probe_result)
        if probe_result.stable:
            self.state.n_stable += 1
        else:
            self.state.n_unstable += 1

        if probe_result.category == "anomaly":
            self.state.n_anomalies += 1
            self.state.anomalies.append({
                "sigma": sigma, "s": s, "alpha": alpha,
                "u": probe_result.u, "stable": probe_result.stable,
                "surprise": probe_result.surprise_score,
            })

        # Check for new region
        new_region = self._detect_new_region(probe_result)
        if new_region and new_region not in self.state.stable_regions:
            self.state.stable_regions.append(new_region)

        return probe_result

    def run_campaign(
        self,
        n_probes: int = 100,
        strategy: str = "mixed",
        progress_callback=None,
    ) -> ScoutState:
        """Run an exploration campaign.

        Strategies:
        - "random": Pure random sampling
        - "lhs": Latin Hypercube Sampling for better coverage
        - "adaptive": Focus on uncertain/interesting regions
        - "mixed": Combination of all strategies
        """
        probes_done = 0

        # Initial LHS phase for coverage
        if strategy in ("lhs", "mixed"):
            n_lhs = min(n_probes // 3, 50)
            lhs_probes = self._generate_latin_hypercube(n_lhs)

            for sigma, s, alpha in lhs_probes:
                result = self.probe(sigma, s, alpha)
                probes_done += 1

                if progress_callback and result:
                    progress_callback(probes_done, n_probes, result)

                if probes_done >= n_probes:
                    break

            # Update predictor after initial phase
            self._update_predictor()

        # Main exploration loop
        while probes_done < n_probes:
            # Choose probe strategy with smart mixing
            roll = self.rng.random()

            if strategy == "random":
                sigma, s, alpha = self._generate_random_probe()

            elif strategy == "ridge":
                # Focus on ridge exploration
                sigma, s, alpha = self._generate_ridge_probe()

            elif strategy == "mixed":
                if roll < 0.3:
                    # 30% - explore near known ridge with variation
                    sigma, s, alpha = self._generate_ridge_probe()
                elif roll < 0.5:
                    # 20% - explore OFF the ridge to find new ridges
                    sigma, s, alpha = self._generate_offridge_probe()
                elif roll < 0.7:
                    # 20% - pure random
                    sigma, s, alpha = self._generate_random_probe()
                elif roll < 0.9:
                    # 20% - adaptive (high uncertainty regions)
                    best_uncertainty = 0
                    best_probe = None
                    for _ in range(15):
                        candidate = self._generate_ridge_probe()
                        _, uncertainty = self._predict_stability(*candidate)
                        if uncertainty > best_uncertainty:
                            best_uncertainty = uncertainty
                            best_probe = candidate
                    sigma, s, alpha = best_probe if best_probe else self._generate_ridge_probe()
                else:
                    # 10% - investigate anomalies
                    if self.state.anomalies:
                        anomaly = self.rng.choice(self.state.anomalies)
                        sigma, s, alpha = self._generate_near_anomaly(anomaly)
                    else:
                        sigma, s, alpha = self._generate_offridge_probe()

            elif strategy == "adaptive":
                # Focus on uncertain regions
                best_uncertainty = 0
                best_probe = None
                for _ in range(20):
                    candidate = self._generate_ridge_probe()
                    _, uncertainty = self._predict_stability(*candidate)
                    if uncertainty > best_uncertainty:
                        best_uncertainty = uncertainty
                        best_probe = candidate
                sigma, s, alpha = best_probe if best_probe else self._generate_ridge_probe()

            else:
                sigma, s, alpha = self._generate_random_probe()

            result = self.probe(sigma, s, alpha)
            if result:
                probes_done += 1

                if progress_callback:
                    progress_callback(probes_done, n_probes, result)

            # Periodically update predictor
            if probes_done % 20 == 0:
                self._update_predictor()

        return self.state

    def summarize(self) -> Dict:
        """Generate summary of exploration."""
        if not self.state.probes:
            return {}

        stable_probes = [p for p in self.state.probes if p.stable]

        summary = {
            "total_probes": len(self.state.probes),
            "stable_count": self.state.n_stable,
            "unstable_count": self.state.n_unstable,
            "anomaly_count": self.state.n_anomalies,
            "stability_rate": self.state.n_stable / len(self.state.probes),
            "n_regions": len(self.state.stable_regions),
        }

        if stable_probes:
            summary["stable_u_range"] = (
                min(p.u for p in stable_probes),
                max(p.u for p in stable_probes),
            )
            summary["stable_alpha_range"] = (
                min(p.alpha for p in stable_probes),
                max(p.alpha for p in stable_probes),
            )

        # Surprising discoveries
        surprising = sorted(self.state.probes, key=lambda p: -p.surprise_score)[:10]
        summary["top_surprises"] = [
            {"sigma": p.sigma, "s": p.s, "alpha": p.alpha,
             "stable": p.stable, "surprise": p.surprise_score}
            for p in surprising if p.surprise_score > 0.1
        ]

        return summary


def main():
    parser = argparse.ArgumentParser(description="Scout K7 moduli space")
    parser.add_argument("--n-probes", type=int, default=100, help="Number of probes")
    parser.add_argument("--strategy", choices=["random", "lhs", "adaptive", "mixed", "ridge"],
                       default="mixed", help="Exploration strategy")
    parser.add_argument("--samples", type=int, default=100, help="Coordinate samples per probe")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", "-o", type=Path, help="Output directory")
    parser.add_argument("--resume", type=Path, help="Resume from previous state")

    args = parser.parse_args()

    print("=" * 60)
    print("K7 Moduli Space Scout")
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"Probes: {args.n_probes}")
    print()

    # Load baseline
    print("Loading baseline data...")
    baseline = load_baseline_data("1_6", num_samples=args.samples)
    config = DeformationConfig()

    # CRITICAL: Calibrate config from baseline at (1,1,0)
    # This is what explore_grid does but explore_one_point alone doesn't
    print("Calibrating from baseline (1,1,0)...")
    baseline_result = explore_one_point(baseline, 1.0, 1.0, 0.0, config)

    if baseline_result.phi_norm_mean is not None and not np.isnan(baseline_result.phi_norm_mean):
        config.baseline_phi_norm = baseline_result.phi_norm_mean
        config.phi_norm_target = baseline_result.phi_norm_mean
        print(f"  phi_norm baseline: {baseline_result.phi_norm_mean:.4f}")

    if baseline_result.kappa_T is not None and not np.isnan(baseline_result.kappa_T):
        config.baseline_kappa = baseline_result.kappa_T
        config.kappa_target = baseline_result.kappa_T
        print(f"  kappa_T baseline: {baseline_result.kappa_T:.4f}")

    print(f"  Baseline stable: {baseline_result.stable}")
    print()

    # Initialize scout
    scout = ModuliScout(baseline, config)
    if args.seed:
        scout.rng = np.random.default_rng(args.seed)

    # Resume if provided
    if args.resume and args.resume.exists():
        with open(args.resume / "scout_state.pkl", "rb") as f:
            scout.state = pickle.load(f)
        print(f"Resumed from {len(scout.state.probes)} previous probes")

    # Progress callback
    def progress(i, total, result):
        status = "STABLE" if result.stable else "unstable"
        cat = result.category
        surprise = f"(!{result.surprise_score:.2f})" if result.surprise_score > 0.3 else ""

        bar = "=" * (30 * i // total) + "-" * (30 - 30 * i // total)
        print(f"\r[{bar}] {i}/{total} ({result.sigma:.2f},{result.s:.2f},{result.alpha:+.2f})"
              f" u={result.u:.2f} -> {status} [{cat}] {surprise}    ", end="")

    # Run campaign
    print(f"\nLaunching {args.n_probes} probes...")
    print()

    state = scout.run_campaign(args.n_probes, args.strategy, progress)
    print()
    print()

    # Summary
    summary = scout.summarize()

    print("=" * 60)
    print("EXPLORATION SUMMARY")
    print("=" * 60)
    print()
    print(f"Total probes: {summary['total_probes']}")
    print(f"Stable: {summary['stable_count']} ({100*summary['stability_rate']:.1f}%)")
    print(f"Anomalies: {summary['anomaly_count']}")
    print(f"Distinct regions: {summary['n_regions']}")

    if "stable_u_range" in summary:
        print()
        print(f"Stable u range: [{summary['stable_u_range'][0]:.2f}, {summary['stable_u_range'][1]:.2f}]")
        print(f"Stable alpha range: [{summary['stable_alpha_range'][0]:.2f}, {summary['stable_alpha_range'][1]:.2f}]")

    if summary.get("top_surprises"):
        print()
        print("Surprising discoveries:")
        for s in summary["top_surprises"]:
            status = "STABLE" if s["stable"] else "unstable"
            print(f"  ({s['sigma']:.2f}, {s['s']:.2f}, {s['alpha']:+.2f}) -> {status} (surprise={s['surprise']:.2f})")

    # Report on regions
    if state.stable_regions:
        print()
        print("Stable regions found:")
        for i, region in enumerate(state.stable_regions):
            print(f"  Region {i+1}: center={region['center']}, "
                  f"u=[{region['u_range'][0]:.2f},{region['u_range'][1]:.2f}], "
                  f"n_points={region['n_points']}")

    # Save results
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = meta_hodge_dir / "artifacts" / "scout_campaigns" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save state for resuming
    with open(output_dir / "scout_state.pkl", "wb") as f:
        pickle.dump(state, f)

    # Save human-readable results
    probes_data = [
        {
            "sigma": p.sigma, "s": p.s, "alpha": p.alpha,
            "u": p.u, "v": p.v,
            "stable": p.stable, "category": p.category,
            "det_mean": p.det_mean, "kappa_T": p.kappa_T,
            "surprise": p.surprise_score,
        }
        for p in state.probes
    ]

    with open(output_dir / "probes.json", "w") as f:
        json.dump(probes_data, f, indent=2)

    with open(output_dir / "summary.json", "w") as f:
        # Convert tuples to lists for JSON
        summary_json = {}
        for k, v in summary.items():
            if isinstance(v, tuple):
                summary_json[k] = list(v)
            else:
                summary_json[k] = v
        json.dump(summary_json, f, indent=2)

    print()
    print(f"Results saved to {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
