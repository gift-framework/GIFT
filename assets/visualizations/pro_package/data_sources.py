"""
Data access and scientific context helpers for the professional visualization suite.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from giftpy.core.framework import GIFT
except ImportError:  # pragma: no cover - fallback when giftpy not installed
    GIFT = None  # type: ignore


def _generate_e8_roots() -> pd.DataFrame:
    """
    Construct the 240-point E8 root system in R^8.

    Returns
    -------
    pd.DataFrame
        Columns: x0..x7 cartesian components, `family` (vector family id),
        `norm`, and `type` (short/long).
    """
    roots: list[Tuple[Tuple[float, ...], str]] = []

    # 112 roots of the form (±1, ±1, 0^6) with even permutations.
    base = np.eye(8)
    for i in range(8):
        for j in range(i + 1, 8):
            for signs in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
                vec = np.zeros(8)
                vec[i], vec[j] = signs
                roots.append((tuple(vec), "vector"))

    # 128 roots of the form (±1/2, ..., ±1/2) with even number of minus signs.
    half = 0.5
    for bits in range(256):
        signs = np.array([half if (bits >> k) & 1 else -half for k in range(8)])
        if int((signs < 0).sum()) % 2 == 0:
            roots.append((tuple(signs), "spinor"))

    df = pd.DataFrame(
        [(*vec, family) for vec, family in roots],
        columns=[f"x{i}" for i in range(8)] + ["family"],
    )
    df["norm"] = np.linalg.norm(df[[f"x{i}" for i in range(8)]].to_numpy(), axis=1)
    df["type"] = np.where(np.isclose(df["norm"], np.sqrt(2)), "long", "short")
    return df


@lru_cache(maxsize=1)
def load_e8_root_system() -> pd.DataFrame:
    """
    Return cached DataFrame of the canonical E8 root system.
    """
    return _generate_e8_roots()


def load_k7_structure() -> Dict[str, float]:
    """
    Provide quick access to the K7 manifold cohomology data used in the framework.
    """
    return {
        "b0": 1,
        "b2": 21,
        "b3": 77,
        "h_star": 99,
        "weyl_factor": 5,
        "rank_e8": 8,
    }


def load_observables(gift: Optional["GIFT"] = None, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch the full observable table from a GIFT instance.
    """
    if gift is None:
        if GIFT is None:
            raise RuntimeError(
                "GIFT framework is unavailable; ensure giftpy is installed or importable."
            )
        gift = GIFT()
    return gift.compute_all(use_cache=use_cache)


def summarize_precision_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sector-level statistics required by the dashboard visualizations.
    """
    if "sector" not in df.columns or "deviation_%" not in df.columns:
        raise ValueError("DataFrame must contain 'sector' and 'deviation_%' columns.")

    grouped = (
        df.groupby("sector")["deviation_%"]
        .agg(["mean", "max", "min", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_deviation",
                "max": "max_deviation",
                "min": "min_deviation",
                "count": "observables",
            }
        )
    )
    grouped.sort_values("mean_deviation", inplace=True)
    return grouped


def prepare_precision_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot observables into a heatmap-friendly structure (sector vs observable).
    """
    pivot = df.pivot_table(
        index="sector",
        columns="observable",
        values="deviation_%",
    )
    return pivot.sort_index()


def ensure_output_directory(path: Path) -> Path:
    """
    Guarantee that the target directory exists.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path












