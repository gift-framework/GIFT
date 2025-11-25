"""
Photorealistic-inspired (interactive) rendering of the E8 root system.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import plotly.graph_objects as go

from .. import data_sources
from ..helpers import build_output_paths, export_figure


def _principal_components(matrix: np.ndarray, ndim: int = 3) -> np.ndarray:
    """
    Lightweight PCA that does not require scikit-learn at runtime.
    """
    centered = matrix - matrix.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order[:ndim]]
    return centered @ basis


def render(
    output_dir,
    config: Mapping[str, Any],
    *,
    gift=None,
    show: bool = False,
) -> dict:
    """
    Render the E8 root system projection.
    """
    df = data_sources.load_e8_root_system()
    coords = df[[f"x{i}" for i in range(8)]].to_numpy()
    projected = _principal_components(coords, 3)

    palette = config.get("palette", {})
    sector_colors = palette.get("sector_colors", {})
    family_colors = {
        "vector": sector_colors.get("Topology", "#f4d35e"),
        "spinor": palette.get("accent_secondary", "#ff8c82"),
    }
    traces = []
    for family, group in df.assign(x=projected[:, 0], y=projected[:, 1], z=projected[:, 2]).groupby(
        "family"
    ):
        traces.append(
            go.Scatter3d(
                x=group["x"],
                y=group["y"],
                z=group["z"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=family_colors.get(family, palette.get("accent", "#8dd3ff")),
                    opacity=0.95,
                ),
                name=f"{family.title()} family ({len(group)} pts)",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="E₈ Root System – PCA Projection",
        scene=dict(
            xaxis_title="PC₁",
            yaxis_title="PC₂",
            zaxis_title="PC₃",
            bgcolor=palette.get("background", "#050505"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            zaxis=dict(showgrid=False),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    output_paths = build_output_paths(output_dir, "e8_root_system_pro")
    export_figure(fig, output_paths, config, show)
    return {"figure": fig, "outputs": output_paths}

