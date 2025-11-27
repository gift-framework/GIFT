"""
Precision matrix heatmap showing deviations per observable.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import plotly.graph_objects as go

from .. import data_sources
from ..helpers import build_output_paths, export_figure


def render(
    output_dir,
    config: Mapping[str, Any],
    *,
    gift=None,
    show: bool = False,
) -> dict:
    """
    Render the precision matrix heatmap.
    """
    df = data_sources.load_observables(gift=gift)
    matrix = data_sources.prepare_precision_matrix(df)
    palette = config.get("palette", {})

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns,
            y=matrix.index,
            colorscale="Viridis",
            colorbar=dict(title="Deviation (%)"),
        )
    )
    fig.update_layout(
        title="GIFT Precision Matrix",
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=120, r=40, t=80, b=120),
    )

    output_paths = build_output_paths(output_dir, "precision_matrix_pro")
    export_figure(fig, output_paths, config, show)
    return {"figure": fig, "outputs": output_paths}








