"""
Dimensional reduction visualization combining Sankey flow and annotation panel.
"""
from __future__ import annotations

from typing import Any, Mapping

import plotly.graph_objects as go

from .. import data_sources
from ..helpers import build_output_paths, export_figure


def render(output_dir, config: Mapping[str, Any], *, gift=None, show: bool = False) -> dict:
    """
    Render the E8×E8 → K7 → SM dimensional flow figure.
    """
    k7 = data_sources.load_k7_structure()
    palette = config.get("palette", {})
    sector_colors = palette.get("sector_colors", {})

    labels = [
        "E₈ × E₈ (496D)",
        "K₇ Cohomology (99D)",
        "Standard Model (4D)",
    ]
    values = [496, 99, 4]

    sankey = go.Sankey(
        node=dict(
            label=labels,
            color=[
                palette.get("accent", "#8dd3ff"),
                sector_colors.get("Topology", "#f4d35e"),
                sector_colors.get("Gauge", "#7ec8e3"),
            ],
        ),
        link=dict(
            source=[0, 1],
            target=[1, 2],
            value=values[:-1],
            color=[palette.get("accent", "#8dd3ff"), sector_colors.get("Gauge", "#7ec8e3")],
        ),
    )

    fig = go.Figure(data=[sankey])
    fig.update_layout(
        title="Dimensional Reduction Flow",
        font=dict(size=config.get("fonts", {}).get("size_base", 16)),
        margin=dict(l=40, r=40, t=80, b=40),
        annotations=[
            dict(
                text=(
                    f"b₂ = {k7['b2']}, b₃ = {k7['b3']}<br>"
                    f"H* = {k7['h_star']} (Information preservation 496 → 99 → 4)"
                ),
                x=0.5,
                y=-0.15,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color=palette.get("text_secondary", "#9ca5b4")),
            )
        ],
    )

    output_paths = build_output_paths(output_dir, "dimensional_flow_pro")
    export_figure(fig, output_paths, config, show)
    return {"figure": fig, "outputs": output_paths}