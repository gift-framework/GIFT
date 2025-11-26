"""
Entrypoints for the professional visualization package.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

from . import data_sources, helpers
from .figures import registry as figure_registry


def render_suite(
    *,
    figures: Optional[Iterable[str]] = None,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    show: bool = False,
    gift=None,
) -> dict:
    """
    Render one or many figures and return metadata for downstream tooling.
    """
    config = helpers.load_config(config_path)
    helpers.configure_plotly_theme(config)
    output = {}

    available = figure_registry()
    selected = list(figures) if figures else list(available.keys())

    root = Path(output_dir) if output_dir else Path("assets/visualizations/outputs/pro")
    root.mkdir(parents=True, exist_ok=True)

    for key in selected:
        if key not in available:
            raise ValueError(f"Unknown figure '{key}'. Available: {list(available)}")
        render_fn = available[key]
        result = render_fn(output_dir=root, config=config, show=show, gift=gift)
        output[key] = result

    return {"config": config, "outputs": output}







