"""
Shared utilities for the professional visualization toolkit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import plotly.io as pio

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config.json"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load the visualization style/configuration JSON.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_plotly_theme(config: Mapping[str, Any]) -> None:
    """
    Apply baseline theme parameters in Plotly.
    """
    palette = config.get("palette", {})
    font = config.get("fonts", {})
    template = {
        "layout": {
            "paper_bgcolor": palette.get("background", "#050505"),
            "plot_bgcolor": palette.get("background", "#050505"),
            "font": {
                "family": font.get("primary", "IBM Plex Sans"),
                "color": palette.get("text_primary", "#f2f2f2"),
                "size": font.get("size_base", 16),
            },
        }
    }
    pio.templates["gift_pro"] = template
    pio.templates.default = "gift_pro"


def build_output_paths(output_root: Path, stem: str) -> Dict[str, Path]:
    """
    Construct canonical filenames for a figure stem.
    """
    return {
        "html": output_root / f"{stem}.html",
        "png": output_root / f"{stem}.png",
    }


def export_figure(fig, output_paths: Mapping[str, Path], config: Mapping[str, Any], show: bool) -> None:
    """
    Save a Plotly figure to the configured outputs.
    """
    export_cfg = config.get("export", {})
    if "html" in output_paths:
        fig.write_html(str(output_paths["html"]), include_plotlyjs="cdn")
    if "png" in output_paths:
        try:
            fig.write_image(
                str(output_paths["png"]),
                width=export_cfg.get("image_width", 1600),
                height=export_cfg.get("image_height", 900),
                scale=export_cfg.get("image_scale", 2),
            )
        except ValueError as exc:
            # Kaleido may be missing in minimal environments; degrade gracefully.
            print(f"[pro_package] Skipping PNG export: {exc}")
    if show:
        fig.show()

