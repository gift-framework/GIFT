"""Export utilities for GIFT predictions."""
from __future__ import annotations

from typing import TYPE_CHECKING
import pandas as pd
from pathlib import Path

if TYPE_CHECKING:
    from giftpy.core.framework import GIFT


def export_predictions(gift: GIFT, filename: str, format: str = "csv", **kwargs):
    """
    Export GIFT predictions to various formats.

    Parameters
    ----------
    gift : GIFT
        GIFT framework instance
    filename : str
        Output filename
    format : str
        Export format: 'csv', 'json', 'latex', 'html', 'excel'
    **kwargs
        Additional arguments for pandas export functions
    """
    results = gift.compute_all()

    if format == "csv":
        results.to_csv(filename, index=False, **kwargs)
    elif format == "json":
        results.to_json(filename, orient="records", indent=2, **kwargs)
    elif format == "latex":
        results.to_latex(filename, index=False, **kwargs)
    elif format == "html":
        results.to_html(filename, index=False, **kwargs)
    elif format == "excel":
        results.to_excel(filename, index=False, **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Exported {len(results)} observables to {filename} ({format})")
