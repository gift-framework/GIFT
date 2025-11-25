"""Visualization utilities for GIFT predictions."""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from giftpy.core.framework import GIFT


def _render_pro_suite(gift: "GIFT", **kwargs):
    """
    Delegate to the assets.visualizations.pro_package module at runtime.
    """
    pro_pkg = import_module("assets.visualizations.pro_package")
    output = pro_pkg.render_suite(gift=gift, **kwargs)
    return output


def plot_predictions(
    gift: "GIFT",
    kind: str = "all",
    filename: Optional[str] = None,
    show: bool = True,
    **kwargs,
):
    """
    Plot or export GIFT predictions.
    """
    if kind == "pro-suite":
        _render_pro_suite(gift, show=show, **kwargs)
        return

    try:
        import matplotlib.pyplot as plt  # noqa: F401  # pylint: disable=unused-import

        if kind == "all":
            validation = gift.validate(verbose=False)
            validation.plot(filename=filename)
        else:
            print(f"Plot type '{kind}' not yet implemented")

    except ImportError:
        print("Matplotlib not installed. Install with: pip install matplotlib")
