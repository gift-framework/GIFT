"""Configuration and discovery utilities for the meta-hodge pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["VersionInfo", "locate_historical_assets", "DEFAULT_VERSION_PRIORITY", "DEFAULT_SAMPLE_SIZE"]


@dataclass
class VersionInfo:
    """Lightweight descriptor for a historical experiment version."""

    version: str
    notebook_paths: List[Path]
    checkpoint_paths: List[Path]
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "notebooks": [str(p) for p in self.notebook_paths],
                "checkpoints": [str(p) for p in self.checkpoint_paths],
                "notes": self.notes,
            },
            indent=2,
        )


DEFAULT_VERSION_PRIORITY: List[str] = [
    "1.8",
    "1_8",
    "1_7",
    "1_6",
    "1_5",
    "1_4",
    "1_3",
    "1_2",
    "1_1",
    "1.0",
    "0.9",
    "0.8",
    "0.7",
    "0.6",
    "0.5",
    "0.4",
    "0.3",
    "0.2",
    "0.1",
]
DEFAULT_SAMPLE_SIZE: int = 2000


GLOB_NOTEBOOK_PATTERNS = ["**/*K7*.ipynb", "**/*G2*.ipynb", "**/*PINN*.ipynb"]
GLOB_CHECKPOINT_PATTERNS = ["**/*.pt", "**/*.pth", "**/*checkpoint*.ckpt"]


def locate_historical_assets(base_dir: Optional[Path] = None) -> Dict[str, VersionInfo]:
    """Scan the repository for notebooks and checkpoints for each version.

    The function avoids hard-coding fragile paths by globbing within the
    ``G2_ML`` directory. It groups results by the parent directory name, which
    typically encodes the semantic version (e.g., ``1_8`` or ``0.7``).
    """

    base = base_dir or Path(__file__).resolve().parents[1]
    g2_dir = base / "G2_ML"
    registry: Dict[str, VersionInfo] = {}
    if not g2_dir.exists():
        return registry

    for subdir in sorted(p for p in g2_dir.iterdir() if p.is_dir()):
        notebook_paths: List[Path] = []
        checkpoint_paths: List[Path] = []
        for pattern in GLOB_NOTEBOOK_PATTERNS:
            notebook_paths.extend(subdir.glob(pattern))
        for pattern in GLOB_CHECKPOINT_PATTERNS:
            checkpoint_paths.extend(subdir.glob(pattern))

        version_label = subdir.name
        if not notebook_paths and not checkpoint_paths:
            continue

        registry[version_label] = VersionInfo(
            version=version_label,
            notebook_paths=sorted(notebook_paths),
            checkpoint_paths=sorted(checkpoint_paths),
            notes="auto-discovered",
        )

    return registry
