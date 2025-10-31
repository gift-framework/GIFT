from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


IGNORE_DIR_NAMES = {".git", "__pycache__", ".ipynb_checkpoints", ".venv", "venv", "node_modules"}


def discover_files(root: Path, patterns: Iterable[str]) -> List[Path]:
    found: List[Path] = []
    for pat in patterns:
        for p in root.glob(pat):
            if any(part in IGNORE_DIR_NAMES for part in p.parts):
                continue
            if p.is_file():
                found.append(p)
    return sorted(set(found))


