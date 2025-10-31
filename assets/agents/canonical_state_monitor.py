from __future__ import annotations

import json
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base import AgentResult, BaseAgent
from .utils.fs import discover_files
from .utils.markdown import extract_status_tags


STATUS_ORDER = [
    "EXPLORATORY",
    "PHENOMENOLOGICAL",
    "THEORETICAL",
    "DERIVED",
    "TOPOLOGICAL",
    "PROVEN",
]
RANK: Dict[str, int] = {s: i for i, s in enumerate(STATUS_ORDER)}


def best_status(tags: List[str]) -> str | None:
    if not tags:
        return None
    return sorted(tags, key=lambda s: RANK.get(s, -1))[-1]


class CanonicalStateMonitor(BaseAgent):
    name = "canonical"

    def run(self, root: Path) -> AgentResult:
        publications = discover_files(root, ["publications/**/*.md"])
        baseline_path = root / "assets/agents/reports/canonical_state.json"
        baseline: Dict[str, str] = {}
        if baseline_path.exists():
            try:
                baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            except Exception:
                baseline = {}

        current: Dict[str, str] = {}
        upgrades: List[Dict[str, Any]] = []

        for p in publications:
            # Optimize: skip if mtime <= baseline's recorded mtime (would need mtime tracking in baseline)
            # For now, parse all files but could add mtime caching in future
            text = p.read_text(encoding="utf-8", errors="replace")
            tags = extract_status_tags(text)
            status = best_status(tags)
            if status is None:
                continue
            key = str(p)
            current[key] = status
            old = baseline.get(key)
            if old and RANK.get(status, -1) > RANK.get(old, -1):
                upgrades.append({"file": key, "from": old, "to": status})

        # Merge baseline and current (keep best status for each file)
        merged_baseline = baseline.copy()
        for key, status in current.items():
            old = merged_baseline.get(key)
            if not old or RANK.get(status, -1) > RANK.get(old, -1):
                merged_baseline[key] = status

        # Write addon if upgrades found
        if upgrades:
            addons_dir = root / "publications/addons"
            addons_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            hostname = socket.gethostname()
            addon = addons_dir / f"{ts}_{hostname}-canonical-upgrades.md"
            lines = [
                "# Canonical status upgrades\n",
                "\n",
                f"Detected on: {hostname}\n",
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
                "This note records improvements in canonical statuses detected by the agents.\n\n",
                "| File | From | To |\n",
                "|------|------|----|\n",
            ]
            for u in upgrades:
                lines.append(f"| {u['file']} | {u['from']} | {u['to']} |\n")
            addon.write_text("".join(lines), encoding="utf-8")

        # Update baseline with merged content
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(merged_baseline, ensure_ascii=False, indent=2), encoding="utf-8")

        ok = True
        summary = (
            f"Canonical monitoring OK. Upgrades: {len(upgrades)}." if upgrades else "Canonical monitoring OK. No upgrades detected."
        )
        issues = [{"upgrade": u} for u in upgrades]
        return AgentResult(self.name, ok, issues, summary)


