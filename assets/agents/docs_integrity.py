from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .base import AgentResult, BaseAgent
from .utils.fs import discover_files
from .utils.markdown import collect_headings


class DocsIntegrityAgent(BaseAgent):
    name = "docs"

    def run(self, root: Path) -> AgentResult:
        md_files = discover_files(root, ["README.md", "docs/**/*.md", "publications/**/*.md"])
        issues: List[Dict[str, Any]] = []

        # Duplicate heading anchors per file
        for p in md_files:
            text = p.read_text(encoding="utf-8", errors="replace")
            ids = collect_headings(text)
            seen: Dict[str, int] = {}
            for k in ids.keys():
                seen[k] = seen.get(k, 0) + 1
            dups = [k for k, c in seen.items() if c > 1]
            if dups:
                issues.append({"type": "duplicate_heading", "file": str(p), "anchors": dups})

        ok = len(issues) == 0
        summary = ("Docs integrity OK." if ok else f"Docs integrity issues: {len(issues)} findings.")
        return AgentResult(self.name, ok, issues, summary)


