from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

from .base import AgentResult, BaseAgent
from .utils.fs import discover_files


class NotebookExecutionAgent(BaseAgent):
    name = "notebooks"

    def run(self, root: Path) -> AgentResult:
        issues: List[Dict[str, Any]] = []
        notebooks = discover_files(root, ["publications/**/*.ipynb", "assets/visualizations/**/*.ipynb"])
        have_papermill = shutil.which("papermill") is not None
        have_jupyter = shutil.which("jupyter") is not None

        if not notebooks:
            return AgentResult(self.name, True, [], "No notebooks found.")

        if not (have_papermill or have_jupyter):
            return AgentResult(self.name, True, [{"info": "papermill/jupyter not installed; skipping execution"}], "Skipping execution (dependencies missing).")

        # Conservative: report-only for now; execution can be enabled later.
        info = [{"notebook": str(nb)} for nb in notebooks]
        return AgentResult(self.name, True, info, f"Found {len(notebooks)} notebooks (execution disabled by default).")


