from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

from .base import AgentResult, BaseAgent
from .utils.fs import discover_files


ASCII_LINE = re.compile(r"^[\x00-\x7F]+$")

REPLACEMENTS = {
    "θ": "theta",
    "τ": "tau",
    "μ": "mu",
    "ν": "nu",
    "π": "pi",
    "Ω": "Omega",
    "Δ": "Delta",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ξ": "xi",
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "×": "x",
    "–": "-",
    "—": "-",
    "'": "'",
    "°": "deg",
}


def transliterate_identifier(text: str) -> str:
    text = "".join(REPLACEMENTS.get(ch, ch) for ch in text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if ord(ch) < 128)
    text = re.sub(r"\W+", "_", text).strip("_")
    return text or "id"


class UnicodeSanitizerAgent(BaseAgent):
    name = "unicode"

    def run(self, root: Path) -> AgentResult:
        targets = discover_files(root, ["**/*.py", "**/*.md"])
        issues: List[Dict[str, Any]] = []
        for path in targets:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                issues.append({"file": str(path), "error": f"read_error: {e}"})
                continue
            if ASCII_LINE.match(content):
                continue
            non_ascii_lines: List[Dict[str, Any]] = []
            for i, line in enumerate(content.splitlines(), start=1):
                if not ASCII_LINE.match(line):
                    non_ascii_lines.append({"line": i, "snippet": line[:160]})
            if non_ascii_lines:
                issues.append({"file": str(path), "non_ascii_lines": non_ascii_lines})
        ok = len(issues) == 0
        summary = ("No non-ASCII content detected." if ok else f"Found non-ASCII content in {len(issues)} files.")
        return AgentResult(self.name, ok, issues, summary)


