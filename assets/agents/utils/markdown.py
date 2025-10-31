from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple


MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_links(text: str) -> List[Tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in MD_LINK.finditer(text)]


def collect_headings(text: str) -> Dict[str, str]:
    ids: Dict[str, str] = {}
    for line in text.splitlines():
        m = MD_HEADING.match(line.strip())
        if not m:
            continue
        title = m.group(2).strip()
        anchor = slugify(title)
        ids[anchor] = title
    return ids


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\s_-]", "", title)
    slug = re.sub(r"\s+", "-", slug).strip("-").lower()
    return slug


def extract_status_tags(text: str) -> List[str]:
    return re.findall(r"\*\*(PROVEN|TOPOLOGICAL|DERIVED|THEORETICAL|PHENOMENOLOGICAL|EXPLORATORY)\*\*", text)


