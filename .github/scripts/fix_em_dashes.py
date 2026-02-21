#!/usr/bin/env python3
"""
Em Dash Eradication Script for GIFT Documentation.

Replaces em dashes (—) and en dashes used as em dashes (–) with
contextually appropriate punctuation:

  - Paired em dashes  "foo — bar — baz"  →  "foo (bar) baz"
  - Before elaboration "foo — the bar"    →  "foo: the bar"
  - Default fallback   "foo — bar"        →  "foo, bar"

Run standalone:
    python .github/scripts/fix_em_dashes.py [--check] [--verbose]

In --check mode, exits with code 1 if em dashes are found (no files modified).
Without --check, fixes files in place.
"""

import re
import sys
import argparse
from pathlib import Path

# Characters to eradicate
EM_DASH = '\u2014'  # —
EN_DASH_AS_EM = ' \u2013 '  # – used with spaces (em dash style)

# Lowercase words that typically introduce an elaboration (colon-worthy)
COLON_STARTERS = {
    'the', 'a', 'an', 'it', 'this', 'that', 'these', 'those',
    'they', 'we', 'he', 'she', 'its', 'their', 'our',
    'for', 'if', 'when', 'because', 'since', 'whether',
    'namely', 'specifically', 'i.e.', 'e.g.',
    'no', 'not', 'only', 'any', 'every', 'each',
}


def find_markdown_files(repo_root: Path) -> list[Path]:
    """Find all markdown files in linted directories."""
    dirs = [
        repo_root / 'publications' / 'papers' / 'markdown',
        repo_root / 'docs',
        repo_root / 'publications' / 'validation',
        repo_root / 'publications' / 'references',
        repo_root / 'publications' / 'outreach',
    ]
    files = []
    for d in dirs:
        if d.exists():
            files.extend(d.rglob('*.md'))
    # Also check root-level md files
    for f in repo_root.glob('*.md'):
        files.append(f)
    return sorted(set(files))


def replace_paired_em_dashes(line: str) -> str:
    """Replace paired em dashes with parentheses.

    "foo — bar baz — qux" → "foo (bar baz) qux"
    """
    # Pattern: text — content — text  (greedy on the inner part, non-greedy match)
    pattern = rf'\s*{EM_DASH}\s*(.+?)\s*{EM_DASH}\s*'
    match = re.search(pattern, line)
    if match:
        inner = match.group(1).strip()
        # Replace the matched region: add space before (, space after )
        start, end = match.start(), match.end()
        before = line[:start]
        after = line[end:]
        # Ensure spacing: "word (inner) word"
        sep_before = ' ' if before and not before.endswith(' ') else ''
        sep_after = ' ' if after and not after.startswith(' ') and not after.startswith(',') and not after.startswith('.') else ''
        line = f"{before}{sep_before}({inner}){sep_after}{after}"
    return line


def replace_single_em_dash(line: str) -> str:
    """Replace a single em dash with colon or comma based on context."""
    if EM_DASH not in line:
        return line

    # Split on em dash (handle " — " and "—")
    pattern = rf'\s*{EM_DASH}\s*'
    parts = re.split(pattern, line, maxsplit=1)
    if len(parts) != 2:
        return line

    before, after = parts
    before = before.rstrip()
    after = after.lstrip()

    if not after:
        return line

    # Decide: colon or comma?
    first_word = after.split()[0].lower().rstrip('.,;:') if after.split() else ''

    if first_word in COLON_STARTERS:
        sep = ':'
    elif after[0].isupper() and not before.endswith(('i.e.', 'e.g.')):
        # Capital letter after dash often signals an independent clause → colon
        sep = ':'
    else:
        sep = ','

    return f"{before}{sep} {after}"


def fix_line(line: str) -> str:
    """Fix all em dashes in a single line."""
    if EM_DASH not in line and EN_DASH_AS_EM not in line:
        return line

    # Normalize en-dash-as-em to em dash first
    line = line.replace(EN_DASH_AS_EM, f' {EM_DASH} ')

    # Skip lines in code blocks (caller handles block-level tracking)
    if line.strip().startswith('```'):
        return line

    # Count em dashes
    count = line.count(EM_DASH)

    if count >= 2:
        # Try paired replacement first
        line = replace_paired_em_dashes(line)
        # If any remain, handle them as singles
        while EM_DASH in line:
            line = replace_single_em_dash(line)
    elif count == 1:
        line = replace_single_em_dash(line)

    return line


def fix_content(content: str) -> str:
    """Fix all em dashes in a document, respecting code blocks."""
    lines = content.split('\n')
    result = []
    in_code_block = False

    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            result.append(line)
            continue

        if in_code_block:
            result.append(line)
            continue

        result.append(fix_line(line))

    return '\n'.join(result)


def check_file(filepath: Path) -> list[tuple[int, str]]:
    """Return list of (line_number, line) containing em dashes."""
    content = filepath.read_text(encoding='utf-8')
    hits = []
    in_code_block = False

    for i, line in enumerate(content.split('\n'), 1):
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if EM_DASH in line:
            hits.append((i, line.strip()))

    return hits


def main():
    parser = argparse.ArgumentParser(description='Eradicate em dashes from GIFT docs')
    parser.add_argument('--check', action='store_true',
                        help='Check only, do not modify files (exit 1 if found)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show each replacement')
    parser.add_argument('files', nargs='*', type=Path,
                        help='Specific files to process (default: all docs)')
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent
    files = args.files if args.files else find_markdown_files(repo_root)

    total_found = 0
    total_fixed = 0
    affected_files = []

    for filepath in files:
        if not filepath.exists():
            print(f"  SKIP {filepath} (not found)")
            continue

        hits = check_file(filepath)
        if not hits:
            continue

        total_found += len(hits)
        rel = filepath.relative_to(repo_root) if filepath.is_relative_to(repo_root) else filepath

        if args.check:
            affected_files.append(rel)
            for line_num, line in hits:
                print(f"  {rel}:{line_num}: {line[:80]}")
        else:
            content = filepath.read_text(encoding='utf-8')
            fixed = fix_content(content)
            if fixed != content:
                filepath.write_text(fixed, encoding='utf-8')
                n = len(hits)
                total_fixed += n
                affected_files.append(rel)
                print(f"  FIXED {rel} ({n} em dash{'es' if n > 1 else ''})")
                if args.verbose:
                    for line_num, line in hits:
                        print(f"    L{line_num}: {line[:80]}")

    print()
    if args.check:
        if total_found > 0:
            print(f"FAILED: {total_found} em dash(es) found in {len(affected_files)} file(s)")
            print("Run: python .github/scripts/fix_em_dashes.py")
            sys.exit(1)
        else:
            print("PASSED: No em dashes found")
            sys.exit(0)
    else:
        if total_fixed > 0:
            print(f"Fixed {total_fixed} em dash(es) in {len(affected_files)} file(s)")
        else:
            print("No em dashes found, nothing to fix")
        sys.exit(0)


if __name__ == '__main__':
    main()
