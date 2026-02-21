#!/usr/bin/env python3
"""
Documentation Linter - CI Script for GIFT Documentation

Checks documentation quality and consistency:
1. No internal jargon (per CLAUDE.md guidelines)
2. Valid cross-references between documents
3. Consistent notation (Unicode vs ASCII)
4. No evolutionary language ("In v3.1, we improved...")
5. Proper status classifications (PROVEN, TOPOLOGICAL, etc.)
6. Valid markdown structure

Fails CI on errors, warns on style issues.
"""

import re
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

# =============================================================================
# LINT RULES
# =============================================================================

# Internal jargon to flag (from CLAUDE.md)
# Note: Some terms like "Fano structure" are valid in explanatory contexts
INTERNAL_JARGON = {
    r'\bB4\s*axiom': 'Use "Lagrange identity for G₂-invariant cross product" instead of "B4 axiom"',
    r'\bepsilon\s*contraction': 'Use "G₂ structure constants" instead of "epsilon contraction"',
    r'\baxiom\s*resolution': 'Avoid internal axiom numbering; describe the mathematical property',
    r'\baxiom\s+\d+/\d+': 'Avoid internal axiom numbering; describe the mathematical property',
}

# Evolutionary language patterns to avoid
EVOLUTIONARY_PATTERNS = {
    r'[Ii]n\s+v\d+\.\d+': 'Avoid evolutionary language like "In v3.1, we improved..."; state current results only',
    r'[Ww]e\s+improved': 'Avoid evolutionary language; state current results only',
    r'[Pp]reviously': 'Avoid evolutionary language; state current results only',
    r'[Uu]pdated?\s+from': 'Avoid evolutionary language; state current results only',
    r'[Nn]ow\s+(?:we|the|it)\s+(?:use|have|is)': 'Consider rephrasing to avoid "now" temporal references',
}

# Valid status classifications
VALID_STATUSES = {
    'PROVEN',
    'PROVEN (Lean 4)',
    'PROVEN (Coq)',
    'TOPOLOGICAL',
    'THEORETICAL',
    'SPECULATIVE',
    'HEURISTIC',
    'DERIVED',
    'PHENOMENOLOGICAL',
    'EXPLORATORY',
    'CONSISTENT',
    'COMPLETE',
    'EXPERIMENTAL',
}

# Notation preferences (warn on non-preferred)
NOTATION_PREFERENCES = {
    r'sin\^2': ('sin²', 'Prefer Unicode superscript: sin² instead of sin^2'),
    r'theta_W': ('θ_W', 'Prefer Unicode: θ_W instead of theta_W'),
    r'\bb_2\b': ('b₂', 'Prefer Unicode subscript: b₂ instead of b_2'),
    r'\bb_3\b': ('b₃', 'Prefer Unicode subscript: b₃ instead of b_3'),
    r'\bE_8\b': ('E₈', 'Prefer Unicode subscript: E₈ instead of E_8'),
    r'\bG_2\b': ('G₂', 'Prefer Unicode subscript: G₂ instead of G_2'),
    r'\bK_7\b': ('K₇', 'Prefer Unicode subscript: K₇ instead of K_7'),
    r'<=': ('≤', 'Prefer Unicode: ≤ instead of <='),
    r'>=': ('≥', 'Prefer Unicode: ≥ instead of >='),
    r'!=': ('≠', 'Prefer Unicode: ≠ instead of !='),
    r'\+-': ('±', 'Prefer Unicode: ± instead of +-'),
}

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class LintIssue:
    file: str
    line: int
    column: int
    severity: str  # 'error', 'warning', 'info'
    category: str
    message: str
    context: str  # The line content


@dataclass
class CrossReference:
    source_file: str
    target: str
    line: int
    ref_type: str  # 'section', 'supplement', 'external'


# =============================================================================
# LINTING FUNCTIONS
# =============================================================================

def check_internal_jargon(content: str, filename: str) -> List[LintIssue]:
    """Check for internal jargon that should be replaced with standard terms."""
    issues = []
    lines = content.split('\n')

    for line_num, line in enumerate(lines, 1):
        for pattern, message in INTERNAL_JARGON.items():
            for match in re.finditer(pattern, line, re.IGNORECASE):
                issues.append(LintIssue(
                    file=filename,
                    line=line_num,
                    column=match.start() + 1,
                    severity='error',
                    category='jargon',
                    message=message,
                    context=line.strip()
                ))

    return issues


def check_evolutionary_language(content: str, filename: str) -> List[LintIssue]:
    """Check for evolutionary language that should be avoided."""
    issues = []
    lines = content.split('\n')

    for line_num, line in enumerate(lines, 1):
        # Skip code blocks
        if line.strip().startswith('```') or line.strip().startswith('    '):
            continue

        for pattern, message in EVOLUTIONARY_PATTERNS.items():
            for match in re.finditer(pattern, line):
                issues.append(LintIssue(
                    file=filename,
                    line=line_num,
                    column=match.start() + 1,
                    severity='warning',
                    category='evolutionary',
                    message=message,
                    context=line.strip()
                ))

    return issues


def check_notation_consistency(content: str, filename: str) -> List[LintIssue]:
    """Check for non-preferred notation styles."""
    issues = []
    lines = content.split('\n')

    for line_num, line in enumerate(lines, 1):
        # Skip code blocks and LaTeX math
        if line.strip().startswith('```'):
            continue

        for pattern, (preferred, message) in NOTATION_PREFERENCES.items():
            for match in re.finditer(pattern, line):
                # Don't flag if already in a LaTeX context
                pre_context = line[:match.start()]
                if '$' in pre_context and pre_context.count('$') % 2 == 1:
                    continue  # Inside LaTeX

                issues.append(LintIssue(
                    file=filename,
                    line=line_num,
                    column=match.start() + 1,
                    severity='info',
                    category='notation',
                    message=message,
                    context=line.strip()
                ))

    return issues


def check_status_classifications(content: str, filename: str) -> List[LintIssue]:
    """Check that status classifications are valid."""
    issues = []
    lines = content.split('\n')

    # Pattern: **Status**: VALUE (must be bold or at start of definition)
    # Be more specific to avoid false positives
    status_patterns = [
        r'\*\*[Ss]tatus\*\*\s*:\s*([A-Z][A-Z\s\(\)0-9]+)',  # **Status**: PROVEN
        r'^\|\s*[Ss]tatus\s*\|\s*([A-Z][A-Z\s\(\)]+)\s*\|',  # | Status | PROVEN |
        r'^[Ss]tatus\s*:\s*([A-Z][A-Z\s\(\)0-9]+)',  # Status: PROVEN (at line start)
    ]

    for line_num, line in enumerate(lines, 1):
        for pattern in status_patterns:
            for match in re.finditer(pattern, line):
                status = match.group(1).strip()

                # Skip if too short or looks like noise
                if len(status) < 4:
                    continue

                # Check if it's a valid status
                valid = any(status.upper().startswith(s) for s in VALID_STATUSES)

                if not valid:
                    issues.append(LintIssue(
                        file=filename,
                        line=line_num,
                        column=match.start() + 1,
                        severity='warning',
                        category='status',
                        message=f"Unknown status '{status}'. Valid: {', '.join(sorted(VALID_STATUSES))}",
                        context=line.strip()
                    ))

    return issues


def extract_cross_references(content: str, filename: str) -> List[CrossReference]:
    """Extract all cross-references from a document."""
    refs = []
    lines = content.split('\n')

    for line_num, line in enumerate(lines, 1):
        # Pattern: [text](#section-anchor)
        internal_pattern = r'\[([^\]]+)\]\(#([^)]+)\)'
        for match in re.finditer(internal_pattern, line):
            refs.append(CrossReference(
                source_file=filename,
                target=match.group(2),
                line=line_num,
                ref_type='section'
            ))

        # Pattern: [text](S1_foundations.md) or similar
        supplement_pattern = r'\[([^\]]+)\]\(([^)]+\.md)(?:#[^)]*)?\)'
        for match in re.finditer(supplement_pattern, line):
            refs.append(CrossReference(
                source_file=filename,
                target=match.group(2),
                line=line_num,
                ref_type='supplement'
            ))

        # Pattern: "see S1" or "see Supplement 1"
        see_pattern = r'[Ss]ee\s+(?:S(?:upplement\s*)?)?(\d)'
        for match in re.finditer(see_pattern, line):
            refs.append(CrossReference(
                source_file=filename,
                target=f'S{match.group(1)}',
                line=line_num,
                ref_type='supplement'
            ))

    return refs


def extract_section_anchors(content: str) -> Set[str]:
    """Extract all section anchors (headers) from a document."""
    anchors = set()

    # Pattern: ## Header Text
    header_pattern = r'^#{1,6}\s+(.+)$'

    for match in re.finditer(header_pattern, content, re.MULTILINE):
        header = match.group(1).strip()
        # Convert to anchor format (lowercase, dashes)
        anchor = re.sub(r'[^\w\s-]', '', header.lower())
        anchor = re.sub(r'\s+', '-', anchor)
        anchors.add(anchor)

    return anchors


def check_cross_references(docs_dir: Path) -> List[LintIssue]:
    """Check that all cross-references are valid."""
    issues = []

    # Collect all documents and their anchors
    doc_anchors = {}
    doc_refs = {}

    for md_file in docs_dir.glob('*.md'):
        content = md_file.read_text(encoding='utf-8')
        filename = md_file.name

        doc_anchors[filename] = extract_section_anchors(content)
        doc_refs[filename] = extract_cross_references(content, filename)

    # Validate references
    for filename, refs in doc_refs.items():
        for ref in refs:
            if ref.ref_type == 'section':
                # Check if anchor exists in same file
                if ref.target not in doc_anchors.get(filename, set()):
                    issues.append(LintIssue(
                        file=filename,
                        line=ref.line,
                        column=1,
                        severity='warning',
                        category='cross-ref',
                        message=f"Section anchor '#{ref.target}' not found in document",
                        context=f"Reference to #{ref.target}"
                    ))

            elif ref.ref_type == 'supplement':
                # Check if target file exists
                target_file = ref.target
                if target_file.startswith('S') and len(target_file) <= 2:
                    # Convert S1 -> GIFT_v3.3_S1_foundations.md pattern
                    continue  # Skip abbreviated references for now

                if target_file not in doc_anchors:
                    issues.append(LintIssue(
                        file=filename,
                        line=ref.line,
                        column=1,
                        severity='warning',
                        category='cross-ref',
                        message=f"Referenced file '{target_file}' not found",
                        context=f"Reference to {target_file}"
                    ))

    return issues


def check_markdown_structure(content: str, filename: str) -> List[LintIssue]:
    """Check markdown structural issues."""
    issues = []
    lines = content.split('\n')

    # Track header hierarchy
    prev_level = 0
    in_code_block = False

    for line_num, line in enumerate(lines, 1):
        # Track code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        # Check header hierarchy (no skipping levels)
        header_match = re.match(r'^(#{1,6})\s+', line)
        if header_match:
            level = len(header_match.group(1))
            if level > prev_level + 1 and prev_level > 0:
                issues.append(LintIssue(
                    file=filename,
                    line=line_num,
                    column=1,
                    severity='info',
                    category='structure',
                    message=f"Header level skipped: h{prev_level} to h{level}",
                    context=line.strip()
                ))
            prev_level = level

        # Check for very long lines (>200 chars, excluding tables and math)
        if len(line) > 200 and not line.strip().startswith('|') and '$' not in line:
            issues.append(LintIssue(
                file=filename,
                line=line_num,
                column=200,
                severity='info',
                category='structure',
                message=f"Very long line ({len(line)} chars)",
                context=line[:50] + '...'
            ))

    return issues


# =============================================================================
# MAIN
# =============================================================================

def lint_file(filepath: Path) -> List[LintIssue]:
    """Run all linters on a single file."""
    content = filepath.read_text(encoding='utf-8')
    filename = filepath.name

    issues = []
    issues.extend(check_internal_jargon(content, filename))
    issues.extend(check_evolutionary_language(content, filename))
    issues.extend(check_notation_consistency(content, filename))
    issues.extend(check_status_classifications(content, filename))
    issues.extend(check_markdown_structure(content, filename))

    return issues


def main():
    print("=" * 70)
    print("GIFT Documentation Linter")
    print("=" * 70)
    print()

    # Paths
    repo_root = Path(__file__).parent.parent.parent
    docs_dirs = [
        repo_root / 'publications' / 'papers' / 'markdown',
        repo_root / 'docs'
    ]

    all_issues = []

    # Lint each directory
    for docs_dir in docs_dirs:
        if not docs_dir.exists():
            continue

        print(f"[*] Linting {docs_dir.relative_to(repo_root)}...")

        for md_file in docs_dir.glob('*.md'):
            print(f"    - {md_file.name}")
            issues = lint_file(md_file)
            all_issues.extend(issues)

    # Cross-reference check (across all docs)
    print()
    print("[*] Checking cross-references...")
    for docs_dir in docs_dirs:
        if docs_dir.exists():
            xref_issues = check_cross_references(docs_dir)
            all_issues.extend(xref_issues)

    # Summary by severity
    errors = [i for i in all_issues if i.severity == 'error']
    warnings = [i for i in all_issues if i.severity == 'warning']
    infos = [i for i in all_issues if i.severity == 'info']

    print()
    print("=" * 70)
    print("LINT SUMMARY")
    print("=" * 70)
    print(f"  Errors:   {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Info:     {len(infos)}")
    print()

    # Show errors
    if errors:
        print("ERRORS (must fix):")
        for issue in errors:
            print(f"  {issue.file}:{issue.line}:{issue.column}")
            print(f"    [{issue.category}] {issue.message}")
            print(f"    > {issue.context[:80]}")
            print()

    # Show warnings (first 10)
    if warnings:
        print("WARNINGS (should fix):")
        for issue in warnings[:10]:
            print(f"  {issue.file}:{issue.line} [{issue.category}] {issue.message}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
        print()

    # Save report
    report = {
        'issues': [
            {
                'file': i.file,
                'line': i.line,
                'column': i.column,
                'severity': i.severity,
                'category': i.category,
                'message': i.message,
                'context': i.context
            }
            for i in all_issues
        ],
        'summary': {
            'errors': len(errors),
            'warnings': len(warnings),
            'info': len(infos),
            'total': len(all_issues)
        },
        'by_category': {
            cat: len([i for i in all_issues if i.category == cat])
            for cat in set(i.category for i in all_issues)
        }
    }

    report_path = repo_root / 'docs_lint_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")

    # Exit code (fail only on errors)
    if errors:
        print()
        print("CI FAILED: Documentation errors detected")
        sys.exit(1)
    else:
        print()
        print("CI PASSED: Documentation lint complete")
        sys.exit(0)


if __name__ == '__main__':
    main()
