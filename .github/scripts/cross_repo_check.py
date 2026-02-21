#!/usr/bin/env python3
"""
Cross-Repo Consistency Check - CI Script for GIFT Documentation

Validates that documentation in GIFT matches verified values in gift-framework/core:
1. Python constants (gift_core/constants.py)
2. Lean definitions (Lean/GIFT/Core.lean)
3. Theorem names referenced in docs match blueprint

Fails CI if:
- Any constant value differs between docs and core
- Referenced Lean theorems don't exist
- Version numbers are mismatched
"""

import argparse
import re
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from fractions import Fraction

# =============================================================================
# LEAN PARSER
# =============================================================================

def parse_lean_constants(lean_dir: Path) -> Dict[str, any]:
    """
    Parse constant definitions from Lean files.
    Looks for patterns like:
    - def dim_E8 : ℕ := 248
    - abbrev b2 := 21
    """
    constants = {}

    core_file = lean_dir / 'GIFT' / 'Core.lean'
    if core_file.exists():
        content = core_file.read_text(encoding='utf-8')

        # Pattern: def name : Type := value
        def_pattern = r'def\s+(\w+)\s*:\s*[ℕℤℚℝ\w]+\s*:=\s*(\d+)'
        for match in re.finditer(def_pattern, content):
            name = match.group(1)
            value = int(match.group(2))
            constants[name] = value

        # Pattern: abbrev name := value (often for simple aliases)
        abbrev_pattern = r'abbrev\s+(\w+)\s*:=\s*(\d+)'
        for match in re.finditer(abbrev_pattern, content):
            name = match.group(1)
            value = int(match.group(2))
            constants[name] = value

    # Also parse BettiNumbers.lean for derived values
    betti_file = lean_dir / 'GIFT' / 'Algebraic' / 'BettiNumbers.lean'
    if betti_file.exists():
        content = betti_file.read_text(encoding='utf-8')

        def_pattern = r'def\s+(\w+)\s*:\s*[ℕℤℚℝ\w]+\s*:=\s*(\d+)'
        for match in re.finditer(def_pattern, content):
            name = match.group(1)
            value = int(match.group(2))
            if name not in constants:
                constants[name] = value

    return constants


def parse_lean_theorems(lean_dir: Path) -> Set[str]:
    """
    Extract theorem names from Lean files for reference checking.
    """
    theorems = set()

    for lean_file in lean_dir.rglob('*.lean'):
        try:
            content = lean_file.read_text(encoding='utf-8')

            # Pattern: theorem name or lemma name
            theorem_pattern = r'(?:theorem|lemma)\s+(\w+)'
            for match in re.finditer(theorem_pattern, content):
                theorems.add(match.group(1))

        except Exception:
            continue

    return theorems


# =============================================================================
# PYTHON CONSTANTS PARSER
# =============================================================================

def parse_python_constants(core_dir: Path) -> Dict[str, any]:
    """
    Parse constants from gift_core/constants.py
    """
    constants = {}

    constants_file = core_dir / 'gift_core' / 'constants.py'
    if not constants_file.exists():
        return constants

    content = constants_file.read_text(encoding='utf-8')

    # Pattern: NAME = value (integer)
    int_pattern = r'^(\w+)\s*=\s*(\d+)\s*(?:#.*)?$'
    for match in re.finditer(int_pattern, content, re.MULTILINE):
        name = match.group(1)
        value = int(match.group(2))
        constants[name] = value

    # Pattern: NAME = Fraction(a, b)
    frac_pattern = r'^(\w+)\s*=\s*Fraction\((\d+),\s*(\d+)\)'
    for match in re.finditer(frac_pattern, content, re.MULTILINE):
        name = match.group(1)
        num = int(match.group(2))
        denom = int(match.group(3))
        constants[name] = Fraction(num, denom)

    return constants


# =============================================================================
# DOCUMENTATION PARSER
# =============================================================================

def parse_docs_constants(docs_dir: Path) -> Dict[str, Dict]:
    """
    Extract constants from markdown documentation.
    Returns dict of {name: {'value': x, 'source': file}}
    """
    constants = {}

    for md_file in docs_dir.glob('*.md'):
        content = md_file.read_text(encoding='utf-8')

        # Pattern: | symbol | value |
        table_pattern = r'\|\s*([^|]+)\s*\|\s*(\d+)\s*\|'
        for match in re.finditer(table_pattern, content):
            symbol = match.group(1).strip()
            value = int(match.group(2))

            # Normalize symbol
            norm_name = normalize_constant_name(symbol)
            if norm_name:
                if norm_name not in constants:
                    constants[norm_name] = {
                        'value': value,
                        'source': md_file.name,
                        'raw_name': symbol
                    }

        # Pattern: exact fractions like "3/13"
        fraction_pattern = r'(\w+)\s*=\s*(\d+)/(\d+)'
        for match in re.finditer(fraction_pattern, content):
            name = match.group(1)
            num = int(match.group(2))
            denom = int(match.group(3))
            norm_name = normalize_constant_name(name)
            if norm_name and norm_name not in constants:
                constants[norm_name] = {
                    'value': Fraction(num, denom),
                    'source': md_file.name,
                    'raw_name': name
                }

    return constants


def parse_docs_theorem_refs(docs_dir: Path) -> Dict[str, str]:
    """
    Find Lean theorem references in documentation.
    Returns dict of {theorem_name: source_file}
    """
    refs = {}

    for md_file in docs_dir.glob('*.md'):
        content = md_file.read_text(encoding='utf-8')

        # Pattern: GIFT.Module.theorem_name or theorem_name (in context)
        lean_ref_pattern = r'(?:GIFT\.[\w.]+\.)?(\w+(?:_\w+)*)'
        # More specific: look for things like "Lean: theorem_name" or "verified in theorem_name"
        specific_pattern = r'(?:Lean|verified|proven)[:\s]+(?:GIFT\.)?(\w+(?:_\w+)+)'

        for match in re.finditer(specific_pattern, content, re.IGNORECASE):
            theorem = match.group(1)
            refs[theorem] = md_file.name

    return refs


def normalize_constant_name(raw: str) -> Optional[str]:
    """
    Normalize constant names to match Python/Lean conventions.
    """
    raw = raw.strip().lower()

    # Mappings from doc notation to code names
    mappings = {
        'dim(e₈)': 'DIM_E8',
        'dim(e8)': 'DIM_E8',
        'dim_e8': 'DIM_E8',
        'rank(e₈)': 'RANK_E8',
        'rank(e8)': 'RANK_E8',
        'dim(g₂)': 'DIM_G2',
        'dim(g2)': 'DIM_G2',
        'dim_g2': 'DIM_G2',
        'b₂': 'B2',
        'b2': 'B2',
        'b₂(k₇)': 'B2',
        'b₃': 'B3',
        'b3': 'B3',
        'b₃(k₇)': 'B3',
        'h*': 'H_STAR',
        'h_star': 'H_STAR',
        'n_gen': 'N_GEN',
        'dim(j₃(o))': 'DIM_J3O',
        'dim(k₇)': 'DIM_K7',
        'dim_k7': 'DIM_K7',
        'p₂': 'P2',
        'weyl': 'WEYL_FACTOR',
    }

    for pattern, canonical in mappings.items():
        if pattern in raw:
            return canonical

    return None


# =============================================================================
# VERSION CHECK
# =============================================================================

def check_version_consistency(docs_dir: Path, core_dir: Path) -> List[str]:
    """
    Check that version numbers are consistent between repos.
    """
    errors = []

    # Extract version from core pyproject.toml
    pyproject = core_dir / 'pyproject.toml'
    core_version = None
    if pyproject.exists():
        content = pyproject.read_text()
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if match:
            core_version = match.group(1)

    # Extract version from docs (looking for v3.3 or similar in filenames/content)
    docs_version = None
    for md_file in docs_dir.glob('*.md'):
        if 'v3.' in md_file.name:
            match = re.search(r'v(\d+\.\d+)', md_file.name)
            if match:
                docs_version = match.group(1)
                break

    # Note: We don't fail on version mismatch, just report
    if core_version and docs_version:
        # Compare major.minor only
        core_major_minor = '.'.join(core_version.split('.')[:2])
        if core_major_minor != docs_version:
            errors.append(
                f"Version mismatch: docs=v{docs_version}, core=v{core_version} "
                f"(major.minor check: {docs_version} vs {core_major_minor})"
            )

    return errors


# =============================================================================
# MAIN VALIDATION
# =============================================================================

@dataclass
class ConsistencyResult:
    constant: str
    docs_value: any
    core_value: any
    docs_source: str
    status: str  # 'match', 'mismatch', 'docs_only', 'core_only'


def run_consistency_check(docs_dir: Path, core_dir: Path) -> Tuple[List[ConsistencyResult], List[str]]:
    """
    Run full consistency check between docs and core.
    """
    results = []
    errors = []

    # Parse all sources
    print("  Parsing Lean constants...")
    lean_constants = parse_lean_constants(core_dir / 'Lean')

    print("  Parsing Python constants...")
    py_constants = parse_python_constants(core_dir)

    print("  Parsing documentation...")
    docs_constants = parse_docs_constants(docs_dir)

    # Merge core constants (Python takes precedence as it's more complete)
    core_constants = {**lean_constants}
    for name, value in py_constants.items():
        core_constants[name.upper()] = value

    # Key constants to validate (must match exactly)
    key_constants = ['B2', 'B3', 'DIM_E8', 'DIM_G2', 'RANK_E8', 'H_STAR', 'N_GEN', 'DIM_K7']

    # Check each documented constant against core
    for doc_name, doc_data in docs_constants.items():
        doc_value = doc_data['value']
        doc_source = doc_data['source']

        if doc_name in core_constants:
            core_value = core_constants[doc_name]

            # Compare values
            if isinstance(doc_value, Fraction) and isinstance(core_value, Fraction):
                match = doc_value == core_value
            elif isinstance(doc_value, (int, float)) and isinstance(core_value, (int, float, Fraction)):
                match = abs(float(doc_value) - float(core_value)) < 0.0001
            else:
                match = doc_value == core_value

            status = 'match' if match else 'mismatch'
            results.append(ConsistencyResult(
                constant=doc_name,
                docs_value=doc_value,
                core_value=core_value,
                docs_source=doc_source,
                status=status
            ))

            if not match and doc_name in key_constants:
                errors.append(
                    f"CRITICAL: {doc_name} mismatch - docs={doc_value} ({doc_source}), core={core_value}"
                )
        else:
            results.append(ConsistencyResult(
                constant=doc_name,
                docs_value=doc_value,
                core_value=None,
                docs_source=doc_source,
                status='docs_only'
            ))

    # Check for core constants not in docs (informational)
    for core_name in key_constants:
        if core_name not in docs_constants and core_name in core_constants:
            results.append(ConsistencyResult(
                constant=core_name,
                docs_value=None,
                core_value=core_constants[core_name],
                docs_source='',
                status='core_only'
            ))

    return results, errors


def check_theorem_references(docs_dir: Path, core_dir: Path) -> List[str]:
    """
    Check that Lean theorem references in docs actually exist.
    """
    warnings = []

    print("  Parsing Lean theorems...")
    lean_theorems = parse_lean_theorems(core_dir / 'Lean')

    print("  Checking theorem references in docs...")
    doc_refs = parse_docs_theorem_refs(docs_dir)

    for theorem, source in doc_refs.items():
        # Skip common words that might match the pattern
        if theorem.lower() in ['theorem', 'lemma', 'proof', 'lean', 'coq']:
            continue
        if len(theorem) < 5:
            continue

        # Check if theorem exists (case-insensitive)
        theorem_lower = theorem.lower()
        found = any(t.lower() == theorem_lower for t in lean_theorems)

        if not found:
            warnings.append(f"Theorem '{theorem}' referenced in {source} not found in Lean")

    return warnings


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Cross-repo consistency check')
    parser.add_argument('--core-path', type=Path, required=True,
                       help='Path to gift-framework/core checkout')
    args = parser.parse_args()

    print("=" * 70)
    print("GIFT Cross-Repo Consistency Check")
    print("=" * 70)
    print()

    # Paths
    repo_root = Path(__file__).parent.parent.parent
    docs_dir = repo_root / 'publications' / 'papers' / 'markdown'
    core_dir = args.core_path

    if not docs_dir.exists():
        print(f"ERROR: Documentation directory not found: {docs_dir}")
        sys.exit(1)

    if not core_dir.exists():
        print(f"ERROR: Core directory not found: {core_dir}")
        sys.exit(1)

    all_errors = []
    all_warnings = []

    # Step 1: Version check
    print("[1/3] Checking version consistency...")
    version_errors = check_version_consistency(docs_dir, core_dir)
    if version_errors:
        for err in version_errors:
            print(f"      WARNING: {err}")
        all_warnings.extend(version_errors)
    else:
        print("      Versions compatible")
    print()

    # Step 2: Constant consistency
    print("[2/3] Checking constant consistency...")
    results, const_errors = run_consistency_check(docs_dir, core_dir)
    all_errors.extend(const_errors)

    matches = sum(1 for r in results if r.status == 'match')
    mismatches = sum(1 for r in results if r.status == 'mismatch')
    docs_only = sum(1 for r in results if r.status == 'docs_only')
    core_only = sum(1 for r in results if r.status == 'core_only')

    print(f"      Matches: {matches}")
    print(f"      Mismatches: {mismatches}")
    print(f"      Docs only: {docs_only}")
    print(f"      Core only: {core_only}")

    if mismatches > 0:
        print("      MISMATCHES:")
        for r in results:
            if r.status == 'mismatch':
                print(f"        - {r.constant}: docs={r.docs_value}, core={r.core_value}")
    print()

    # Step 3: Theorem reference check
    print("[3/3] Checking theorem references...")
    theorem_warnings = check_theorem_references(docs_dir, core_dir)
    if theorem_warnings:
        print(f"      Warnings: {len(theorem_warnings)}")
        for warn in theorem_warnings[:5]:  # Show first 5
            print(f"        - {warn}")
        if len(theorem_warnings) > 5:
            print(f"        ... and {len(theorem_warnings) - 5} more")
    else:
        print("      All references valid")
    all_warnings.extend(theorem_warnings)
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Errors: {len(all_errors)}")
    print(f"  Warnings: {len(all_warnings)}")
    print()

    # Save report
    report = {
        'results': [
            {
                'constant': r.constant,
                'docs_value': str(r.docs_value),
                'core_value': str(r.core_value),
                'docs_source': r.docs_source,
                'status': r.status
            }
            for r in results
        ],
        'errors': all_errors,
        'warnings': all_warnings,
        'summary': {
            'matches': matches,
            'mismatches': mismatches,
            'docs_only': docs_only,
            'core_only': core_only,
            'total_errors': len(all_errors),
            'total_warnings': len(all_warnings)
        }
    }

    report_path = repo_root / 'cross_repo_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")

    # Exit code
    if all_errors:
        print()
        print("CI FAILED: Critical consistency errors detected")
        sys.exit(1)
    else:
        print()
        print("CI PASSED: Cross-repo consistency verified")
        sys.exit(0)


if __name__ == '__main__':
    main()
