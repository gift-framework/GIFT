#!/usr/bin/env python3
"""
GIFT Atlas Generator
Generates organized markdown, CSV, and JSON from GIFT_ATLAS.yaml
"""

import csv
import json
import yaml
from pathlib import Path
from datetime import datetime

ATLAS_FILE = Path(__file__).parent / "GIFT_ATLAS.yaml"
OUTPUT_DIR = Path(__file__).parent / "generated"


def load_atlas():
    """Load the YAML atlas file."""
    with open(ATLAS_FILE, 'r') as f:
        return yaml.safe_load(f)


def generate_constants_md(atlas: dict) -> str:
    """Generate CONSTANTS.md from atlas data."""
    constants = atlas['constants']

    lines = [
        "# GIFT Fundamental Constants",
        "",
        f"*Auto-generated from GIFT_ATLAS.yaml on {datetime.now().strftime('%Y-%m-%d')}*",
        "",
        "---",
        "",
        "## Topology (K7 Manifold)",
        "",
        "| Symbol | Value | Origin | LaTeX |",
        "|--------|-------|--------|-------|",
    ]

    for key, val in constants.get('topology', {}).items():
        lines.append(f"| {key} | {val['value']} | {val['origin']} | `{val.get('latex', '')}` |")

    lines.extend([
        "",
        "## Algebra (Lie Groups)",
        "",
        "| Symbol | Value | Origin | LaTeX |",
        "|--------|-------|--------|-------|",
    ])

    for key, val in constants.get('algebra', {}).items():
        lines.append(f"| {key} | {val['value']} | {val['origin']} | `{val.get('latex', '')}` |")

    lines.extend([
        "",
        "## Physics (Standard Model)",
        "",
        "| Symbol | Value | Origin | LaTeX |",
        "|--------|-------|--------|-------|",
    ])

    for key, val in constants.get('physics', {}).items():
        lines.append(f"| {key} | {val['value']} | {val['origin']} | `{val.get('latex', '')}` |")

    lines.extend([
        "",
        "## Sequences",
        "",
        "### Fibonacci",
        "",
        "| n | F_n |",
        "|---|-----|",
    ])

    for key, val in constants.get('sequences', {}).get('fibonacci', {}).items():
        lines.append(f"| {key} | {val} |")

    lines.extend([
        "",
        "### Lucas",
        "",
        "| n | L_n |",
        "|---|-----|",
    ])

    for key, val in constants.get('sequences', {}).get('lucas', {}).items():
        lines.append(f"| {key} | {val} |")

    lines.extend([
        "",
        "### Fermat Primes",
        "",
        "| n | F_n |",
        "|---|-----|",
    ])

    for key, val in constants.get('sequences', {}).get('fermat', {}).items():
        lines.append(f"| {key} | {val} |")

    return "\n".join(lines)


def generate_observables_md(atlas: dict) -> str:
    """Generate OBSERVABLES.md from atlas data."""
    obs = atlas['observables']

    lines = [
        "# GIFT Physical Observables",
        "",
        f"*Auto-generated from GIFT_ATLAS.yaml on {datetime.now().strftime('%Y-%m-%d')}*",
        "",
        "---",
        "",
    ]

    for section, items in obs.items():
        section_title = section.replace('_', ' ').title()
        lines.extend([
            f"## {section_title}",
            "",
            "| # | Name | Formula | Predicted | Experimental | Deviation |",
            "|---|------|---------|-----------|--------------|-----------|",
        ])

        for item in items:
            lines.append(
                f"| {item['id']} | {item['name']} | `{item['formula']}` | "
                f"{item['predicted']} | {item['experimental']} | {item['deviation_pct']}% |"
            )

        lines.append("")

    return "\n".join(lines)


def generate_correspondences_md(atlas: dict) -> str:
    """Generate CORRESPONDENCES.md from atlas data."""
    corr = atlas['correspondences']

    lines = [
        "# GIFT Mathematical Correspondences",
        "",
        f"*Auto-generated from GIFT_ATLAS.yaml on {datetime.now().strftime('%Y-%m-%d')}*",
        "",
        "---",
        "",
        "## Bernoulli-GIFT Correspondence",
        "",
        f"*{corr['bernoulli']['description']}*",
        "",
        "| Lie Group | Rank | B_index | Primes | GIFT Interpretation |",
        "|-----------|------|---------|--------|---------------------|",
    ]

    for m in corr['bernoulli']['mappings']:
        primes = ', '.join(map(str, m['denom_primes']))
        interp = ', '.join([f"{k}={v}" for k, v in m['gift_interpretation'].items()])
        lines.append(f"| {m['group']} | {m['rank']} | B_{m['bernoulli_index']} | {primes} | {interp} |")

    lines.extend([
        "",
        "## Fibonacci-GIFT Correspondence",
        "",
        f"*{corr['fibonacci']['description']}*",
        "",
        "| F_n | Value | GIFT Constant |",
        "|-----|-------|---------------|",
    ])

    for key, val in corr['fibonacci']['mappings'].items():
        lines.append(f"| {key} | {val['value']} | {val['gift']} |")

    lines.extend([
        "",
        "## Lucas-GIFT Correspondence",
        "",
        f"*{corr['lucas']['description']}*",
        "",
        "| L_n | Value | GIFT Constant |",
        "|-----|-------|---------------|",
    ])

    for key, val in corr['lucas']['mappings'].items():
        lines.append(f"| {key} | {val['value']} | {val['gift']} |")

    lines.extend([
        "",
        "## Fermat-GIFT Correspondence",
        "",
        f"*{corr['fermat']['description']}*",
        "",
        "| F_n | Value | GIFT Constant |",
        "|-----|-------|---------------|",
    ])

    for key, val in corr['fermat']['mappings'].items():
        lines.append(f"| {key} | {val['value']} | {val['gift']} |")

    return "\n".join(lines)


def generate_sporadic_md(atlas: dict) -> str:
    """Generate SPORADIC_GROUPS.md from atlas data."""
    spor = atlas['sporadic_groups']

    lines = [
        "# Moonshine-GIFT Correspondence",
        "",
        f"*Auto-generated from GIFT_ATLAS.yaml on {datetime.now().strftime('%Y-%m-%d')}*",
        "",
        f"*{spor['description']}*",
        "",
        "---",
        "",
        "## All 26 Sporadic Groups",
        "",
        "| # | Group | Symbol | dim(V1) | GIFT Formula | Verified |",
        "|---|-------|--------|---------|--------------|----------|",
    ]

    for i, g in enumerate(spor['groups'], 1):
        verified = "Yes" if g['verified'] else "~"
        lines.append(
            f"| {i} | {g['name']} | {g['symbol']} | {g['dim_V1']} | `{g['formula']}` | {verified} |"
        )

    lines.extend([
        "",
        "## j-Invariant",
        "",
        f"**Constant term**: {spor['j_invariant']['constant_term']}",
        "",
        "Formulas:",
    ])

    for f in spor['j_invariant']['formulas']:
        lines.append(f"- `{f}`")

    lines.extend([
        "",
        f"**First coefficient**: {spor['j_invariant']['first_coefficient']}",
        f"",
        f"*{spor['j_invariant']['note']}*",
        "",
        "## Monster Primes",
        "",
        f"*{spor['monster_primes']['description']}*",
        "",
        f"Primes: {', '.join(map(str, spor['monster_primes']['primes']))}",
    ])

    return "\n".join(lines)


def generate_prime_atlas_md(atlas: dict) -> str:
    """Generate PRIME_ATLAS.md from atlas data."""
    pa = atlas['prime_atlas']

    lines = [
        "# GIFT Prime Atlas",
        "",
        f"*Auto-generated from GIFT_ATLAS.yaml on {datetime.now().strftime('%Y-%m-%d')}*",
        "",
        f"*{pa['description']}*",
        "",
        f"**Total primes < 200**: {pa['total_primes']}",
        f"**Coverage**: {pa['coverage']}",
        "",
        "---",
        "",
        "## The Three Generators",
        "",
        "| Generator | Value | Operation | Range |",
        "|-----------|-------|-----------|-------|",
    ]

    for g in pa['generators']:
        lines.append(f"| {g['name']} | {g['value']} | {g['operation']} | {g['range']} |")

    lines.extend([
        "",
        "## Tier 1: Direct Constants",
        "",
        "| Prime | GIFT |",
        "|-------|------|",
    ])

    for prime, gift in pa['tier1_direct']['primes'].items():
        lines.append(f"| {prime} | {gift} |")

    lines.extend([
        "",
        "## Tier 2: b3 Generated",
        "",
        "| Prime | Formula |",
        "|-------|---------|",
    ])

    for prime, formula in pa['tier2_b3']['primes'].items():
        lines.append(f"| {prime} | {formula} |")

    lines.extend([
        "",
        "## Tier 3: H* Generated",
        "",
        "| Prime | Formula |",
        "|-------|---------|",
    ])

    for prime, formula in pa['tier3_H_star']['primes'].items():
        lines.append(f"| {prime} | {formula} |")

    lines.extend([
        "",
        "## Tier 4: dim_E8 Generated",
        "",
        "| Prime | Formula |",
        "|-------|---------|",
    ])

    for prime, formula in pa['tier4_E8']['primes'].items():
        lines.append(f"| {prime} | {formula} |")

    lines.extend([
        "",
        "## Special Primes",
        "",
        "### Heegner Numbers",
        "",
        f"*{pa['special_primes']['heegner']['description']}*",
        "",
        f"Numbers: {', '.join(map(str, pa['special_primes']['heegner']['numbers']))}",
        "",
        "### Mersenne Primes",
        "",
        "| M_n | Value | GIFT |",
        "|-----|-------|------|",
    ])

    for key, val in pa['special_primes']['mersenne'].items():
        if key != 'description':
            lines.append(f"| {key} | {val['value']} | {val['gift']} |")

    return "\n".join(lines)


def generate_relations_md(atlas: dict) -> str:
    """Generate RELATIONS.md from atlas data."""
    rels = atlas['relations']

    lines = [
        "# GIFT Relations Catalog",
        "",
        f"*Auto-generated from GIFT_ATLAS.yaml on {datetime.now().strftime('%Y-%m-%d')}*",
        "",
        f"**Total relations**: ~{atlas['metadata']['total_relations']}",
        "",
        "---",
        "",
    ]

    for section, items in rels.items():
        section_title = section.replace('_', ' ').title()
        lines.extend([
            f"## {section_title} Relations",
            "",
            "| # | Name | Formula | Domain |",
            "|---|------|---------|--------|",
        ])

        for item in items:
            status = f" ({item.get('status', '')})" if item.get('status') else ""
            lines.append(
                f"| {item['id']} | {item['name']}{status} | `{item['formula']}` | {item['domain']} |"
            )

        lines.append("")

    return "\n".join(lines)


def generate_exhaustive_csv(atlas: dict) -> str:
    """Generate a single exhaustive CSV with all data."""
    import io
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(['category', 'subcategory', 'id', 'name', 'value', 'formula',
                     'origin', 'experimental', 'deviation_pct', 'status', 'domain', 'notes'])

    # Constants - Topology
    for key, val in atlas['constants'].get('topology', {}).items():
        writer.writerow(['constant', 'topology', '', key, val['value'], '',
                        val['origin'], '', '', '', '', val.get('latex', '')])

    # Constants - Algebra
    for key, val in atlas['constants'].get('algebra', {}).items():
        writer.writerow(['constant', 'algebra', '', key, val['value'], '',
                        val['origin'], '', '', '', '', val.get('latex', '')])

    # Constants - Physics
    for key, val in atlas['constants'].get('physics', {}).items():
        writer.writerow(['constant', 'physics', '', key, val['value'], '',
                        val['origin'], '', '', '', '', val.get('latex', '')])

    # Constants - Sequences
    for seq_name, seq_data in atlas['constants'].get('sequences', {}).items():
        for key, val in seq_data.items():
            writer.writerow(['constant', f'sequence_{seq_name}', '', key, val, '',
                            seq_name, '', '', '', '', ''])

    # Constants - Derived
    for key, val in atlas['constants'].get('derived', {}).items():
        writer.writerow(['constant', 'derived', '', key, val['value'],
                        val.get('formula', ''), val['origin'], '', '', '', '', ''])

    # Observables
    for section, items in atlas['observables'].items():
        for item in items:
            writer.writerow(['observable', section, item['id'], item['name'],
                            item['predicted'], item['formula'], '',
                            item['experimental'], item['deviation_pct'],
                            item['status'], '', item.get('unit', '')])

    # Sporadic Groups
    for i, g in enumerate(atlas['sporadic_groups']['groups'], 1):
        writer.writerow(['sporadic_group', '', i, g['name'], g['dim_V1'],
                        g['formula'], '', '', '',
                        'verified' if g['verified'] else 'unverified',
                        'moonshine', g['symbol']])

    # Prime Atlas - Direct
    for prime, gift in atlas['prime_atlas']['tier1_direct']['primes'].items():
        writer.writerow(['prime', 'tier1_direct', '', gift, prime, '',
                        'direct constant', '', '', '', 'primes', ''])

    # Prime Atlas - b3
    for prime, formula in atlas['prime_atlas']['tier2_b3']['primes'].items():
        writer.writerow(['prime', 'tier2_b3', '', f'prime_{prime}', prime,
                        formula, 'b3 generated', '', '', '', 'primes', ''])

    # Prime Atlas - H*
    for prime, formula in atlas['prime_atlas']['tier3_H_star']['primes'].items():
        writer.writerow(['prime', 'tier3_H_star', '', f'prime_{prime}', prime,
                        formula, 'H* generated', '', '', '', 'primes', ''])

    # Prime Atlas - E8
    for prime, formula in atlas['prime_atlas']['tier4_E8']['primes'].items():
        writer.writerow(['prime', 'tier4_E8', '', f'prime_{prime}', prime,
                        formula, 'E8 generated', '', '', '', 'primes', ''])

    # Relations
    for section, items in atlas['relations'].items():
        for item in items:
            writer.writerow(['relation', section, item['id'], item['name'], '',
                            item['formula'], '', '', '', item.get('status', ''),
                            item['domain'], item.get('precision', '')])

    # Correspondences - Bernoulli
    for m in atlas['correspondences']['bernoulli']['mappings']:
        interp = '; '.join([f"{k}={v}" for k, v in m['gift_interpretation'].items()])
        writer.writerow(['correspondence', 'bernoulli', '', m['group'],
                        m['bernoulli_index'], f"denom_primes: {m['denom_primes']}",
                        atlas['correspondences']['bernoulli']['description'],
                        '', '', '', 'bernoulli', interp])

    # Correspondences - Fibonacci
    for key, val in atlas['correspondences']['fibonacci']['mappings'].items():
        writer.writerow(['correspondence', 'fibonacci', '', key, val['value'],
                        '', val['gift'], '', '', '', 'fibonacci', ''])

    # Correspondences - Lucas
    for key, val in atlas['correspondences']['lucas']['mappings'].items():
        writer.writerow(['correspondence', 'lucas', '', key, val['value'],
                        '', val['gift'], '', '', '', 'lucas', ''])

    # Correspondences - Fermat
    for key, val in atlas['correspondences']['fermat']['mappings'].items():
        writer.writerow(['correspondence', 'fermat', '', key, val['value'],
                        '', val['gift'], '', '', '', 'fermat', ''])

    return output.getvalue()


def generate_exhaustive_json(atlas: dict) -> str:
    """Generate a single exhaustive JSON with all data in flat and nested formats."""

    # Build a comprehensive JSON structure
    output = {
        "metadata": atlas['metadata'],
        "generated": datetime.now().isoformat(),

        # Flat lists for easy querying
        "flat": {
            "constants": [],
            "observables": [],
            "relations": [],
            "sporadic_groups": [],
            "primes": [],
            "correspondences": []
        },

        # Original nested structure
        "nested": atlas
    }

    # Flatten constants
    for category in ['topology', 'algebra', 'physics', 'derived']:
        for key, val in atlas['constants'].get(category, {}).items():
            output['flat']['constants'].append({
                "id": key,
                "category": category,
                "value": val.get('value', val) if isinstance(val, dict) else val,
                "origin": val.get('origin', '') if isinstance(val, dict) else '',
                "latex": val.get('latex', '') if isinstance(val, dict) else '',
                "formula": val.get('formula', '') if isinstance(val, dict) else ''
            })

    # Flatten sequences
    for seq_name, seq_data in atlas['constants'].get('sequences', {}).items():
        for key, val in seq_data.items():
            output['flat']['constants'].append({
                "id": key,
                "category": f"sequence_{seq_name}",
                "value": val,
                "origin": seq_name,
                "latex": "",
                "formula": ""
            })

    # Flatten observables
    for section, items in atlas['observables'].items():
        for item in items:
            output['flat']['observables'].append({
                "id": item['id'],
                "section": section,
                "name": item['name'],
                "formula": item['formula'],
                "predicted": item['predicted'],
                "experimental": item['experimental'],
                "uncertainty": item.get('uncertainty', ''),
                "deviation_pct": item['deviation_pct'],
                "status": item['status'],
                "unit": item.get('unit', '')
            })

    # Flatten relations
    for section, items in atlas['relations'].items():
        for item in items:
            output['flat']['relations'].append({
                "id": item['id'],
                "section": section,
                "name": item['name'],
                "formula": item['formula'],
                "domain": item['domain'],
                "status": item.get('status', ''),
                "precision": item.get('precision', '')
            })

    # Flatten sporadic groups
    for i, g in enumerate(atlas['sporadic_groups']['groups'], 1):
        output['flat']['sporadic_groups'].append({
            "rank": i,
            "name": g['name'],
            "symbol": g['symbol'],
            "dim_V1": g['dim_V1'],
            "formula": g['formula'],
            "factors": g.get('factors', ''),
            "verified": g['verified']
        })

    # Flatten primes
    for tier, tier_key in [('tier1_direct', 'tier1'), ('tier2_b3', 'tier2'),
                            ('tier3_H_star', 'tier3'), ('tier4_E8', 'tier4')]:
        for prime, formula in atlas['prime_atlas'].get(tier, {}).get('primes', {}).items():
            output['flat']['primes'].append({
                "prime": int(prime),
                "tier": tier_key,
                "formula": formula,
                "generator": atlas['prime_atlas'].get(tier, {}).get('description', '')
            })

    # Flatten correspondences
    for corr_type in ['bernoulli', 'fibonacci', 'lucas', 'fermat']:
        corr_data = atlas['correspondences'].get(corr_type, {})
        if 'mappings' in corr_data:
            if corr_type == 'bernoulli':
                for m in corr_data['mappings']:
                    output['flat']['correspondences'].append({
                        "type": corr_type,
                        "id": m['group'],
                        "rank": m['rank'],
                        "bernoulli_index": m['bernoulli_index'],
                        "primes": m['denom_primes'],
                        "gift_interpretation": m['gift_interpretation']
                    })
            else:
                for key, val in corr_data['mappings'].items():
                    output['flat']['correspondences'].append({
                        "type": corr_type,
                        "id": key,
                        "value": val['value'],
                        "gift": val['gift']
                    })

    # Add summary stats
    output['stats'] = {
        "total_constants": len(output['flat']['constants']),
        "total_observables": len(output['flat']['observables']),
        "total_relations": len(output['flat']['relations']),
        "total_sporadic_groups": len(output['flat']['sporadic_groups']),
        "total_primes": len(output['flat']['primes']),
        "total_correspondences": len(output['flat']['correspondences'])
    }

    return json.dumps(output, indent=2, ensure_ascii=False)


def generate_index_md(atlas: dict) -> str:
    """Generate README.md index for the atlas."""
    meta = atlas['metadata']

    return f"""# GIFT Atlas

*Unified source of truth for GIFT constants, relations, and correspondences*

**Version**: {meta['version']}
**Date**: {meta['date']}
**Total Relations**: {meta['total_relations']}
**Total Observables**: {meta['total_observables']}

---

## Structure

| File | Description |
|------|-------------|
| [GIFT_ATLAS.yaml](GIFT_ATLAS.yaml) | Master data file (source of truth) |
| [generated/CONSTANTS.md](generated/CONSTANTS.md) | All fundamental constants |
| [generated/OBSERVABLES.md](generated/OBSERVABLES.md) | Physical predictions (39 total) |
| [generated/CORRESPONDENCES.md](generated/CORRESPONDENCES.md) | Bernoulli, Fibonacci, Lucas, Fermat |
| [generated/SPORADIC_GROUPS.md](generated/SPORADIC_GROUPS.md) | Moonshine correspondence (26 groups) |
| [generated/PRIME_ATLAS.md](generated/PRIME_ATLAS.md) | Prime expressibility (46 primes) |
| [generated/RELATIONS.md](generated/RELATIONS.md) | Master relation catalog |
| [generated/GIFT_ATLAS.csv](generated/GIFT_ATLAS.csv) | Exhaustive CSV export |
| [generated/GIFT_ATLAS.json](generated/GIFT_ATLAS.json) | Exhaustive JSON export |

---

## Quick Stats

- **Constants**: ~50 fundamental values
- **Observables**: 39 physical predictions
- **Relations**: ~210 mathematical identities
- **Sporadic groups**: 26/26 with GIFT dimensions
- **Primes < 200**: 46/46 expressible (100%)

---

## Regenerating

```bash
python generate_atlas.py
```

This will regenerate all markdown files from the YAML source.
"""


def main():
    """Main entry point."""
    print("Loading GIFT_ATLAS.yaml...")
    atlas = load_atlas()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Generate all markdown files
    generators = [
        ("CONSTANTS.md", generate_constants_md),
        ("OBSERVABLES.md", generate_observables_md),
        ("CORRESPONDENCES.md", generate_correspondences_md),
        ("SPORADIC_GROUPS.md", generate_sporadic_md),
        ("PRIME_ATLAS.md", generate_prime_atlas_md),
        ("RELATIONS.md", generate_relations_md),
    ]

    for filename, generator in generators:
        print(f"Generating {filename}...")
        content = generator(atlas)
        output_path = OUTPUT_DIR / filename
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"  -> {output_path}")

    # Generate exhaustive CSV
    print("Generating GIFT_ATLAS.csv...")
    csv_content = generate_exhaustive_csv(atlas)
    csv_path = OUTPUT_DIR / "GIFT_ATLAS.csv"
    with open(csv_path, 'w', newline='') as f:
        f.write(csv_content)
    print(f"  -> {csv_path}")

    # Generate exhaustive JSON
    print("Generating GIFT_ATLAS.json...")
    json_content = generate_exhaustive_json(atlas)
    json_path = OUTPUT_DIR / "GIFT_ATLAS.json"
    with open(json_path, 'w') as f:
        f.write(json_content)
    print(f"  -> {json_path}")

    # Generate index
    print("Generating README.md...")
    index_content = generate_index_md(atlas)
    index_path = Path(__file__).parent / "README.md"
    with open(index_path, 'w') as f:
        f.write(index_content)
    print(f"  -> {index_path}")

    print("\nDone! All files generated.")


if __name__ == "__main__":
    main()
