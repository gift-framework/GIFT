"""
Fix all syntax and compatibility issues in K7_Torsion_v1_0c.ipynb
"""
import json
import re
from pathlib import Path

def fix_fstring_syntax(source):
    """Fix common f-string syntax errors."""
    # Fix: {var["key"]'} -> {var["key"]}
    source = re.sub(r'\{([^}]+)\}\'(?=\))', r'{\1}', source)

    # Fix: print(f'... (more specific patterns)
    lines = source.split('\n')
    fixed_lines = []

    for line in lines:
        # Pattern: print(f'..{grid["key"]'}')
        if 'print(f' in line:
            # Count quotes to detect imbalance
            in_fstring = False
            fixed_line = []
            i = 0
            while i < len(line):
                char = line[i]

                # Detect f-string start
                if i < len(line) - 1 and char == 'f' and line[i+1] in ['"', "'"]:
                    in_fstring = True
                    fixed_line.append(char)
                    i += 1
                    continue

                # Inside f-string, check for bad patterns
                if in_fstring and char == '}':
                    # Check if next char is quote before closing paren
                    if i < len(line) - 2 and line[i+1] == "'" and line[i+2] == ')':
                        fixed_line.append(char)
                        i += 2  # Skip the extra quote
                        continue

                fixed_line.append(char)
                i += 1

            line = ''.join(fixed_line)

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_torch_load(source):
    """Add weights_only=False to torch.load calls."""
    # Pattern: torch.load(path)
    source = re.sub(
        r'torch\.load\(([^)]+)\)(?!\s*,\s*weights_only)',
        r'torch.load(\1, weights_only=False)',
        source
    )
    return source

def fix_matplotlib_style(source):
    """Fix deprecated matplotlib style."""
    # seaborn-v0_8-darkgrid -> seaborn-v0_8 or just remove
    source = source.replace("plt.style.use('seaborn-v0_8-darkgrid')",
                           "# plt.style.use('seaborn-v0_8-darkgrid')  # Colab compatibility")
    return source

def add_colab_compatibility(source, cell_id):
    """Add Colab-specific fixes."""
    # For imports cell, add fallback for seaborn style
    if cell_id == 'imports':
        if 'plt.style.use' in source:
            source = source.replace(
                "plt.style.use('seaborn-v0_8-darkgrid')",
                "try:\n    plt.style.use('seaborn-v0_8-darkgrid')\nexcept:\n    plt.style.use('default')  # Colab fallback"
            )
    return source

def fix_indentation(source):
    """Fix common indentation issues."""
    lines = source.split('\n')
    fixed_lines = []

    for line in lines:
        # Don't modify empty lines or comments
        if not line.strip() or line.strip().startswith('#'):
            fixed_lines.append(line)
            continue

        # Check for mixed tabs/spaces (convert to spaces)
        if '\t' in line:
            # Replace tabs with 4 spaces
            line = line.replace('\t', '    ')

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_cell_source(source, cell_id):
    """Apply all fixes to a cell's source."""
    source = fix_fstring_syntax(source)
    source = fix_torch_load(source)
    source = fix_matplotlib_style(source)
    source = add_colab_compatibility(source, cell_id)
    source = fix_indentation(source)
    return source

# Load notebook
nb_path = Path('K7_Torsion_v1_0c.ipynb')
backup_path = Path('K7_Torsion_v1_0c_backup.ipynb')

print('Loading notebook...')
with open(nb_path) as f:
    nb = json.load(f)

# Create backup
print(f'Creating backup: {backup_path}')
with open(backup_path, 'w') as f:
    json.dump(nb, f, indent=1)

# Fix each cell
print('Fixing cells...')
fixes_applied = 0

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue

    original_source = ''.join(cell.get('source', []))
    cell_id = cell.get('id', f'cell_{i}')

    fixed_source = fix_cell_source(original_source, cell_id)

    if fixed_source != original_source:
        # Update cell source (keep as list of lines)
        cell['source'] = [fixed_source]
        fixes_applied += 1
        print(f'  Fixed cell {i} ({cell_id})')

# Save fixed notebook
print(f'\nApplied {fixes_applied} fixes')
print(f'Saving to {nb_path}...')

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print('Done!')
print(f'\nBackup saved as: {backup_path}')
print(f'Fixed notebook: {nb_path}')
