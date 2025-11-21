"""
Comprehensive fix for all notebook issues
"""
import json
import re

def fix_unterminated_fstrings(source):
    """Fix unterminated f-strings by adding missing closing quotes."""
    lines = source.split('\n')
    fixed_lines = []

    for line in lines:
        # Pattern: print(f'...{var}) but missing closing quote
        # Should be: print(f'...{var}')
        if 'print(f' in line:
            # Count opening and closing quotes
            single_quotes = line.count("'")
            double_quotes = line.count('"')

            # If inside an f-string (starts with f'), check if balanced
            if "print(f'" in line or 'print(f"' in line:
                # Find the last closing paren
                if line.rstrip().endswith(')') and not line.rstrip().endswith(')'):
                    # Check if there's a quote before the paren
                    if line.rstrip()[-2] not in ['"', "'"]:
                        # Add missing quote
                        if "print(f'" in line:
                            line = line.rstrip()[:-1] + "')"
                        else:
                            line = line.rstrip()[:-1] + '")'

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_indentation(source):
    """Fix unexpected indentation issues."""
    lines = source.split('\n')
    fixed_lines = []
    prev_indent = 0

    for i, line in enumerate(lines):
        if not line.strip():
            fixed_lines.append(line)
            continue

        # Measure current indentation
        current_indent = len(line) - len(line.lstrip())

        # Check for orphaned indented lines
        if i > 0 and current_indent > 0:
            prev_line = lines[i-1].strip()
            # If previous line doesn't end with : or \, this indent might be wrong
            if prev_line and not prev_line.endswith((':',  '\\', ',')):
                # Check if it's part of a multi-line statement
                if not any(keyword in prev_line for keyword in ['def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'with ']):
                    # Likely an indentation error - dedent to match previous
                    if i > 1:
                        line = line.lstrip()
                        # Add back the previous indentation level
                        prev_prev = lines[i-2] if i > 1 else ''
                        prev_prev_indent = len(prev_prev) - len(prev_prev.lstrip())
                        line = ' ' * prev_prev_indent + line

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

# Load notebook
print('Loading notebook...')
with open('K7_Torsion_v1_0c_backup.ipynb') as f:  # Start from backup
    nb = json.load(f)

print('Applying comprehensive fixes...')

# Process each code cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue

    cell_id = cell.get('id', f'cell_{i}')
    source = cell['source'][0] if cell['source'] else ''

    if not source.strip():
        continue

    original_source = source

    # Apply fixes
    source = fix_indentation(source)
    source = fix_unterminated_fstrings(source)

    # Specific fixes
    # torch.load
    if 'torch.load' in source and 'weights_only=False' not in source:
        source = re.sub(
            r'torch\.load\(([^)]+)\)(?!\s*,)',
            r'torch.load(\1, weights_only=False)',
            source
        )

    # matplotlib style
    if "plt.style.use('seaborn-v0_8-darkgrid')" in source:
        source = source.replace(
            "plt.style.use('seaborn-v0_8-darkgrid')",
            "try:\n    plt.style.use('seaborn-v0_8-darkgrid')\nexcept:\n    pass  # Colab compatibility"
        )

    if source != original_source:
        cell['source'] = [source]
        print(f'  Fixed cell {i} ({cell_id})')

# Save
output_path = 'K7_Torsion_v1_0c.ipynb'
print(f'\nSaving to {output_path}...')
with open(output_path, 'w') as f:
    json.dump(nb, f, indent=1)

print('Done!')

# Validate
print('\nValidating syntax...')
import ast
errors = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = cell['source'][0] if cell['source'] else ''
    if not source.strip():
        continue
    try:
        ast.parse(source)
    except SyntaxError as e:
        errors += 1
        cell_id = cell.get('id', f'cell_{i}')
        print(f'  Error in cell {i} ({cell_id}): {e.msg}')

if errors == 0:
    print('  All cells valid!')
else:
    print(f'  Found {errors} errors')
