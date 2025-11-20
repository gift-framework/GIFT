"""
Final fix: ensure ALL torch.load calls have weights_only=False
"""
import json
import re

with open('K7_Torsion_v1_0c.ipynb') as f:
    nb = json.load(f)

print('Fixing ALL torch.load calls...\n')

fixed_cells = []

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code' or not cell['source']:
        continue

    source = cell['source'][0]
    original = source

    # Find ALL torch.load patterns and fix them
    patterns = [
        (r'torch\.load\(([^,)]+)\)(?!\s*,\s*weights_only)', r'torch.load(\1, weights_only=False)'),
        (r'torch\.load\(([^,)]+),\s*weights_only=True\)', r'torch.load(\1, weights_only=False)'),
    ]

    for pattern, replacement in patterns:
        source = re.sub(pattern, replacement, source)

    # Also catch edge cases with newlines
    if 'torch.load(' in source and 'weights_only=False' not in source:
        # Manual fix for multi-line torch.load
        lines = source.split('\n')
        for j, line in enumerate(lines):
            if 'torch.load(' in line and ')' in line and 'weights_only' not in line:
                # Simple case: torch.load(something)
                line = line.replace('torch.load(', 'torch.load(').replace(')', ', weights_only=False)')
                lines[j] = line
        source = '\n'.join(lines)

    if source != original:
        cell['source'] = [source]
        cell_id = cell.get('id', f'cell_{i}')
        fixed_cells.append(cell_id)
        print(f'Fixed cell {i} ({cell_id})')

# Special manual fix for checkpoint_manager if still broken
for cell in nb['cells']:
    if cell.get('id') == 'checkpoint_manager':
        source = cell['source'][0]

        # Find the load method and fix it explicitly
        if 'def load(self' in source and 'torch.load(path, weights_only=False)' not in source:
            source = source.replace(
                'checkpoint = torch.load(path)',
                'checkpoint = torch.load(path, weights_only=False)'
            )
            source = source.replace(
                'checkpoint = torch.load(path, weights_only=True)',
                'checkpoint = torch.load(path, weights_only=False)'
            )
            cell['source'] = [source]
            print('Manually fixed checkpoint_manager.load()')
            if 'checkpoint_manager' not in fixed_cells:
                fixed_cells.append('checkpoint_manager')

# Same for load_helpers
for cell in nb['cells']:
    if cell.get('id') == 'load_helpers':
        source = cell['source'][0]

        if 'torch.load(ckpt_path' in source and 'weights_only=False' not in source:
            source = source.replace(
                'final_ckpt = torch.load(ckpt_path)',
                'final_ckpt = torch.load(ckpt_path, weights_only=False)'
            )
            cell['source'] = [source]
            print('Manually fixed load_helpers')
            if 'load_helpers' not in fixed_cells:
                fixed_cells.append('load_helpers')

# Save
with open('K7_Torsion_v1_0c.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f'\nTotal cells fixed: {len(fixed_cells)}')
print('Fixed cells:', ', '.join(fixed_cells))

# Verify
print('\nVerifying all torch.load calls...')
issues = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code' or not cell['source']:
        continue
    source = cell['source'][0]
    if 'torch.load(' in source:
        cell_id = cell.get('id', f'cell_{i}')
        if 'weights_only=False' in source:
            print(f'  [OK] {cell_id}')
        else:
            print(f'  [FAIL] {cell_id} - still missing weights_only=False')
            issues.append(cell_id)

if issues:
    print(f'\nWARNING: {len(issues)} cells still have issues!')
else:
    print('\n✓ All torch.load calls now have weights_only=False')
