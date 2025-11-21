"""
Manual fixes for specific problematic cells
"""
import json

with open('K7_Torsion_v1_0c.ipynb') as f:
    nb = json.load(f)

# Find and fix specific cells by ID
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue

    cell_id = cell.get('id', '')
    source = cell['source'][0] if cell['source'] else ''

    # Fix grid_construction cell
    if cell_id == 'grid_construction':
        print(f'Fixing {cell_id}...')
        # Rewrite the problematic print statements
        new_source = source.split('\n')
        for i, line in enumerate(new_source):
            if line.strip().startswith("print(f'Grid shape"):
                new_source[i] = 'print(f"Grid shape: {grid[\'shape\']}")'
            elif line.strip().startswith("print(f'Total nodes"):
                new_source[i] = 'print(f"Total nodes: {grid[\'n_nodes\']:,}")'
            elif line.strip().startswith("print(f'e range"):
                new_source[i] = 'print(f"e range: [{grid[\'e\'][0]:.3f}, {grid[\'e\'][-1]:.3f}]")'
            elif line.strip().startswith("print(f'π range"):
                new_source[i] = 'print(f"π range: [{grid[\'pi\'][0]:.3f}, {grid[\'pi\'][-1]:.3f}]")'
            elif line.strip().startswith("print(f'φ range"):
                new_source[i] = 'print(f"φ range: [{grid[\'phi\'][0]:.3f}, {grid[\'phi\'][-1]:.3f}]")'
            elif 'Spacings: de=' in line:
                # Multi-line print, rebuild it
                new_source[i] = 'print(f"Spacings: de={grid[\'spacings\'][\'de\']:.4f}, "'
                new_source[i+1] = '      f"dπ={grid[\'spacings\'][\'dpi\']:.4f}, "'
                new_source[i+2] = '      f"dφ={grid[\'spacings\'][\'dphi\']:.4f}")'

        cell['source'] = ['\n'.join(new_source)]

    # Fix checkpoint_manager - add weights_only=False
    if cell_id == 'checkpoint_manager':
        print(f'Fixing {cell_id}...')
        if 'torch.load(path, weights_only=False)' not in source:
            source = source.replace(
                'checkpoint = torch.load(path, weights_only=False)',
                'checkpoint = torch.load(path, weights_only=False)'
            )
            if 'checkpoint = torch.load(path)' in source:
                source = source.replace(
                    'checkpoint = torch.load(path)',
                    'checkpoint = torch.load(path, weights_only=False)'
                )
                cell['source'] = [source]

    # Fix load_helpers
    if cell_id == 'load_helpers':
        print(f'Fixing {cell_id}...')
        if 'torch.load(ckpt_path, weights_only=False)' not in source:
            source = source.replace(
                'final_ckpt = torch.load(ckpt_path)',
                'final_ckpt = torch.load(ckpt_path, weights_only=False)'
            )
            cell['source'] = [source]

# Save
with open('K7_Torsion_v1_0c.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('\nManual fixes complete!')
