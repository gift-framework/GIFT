"""
Fix cell 15 grid_construction f-string errors
"""
import json

with open('K7_Torsion_v1_0c.ipynb') as f:
    nb = json.load(f)

# Find and fix cell 15
for i, cell in enumerate(nb['cells']):
    if cell.get('id') != 'grid_construction':
        continue

    source = cell.get('source', '')
    if isinstance(source, list):
        source = ''.join(source)

    # Rewrite the entire problematic section
    # Find the print statements section
    lines = source.split('\n')
    new_lines = []

    for line in lines:
        # Replace each problematic print
        if "print(f'Grid shape" in line:
            new_lines.append('print(f"Grid shape: {grid[\'shape\']}")')
        elif "print(f'Total nodes" in line:
            new_lines.append('print(f"Total nodes: {grid[\'n_nodes\']:,}")')
        elif "print(f'e range" in line:
            new_lines.append('print(f"e range: [{grid[\'e\'][0]:.3f}, {grid[\'e\'][-1]:.3f}]")')
        elif "range: [{grid" in line and "pi" in line:
            # pi range line
            new_lines.append('print(f"pi range: [{grid[\'pi\'][0]:.3f}, {grid[\'pi\'][-1]:.3f}]")')
        elif "range: [{grid" in line and "phi" in line:
            # phi range line
            new_lines.append('print(f"phi range: [{grid[\'phi\'][0]:.3f}, {grid[\'phi\'][-1]:.3f}]")')
        elif "Spacings: de=" in line:
            # Multi-line spacing, replace all 3 lines
            new_lines.append('print(f"Spacings: de={grid[\'spacings\'][\'de\']:.4f}, " \\')
        elif "dpi" in line and ":.4f" in line and len(new_lines) > 0 and "Spacings" in new_lines[-1]:
            new_lines.append('      f"dpi={grid[\'spacings\'][\'dpi\']:.4f}, " \\')
        elif "dphi" in line and ":.4f" in line and len(new_lines) > 0:
            new_lines.append('      f"dphi={grid[\'spacings\'][\'dphi\']:.4f}")')
        else:
            new_lines.append(line)

    cell['source'] = '\n'.join(new_lines)
    break

# Save
with open('K7_Torsion_v1_0c.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

with open('fix_cell_15.log', 'w') as f:
    f.write('Cell 15 fixed\n')
