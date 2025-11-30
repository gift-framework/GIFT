# Professional Visualization Outputs

This directory stores the artifacts produced by the professional-grade visualization suite (`assets.visualizations.pro_package`). The figures are *not* checked into version control, but we keep this README so that the directory is always present and documented.

## Expected files

- `e8_root_system_pro.(html|png)`
- `dimensional_flow_pro.(html|png)`
- `precision_matrix_pro.(html|png)`

Additional figures can be added via the CLI (`python -m assets.visualizations.pro_package.cli`) or by running `python scripts/generate_pro_visuals.py`.

## Regeneration

```bash
python scripts/generate_pro_visuals.py
```

The command reads `assets/visualizations/pro_package/config.json` for shared styling instructions.











