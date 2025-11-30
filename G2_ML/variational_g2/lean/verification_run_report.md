# det(g) interval verification run

## Command
```
python G2_ML/variational_g2/lean/verify_det_g.py --direct --output G2_ML/variational_g2/lean/verification_result.json --tolerance 1e-6
```

## Outcome
- **Point evaluation:** VERIFIED with det(g) interval `[2.03125, 2.03125]`, matching the target `65/32` within `1e-6` tolerance.
- **With uncertainty (ε = 0.001):** PARTIAL; the interval `[2.00438, 2.05845]` overlaps the tolerance band but is not fully contained in `[2.02125, 2.04125]`.
- Machine-readable certificate written to `G2_ML/variational_g2/lean/verification_result.json`.

## Notes
This run exercises the deterministic "direct" verifier (no neural network weights). The uncertain pass remains partial, so a tighter uncertainty radius or refined enclosure would be needed to upgrade it to a full proof.
