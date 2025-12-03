# G2_ML v1.9b - Hodge Pure (Fixed Wedge)

**Fixes from v1.9:**
- Proper wedge product with 105 Levi-Civita terms
- Text-based training output (no matplotlib refresh spam)
- Correct JSON serialization (no numpy/torch bools)
- Tau division by zero handling
- Added 35/42 detection (b3_local/b3_global split)

## Key Finding from v1.9

The spectrum showed **35 non-zero modes** (not 43), matching b3_local = C(7,3).

This means the Yukawa couples only the **local** H3 modes. The 42 global modes (from TCS topology) don't participate because our network generates them artificially.

## Training Output Format

Every 100 epochs:
```
 Epoch |       Loss |      Ortho |     Closed |       Best
----------------------------------------------------------
   100 |   1.23e-02 |   4.56e-03 |   7.89e-03 |   1.23e-02
```

For H3, adds G2 compatibility column.

## Files

- `K7_Hodge_Pure_v1_9b.ipynb` - All-in-one Colab notebook

## Expected Output

With proper wedge:
- n_visible ~ 35 (b3_local)
- Gap at 34->35 should be largest
- Modes 35-76 near zero

For 43/77 to emerge, we need real TCS structure in global modes.
