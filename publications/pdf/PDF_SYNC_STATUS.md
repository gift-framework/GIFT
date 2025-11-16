# PDF Synchronization Status

**Last checked**: 2025-11-16 16:34:46

## Status Summary

| PDF | Status | PDF Date | Markdown Date |
|-----|--------|----------|---------------|
| Supp_A.pdf | ✓ In sync | 2025-11-16 10:04:18 | 2025-11-16 10:04:18 |
| Supp_B.pdf | ✓ In sync | 2025-11-16 10:04:18 | 2025-11-16 10:04:18 |
| Supp_C.pdf | ✓ In sync | 2025-11-16 10:04:18 | 2025-11-16 10:04:18 |
| Supp_D.pdf | ✓ In sync | 2025-11-16 10:04:18 | 2025-11-16 10:04:18 |
| Supp_E.pdf | ✓ In sync | 2025-11-16 10:04:18 | 2025-11-16 10:04:18 |
| Supp_F.pdf | ✓ In sync | 2025-11-16 10:04:18 | 2025-11-16 10:04:18 |
| gift-main.pdf | ✓ In sync | 2025-11-16 10:04:18 | 2025-11-16 10:04:18 |
| gift_extensions.pdf | ✓ In sync | 2025-11-16 10:04:18 | 2025-11-16 10:04:18 |

## How to Regenerate PDFs

If PDFs are out of sync, regenerate using:

```bash
# Using Pandoc
pandoc publications/gift_main.md -o publications/pdf/gift-main.pdf
pandoc publications/supplements/A_math_foundations.md -o publications/pdf/Supp_A.pdf
# ... repeat for other files
```

Or use batch script:

```bash
cd publications
for md in *.md; do
    pandoc "$md" -o "pdf/${md%.md}.pdf"
done
```
