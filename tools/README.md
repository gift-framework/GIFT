# GIFT Framework Tools

Utility scripts for repository maintenance and validation.

## Available Tools

### `check_pdf_sync.py`

Verify synchronization between PDF files and their markdown sources.

**Usage**:
```bash
python3 tools/check_pdf_sync.py
```

**Output**:
- Console report showing sync status of all PDFs
- Creates `publications/pdf/PDF_SYNC_STATUS.md` with detailed status

**Checks**:
- ✓ PDFs in sync with markdown sources
- ⚠ PDFs out of sync (markdown newer than PDF)
- ✗ Missing PDFs or markdown files

**When to run**:
- After updating any publication markdown files
- Before creating a new release
- When verifying repository integrity

## Adding New Tools

Place utility scripts in this directory following these conventions:

1. **Naming**: Use descriptive names (e.g., `check_*.py`, `validate_*.py`)
2. **Documentation**: Include docstrings and usage examples
3. **Shebang**: Start with `#!/usr/bin/env python3`
4. **Executable**: Make scripts executable (`chmod +x script.py`)
5. **README**: Document in this file

## Planned Tools

Future utilities that could be added:

- `validate_links.py` - Check all markdown links
- `check_citations.py` - Verify citation consistency
- `generate_pdfs.py` - Batch regenerate all PDFs
- `check_notation.py` - Verify notation consistency across documents
- `update_changelog.py` - Automated CHANGELOG generation

---

**Directory**: `/tools`
**Purpose**: Repository maintenance and validation
**Last updated**: 2025-11-16
