# LMFDB Access Guide for Dirichlet L-Function Zeros

**Status**: REFERENCE DOCUMENT
**Date**: February 2026

---

## 1. Key Finding

**LMFDB does NOT provide precomputed Dirichlet L-function zeros** like it does for Riemann zeta (103.8 billion zeros available).

For Dirichlet L-functions, you must **compute zeros yourself** using the character data from LMFDB.

---

## 2. LMFDB API (Now Supports JSON)

The LMFDB API now supports JSON output (previously HTML only).

### API Endpoint

```
https://www.lmfdb.org/api/
```

### Query Examples

```bash
# Get L-functions with conductor 100
curl "https://www.lmfdb.org/api/lfunc_lfunctions/?conductor=100&_format=json"

# Get Dirichlet characters of modulus 13
curl "https://www.lmfdb.org/api/char_dirichlet/?modulus=13&_format=json"
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `_format` | Output format: `json`, `yaml`, `html` |
| `conductor` | Filter by conductor |
| `degree` | 1 for Dirichlet L-functions |
| `_limit` | Max results (default 100, max 10000) |

---

## 3. Python Access: lmfdb-lite

### Installation

```bash
pip install "lmfdb-lite[pgsource] @ git+https://github.com/roed314/lmfdb-lite.git"
```

### Usage

```python
from lmfdb_lite import db

# Search for Dirichlet characters
chars = db.char_dirichlet.search({"conductor": 13})

# Get L-function metadata
lfuncs = db.lfunc_lfunctions.search({"degree": 1, "conductor": 13})

# Count results
count = db.lfunc_lfunctions.count({"conductor": 13})
```

---

## 4. Computing Dirichlet L-Function Zeros

Since LMFDB doesn't have precomputed zeros, use these libraries:

### Option A: SageMath (Recommended for Exploration)

```python
from sage.all import *

# Create Dirichlet character of conductor q
G = DirichletGroup(q)
chi = G[j]  # j-th character

# Create L-function
L = Lfunction_from_character(chi, type="int")

# Find zeros in range [a, b]
zeros = L.find_zeros(0, 1000, step_size=0.1)
```

### Option B: mpmath (Pure Python)

```python
import mpmath as mp
mp.mp.dps = 50  # 50 decimal places

def dirichlet_L(s, chi_values, q):
    """Evaluate L(s, chi) for Dirichlet character."""
    result = mp.mpf(0)
    for n in range(1, 10000):
        chi_n = chi_values[n % q]
        result += chi_n / mp.power(n, s)
    return result

# Find zeros by searching for sign changes
```

### Option C: Arb Library (Rigorous, Fastest)

For high-precision rigorous computation:

```c
#include "acb_dirichlet.h"

// Use acb_dirichlet_hardy_z for zero-finding
// Zeros are where Hardy Z-function changes sign
```

Python binding via `python-flint`:

```bash
pip install python-flint
```

---

## 5. Riemann Zeta Zeros (Available)

For comparison, LMFDB has extensive Riemann zeta zero data:

### Online Access

```
https://www.lmfdb.org/zeros/zeta/
```

### Bulk Download

```
https://beta.lmfdb.org/riemann-zeta-zeros/
```

### SageMath Access

```python
from sage.databases.odlyzko import zeta_zeros
zeros = zeta_zeros()  # First 2,001,052 zeros
```

### Odlyzko Tables

```
https://www-users.cs.umn.edu/~odlyzko/zeta_tables/
```

---

## 6. Recommended Workflow for GIFT Test

### Step 1: Get Characters from LMFDB

```python
from lmfdb_lite import db

GIFT_CONDUCTORS = [6, 7, 8, 11, 13, 14, 17, 21, 27, 77, 99]

for q in GIFT_CONDUCTORS:
    chars = list(db.char_dirichlet.search({"conductor": q, "primitive": True}))
    print(f"Conductor {q}: {len(chars)} primitive characters")
```

### Step 2: Compute Zeros (SageMath Colab)

```python
# In Google Colab with SageMath kernel
zeros_by_conductor = {}

for q in GIFT_CONDUCTORS:
    G = DirichletGroup(q)
    for chi in G:
        if chi.is_primitive():
            L = Lfunction_from_character(chi)
            zeros = L.find_zeros(0, 500, 0.05)
            zeros_by_conductor.setdefault(q, []).extend(zeros)
```

### Step 3: Run Selectivity Test

```python
# Use the fit_recurrence function from conductor_selectivity_test.py
from conductor_selectivity_test import fit_recurrence, GIFT_LAGS

for q, zeros in zeros_by_conductor.items():
    coeffs, error = fit_recurrence(zeros, GIFT_LAGS)
    R = compute_fibonacci_constraint(coeffs, GIFT_LAGS)
    print(f"q={q}: R={R:.4f}")
```

---

## 7. Alternative: Approximate via Riemann Zeros

If computing Dirichlet zeros is too slow, the current proxy approach is:

1. Use mpmath.zetazero() for Riemann zeros
2. Apply conductor-dependent windowing
3. Apply conductor-dependent scaling

This is what `Conductor_Selectivity_mpmath.ipynb` does.

**Limitation**: Not real L-function zeros, just scaled ζ(s) zeros.

---

## 8. Resources

| Resource | URL |
|----------|-----|
| LMFDB Main | https://www.lmfdb.org/ |
| LMFDB API | https://www.lmfdb.org/api/ |
| lmfdb-lite | https://github.com/roed314/lmfdb-lite |
| SageMath L-functions | https://doc.sagemath.org/html/en/reference/lfunctions/ |
| Arb Library | https://arblib.org/acb_dirichlet.html |
| Odlyzko Tables | https://www-users.cs.umn.edu/~odlyzko/zeta_tables/ |

---

*GIFT Framework — Riemann Research*
*February 2026*
