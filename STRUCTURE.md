# Repository Structure

This repository contains the theoretical documentation for GIFT v2.3.

## Directory Layout

```
GIFT/
├── publications/                     # Theoretical documents
│   ├── README.md                    # Overview and reading guide
│   ├── markdown/                    # Main documents
│   │   ├── gift_2_3_main.md        # Core paper
│   │   └── S1-S7 supplements       # Mathematical details
│   ├── references/                  # Quick reference documents
│   ├── tex/                         # LaTeX sources
│   └── pdf/                         # Generated PDFs
│
└── docs/                            # Supporting documentation
    ├── FAQ.md                       # Common questions
    ├── GLOSSARY.md                  # Technical terms
    └── PHILOSOPHY.md                # Foundational perspective
```

## Quick Navigation

| Looking for... | Go to |
|----------------|-------|
| Framework overview | `README.md` |
| Complete theory | `publications/markdown/gift_2_3_main.md` |
| All 39 observables | `publications/references/GIFT_v23_Observable_Reference.md` |
| Proofs | `publications/markdown/S4_complete_derivations_v23.md` |
| Formal verification | [gift-framework/core](https://github.com/gift-framework/core) |
| Technical definitions | `docs/GLOSSARY.md` |

## Related Repositories

| Repository | Content |
|------------|---------|
| [gift-framework/core](https://github.com/gift-framework/core) | Formal proofs (Lean 4 + Coq), K₇ metric pipeline, validation code |
