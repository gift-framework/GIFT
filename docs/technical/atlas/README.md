# GIFT Atlas

*Unified source of truth for GIFT constants, relations, and correspondences*

**Version**: 1.0
**Date**: 2025-12-11
**Total Relations**: 210
**Total Observables**: 39

---

## Structure

| File | Description |
|------|-------------|
| [GIFT_ATLAS.yaml](GIFT_ATLAS.yaml) | Master data file (source of truth) |
| [generated/CONSTANTS.md](generated/CONSTANTS.md) | All fundamental constants |
| [generated/OBSERVABLES.md](generated/OBSERVABLES.md) | Physical predictions (39 total) |
| [generated/CORRESPONDENCES.md](generated/CORRESPONDENCES.md) | Bernoulli, Fibonacci, Lucas, Fermat |
| [generated/SPORADIC_GROUPS.md](generated/SPORADIC_GROUPS.md) | Moonshine correspondence (26 groups) |
| [generated/PRIME_ATLAS.md](generated/PRIME_ATLAS.md) | Prime expressibility (46 primes) |
| [generated/RELATIONS.md](generated/RELATIONS.md) | Master relation catalog |
| [generated/GIFT_ATLAS.csv](generated/GIFT_ATLAS.csv) | Exhaustive CSV export |
| [generated/GIFT_ATLAS.json](generated/GIFT_ATLAS.json) | Exhaustive JSON export |

---

## Quick Stats

- **Constants**: ~50 fundamental values
- **Observables**: 39 physical predictions
- **Relations**: ~210 mathematical identities
- **Sporadic groups**: 26/26 with GIFT dimensions
- **Primes < 200**: 46/46 expressible (100%)

---

## Regenerating

```bash
python generate_atlas.py
```

This will regenerate all markdown files from the YAML source.
