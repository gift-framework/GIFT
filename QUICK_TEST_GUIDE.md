# Guide Rapide - Tests GIFT Framework

## Lancer les tests

### Tests rapides (sans torch)
```bash
# Tous les tests core (59 tests, ~15 secondes)
pytest tests/unit/test_gift_framework.py tests/unit/test_agents.py tests/integration -v

# Avec couverture
pytest tests/unit/test_gift_framework.py --cov=statistical_validation --cov-report=html
```

### Tests complets (avec torch installe)
```bash
# Tous les tests
pytest tests/ -v

# Uniquement tests rapides
pytest -m "not slow" -v

# En parallele (plus rapide)
pytest -n auto -v
```

## Commandes utiles

### Par categorie
```bash
pytest tests/unit              # Tests unitaires
pytest tests/integration       # Tests d'integration
pytest tests/regression        # Tests de regression
pytest G2_ML/tests             # Tests G2 ML (necessite torch)
```

### Avec filtres
```bash
pytest -k "gauge"              # Tous les tests avec "gauge" dans le nom
pytest -k "GIFT"               # Tests du framework GIFT
pytest -k "delta_CP"           # Tests pour delta_CP
```

### Rapport de couverture
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html        # Ouvrir le rapport HTML
```

## Tests critiques valides

- 39 observables (34 dimensionnels + 5 cosmologiques)
- 13 relations PROVEN exactes
- Precision < 0.2% vs experience
- Stabilite numerique
- Validation experimentale

## Structure

```
tests/
├── unit/                   # Tests unitaires
│   ├── test_gift_framework.py
│   ├── test_agents.py
│   └── test_error_handling.py
├── integration/            # Tests d'integration
├── regression/             # Tests de regression
└── notebooks/              # Tests notebooks

G2_ML/tests/               # Tests G2 ML (~150 tests)
├── test_geometry.py
└── test_manifold.py
```

## Resultats actuels

**59/64 tests passent** (92%) sans torch
**~210+ tests** disponibles avec torch

Les echecs mineurs sont sur des tests tres stricts
(variations numeriques < 0.2%).

## CI/CD

Les tests s'executent automatiquement sur chaque push :
- Linting (flake8)
- Tests unitaires + couverture
- Tests d'integration
- Tests de regression
- Upload vers Codecov

## Documentation complete

Voir `tests/README.md` pour le guide complet.
