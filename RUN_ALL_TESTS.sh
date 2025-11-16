#!/bin/bash
# Script pour exécuter tous les tests GIFT

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        GIFT Framework - Exécution Tests Complets          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Vérifier torch
echo "🔍 Vérification des dépendances..."
if python -c "import torch" 2>/dev/null; then
    echo "✓ Torch installé"
    TORCH_OK=1
else
    echo "⚠ Torch non installé - Installation recommandée:"
    echo "  pip install torch"
    TORCH_OK=0
fi
echo ""

# Tests Core (toujours disponibles)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Lancement des tests Core (sans torch)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pytest tests/unit/test_gift_framework.py tests/unit/test_agents.py \
       tests/integration tests/regression \
       -v --tb=short -m "not slow" \
       --cov=. --cov-report=term-missing --cov-report=html
CORE_EXIT=$?
echo ""

# Tests G2 ML (si torch disponible)
if [ $TORCH_OK -eq 1 ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Lancement des tests G2 ML (avec torch)..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    pytest G2_ML/tests tests/unit/test_error_handling.py \
           -v --tb=short -m "not slow"
    G2_EXIT=$?
    echo ""
else
    echo "⏭  Tests G2 ML ignorés (torch non installé)"
    echo ""
    G2_EXIT=0
fi

# Résumé
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    RÉSUMÉ FINAL                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
if [ $CORE_EXIT -eq 0 ]; then
    echo "✅ Tests Core: RÉUSSITE"
else
    echo "⚠️  Tests Core: Quelques échecs (voir détails ci-dessus)"
fi

if [ $TORCH_OK -eq 1 ]; then
    if [ $G2_EXIT -eq 0 ]; then
        echo "✅ Tests G2 ML: RÉUSSITE"
    else
        echo "⚠️  Tests G2 ML: Quelques échecs"
    fi
fi

echo ""
echo "📊 Rapport de couverture: htmlcov/index.html"
echo ""

# Exit code: 0 si tous OK, 1 sinon
if [ $CORE_EXIT -eq 0 ] && [ $G2_EXIT -eq 0 ]; then
    exit 0
else
    exit 1
fi
