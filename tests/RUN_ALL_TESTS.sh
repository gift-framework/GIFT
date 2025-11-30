#!/bin/bash
# Script pour executer tous les tests GIFT

echo "================================================================"
echo "        GIFT Framework - Execution Tests Complets              "
echo "================================================================"
echo ""

# Verifier torch
echo "[CHECK] Verification des dependances..."
if python -c "import torch" 2>/dev/null; then
    echo "[OK] Torch installe"
    TORCH_OK=1
else
    echo "[WARNING] Torch non installe - Installation recommandee:"
    echo "  pip install torch"
    TORCH_OK=0
fi
echo ""

# Tests Core (toujours disponibles)
echo "------------------------------------------------------------"
echo "Lancement des tests Core (sans torch)..."
echo "------------------------------------------------------------"
pytest tests/unit/test_gift_framework.py tests/unit/test_agents.py \
       tests/integration tests/regression \
       -v --tb=short -m "not slow" \
       --cov=. --cov-report=term-missing --cov-report=html
CORE_EXIT=$?
echo ""

# Tests G2 ML (si torch disponible)
if [ $TORCH_OK -eq 1 ]; then
    echo "------------------------------------------------------------"
    echo "Lancement des tests G2 ML (avec torch)..."
    echo "------------------------------------------------------------"
    pytest G2_ML/tests tests/unit/test_error_handling.py \
           -v --tb=short -m "not slow"
    G2_EXIT=$?
    echo ""
else
    echo "[SKIP] Tests G2 ML ignores (torch non installe)"
    echo ""
    G2_EXIT=0
fi

# Resume
echo "================================================================"
echo "                    RESUME FINAL                               "
echo "================================================================"
if [ $CORE_EXIT -eq 0 ]; then
    echo "[OK] Tests Core: REUSSITE"
else
    echo "[WARNING] Tests Core: Quelques echecs (voir details ci-dessus)"
fi

if [ $TORCH_OK -eq 1 ]; then
    if [ $G2_EXIT -eq 0 ]; then
        echo "[OK] Tests G2 ML: REUSSITE"
    else
        echo "[WARNING] Tests G2 ML: Quelques echecs"
    fi
fi

echo ""
echo "Rapport de couverture: htmlcov/index.html"
echo ""

# Exit code: 0 si tous OK, 1 sinon
if [ $CORE_EXIT -eq 0 ] && [ $G2_EXIT -eq 0 ]; then
    exit 0
else
    exit 1
fi
