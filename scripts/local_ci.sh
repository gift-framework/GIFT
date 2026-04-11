#!/usr/bin/env bash
# Local CI runner for the GIFT repo.
# Runs the same checks the GitHub Actions workflows do, BEFORE you push.
#
# Usage:
#   ./scripts/local_ci.sh           # run all checks
#   ./scripts/local_ci.sh --fix     # run all checks AND auto-fix where possible
#   ./scripts/local_ci.sh docs      # only the docs linter
#   ./scripts/local_ci.sh consistency  # only cross-repo consistency
#   ./scripts/local_ci.sh observables  # only observable calculator
#
# Exit code 0 = all pass, non-zero = at least one check failed.

set -e
cd "$(dirname "$0")/.."

ROOT="$(pwd)"
FAIL=0
FIX_MODE=0
TARGET="all"

# Parse args
for arg in "$@"; do
    case "$arg" in
        --fix) FIX_MODE=1 ;;
        docs|consistency|observables|all) TARGET="$arg" ;;
        -h|--help)
            head -16 "$0"
            exit 0
            ;;
    esac
done

green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
blue()  { printf '\033[34m%s\033[0m\n' "$*"; }

run_check() {
    local name="$1"
    shift
    blue ""
    blue "============================================================"
    blue " $name"
    blue "============================================================"
    if "$@"; then
        green "  ✓ $name PASSED"
    else
        red "  ✗ $name FAILED"
        FAIL=1
    fi
}

# ----------------------------------------------------------------------
# 1. Docs linter (em-dashes, evolutionary language, status, cross-refs)
# ----------------------------------------------------------------------
docs_check() {
    if [ "$FIX_MODE" -eq 1 ]; then
        blue "  → Auto-fixing em-dashes first (recursive)..."
        python3 .github/scripts/fix_em_dashes.py 2>&1 || true
    fi
    python3 .github/scripts/docs_linter.py || return 1
    # The CI workflow also runs fix_em_dashes.py --check (recursive on docs/,
    # so it catches the wiki/ subdirectory). Mirror that exactly.
    blue ""
    blue "  → Running em-dash check (recursive on docs/)..."
    python3 .github/scripts/fix_em_dashes.py --check
}

# ----------------------------------------------------------------------
# 2. Cross-repo consistency (only if gift-framework/core is cloned next door)
# ----------------------------------------------------------------------
consistency_check() {
    local CORE_PATH="../core"
    if [ ! -d "$CORE_PATH" ]; then
        echo "  ⚠ ../core/ not found — skipping cross-repo consistency check"
        echo "    (clone gift-framework/core next to this repo to enable it)"
        return 0
    fi
    python3 .github/scripts/cross_repo_check.py --core-path "$CORE_PATH"
}

# ----------------------------------------------------------------------
# 3. Observable calculator
# ----------------------------------------------------------------------
observables_check() {
    if [ ! -f .github/scripts/observable_calculator.py ]; then
        echo "  ⚠ observable_calculator.py not found — skipping"
        return 0
    fi
    python3 .github/scripts/observable_calculator.py
}

# Dispatch
case "$TARGET" in
    all)
        run_check "Docs Linter"          docs_check
        run_check "Cross-Repo Consistency" consistency_check
        run_check "Observable Calculator"  observables_check
        ;;
    docs)         run_check "Docs Linter"          docs_check ;;
    consistency)  run_check "Cross-Repo Consistency" consistency_check ;;
    observables)  run_check "Observable Calculator"  observables_check ;;
esac

echo ""
if [ $FAIL -eq 0 ]; then
    green "============================================================"
    green " ✅  All local CI checks passed. Safe to push."
    green "============================================================"
    exit 0
else
    red "============================================================"
    red " ❌  Local CI failed. Fix issues before pushing."
    red "============================================================"
    if [ "$FIX_MODE" -eq 0 ]; then
        echo ""
        echo "  Tip: re-run with --fix to auto-fix em-dashes and other safe edits."
    fi
    exit 1
fi
