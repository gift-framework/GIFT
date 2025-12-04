#!/usr/bin/env bash
# =============================================================================
# GIFT Framework - Core Repository Setup Script
# =============================================================================
# Clones or updates the gift-framework/core repository containing
# the formal Lean 4 and Coq proofs.
#
# Usage:
#   ./setup_core.sh          # Clone or update
#   ./setup_core.sh --force  # Force fresh clone
#
# Version: 1.0
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "${SCRIPT_DIR}")"
ROOT_DIR="$(dirname "${PIPELINE_DIR}")"

source "${PIPELINE_DIR}/config.env"

CORE_PATH="${ROOT_DIR}/${CORE_DIR}"

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
log() {
    echo "[$(date -u +"%H:%M:%S")] $*"
}

clone_core() {
    log "Cloning gift-framework/core..."
    mkdir -p "$(dirname "${CORE_PATH}")"
    git clone --depth 1 "${CORE_REPO}" "${CORE_PATH}"
    log "Clone complete: ${CORE_PATH}"
}

update_core() {
    log "Updating gift-framework/core..."
    cd "${CORE_PATH}"
    git fetch origin
    git reset --hard origin/main
    cd "${ROOT_DIR}"
    log "Update complete"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    local force=false

    if [[ "${1:-}" == "--force" ]]; then
        force=true
    fi

    if [[ "${force}" == true ]] && [[ -d "${CORE_PATH}" ]]; then
        log "Force mode: removing existing clone"
        rm -rf "${CORE_PATH}"
    fi

    if [[ -d "${CORE_PATH}/.git" ]]; then
        update_core
    else
        clone_core
    fi

    # Verify structure
    log "Verifying core repository structure..."

    local errors=0

    if [[ ! -d "${ROOT_DIR}/${LEAN_DIR}" ]]; then
        log "Warning: Lean directory not found at ${LEAN_DIR}"
        ((errors++))
    else
        log "Lean directory: OK"
    fi

    if [[ ! -d "${ROOT_DIR}/${COQ_DIR}" ]]; then
        log "Warning: Coq directory not found at ${COQ_DIR}"
        ((errors++))
    else
        log "Coq directory: OK"
    fi

    if [[ ${errors} -gt 0 ]]; then
        log "Setup completed with ${errors} warning(s)"
        log "Note: Directory structure in core may differ. Check ${CORE_REPO}"
        return 0
    fi

    log "Setup complete. Core repository ready at ${CORE_PATH}"
}

main "$@"
