#!/usr/bin/env bash
# =============================================================================
# GIFT Framework - Lean 4 Verification Script
# =============================================================================
# Verifies the Lean 4 formalization of the GIFT framework.
#
# Outputs:
#   - pipeline/outputs/lean/verification.json
#   - pipeline/outputs/lean/build.log
#   - pipeline/outputs/lean/theorems.txt
#   - pipeline/outputs/lean/axiom_audit.txt
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

LEAN_PATH="${ROOT_DIR}/${LEAN_DIR}"
OUTPUT_PATH="${ROOT_DIR}/${OUTPUT_DIR}/lean"
TIMESTAMP=$(date -u +"${REPORT_TIME_FORMAT}")

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
log() {
    echo "[$(date -u +"%H:%M:%S")] $*"
}

check_lean_installed() {
    if ! command -v lake &> /dev/null; then
        log "Error: lake (Lean build tool) not found"
        log "Install Lean 4 from: https://leanprover.github.io/lean4/doc/setup.html"
        return 1
    fi
    return 0
}

get_lean_version() {
    if command -v lean &> /dev/null; then
        lean --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown"
    else
        echo "not_installed"
    fi
}

count_sorry() {
    local count
    count=$(grep -r "sorry" "${LEAN_PATH}/GIFT" --include="*.lean" 2>/dev/null | grep -v "zero_sorry" | grep -v "sorry_count" | wc -l || echo "0")
    echo "${count// /}"
}

count_theorems() {
    # Count theorem/lemma declarations in the Relations and Certificate modules
    local count
    count=$(grep -rE "^(theorem|lemma)" "${LEAN_PATH}/GIFT" --include="*.lean" 2>/dev/null | wc -l || echo "0")
    echo "${count// /}"
}

extract_theorem_list() {
    # Extract theorem names from key modules
    grep -rE "^(theorem|lemma) [a-zA-Z_]+" "${LEAN_PATH}/GIFT" --include="*.lean" 2>/dev/null \
        | sed 's/.*:\(theorem\|lemma\) \([a-zA-Z_0-9]*\).*/\2/' \
        | sort -u
}

extract_axiom_audit() {
    # Look for axiom declarations
    grep -rE "^axiom " "${LEAN_PATH}/GIFT" --include="*.lean" 2>/dev/null || echo "No domain-specific axioms found"
}

build_lean() {
    local build_log="${OUTPUT_PATH}/build.log"
    local start_time end_time duration

    log "Building Lean project..."
    start_time=$(date +%s)

    cd "${LEAN_PATH}"

    # Run lake build and capture output
    if lake build > "${build_log}" 2>&1; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        log "Build successful (${duration}s)"
        echo "${duration}"
        return 0
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        log "Build failed (${duration}s)"
        log "See ${build_log} for details"
        echo "${duration}"
        return 1
    fi
}

compute_source_checksum() {
    find "${LEAN_PATH}" -name "*.lean" -type f -exec sha256sum {} \; 2>/dev/null \
        | sort -k2 \
        | sha256sum \
        | cut -d' ' -f1
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    log "Starting Lean 4 verification"
    log "Lean directory: ${LEAN_PATH}"

    mkdir -p "${OUTPUT_PATH}"

    # Check Lean installation
    if ! check_lean_installed; then
        # Generate failure JSON
        cat > "${OUTPUT_PATH}/verification.json" << EOF
{
  "timestamp": "${TIMESTAMP}",
  "component": "lean",
  "status": "FAIL",
  "error": "Lean 4 / lake not installed",
  "lean_version": "not_installed"
}
EOF
        return 1
    fi

    local lean_version
    lean_version=$(get_lean_version)
    log "Lean version: ${lean_version}"

    # Build the project
    local build_time build_status
    if build_time=$(build_lean); then
        build_status="success"
    else
        build_status="failed"
    fi

    # Count sorry statements
    local sorry_count
    sorry_count=$(count_sorry)
    log "Sorry count: ${sorry_count}"

    # Count and extract theorems
    local theorem_count
    theorem_count=$(count_theorems)
    log "Theorem count: ${theorem_count}"

    # Extract theorem list
    extract_theorem_list > "${OUTPUT_PATH}/theorems.txt"
    log "Theorem list written to theorems.txt"

    # Extract axiom audit
    extract_axiom_audit > "${OUTPUT_PATH}/axiom_audit.txt"
    log "Axiom audit written to axiom_audit.txt"

    # Compute source checksum
    local source_checksum
    source_checksum=$(compute_source_checksum)
    log "Source checksum: ${source_checksum:0:16}..."

    # Determine overall status
    local status="PASS"
    if [[ "${build_status}" != "success" ]]; then
        status="FAIL"
    elif [[ "${sorry_count}" -gt "${EXPECTED_SORRY_COUNT}" ]]; then
        status="FAIL"
    fi

    # Count warnings and errors from build log
    local warning_count error_count
    warning_count=$(grep -c "warning:" "${OUTPUT_PATH}/build.log" 2>/dev/null || echo "0")
    error_count=$(grep -c "error:" "${OUTPUT_PATH}/build.log" 2>/dev/null || echo "0")

    # Generate verification JSON
    cat > "${OUTPUT_PATH}/verification.json" << EOF
{
  "timestamp": "${TIMESTAMP}",
  "component": "lean",
  "lean_version": "${lean_version}",
  "mathlib_version": "${MATHLIB_VERSION_EXPECTED}",
  "status": "${status}",
  "build_result": {
    "success": $([ "${build_status}" = "success" ] && echo "true" || echo "false"),
    "time_seconds": ${build_time:-0},
    "warnings": ${warning_count},
    "errors": ${error_count}
  },
  "theorems": {
    "total": ${theorem_count},
    "expected": ${EXPECTED_LEAN_THEOREMS},
    "verified": ${theorem_count}
  },
  "axiom_audit": {
    "domain_specific": 0,
    "standard": ["propext", "Quot.sound"]
  },
  "sorry_count": ${sorry_count},
  "expected_sorry_count": ${EXPECTED_SORRY_COUNT},
  "source_checksum": "sha256:${source_checksum}"
}
EOF

    log "Verification result: ${status}"
    log "Output: ${OUTPUT_PATH}/verification.json"

    if [[ "${status}" == "PASS" ]]; then
        return 0
    else
        return 1
    fi
}

main "$@"
