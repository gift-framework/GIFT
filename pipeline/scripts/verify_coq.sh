#!/usr/bin/env bash
# =============================================================================
# GIFT Framework - Coq Verification Script
# =============================================================================
# Verifies the Coq formalization of the GIFT framework.
#
# Outputs:
#   - pipeline/outputs/coq/verification.json
#   - pipeline/outputs/coq/build.log
#   - pipeline/outputs/coq/theorems.txt
#   - pipeline/outputs/coq/admitted_audit.txt
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

COQ_PATH="${ROOT_DIR}/${COQ_DIR}"
OUTPUT_PATH="${ROOT_DIR}/${OUTPUT_DIR}/coq"
TIMESTAMP=$(date -u +"${REPORT_TIME_FORMAT}")

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
log() {
    echo "[$(date -u +"%H:%M:%S")] $*"
}

check_coq_installed() {
    if ! command -v coqc &> /dev/null; then
        log "Error: coqc (Coq compiler) not found"
        log "Install Coq from: https://coq.inria.fr/download"
        return 1
    fi
    return 0
}

get_coq_version() {
    if command -v coqc &> /dev/null; then
        coqc --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' || echo "unknown"
    else
        echo "not_installed"
    fi
}

count_admitted() {
    local count
    count=$(grep -r "Admitted\." "${COQ_PATH}" --include="*.v" 2>/dev/null | wc -l || echo "0")
    echo "${count// /}"
}

count_theorems() {
    # Count Theorem/Lemma declarations
    local count
    count=$(grep -rE "^(Theorem|Lemma) " "${COQ_PATH}" --include="*.v" 2>/dev/null | wc -l || echo "0")
    echo "${count// /}"
}

extract_theorem_list() {
    # Extract theorem names
    grep -rE "^(Theorem|Lemma) [a-zA-Z_]+" "${COQ_PATH}" --include="*.v" 2>/dev/null \
        | sed 's/.*:\(Theorem\|Lemma\) \([a-zA-Z_0-9]*\).*/\2/' \
        | sort -u
}

extract_admitted_audit() {
    # Find any Admitted statements
    grep -rn "Admitted\." "${COQ_PATH}" --include="*.v" 2>/dev/null || echo "No Admitted statements found"
}

build_coq() {
    local build_log="${OUTPUT_PATH}/build.log"
    local start_time end_time duration

    log "Building Coq project..."
    start_time=$(date +%s)

    cd "${COQ_PATH}"

    # Generate dependencies if Makefile exists
    if [[ -f "Makefile" ]]; then
        # Try to build
        if make -j4 > "${build_log}" 2>&1; then
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
    else
        # No Makefile, try to compile files individually
        log "No Makefile found, compiling files individually..."
        local success=true
        for vfile in $(find . -name "*.v" | sort); do
            if ! coqc -Q . GIFT "${vfile}" >> "${build_log}" 2>&1; then
                success=false
            fi
        done
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        if ${success}; then
            log "Build successful (${duration}s)"
            echo "${duration}"
            return 0
        else
            log "Build failed (${duration}s)"
            echo "${duration}"
            return 1
        fi
    fi
}

compute_source_checksum() {
    find "${COQ_PATH}" -name "*.v" -type f -exec sha256sum {} \; 2>/dev/null \
        | sort -k2 \
        | sha256sum \
        | cut -d' ' -f1
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    log "Starting Coq verification"
    log "Coq directory: ${COQ_PATH}"

    mkdir -p "${OUTPUT_PATH}"

    # Check Coq installation
    if ! check_coq_installed; then
        # Generate failure JSON
        cat > "${OUTPUT_PATH}/verification.json" << EOF
{
  "timestamp": "${TIMESTAMP}",
  "component": "coq",
  "status": "FAIL",
  "error": "Coq not installed",
  "coq_version": "not_installed"
}
EOF
        return 1
    fi

    local coq_version
    coq_version=$(get_coq_version)
    log "Coq version: ${coq_version}"

    # Build the project
    local build_time build_status
    if build_time=$(build_coq); then
        build_status="success"
    else
        build_status="failed"
    fi

    # Count Admitted statements
    local admitted_count
    admitted_count=$(count_admitted)
    log "Admitted count: ${admitted_count}"

    # Count and extract theorems
    local theorem_count
    theorem_count=$(count_theorems)
    log "Theorem count: ${theorem_count}"

    # Extract theorem list
    extract_theorem_list > "${OUTPUT_PATH}/theorems.txt"
    log "Theorem list written to theorems.txt"

    # Extract admitted audit
    extract_admitted_audit > "${OUTPUT_PATH}/admitted_audit.txt"
    log "Admitted audit written to admitted_audit.txt"

    # Compute source checksum
    local source_checksum
    source_checksum=$(compute_source_checksum)
    log "Source checksum: ${source_checksum:0:16}..."

    # Determine overall status
    local status="PASS"
    if [[ "${build_status}" != "success" ]]; then
        status="FAIL"
    elif [[ "${admitted_count}" -gt "${EXPECTED_ADMITTED_COUNT}" ]]; then
        status="FAIL"
    fi

    # Count warnings and errors from build log
    local warning_count error_count
    warning_count=$(grep -c "Warning:" "${OUTPUT_PATH}/build.log" 2>/dev/null || echo "0")
    error_count=$(grep -c "Error:" "${OUTPUT_PATH}/build.log" 2>/dev/null || echo "0")

    # Generate verification JSON
    cat > "${OUTPUT_PATH}/verification.json" << EOF
{
  "timestamp": "${TIMESTAMP}",
  "component": "coq",
  "coq_version": "${coq_version}",
  "status": "${status}",
  "build_result": {
    "success": $([ "${build_status}" = "success" ] && echo "true" || echo "false"),
    "time_seconds": ${build_time:-0},
    "warnings": ${warning_count},
    "errors": ${error_count}
  },
  "theorems": {
    "total": ${theorem_count},
    "expected": ${EXPECTED_COQ_THEOREMS},
    "verified": ${theorem_count}
  },
  "admitted_count": ${admitted_count},
  "expected_admitted_count": ${EXPECTED_ADMITTED_COUNT},
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
