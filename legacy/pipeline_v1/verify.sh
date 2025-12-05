#!/usr/bin/env bash
# =============================================================================
# GIFT Framework Verification Pipeline
# =============================================================================
# Master script for unified verification of the GIFT v2.3 framework.
#
# Usage:
#   ./verify.sh [command]
#
# Commands:
#   all       Run complete verification (Lean + Coq + G2 + report)
#   lean      Verify Lean 4 framework only
#   coq       Verify Coq framework only
#   g2        Validate G2 metric only
#   report    Generate report from existing results
#   checksums Compute source file checksums
#   clean     Remove all generated outputs
#   status    Show current verification status
#   help      Show this help message
#
# Version: 1.0
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${SCRIPT_DIR}/pipeline"
CONFIG_FILE="${PIPELINE_DIR}/config.env"

# Load configuration
if [[ -f "${CONFIG_FILE}" ]]; then
    source "${CONFIG_FILE}"
else
    echo "Error: Configuration file not found: ${CONFIG_FILE}"
    exit 1
fi

# Derived paths
OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_DIR}"
SCRIPTS_DIR="${PIPELINE_DIR}/scripts"
LEAN_PATH="${SCRIPT_DIR}/${LEAN_DIR}"
COQ_PATH="${SCRIPT_DIR}/${COQ_DIR}"
G2_PATH="${SCRIPT_DIR}/${G2_DIR}"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_FILE="${OUTPUT_DIR}/verification.log"

log() {
    local timestamp
    timestamp=$(date -u +"${REPORT_TIME_FORMAT}")
    echo "[${timestamp}] $*" | tee -a "${LOG_FILE}"
}

log_section() {
    echo ""
    echo "==========================================================================="
    log "$*"
    echo "==========================================================================="
}

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
check_dependencies() {
    local missing=()

    command -v jq >/dev/null 2>&1 || missing+=("jq")
    command -v sha256sum >/dev/null 2>&1 || missing+=("sha256sum")

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "Error: Missing required tools: ${missing[*]}"
        exit 1
    fi
}

get_git_info() {
    local commit_hash branch
    commit_hash=$(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
    branch=$(git -C "${SCRIPT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    echo "${branch}:${commit_hash}"
}

ensure_output_dirs() {
    mkdir -p "${OUTPUT_DIR}"/{lean,coq,g2,reports,checksums}
}

# -----------------------------------------------------------------------------
# Command: all
# -----------------------------------------------------------------------------
cmd_all() {
    local start_time end_time duration
    start_time=$(date +%s)

    log_section "GIFT Framework Verification Pipeline v${PIPELINE_VERSION}"
    log "Timestamp: $(date -u +"${REPORT_TIME_FORMAT}")"
    log "Git: $(get_git_info)"
    log "GIFT Version: ${GIFT_VERSION}"

    ensure_output_dirs
    check_dependencies

    local all_pass=true

    # Step 1: Lean verification
    log_section "[1/4] Lean 4 Verification"
    if cmd_lean; then
        log "Lean: PASS"
    else
        log "Lean: FAIL"
        all_pass=false
    fi

    # Step 2: Coq verification
    log_section "[2/4] Coq Verification"
    if cmd_coq; then
        log "Coq: PASS"
    else
        log "Coq: FAIL"
        all_pass=false
    fi

    # Step 3: G2 validation
    log_section "[3/4] G2 Metric Validation"
    if cmd_g2; then
        log "G2: PASS"
    else
        log "G2: FAIL"
        all_pass=false
    fi

    # Step 4: Report generation
    log_section "[4/4] Report Generation"
    cmd_checksums
    cmd_report

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    log_section "VERIFICATION COMPLETE"
    log "Total duration: ${duration} seconds"

    if ${all_pass}; then
        log "All components: PASS"
        return 0
    else
        log "Some components: FAIL"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Command: lean
# -----------------------------------------------------------------------------
cmd_lean() {
    log "Starting Lean 4 verification..."

    if [[ -x "${SCRIPTS_DIR}/verify_lean.sh" ]]; then
        "${SCRIPTS_DIR}/verify_lean.sh"
    else
        log "Error: verify_lean.sh not found or not executable"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Command: coq
# -----------------------------------------------------------------------------
cmd_coq() {
    log "Starting Coq verification..."

    if [[ -x "${SCRIPTS_DIR}/verify_coq.sh" ]]; then
        "${SCRIPTS_DIR}/verify_coq.sh"
    else
        log "Error: verify_coq.sh not found or not executable"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Command: g2
# -----------------------------------------------------------------------------
cmd_g2() {
    log "Starting G2 metric validation..."

    if [[ -x "${SCRIPTS_DIR}/verify_g2.sh" ]]; then
        "${SCRIPTS_DIR}/verify_g2.sh"
    else
        log "Error: verify_g2.sh not found or not executable"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Command: checksums
# -----------------------------------------------------------------------------
cmd_checksums() {
    log "Computing checksums..."

    if [[ -x "${SCRIPTS_DIR}/compute_checksums.sh" ]]; then
        "${SCRIPTS_DIR}/compute_checksums.sh"
    else
        log "Error: compute_checksums.sh not found or not executable"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Command: report
# -----------------------------------------------------------------------------
cmd_report() {
    log "Generating report..."

    if [[ -x "${SCRIPTS_DIR}/generate_report.sh" ]]; then
        "${SCRIPTS_DIR}/generate_report.sh"
    else
        log "Error: generate_report.sh not found or not executable"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Command: status
# -----------------------------------------------------------------------------
cmd_status() {
    echo "GIFT Framework Verification Status"
    echo "==================================="
    echo ""
    echo "Pipeline Version: ${PIPELINE_VERSION}"
    echo "GIFT Version: ${GIFT_VERSION}"
    echo "Git: $(get_git_info)"
    echo ""

    echo "Output Directory: ${OUTPUT_DIR}"
    echo ""

    echo "Component Status:"

    # Check Lean results
    if [[ -f "${OUTPUT_DIR}/lean/verification.json" ]]; then
        local lean_status
        lean_status=$(jq -r '.status' "${OUTPUT_DIR}/lean/verification.json" 2>/dev/null || echo "ERROR")
        echo "  Lean:  ${lean_status}"
    else
        echo "  Lean:  NOT RUN"
    fi

    # Check Coq results
    if [[ -f "${OUTPUT_DIR}/coq/verification.json" ]]; then
        local coq_status
        coq_status=$(jq -r '.status' "${OUTPUT_DIR}/coq/verification.json" 2>/dev/null || echo "ERROR")
        echo "  Coq:   ${coq_status}"
    else
        echo "  Coq:   NOT RUN"
    fi

    # Check G2 results
    if [[ -f "${OUTPUT_DIR}/g2/validation.json" ]]; then
        local g2_status
        g2_status=$(jq -r '.status' "${OUTPUT_DIR}/g2/validation.json" 2>/dev/null || echo "ERROR")
        echo "  G2:    ${g2_status}"
    else
        echo "  G2:    NOT RUN"
    fi

    # Check report
    local latest_report
    latest_report=$(ls -t "${OUTPUT_DIR}/reports/"*.md 2>/dev/null | head -1)
    if [[ -n "${latest_report}" ]]; then
        echo ""
        echo "Latest Report: ${latest_report}"
    fi
}

# -----------------------------------------------------------------------------
# Command: clean
# -----------------------------------------------------------------------------
cmd_clean() {
    log "Cleaning output directories..."
    rm -rf "${OUTPUT_DIR:?}"/*
    log "Clean complete"
}

# -----------------------------------------------------------------------------
# Command: help
# -----------------------------------------------------------------------------
cmd_help() {
    head -30 "${BASH_SOURCE[0]}" | tail -27 | sed 's/^# //' | sed 's/^#//'
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    local command="${1:-help}"

    case "${command}" in
        all)
            cmd_all
            ;;
        lean)
            ensure_output_dirs
            cmd_lean
            ;;
        coq)
            ensure_output_dirs
            cmd_coq
            ;;
        g2)
            ensure_output_dirs
            cmd_g2
            ;;
        report)
            cmd_report
            ;;
        checksums)
            ensure_output_dirs
            cmd_checksums
            ;;
        status)
            cmd_status
            ;;
        clean)
            cmd_clean
            ;;
        help|--help|-h)
            cmd_help
            ;;
        *)
            echo "Unknown command: ${command}"
            echo "Run './verify.sh help' for usage information."
            exit 1
            ;;
    esac
}

main "$@"
