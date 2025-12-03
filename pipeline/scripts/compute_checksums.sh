#!/usr/bin/env bash
# =============================================================================
# GIFT Framework - Checksum Computation Script
# =============================================================================
# Computes SHA-256 checksums for all source files in the framework.
#
# Outputs:
#   - pipeline/outputs/checksums/manifest.txt
#   - pipeline/outputs/checksums/lean_sources.txt
#   - pipeline/outputs/checksums/coq_sources.txt
#   - pipeline/outputs/checksums/g2_sources.txt
#   - pipeline/outputs/checksums/aggregate.txt
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

OUTPUT_PATH="${ROOT_DIR}/${OUTPUT_DIR}/checksums"
TIMESTAMP=$(date -u +"${REPORT_TIME_FORMAT}")

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
log() {
    echo "[$(date -u +"%H:%M:%S")] $*"
}

compute_lean_checksums() {
    local output_file="${OUTPUT_PATH}/lean_sources.txt"
    local lean_path="${ROOT_DIR}/${LEAN_DIR}"

    echo "# Lean 4 Source Checksums" > "${output_file}"
    echo "# Algorithm: SHA-256" >> "${output_file}"
    echo "# Timestamp: ${TIMESTAMP}" >> "${output_file}"
    echo "# Directory: ${LEAN_DIR}" >> "${output_file}"
    echo "" >> "${output_file}"

    if [[ -d "${lean_path}" ]]; then
        find "${lean_path}" -name "*.lean" -type f | sort | while read -r file; do
            local relative_path="${file#${ROOT_DIR}/}"
            local checksum
            checksum=$(sha256sum "${file}" | cut -d' ' -f1)
            echo "sha256:${checksum}  ${relative_path}" >> "${output_file}"
        done

        # Include lakefile
        if [[ -f "${lean_path}/lakefile.lean" ]]; then
            local checksum
            checksum=$(sha256sum "${lean_path}/lakefile.lean" | cut -d' ' -f1)
            echo "sha256:${checksum}  ${LEAN_DIR}/lakefile.lean" >> "${output_file}"
        fi

        local count
        count=$(grep -c "sha256:" "${output_file}" 2>/dev/null || echo "0")
        log "Lean sources: ${count} files"
    else
        echo "# Directory not found" >> "${output_file}"
        log "Warning: Lean directory not found"
    fi
}

compute_coq_checksums() {
    local output_file="${OUTPUT_PATH}/coq_sources.txt"
    local coq_path="${ROOT_DIR}/${COQ_DIR}"

    echo "# Coq Source Checksums" > "${output_file}"
    echo "# Algorithm: SHA-256" >> "${output_file}"
    echo "# Timestamp: ${TIMESTAMP}" >> "${output_file}"
    echo "# Directory: ${COQ_DIR}" >> "${output_file}"
    echo "" >> "${output_file}"

    if [[ -d "${coq_path}" ]]; then
        find "${coq_path}" -name "*.v" -type f | sort | while read -r file; do
            local relative_path="${file#${ROOT_DIR}/}"
            local checksum
            checksum=$(sha256sum "${file}" | cut -d' ' -f1)
            echo "sha256:${checksum}  ${relative_path}" >> "${output_file}"
        done

        # Include build files
        for build_file in "_CoqProject" "Makefile"; do
            if [[ -f "${coq_path}/${build_file}" ]]; then
                local checksum
                checksum=$(sha256sum "${coq_path}/${build_file}" | cut -d' ' -f1)
                echo "sha256:${checksum}  ${COQ_DIR}/${build_file}" >> "${output_file}"
            fi
        done

        local count
        count=$(grep -c "sha256:" "${output_file}" 2>/dev/null || echo "0")
        log "Coq sources: ${count} files"
    else
        echo "# Directory not found" >> "${output_file}"
        log "Warning: Coq directory not found"
    fi
}

compute_g2_checksums() {
    local output_file="${OUTPUT_PATH}/g2_sources.txt"
    local g2_path="${ROOT_DIR}/${G2_DIR}"

    echo "# G2 Lean Source Checksums" > "${output_file}"
    echo "# Algorithm: SHA-256" >> "${output_file}"
    echo "# Timestamp: ${TIMESTAMP}" >> "${output_file}"
    echo "# Directory: ${G2_DIR}" >> "${output_file}"
    echo "" >> "${output_file}"

    if [[ -d "${g2_path}" ]]; then
        # Lean files
        echo "## Lean Files" >> "${output_file}"
        find "${g2_path}" -maxdepth 1 -name "*.lean" -type f | sort | while read -r file; do
            local relative_path="${file#${ROOT_DIR}/}"
            local checksum
            checksum=$(sha256sum "${file}" | cut -d' ' -f1)
            echo "sha256:${checksum}  ${relative_path}" >> "${output_file}"
        done

        # Python files (top level only for core validation)
        echo "" >> "${output_file}"
        echo "## Python Files" >> "${output_file}"
        find "${g2_path}" -maxdepth 1 -name "*.py" -type f | sort | while read -r file; do
            local relative_path="${file#${ROOT_DIR}/}"
            local checksum
            checksum=$(sha256sum "${file}" | cut -d' ' -f1)
            echo "sha256:${checksum}  ${relative_path}" >> "${output_file}"
        done

        local count
        count=$(grep -c "sha256:" "${output_file}" 2>/dev/null || echo "0")
        log "G2 sources: ${count} files"
    else
        echo "# Directory not found" >> "${output_file}"
        log "Warning: G2 directory not found"
    fi
}

compute_aggregate_checksum() {
    local output_file="${OUTPUT_PATH}/aggregate.txt"

    echo "# Aggregate Checksum" > "${output_file}"
    echo "# Algorithm: SHA-256" >> "${output_file}"
    echo "# Timestamp: ${TIMESTAMP}" >> "${output_file}"
    echo "# Method: SHA-256 of concatenated source checksums" >> "${output_file}"
    echo "" >> "${output_file}"

    # Concatenate all individual checksum files and compute aggregate
    local aggregate
    aggregate=$(cat "${OUTPUT_PATH}/lean_sources.txt" \
                    "${OUTPUT_PATH}/coq_sources.txt" \
                    "${OUTPUT_PATH}/g2_sources.txt" 2>/dev/null \
                | grep "^sha256:" \
                | sort \
                | sha256sum \
                | cut -d' ' -f1)

    echo "sha256:${aggregate}  GIFT_AGGREGATE" >> "${output_file}"

    log "Aggregate checksum: ${aggregate:0:16}..."
}

generate_manifest() {
    local manifest_file="${OUTPUT_PATH}/manifest.txt"

    cat > "${manifest_file}" << EOF
# =============================================================================
# GIFT Framework Checksum Manifest
# =============================================================================
# Generated: ${TIMESTAMP}
# Algorithm: SHA-256
# Pipeline Version: ${PIPELINE_VERSION}
# GIFT Version: ${GIFT_VERSION}
# =============================================================================

EOF

    # Git information
    local git_commit git_branch
    git_commit=$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo "unknown")
    git_branch=$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

    cat >> "${manifest_file}" << EOF
## Repository State

Git Commit: ${git_commit}
Git Branch: ${git_branch}

EOF

    # Include all source checksums
    echo "## Lean 4 Sources" >> "${manifest_file}"
    echo "" >> "${manifest_file}"
    grep "^sha256:" "${OUTPUT_PATH}/lean_sources.txt" 2>/dev/null >> "${manifest_file}" || true
    echo "" >> "${manifest_file}"

    echo "## Coq Sources" >> "${manifest_file}"
    echo "" >> "${manifest_file}"
    grep "^sha256:" "${OUTPUT_PATH}/coq_sources.txt" 2>/dev/null >> "${manifest_file}" || true
    echo "" >> "${manifest_file}"

    echo "## G2 Lean Sources" >> "${manifest_file}"
    echo "" >> "${manifest_file}"
    grep "^sha256:" "${OUTPUT_PATH}/g2_sources.txt" 2>/dev/null >> "${manifest_file}" || true
    echo "" >> "${manifest_file}"

    echo "## Aggregate" >> "${manifest_file}"
    echo "" >> "${manifest_file}"
    grep "^sha256:" "${OUTPUT_PATH}/aggregate.txt" 2>/dev/null >> "${manifest_file}" || true
    echo "" >> "${manifest_file}"

    # Summary statistics
    local lean_count coq_count g2_count total_count
    lean_count=$(grep -c "^sha256:" "${OUTPUT_PATH}/lean_sources.txt" 2>/dev/null || echo "0")
    coq_count=$(grep -c "^sha256:" "${OUTPUT_PATH}/coq_sources.txt" 2>/dev/null || echo "0")
    g2_count=$(grep -c "^sha256:" "${OUTPUT_PATH}/g2_sources.txt" 2>/dev/null || echo "0")
    total_count=$((lean_count + coq_count + g2_count))

    cat >> "${manifest_file}" << EOF
## Summary

| Component | File Count |
|-----------|------------|
| Lean 4    | ${lean_count} |
| Coq       | ${coq_count} |
| G2 Lean   | ${g2_count} |
| **Total** | **${total_count}** |

EOF

    log "Manifest generated: ${manifest_file}"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    log "Starting checksum computation"

    mkdir -p "${OUTPUT_PATH}"

    compute_lean_checksums
    compute_coq_checksums
    compute_g2_checksums
    compute_aggregate_checksum
    generate_manifest

    log "Checksum computation complete"
    log "Output: ${OUTPUT_PATH}/manifest.txt"
}

main "$@"
