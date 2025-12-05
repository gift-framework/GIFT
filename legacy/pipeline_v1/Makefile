# =============================================================================
# GIFT Framework Verification Pipeline - Makefile
# =============================================================================
# Orchestration targets for the unified verification pipeline.
#
# Usage:
#   make [target]
#
# Targets:
#   all        - Run complete verification (default)
#   lean       - Verify Lean 4 framework
#   coq        - Verify Coq framework
#   g2         - Validate G2 metric
#   report     - Generate verification report
#   checksums  - Compute source checksums
#   notebooks  - Execute canonical notebooks
#   clean      - Remove generated outputs
#   help       - Show this help
#
# Version: 1.0
# =============================================================================

.PHONY: all lean coq g2 report checksums notebooks clean help

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SHELL := /bin/bash
PIPELINE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(PIPELINE_DIR)/..
SCRIPTS_DIR := $(PIPELINE_DIR)/scripts
OUTPUT_DIR := $(PIPELINE_DIR)/outputs
NOTEBOOKS_DIR := $(PIPELINE_DIR)/notebooks

# Include configuration
include $(PIPELINE_DIR)/config.env

# Timestamp for reports
TIMESTAMP := $(shell date -u +"%Y%m%d_%H%M%S")
DATE_STAMP := $(shell date -u +"%Y-%m-%d")

# -----------------------------------------------------------------------------
# Default Target
# -----------------------------------------------------------------------------
all: lean coq g2 checksums report
	@echo ""
	@echo "=========================================="
	@echo "GIFT Verification Pipeline Complete"
	@echo "=========================================="
	@echo "Results: $(OUTPUT_DIR)/reports/"
	@echo ""

# -----------------------------------------------------------------------------
# Lean Verification
# -----------------------------------------------------------------------------
lean: $(OUTPUT_DIR)/lean/verification.json

$(OUTPUT_DIR)/lean/verification.json: $(SCRIPTS_DIR)/verify_lean.sh
	@echo "[Lean] Starting verification..."
	@mkdir -p $(OUTPUT_DIR)/lean
	@$(SCRIPTS_DIR)/verify_lean.sh
	@echo "[Lean] Complete"

# -----------------------------------------------------------------------------
# Coq Verification
# -----------------------------------------------------------------------------
coq: $(OUTPUT_DIR)/coq/verification.json

$(OUTPUT_DIR)/coq/verification.json: $(SCRIPTS_DIR)/verify_coq.sh
	@echo "[Coq] Starting verification..."
	@mkdir -p $(OUTPUT_DIR)/coq
	@$(SCRIPTS_DIR)/verify_coq.sh
	@echo "[Coq] Complete"

# -----------------------------------------------------------------------------
# G2 Metric Validation
# -----------------------------------------------------------------------------
g2: $(OUTPUT_DIR)/g2/validation.json

$(OUTPUT_DIR)/g2/validation.json: $(SCRIPTS_DIR)/verify_g2.sh
	@echo "[G2] Starting validation..."
	@mkdir -p $(OUTPUT_DIR)/g2
	@$(SCRIPTS_DIR)/verify_g2.sh
	@echo "[G2] Complete"

# -----------------------------------------------------------------------------
# Checksum Computation
# -----------------------------------------------------------------------------
checksums: $(OUTPUT_DIR)/checksums/manifest.txt

$(OUTPUT_DIR)/checksums/manifest.txt: $(SCRIPTS_DIR)/compute_checksums.sh
	@echo "[Checksums] Computing..."
	@mkdir -p $(OUTPUT_DIR)/checksums
	@$(SCRIPTS_DIR)/compute_checksums.sh
	@echo "[Checksums] Complete"

# -----------------------------------------------------------------------------
# Report Generation
# -----------------------------------------------------------------------------
report: $(OUTPUT_DIR)/reports/GIFT_Verification_Report_$(DATE_STAMP).md

$(OUTPUT_DIR)/reports/GIFT_Verification_Report_$(DATE_STAMP).md: lean coq g2 checksums
	@echo "[Report] Generating..."
	@mkdir -p $(OUTPUT_DIR)/reports
	@$(SCRIPTS_DIR)/generate_report.sh
	@echo "[Report] Complete: $@"

# -----------------------------------------------------------------------------
# Notebook Execution
# -----------------------------------------------------------------------------
notebooks:
	@echo "[Notebooks] Executing canonical notebooks..."
	@for nb in $(NOTEBOOKS_DIR)/*.ipynb; do \
		if [ -f "$$nb" ]; then \
			echo "  Running: $$(basename $$nb)"; \
			jupyter nbconvert --execute --inplace "$$nb" 2>/dev/null || true; \
		fi; \
	done
	@echo "[Notebooks] Complete"

# -----------------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------------
clean:
	@echo "[Clean] Removing generated outputs..."
	@rm -rf $(OUTPUT_DIR)/lean/*
	@rm -rf $(OUTPUT_DIR)/coq/*
	@rm -rf $(OUTPUT_DIR)/g2/*
	@rm -rf $(OUTPUT_DIR)/reports/*
	@rm -rf $(OUTPUT_DIR)/checksums/*
	@rm -f $(OUTPUT_DIR)/verification.log
	@echo "[Clean] Complete"

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:
	@echo "GIFT Framework Verification Pipeline"
	@echo "====================================="
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all        Run complete verification (default)"
	@echo "  lean       Verify Lean 4 framework (13 theorems)"
	@echo "  coq        Verify Coq framework (13 theorems)"
	@echo "  g2         Validate G2 metric (det(g) = 65/32)"
	@echo "  report     Generate verification report"
	@echo "  checksums  Compute source file checksums"
	@echo "  notebooks  Execute canonical notebooks"
	@echo "  clean      Remove all generated outputs"
	@echo "  help       Show this help"
	@echo ""
	@echo "Output directory: $(OUTPUT_DIR)"
	@echo ""
