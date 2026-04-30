.DEFAULT_GOAL := help

# Detect whether we're inside a virtualenv already
VENV_ACTIVATE := .venv/bin/activate
PYTHON        := python3
UV            := uv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

.PHONY: venv
venv:  ## Create a uv virtual environment in .venv/
	$(UV) venv .venv

.PHONY: install
install: venv  ## Install the package + dev dependencies
	$(UV) pip install -e ".[dev]"

.PHONY: install-neuron
install-neuron: venv  ## Install the package + Neuron extras
	$(UV) pip install -e ".[neuron,dev]"

.PHONY: setup-local
setup-local:  ## Set up a local dev environment (creates .venv, installs package)
	bash scripts/setup_inf2.sh --local

.PHONY: setup-inf2
setup-inf2:  ## Bootstrap a fresh Inf2 / Trn1 instance
	bash scripts/setup_inf2.sh

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

.PHONY: test
test:  ## Run the unit test suite (CPU)
	. $(VENV_ACTIVATE) && pytest tests/ -v

.PHONY: test-fast
test-fast:  ## Run tests, stop on first failure
	. $(VENV_ACTIVATE) && pytest tests/ -x -q

.PHONY: test-cov
test-cov:  ## Run tests with coverage report
	. $(VENV_ACTIVATE) && pytest tests/ --cov=src/lqcd_neuron --cov-report=term-missing

.PHONY: smoke
smoke:  ## Run all example scripts as smoke tests (CPU)
	. $(VENV_ACTIVATE) && \
	    $(PYTHON) examples/01_plaquette.py && \
	    $(PYTHON) examples/02_wilson_dslash.py && \
	    $(PYTHON) examples/03_cg_inversion.py

.PHONY: smoke-neuron
smoke-neuron:  ## Run all examples with Neuron compilation (requires Inf2/Trn1)
	. $(VENV_ACTIVATE) && bash scripts/run_tests.sh --bench --neuron

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

.PHONY: bench
bench:  ## CPU Dslash throughput benchmark
	. $(VENV_ACTIVATE) && $(PYTHON) examples/bench_dslash.py

.PHONY: bench-neuron
bench-neuron:  ## Neuron vs CPU Dslash throughput benchmark (requires Inf2/Trn1)
	. $(VENV_ACTIVATE) && $(PYTHON) examples/bench_dslash.py --neuron

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

.PHONY: lint
lint:  ## Run black + isort in check mode
	. $(VENV_ACTIVATE) && \
	    black --check src/ tests/ examples/ && \
	    isort --check src/ tests/ examples/

.PHONY: fmt
fmt:  ## Auto-format with black + isort
	. $(VENV_ACTIVATE) && \
	    black src/ tests/ examples/ && \
	    isort src/ tests/ examples/

.PHONY: typecheck
typecheck:  ## Run mypy
	. $(VENV_ACTIVATE) && mypy src/lqcd_neuron --ignore-missing-imports

# ---------------------------------------------------------------------------
# Infrastructure (OpenTofu)
# ---------------------------------------------------------------------------

INFRA_DIR := infra

.PHONY: tofu-init
tofu-init:  ## Initialise OpenTofu (download providers)
	tofu -chdir=$(INFRA_DIR) init

.PHONY: tofu-plan
tofu-plan:  ## Show what OpenTofu will create
	tofu -chdir=$(INFRA_DIR) plan

.PHONY: tofu-apply
tofu-apply:  ## Provision the Inf2 instance
	tofu -chdir=$(INFRA_DIR) apply

.PHONY: tofu-apply-auto
tofu-apply-auto:  ## Provision without interactive confirmation (CI use)
	tofu -chdir=$(INFRA_DIR) apply -auto-approve

.PHONY: tofu-destroy
tofu-destroy:  ## Destroy the instance and all infra (prompts for confirmation)
	tofu -chdir=$(INFRA_DIR) destroy

.PHONY: tofu-output
tofu-output:  ## Print instance connection details
	tofu -chdir=$(INFRA_DIR) output

.PHONY: connect
connect:  ## Open an SSH shell on the provisioned instance
	bash scripts/connect_inf2.sh

.PHONY: connect-setup
connect-setup:  ## Bootstrap the instance, then open a shell
	chmod +x scripts/connect_inf2.sh
	bash scripts/connect_inf2.sh --setup

.PHONY: connect-test
connect-test:  ## Run tests on the instance
	chmod +x scripts/connect_inf2.sh
	bash scripts/connect_inf2.sh --test

.PHONY: connect-bench
connect-bench:  ## Run benchmarks on the instance
	chmod +x scripts/connect_inf2.sh
	bash scripts/connect_inf2.sh --bench

.PHONY: tfvars
tvars:  ## Copy the example tfvars file (edit before running tofu-apply)
	cp $(INFRA_DIR)/terraform.tfvars.example $(INFRA_DIR)/terraform.tfvars
	@echo "Edit $(INFRA_DIR)/terraform.tfvars then run: make tofu-apply"

# ---------------------------------------------------------------------------
# Build / publish
# ---------------------------------------------------------------------------

.PHONY: build
build:  ## Build sdist + wheel into dist/
	$(UV) build

.PHONY: clean
clean:  ## Remove build artefacts, caches, and venv
	rm -rf dist/ build/ *.egg-info .eggs/
	rm -rf .venv/ .pytest_cache/ .mypy_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.neff" -delete 2>/dev/null || true

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.PHONY: help
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' \
	    | sort
