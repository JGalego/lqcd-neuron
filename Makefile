.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Virtual-environment detection
# Pick the first activate script that exists, in priority order:
#   1. An already-active venv (VIRTUAL_ENV set by the shell)
#   2. AWS Neuron venv  (aws_neuron_venv_p38)
#   3. uv / pip venv    (.venv)
#   4. Plain venv       (venv)
# Fall back to the uv-managed .venv path so 'make venv' still works.
# ---------------------------------------------------------------------------

ifneq ($(VIRTUAL_ENV),)
  # Already inside an activated venv — no sourcing needed.
  VENV_ACTIVATE := $(VIRTUAL_ENV)/bin/activate
else ifneq ($(wildcard /opt/aws_neuronx_venv_pytorch_2_8/bin/activate),)
  VENV_ACTIVATE := /opt/aws_neuronx_venv_pytorch_2_8/bin/activate
else ifneq ($(wildcard .venv/bin/activate),)
  VENV_ACTIVATE := .venv/bin/activate
else ifneq ($(wildcard venv/bin/activate),)
  VENV_ACTIVATE := venv/bin/activate
else
  VENV_ACTIVATE := .venv/bin/activate
endif

PYTHON := python3
UV     := uv

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
#
# Options (all optional):
#   NEURON=1              compile and run on NeuronCores (default: 0 = CPU only)
#   NO_FUSED=1            disable fused (Ns*Nc)^2 kernels — A/B diagnostic
#   LATTICE=TxZxYxX       benchmark only this lattice size
#   LATTICE="A B C"       benchmark multiple specific sizes (space-separated)
#   BATCH=B1,B2,...       sweep multi-RHS batch sizes (default: 1,8,32)
#
# Examples:
#   make bench
#   make bench NEURON=1
#   make bench NEURON=1 NO_FUSED=1
#   make bench NEURON=1 LATTICE=16x8x8x8
#   make bench NEURON=1 LATTICE="8x8x8x4 16x16x16x16" NO_FUSED=1
#   make bench NEURON=1 BATCH=1,8,32,64
# ---------------------------------------------------------------------------

NEURON   ?= 0
NO_FUSED ?= 0
LATTICE  ?=
BATCH    ?=

_BENCH_FLAGS  = $(if $(filter 1,$(NEURON)),--neuron)
_BENCH_FLAGS += $(if $(filter 1,$(NO_FUSED)),--no-fused)
_BENCH_FLAGS += $(foreach l,$(LATTICE),--lattice $(l))
_BENCH_FLAGS += $(if $(BATCH),--batch-sizes $(BATCH))

.PHONY: bench
bench:  ## Dslash throughput benchmark [NEURON=1] [NO_FUSED=1] [LATTICE=...] [BATCH=1,8,32]
	@echo "Bench config: NEURON=$(NEURON)  NO_FUSED=$(NO_FUSED)  LATTICE=$(if $(LATTICE),$(LATTICE),(all))  BATCH=$(if $(BATCH),$(BATCH),(default))"
	@echo "Flags:        $(_BENCH_FLAGS)"
	@echo ""
	. $(VENV_ACTIVATE) && $(PYTHON) examples/bench_dslash.py $(_BENCH_FLAGS)

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
connect-bench:  ## Run benchmarks on the instance [NEURON=1] [NO_FUSED=1] [LATTICE=...] [BATCH=...]
	chmod +x scripts/connect_inf2.sh
	bash scripts/connect_inf2.sh --bench $(if $(filter 1,$(NEURON)),--neuron) $(if $(filter 1,$(NO_FUSED)),--no-fused) $(foreach l,$(LATTICE),--lattice $(l)) $(if $(BATCH),--batch-sizes $(BATCH))

# ---------------------------------------------------------------------------
# Benchmark job (fire-and-forget, results emailed via SNS)
#
# Options:
#   MODE=persistent   (default) reuse the existing Inf2 instance via SSM
#   MODE=ephemeral    spin up a one-shot Inf2 that auto-terminates
#   WAIT=1            block until the SSM command finishes (persistent only)
#   NEURON=1          add --neuron to the bench (default: on)
#   NO_FUSED=1        add --no-fused
#   LATTICE="A B"     restrict to specific lattice sizes
#
# Examples:
#   make bench-job
#   make bench-job MODE=ephemeral
#   make bench-job NEURON=1 LATTICE="16x16x16x16 24x24x24x24"
#   make bench-job MODE=ephemeral NO_FUSED=1
# ---------------------------------------------------------------------------

MODE   ?= ephemeral
WAIT   ?= 0
# Default to --neuron unless explicitly disabled with NEURON=0.
NEURON ?= 1

_JOB_FLAGS  = $(if $(filter 1,$(NEURON)),--neuron)
_JOB_FLAGS += $(if $(filter 1,$(NO_FUSED)),--no-fused)
_JOB_FLAGS += $(foreach l,$(LATTICE),--lattice $(l))
_JOB_FLAGS += $(if $(BATCH),--batch-sizes $(BATCH))

.PHONY: bench-job
bench-job:  ## Trigger a bench job (results emailed) [MODE=persistent|ephemeral] [WAIT=1]
	chmod +x scripts/trigger_bench_job.sh
	bash scripts/trigger_bench_job.sh \
	    --mode $(MODE) \
	    $(if $(filter 1,$(WAIT)),--wait) \
	    -- $(_JOB_FLAGS)

# ---------------------------------------------------------------------------
# Inspect partial results streamed by an in-flight bench-job to S3.
#   make bench-runs                       # list runs in the bucket
#   make bench-tail RUN=<run_id>          # tail per-lattice JSONL
#   make bench-tail RUN=<run_id> LOG=1    # tail the running bench.log
# ---------------------------------------------------------------------------
_BENCH_BUCKET = $$(tofu -chdir=$(INFRA_DIR) output -raw bench_s3_bucket)
_BENCH_REGION = $$(tofu -chdir=$(INFRA_DIR) output -raw aws_region)

.PHONY: bench-runs
bench-runs:  ## List bench runs uploaded to the S3 bucket
	aws s3 ls --region $(_BENCH_REGION) "s3://$(_BENCH_BUCKET)/runs/"

.PHONY: bench-tail
bench-tail:  ## Tail partial results of an in-flight run [RUN=<id>] [LOG=1]
	@if [ -z "$(RUN)" ]; then echo "Usage: make bench-tail RUN=<run_id> [LOG=1]"; exit 2; fi
	@if [ "$(LOG)" = "1" ]; then \
	    aws s3 cp --region $(_BENCH_REGION) \
	        "s3://$(_BENCH_BUCKET)/runs/$(RUN)/bench.log.partial" -; \
	else \
	    aws s3 cp --region $(_BENCH_REGION) \
	        "s3://$(_BENCH_BUCKET)/runs/$(RUN)/partial/results.jsonl" -; \
	fi

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
