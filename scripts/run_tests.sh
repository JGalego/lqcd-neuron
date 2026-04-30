#!/usr/bin/env bash
# scripts/run_tests.sh
#
# Runs the full test suite and (optionally) the benchmarks on a Neuron instance.
#
# Usage:
#   bash scripts/run_tests.sh                   # pytest only
#   bash scripts/run_tests.sh --bench           # pytest + CPU benchmarks
#   bash scripts/run_tests.sh --bench --neuron  # pytest + Neuron benchmarks
#
# Exit codes:
#   0 — all tests passed (and benchmarks completed without error)
#   1 — one or more tests failed
#
# Environment variables:
#   LQCD_PYTEST_OPTS   -- extra options forwarded to pytest (e.g. "-x -q")
#   LQCD_BENCH_OUTPUT  -- path to write benchmark results (default: stdout)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
cd "${REPO_ROOT}"

# Auto-activate the project venv when not already inside one.
if [[ -z "${VIRTUAL_ENV:-}" && -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/.venv/bin/activate"
fi

RUN_BENCH=0
USE_NEURON=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bench)  RUN_BENCH=1;  shift ;;
        --neuron) USE_NEURON=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PYTEST_OPTS="${LQCD_PYTEST_OPTS:--v}"

# ---------------------------------------------------------------------------
# 1. Verify the environment is activated
# ---------------------------------------------------------------------------
if ! python3 -c "import lqcd_neuron" 2>/dev/null; then
    echo "[run_tests] ERROR: lqcd_neuron not importable."
    echo "  Run: source .venv/bin/activate  (or the DLAMI Neuron venv)"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Print environment info
# ---------------------------------------------------------------------------
echo "=== Environment ==="
python3 - <<'EOF'
import sys, torch
print(f"  Python      : {sys.version.split()[0]}")
print(f"  PyTorch     : {torch.__version__}")
try:
    import torch_neuronx
    print(f"  torch-neuronx: {torch_neuronx.__version__}")
except ImportError:
    print("  torch-neuronx: not installed (CPU mode)")
from lqcd_neuron.neuron import get_device
d = get_device()
print(f"  NeuronDevice: {d}")
EOF
echo ""

# ---------------------------------------------------------------------------
# 3. Run pytest
# ---------------------------------------------------------------------------
echo "=== Unit Tests ==="
# shellcheck disable=SC2086
pytest tests/ ${PYTEST_OPTS}
echo ""

# ---------------------------------------------------------------------------
# 4. Run examples as smoke tests
# ---------------------------------------------------------------------------
echo "=== Example Smoke Tests ==="
for script in examples/01_plaquette.py examples/02_wilson_dslash.py examples/03_cg_inversion.py; do
    echo "--- ${script} ---"
    python3 "${script}"
    echo ""
done

# ---------------------------------------------------------------------------
# 5. Benchmarks (optional)
# ---------------------------------------------------------------------------
if [[ "${RUN_BENCH}" -eq 1 ]]; then
    echo "=== Dslash Throughput Benchmark ==="
    BENCH_CMD="python3 examples/bench_dslash.py"
    if [[ "${USE_NEURON}" -eq 1 ]]; then
        BENCH_CMD="${BENCH_CMD} --neuron"
    fi

    if [[ -n "${LQCD_BENCH_OUTPUT:-}" ]]; then
        ${BENCH_CMD} | tee "${LQCD_BENCH_OUTPUT}"
    else
        ${BENCH_CMD}
    fi
fi

echo "=== All steps completed successfully ==="
