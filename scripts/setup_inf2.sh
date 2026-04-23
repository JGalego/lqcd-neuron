#!/usr/bin/env bash
# scripts/setup_inf2.sh
#
# One-shot bootstrap for an Inf2 / Trn1 instance.
#
# Usage:
#   bash scripts/setup_inf2.sh [--repo-url <git-url>] [--branch <branch>]
#
# What it does:
#   1. Activates the DLAMI Neuron virtualenv (or creates a fresh uv venv
#      on non-DLAMI Ubuntu / Amazon Linux 2023 instances).
#   2. Clones / updates the lqcd-neuron repository.
#   3. Installs the package with the [neuron] extras.
#   4. Exports INSTANCE_TYPE so the hardware-detection heuristic works.
#   5. Prints a quick sanity-check.
#
# Tested AMIs:
#   - Deep Learning AMI Neuron (Ubuntu 22.04) — uses DLAMI venv path
#   - Amazon Linux 2023 + manual Neuron SDK install
#
# Reference:
#   https://awsdocs-neuron.readthedocs-hosted.com/en/latest/
#     general/setup/torch-neuronx.html

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (override via CLI flags)
# ---------------------------------------------------------------------------
REPO_URL="${LQCD_NEURON_REPO_URL:-https://github.com/JGalego/lqcd-neuron}"
BRANCH="${LQCD_NEURON_BRANCH:-main}"
INSTALL_DIR="${HOME}/lqcd-neuron"

# DLAMI ships a ready-made Neuron virtualenv; fall back to uv otherwise.
DLAMI_VENV="/opt/aws_neuronx_venv_pytorch_2_8"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-url) REPO_URL="$2"; shift 2 ;;
        --branch)   BRANCH="$2";   shift 2 ;;
        --install-dir) INSTALL_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

log() { echo "[setup] $*"; }

# ---------------------------------------------------------------------------
# Step 1: Activate / create Python environment
# ---------------------------------------------------------------------------
if [[ -d "${DLAMI_VENV}" ]]; then
    log "DLAMI Neuron venv detected at ${DLAMI_VENV}"
    # shellcheck disable=SC1090
    source "${DLAMI_VENV}/bin/activate"
    USING_DLAMI=1
else
    log "DLAMI venv not found — creating fresh env with uv"
    command -v uv >/dev/null 2>&1 || \
        curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.cargo/bin:${PATH}"

    uv venv "${INSTALL_DIR}/.venv"
    # shellcheck disable=SC1090
    source "${INSTALL_DIR}/.venv/bin/activate"
    USING_DLAMI=0
fi

# ---------------------------------------------------------------------------
# Step 2: Clone / update repository
# ---------------------------------------------------------------------------
if [[ -d "${INSTALL_DIR}/.git" ]]; then
    log "Updating existing repo at ${INSTALL_DIR}"
    git -C "${INSTALL_DIR}" fetch origin
    git -C "${INSTALL_DIR}" checkout "${BRANCH}"
    git -C "${INSTALL_DIR}" pull --ff-only origin "${BRANCH}"
else
    log "Cloning ${REPO_URL} (branch: ${BRANCH}) → ${INSTALL_DIR}"
    git clone --branch "${BRANCH}" "${REPO_URL}" "${INSTALL_DIR}"
fi

cd "${INSTALL_DIR}"

# ---------------------------------------------------------------------------
# Step 3: Install package
# ---------------------------------------------------------------------------
if [[ "${USING_DLAMI}" -eq 1 ]]; then
    # On DLAMI, torch-neuronx is already installed; skip the [neuron] extra
    # to avoid version conflicts.
    log "Installing lqcd-neuron (core only — DLAMI provides torch-neuronx)"
    pip install -e "." --quiet
else
    log "Installing lqcd-neuron[neuron]"
    uv pip install -e ".[neuron]"
fi

# ---------------------------------------------------------------------------
# Step 4: Export INSTANCE_TYPE for hardware detection heuristic
# ---------------------------------------------------------------------------
# Try the EC2 instance-metadata service (IMDSv2, no auth required for
# the instance-type endpoint when called from within the instance).
IMDS_TOKEN=$(curl -sf -X PUT \
    "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null || true)

if [[ -n "${IMDS_TOKEN}" ]]; then
    INSTANCE_TYPE=$(curl -sf \
        "http://169.254.169.254/latest/meta-data/instance-type" \
        -H "X-aws-ec2-metadata-token: ${IMDS_TOKEN}" 2>/dev/null || echo "unknown")
    export INSTANCE_TYPE
    log "Instance type from IMDS: ${INSTANCE_TYPE}"
else
    log "IMDS not reachable — set INSTANCE_TYPE manually if needed"
fi

# ---------------------------------------------------------------------------
# Step 5: Quick sanity check
# ---------------------------------------------------------------------------
log "Running sanity check …"
python3 - <<'EOF'
from lqcd_neuron.neuron import get_device
from lqcd_neuron.core import LatticeGeometry, GaugeField
from lqcd_neuron.observables import plaquette

d = get_device()
print(f"  NeuronDevice : {d}")
print(f"  is_neuron    : {d.is_neuron}")

geom  = LatticeGeometry(T=4, Z=4, Y=4, X=4)
U_cold = GaugeField.cold(geom)
P = plaquette(U_cold)
print(f"  Cold plaquette: {P:.10f}  (expected 1.0)")
assert abs(P - 1.0) < 1e-5, "Plaquette sanity check failed!"
print("  ✓ All OK")
EOF

log "Setup complete.  Activate the env with:"
if [[ "${USING_DLAMI}" -eq 1 ]]; then
    echo "  source ${DLAMI_VENV}/bin/activate"
else
    echo "  source ${INSTALL_DIR}/.venv/bin/activate"
fi
