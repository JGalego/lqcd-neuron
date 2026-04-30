#!/usr/bin/env bash
# scripts/connect_inf2.sh
#
# Connect to the Inf2 instance provisioned by OpenTofu, optionally running
# the bootstrap and test scripts remotely.
#
# Usage:
#   bash scripts/connect_inf2.sh                       # interactive SSH shell
#   bash scripts/connect_inf2.sh --setup               # bootstrap instance, then shell
#   bash scripts/connect_inf2.sh --setup --test        # bootstrap + run tests
#   bash scripts/connect_inf2.sh --setup --bench       # bootstrap + full benchmark
#   bash scripts/connect_inf2.sh --ssm                 # connect via SSM (no port 22)
#
# Prerequisites:
#   - OpenTofu apply has been run (cd infra && tofu apply)
#   - aws CLI is configured with credentials that can describe the instance
#   - For --ssm: aws CLI v2 + Session Manager plugin installed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INFRA_DIR="${REPO_ROOT}/infra"
RUN_SETUP=0
RUN_TESTS=0
RUN_BENCH=0
USE_SSM=0
EXTRA_SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30"

DLAMI_VENV="/opt/aws_neuronx_venv_pytorch_2_8"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --setup)  RUN_SETUP=1; shift ;;
        --test)   RUN_TESTS=1; shift ;;
        --bench)  RUN_BENCH=1; shift ;;
        --ssm)    USE_SSM=1;   shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Read outputs from OpenTofu state
# ---------------------------------------------------------------------------
if ! command -v tofu &>/dev/null; then
    echo "ERROR: 'tofu' not found. Install OpenTofu: https://opentofu.org/docs/intro/install/"
    exit 1
fi

echo "[connect] Reading OpenTofu outputs from ${INFRA_DIR} …"
INSTANCE_ID=$(tofu -chdir="${INFRA_DIR}" output -raw instance_id   2>/dev/null)
PUBLIC_DNS=$(tofu  -chdir="${INFRA_DIR}" output -raw public_dns     2>/dev/null)
KEY_PATH=$(tofu    -chdir="${INFRA_DIR}" output -raw private_key_path 2>/dev/null)
AWS_REGION=$(tofu  -chdir="${INFRA_DIR}" output -raw aws_region     2>/dev/null || \
             tofu  -chdir="${INFRA_DIR}" output -json | python3 -c \
             "import sys,json; d=json.load(sys.stdin); print(d.get('aws_region',{}).get('value','us-east-2'))")

echo "[connect] Instance : ${INSTANCE_ID}"
echo "[connect] Host     : ${PUBLIC_DNS}"
echo "[connect] Key      : ${KEY_PATH}"

# ---------------------------------------------------------------------------
# 2. Wait until the instance is reachable
# ---------------------------------------------------------------------------
wait_for_ssh() {
    local host="$1" key="$2"
    echo "[connect] Waiting for SSH on ${host} …"
    for i in $(seq 1 30); do
        if ssh -i "${key}" ${EXTRA_SSH_OPTS} \
               -o BatchMode=yes \
               "ubuntu@${host}" "true" 2>/dev/null; then
            echo "[connect] SSH is up."
            return 0
        fi
        echo "  attempt ${i}/30 — retrying in 10 s"
        sleep 10
    done
    echo "ERROR: Timed out waiting for SSH."
    exit 1
}

# ---------------------------------------------------------------------------
# 3. SSM path (no port 22 required)
# ---------------------------------------------------------------------------
if [[ "${USE_SSM}" -eq 1 ]]; then
    echo "[connect] Connecting via SSM Session Manager …"
    aws ssm start-session \
        --target "${INSTANCE_ID}" \
        --region "${AWS_REGION}"
    exit 0
fi

# ---------------------------------------------------------------------------
# 4. SSH path
# ---------------------------------------------------------------------------
wait_for_ssh "${PUBLIC_DNS}" "${KEY_PATH}"

SSH="ssh -i ${KEY_PATH} ${EXTRA_SSH_OPTS} ubuntu@${PUBLIC_DNS}"

# ---------------------------------------------------------------------------
# 5. Bootstrap (idempotent — setup_inf2.sh checks what's already done)
# ---------------------------------------------------------------------------
if [[ "${RUN_SETUP}" -eq 1 ]]; then
    echo "[connect] Running bootstrap script on the instance …"
    $SSH "bash -s" < "${REPO_ROOT}/scripts/setup_inf2.sh"
fi

# ---------------------------------------------------------------------------
# 6. Tests (optional)
# ---------------------------------------------------------------------------
if [[ "${RUN_TESTS}" -eq 1 ]]; then
    BENCH_FLAG=""
    [[ "${RUN_BENCH}" -eq 1 ]] && BENCH_FLAG="--bench --neuron"

    echo "[connect] Running test suite on the instance …"
    $SSH <<REMOTE
set -euo pipefail
[[ -d "\${DLAMI_VENV}" ]] && source "\${DLAMI_VENV}/bin/activate" || source ~/lqcd-neuron/.venv/bin/activate
cd ~/lqcd-neuron
bash scripts/run_tests.sh ${BENCH_FLAG}
REMOTE
fi

# ---------------------------------------------------------------------------
# 7. Interactive shell (default if no --test/--bench)
# ---------------------------------------------------------------------------
if [[ "${RUN_TESTS}" -eq 0 && "${RUN_BENCH}" -eq 0 ]]; then
    echo "[connect] Opening interactive shell …"
    echo "[connect] Tip: run 'bash scripts/run_tests.sh --bench --neuron' once inside."
    exec $SSH
fi
