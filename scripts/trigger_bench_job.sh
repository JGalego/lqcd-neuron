#!/usr/bin/env bash
# scripts/trigger_bench_job.sh
#
# Fire off a benchmark job on AWS and have the results emailed to the
# address subscribed to the SNS topic.  Two modes:
#
#   --mode persistent   (default)
#       Sends an SSM RunCommand to the existing Inf2 instance provisioned
#       by `tofu apply`.  The instance must be running.  Cheapest if you
#       already have it up.
#
#   --mode ephemeral
#       Launches a one-shot Inf2 from the bench launch template, runs the
#       benchmark, emails the results, and self-terminates
#       (instance_initiated_shutdown_behavior=terminate).  Use this when
#       the persistent instance is stopped or you want a clean run.
#
# Usage:
#   bash scripts/trigger_bench_job.sh                    # persistent, --neuron
#   bash scripts/trigger_bench_job.sh --mode ephemeral
#   bash scripts/trigger_bench_job.sh -- --no-fused --lattice 16x16x16x16
#
# Anything after `--` is forwarded verbatim to bench_dslash.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFRA_DIR="${REPO_ROOT}/infra"

MODE="persistent"
WAIT=0
BENCH_ARGS_DEFAULT="--neuron"
BENCH_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)        MODE="$2"; shift 2 ;;
        --persistent)  MODE="persistent"; shift ;;
        --ephemeral)   MODE="ephemeral"; shift ;;
        --wait)        WAIT=1; shift ;;
        --)            shift; BENCH_ARGS="$*"; break ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *)
            # Allow forwarding without explicit `--`
            BENCH_ARGS="${BENCH_ARGS} $1"; shift ;;
    esac
done

BENCH_ARGS="$(echo "${BENCH_ARGS:-${BENCH_ARGS_DEFAULT}}" | xargs)"

log() { echo "[trigger] $*"; }

command -v tofu >/dev/null 2>&1 || {
    echo "ERROR: 'tofu' not found.  Run 'make tofu-apply' first." >&2; exit 1; }
command -v aws >/dev/null 2>&1 || {
    echo "ERROR: AWS CLI v2 is required." >&2; exit 1; }

# ---------------------------------------------------------------------------
# Read OpenTofu outputs
# ---------------------------------------------------------------------------
AWS_REGION=$(tofu  -chdir="${INFRA_DIR}" output -raw aws_region)
SNS_ARN=$(tofu     -chdir="${INFRA_DIR}" output -raw bench_sns_topic_arn)
S3_BUCKET=$(tofu   -chdir="${INFRA_DIR}" output -raw bench_s3_bucket)

case "${MODE}" in
    persistent)
        INSTANCE_ID=$(tofu -chdir="${INFRA_DIR}" output -raw instance_id 2>/dev/null || true)
        if [[ -z "${INSTANCE_ID}" || "${INSTANCE_ID}" == "null" ]]; then
            cat >&2 <<EOF
ERROR: No persistent instance is provisioned (instance_id output is null).

The infra defaults to skip_persistent_instance = true, which provisions
only the shared bench infra.  Either:

  - run an ephemeral one-shot:   make bench-job MODE=ephemeral
  - or enable the persistent box: set 'skip_persistent_instance = false'
                                  in infra/terraform.tfvars and re-apply.
EOF
            exit 1
        fi
        log "Mode        : persistent"
        log "Instance    : ${INSTANCE_ID}"
        ;;
    ephemeral)
        LAUNCH_TEMPLATE_ID=$(tofu -chdir="${INFRA_DIR}" output -raw bench_launch_template_id)
        log "Mode        : ephemeral"
        log "Launch tmpl : ${LAUNCH_TEMPLATE_ID}"
        ;;
    *)
        echo "ERROR: --mode must be 'persistent' or 'ephemeral'" >&2; exit 1 ;;
esac
log "Region      : ${AWS_REGION}"
log "SNS topic   : ${SNS_ARN}"
log "S3 bucket   : ${S3_BUCKET}"
log "Bench args  : ${BENCH_ARGS}"

# ---------------------------------------------------------------------------
# Persistent: send via SSM RunCommand
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "persistent" ]]; then
    # We embed run_bench_job.sh into the document so the on-instance copy
    # doesn't have to be in sync with this branch.  The script reads
    # BENCH_ARGS / SNS_TOPIC_ARN / S3_BUCKET / AWS_REGION from the env.
    SCRIPT_PATH="${REPO_ROOT}/scripts/run_bench_job.sh"
    [[ -f "${SCRIPT_PATH}" ]] || {
        echo "ERROR: ${SCRIPT_PATH} missing" >&2; exit 1; }

    # Quote values once locally so they survive embedding in the document.
    Q_BENCH_ARGS=$(printf '%q' "${BENCH_ARGS}")
    Q_SNS_ARN=$(printf    '%q' "${SNS_ARN}")
    Q_S3_BUCKET=$(printf  '%q' "${S3_BUCKET}")
    Q_AWS_REGION=$(printf '%q' "${AWS_REGION}")

    REMOTE_CMD=$(cat <<REMOTE
set -euo pipefail
cat > /tmp/run_bench_job.sh <<'BENCH_EOF'
$(cat "${SCRIPT_PATH}")
BENCH_EOF
chmod +x /tmp/run_bench_job.sh
sudo -u ubuntu -H \
    BENCH_ARGS=${Q_BENCH_ARGS} \
    SNS_TOPIC_ARN=${Q_SNS_ARN} \
    S3_BUCKET=${Q_S3_BUCKET} \
    AWS_REGION=${Q_AWS_REGION} \
    bash /tmp/run_bench_job.sh
REMOTE
)

    PAYLOAD=$(python3 -c '
import json, sys
cmd = sys.stdin.read()
print(json.dumps({"commands": [cmd], "executionTimeout": ["7200"]}))
' <<<"${REMOTE_CMD}")

    log "Sending SSM RunCommand …"
    CMD_ID=$(aws ssm send-command \
        --region "${AWS_REGION}" \
        --instance-ids "${INSTANCE_ID}" \
        --document-name "AWS-RunShellScript" \
        --comment "lqcd-neuron bench job" \
        --cloud-watch-output-config '{"CloudWatchOutputEnabled":true}' \
        --parameters "${PAYLOAD}" \
        --query "Command.CommandId" \
        --output text)
    log "Command ID  : ${CMD_ID}"
    log "An email will be sent to the SNS-subscribed address when the run finishes."

    if [[ "${WAIT}" -eq 1 ]]; then
        log "Waiting for completion (this can take a while) …"
        aws ssm wait command-executed \
            --region "${AWS_REGION}" \
            --command-id "${CMD_ID}" \
            --instance-id "${INSTANCE_ID}" || true
        aws ssm get-command-invocation \
            --region "${AWS_REGION}" \
            --command-id "${CMD_ID}" \
            --instance-id "${INSTANCE_ID}" \
            --query "{Status:Status,StatusDetails:StatusDetails}" \
            --output table
    else
        log "Tail with: aws ssm get-command-invocation \\"
        log "    --region ${AWS_REGION} --command-id ${CMD_ID} \\"
        log "    --instance-id ${INSTANCE_ID}"
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# Ephemeral: launch from the bench launch template with one-shot user-data
# ---------------------------------------------------------------------------
REPO_URL="${LQCD_NEURON_REPO_URL:-https://github.com/JGalego/lqcd-neuron}"
BRANCH="${LQCD_NEURON_BRANCH:-main}"

# Quote each value once on the local side so it lands in the user-data as
# a single shell token regardless of spaces / metacharacters.
Q_BENCH_ARGS=$(printf '%q' "${BENCH_ARGS}")
Q_SNS_ARN=$(printf   '%q' "${SNS_ARN}")
Q_S3_BUCKET=$(printf '%q' "${S3_BUCKET}")
Q_AWS_REGION=$(printf '%q' "${AWS_REGION}")
Q_REPO_URL=$(printf  '%q' "${REPO_URL}")
Q_BRANCH=$(printf    '%q' "${BRANCH}")

USER_DATA=$(cat <<EOF
#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/lqcd-bench-userdata.log) 2>&1
echo "[user-data] starting at \$(date -u --iso-8601=seconds)"

REPO_URL=${Q_REPO_URL}
BRANCH=${Q_BRANCH}

# Run the bench as the ubuntu user.  We clone the repo fresh so the
# instance picks up whatever branch was requested at trigger time.
sudo -u ubuntu -H \\
    BENCH_ARGS=${Q_BENCH_ARGS} \\
    SNS_TOPIC_ARN=${Q_SNS_ARN} \\
    S3_BUCKET=${Q_S3_BUCKET} \\
    AWS_REGION=${Q_AWS_REGION} \\
    EPHEMERAL=1 \\
    LQCD_NEURON_REPO_URL="\${REPO_URL}" \\
    GIT_BRANCH="\${BRANCH}" \\
    bash -c '
        set -euo pipefail
        cd "\$HOME"
        if [[ ! -d lqcd-neuron/.git ]]; then
            git clone --branch "\$GIT_BRANCH" "\$LQCD_NEURON_REPO_URL" lqcd-neuron
        fi
        cd lqcd-neuron
        git fetch --quiet origin "\$GIT_BRANCH" || true
        git checkout --quiet "\$GIT_BRANCH"
        git reset --hard "origin/\$GIT_BRANCH"
        export REPO_DIR="\$HOME/lqcd-neuron"
        bash scripts/run_bench_job.sh
    '

# Belt-and-braces self-termination in case run_bench_job.sh's shutdown failed.
shutdown -h +5 "lqcd-neuron ephemeral bench complete" || true
EOF
)

# AWS expects user-data as base64 when passed via --user-data with the
# CLI (it's actually the raw text, but base64 sidesteps shell quoting).
USER_DATA_B64=$(printf '%s' "${USER_DATA}" | base64 -w0)

log "Launching one-shot Inf2 from launch template ${LAUNCH_TEMPLATE_ID} …"
INSTANCE_JSON=$(aws ec2 run-instances \
    --region "${AWS_REGION}" \
    --launch-template "LaunchTemplateId=${LAUNCH_TEMPLATE_ID}" \
    --user-data "${USER_DATA_B64}" \
    --output json)

EPHEMERAL_ID=$(echo "${INSTANCE_JSON}" | python3 -c 'import sys,json; print(json.load(sys.stdin)["Instances"][0]["InstanceId"])')
log "Launched     : ${EPHEMERAL_ID}"
log "Tagging      : Lifecycle=ephemeral, BenchTriggeredAt=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
aws ec2 create-tags \
    --region "${AWS_REGION}" \
    --resources "${EPHEMERAL_ID}" \
    --tags "Key=BenchTriggeredAt,Value=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >/dev/null

log "The instance will email results, then self-terminate."
log "Watch progress with:"
log "  aws ec2 describe-instance-status --region ${AWS_REGION} --instance-ids ${EPHEMERAL_ID}"
