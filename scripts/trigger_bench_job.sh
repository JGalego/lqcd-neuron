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
# Pre-flight: refuse to launch with a dirty tree or unpushed commits.
# The remote bench instance always pulls from origin/<branch>, so anything
# that isn't pushed is invisible to it. Skip with LQCD_SKIP_GIT_CHECK=1.
# ---------------------------------------------------------------------------
if [[ "${LQCD_SKIP_GIT_CHECK:-0}" != "1" ]]; then
    if ! git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        log "WARN: ${REPO_ROOT} is not a git checkout, skipping git pre-flight."
    else
        BRANCH_LOCAL="${LQCD_NEURON_BRANCH:-$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)}"
        if ! git -C "${REPO_ROOT}" diff --quiet \
            || ! git -C "${REPO_ROOT}" diff --cached --quiet; then
            echo "ERROR: working tree has uncommitted changes on '${BRANCH_LOCAL}'." >&2
            echo "       Commit/stash them, or re-run with LQCD_SKIP_GIT_CHECK=1." >&2
            exit 1
        fi
        log "Fetching origin/${BRANCH_LOCAL} to check for unpushed commits …"
        if git -C "${REPO_ROOT}" fetch --quiet origin "${BRANCH_LOCAL}" 2>/dev/null; then
            AHEAD=$(git -C "${REPO_ROOT}" rev-list --count "origin/${BRANCH_LOCAL}..HEAD" 2>/dev/null || echo 0)
            if [[ "${AHEAD}" -gt 0 ]]; then
                echo "ERROR: local '${BRANCH_LOCAL}' is ${AHEAD} commit(s) ahead of origin." >&2
                echo "       Push first, or re-run with LQCD_SKIP_GIT_CHECK=1." >&2
                git -C "${REPO_ROOT}" log --oneline "origin/${BRANCH_LOCAL}..HEAD" >&2 || true
                exit 1
            fi
        else
            log "WARN: could not fetch origin/${BRANCH_LOCAL}; skipping ahead-check."
        fi
        log "Git pre-flight OK (clean tree, no unpushed commits on ${BRANCH_LOCAL})."
    fi
fi

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
    log "Attach a shell with:"
    log "  aws ssm start-session --region ${AWS_REGION} --target ${INSTANCE_ID}"

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

# Embed the on-instance bench script directly so the ephemeral box does not
# depend on the orchestration script being pushed (only the bench code does).
SCRIPT_PATH="${REPO_ROOT}/scripts/run_bench_job.sh"
[[ -f "${SCRIPT_PATH}" ]] || {
    echo "ERROR: ${SCRIPT_PATH} missing" >&2; exit 1; }
RUN_BENCH_JOB_SH=$(cat "${SCRIPT_PATH}")

USER_DATA=$(cat <<EOF
#!/bin/bash
exec > >(tee /var/log/lqcd-bench-userdata.log) 2>&1
echo "[user-data] starting at \$(date -u --iso-8601=seconds)"

# Hard wallclock kill switch: no matter what wedges below, the instance
# dies in 2 hours.  Cancellable from inside the workload via 'shutdown -c'
# if a longer run is ever needed.
shutdown -h +120 "lqcd-neuron bench wallclock" || true

# Belt-and-braces teardown: fires on success, error, or signal.  Replaces
# the old trailing 'shutdown' which 'set -e' could skip on bootstrap failure.
_lqcd_teardown() {
    rc=\$?
    echo "[user-data] teardown rc=\${rc} at \$(date -u --iso-8601=seconds)"
    if [[ "\${rc}" -ne 0 ]] && [[ "\${LQCD_SNS_NOTIFIED:-0}" != "1" ]]; then
        # Bootstrap failed before run_bench_job.sh could publish a result.
        # Send a short failure ping so the run is not silently lost.
        aws sns publish \\
            --region ${Q_AWS_REGION} \\
            --topic-arn ${Q_SNS_ARN} \\
            --subject "[lqcd-neuron] bench BOOTSTRAP FAILED (rc=\${rc})" \\
            --message "\$(tail -c 8000 /var/log/lqcd-bench-userdata.log 2>/dev/null \\
                || echo 'no log available')" >/dev/null 2>&1 || true
    fi
    shutdown -h +2 "lqcd-neuron ephemeral bench done" || true
}
trap _lqcd_teardown EXIT
set -uo pipefail

REPO_URL=${Q_REPO_URL}
BRANCH=${Q_BRANCH}

# Drop the embedded run_bench_job.sh in /tmp so it runs even if the
# orchestration changes haven't been pushed to the remote branch yet.
cat > /tmp/run_bench_job.sh <<'BENCH_EOF'
${RUN_BENCH_JOB_SH}
BENCH_EOF
chmod +x /tmp/run_bench_job.sh
chown ubuntu:ubuntu /tmp/run_bench_job.sh

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
        set -uo pipefail
        cd "\$HOME"
        if [[ ! -d lqcd-neuron/.git ]]; then
            git clone --branch "\$GIT_BRANCH" "\$LQCD_NEURON_REPO_URL" lqcd-neuron
        fi
        cd lqcd-neuron
        git fetch --quiet origin "\$GIT_BRANCH" || true
        git checkout --quiet "\$GIT_BRANCH"
        git reset --hard "origin/\$GIT_BRANCH"
        export REPO_DIR="\$HOME/lqcd-neuron"
        bash /tmp/run_bench_job.sh
    '
BENCH_RC=\$?
# run_bench_job.sh handles its own SNS publish on success or bench failure;
# only the trap needs to fire for bootstrap-level failures.
if [[ "\${BENCH_RC}" -le 1 ]]; then
    export LQCD_SNS_NOTIFIED=1
fi
exit "\${BENCH_RC}"
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
log "Attach a shell with:"
log "  aws ssm start-session --region ${AWS_REGION} --target ${EPHEMERAL_ID}"
