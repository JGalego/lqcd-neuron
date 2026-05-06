#!/usr/bin/env bash
# scripts/run_bench_job.sh
#
# On-instance benchmark job: runs examples/bench_dslash.py, archives the
# log to S3, and publishes a summary (with a presigned download URL) to
# the SNS topic configured on the instance.
#
# Inputs (env vars, with sensible fallbacks):
#   BENCH_ARGS          extra args to pass to bench_dslash.py
#                       (default: "--neuron")
#   SNS_TOPIC_ARN       SNS topic to publish to. If unset, derived from
#                       the BenchSnsTopicArn instance tag.
#   S3_BUCKET           S3 bucket for log upload. If unset, derived from
#                       the BenchS3Bucket instance tag.
#   AWS_REGION          AWS region. If unset, derived from IMDS.
#   EPHEMERAL           "1" => `sudo shutdown -h now` after publishing.
#   REPO_DIR            checkout location (default: ~/lqcd-neuron).
#   GIT_BRANCH          branch to pull (default: current branch, else main).
#
# Exit codes:
#   0  bench succeeded, email sent
#   1  bench failed (failure email still sent)
#   2  config error (no SNS topic / bucket resolvable)

set -uo pipefail

log() { echo "[bench-job] $*" >&2; }

# ---------------------------------------------------------------------------
# Resolve config from instance tags / IMDS when not provided
# ---------------------------------------------------------------------------
imds_token() {
    curl -sf -X PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 3600" 2>/dev/null
}
imds_get() {
    local token="$1" path="$2"
    curl -sf "http://169.254.169.254/latest/meta-data/${path}" \
        -H "X-aws-ec2-metadata-token: ${token}" 2>/dev/null
}

TOKEN=$(imds_token || true)
INSTANCE_ID=${INSTANCE_ID:-$(imds_get "${TOKEN}" "instance-id" || echo "unknown")}
INSTANCE_TYPE=${INSTANCE_TYPE:-$(imds_get "${TOKEN}" "instance-type" || echo "unknown")}
AVAIL_ZONE=$(imds_get "${TOKEN}" "placement/availability-zone" || echo "unknown")
AWS_REGION=${AWS_REGION:-${AVAIL_ZONE%[a-z]}}
AWS_REGION=${AWS_REGION:-us-east-2}

# Pull tags off the instance if SNS / S3 weren't provided directly
if [[ -z "${SNS_TOPIC_ARN:-}" || -z "${S3_BUCKET:-}" ]] \
   && [[ "${INSTANCE_ID}" != "unknown" ]] \
   && command -v aws >/dev/null 2>&1; then
    log "Resolving SNS topic / S3 bucket from instance tags …"
    TAGS_JSON=$(aws ec2 describe-tags \
        --region "${AWS_REGION}" \
        --filters "Name=resource-id,Values=${INSTANCE_ID}" \
        --output json 2>/dev/null || echo '{"Tags":[]}')
    SNS_TOPIC_ARN=${SNS_TOPIC_ARN:-$(echo "${TAGS_JSON}" \
        | python3 -c "import sys,json; t=json.load(sys.stdin)['Tags']; print(next((x['Value'] for x in t if x['Key']=='BenchSnsTopicArn'),''))")}
    S3_BUCKET=${S3_BUCKET:-$(echo "${TAGS_JSON}" \
        | python3 -c "import sys,json; t=json.load(sys.stdin)['Tags']; print(next((x['Value'] for x in t if x['Key']=='BenchS3Bucket'),''))")}
fi

if [[ -z "${SNS_TOPIC_ARN:-}" ]] || [[ -z "${S3_BUCKET:-}" ]]; then
    log "ERROR: SNS_TOPIC_ARN and/or S3_BUCKET could not be resolved."
    log "       Set them explicitly or ensure the instance has the"
    log "       BenchSnsTopicArn / BenchS3Bucket tags."
    exit 2
fi

BENCH_ARGS=${BENCH_ARGS:---neuron}
REPO_DIR=${REPO_DIR:-${HOME}/lqcd-neuron}
EPHEMERAL=${EPHEMERAL:-0}

DLAMI_VENV="/opt/aws_neuronx_venv_pytorch_2_8"

log "Region          : ${AWS_REGION}"
log "Instance        : ${INSTANCE_ID} (${INSTANCE_TYPE})"
log "SNS topic       : ${SNS_TOPIC_ARN}"
log "S3 bucket       : ${S3_BUCKET}"
log "Bench args      : ${BENCH_ARGS}"
log "Ephemeral       : ${EPHEMERAL}"

# ---------------------------------------------------------------------------
# Repo + venv
# ---------------------------------------------------------------------------
if [[ ! -d "${REPO_DIR}/.git" ]]; then
    log "Cloning repo into ${REPO_DIR} …"
    git clone "${LQCD_NEURON_REPO_URL:-https://github.com/JGalego/lqcd-neuron}" \
        "${REPO_DIR}"
fi
cd "${REPO_DIR}"
git fetch --quiet origin || true
GIT_BRANCH=${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}
git checkout --quiet "${GIT_BRANCH}" 2>/dev/null || true
git pull --ff-only --quiet origin "${GIT_BRANCH}" 2>/dev/null || true
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

if [[ -d "${DLAMI_VENV}" ]]; then
    # shellcheck disable=SC1091
    source "${DLAMI_VENV}/bin/activate"
    pip install -e "." --quiet >/dev/null 2>&1 || true
elif [[ -d "${REPO_DIR}/.venv" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_DIR}/.venv/bin/activate"
fi

# ---------------------------------------------------------------------------
# Collect host metadata
# ---------------------------------------------------------------------------
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ID="${TIMESTAMP}-${INSTANCE_ID}"
LOG_DIR=$(mktemp -d -t bench-job-XXXX)
LOG_FILE="${LOG_DIR}/bench.log"
META_FILE="${LOG_DIR}/meta.txt"

{
    echo "==== lqcd-neuron benchmark run ===="
    echo "run_id        : ${RUN_ID}"
    echo "timestamp_utc : $(date -u --iso-8601=seconds)"
    echo "instance_id   : ${INSTANCE_ID}"
    echo "instance_type : ${INSTANCE_TYPE}"
    echo "region        : ${AWS_REGION}"
    echo "az            : ${AVAIL_ZONE}"
    echo "hostname      : $(hostname)"
    echo "kernel        : $(uname -r)"
    echo "nproc         : $(nproc)"
    echo "git_sha       : ${GIT_SHA}"
    echo "git_branch    : ${GIT_BRANCH}"
    echo "bench_args    : ${BENCH_ARGS}"
    echo "python        : $(python3 --version 2>&1)"
    echo "torch         : $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo n/a)"
    echo "torch_neuronx : $(python3 -c 'import torch_neuronx; print(torch_neuronx.__version__)' 2>/dev/null || echo n/a)"
    echo
    echo "---- neuron-ls ----"
    if command -v neuron-ls >/dev/null 2>&1; then
        neuron-ls 2>&1 || true
    else
        echo "(neuron-ls not available)"
    fi
} | tee "${META_FILE}"

# ---------------------------------------------------------------------------
# Run the benchmark
# ---------------------------------------------------------------------------
log "Running: python examples/bench_dslash.py ${BENCH_ARGS}"
START_TS=$(date +%s)
set +e
# shellcheck disable=SC2086
python3 examples/bench_dslash.py ${BENCH_ARGS} 2>&1 | tee "${LOG_FILE}"
BENCH_EXIT=${PIPESTATUS[0]}
set -e
END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))

if [[ "${BENCH_EXIT}" -eq 0 ]]; then
    STATUS="OK"
else
    STATUS="FAILED (exit=${BENCH_EXIT})"
fi
log "Bench finished in ${DURATION}s — ${STATUS}"

# ---------------------------------------------------------------------------
# Combine meta + log, upload to S3, presign
# ---------------------------------------------------------------------------
FULL_LOG="${LOG_DIR}/full.log"
{
    cat "${META_FILE}"
    echo
    echo "duration_seconds : ${DURATION}"
    echo "status           : ${STATUS}"
    echo
    echo "---- bench output ----"
    cat "${LOG_FILE}"
} > "${FULL_LOG}"

S3_KEY="runs/${RUN_ID}/bench.log"
S3_URI="s3://${S3_BUCKET}/${S3_KEY}"
log "Uploading log to ${S3_URI}"
aws s3 cp --region "${AWS_REGION}" --quiet "${FULL_LOG}" "${S3_URI}" \
    || log "WARN: S3 upload failed (continuing so we still email)."

# 7-day presigned URL for one-click download from the email
PRESIGNED_URL=$(aws s3 presign --region "${AWS_REGION}" \
    --expires-in 604800 "${S3_URI}" 2>/dev/null || echo "")

# ---------------------------------------------------------------------------
# Build summary email body
# ---------------------------------------------------------------------------
SUMMARY_FILE="${LOG_DIR}/summary.txt"
{
    echo "lqcd-neuron benchmark — ${STATUS}"
    echo
    echo "run_id       : ${RUN_ID}"
    echo "instance     : ${INSTANCE_ID}  (${INSTANCE_TYPE}, ${AWS_REGION})"
    echo "git          : ${GIT_BRANCH} @ ${GIT_SHA}"
    echo "bench args   : ${BENCH_ARGS}"
    echo "duration     : ${DURATION}s"
    echo "log archive  : ${S3_URI}"
    if [[ -n "${PRESIGNED_URL}" ]]; then
        echo
        echo "Download (valid 7 days):"
        echo "  ${PRESIGNED_URL}"
    fi
    echo
    echo "---- neuron-ls (head) ----"
    head -n 20 "${META_FILE}" || true
    echo
    echo "---- bench output (tail) ----"
    # SNS email body is capped at 256 KB; keep it well under that.
    tail -c 60000 "${LOG_FILE}"
} > "${SUMMARY_FILE}"

SUBJECT="[lqcd-neuron] bench ${STATUS} — ${INSTANCE_TYPE} — ${TIMESTAMP}"
# SNS subject is capped at 100 chars and forbids newlines / control chars.
SUBJECT=$(echo "${SUBJECT}" | tr -d '\n\r' | cut -c1-100)

log "Publishing to SNS …"
aws sns publish \
    --region "${AWS_REGION}" \
    --topic-arn "${SNS_TOPIC_ARN}" \
    --subject "${SUBJECT}" \
    --message "file://${SUMMARY_FILE}" >/dev/null \
    || log "WARN: SNS publish failed."

log "Done."

# ---------------------------------------------------------------------------
# Ephemeral mode: terminate self
# ---------------------------------------------------------------------------
if [[ "${EPHEMERAL}" == "1" ]]; then
    log "EPHEMERAL=1 — shutting down (instance is configured to terminate)."
    sudo shutdown -h +1 "lqcd-neuron bench job complete" || true
fi

exit "${BENCH_EXIT}"
