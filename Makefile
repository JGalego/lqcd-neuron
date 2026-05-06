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
#   WALLCLOCK=N       ephemeral wallclock kill switch in minutes
#                     (default 120; set 0 to disable for long sweeps)
#
# Examples:
#   make bench-job
#   make bench-job MODE=ephemeral
#   make bench-job NEURON=1 LATTICE="16x16x16x16 24x24x24x24"
#   make bench-job MODE=ephemeral NO_FUSED=1
#   make bench-job MODE=ephemeral WALLCLOCK=480
#   make bench-job MODE=ephemeral OPTLEVEL=1
# ---------------------------------------------------------------------------

MODE   ?= ephemeral
WAIT   ?= 0
# Default to --neuron unless explicitly disabled with NEURON=0.
NEURON ?= 1

_JOB_FLAGS  = $(if $(filter 1,$(NEURON)),--neuron)
_JOB_FLAGS += $(if $(filter 1,$(NO_FUSED)),--no-fused)
_JOB_FLAGS += $(foreach l,$(LATTICE),--lattice $(l))
_JOB_FLAGS += $(if $(BATCH),--batch-sizes $(BATCH))
_JOB_FLAGS += $(if $(OPTLEVEL),--optlevel $(OPTLEVEL))

.PHONY: bench-job
bench-job:  ## Trigger a bench job (results emailed) [MODE=persistent|ephemeral] [WAIT=1] [WALLCLOCK=min]
	chmod +x scripts/trigger_bench_job.sh
	bash scripts/trigger_bench_job.sh \
	    --mode $(MODE) \
	    $(if $(filter 1,$(WAIT)),--wait) \
	    $(if $(WALLCLOCK),--wallclock-minutes $(WALLCLOCK)) \
	    -- $(_JOB_FLAGS)

# ---------------------------------------------------------------------------
# Inspect partial results streamed by an in-flight bench-job to S3.
#   make bench-runs                       # list runs in the bucket
#   make bench-tail                       # average results across ALL runs
#   make bench-tail RUN=<run_id>          # per-lattice results as a table
#   make bench-tail RUN=<run_id> RAW=1    # raw JSONL (one line per entry)
#   make bench-tail RUN=<run_id> LOG=1    # tail the running bench.log
# ---------------------------------------------------------------------------
_BENCH_BUCKET = $$(tofu -chdir=$(INFRA_DIR) output -raw bench_s3_bucket)
_BENCH_REGION = $$(tofu -chdir=$(INFRA_DIR) output -raw aws_region)

# Classify a run from its S3 artefacts (and a single EC2 cross-check):
#   DONE OK / DONE FAILED / DONE ?  -- final bench.log present
#   RUNNING                          -- partial/* present, instance alive,
#                                       last partial < STALE_MIN ago
#   STALE                            -- partial/* present, last update >
#                                       STALE_MIN ago, instance still alive
#   ABANDONED                        -- partial/* present but the launching
#                                       EC2 instance is gone (terminated or
#                                       never existed in this account)
#   UNKNOWN                          -- prefix exists with no usable artefacts
# Tunable: STALE_MIN env var (default 30 minutes).
define _BENCH_RUNS_PY
import datetime as dt
import json, os, subprocess, sys, collections

REGION = os.environ["_BENCH_REGION"]
BUCKET = os.environ["_BENCH_BUCKET"]
STALE_MIN = int(os.environ.get("STALE_MIN", "30"))

# ANSI colors; disabled when stdout is not a TTY or NO_COLOR is set.
USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else s

def _colorize(status):
    if status.startswith("DONE OK"):
        return _c("1;32", status)         # bold green
    if status.startswith("DONE FAILED"):
        return _c("1;31", status)         # bold red
    if status == "RUNNING":
        return _c("1;33", status)         # bold yellow
    if status == "STALE":
        return _c("1;35", status)         # bold magenta
    if status == "ABANDONED":
        return _c("1;38;5;93", status)    # bold purple (256-color)
    if status == "UNKNOWN" or status.endswith("?"):
        return _c("1;90", status)         # bold gray
    return status

# stdin: JSON array of {Key, LastModified} from list-objects-v2
try:
    objects = json.loads(sys.stdin.read() or "[]") or []
except json.JSONDecodeError:
    objects = []

def _parse_iso(s):
    # AWS returns e.g. '2026-05-06T15:42:53+00:00'
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None

runs = collections.defaultdict(lambda: {
    "final": False, "partial": False, "last_partial": None,
})
for obj in objects:
    k = obj.get("Key", "")
    parts = k.split("/", 2)
    if len(parts) < 3 or parts[0] != "runs":
        continue
    run_id, tail = parts[1], parts[2]
    info = runs[run_id]
    if tail == "bench.log":
        info["final"] = True
    elif tail.startswith("partial/") or tail.startswith("bench.log.partial"):
        info["partial"] = True
        ts = _parse_iso(obj.get("LastModified", ""))
        if ts and (info["last_partial"] is None or ts > info["last_partial"]):
            info["last_partial"] = ts

def _final_status(run_id):
    # One small GET per completed run; the status line is in the meta header.
    try:
        out = subprocess.check_output(
            ["aws", "s3", "cp", "--region", REGION, "--quiet",
             f"s3://{BUCKET}/runs/{run_id}/bench.log", "-"],
            stderr=subprocess.DEVNULL,
        ).decode(errors="replace")
    except subprocess.CalledProcessError:
        return "?"
    for line in out.splitlines():
        if line.startswith("status"):
            _, _, val = line.partition(":")
            val = val.strip()
            if val.startswith("OK"):
                return "OK"
            if val.startswith("FAILED"):
                return val.upper()
            return val.upper() or "?"
    return "?"

# Cross-check EC2 for the instance ids embedded in still-active run ids.
# Run id format: <TIMESTAMP>-<INSTANCE_ID>, e.g. 20260506T154253Z-i-0ae62da2330b0aab1
def _instance_id(run_id):
    rest = run_id.split("-", 1)[1] if "-" in run_id else ""
    return rest if rest.startswith("i-") else None

needs_ec2 = {
    _instance_id(r) for r, info in runs.items()
    if not info["final"] and _instance_id(r)
}
needs_ec2.discard(None)

alive = set()
if needs_ec2:
    try:
        out = subprocess.check_output(
            ["aws", "ec2", "describe-instances", "--region", REGION,
             "--instance-ids", *sorted(needs_ec2),
             "--query", "Reservations[].Instances[].[InstanceId,State.Name]",
             "--output", "json"],
            stderr=subprocess.DEVNULL,
        )
        for iid, state in json.loads(out or "[]"):
            # 'terminated' / 'shutting-down' instances still appear here for
            # ~1h after teardown; treat them as not-alive.
            if state in ("pending", "running", "stopping", "stopped"):
                alive.add(iid)
    except subprocess.CalledProcessError:
        # If any id has been purged from EC2 entirely, describe-instances
        # fails the whole call.  Fall back to per-id checks.
        for iid in sorted(needs_ec2):
            try:
                out = subprocess.check_output(
                    ["aws", "ec2", "describe-instances", "--region", REGION,
                     "--instance-ids", iid,
                     "--query", "Reservations[].Instances[].State.Name",
                     "--output", "text"],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
                if out in ("pending", "running", "stopping", "stopped"):
                    alive.add(iid)
            except subprocess.CalledProcessError:
                pass

now = dt.datetime.now(dt.timezone.utc)
stale_delta = dt.timedelta(minutes=STALE_MIN)

rows = []
for run_id in sorted(runs):
    info = runs[run_id]
    if info["final"]:
        status = "DONE " + _final_status(run_id)
    elif info["partial"]:
        iid = _instance_id(run_id)
        if iid and iid not in alive:
            status = "ABANDONED"
        elif info["last_partial"] and (now - info["last_partial"]) > stale_delta:
            status = "STALE"
        else:
            status = "RUNNING"
    else:
        status = "UNKNOWN"
    ts = run_id.split("-", 1)[0]
    started = (
        f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
        if len(ts) >= 15 else ts
    )
    rows.append((started, run_id, status))

w0 = max([len("started (UTC)")] + [len(r[0]) for r in rows])
w1 = max([len("run id")]        + [len(r[1]) for r in rows])
w2 = max([len("status")]        + [len(r[2]) for r in rows])
print(f"{'started (UTC)'.ljust(w0)}  {'run id'.ljust(w1)}  {'status'.ljust(w2)}")
print("-" * w0 + "  " + "-" * w1 + "  " + "-" * w2)
for r in rows:
    # Pad first (raw width), then colorize -- ANSI codes have zero printed width.
    status_cell = r[2].ljust(w2)
    print(f"{r[0].ljust(w0)}  {r[1].ljust(w1)}  {_colorize(status_cell)}")
endef
export _BENCH_RUNS_PY

.PHONY: bench-runs
bench-runs:  ## List bench runs in S3 [STALE_MIN=30]
	@_BENCH_REGION=$(_BENCH_REGION) _BENCH_BUCKET=$(_BENCH_BUCKET) \
	 STALE_MIN=$(or $(STALE_MIN),30) \
	 aws s3api list-objects-v2 --region $(_BENCH_REGION) \
	    --bucket $(_BENCH_BUCKET) --prefix runs/ \
	    --query 'Contents[].{Key:Key,LastModified:LastModified}' \
	    --output json 2>/dev/null \
	    | _BENCH_REGION=$(_BENCH_REGION) _BENCH_BUCKET=$(_BENCH_BUCKET) \
	      STALE_MIN=$(or $(STALE_MIN),30) \
	      python3 -c "$$_BENCH_RUNS_PY"

define _BENCH_TAIL_PY
import json, os, sys

rows = [json.loads(l) for l in sys.stdin if l.strip()]
if not rows:
    print("(no partial results yet)"); sys.exit(0)

# ANSI bold green for the winning throughput column; disabled when not a
# TTY or when NO_COLOR is set (https://no-color.org/).
USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
BOLD = "\033[1;32m" if USE_COLOR else ""
RESET = "\033[0m"  if USE_COLOR else ""

THROUGHPUT_KEYS = ("cpu", "neuron", "batched", "multicore")

def _num(r, k):
    v = r.get(k)
    return v if isinstance(v, (int, float)) else None

def _fmt(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else ""

def _speedup(r):
    cpu = _num(r, "cpu")
    if not cpu:
        return ""
    best = max((v for v in (_num(r, k) for k in THROUGHPUT_KEYS[1:]) if v is not None), default=None)
    if best is None:
        return ""
    return f"{best / cpu:.1f}x"

cols = [
    ("t_utc",     "time (UTC)", lambda r: r.get("t_utc", "")),
    ("label",     "Lattice",    lambda r: r.get("label", "")),
    ("B",         "B",          lambda r: str(r.get("B", ""))),
    ("cpu",       "CPU",        lambda r: _fmt(_num(r, "cpu"))),
    ("neuron",    "Neuron",     lambda r: _fmt(_num(r, "neuron"))),
    ("batched",   "Batched",    lambda r: _fmt(_num(r, "batched"))),
    ("multicore", "Multicore",  lambda r: _fmt(_num(r, "multicore"))),
    ("speedup",   "Speedup",    _speedup),
    ("note",      "note",       lambda r: r.get("skipped") or r.get("neuron_error") or r.get("batched_error") or ""),
]
data = [[fn(r) for _, _, fn in cols] for r in rows]
# Drop columns that are empty for every row (e.g. t_utc in averaged mode).
keep = [i for i in range(len(cols)) if any(row[i] for row in data)]
cols = [cols[i] for i in keep]
data = [[row[i] for i in keep] for row in data]
widths = [max(len(h), *(len(row[i]) for row in data)) for i, (_, h, _) in enumerate(cols)]
key_to_idx = {k: i for i, (k, _, _) in enumerate(cols)}

# Pad first (so column widths line up), then wrap the winning cell in
# ANSI codes -- doing it this order keeps alignment correct since the
# escape sequences have zero printed width.
header = "  ".join(h.ljust(w) for (_, h, _), w in zip(cols, widths))
print(header)
print("  ".join("-" * w for w in widths))
for row, raw in zip(data, rows):
    cells = [val.ljust(w) for val, w in zip(row, widths)]
    nums = [(k, _num(raw, k)) for k in THROUGHPUT_KEYS]
    nums = [(k, v) for k, v in nums if v is not None]
    if nums:
        winner = max(nums, key=lambda kv: kv[1])[0]
        if winner in key_to_idx:
            i = key_to_idx[winner]
            cells[i] = f"{BOLD}{cells[i]}{RESET}"
    print("  ".join(cells))
endef
export _BENCH_TAIL_PY

# Aggregator: group records by (label, B) and average the throughput
# columns across runs.  Used when `make bench-tail` is invoked without RUN.
define _BENCH_AVG_PY
import collections, json, sys

THROUGHPUT_KEYS = ("cpu", "neuron", "batched", "multicore")
groups = collections.OrderedDict()
runs = set()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    run_id = r.get("_run") or r.get("run_id") or ""
    if run_id:
        runs.add(run_id)
    key = (r.get("label", ""), r.get("B", ""))
    g = groups.setdefault(key, {
        "label": key[0], "B": key[1],
        "_sum": collections.defaultdict(float),
        "_n":   collections.defaultdict(int),
        "_runs": set(),
        "_count": 0,
    })
    g["_count"] += 1
    if run_id:
        g["_runs"].add(run_id)
    for k in THROUGHPUT_KEYS:
        v = r.get(k)
        if isinstance(v, (int, float)):
            g["_sum"][k] += v
            g["_n"][k]   += 1

def _vol(label):
    try:
        return [int(x) for x in str(label).split("x")]
    except (TypeError, ValueError):
        return [0]

print(f"# averaged across {len(runs) or '?'} run(s), {sum(g['_count'] for g in groups.values())} record(s)",
      file=sys.stderr)

for key in sorted(groups, key=lambda k: (_vol(k[0]), k[1] if isinstance(k[1], int) else 0)):
    g = groups[key]
    out = {"label": g["label"], "B": g["B"]}
    for k in THROUGHPUT_KEYS:
        if g["_n"][k]:
            out[k] = g["_sum"][k] / g["_n"][k]
    out["note"] = ", ".join(sorted(g["_runs"])) if g["_runs"] else f"avg(n={g['_count']})"
    print(json.dumps(out))
endef
export _BENCH_AVG_PY

.PHONY: bench-tail
bench-tail:  ## Tail partial results [no RUN: avg across all runs] [RUN=<id>] [RAW=1] [LOG=1]
	@if [ "$(LOG)" = "1" ] && [ -z "$(RUN)" ]; then \
	    echo "LOG=1 requires RUN=<run_id>"; exit 2; \
	fi
	@if [ -n "$(RUN)" ]; then \
	    if [ "$(LOG)" = "1" ]; then \
	        aws s3 cp --region $(_BENCH_REGION) \
	            "s3://$(_BENCH_BUCKET)/runs/$(RUN)/bench.log.partial" -; \
	    elif [ "$(RAW)" = "1" ]; then \
	        aws s3 cp --region $(_BENCH_REGION) \
	            "s3://$(_BENCH_BUCKET)/runs/$(RUN)/partial/results.jsonl" -; \
	    else \
	        aws s3 cp --region $(_BENCH_REGION) \
	            "s3://$(_BENCH_BUCKET)/runs/$(RUN)/partial/results.jsonl" - \
	        | python3 -c "$$_BENCH_TAIL_PY"; \
	    fi; \
	else \
	    runs=$$(aws s3 ls --region $(_BENCH_REGION) "s3://$(_BENCH_BUCKET)/runs/" \
	        | awk '/PRE / {gsub("/","",$$2); print $$2}'); \
	    if [ -z "$$runs" ]; then echo "(no runs in bucket)"; exit 0; fi; \
	    tmp=$$(mktemp); \
	    trap 'rm -f $$tmp' EXIT; \
	    for r in $$runs; do \
	        aws s3 cp --region $(_BENCH_REGION) \
	            "s3://$(_BENCH_BUCKET)/runs/$$r/partial/results.jsonl" - \
	            2>/dev/null \
	            | awk -v r="$$r" 'NF { sub(/^{/, "{\"_run\":\"" r "\","); print }' \
	            >> $$tmp || true; \
	    done; \
	    if [ ! -s $$tmp ]; then \
	        echo "(no partial results found across $$(echo $$runs | wc -w) run(s))"; \
	        exit 0; \
	    fi; \
	    if [ "$(RAW)" = "1" ]; then \
	        python3 -c "$$_BENCH_AVG_PY" < $$tmp; \
	    else \
	        python3 -c "$$_BENCH_AVG_PY" < $$tmp | python3 -c "$$_BENCH_TAIL_PY"; \
	    fi; \
	fi

.PHONY: bench-rm
bench-rm:  ## Delete a bench run from S3 (RUN=<id> [YES=1] to skip prompt)
	@if [ -z "$(RUN)" ]; then \
	    echo "Usage: make bench-rm RUN=<run_id> [YES=1]"; exit 2; \
	fi
	@prefix="s3://$(_BENCH_BUCKET)/runs/$(RUN)/"; \
	count=$$(aws s3 ls --region $(_BENCH_REGION) --recursive "$$prefix" \
	    2>/dev/null | wc -l); \
	if [ "$$count" -eq 0 ]; then \
	    echo "(no objects under $$prefix)"; exit 0; \
	fi; \
	echo "About to delete $$count object(s) under $$prefix"; \
	if [ "$(YES)" != "1" ]; then \
	    printf "Continue? [y/N] "; read ans; \
	    case "$$ans" in y|Y|yes|YES) ;; *) echo "aborted."; exit 1 ;; esac; \
	fi; \
	aws s3 rm --region $(_BENCH_REGION) --recursive "$$prefix"

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
