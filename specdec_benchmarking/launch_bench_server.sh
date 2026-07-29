#!/usr/bin/env bash
# Launch vLLM servers for PD / PD+SD / P/D-disagg bench topologies.
#
# Examples:
#   ./launch_bench_server.sh --list
#   ./launch_bench_server.sh PD1_b8192 --devices 0 --port 8000
#   ./launch_bench_server.sh PD2SD1 --devices 0,1 --port 8000 --batched-tokens 8192
#   ./launch_bench_server.sh PD4SD1_b8192 --devices 0,1,2,3 --port 8000 \
#       --no-async-scheduling --enable-logging-iteration-details --enable-disagg-profile
#   ./launch_bench_server.sh V1SD1_b8192 --devices 0 --draft-devices 1 --port 8000
#   ./launch_bench_server.sh V4SD1_b8192 --devices 0,1,2,3 --draft-devices 4 --port 8000 \
#       --no-wave-schedule --no-disagg-async --no-async-scheduling \
#       --enable-logging-iteration-details --nixl-log-every 0 --enable-disagg-profile
#   ./launch_bench_server.sh P2_D2SD1 \
#       --prefill-devices 0,1 --decode-devices 2,3 \
#       --prefill-port 8100 --decode-port 8200 --batched-tokens 16384
#   ./launch_bench_server.sh PD4SD4_b4096 --print-only
#   ./launch_bench_server.sh PD1 --devices 0 --port 8000 -- --max-model-len 8192
#
# Case grammar:
#   PD{tp}[SD{draft_tp}][_b{batched}]     colocated prefill+decode
#   P{ptp}_D{dtp}[SD{draft_tp}][_b{batched}]   PD disagg (SD on decode)
#   V{vtp}SD{draft_gpus}[_b{batched}]          Disagg-DFlash verify + remote draft
#
# Defaults match recent gpt-oss-20b / DFlash / Nixl runs on this machine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-openai/gpt-oss-20b}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/gpt-oss-20b-DFlash}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-600}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NIXL_PREFILL_PORT="${NIXL_PREFILL_PORT:-5600}"
NIXL_DECODE_PORT="${NIXL_DECODE_PORT:-5601}"
PROXY_SCRIPT="${PROXY_SCRIPT:-}"
PROXY_PORT="${PROXY_PORT:-8000}"
# Disagg-DFlash (V*SD*): ZMQ control-plane endpoint (speculate RPC / draft tokens).
# --draft-comm ipc|tcp selects the default bind/connect URLs (overridable via
# --draft-bind / --draft-addr). tcp://127.0.0.1 may use OS/ZMQ loopback shm.
DRAFT_COMM="${DRAFT_COMM:-ipc}"
DRAFT_ZMQ_PORT="${DRAFT_ZMQ_PORT:-50051}"
DRAFT_IPC_PATH="${DRAFT_IPC_PATH:-/tmp/vllm_dflash_draft.ipc}"
DRAFT_BIND="${DRAFT_BIND:-}"
DRAFT_ADDR="${DRAFT_ADDR:-}"
DISAGG_DFLASH_TRANSPORT="${DISAGG_DFLASH_TRANSPORT:-nixl}"
WAVE_SIZE="${WAVE_SIZE:-}"
WAVE_SCHEDULE=1
DISAGG_ASYNC=1
# vLLM async scheduling (batch queue). On by default (omit flag → vLLM default).
ASYNC_SCHEDULING=1
# Off by default: SD timing / Disagg profile use CUDA synchronize and skew TPOT.
ENABLE_DISAGG_PROFILE=0
# NIXL/ZMQ transfer INFO logs: -1 off (default), 0 every xfer, N every Nth.
# Wall-clock + NIXL telemetry only — no extra CUDA sync for logging.
NIXL_LOG_EVERY="${NIXL_LOG_EVERY:--1}"
# Live terminal + file under startup_logs/<tag>_<role>_<timestamp>.log
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/startup_logs}"

# Synthetic acceptance rates used in prior PD*SD* sweeps.
SYNTH_RATES='[0.9184485330681254,0.8179331856606844,0.7422584874101532,0.6783825324352425,0.6191175805795398,0.5591293341169025,0.49742326296279554]'

CASE=""
BATCHED=""
DEVICES=""
PORT=""
PREFILL_DEVICES=""
DECODE_DEVICES=""
DRAFT_DEVICES=""
PREFILL_PORT="8100"
DECODE_PORT="8200"
PRINT_ONLY=0
DO_PROXY=0
LIST_ONLY=0
ENABLE_LOGGING_ITERATION_DETAILS=0
EXTRA_ARGS=()

CASES=(
  PD1 PD2 PD4
  PD1SD1 PD2SD2 PD4SD4
  PD2SD1 PD4SD1
  P1_D1 P2_D2
  P1_D1SD1 P2_D2SD2 P2_D2SD1
  V1SD1 V2SD1 V4SD1
)
BATCHED_SWEEP=(4096 8192 16384)

usage() {
  cat <<'EOF'
Usage:
  launch_bench_server.sh <CASE[_bBATCHED]> [options] [-- extra vllm args...]
  launch_bench_server.sh --list

Options:
  --batched-tokens N       max-num-batched-tokens (or use _bN in CASE)
  --max-num-seqs N         max-num-seqs (default: 600, or MAX_NUM_SEQS env)
  --devices IDS            CUDA_VISIBLE_DEVICES for colocated / verify (e.g. 0,1)
  --port PORT              HTTP port for colocated / verify server (default 8000)
  --prefill-devices IDS    GPUs for prefill (P/D disagg)
  --decode-devices IDS     GPUs for decode (P/D disagg)
  --draft-devices IDS      GPUs for Disagg-DFlash draft server (V*SD*)
  --prefill-port PORT      Prefill HTTP port (default 8100)
  --decode-port PORT       Decode HTTP port (default 8200)
  --draft-bind ADDR        Draft server bind (default from --draft-comm)
  --draft-addr ADDR        Verify→draft connect addr (default from --draft-comm)
  --draft-comm MODE        ZMQ endpoint mode: ipc (default) or tcp.
                           ipc → ipc:///tmp/vllm_dflash_draft.ipc
                           tcp → tcp://127.0.0.1:50051 (loopback; may use shm)
  --wave-size N            disagg_dflash_wave_size (omit = ceil(ready/2))
  --no-wave-schedule       disagg_dflash_wave_schedule=false
  --no-disagg-async        disagg_dflash_async_complete=false (sync propose)
  --no-async-scheduling    Pass --no-async-scheduling to vllm serve
                           (disables EngineCore batch-queue async scheduling)
  --enable-disagg-profile  Opt-in SD timing logs (same [DisaggDFlash][timing]
                           format for colocated + disagg) + Disagg/DFlash profile
                           (CUDA sync — skews latency; off by default for benches)
  --nixl-log-every N       NIXL/ZMQ transfer INFO logs on verify + draft:
                           -1 off (default), 0 every transfer, N every Nth.
                           Wall-clock + NIXL telemetry (no extra CUDA sync).
  --proxy                  Also launch toy_proxy_server.py on --proxy-port
  --proxy-port PORT        Client-facing proxy port (default 8000)
  --proxy-script PATH      Override path to toy_proxy_server.py
  --log-dir DIR            Where to write startup logs (default: ./startup_logs)
  --enable-logging-iteration-details
                           Opt-in: pass to vllm serve and (for V*SD*) draft
                           server. Off by default.
  --print-only             Print commands, do not exec
  --list                   List supported cases / full sweep matrix
  -h, --help               Show this help

Cases:
  PD{tp}[SD{draft_tp}][_bN]           colocated prefill+decode (+ optional SD)
  P{ptp}_D{dtp}[SD{draft_tp}][_bN]    P/D KV disagg (SD on decode)
  V{vtp}SD{draft_gpus}[_bN]           Disagg-DFlash: verify + remote draft server
                                      e.g. V1SD1_b8192  (verify GPU0, draft GPU1)

Logs:
  stdout+stderr are teed live to the terminal and to
  <log-dir>/<tag>_<role>_YYYYMMDD_HHMMSS.log

Env overrides:
  MODEL DRAFT_MODEL NUM_SPEC_TOKENS GPU_MEM_UTIL MAX_NUM_SEQS MAX_MODEL_LEN
  BLOCK_SIZE NIXL_PREFILL_PORT NIXL_DECODE_PORT LOG_DIR PROXY_WAIT_TIMEOUT
  DRAFT_BIND DRAFT_ADDR DRAFT_COMM DRAFT_ZMQ_PORT DRAFT_IPC_PATH
  DISAGG_DFLASH_TRANSPORT WAVE_SIZE
EOF
}

list_matrix() {
  echo "Supported base cases:"
  printf '  %s\n' "${CASES[@]}"
  echo
  echo "Batched-token sweep values: ${BATCHED_SWEEP[*]}"
  echo "Full tag form: <CASE>_b<BATCHED>  e.g. PD2SD1_b8192  or  V1SD1_b8192"
  echo
  echo "Colocated SD examples:"
  echo "  ./launch_bench_server.sh PD1SD1_b8192 --devices 0 --port 8000 \\"
  echo "      --no-async-scheduling --enable-logging-iteration-details --enable-disagg-profile"
  echo "  ./launch_bench_server.sh PD2SD1_b8192 --devices 0,1 --port 8000 \\"
  echo "      --no-async-scheduling --enable-logging-iteration-details --enable-disagg-profile"
  echo "  ./launch_bench_server.sh PD4SD1_b8192 --devices 0,1,2,3 --port 8000 \\"
  echo "      --no-async-scheduling --enable-logging-iteration-details --enable-disagg-profile"
  echo
  echo "Disagg-DFlash: V{verify_tp}SD{draft_gpus}_bN  (remote draft via NIXL)"
  echo "  ./launch_bench_server.sh V1SD1_b8192 --devices 0 --draft-devices 1 --port 8000 \\"
  echo "      --no-wave-schedule --no-disagg-async --no-async-scheduling \\"
  echo "      --enable-logging-iteration-details --nixl-log-every 0 --enable-disagg-profile"
  echo "  ./launch_bench_server.sh V2SD1_b8192 --devices 0,1 --draft-devices 2 --port 8000 \\"
  echo "      --no-wave-schedule --no-disagg-async --no-async-scheduling \\"
  echo "      --enable-logging-iteration-details --nixl-log-every 0 --enable-disagg-profile"
  echo "  ./launch_bench_server.sh V4SD1_b8192 --devices 0,1,2,3 --draft-devices 4 --port 8000 \\"
  echo "      --no-wave-schedule --no-disagg-async --no-async-scheduling \\"
  echo "      --enable-logging-iteration-details --nixl-log-every 0 --enable-disagg-profile"
  echo
  echo "=== Full sweep (case × batched) ==="
  for c in "${CASES[@]}"; do
    for b in "${BATCHED_SWEEP[@]}"; do
      echo "  ${c}_b${b}"
    done
  done
}

die() { echo "error: $*" >&2; exit 1; }

# Parse CASE -> MODE / TP / DRAFT_TP / optional embedded batched
parse_case() {
  local raw="$1"
  local base="$raw"
  BATCHED_FROM_CASE=""

  if [[ "$raw" =~ ^(.*)_b([0-9]+)$ ]]; then
    base="${BASH_REMATCH[1]}"
    BATCHED_FROM_CASE="${BASH_REMATCH[2]}"
  fi

  MODE=""
  TP=""
  PREFILL_TP=""
  DECODE_TP=""
  DRAFT_TP=""
  VERIFY_TP=""
  DRAFT_GPUS=""

  if [[ "$base" =~ ^V([0-9]+)SD([0-9]+)$ ]]; then
    MODE=sd_disagg
    VERIFY_TP="${BASH_REMATCH[1]}"
    DRAFT_GPUS="${BASH_REMATCH[2]}"
  elif [[ "$base" =~ ^PD([0-9]+)SD([0-9]+)$ ]]; then
    MODE=colocated_sd
    TP="${BASH_REMATCH[1]}"
    DRAFT_TP="${BASH_REMATCH[2]}"
  elif [[ "$base" =~ ^PD([0-9]+)$ ]]; then
    MODE=colocated
    TP="${BASH_REMATCH[1]}"
  elif [[ "$base" =~ ^P([0-9]+)_D([0-9]+)SD([0-9]+)$ ]]; then
    MODE=disagg_sd
    PREFILL_TP="${BASH_REMATCH[1]}"
    DECODE_TP="${BASH_REMATCH[2]}"
    DRAFT_TP="${BASH_REMATCH[3]}"
  elif [[ "$base" =~ ^P([0-9]+)_D([0-9]+)$ ]]; then
    MODE=disagg
    PREFILL_TP="${BASH_REMATCH[1]}"
    DECODE_TP="${BASH_REMATCH[2]}"
  else
    die "unrecognized case '$raw' (try --list)"
  fi

  CASE_BASE="$base"
}

default_devices_for_tp() {
  local n="$1"
  case "$n" in
    1) echo "0" ;;
    2) echo "0,1" ;;
    4) echo "0,1,2,3" ;;
    *)
      local ids=()
      local i
      for ((i = 0; i < n; i++)); do ids+=("$i"); done
      (IFS=,; echo "${ids[*]}")
      ;;
  esac
}

default_disagg_devices() {
  # Prefer packing P then D on consecutive GPUs.
  local ptp="$1" dtp="$2"
  local p_ids=() d_ids=()
  local i
  for ((i = 0; i < ptp; i++)); do p_ids+=("$i"); done
  for ((i = 0; i < dtp; i++)); do d_ids+=("$((ptp + i))"); done
  PREFILL_DEVICES_DEFAULT=$(IFS=,; echo "${p_ids[*]}")
  DECODE_DEVICES_DEFAULT=$(IFS=,; echo "${d_ids[*]}")
}

default_sd_disagg_devices() {
  # Verify on first VERIFY_TP GPUs; draft on the next DRAFT_GPUS GPUs.
  local vtp="$1" dgpus="$2"
  local v_ids=() d_ids=()
  local i
  for ((i = 0; i < vtp; i++)); do v_ids+=("$i"); done
  for ((i = 0; i < dgpus; i++)); do d_ids+=("$((vtp + i))"); done
  VERIFY_DEVICES_DEFAULT=$(IFS=,; echo "${v_ids[*]}")
  DRAFT_DEVICES_DEFAULT=$(IFS=,; echo "${d_ids[*]}")
}

resolve_draft_endpoints() {
  # Fill DRAFT_BIND / DRAFT_ADDR from --draft-comm unless explicitly set.
  local comm
  comm=$(echo "${DRAFT_COMM}" | tr '[:upper:]' '[:lower:]')
  case "$comm" in
    ipc)
      # ipc:///path — three slashes (empty host + absolute path).
      DRAFT_BIND="${DRAFT_BIND:-ipc://${DRAFT_IPC_PATH}}"
      DRAFT_ADDR="${DRAFT_ADDR:-ipc://${DRAFT_IPC_PATH}}"
      ;;
    tcp)
      # Bind+connect on 127.0.0.1 (not 0.0.0.0): ZMQ/docs note loopback
      # can use shared-memory transport and avoid the full TCP stack.
      DRAFT_BIND="${DRAFT_BIND:-tcp://127.0.0.1:${DRAFT_ZMQ_PORT}}"
      DRAFT_ADDR="${DRAFT_ADDR:-tcp://127.0.0.1:${DRAFT_ZMQ_PORT}}"
      ;;
    *)
      die "--draft-comm must be 'ipc' or 'tcp' (got '${DRAFT_COMM}')"
      ;;
  esac
  DRAFT_COMM="$comm"
}

count_csv() {
  local s="$1"
  if [[ -z "$s" ]]; then echo 0; return; fi
  awk -F, '{print NF}' <<<"$s"
}

spec_json() {
  local draft_tp="$1"
  # Colocated SD: set draft_tensor_parallel_size.
  printf '{"method":"dflash","model":"%s","num_speculative_tokens":%s,"draft_tensor_parallel_size":%s,"rejection_sample_method":"synthetic","synthetic_acceptance_rates":%s}' \
    "$DRAFT_MODEL" "$NUM_SPEC_TOKENS" "$draft_tp" "$SYNTH_RATES"
}

spec_json_sd_disagg() {
  # Remote DFlash: verify does not load draft weights; RPCs to draft server.
  local wave_sched="true"
  local async_c="true"
  local profile="false"
  local wave_size_json="null"
  [[ "$WAVE_SCHEDULE" -eq 1 ]] || wave_sched="false"
  [[ "$DISAGG_ASYNC" -eq 1 ]] || async_c="false"
  [[ "$ENABLE_DISAGG_PROFILE" -eq 1 ]] && profile="true"
  if [[ -n "$WAVE_SIZE" ]]; then
    wave_size_json="$WAVE_SIZE"
  fi
  # Same synthetic acceptance as colocated PD*SD* so paper A/B compares
  # latency/overlap, not draft quality.
  printf '{"method":"dflash","model":"%s","num_speculative_tokens":%s,"rejection_sample_method":"synthetic","synthetic_acceptance_rates":%s,"disagg_dflash_address":"%s","disagg_dflash_transport":"%s","disagg_dflash_cross_step":true,"disagg_dflash_async_complete":%s,"disagg_dflash_wave_schedule":%s,"disagg_dflash_wave_size":%s,"disagg_dflash_profile":%s,"attention_backend":"FLASHINFER"}' \
    "$DRAFT_MODEL" "$NUM_SPEC_TOKENS" "$SYNTH_RATES" \
    "$DRAFT_ADDR" "$DISAGG_DFLASH_TRANSPORT" \
    "$async_c" "$wave_sched" "$wave_size_json" "$profile"
}

kv_json() {
  local role="$1" # kv_producer | kv_consumer
  printf '{"kv_connector":"NixlConnector","kv_role":"%s","kv_buffer_device":"cuda"}' "$role"
}

common_flags() {
  # Shared flags for all roles.
  cat <<EOF
--served-model-name ${MODEL}
--dtype auto
--gpu-memory-utilization ${GPU_MEM_UTIL}
--max-num-seqs ${MAX_NUM_SEQS}
--max-model-len ${MAX_MODEL_LEN}
--max-num-batched-tokens ${BATCHED}
--block-size ${BLOCK_SIZE}
--no-enable-prefix-caching
--enable-mfu-metrics
--enable-auto-tool-choice
--tool-call-parser openai
--reasoning-parser openai_gptoss
--enable-prompt-tokens-details
EOF
}

find_proxy_script() {
  if [[ -n "$PROXY_SCRIPT" && -f "$PROXY_SCRIPT" ]]; then
    echo "$PROXY_SCRIPT"
    return
  fi
  # Nixl PD toy proxy (preferred) — correctly threads kv_transfer_params P→D.
  local candidates=(
    "${SCRIPT_DIR}/../tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
    "${HOME}/vllm_2307/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
    "${HOME}/vllm-sd-disagg/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -f "$c" ]]; then
      # Resolve to absolute path for logs / tips.
      (cd "$(dirname "$c")" && echo "$(pwd)/$(basename "$c")")
      return
    fi
  done
  return 1
}

# Wait until HTTP GET url returns 200 (engines ready before proxy).
wait_http_ready() {
  local url="$1"
  local name="$2"
  local timeout_s="${3:-${PROXY_WAIT_TIMEOUT:-600}}"
  local start now code
  start=$(date +%s)
  echo "# waiting for ${name} at ${url} (timeout ${timeout_s}s)..."
  while true; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 --max-time 5 "$url" || true)
    if [[ "$code" == "200" ]]; then
      echo "# ${name} ready (HTTP 200)"
      return 0
    fi
    now=$(date +%s)
    if [[ $((now - start)) -ge "$timeout_s" ]]; then
      die "${name} not ready after ${timeout_s}s (last HTTP ${code:-000}) -- check its log"
    fi
    sleep 2
  done
}

shell_join() {
  # Human-readable join; quote only tokens that need it.
  local out="" t
  for t in "$@"; do
    if [[ "$t" =~ [[:space:]|{}\"\'\\] ]]; then
      out+=" $(printf '%q' "$t")"
    else
      out+=" $t"
    fi
  done
  echo "${out# }"
}

run_cmd() {
  # $1=human description  $2=role slug for logfile  rest=command
  local desc="$1"
  local role="$2"
  shift 2
  local ts logfile
  ts=$(date +%Y%m%d_%H%M%S)
  logfile="${LOG_DIR}/${TAG}_${role}_${ts}.log"

  echo
  echo "# ---- ${desc} ----"
  echo "# log -> ${logfile}"
  shell_join "$@"
  if [[ "$PRINT_ONLY" -eq 1 ]]; then
    return 0
  fi
  mkdir -p "$LOG_DIR"
  # Header in the log file for later attribution.
  {
    echo "# launched: $(date -Is)"
    echo "# tag=${TAG} role=${role}"
    echo "# cmd: $(shell_join "$@")"
    echo "# ----"
  } >"$logfile"
  # Live on terminal + append to file (stdout and stderr).
  if command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL -eL "$@" 2>&1 | tee -a "$logfile" &
  else
    "$@" 2>&1 | tee -a "$logfile" &
  fi
  echo "# pid=$!"
}

# ---------- arg parse ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --list) LIST_ONLY=1; shift ;;
    --print-only) PRINT_ONLY=1; shift ;;
    --proxy) DO_PROXY=1; shift ;;
    --enable-logging-iteration-details) ENABLE_LOGGING_ITERATION_DETAILS=1; shift ;;
    --batched-tokens) BATCHED="${2:?}"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="${2:?}"; shift 2 ;;
    --devices) DEVICES="${2:?}"; shift 2 ;;
    --port) PORT="${2:?}"; shift 2 ;;
    --prefill-devices) PREFILL_DEVICES="${2:?}"; shift 2 ;;
    --decode-devices) DECODE_DEVICES="${2:?}"; shift 2 ;;
    --draft-devices) DRAFT_DEVICES="${2:?}"; shift 2 ;;
    --prefill-port) PREFILL_PORT="${2:?}"; shift 2 ;;
    --decode-port) DECODE_PORT="${2:?}"; shift 2 ;;
    --draft-bind) DRAFT_BIND="${2:?}"; shift 2 ;;
    --draft-addr) DRAFT_ADDR="${2:?}"; shift 2 ;;
    --draft-comm) DRAFT_COMM="${2:?}"; shift 2 ;;
    --wave-size) WAVE_SIZE="${2:?}"; shift 2 ;;
    --no-wave-schedule) WAVE_SCHEDULE=0; shift ;;
    --no-disagg-async) DISAGG_ASYNC=0; shift ;;
    --no-async-scheduling) ASYNC_SCHEDULING=0; shift ;;
    --enable-disagg-profile) ENABLE_DISAGG_PROFILE=1; shift ;;
    --nixl-log-every) NIXL_LOG_EVERY="${2:?}"; shift 2 ;;
    --proxy-port) PROXY_PORT="${2:?}"; shift 2 ;;
    --proxy-script) PROXY_SCRIPT="${2:?}"; shift 2 ;;
    --log-dir) LOG_DIR="${2:?}"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -*)
      die "unknown option: $1 (use -- for extra vllm args)"
      ;;
    *)
      if [[ -n "$CASE" ]]; then
        die "unexpected positional: $1 (case already set to $CASE)"
      fi
      CASE="$1"
      shift
      ;;
  esac
done

if [[ "$LIST_ONLY" -eq 1 ]]; then
  list_matrix
  exit 0
fi

[[ -n "$CASE" ]] || { usage; die "CASE is required"; }

# Resolve batched tokens: flag wins, else _bN, else error.
parse_case "$CASE"
if [[ -z "$BATCHED" ]]; then
  if [[ -n "$BATCHED_FROM_CASE" ]]; then
    BATCHED="$BATCHED_FROM_CASE"
  else
    die "set --batched-tokens N or use ${CASE_BASE}_bN"
  fi
fi
# Recompute tag with resolved batched
TAG="${CASE_BASE}_b${BATCHED}"

if [[ "$MODE" == "sd_disagg" ]]; then
  resolve_draft_endpoints
fi

echo "# case=${CASE_BASE}  tag=${TAG}  mode=${MODE}  batched=${BATCHED}"
if [[ "$MODE" == "sd_disagg" ]]; then
  echo "# draft-comm=${DRAFT_COMM}  bind=${DRAFT_BIND}  addr=${DRAFT_ADDR}"
fi

# ---------- build + launch ----------
COMMON=()
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  # shellcheck disable=SC2206
  COMMON+=($line)
done < <(common_flags)
COMMON+=("${EXTRA_ARGS[@]}")
if [[ "$ENABLE_LOGGING_ITERATION_DETAILS" -eq 1 ]]; then
  COMMON+=(--enable-logging-iteration-details)
fi
if [[ "$ASYNC_SCHEDULING" -eq 0 ]]; then
  COMMON+=(--no-async-scheduling)
fi

case "$MODE" in
  colocated|colocated_sd)
    PORT="${PORT:-8000}"
    DEVICES="${DEVICES:-$(default_devices_for_tp "$TP")}"
    n_dev=$(count_csv "$DEVICES")
    [[ "$n_dev" -eq "$TP" ]] || die "PD${TP} needs ${TP} devices, got '${DEVICES}' (${n_dev})"

    cmd=(
      env
      VLLM_USE_V2_MODEL_RUNNER=1
      HF_HUB_OFFLINE=1
      "CUDA_VISIBLE_DEVICES=${DEVICES}"
      vllm serve "$MODEL"
      --port "$PORT"
      --tensor-parallel-size "$TP"
      "${COMMON[@]}"
    )
    if [[ "$MODE" == colocated_sd ]]; then
      cmd+=(--speculative-config "$(spec_json "$DRAFT_TP")")
    fi
    if [[ "$ENABLE_DISAGG_PROFILE" -eq 1 ]]; then
      # Same --enable-sd-timing-model log line as Disagg (mode=colocated).
      cmd+=(--enable-sd-timing-model)
    fi
    run_cmd "${TAG} colocated tp=${TP} devices=${DEVICES} port=${PORT}" \
      "colocated" "${cmd[@]}"
    if [[ "$PRINT_ONLY" -eq 0 ]]; then
      wait
    fi
    ;;

  disagg|disagg_sd)
    default_disagg_devices "$PREFILL_TP" "$DECODE_TP"
    PREFILL_DEVICES="${PREFILL_DEVICES:-$PREFILL_DEVICES_DEFAULT}"
    DECODE_DEVICES="${DECODE_DEVICES:-$DECODE_DEVICES_DEFAULT}"
    np=$(count_csv "$PREFILL_DEVICES")
    nd=$(count_csv "$DECODE_DEVICES")
    [[ "$np" -eq "$PREFILL_TP" ]] || die "prefill TP=${PREFILL_TP} but --prefill-devices='${PREFILL_DEVICES}'"
    [[ "$nd" -eq "$DECODE_TP" ]] || die "decode TP=${DECODE_TP} but --decode-devices='${DECODE_DEVICES}'"

    p_cmd=(
      env
      VLLM_USE_V2_MODEL_RUNNER=1
      HF_HUB_OFFLINE=1
      UCX_NET_DEVICES=all
      "VLLM_NIXL_SIDE_CHANNEL_PORT=${NIXL_PREFILL_PORT}"
      "CUDA_VISIBLE_DEVICES=${PREFILL_DEVICES}"
      vllm serve "$MODEL"
      --port "$PREFILL_PORT"
      --tensor-parallel-size "$PREFILL_TP"
      --kv-transfer-config "$(kv_json kv_producer)"
      "${COMMON[@]}"
    )
    d_cmd=(
      env
      VLLM_USE_V2_MODEL_RUNNER=1
      HF_HUB_OFFLINE=1
      UCX_NET_DEVICES=all
      "VLLM_NIXL_SIDE_CHANNEL_PORT=${NIXL_DECODE_PORT}"
      "CUDA_VISIBLE_DEVICES=${DECODE_DEVICES}"
      vllm serve "$MODEL"
      --port "$DECODE_PORT"
      --tensor-parallel-size "$DECODE_TP"
      --kv-transfer-config "$(kv_json kv_consumer)"
      "${COMMON[@]}"
    )
    if [[ "$MODE" == disagg_sd ]]; then
      d_cmd+=(--speculative-config "$(spec_json "$DRAFT_TP")")
    fi

    run_cmd "${TAG} PREFILL tp=${PREFILL_TP} devices=${PREFILL_DEVICES} port=${PREFILL_PORT}" \
      "prefill" "${p_cmd[@]}"
    run_cmd "${TAG} DECODE  tp=${DECODE_TP} devices=${DECODE_DEVICES} port=${DECODE_PORT}" \
      "decode" "${d_cmd[@]}"

    if [[ "$DO_PROXY" -eq 1 ]]; then
      pscript=$(find_proxy_script) || die "toy_proxy_server.py not found; set --proxy-script"
      if [[ "$PRINT_ONLY" -eq 0 ]]; then
        # Wait for engines before proxy; toy_proxy does not probe /v1/models itself.
        wait_http_ready "http://localhost:${PREFILL_PORT}/v1/models" "prefill" \
          "${PROXY_WAIT_TIMEOUT:-600}"
        wait_http_ready "http://localhost:${DECODE_PORT}/v1/models" "decode" \
          "${PROXY_WAIT_TIMEOUT:-600}"
      fi
      # CLI: tests/v1/kv_connector/nixl_integration/toy_proxy_server.py
      px_cmd=(
        python3 "$pscript"
        --host 127.0.0.1
        --port "$PROXY_PORT"
        --prefiller-hosts localhost
        --prefiller-ports "$PREFILL_PORT"
        --decoder-hosts localhost
        --decoder-ports "$DECODE_PORT"
      )
      run_cmd "${TAG} PROXY port=${PROXY_PORT}" "proxy" "${px_cmd[@]}"
    else
      echo
      echo "# tip: for client traffic, start the Nixl toy proxy, e.g.:"
      echo "#   ./launch_bench_server.sh ${TAG} ... --proxy"
      echo "# or (after engines are up):"
      proxy_path="$(find_proxy_script 2>/dev/null || true)"
      if [[ -n "${proxy_path}" ]]; then
        echo "#   python3 ${proxy_path} --host 127.0.0.1 --port ${PROXY_PORT} \\"
        echo "#       --prefiller-hosts localhost --prefiller-ports ${PREFILL_PORT} \\"
        echo "#       --decoder-hosts localhost --decoder-ports ${DECODE_PORT}"
      fi
    fi

    if [[ "$PRINT_ONLY" -eq 0 ]]; then
      wait
    fi
    ;;

  sd_disagg)
    # Disagg-DFlash: verify (vllm serve) + remote draft_server on separate GPUs.
    PORT="${PORT:-8000}"
    default_sd_disagg_devices "$VERIFY_TP" "$DRAFT_GPUS"
    DEVICES="${DEVICES:-$VERIFY_DEVICES_DEFAULT}"
    DRAFT_DEVICES="${DRAFT_DEVICES:-$DRAFT_DEVICES_DEFAULT}"
    n_v=$(count_csv "$DEVICES")
    n_d=$(count_csv "$DRAFT_DEVICES")
    [[ "$n_v" -eq "$VERIFY_TP" ]] || die "V${VERIFY_TP} needs ${VERIFY_TP} verify devices, got '${DEVICES}'"
    [[ "$n_d" -eq "$DRAFT_GPUS" ]] || die "SD${DRAFT_GPUS} needs ${DRAFT_GPUS} draft devices, got '${DRAFT_DEVICES}'"

    draft_cmd=(
      env
      VLLM_USE_V2_MODEL_RUNNER=1
      HF_HUB_OFFLINE=1
      UCX_NET_DEVICES=all
      "CUDA_VISIBLE_DEVICES=${DRAFT_DEVICES}"
      python3 -m vllm.entrypoints.dflash_draft_server
      --draft-model "$DRAFT_MODEL"
      --target-model "$MODEL"
      --num-speculative-tokens "$NUM_SPEC_TOKENS"
      --bind "$DRAFT_BIND"
      --transport "$DISAGG_DFLASH_TRANSPORT"
      --max-model-len "$MAX_MODEL_LEN"
      --max-num-seqs "$MAX_NUM_SEQS"
      --gpu-memory-utilization "$GPU_MEM_UTIL"
      --block-size "$BLOCK_SIZE"
      --attention-backend FLASH_ATTN
    )
    # Same opt-in as verify: only when --enable-logging-iteration-details.
    if [[ "$ENABLE_LOGGING_ITERATION_DETAILS" -eq 1 ]]; then
      draft_cmd+=(--enable-logging-iteration-details)
    fi
    if [[ "$NIXL_LOG_EVERY" != "-1" ]]; then
      draft_cmd+=(--disagg-dflash-nixl-log-every "$NIXL_LOG_EVERY")
    fi
    verify_cmd=(
      env
      VLLM_USE_V2_MODEL_RUNNER=1
      HF_HUB_OFFLINE=1
      UCX_NET_DEVICES=all
      "CUDA_VISIBLE_DEVICES=${DEVICES}"
      vllm serve "$MODEL"
      --port "$PORT"
      --tensor-parallel-size "$VERIFY_TP"
      --speculative-config "$(spec_json_sd_disagg)"
      "${COMMON[@]}"
    )
    if [[ "$ENABLE_DISAGG_PROFILE" -eq 1 ]]; then
      draft_cmd+=(--enable-sd-timing-model --enable-dflash-draft-profile)
      verify_cmd+=(--enable-sd-timing-model --enable-disagg-dflash-profile)
    fi
    if [[ "$NIXL_LOG_EVERY" != "-1" ]]; then
      verify_cmd+=(--disagg-dflash-nixl-log-every "$NIXL_LOG_EVERY")
    fi

    run_cmd "${TAG} DRAFT devices=${DRAFT_DEVICES} bind=${DRAFT_BIND} transport=${DISAGG_DFLASH_TRANSPORT}" \
      "draft" "${draft_cmd[@]}"
    # Give draft a moment to bind before verify HELLO (print-only skips wait).
    if [[ "$PRINT_ONLY" -eq 0 ]]; then
      sleep 2
    fi
    run_cmd "${TAG} VERIFY tp=${VERIFY_TP} devices=${DEVICES} port=${PORT} draft=${DRAFT_ADDR}" \
      "verify" "${verify_cmd[@]}"

    echo
    echo "# Disagg-DFlash: for overlap debug logs add --enable-disagg-profile (skews latency)"
    echo "# NIXL xfer logs (no extra CUDA sync): --nixl-log-every 0"
    echo "# A/B: --no-wave-schedule and/or --no-disagg-async"
    echo "# Sync EngineCore: --no-async-scheduling"

    if [[ "$PRINT_ONLY" -eq 0 ]]; then
      wait
    fi
    ;;
esac
