#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source conda.sh

MODE="${1:-launch}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION_NAME="${SESSION_NAME:-group_inf_seq_long_${RUN_STAMP}}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SOURCE_STAMP="${SOURCE_STAMP:-20260506_142428}"
SOURCE_AGG_GLOB="${SOURCE_AGG_GLOB:-results/group_influence/aggregation/*${SOURCE_STAMP}*/aggregation_result.json}"
ORDER_POLICIES="${ORDER_POLICIES:-small_abs_cluster_influence_first}"
CACHE_ROOT="${CACHE_ROOT:-results/group_influence}"
RUN_ROOT="${RUN_ROOT:-results/group_influence_sequential_existing/${RUN_STAMP}}"
MAX_RUNS="${MAX_RUNS:-}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_WORKERS="${NUM_WORKERS:-${#GPU_ARRAY[@]}}"

build_jobs() {
  JOBS=()
  local line
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    JOBS+=("$line")
  done < <(
    SOURCE_AGG_GLOB="$SOURCE_AGG_GLOB" RUN_STAMP="$RUN_STAMP" MAX_RUNS="$MAX_RUNS" python - <<'PY'
import glob
import json
import os
import sys

paths = sorted(glob.glob(os.environ["SOURCE_AGG_GLOB"]))
max_runs = int(os.environ.get("MAX_RUNS") or 0)
emitted = 0
for result_path in paths:
    agg_dir = os.path.dirname(result_path)
    metadata_path = os.path.join(agg_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        continue
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    config = metadata.get("config", {})
    candidate_dir = config.get("candidate_dir")
    clustering_dir = config.get("clustering_dir")
    if not candidate_dir or not clustering_dir:
        print(f"skip missing dirs: {agg_dir}", file=sys.stderr)
        continue
    run_id = f"{os.path.basename(agg_dir)}_seq_{os.environ['RUN_STAMP']}"
    print("\t".join([agg_dir, candidate_dir, clustering_dir, run_id]))
    emitted += 1
    if max_runs and emitted >= max_runs:
        break
PY
  )
}

run_job() {
  local job_index="$1"
  local agg_dir="$2"
  local candidate_dir="$3"
  local clustering_dir="$4"
  local run_id="$5"
  local seq_dir="${CACHE_ROOT}/sequential_aggregation/${run_id}"

  echo "[job ${job_index}] gpu=${CUDA_VISIBLE_DEVICES} aggregation_dir=${agg_dir}"
  if [[ -d "$seq_dir" && ! -f "${seq_dir}/sequential_result.csv" ]]; then
    echo "[job ${job_index}] remove incomplete sequential_dir=${seq_dir}"
    rm -rf "$seq_dir"
  fi
  if [[ -f "${seq_dir}/sequential_result.csv" ]]; then
    echo "[job ${job_index}] reuse sequential_dir=${seq_dir}"
    return
  fi

  python experiments/group_influence/run_sequential_aggregation.py \
    --candidate-dir "$candidate_dir" \
    --clustering-dir "$clustering_dir" \
    --baseline-aggregation-dir "$agg_dir" \
    --order-policies "$ORDER_POLICIES" \
    --cache-root "$CACHE_ROOT" \
    --run-id "$run_id"
}

run_worker() {
  local worker_id="$1"
  local gpu_id="$2"

  source conda.sh
  export CUDA_VISIBLE_DEVICES="$gpu_id"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/.cache}"
  mkdir -p "$RUN_ROOT/logs" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

  build_jobs
  echo "[worker ${worker_id}] gpu=${gpu_id} total_jobs=${#JOBS[@]} run_stamp=${RUN_STAMP}"
  local idx job agg_dir candidate_dir clustering_dir run_id
  for idx in "${!JOBS[@]}"; do
    if (( idx % NUM_WORKERS != worker_id )); then
      continue
    fi
    job="${JOBS[$idx]}"
    IFS=$'\t' read -r agg_dir candidate_dir clustering_dir run_id <<< "$job"
    run_job "$idx" "$agg_dir" "$candidate_dir" "$clustering_dir" "$run_id"
  done
  touch "${RUN_ROOT}/worker_${worker_id}.done"
  echo "[worker ${worker_id}] all assigned jobs done"
}

launch_tmux() {
  build_jobs
  mkdir -p "$RUN_ROOT/logs"
  {
    echo "run_stamp=${RUN_STAMP}"
    echo "session=${SESSION_NAME}"
    echo "gpu_ids=${GPU_IDS}"
    echo "num_workers=${NUM_WORKERS}"
    echo "source_stamp=${SOURCE_STAMP}"
    echo "source_agg_glob=${SOURCE_AGG_GLOB}"
    echo "order_policies=${ORDER_POLICIES}"
    echo "cache_root=${CACHE_ROOT}"
    echo "run_root=${RUN_ROOT}"
    echo "max_runs=${MAX_RUNS:-all}"
    echo "total_sequential_runs=${#JOBS[@]}"
  } > "${RUN_ROOT}/run_config.txt"

  if (( ${#JOBS[@]} == 0 )); then
    echo "error: no aggregation jobs matched SOURCE_AGG_GLOB=${SOURCE_AGG_GLOB}" >&2
    exit 1
  fi
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "error: tmux session already exists: ${SESSION_NAME}" >&2
    exit 1
  fi

  local worker_id gpu_id command
  for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
    gpu_id="${GPU_ARRAY[$((worker_id % ${#GPU_ARRAY[@]}))]}"
    command="RUN_STAMP='${RUN_STAMP}' SESSION_NAME='${SESSION_NAME}' GPU_IDS='${GPU_IDS}' NUM_WORKERS='${NUM_WORKERS}' SOURCE_STAMP='${SOURCE_STAMP}' SOURCE_AGG_GLOB='${SOURCE_AGG_GLOB}' ORDER_POLICIES='${ORDER_POLICIES}' CACHE_ROOT='${CACHE_ROOT}' RUN_ROOT='${RUN_ROOT}' MAX_RUNS='${MAX_RUNS}' bash '${SCRIPT_DIR}/run_group_influence_sequential_existing.sh' worker '${worker_id}' '${gpu_id}' 2>&1 | tee '${RUN_ROOT}/logs/worker_${worker_id}.log'"
    if [[ "$worker_id" == "0" ]]; then
      tmux new-session -d -s "$SESSION_NAME" -n "gpu${gpu_id}" "$command"
    else
      tmux split-window -t "$SESSION_NAME" "$command"
      tmux select-layout -t "$SESSION_NAME" tiled >/dev/null
    fi
  done

  echo "[group-influence-sequential-existing]"
  echo "session=${SESSION_NAME}"
  echo "run_root=${RUN_ROOT}"
  echo "total_sequential_runs=${#JOBS[@]}"
  echo "order_policies=${ORDER_POLICIES}"
  echo "attach: tmux attach -t ${SESSION_NAME}"
}

status() {
  build_jobs
  local total done_count tmux_state
  done_count=0
  local job agg_dir candidate_dir clustering_dir run_id seq_dir
  for job in "${JOBS[@]}"; do
    IFS=$'\t' read -r agg_dir candidate_dir clustering_dir run_id <<< "$job"
    seq_dir="${CACHE_ROOT}/sequential_aggregation/${run_id}"
    if [[ -f "${seq_dir}/sequential_result.csv" ]]; then
      done_count=$((done_count + 1))
    fi
  done
  total="${#JOBS[@]}"
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux_state="alive"
  else
    tmux_state="not-found"
  fi
  echo "[group-influence-sequential-existing-status]"
  echo "session=${SESSION_NAME} (${tmux_state})"
  echo "run_root=${RUN_ROOT}"
  echo "done=${done_count}/${total}"
  if [[ -d "${RUN_ROOT}/logs" ]]; then
    echo "worker_done=$(find "${RUN_ROOT}" -maxdepth 1 -name 'worker_*.done' | wc -l)/${NUM_WORKERS}"
    if rg -n -i "traceback|runtimeerror|cuda out of memory|killed|exception" "${RUN_ROOT}/logs" >/tmp/group_inf_seq_status_errors.$$ 2>/dev/null; then
      echo "errors=present"
      head -n 20 /tmp/group_inf_seq_status_errors.$$
    else
      echo "errors=none-observed"
    fi
    rm -f /tmp/group_inf_seq_status_errors.$$
  fi
}

case "$MODE" in
  launch)
    launch_tmux
    ;;
  worker)
    run_worker "$2" "$3"
    ;;
  status)
    status
    ;;
  *)
    echo "error: invalid mode '${MODE}' (use: launch|worker|status)" >&2
    exit 1
    ;;
esac
