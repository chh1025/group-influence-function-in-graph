#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-launch}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION_NAME="${SESSION_NAME:-group_inf_mvp_small_${RUN_STAMP}}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
DATASETS="${DATASETS:-cora_public,citeseer_public,texas,cornell}"
MODELS="${MODELS:-GCN}"
LAYERS="${LAYERS:-2}"
CANDIDATE_TYPES="${CANDIDATE_TYPES:-random,top_abs,mixed}"
FEATURE_TYPES="${FEATURE_TYPES:-cheap}"
CLUSTERING_METHODS="${CLUSTERING_METHODS:-cheap_kmeans,random}"
NUM_CANDIDATES="${NUM_CANDIDATES:-100}"
POOL_SIZE="${POOL_SIZE:-200}"
NUM_CLUSTERS="${NUM_CLUSTERS:-5}"
OUTPUT_NODE_SCOPE="${OUTPUT_NODE_SCOPE:-eval}"
OUTPUT_PCA_DIM="${OUTPUT_PCA_DIM:-32}"
SEED="${SEED:-0}"
EVAL_METRIC="${EVAL_METRIC:-mean_validation_loss}"
HESSIAN_TYPE="${HESSIAN_TYPE:-GNH}"
LR="${LR:-0.01}"
HIDDEN_DIM="${HIDDEN_DIM:-16}"
EPOCHS="${EPOCHS:-1000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
DAMP="${DAMP:-0.1}"
SCALE="${SCALE:-32}"
LISSA_ITER="${LISSA_ITER:-100}"
PBRF_EPOCHS="${PBRF_EPOCHS:-1}"
PBRF_WEIGHT_DECAY="${PBRF_WEIGHT_DECAY:-0.0}"
NUM_HEADS="${NUM_HEADS:-8}"
CACHE_ROOT="${CACHE_ROOT:-results/group_influence}"
RUN_ROOT="${RUN_ROOT:-results/group_influence_mvp_small/${RUN_STAMP}}"
SKIP_PBRF="${SKIP_PBRF:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_WORKERS="${NUM_WORKERS:-${#GPU_ARRAY[@]}}"

safe_id() {
  printf "%s" "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_-' '_'
}

build_jobs() {
  JOBS=()
  IFS=',' read -r -a DATASET_ARRAY <<< "$DATASETS"
  IFS=',' read -r -a MODEL_ARRAY <<< "$MODELS"
  IFS=',' read -r -a LAYER_ARRAY <<< "$LAYERS"
  IFS=',' read -r -a CANDIDATE_ARRAY <<< "$CANDIDATE_TYPES"
  for dataset in "${DATASET_ARRAY[@]}"; do
    for model in "${MODEL_ARRAY[@]}"; do
      for layer in "${LAYER_ARRAY[@]}"; do
        for candidate_type in "${CANDIDATE_ARRAY[@]}"; do
          JOBS+=("${dataset}|${model}|${layer}|${candidate_type}")
        done
      done
    done
  done
}

run_candidate_job() {
  local job_index="$1"
  local dataset="$2"
  local model="$3"
  local layer="$4"
  local candidate_type="$5"

  local dataset_id model_id candidate_id base_id candidate_run_id candidate_dir pool_size
  dataset_id="$(safe_id "$dataset")"
  model_id="$(safe_id "$model")"
  candidate_id="$(safe_id "$candidate_type")"
  base_id="${dataset_id}_${model_id}_l${layer}_${candidate_id}_n${NUM_CANDIDATES}_${RUN_STAMP}"
  candidate_run_id="${base_id}"
  candidate_dir="${CACHE_ROOT}/candidate_edges/${candidate_run_id}"
  pool_size="$NUM_CANDIDATES"
  if [[ "$candidate_type" == "top_abs" || "$candidate_type" == "mixed" || "$candidate_type" == "top_positive_negative" ]]; then
    pool_size="$POOL_SIZE"
  fi

  echo "[job ${job_index}] dataset=${dataset} model=${model} layer=${layer} candidate=${candidate_type} gpu=${CUDA_VISIBLE_DEVICES}"

  if [[ -d "$candidate_dir" && ! -f "${candidate_dir}/candidate_edges.pt" ]]; then
    echo "[job ${job_index}] remove incomplete candidate_dir=${candidate_dir}"
    rm -rf "$candidate_dir"
  fi
  if [[ ! -f "${candidate_dir}/candidate_edges.pt" ]]; then
    candidate_args=(
      --dataset "$dataset"
      --model "$model"
      --num-layers "$layer"
      --hidden-dim "$HIDDEN_DIM"
      --seed "$SEED"
      --element-type edge_removal
      --candidate-type "$candidate_type"
      --num-candidates "$NUM_CANDIDATES"
      --pool-size "$pool_size"
      --eval-metric "$EVAL_METRIC"
      --hessian-type "$HESSIAN_TYPE"
      --lr "$LR"
      --epochs "$EPOCHS"
      --weight-decay "$WEIGHT_DECAY"
      --damp "$DAMP"
      --scale "$SCALE"
      --lissa-iter "$LISSA_ITER"
      --pbrf-epochs "$PBRF_EPOCHS"
      --pbrf-weight-decay "$PBRF_WEIGHT_DECAY"
      --num-heads "$NUM_HEADS"
      --cache-root "$CACHE_ROOT"
      --run-id "$candidate_run_id"
    )
    if [[ "$candidate_type" == "mixed" || "$candidate_type" == "top_positive_negative" ]]; then
      local mixed_positive_count mixed_negative_count
      mixed_positive_count=$(( NUM_CANDIDATES / 2 ))
      mixed_negative_count=$(( NUM_CANDIDATES - mixed_positive_count ))
      candidate_args+=(--mixed-positive-count "$mixed_positive_count" --mixed-negative-count "$mixed_negative_count")
    fi
    python experiments/group_influence/build_candidate_edges.py "${candidate_args[@]}"
  else
    echo "[job ${job_index}] reuse candidate_dir=${candidate_dir}"
  fi

  IFS=',' read -r -a FEATURE_ARRAY <<< "$FEATURE_TYPES"
  IFS=',' read -r -a CLUSTER_ARRAY <<< "$CLUSTERING_METHODS"
  for feature_type in "${FEATURE_ARRAY[@]}"; do
    for clustering_method in "${CLUSTER_ARRAY[@]}"; do
      local feature_id cluster_id cluster_run_id cluster_dir agg_run_id agg_args
      feature_id="$(safe_id "$feature_type")"
      cluster_id="$(safe_id "$clustering_method")"
      cluster_run_id="${base_id}_${feature_id}_${cluster_id}_k${NUM_CLUSTERS}"
      cluster_dir="${CACHE_ROOT}/feature_clustering/${cluster_run_id}"
      if [[ -d "$cluster_dir" && ! -f "${cluster_dir}/cluster_labels.pt" ]]; then
        echo "[job ${job_index}] remove incomplete clustering_dir=${cluster_dir}"
        rm -rf "$cluster_dir"
      fi
      if [[ ! -f "${cluster_dir}/cluster_labels.pt" ]]; then
        cluster_args=(
          --candidate-dir "$candidate_dir"
          --feature-type "$feature_type"
          --clustering-method "$clustering_method"
          --num-clusters "$NUM_CLUSTERS"
          --run-id "$cluster_run_id"
          --cache-root "$CACHE_ROOT"
        )
        if [[ "$feature_type" == "logits_delta" ]]; then
          cluster_args+=(--output-node-scope "$OUTPUT_NODE_SCOPE" --output-pca-dim "$OUTPUT_PCA_DIM")
        fi
        python experiments/group_influence/run_feature_clustering.py "${cluster_args[@]}"
      else
        echo "[job ${job_index}] reuse clustering_dir=${cluster_dir}"
      fi

      agg_run_id="${cluster_run_id}_cind_pbrf${PBRF_EPOCHS}"
      agg_args=(
        --candidate-dir "$candidate_dir"
        --clustering-dir "$cluster_dir"
        --aggregation-method independent_cluster_sum
        --pbrf-epochs "$PBRF_EPOCHS"
        --cache-root "$CACHE_ROOT"
        --run-id "$agg_run_id"
      )
      if [[ "$SKIP_PBRF" == "1" ]]; then
        agg_args+=(--skip-pbrf)
      fi
      if [[ -d "${CACHE_ROOT}/aggregation/${agg_run_id}" && ! -f "${CACHE_ROOT}/aggregation/${agg_run_id}/aggregation_result.csv" ]]; then
        echo "[job ${job_index}] remove incomplete aggregation_dir=${CACHE_ROOT}/aggregation/${agg_run_id}"
        rm -rf "${CACHE_ROOT}/aggregation/${agg_run_id}"
      fi
      if [[ ! -f "${CACHE_ROOT}/aggregation/${agg_run_id}/aggregation_result.csv" ]]; then
        python experiments/group_influence/run_aggregation.py "${agg_args[@]}"
      else
        echo "[job ${job_index}] reuse aggregation_dir=${CACHE_ROOT}/aggregation/${agg_run_id}"
      fi
    done
  done
  echo "[job ${job_index}] done"
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
  local idx job dataset model layer candidate_type
  for idx in "${!JOBS[@]}"; do
    if (( idx % NUM_WORKERS != worker_id )); then
      continue
    fi
    job="${JOBS[$idx]}"
    IFS='|' read -r dataset model layer candidate_type <<< "$job"
    run_candidate_job "$idx" "$dataset" "$model" "$layer" "$candidate_type"
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
    echo "datasets=${DATASETS}"
    echo "models=${MODELS}"
    echo "layers=${LAYERS}"
    echo "candidate_types=${CANDIDATE_TYPES}"
    echo "feature_types=${FEATURE_TYPES}"
    echo "clustering_methods=${CLUSTERING_METHODS}"
    echo "num_candidates=${NUM_CANDIDATES}"
    echo "pool_size=${POOL_SIZE}"
    echo "num_clusters=${NUM_CLUSTERS}"
    echo "output_node_scope=${OUTPUT_NODE_SCOPE}"
    echo "output_pca_dim=${OUTPUT_PCA_DIM}"
    echo "pbrf_epochs=${PBRF_EPOCHS}"
    echo "skip_pbrf=${SKIP_PBRF}"
    echo "total_candidate_jobs=${#JOBS[@]}"
  } > "${RUN_ROOT}/run_config.txt"

  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "error: tmux session already exists: ${SESSION_NAME}" >&2
    exit 1
  fi

  local worker_id gpu_id command
  for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
    gpu_id="${GPU_ARRAY[$((worker_id % ${#GPU_ARRAY[@]}))]}"
    command="RUN_STAMP='${RUN_STAMP}' SESSION_NAME='${SESSION_NAME}' GPU_IDS='${GPU_IDS}' NUM_WORKERS='${NUM_WORKERS}' DATASETS='${DATASETS}' MODELS='${MODELS}' LAYERS='${LAYERS}' CANDIDATE_TYPES='${CANDIDATE_TYPES}' FEATURE_TYPES='${FEATURE_TYPES}' CLUSTERING_METHODS='${CLUSTERING_METHODS}' NUM_CANDIDATES='${NUM_CANDIDATES}' POOL_SIZE='${POOL_SIZE}' NUM_CLUSTERS='${NUM_CLUSTERS}' OUTPUT_NODE_SCOPE='${OUTPUT_NODE_SCOPE}' OUTPUT_PCA_DIM='${OUTPUT_PCA_DIM}' SEED='${SEED}' EVAL_METRIC='${EVAL_METRIC}' HESSIAN_TYPE='${HESSIAN_TYPE}' LR='${LR}' HIDDEN_DIM='${HIDDEN_DIM}' EPOCHS='${EPOCHS}' WEIGHT_DECAY='${WEIGHT_DECAY}' DAMP='${DAMP}' SCALE='${SCALE}' LISSA_ITER='${LISSA_ITER}' PBRF_EPOCHS='${PBRF_EPOCHS}' PBRF_WEIGHT_DECAY='${PBRF_WEIGHT_DECAY}' NUM_HEADS='${NUM_HEADS}' CACHE_ROOT='${CACHE_ROOT}' RUN_ROOT='${RUN_ROOT}' SKIP_PBRF='${SKIP_PBRF}' bash '${SCRIPT_DIR}/run_group_influence_mvp_small.sh' worker '${worker_id}' '${gpu_id}' 2>&1 | tee '${RUN_ROOT}/logs/worker_${worker_id}.log'"
    if [[ "$worker_id" == "0" ]]; then
      tmux new-session -d -s "$SESSION_NAME" -n "gpu${gpu_id}" "$command"
    else
      tmux split-window -t "$SESSION_NAME" "$command"
      tmux select-layout -t "$SESSION_NAME" tiled >/dev/null
    fi
  done

  echo "[group-influence-mvp-small]"
  echo "session=${SESSION_NAME}"
  echo "run_root=${RUN_ROOT}"
  echo "total_candidate_jobs=${#JOBS[@]}"
  echo "total_aggregation_runs=$(( ${#JOBS[@]} * $(awk -F, '{print NF}' <<< "$FEATURE_TYPES") * $(awk -F, '{print NF}' <<< "$CLUSTERING_METHODS") ))"
  echo "attach: tmux attach -t ${SESSION_NAME}"
}

case "$MODE" in
  launch)
    launch_tmux
    ;;
  worker)
    run_worker "$2" "$3"
    ;;
  *)
    echo "error: invalid mode '${MODE}' (use: launch|worker)" >&2
    exit 1
    ;;
esac
