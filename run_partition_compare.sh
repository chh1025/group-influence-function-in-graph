#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROFILE="${1:-small}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "/data_seoul/undergrad_hh/miniconda3/envs/EIF/bin/python" ]]; then
  PYTHON_BIN="/data_seoul/undergrad_hh/miniconda3/envs/EIF/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "error: neither project python nor system python is available." >&2
  exit 1
fi

RATIOS="${RATIOS:-1,5,10,30,50,80}"
PARTITION_METHODS="${PARTITION_METHODS:-metis,spectral,local_ppr}"
PARTITION_STRATEGY="${PARTITION_STRATEGY:-}"
PARTITION_STRATEGIES="${PARTITION_STRATEGIES:-}"
ELEMENT_TYPE="${ELEMENT_TYPE:-edge_removal}"
NUM_CLUSTERS="${NUM_CLUSTERS:-3}"
NUM_CLUSTERS_LIST="${NUM_CLUSTERS_LIST:-}"
NUM_REMOVAL_CANDIDATES="${NUM_REMOVAL_CANDIDATES:-50}"
MODELS="${MODELS:-}"
LAYERS="${LAYERS:-}"
DATASETS="${DATASETS:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MAX_WORKERS="${MAX_WORKERS:-}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-1.0}"
SEED="${SEED:-0}"
HESSIAN_TYPE="${HESSIAN_TYPE:-GNH}"
EVAL_METRIC="${EVAL_METRIC:-mean_validation_loss}"
LR="${LR:-0.01}"
HIDDEN_DIM="${HIDDEN_DIM:-16}"
EPOCHS="${EPOCHS:-1000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
PBRF_EPOCHS="${PBRF_EPOCHS:-1000}"
LISSA_ITER="${LISSA_ITER:-1000}"
SCALE="${SCALE:-32}"
PARTITION_SHARED_ENDPOINT_BONUS="${PARTITION_SHARED_ENDPOINT_BONUS:-1.0}"
PARTITION_DISTANCE_SCALE="${PARTITION_DISTANCE_SCALE:-1.0}"
PARTITION_DISTANCE_MAX_HOPS="${PARTITION_DISTANCE_MAX_HOPS:-3}"
PARTITION_MIN_WEIGHT="${PARTITION_MIN_WEIGHT:-1e-8}"
HYBRID_CROSS_OWNER_AFFINITY_SCALE="${HYBRID_CROSS_OWNER_AFFINITY_SCALE:-}"
COCO_EPOCHS="${COCO_EPOCHS:-200}"
COCO_LR="${COCO_LR:-0.001}"
COCO_HIDDEN_DIM="${COCO_HIDDEN_DIM:-256}"
COCO_ACTIVATION="${COCO_ACTIVATION:-ident}"
COCO_COMPACT_K="${COCO_COMPACT_K:-64}"
COCO_STAGE_NUM="${COCO_STAGE_NUM:-10}"
COCO_BETA="${COCO_BETA:-1.0}"
COCO_FILTER_STEPS="${COCO_FILTER_STEPS:-2}"
COCO_ALPHA="${COCO_ALPHA:-0.2}"
COCO_CONSISTENCY_T="${COCO_CONSISTENCY_T:-0.02}"
COCO_MEMORY_SIZE="${COCO_MEMORY_SIZE:-0}"
COCO_MEMORY_MULTIPLIER="${COCO_MEMORY_MULTIPLIER:-10}"
COCO_PCA_DIM="${COCO_PCA_DIM:--1}"
COCO_USE_DIFFUSION="${COCO_USE_DIFFUSION:-1}"
COCO_KMEANS_DISTANCE="${COCO_KMEANS_DISTANCE:-euclidean}"
COCO_DEVICE="${COCO_DEVICE:-auto}"
COCO_CACHE_DIR="${COCO_CACHE_DIR:-results/coco_line_graph_cache}"
COCO_FORCE_RETRAIN="${COCO_FORCE_RETRAIN:-0}"
COCO_MAX_LINE_GRAPH_NODES="${COCO_MAX_LINE_GRAPH_NODES:-6000}"
AUTO_K_METHOD="${AUTO_K_METHOD:-none}"
AUTO_K_MIN="${AUTO_K_MIN:-1}"
AUTO_K_MAX="${AUTO_K_MAX:-8}"
AUTO_K_MAX_RATIO="${AUTO_K_MAX_RATIO:-1.0}"
AUTO_K_MIN_CLUSTER_SIZE="${AUTO_K_MIN_CLUSTER_SIZE:-1}"
AUTO_K_MAX_CLUSTER_SIZE="${AUTO_K_MAX_CLUSTER_SIZE:-0}"
AUTO_K_TINY_GRAPH_THRESHOLD="${AUTO_K_TINY_GRAPH_THRESHOLD:-4}"
AUTO_K_NUM_RESTARTS="${AUTO_K_NUM_RESTARTS:-10}"
AUTO_K_RANDOM_SEED="${AUTO_K_RANDOM_SEED:-0}"
AUTO_K_SILHOUETTE_METRIC="${AUTO_K_SILHOUETTE_METRIC:-euclidean}"
AUTO_K_STABILITY_TRIALS="${AUTO_K_STABILITY_TRIALS:-10}"
AUTO_K_STABILITY_EDGE_DROPOUT="${AUTO_K_STABILITY_EDGE_DROPOUT:-0.05}"
AUTO_K_WEAK_SILHOUETTE_THRESHOLD="${AUTO_K_WEAK_SILHOUETTE_THRESHOLD:-0.05}"
AUTO_K_PREFER_SMALLER_K="${AUTO_K_PREFER_SMALLER_K:-1}"

ARGS=(
  --profile "$PROFILE"
  --ratios "$RATIOS"
  --partition-methods "$PARTITION_METHODS"
  --element-type "$ELEMENT_TYPE"
  --num-clusters "$NUM_CLUSTERS"
  --num-removal-candidates "$NUM_REMOVAL_CANDIDATES"
  --gpu-ids "$GPU_IDS"
  --poll-interval-sec "$POLL_INTERVAL_SEC"
  --seed "$SEED"
  --hessian-type "$HESSIAN_TYPE"
  --eval-metric "$EVAL_METRIC"
  --lr "$LR"
  --hidden-dim "$HIDDEN_DIM"
  --epochs "$EPOCHS"
  --weight-decay "$WEIGHT_DECAY"
  --pbrf-epochs "$PBRF_EPOCHS"
  --lissa-iter "$LISSA_ITER"
  --scale "$SCALE"
  --partition-shared-endpoint-bonus "$PARTITION_SHARED_ENDPOINT_BONUS"
  --partition-distance-scale "$PARTITION_DISTANCE_SCALE"
  --partition-distance-max-hops "$PARTITION_DISTANCE_MAX_HOPS"
  --partition-min-weight "$PARTITION_MIN_WEIGHT"
  --coco-epochs "$COCO_EPOCHS"
  --coco-lr "$COCO_LR"
  --coco-hidden-dim "$COCO_HIDDEN_DIM"
  --coco-activation "$COCO_ACTIVATION"
  --coco-compact-k "$COCO_COMPACT_K"
  --coco-stage-num "$COCO_STAGE_NUM"
  --coco-beta "$COCO_BETA"
  --coco-filter-steps "$COCO_FILTER_STEPS"
  --coco-alpha "$COCO_ALPHA"
  --coco-consistency-t "$COCO_CONSISTENCY_T"
  --coco-memory-size "$COCO_MEMORY_SIZE"
  --coco-memory-multiplier "$COCO_MEMORY_MULTIPLIER"
  --coco-pca-dim "$COCO_PCA_DIM"
  --coco-use-diffusion "$COCO_USE_DIFFUSION"
  --coco-kmeans-distance "$COCO_KMEANS_DISTANCE"
  --coco-device "$COCO_DEVICE"
  --coco-cache-dir "$COCO_CACHE_DIR"
  --coco-force-retrain "$COCO_FORCE_RETRAIN"
  --coco-max-line-graph-nodes "$COCO_MAX_LINE_GRAPH_NODES"
  --auto-k-method "$AUTO_K_METHOD"
  --auto-k-min "$AUTO_K_MIN"
  --auto-k-max "$AUTO_K_MAX"
  --auto-k-max-ratio "$AUTO_K_MAX_RATIO"
  --auto-k-min-cluster-size "$AUTO_K_MIN_CLUSTER_SIZE"
  --auto-k-max-cluster-size "$AUTO_K_MAX_CLUSTER_SIZE"
  --auto-k-tiny-graph-threshold "$AUTO_K_TINY_GRAPH_THRESHOLD"
  --auto-k-num-restarts "$AUTO_K_NUM_RESTARTS"
  --auto-k-random-seed "$AUTO_K_RANDOM_SEED"
  --auto-k-silhouette-metric "$AUTO_K_SILHOUETTE_METRIC"
  --auto-k-stability-trials "$AUTO_K_STABILITY_TRIALS"
  --auto-k-stability-edge-dropout "$AUTO_K_STABILITY_EDGE_DROPOUT"
  --auto-k-weak-silhouette-threshold "$AUTO_K_WEAK_SILHOUETTE_THRESHOLD"
  --auto-k-prefer-smaller-k "$AUTO_K_PREFER_SMALLER_K"
)

if [[ -n "${PARTITION_STRATEGY}" ]]; then
  ARGS+=(--partition-strategy "$PARTITION_STRATEGY")
fi

if [[ -n "${PARTITION_STRATEGIES}" ]]; then
  ARGS+=(--partition-strategies "$PARTITION_STRATEGIES")
fi

if [[ -n "${SUMMARY_CSV:-}" ]]; then
  ARGS+=(--summary-csv "$SUMMARY_CSV")
fi

if [[ -n "${JOB_DIR:-}" ]]; then
  ARGS+=(--job-dir "$JOB_DIR")
fi

if [[ -n "${MODELS}" ]]; then
  ARGS+=(--models "$MODELS")
fi

if [[ -n "${LAYERS}" ]]; then
  ARGS+=(--layers "$LAYERS")
fi

if [[ -n "${DATASETS}" ]]; then
  ARGS+=(--datasets "$DATASETS")
fi

if [[ -n "${NUM_CLUSTERS_LIST}" ]]; then
  ARGS+=(--num-clusters-list "$NUM_CLUSTERS_LIST")
fi

if [[ -n "${HYBRID_CROSS_OWNER_AFFINITY_SCALE}" ]]; then
  ARGS+=(--hybrid-cross-owner-affinity-scale "$HYBRID_CROSS_OWNER_AFFINITY_SCALE")
fi

if [[ -n "${MAX_RUNS:-}" ]]; then
  ARGS+=(--max-runs "$MAX_RUNS")
fi

if [[ -n "${MAX_WORKERS}" ]]; then
  ARGS+=(--max-workers "$MAX_WORKERS")
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/.cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

cat <<EOF
[partition-compare]
profile=$PROFILE
python=$PYTHON_BIN
ratios=$RATIOS
partition_methods=$PARTITION_METHODS
partition_strategy=${PARTITION_STRATEGY:-candidate_local_affinity}
partition_strategies=${PARTITION_STRATEGIES:-single}
element_type=$ELEMENT_TYPE
num_clusters=${NUM_CLUSTERS_LIST:-$NUM_CLUSTERS}
num_removal_candidates=$NUM_REMOVAL_CANDIDATES
models=${MODELS:-profile-default}
layers=${LAYERS:-profile-default}
datasets=${DATASETS:-profile-default}
gpu_ids=$GPU_IDS
max_workers=${MAX_WORKERS:-auto}
job_dir=${JOB_DIR:-auto}
poll_interval_sec=$POLL_INTERVAL_SEC
seed=$SEED
auto_k_method=$AUTO_K_METHOD
auto_k_range=${AUTO_K_MIN}-${AUTO_K_MAX}
auto_k_min_cluster_size=$AUTO_K_MIN_CLUSTER_SIZE
auto_k_max_cluster_size=$AUTO_K_MAX_CLUSTER_SIZE
hessian_type=$HESSIAN_TYPE
eval_metric=$EVAL_METRIC
lr=$LR
hidden_dim=$HIDDEN_DIM
epochs=$EPOCHS
weight_decay=$WEIGHT_DECAY
pbrf_epochs=$PBRF_EPOCHS
lissa_iter=$LISSA_ITER
scale=$SCALE
hybrid_cross_owner_affinity_scale=${HYBRID_CROSS_OWNER_AFFINITY_SCALE:-default}
coco_epochs=$COCO_EPOCHS
coco_hidden_dim=$COCO_HIDDEN_DIM
coco_cache_dir=$COCO_CACHE_DIR
coco_max_line_graph_nodes=$COCO_MAX_LINE_GRAPH_NODES
EOF

"$PYTHON_BIN" run_partition_influence_exp.py "${ARGS[@]}"
