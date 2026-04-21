#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Purpose:
# - Run the groupwise-metric influence comparison sweep over the same model/layer/dataset
#   grid style used by run_experiments.sh.
# - Each run writes a candidate_results.csv that contains:
#   calculate_influence_total            -> existing method
#   clusterwise_fixed_theta_total        -> new groupwise method
#   pbrf_total                           -> actual proxy target
#
# Default behavior:
# - profile: small
# - experiment family: large_drop_influence-style ratio sweep
# - concurrency: up to 4 jobs in parallel, one GPU per job via main.py HYDRA_JOB_NUM % 4
#
# Important:
# - Groupwise proxy extraction is much more expensive than the original pipeline.
# - For that reason the default NUM_REMOVAL_CANDIDATES is 5, not 50.
# - If you want the original candidate count, override:
#     NUM_REMOVAL_CANDIDATES=50 ./run_groupwise_compare.sh small

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

MODELS_FULL="${MODELS_FULL:-sgc,gcn,gat,chebnet}"
LAYERS_FULL="${LAYERS_FULL:-2,4,6,12}"
DATASETS_FULL="${DATASETS_FULL:-cora,citeseer,pubmed,computers,photo,chameleon,actor,squirrel,texas,cornell}"

MODELS_SMALL="${MODELS_SMALL:-gcn,gat}"
LAYERS_SMALL="${LAYERS_SMALL:-2,4}"
DATASETS_SMALL="${DATASETS_SMALL:-cora,citeseer,texas,cornell}"

case "$PROFILE" in
  small)
    MODELS="${MODELS:-$MODELS_SMALL}"
    LAYERS="${LAYERS:-$LAYERS_SMALL}"
    DATASETS="${DATASETS:-$DATASETS_SMALL}"
    ;;
  full)
    MODELS="${MODELS:-$MODELS_FULL}"
    LAYERS="${LAYERS:-$LAYERS_FULL}"
    DATASETS="${DATASETS:-$DATASETS_FULL}"
    ;;
  *)
    echo "error: invalid profile '$PROFILE' (use: small|full)" >&2
    exit 1
    ;;
esac

RATIOS="${RATIOS:-1,5,10,20,30,50,70,90}"
SEED="${SEED:-0}"
NUM_REMOVAL_CANDIDATES="${NUM_REMOVAL_CANDIDATES:-50}"

EVAL_METRIC="${EVAL_METRIC:-mean_validation_loss}"
ELEMENT_TYPE="${ELEMENT_TYPE:-edge_removal}"
HESSIAN_TYPE="${HESSIAN_TYPE:-GNH}"
LISSA_ITER="${LISSA_ITER:-1000}"
SCALE="${SCALE:-32}"

GROUPWISE_ALPHA_REPR="${GROUPWISE_ALPHA_REPR:-1.0}"
GROUPWISE_BETA_PROXY="${GROUPWISE_BETA_PROXY:-1.0}"
GROUPWISE_PROBE_DIM="${GROUPWISE_PROBE_DIM:-4}"
GROUPWISE_PROBE_SEED="${GROUPWISE_PROBE_SEED:-0}"
GROUPWISE_NORMALIZE_PROXIES="${GROUPWISE_NORMALIZE_PROXIES:-1}"
GROUPWISE_DAMPING="${GROUPWISE_DAMPING:-1e-3}"
GROUPWISE_SPLIT_THRESHOLD="${GROUPWISE_SPLIT_THRESHOLD:-1e-6}"
GROUPWISE_MERGE_THRESHOLD="${GROUPWISE_MERGE_THRESHOLD:-1e-6}"
GROUPWISE_MOVE_THRESHOLD="${GROUPWISE_MOVE_THRESHOLD:-1e-6}"
GROUPWISE_RADIUS_UPDATE_RHO="${GROUPWISE_RADIUS_UPDATE_RHO:-1.0}"
GROUPWISE_MAX_OUTER_ITERS="${GROUPWISE_MAX_OUTER_ITERS:-20}"
GROUPWISE_CHAIN_METHOD="${GROUPWISE_CHAIN_METHOD:-nearest_neighbor}"
GROUPWISE_METRIC_ENERGY_MODE="${GROUPWISE_METRIC_ENERGY_MODE:-scalar_proxy}"
GROUPWISE_MERGE_TOPK="${GROUPWISE_MERGE_TOPK:-10}"
GROUPWISE_MAX_REASSIGN_CANDIDATES_PER_GROUP="${GROUPWISE_MAX_REASSIGN_CANDIDATES_PER_GROUP:-5}"
GROUPWISE_REASSIGN_TARGET_TOPK="${GROUPWISE_REASSIGN_TARGET_TOPK:-2}"
GROUPWISE_MIN_GROUP_SIZE="${GROUPWISE_MIN_GROUP_SIZE:-2}"

MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
SWEEP_DIR="${SWEEP_DIR:-multirun_groupwise_compare}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
mkdir -p "$MPLCONFIGDIR"

cat <<EOF
[groupwise-compare]
profile=$PROFILE
python=$PYTHON_BIN
models=$MODELS
layers=$LAYERS
datasets=$DATASETS
ratios=$RATIOS
num_removal_candidates=$NUM_REMOVAL_CANDIDATES
hessian_type=$HESSIAN_TYPE
eval_metric=$EVAL_METRIC
lissa_iter=$LISSA_ITER
scale=$SCALE
hydra.sweeper.max_batch_size=$MAX_BATCH_SIZE
hydra.sweep.dir=$SWEEP_DIR
EOF

"$PYTHON_BIN" main.py -m \
  hydra.sweep.dir="$SWEEP_DIR" \
  hydra.sweeper.max_batch_size="$MAX_BATCH_SIZE" \
  experiment=large_drop_influence \
  model="$MODELS" \
  model.num_layers="$LAYERS" \
  dataset="$DATASETS" \
  seed="$SEED" \
  run.element_type="$ELEMENT_TYPE" \
  run.num_removal_candidates="$NUM_REMOVAL_CANDIDATES" \
  influence.hessian_type="$HESSIAN_TYPE" \
  influence.eval_metric="$EVAL_METRIC" \
  influence.lissa_iter="$LISSA_ITER" \
  influence.scale="$SCALE" \
  experiment.ratio_group_elem="$RATIOS" \
  experiment.metric_mode=groupwise \
  experiment.influence_mode=calculate_influence \
  experiment.groupwise_alpha_repr="$GROUPWISE_ALPHA_REPR" \
  experiment.groupwise_beta_proxy="$GROUPWISE_BETA_PROXY" \
  experiment.groupwise_probe_dim="$GROUPWISE_PROBE_DIM" \
  experiment.groupwise_probe_seed="$GROUPWISE_PROBE_SEED" \
  experiment.groupwise_normalize_proxies="$GROUPWISE_NORMALIZE_PROXIES" \
  experiment.groupwise_damping="$GROUPWISE_DAMPING" \
  experiment.groupwise_split_threshold="$GROUPWISE_SPLIT_THRESHOLD" \
  experiment.groupwise_merge_threshold="$GROUPWISE_MERGE_THRESHOLD" \
  experiment.groupwise_move_threshold="$GROUPWISE_MOVE_THRESHOLD" \
  experiment.groupwise_radius_update_rho="$GROUPWISE_RADIUS_UPDATE_RHO" \
  experiment.groupwise_max_outer_iters="$GROUPWISE_MAX_OUTER_ITERS" \
  experiment.groupwise_chain_method="$GROUPWISE_CHAIN_METHOD" \
  experiment.groupwise_metric_energy_mode="$GROUPWISE_METRIC_ENERGY_MODE" \
  experiment.groupwise_merge_topk="$GROUPWISE_MERGE_TOPK" \
  experiment.groupwise_max_reassign_candidates_per_group="$GROUPWISE_MAX_REASSIGN_CANDIDATES_PER_GROUP" \
  experiment.groupwise_reassign_target_topk="$GROUPWISE_REASSIGN_TARGET_TOPK" \
  experiment.groupwise_min_group_size="$GROUPWISE_MIN_GROUP_SIZE"
