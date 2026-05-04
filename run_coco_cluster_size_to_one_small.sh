#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ./conda.sh

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

# Continue the previous manual edge_cluster_size sweep beyond 512.
# Previous sweep reached:
# - cora_public: edge_cluster_size:512 -> K=11
# - citeseer_public: edge_cluster_size:512 -> K=9
# - cornell/texas: edge_cluster_size:512 -> K=1 already
#
# This continuation therefore runs only cora_public and citeseer_public.
# edge_cluster_size:8192 is larger than both full edge counts
# (cora=5278, citeseer=4552), so it reaches K=1.

export PARTITION_METHODS="${PARTITION_METHODS:-coco}"
export PARTITION_STRATEGY="${PARTITION_STRATEGY:-coco_full_line_graph_assignment}"
export ELEMENT_TYPE="${ELEMENT_TYPE:-edge_removal}"
export AUTO_K_METHOD="${AUTO_K_METHOD:-none}"

export NUM_CLUSTERS_LIST="${NUM_CLUSTERS_LIST:-edge_cluster_size:1024,edge_cluster_size:2048,edge_cluster_size:4096,edge_cluster_size:8192}"
export RATIOS="${RATIOS:-1,10,30,50,80}"
export DATASETS="${DATASETS:-cora_public,citeseer_public}"
export MODELS="${MODELS:-GCN,GAT}"
export LAYERS="${LAYERS:-2,4}"
export NUM_REMOVAL_CANDIDATES="${NUM_REMOVAL_CANDIDATES:-100}"
export SEED="${SEED:-0}"

export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export MAX_WORKERS="${MAX_WORKERS:-4}"
export POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-1.0}"

export HESSIAN_TYPE="${HESSIAN_TYPE:-GNH}"
export EVAL_METRIC="${EVAL_METRIC:-mean_validation_loss}"
export LR="${LR:-0.01}"
export HIDDEN_DIM="${HIDDEN_DIM:-16}"
export EPOCHS="${EPOCHS:-1000}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
export PBRF_EPOCHS="${PBRF_EPOCHS:-1000}"
export LISSA_ITER="${LISSA_ITER:-1000}"
export SCALE="${SCALE:-32}"

export COCO_USE_DIFFUSION="${COCO_USE_DIFFUSION:-1}"
export COCO_EPOCHS="${COCO_EPOCHS:-200}"
export COCO_LR="${COCO_LR:-0.001}"
export COCO_HIDDEN_DIM="${COCO_HIDDEN_DIM:-256}"
export COCO_ACTIVATION="${COCO_ACTIVATION:-ident}"
export COCO_COMPACT_K="${COCO_COMPACT_K:-64}"
export COCO_STAGE_NUM="${COCO_STAGE_NUM:-10}"
export COCO_BETA="${COCO_BETA:-1.0}"
export COCO_FILTER_STEPS="${COCO_FILTER_STEPS:-2}"
export COCO_ALPHA="${COCO_ALPHA:-0.2}"
export COCO_CONSISTENCY_T="${COCO_CONSISTENCY_T:-0.02}"
export COCO_MEMORY_SIZE="${COCO_MEMORY_SIZE:-0}"
export COCO_MEMORY_MULTIPLIER="${COCO_MEMORY_MULTIPLIER:-10}"
export COCO_PCA_DIM="${COCO_PCA_DIM:--1}"
export COCO_KMEANS_DISTANCE="${COCO_KMEANS_DISTANCE:-euclidean}"
export COCO_DEVICE="${COCO_DEVICE:-auto}"
export COCO_CACHE_DIR="${COCO_CACHE_DIR:-results/coco_line_graph_cache}"
export COCO_FORCE_RETRAIN="${COCO_FORCE_RETRAIN:-0}"
export COCO_MAX_LINE_GRAPH_NODES="${COCO_MAX_LINE_GRAPH_NODES:-6000}"

export SUMMARY_CSV="${SUMMARY_CSV:-results/partition_compare_summary_coco_cluster_size_to_one_${TS}.csv}"
export JOB_DIR="${JOB_DIR:-results/partition_compare_jobs/coco_cluster_size_to_one_${TS}}"

exec bash run_partition_compare.sh small
