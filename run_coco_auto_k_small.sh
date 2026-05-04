#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ./conda.sh

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

export PARTITION_METHODS="${PARTITION_METHODS:-coco}"
export PARTITION_STRATEGY="${PARTITION_STRATEGY:-coco_full_line_graph_assignment}"
export ELEMENT_TYPE="${ELEMENT_TYPE:-edge_removal}"

export AUTO_K_METHOD="${AUTO_K_METHOD:-eigengap_silhouette_hybrid}"
export AUTO_K_MIN="${AUTO_K_MIN:-1}"
export AUTO_K_MAX="${AUTO_K_MAX:-8}"
export AUTO_K_MAX_RATIO="${AUTO_K_MAX_RATIO:-1.0}"
export AUTO_K_MIN_CLUSTER_SIZE="${AUTO_K_MIN_CLUSTER_SIZE:-1}"
export AUTO_K_MAX_CLUSTER_SIZE="${AUTO_K_MAX_CLUSTER_SIZE:-0}"
export AUTO_K_TINY_GRAPH_THRESHOLD="${AUTO_K_TINY_GRAPH_THRESHOLD:-4}"
export AUTO_K_NUM_RESTARTS="${AUTO_K_NUM_RESTARTS:-10}"
export AUTO_K_RANDOM_SEED="${AUTO_K_RANDOM_SEED:-0}"
export AUTO_K_SILHOUETTE_METRIC="${AUTO_K_SILHOUETTE_METRIC:-euclidean}"
export AUTO_K_STABILITY_TRIALS="${AUTO_K_STABILITY_TRIALS:-10}"
export AUTO_K_STABILITY_EDGE_DROPOUT="${AUTO_K_STABILITY_EDGE_DROPOUT:-0.05}"
export AUTO_K_WEAK_SILHOUETTE_THRESHOLD="${AUTO_K_WEAK_SILHOUETTE_THRESHOLD:-0.05}"
export AUTO_K_PREFER_SMALLER_K="${AUTO_K_PREFER_SMALLER_K:-1}"

export NUM_CLUSTER_LIST="${NUM_CLUSTERS:-edge_cluster_size:8,edge_cluster_size:16,edge_cluster_size:32,edge_cluster_size:64,edge_cluster_size:128,edge_cluster_size:256,edge_cluster_size:512}"
export RATIOS="${RATIOS:-30,50,80}"
export NUM_REMOVAL_CANDIDATES="${NUM_REMOVAL_CANDIDATES:-20}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export MAX_WORKERS="${MAX_WORKERS:-4}"

export COCO_USE_DIFFUSION="${COCO_USE_DIFFUSION:-0}"
export COCO_EPOCHS="${COCO_EPOCHS:-200}"
export COCO_CACHE_DIR="${COCO_CACHE_DIR:-results/coco_line_graph_cache}"
export COCO_FORCE_RETRAIN="${COCO_FORCE_RETRAIN:-0}"
export COCO_MAX_LINE_GRAPH_NODES="${COCO_MAX_LINE_GRAPH_NODES:-6000}"

export SUMMARY_CSV="${SUMMARY_CSV:-results/partition_compare_summary_coco_auto_k_small_${TS}.csv}"
export JOB_DIR="${JOB_DIR:-results/partition_compare_jobs/coco_auto_k_small_${TS}}"

exec bash run_partition_compare.sh small
