#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROFILE="${1:-small}"
RUN_NAME="${RUN_NAME:-metis_k_sweep_high_ratio_gcn_gat4}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export PARTITION_METHODS="${PARTITION_METHODS:-metis}"
export NUM_CLUSTERS_LIST="${NUM_CLUSTERS_LIST:-1,2,3,4,6,8}"
export RATIOS="${RATIOS:-30,50,80}"
export MODELS="${MODELS:-GCN,GAT}"
export LAYERS="${LAYERS:-4}"
export DATASETS="${DATASETS:-cora_public,citeseer_public,texas,cornell}"
export NUM_REMOVAL_CANDIDATES="${NUM_REMOVAL_CANDIDATES:-50}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export SUMMARY_CSV="${SUMMARY_CSV:-results/partition_compare_summary_${RUN_NAME}_${TIMESTAMP}.csv}"
export JOB_DIR="${JOB_DIR:-results/partition_compare_jobs/${RUN_NAME}_${TIMESTAMP}}"

cat <<EOF
[partition-compare-metis-k-sweep]
run_name=$RUN_NAME
timestamp=$TIMESTAMP
profile=$PROFILE
partition_methods=$PARTITION_METHODS
num_clusters_list=$NUM_CLUSTERS_LIST
ratios=$RATIOS
models=$MODELS
layers=$LAYERS
datasets=$DATASETS
summary_csv=$SUMMARY_CSV
job_dir=$JOB_DIR
EOF

exec ./run_partition_compare.sh "$PROFILE"
