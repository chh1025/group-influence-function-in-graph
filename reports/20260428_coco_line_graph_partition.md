# CoCo Line-Graph Partition For Edge-Removal Influence

## Scope

This implementation adds CoCo-based clustering as a partition backend for edge-removal influence experiments.

For edge edits, each graph edge is converted into one line-graph node. The line-graph node feature is the mean of the two endpoint node features:

```text
x_(u,v) = 0.5 * (x_u + x_v)
```

The main experiment target is `edge_removal`. Insertion fallback support exists in code, but the planned runs should use removal only.

## New Modes

Use:

```bash
--partition-methods coco
--partition-strategy coco_full_line_graph_assignment
--element-type edge_removal
```

`coco_full_line_graph_assignment` trains CoCo once on the full training-graph line graph, then assigns candidate removal edges by their learned line-graph cluster label.

`coco_candidate_line_graph` trains CoCo independently on each candidate edge set. This is mainly a debugging/ablation mode because it is more expensive.

## 4-GPU Scheduler

The existing `run_partition_influence_exp.py` scheduler is still the entry point. It launches one worker per free GPU and immediately dispatches the next run when a worker exits.

Use:

```bash
source ./conda.sh
PARTITION_METHODS=coco \
PARTITION_STRATEGY=coco_full_line_graph_assignment \
ELEMENT_TYPE=edge_removal \
GPU_IDS=0,1,2,3 \
MAX_WORKERS=4 \
./run_partition_compare.sh small
```

## NUM_CLUSTERS_LIST Sweep Syntax

`NUM_CLUSTERS_LIST` can mix fixed K values, total-edge ratios, and graph-derived rules.

Examples:

```bash
NUM_CLUSTERS_LIST=edge_cluster_size:16,edge_cluster_size:32,edge_cluster_size:64,sqrt_edges,avg_degree
```

Supported tokens:

- fixed integers: `8`, `16`
- full graph edge cluster-size target: `edge_cluster_size:32`
- candidate edge cluster-size target: `candidate_cluster_size:16`
- candidate edge fraction/percent: `candidate_ratio:0.1`, `candidate_pct:10`
- total undirected-edge fraction: `edge_ratio:0.01`
- total undirected-edge percent: `edge_pct:1`
- total node fraction/percent: `node_ratio:0.01`, `node_pct:1`
- graph-scale rules: `sqrt_edges`, `sqrt_nodes`, `log2_edges`, `log2_nodes`, `avg_degree`
- candidate-scale rules: `sqrt_candidate_edges`, `log2_candidate_edges`
- scaled graph rules: `sqrt_edges:0.5`, `avg_degree:2`

Resolved K is capped by the candidate edge count for that dataset/ratio and duplicate resolved K values are skipped within the same dataset/ratio.

## Recommended K Sweep Basis

Do not use `K=2,3` as the main CoCo sweep. Those values are useful only as sanity checks.

For `coco_full_line_graph_assignment`, the primary K axis should target the average full-line-graph cluster size:

```bash
NUM_CLUSTERS_LIST=edge_cluster_size:16,edge_cluster_size:32,edge_cluster_size:64,sqrt_edges,avg_degree
```

This asks whether CoCo should form fine, medium, or coarse edge communities before candidate removal groups are mapped to cluster labels.

For candidate-level ablations, use candidate-size-normalized K:

```bash
NUM_CLUSTERS_LIST=candidate_cluster_size:8,candidate_cluster_size:16,candidate_cluster_size:32,sqrt_candidate_edges
```

This keeps the expected cluster granularity comparable across ratios.

## Smoke-Tested Command

```bash
source ./conda.sh
python run_partition_influence_exp.py \
  --profile small \
  --partition-methods coco \
  --partition-strategy coco_full_line_graph_assignment \
  --element-type edge_removal \
  --models GCN \
  --layers 2 \
  --datasets texas \
  --ratios 1 \
  --num-clusters 2 \
  --num-removal-candidates 1 \
  --epochs 1000 \
  --pbrf-epochs 1 \
  --lissa-iter 100 \
  --scale 32 \
  --coco-epochs 1 \
  --coco-hidden-dim 8 \
  --coco-compact-k 4 \
  --coco-stage-num 2 \
  --coco-use-diffusion 0 \
  --gpu-ids 0 \
  --max-workers 1 \
  --max-runs 1
```

The smoke run completed successfully and wrote CoCo partition diagnostics into `candidate_results.csv`.

## Recommended Experiment Grid

Pilot:

- partition method: `coco`
- partition strategy: `coco_full_line_graph_assignment`
- element type: `edge_removal`
- datasets: `texas,cornell`
- models: `GCN,GAT`
- layers: `2`
- ratios: `30,50`
- K specs: `edge_cluster_size:16,edge_cluster_size:32,edge_cluster_size:64,sqrt_edges,avg_degree`
- candidates: `20`

Main comparison:

- strategies:
  - `coco_full_line_graph_assignment`
  - `candidate_local_affinity`
  - `global_training_graph_assignment`
- datasets: `cora_public,citeseer_public,texas,cornell`
- models: `GCN,GAT`
- layers: `2,4`
- ratios: `30,50,80`
- K specs: `edge_cluster_size:16,edge_cluster_size:32,edge_cluster_size:64,sqrt_edges,avg_degree`
- candidates: `50`

Primary metrics:

1. Spearman
2. sign accuracy
3. MAE
4. runtime

## Notes

- Full line-graph CoCo labels are cached under `results/coco_line_graph_cache` unless `COCO_CACHE_DIR=none`.
- Default `COCO_MAX_LINE_GRAPH_NODES=6000` protects against dense line-graph/PPR memory blowups.
- For larger datasets, start with `COCO_USE_DIFFUSION=0` or raise the node limit only after estimating line-graph size.
