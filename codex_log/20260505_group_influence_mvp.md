# 2026-05-05 Group Influence MVP

## Context

- Started from branch `experiments/coco-clustering`.
- Created branch `experiments/group-influence-mvp`.
- Goal: implement only enough of Phase 0 to run one small baseline experiment.

## Implementation Plan

1. Add a reusable baseline wrapper around existing training, influence, and PBRF APIs.
2. Add a cache helper for config-hashed result directories.
3. Add a first executable script for `A(S)`, `H(S)`, and `E(S)` on one small random edge set.
4. Verify with a tiny Cora/GCN run before adding clustering.

## Implemented

- Added `src/group_influence/baseline_api.py`.
  - Reuses existing `DataLoader`, `GNN`, vanilla checkpoint loading/training, `GraphInfluenceModule`, and `calculate_pbrf`.
  - Exposes wrappers for:
    - `compute_actual_pbrf`
    - `compute_heo_oneshot`
    - `compute_single_edge_sum`
- Added `src/group_influence/cache.py`.
  - Config hash helper.
  - JSON, CSV, and tensor cache writers.
- Added `experiments/group_influence/run_baselines.py`.
  - Builds one random edge set.
  - Computes `A(S)`, `H(S)`, and `E(S)`.
  - Saves:
    - `candidate_edges.pt`
    - `metadata.json`
    - `baseline_result.json`
    - `baseline_result.csv`
    - `single_edge_influences.csv`

## Verification

Initial overly small smoke failed:

```bash
python experiments/group_influence/run_baselines.py \
  --dataset cora_public \
  --model GCN \
  --num-layers 2 \
  --seed 0 \
  --num-edges 1 \
  --epochs 1 \
  --lissa-iter 1 \
  --pbrf-epochs 1 \
  --scale 32 \
  --skip-pbrf \
  --run-id smoke_skip_pbrf
```

Reason: LiSSA returned NaN with `lissa_iter=1`. This is not a useful smoke setting.

Fast wrapper smoke passed with PBRF skipped:

```bash
python experiments/group_influence/run_baselines.py \
  --dataset cora_public \
  --model GCN \
  --num-layers 2 \
  --seed 0 \
  --num-edges 1 \
  --epochs 1000 \
  --lissa-iter 100 \
  --pbrf-epochs 1 \
  --scale 32 \
  --skip-pbrf \
  --run-id smoke_skip_pbrf_lissa100_fix
```

First actual group baseline passed:

```bash
python experiments/group_influence/run_baselines.py \
  --dataset cora_public \
  --model GCN \
  --num-layers 2 \
  --seed 0 \
  --num-edges 2 \
  --epochs 1000 \
  --lissa-iter 100 \
  --pbrf-epochs 1 \
  --scale 32 \
  --run-id first_group2_pbrf_lissa100
```

Output directory:

```text
results/group_influence/baselines/first_group2_pbrf_lissa100
```

Key result:

```text
A(S) = -0.00047791004180908203
H(S) = -0.00037112965947017074
E(S) = -0.00037114480073796585
|A(S)-H(S)| = 0.0001067803823389113
|A(S)-E(S)| = 0.00010676524107111618
sign_match_H = True
sign_match_E = True
```

Notes:

- With only two random edges, `H(S)` and `E(S)` are still very close.
- This run is a pipeline smoke, not a scientific result.
- Next step should add Phase 1 candidate construction so `top_abs` and mixed positive/negative candidate sets can reuse single-edge caches.

## Launched Small 20% Drop Partition Baseline

Launched the existing partition comparison scheduler in tmux.

Session:

```text
partition_small_drop20
```

Command:

```bash
source ./conda.sh
export TS=20260505_015921
export RATIOS=20
export GPU_IDS=0,1,2,3
export MAX_WORKERS=4
export POLL_INTERVAL_SEC=1.0
export SUMMARY_CSV=results/partition_compare_summary_small_drop20_20260505_015921.csv
export JOB_DIR=results/partition_compare_jobs/small_drop20_20260505_015921
bash run_partition_compare.sh small
```

Run settings:

- profile: `small`
- drop ratio: `20%` of all edges via `large_drop_influence`
- element type: `edge_removal`
- partition methods: `metis,spectral,local_ppr`
- partition strategy: `candidate_local_affinity`
- num clusters: `3`
- num removal candidates: `50`
- GPUs: `0,1,2,3`
- max workers: `4`

Initial worker checks:

- `run_0001`: `metis`, `GCN`, layer `2`, `cora_public`, ratio `20`
- `run_0002`: `metis`, `GCN`, layer `2`, `citeseer_public`, ratio `20`
- `run_0003`: `metis`, `GCN`, layer `2`, `texas`, ratio `20`
- `run_0004`: `metis`, `GCN`, layer `2`, `cornell`, ratio `20`

The scheduler launched four workers concurrently and will continue assigning new runs to free GPUs from the planned small-profile run queue.
