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

## Small 20% Drop Partition Baseline Results

Checked on 2026-05-06 12:05 KST.

Output:

```text
results/partition_compare_summary_small_drop20_20260505_015921.csv
```

Completion:

- total runs: `48`
- status `ok`: `48`
- failed: `0`

Overall interpretation:

- The run completed successfully.
- This default `candidate_local_affinity` partition baseline does not show a clear aggregate improvement over the one-shot baseline.
- The baseline and clusterwise estimates are very close in many runs, and clusterwise is often slightly worse by MAE.
- There is one unstable outlier configuration: `cornell / GAT / layer 4`, with MAE around `41531` for all three partition methods. Result dirs include `nan_nan_0.61_*`, so this should be treated as a PBRF/model instability case before drawing scientific conclusions.

Aggregate excluding the outlier rows (`baseline_pbrf_mae < 100` and `cluster_pbrf_mae < 100`):

```text
method      n   wins  baseline_mae  cluster_mae  mean_delta(cluster-baseline)
metis      15   5     0.099814      0.100254     +0.000439
spectral   15   7     0.099814      0.100196     +0.000382
local_ppr  15   3     0.099814      0.102639     +0.002825
```

Notes:

- `spectral` has the most MAE wins (`7/15`) after excluding the outlier, but the mean MAE is still slightly worse than baseline.
- `metis` is closest to baseline on mean MAE among the three methods.
- `local_ppr` is the weakest of the three by mean MAE in this setting.
- Baseline sign accuracy and cluster sign accuracy are almost unchanged overall.

Best meaningful improvements:

- `texas / GAT / layer 4`: clusterwise improves MAE by about `0.02-0.024` depending on method.
- `cornell / GAT / layer 2`: `metis` improves MAE by about `0.0085`.

Worst meaningful regressions:

- `texas / GAT / layer 2`: clusterwise is worse, especially `local_ppr` and `spectral`.
- `texas / GCN / layer 2`: `metis` and `local_ppr` regress noticeably.

Next recommendation:

Use these results as a completed sanity baseline, not as evidence that grouping helps. Move to the planned Phase 1/3 influence-feature candidate construction and cheap influence-feature clustering, because this structural partitioning baseline is too weak to validate the influence-geometry hypothesis.

## Phase 1 Candidate Edge Construction

Implemented after the structural partition baseline.

Added:

- `src/group_influence/candidates.py`
- `experiments/group_influence/build_candidate_edges.py`

Supported candidate types:

- `random`
- `top_abs`
- `mixed`
- `top_positive_negative`

Saved outputs per run:

```text
metadata.json
edge_pool.pt
candidate_edges.pt
candidate_edge_scores.csv
pool_edge_scores.csv  # for scored pool methods
```

Smoke checks:

```bash
python experiments/group_influence/build_candidate_edges.py \
  --dataset cora_public \
  --model GCN \
  --num-layers 2 \
  --seed 0 \
  --candidate-type random \
  --num-candidates 4 \
  --pool-size 4 \
  --epochs 1000 \
  --lissa-iter 100 \
  --scale 32 \
  --run-id smoke_random_4
```

```bash
python experiments/group_influence/build_candidate_edges.py \
  --dataset cora_public \
  --model GCN \
  --num-layers 2 \
  --seed 0 \
  --candidate-type top_abs \
  --num-candidates 4 \
  --pool-size 8 \
  --epochs 1000 \
  --lissa-iter 100 \
  --scale 32 \
  --run-id smoke_top_abs_4_pool8
```

```bash
python experiments/group_influence/build_candidate_edges.py \
  --dataset cora_public \
  --model GCN \
  --num-layers 2 \
  --seed 0 \
  --candidate-type mixed \
  --num-candidates 4 \
  --pool-size 12 \
  --mixed-positive-count 2 \
  --mixed-negative-count 2 \
  --epochs 1000 \
  --lissa-iter 100 \
  --scale 32 \
  --run-id smoke_mixed_2_2_pool12
```

Also updated `run_baselines.py` to load a saved candidate set:

```bash
python experiments/group_influence/run_baselines.py \
  --dataset cora_public \
  --model GCN \
  --num-layers 2 \
  --seed 0 \
  --candidate-edges-path results/group_influence/candidate_edges/smoke_top_abs_4_pool8/candidate_edges.pt \
  --candidate-type top_abs \
  --num-edges 4 \
  --epochs 1000 \
  --lissa-iter 100 \
  --pbrf-epochs 1 \
  --scale 32 \
  --skip-pbrf \
  --run-id smoke_baseline_loaded_top_abs_4
```

Key smoke result:

```text
candidate_set=top_abs_4 candidate_edges=4 pool_edges=8
loaded baseline H(S)=-0.00019106602
loaded baseline E(S)=-0.00019023675
```

Next step:

Build Phase 3 cheap influence-feature clustering on top of `candidate_edge_scores.csv`, starting with `cheap_kmeans` and `random` clustering baselines.

## Phase 3/6 Cheap Clustering and Independent Aggregation

Implementation in progress after Phase 1 was committed as:

```text
5adf25c [Exp] Add group influence candidate builder
```

Added reusable modules:

- `src/group_influence/features.py`
- `src/group_influence/clustering.py`
- `src/group_influence/aggregation.py`

Added scripts:

- `experiments/group_influence/run_feature_clustering.py`
- `experiments/group_influence/run_aggregation.py`

Cheap feature vector:

```text
single_edge_influence
abs_single_edge_influence
single_edge_parameter_shift
single_edge_message_passing
degree_u
degree_v
endpoint_logit_cosine
endpoint_logit_l2
```

Clustering outputs:

```text
metadata.json
clustering_result.json
cluster_assignments.csv
cluster_summary.csv
candidate_edges.pt
cheap_features.pt
normalized_features.pt
cluster_labels.pt
cluster_centroids.pt
```

Aggregation outputs:

```text
metadata.json
aggregation_result.json
aggregation_result.csv
independent_cluster_rows.csv
candidate_edges.pt
cluster_labels.pt
```

Planned smoke command:

```bash
python experiments/group_influence/run_feature_clustering.py \
  --candidate-dir results/group_influence/candidate_edges/smoke_top_abs_4_pool8 \
  --clustering-method cheap_kmeans \
  --num-clusters 2 \
  --run-id smoke_top_abs_4_kmeans2
```

Then:

```bash
python experiments/group_influence/run_aggregation.py \
  --candidate-dir results/group_influence/candidate_edges/smoke_top_abs_4_pool8 \
  --clustering-dir results/group_influence/feature_clustering/smoke_top_abs_4_kmeans2 \
  --skip-pbrf \
  --run-id smoke_top_abs_4_kmeans2_independent
```

Smoke results:

```text
results/group_influence/feature_clustering/smoke_top_abs_4_kmeans2
method=cheap_kmeans K=2 total=-0.00019023675 cancel=0.00026383734 sign_purity=0.7500 within_var=1.7815732
```

```text
results/group_influence/feature_clustering/smoke_top_abs_4_random2
method=random K=2 total=-0.00019023675 cancel=0.00036294568 sign_purity=1.0000 within_var=4.3881617
```

```text
results/group_influence/baselines/smoke_baseline_loaded_top_abs_4_infer_count
A(S)=skipped H(S)=-0.00019106554 E(S)=-0.00019023692
```

```text
results/group_influence/aggregation/smoke_top_abs_4_kmeans2_independent_infer_count
A=skipped H=-0.00019106566 E=-0.0001902365 C_ind=-0.00019023687
```

Actual PBRF smoke with `pbrf_epochs=1`:

```text
results/group_influence/aggregation/smoke_top_abs_4_kmeans2_independent_pbrf1
A=-0.0002951622 H=-0.00019106544 E=-0.00019023652 C_ind=-0.00019023681
abs_error_H=0.00010409676
abs_error_E=0.00010492568
abs_error_C_ind=0.00010492539
```

Implementation note:

`run_baselines.py` and `run_aggregation.py` now infer `num_group_elem` from `candidate_edges.pt` when a saved candidate set is loaded. This keeps PBRF checkpoint/result directories under the correct `Nedges` path.
