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

## MVP Small Grid Tmux Run

Launched on 2026-05-06 13:08 KST.

Launcher:

```text
run_group_influence_mvp_small.sh
```

Tmux session:

```text
group_inf_mvp_small_20260506_130854
```

Run root:

```text
results/group_influence_mvp_small/20260506_130854
```

Grid:

```text
datasets: cora_public,citeseer_public,texas,cornell
model: GCN
layers: 2
candidate_types: random,top_abs,mixed
num_candidates: 100
top_abs/mixed pool_size: 200
cluster_methods: cheap_kmeans,random
num_clusters: 5
pbrf_epochs: 1
gpus: 0,1,2,3
```

Planned work:

```text
candidate jobs: 12
aggregation runs: 24
```

Initial health check:

```text
candidate dirs: 8 / 12
clustering dirs: 12 / 24
aggregation dirs: 11 / 24
tmux workers alive: 4 / 4
errors in worker logs: none observed
```

Resume note:

The first launch exposed a scalar conversion bug in `baseline_api._scalar` when a cluster returned an integer zero tensor. Fixed by casting non-floating tensors to float before averaging.

Also updated the launcher to resume by completion files instead of directory existence:

```text
candidate_edges.pt
cluster_labels.pt
aggregation_result.csv
```

Relaunched with the same run stamp/session after the fix:

```text
session: group_inf_output_long_20260506_142428
candidate dirs complete: 5 / 84
clustering dirs complete: 17 / 336
aggregation dirs complete: 13 / 336
tmux workers alive: 4 / 4
errors in fresh worker logs: none observed
```

Useful commands:

```bash
tmux attach -t group_inf_mvp_small_20260506_130854
tail -f results/group_influence_mvp_small/20260506_130854/logs/worker_0.log
```

Completion check:

```text
candidate dirs: 12 / 12
clustering dirs: 24 / 24
aggregation dirs: 24 / 24
done workers: 4 / 4
tmux session: finished
errors in worker logs: none observed
```

Aggregate over 24 aggregation runs:

```text
method        n   mean_err_H   mean_err_E   mean_err_C_ind   wins(H/E/C_ind)   sign(H/E/C_ind)
cheap_kmeans 12  0.0454866    0.313787     0.0604775        8/3/1             6/6/7
random       12  0.0454866    0.313787     0.0796885        8/2/2             6/6/4
```

By dataset:

```text
dataset          n   mean_err_H   mean_err_E   mean_err_C_ind   wins(H/E/C_ind)
cora_public      6   0.000904944  0.000600256  0.000637571      0/3/3
citeseer_public  6   0.00130138   0.00173252   0.00154241       6/0/0
cornell          6   0.100392     0.104365     0.11444         4/2/0
texas            6   0.0793476    1.14845      0.163712        6/0/0
```

Interpretation:

- `cheap_kmeans` improves average `C_ind` error versus random clustering in this small grid.
- `C_ind` is not yet competitive with the best of `H` and `E` overall.
- The promising cases are mostly `cora_public`, where `C_ind` wins 3 of 6 runs.
- `texas` shows large regressions for cluster aggregation, especially on `top_abs` and `mixed`.

## Phase 4 Output-Space Clustering Setup

Next experiment target:

```text
feature_type: logits_delta
distance/clustering: PCA-reduced Euclidean k-means
aggregation: independent_cluster_sum
```

Implementation:

- `src/group_influence/features.py` now supports `build_output_delta_features`.
- `experiments/group_influence/run_feature_clustering.py` supports `--feature-type logits_delta`.
- `run_group_influence_mvp_small.sh` now supports `FEATURE_TYPES=cheap,logits_delta`.

Feature construction:

```text
For each candidate edge e:
  1. apply the single edge edit to the graph with fixed model parameters
  2. compute logits_delta = logits(G_e) - logits(G)
  3. flatten over selected nodes/classes
  4. reduce with PCA to --output-pca-dim, default 32
```

Smoke:

```text
results/group_influence/feature_clustering/smoke_top_abs_4_logits_delta_kmeans2
feature_type=logits_delta
raw_feature_dim=18956
explained_variance_ratio_sum=0.8414692878723145
```

Actual PBRF smoke:

```text
results/group_influence/aggregation/smoke_top_abs_4_logits_delta_kmeans2_independent_pbrf1
A=-0.0002951622 H=-0.0001910656 E=-0.00019023685 C_ind=-0.00019023717
abs_error_H=0.00010409660
abs_error_C_ind=0.00010492503
```

## Output-Space Long Grid Tmux Run

Launched on 2026-05-06 14:24 KST.

Tmux session:

```text
group_inf_output_long_20260506_142428
```

Run root:

```text
results/group_influence_output_long/20260506_142428
```

Grid:

```text
datasets: cora_public,citeseer_public,pubmed_public,texas,cornell,chameleon,squirrel
models: GCN,GAT
layers: 2,4
candidate_types: random,top_abs,mixed
feature_types: cheap,logits_delta
clustering_methods: cheap_kmeans,random
num_candidates: 200
pool_size: 500
num_clusters: 5
output_node_scope: eval
output_pca_dim: 32
lissa_iter: 300
pbrf_epochs: 5
gpus: 0,1,2,3
```

Planned work:

```text
candidate jobs: 84
aggregation runs: 336
```

Initial health check:

```text
candidate dirs: 4 / 84
clustering dirs: 7 / 336
aggregation dirs: 4 / 336
tmux workers alive: 4 / 4
errors in worker logs: none observed
```

Useful commands:

```bash
tmux attach -t group_inf_output_long_20260506_142428
tail -f results/group_influence_output_long/20260506_142428/logs/worker_0.log
```

Completion check:

```text
candidate dirs: 84 / 84
clustering dirs: 336 / 336
aggregation dirs: 336 / 336
done workers: 4 / 4
tmux session: finished
errors in final worker logs: none observed after resume
```

Machine-readable aggregate:

```text
results/group_influence_output_long/20260506_142428/aggregation_long_results.csv
results/group_influence_output_long/20260506_142428/summary_by_feature_clustering.csv
results/group_influence_output_long/20260506_142428/summary_by_dataset.csv
```

Overall aggregation result:

```text
n=336
mean_err_H=0.076594
mean_err_E=0.274103
mean_err_C_ind=0.090721
median_err_H=0.004046
median_err_E=0.002338
median_err_C_ind=0.003927
wins(H/E/C_ind)=114/154/68
sign_match(H/E/C_ind)=252/272/236
```

By feature/clustering:

```text
feature        clustering      n   mean_err_C_ind  median_err_C_ind  wins(H/E/C_ind)
cheap          cheap_kmeans    84  0.094739        0.004035          28/38/18
cheap          random          84  0.092266        0.003367          30/38/16
logits_delta   cheap_kmeans    84  0.083614        0.004347          26/41/17
logits_delta   random          84  0.092266        0.003367          30/37/17
```

Interpretation:

- `logits_delta + cheap_kmeans` has the best mean `C_ind` error, mostly due to improvements on large-error WebKB cases.
- Median error still favors the random partition baseline, so the output-space clustering is not uniformly better.
- `C_ind` is still not generally competitive with `H`/`E`: it wins only 68 of 336 runs.
- `E` has the most wins overall, mostly on low-error citation/heterophily datasets.
- The most promising regime is `GAT layer 4` with `logits_delta + cheap_kmeans`: mean `C_ind` beats both `H` and `E` in that slice.

Dataset notes:

```text
cora_public: C_ind is competitive; mean C is best overall.
texas: logits_delta + cheap_kmeans improves over H/E on average, but other clusterings regress.
cornell: random clustering has surprisingly strong C_ind cases; logits_delta kmeans is unstable.
chameleon/pubmed/squirrel: errors are very small; E is usually strongest.
citeseer_public: H/E dominate; C_ind rarely wins.
```

## 2026-05-06 Phase 6: sequential cluster aggregation

Motivation:

```text
Independent cluster aggregation C_ind estimates every cluster on the original graph.
The next check is C_seq: estimate a cluster, apply that cluster's edge edits to the
working graph, then estimate the next cluster on the updated graph.
```

Implementation:

```text
src/group_influence/aggregation.py
  - added compute_cluster_sequential_graph_only
  - supports cluster_id, random, cluster_size_asc,
    small_abs_cluster_influence_first, large_abs_cluster_influence_first

experiments/group_influence/run_sequential_aggregation.py
  - reuses existing candidate/clustering artifacts
  - reuses baseline aggregation_result.json for A/H/E/C_ind and PBRF
  - writes sequential_result.csv/json and per-step CSVs

run_group_influence_sequential_existing.sh
  - launches C_seq over existing aggregation runs in tmux
  - round-robin workers over GPU_IDS
  - default source stamp: 20260506_142428
  - default order policy: small_abs_cluster_influence_first
```

Validation:

```text
bash -n run_group_influence_sequential_existing.sh
python -m py_compile src/group_influence/__init__.py \
  src/group_influence/aggregation.py \
  experiments/group_influence/run_sequential_aggregation.py
git diff --check
```

Smoke run:

```text
results/group_influence/sequential_aggregation/smoke_top_abs_4_logits_delta_kmeans2_seq_pbrf1

A=-0.0002951622
H=-0.0001910656
E=-0.0001902369
C_ind=-0.0001902372
C_seq_cluster_id=-0.0001903239
C_seq_small_abs_cluster_influence_first=-0.0001903516
C_seq_large_abs_cluster_influence_first=-0.0001903236

best_seq=small_abs_cluster_influence_first
abs_error_C_ind=0.0001049250
abs_error_C_seq_small_abs_cluster_influence_first=0.0001048106
```

Planned long run:

```bash
ORDER_POLICIES=small_abs_cluster_influence_first \
GPU_IDS=0,1,2,3 \
bash run_group_influence_sequential_existing.sh launch
```

Status command:

```bash
RUN_STAMP=<stamp> SESSION_NAME=group_inf_seq_long_<stamp> \
bash run_group_influence_sequential_existing.sh status
```

Launched long run:

```text
session=group_inf_seq_long_20260506_180948
run_root=results/group_influence_sequential_existing/20260506_180948
source_stamp=20260506_142428
total_sequential_runs=336
order_policies=small_abs_cluster_influence_first
gpu_ids=0,1,2,3
initial status: done=2/336, worker_done=0/4, errors=none-observed
```
