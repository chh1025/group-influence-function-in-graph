# Experiment Plan: Influence-Geometry Grouping for Group Edge Influence

## 0. Goal

We want to test whether grouping edge edits by **influence geometry** is useful for group influence estimation and interpretation in GNNs.

The current concern is that full parameter-space Bregman clustering may be too expensive. Therefore, implement this experiment in staged MVPs.

The implementation should proceed in small, testable steps:

1. Build a reusable baseline harness.
2. Implement cheap influence-feature clustering.
3. Implement output-space Bregman / Mahalanobis clustering.
4. Implement approximate parameter-space Bregman clustering using projection.
5. Implement aggregation methods.
6. Add evaluation and reporting.

Do **not** implement all methods in one monolithic script.
Each phase should produce cached outputs that later phases can reuse.

---

## 1. High-level Definitions

For an edge edit set \(S\), we compare:

### Actual PBRF response

\[
A(S)
\]

This is the actual response obtained by applying the multi-edge edit and evaluating the PBRF-based retrained or fine-tuned response.

### Heo one-shot prediction

\[
H(S)
\]

This is the one-shot multi-edge influence prediction using the existing Heo-style implementation.

### Single-edge sum

\[
E(S)=\sum_{e\in S} H(\{e\})
\]

This is the sum of single-edge influence predictions.

### Independent cluster sum

Given a partition

\[
S = C_1 \sqcup C_2 \sqcup \cdots \sqcup C_K,
\]

define

\[
C_{\mathrm{ind}}(S)
=
\sum_{h=1}^{K} H(C_h).
\]

### Cluster-sequential prediction

For an order \(\pi\),

\[
C_{\mathrm{seq}}^\pi(S)
=
\sum_{h=1}^{K}
H_{z_{h-1}}\left(C_{\pi(h)}\right),
\]

where \(z_{h-1}\) is the current graph/model state before applying cluster \(C_{\pi(h)}\).

At first, implement a cheap version where the graph state is updated sequentially but model parameters are fixed. Parameter-updating sequential prediction can be added later.

---

## 2. Implementation Rules

### Avoid expensive full-matrix computation

Do **not** explicitly construct the full matrix

\[
G = J^\top HJ+\lambda I.
\]

If needed, implement it only as a matrix-vector product function:

```python
def Gv(v):
    # returns (J^T H J + lambda I) v
    ...
```

### Avoid per-edge inverse-HVP

Do **not** compute

\[
G^{-1}\delta g_e
\]

for every edge at the beginning. This is too expensive.

Instead, use one of the following approximations:

1. cheap influence feature vectors,
2. output-space perturbation vectors,
3. probe-projected parameter-space features.

### Cache everything

Each phase should save intermediate results to disk.

Required cache items:

```text
edge_id
edge endpoints
single_edge_influence
message_term if available
output_delta feature if available
projected_parameter_feature if available
cluster_id
cluster influence
A(S), H(S), E(S)
C_ind(S), C_seq(S)
metadata: dataset, model, seed, metric, candidate selection method
```

Use `.pt`, `.pkl`, `.npz`, or `.csv` depending on the existing project style.

---

# Phase 0. Inspect Existing Code and Add Experiment Harness

## Goal

Before adding clustering, identify the existing functions/scripts for:

1. computing actual PBRF response,
2. computing Heo one-shot multi-edge influence,
3. computing single-edge influence,
4. applying edge deletions or insertions,
5. loading datasets and trained models.

## Tasks

- Inspect the repository structure.
- Identify existing APIs for influence computation.
- Add wrapper functions if needed.

Implement or expose the following interface:

```python
def compute_actual_pbrf(edge_set, model, data, config):
    """
    Return actual PBRF response A(S) for a given edge set S.
    """
    pass


def compute_heo_oneshot(edge_set, model, data, config):
    """
    Return Heo-style one-shot multi-edge prediction H(S).
    """
    pass


def compute_single_edge_influence(edge, model, data, config):
    """
    Return single-edge influence H({e}).
    """
    pass


def compute_single_edge_sum(edge_set, model, data, config):
    """
    Return E(S) = sum_e H({e}).
    """
    return sum(compute_single_edge_influence(e, model, data, config) for e in edge_set)
```

If exact names differ in the current repo, use existing naming conventions, but keep these conceptual interfaces.

## Acceptance Criteria

- We can run a small example with one dataset, one model, and a small edge set.
- The script reports:

```text
A(S)
H(S)
E(S)
|A(S)-H(S)|
|A(S)-E(S)|
```

- Results are cached with a config hash or unique run ID.

---

# Phase 1. Candidate Edge Set Construction

## Goal

Create manageable candidate edge sets for group influence experiments.

Do not run on all edges initially.

## Candidate set types

Implement at least:

### 1. Random candidates

```text
random_100
random_200
```

### 2. High absolute influence candidates

Select top-\(m\) edges by

\[
|H(\{e\})|.
\]

Example:

```text
top_abs_100
```

### 3. Harmful and beneficial mixed candidates

Select:

```text
top_positive_50 + top_negative_50
```

This is important for cancellation analysis.

### 4. Optional graph-local candidates

If existing graph-local construction code exists, include it as a baseline.

## Acceptance Criteria

For each candidate set, save:

```text
candidate_edges
single_edge_influence
candidate_type
dataset
model
seed
```

---

# Phase 2. Baseline Evaluation

## Goal

For each candidate edge set \(S\), compute and cache:

\[
A(S), \quad H(S), \quad E(S).
\]

## Tasks

Create a script:

```bash
python experiments/group_influence/run_baselines.py \
  --dataset Cora \
  --model GCN \
  --candidate-set top_abs_100 \
  --metric val_loss \
  --seed 0
```

The script should:

1. load model and data,
2. load candidate edge set,
3. compute \(A(S)\),
4. compute \(H(S)\),
5. compute \(E(S)\),
6. save metrics.

## Metrics

Save:

```text
A
H
E
abs_error_H = |A-H|
abs_error_E = |A-E|
relative_error_H
relative_error_E
sign_match_H
sign_match_E
```

## Acceptance Criteria

- We can compare Heo one-shot and single-edge sum on several candidate sets.
- This phase should run before any clustering method is added.

---

# Phase 3. MVP-1: Cheap Influence-Feature Clustering

## Goal

Test whether simple influence-informed features already produce meaningful clusters.

This is not the final Bregman implementation.
This phase is for fast validation.

## Edge feature vector

For each edge \(e\), construct a low-dimensional feature vector \(z_e\).

Start with available features:

```text
single_edge_influence
absolute_single_edge_influence
edge endpoint degrees
endpoint representation cosine similarity
endpoint representation L2 distance
optional: message propagation term
optional: validation loss influence, Dirichlet energy influence, over-squashing influence
```

Example:

\[
z_e =
[
H(\{e\}),
|H(\{e\})|,
\deg(u),
\deg(v),
\cos(h_u,h_v),
\|h_u-h_v\|_2
].
\]

Normalize features before clustering.

## Clustering methods

Implement:

1. k-means,
2. random clustering baseline,
3. optional spectral clustering if easy.

Arguments:

```bash
python experiments/group_influence/run_feature_clustering.py \
  --dataset Cora \
  --model GCN \
  --candidate-set top_abs_100 \
  --num-clusters 5 \
  --feature-type cheap
```

## Cluster outputs

For each cluster \(C_h\), compute:

\[
I_h = \sum_{e\in C_h} H(\{e\})
\]

and

\[
I_{\mathrm{tot}} = \sum_h I_h.
\]

Compute cancellation:

\[
\mathrm{Cancel}(S)
=
\sum_h |I_h|
-
\left|
\sum_h I_h
\right|.
\]

## Metrics

Save:

```text
cluster_id for each edge
cluster_size
cluster_influence_sum I_h
total_influence_sum
cancellation_score
sign_purity
within_cluster_feature_variance
```

## Acceptance Criteria

- We can visualize or print cluster-level influence summaries.
- We can compare random clustering vs influence-feature clustering.
- We can identify whether positive and negative influence groups are separated.

---

# Phase 4. MVP-2: Output-Space Bregman / Mahalanobis Clustering

## Goal

Implement a more principled but still affordable geometry-based clustering method using output perturbations.

Instead of using full parameter-space perturbations, represent each edge edit by how it changes model outputs.

## Feature construction

For each edge edit \(e\), compute:

\[
\Delta h_e
=
h^{G_e,\theta_s}
-
h^{G,\theta_s}.
\]

Possible choices:

1. use outputs on validation nodes only,
2. use logits instead of hidden representations,
3. use last-layer node embeddings,
4. flatten and optionally reduce dimension with PCA.

## Distance options

Start with Euclidean distance:

\[
d(e_i,e_j)=\|\Delta h_{e_i}-\Delta h_{e_j}\|_2^2.
\]

Then add Mahalanobis-type or loss-Hessian-weighted distance if available:

\[
d(e_i,e_j)
=
\frac12
(\Delta h_i-\Delta h_j)^\top
H_{\mathrm{out}}
(\Delta h_i-\Delta h_j).
\]

This is closer to output-space PBRF Bregman geometry.

## Script

```bash
python experiments/group_influence/run_output_bregman_clustering.py \
  --dataset Cora \
  --model GCN \
  --candidate-set top_abs_100 \
  --num-clusters 5 \
  --feature logits_delta \
  --distance euclidean
```

Support:

```text
--feature logits_delta
--feature embedding_delta
--distance euclidean
--distance diagonal_hessian
```

## Acceptance Criteria

- Output-space perturbation features are cached.
- Clustering results can be compared with Phase 3.
- Cancellation and sign-purity metrics are reported.

---

# Phase 5. MVP-3: Approximate Parameter-Space Bregman Clustering

## Goal

Approximate the parameter-space Mahalanobis-type Bregman geometry:

\[
d_G(x,y)
=
\frac12(x-y)^\top G(x-y),
\]

where

\[
G = J^\top HJ+\lambda I.
\]

Do not explicitly construct \(G\).

## Method: Probe-projected features

We want to avoid computing \(G^{-1}\delta g_e\) for every edge.

Choose probe vectors \(p_1,\dots,p_R\).

For each probe, compute:

\[
u_r = G^{-1}p_r.
\]

Then define edge features:

\[
z_e^{(r)}
=
u_r^\top \delta g_e.
\]

So:

\[
z_e =
[z_e^{(1)},\dots,z_e^{(R)}].
\]

Then cluster \(z_e\) using k-means.

## Probe choices

Implement at least:

1. evaluation gradient probe:
   \[
   p = \nabla_\theta f
   \]
2. random Gaussian probes,
3. optional class-specific validation gradient probes.

Use small \(R\) first:

```text
R = 8, 16, 32
```

## Script

```bash
python experiments/group_influence/run_projected_bregman_clustering.py \
  --dataset Cora \
  --model GCN \
  --candidate-set top_abs_100 \
  --num-clusters 5 \
  --num-probes 16 \
  --probe-type eval_plus_random
```

## Important implementation note

If inverse-HVP already exists in the Heo implementation, reuse it.
If not, implement only a minimal interface and start with diagonal approximation.

## Acceptance Criteria

- Projected features \(z_e\) are cached.
- Clustering runs without per-edge inverse-HVP.
- Runtime is reported.
- Results are compared with Phase 3 and Phase 4.

---

# Phase 6. Aggregation Experiments

## Goal

After obtaining clusters, compare aggregation methods.

For each clustering result, compute:

1. Heo one-shot:
   \[
   H(S)
   \]
2. single-edge sum:
   \[
   E(S)
   \]
3. independent cluster sum:
   \[
   C_{\mathrm{ind}}(S)
   =
   \sum_h H(C_h)
   \]
4. cluster-sequential prediction:
   \[
   C_{\mathrm{seq}}^\pi(S)
   \]

## Independent cluster sum

Implement:

```python
def compute_independent_cluster_sum(clusters, model, data, config):
    total = 0.0
    for C_h in clusters:
        total += compute_heo_oneshot(C_h, model, data, config)
    return total
```

## Cluster-sequential prediction: graph-only version first

At first, implement the cheap version:

1. Start from original graph \(G_0\).
2. For cluster \(C_{\pi(1)}\), compute influence from current graph.
3. Apply the cluster edit to the graph.
4. Move to next cluster.
5. Keep model parameters fixed.
6. Sum stepwise predictions.

Pseudo-code:

```python
def compute_cluster_sequential_graph_only(clusters, order, model, data, config):
    current_data = clone_data(data)
    total_pred = 0.0

    for idx in order:
        C = clusters[idx]
        pred = compute_heo_oneshot(C, model, current_data, config)
        total_pred += pred
        current_data = apply_edge_edits(current_data, C, config)

    return total_pred
```

## Ordering policies

Implement:

1. random order,
2. small cluster influence norm first,
3. large absolute cluster influence first,
4. cluster size ascending.

Use simple policies first.

## Metrics

Compare each prediction with actual PBRF \(A(S)\):

```text
abs_error_H
abs_error_E
abs_error_C_ind
abs_error_C_seq
relative_error
sign_match
```

## Acceptance Criteria

- We can run aggregation comparison for each clustering method.
- Results are saved in a single table.
- At least one script prints a summary ranking methods by absolute error.

---

# Phase 7. Interpretability Metrics

## Goal

Test whether clustering reveals hidden influence structure.

## Metrics

### Cluster influence

\[
I_h=\sum_{e\in C_h}H(\{e\}).
\]

### Total influence

\[
I_{\mathrm{tot}}=\sum_h I_h.
\]

### Cancellation score

\[
\mathrm{Cancel}(S)
=
\sum_h |I_h|
-
\left|
\sum_h I_h
\right|.
\]

### Normalized cancellation score

\[
\mathrm{CancelNorm}(S)
=
\frac{
\sum_h |I_h|-
\left|\sum_h I_h\right|
}{
\sum_h |I_h|+\epsilon
}.
\]

### Sign purity

For each cluster:

```text
positive_ratio = # positive edges / cluster_size
negative_ratio = # negative edges / cluster_size
sign_purity = max(positive_ratio, negative_ratio)
```

Average over clusters.

## Acceptance Criteria

- Report cancellation and sign-purity for each clustering method.
- Compare against random clustering.
- Identify examples where total influence is small but cancellation score is high.

---

# Phase 8. Reporting and Visualization

## Required tables

### Table 1. Baseline prediction

```text
dataset | model | candidate_set | A | H | E | err_H | err_E
```

### Table 2. Clustering interpretability

```text
dataset | model | candidate_set | clustering_method | K | cancellation | sign_purity | within_dispersion
```

### Table 3. Aggregation performance

```text
dataset | model | candidate_set | clustering_method | aggregation_method | prediction | error | sign_match
```

## Required plots

1. cluster influence bar plot:
   - x-axis: cluster ID
   - y-axis: cluster influence \(I_h\)

2. cancellation example:
   - show case where total influence is near zero but positive and negative clusters are large

3. error comparison plot:
   - H vs E vs C_ind vs C_seq

4. runtime plot or table:
   - feature extraction time
   - clustering time
   - aggregation time

## Acceptance Criteria

- Each experiment run should produce a machine-readable result file.
- Add one summary notebook or script to aggregate results.

---

# Phase 9. Suggested Directory Structure

Use the existing project style if different. Otherwise:

```text
experiments/
  group_influence/
    run_baselines.py
    build_candidate_edges.py
    run_feature_clustering.py
    run_output_bregman_clustering.py
    run_projected_bregman_clustering.py
    run_aggregation.py
    summarize_results.py

src/
  group_influence/
    candidates.py
    baseline_api.py
    features.py
    clustering.py
    aggregation.py
    metrics.py
    cache.py
```

---

# Phase 10. Minimal First Milestone

Implement only this first:

1. candidate edge construction,
2. baseline computation \(A(S),H(S),E(S)\),
3. cheap influence-feature clustering,
4. cancellation score,
5. independent cluster sum.

Do not implement projected Bregman clustering or full sequential prediction until this works.

## Minimal command flow

```bash
python experiments/group_influence/build_candidate_edges.py \
  --dataset Cora \
  --model GCN \
  --candidate-type top_abs \
  --num-candidates 100 \
  --seed 0
```

```bash
python experiments/group_influence/run_baselines.py \
  --dataset Cora \
  --model GCN \
  --candidate-set top_abs_100 \
  --metric val_loss \
  --seed 0
```

```bash
python experiments/group_influence/run_feature_clustering.py \
  --dataset Cora \
  --model GCN \
  --candidate-set top_abs_100 \
  --num-clusters 5 \
  --feature-type cheap \
  --seed 0
```

```bash
python experiments/group_influence/run_aggregation.py \
  --dataset Cora \
  --model GCN \
  --candidate-set top_abs_100 \
  --clustering-method cheap_kmeans \
  --aggregation-method independent_cluster_sum \
  --seed 0
```

## Minimal success criteria

- The pipeline runs end-to-end on one dataset/model.
- It produces:
  - baseline prediction errors,
  - cluster assignments,
  - cluster influence summaries,
  - cancellation score,
  - independent cluster sum prediction.

---

# Notes

The purpose of the first implementation is not to prove the full theory.
The purpose is to quickly test whether influence-geometry grouping reveals meaningful subgroup structure and whether cluster-level aggregation is promising.

Prioritize:

1. correctness,
2. caching,
3. small candidate sets,
4. simple baselines,
5. clear result tables.

Do not optimize too early.
