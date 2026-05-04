# Codex Task: Add METIS-based alternatives without removing the current candidate-local partitioning path

## Goal

Extend the current partitioning pipeline with **two additional strategies** while **preserving the existing implementation and default behavior**.

Do **not** delete, replace, or silently change the current candidate-local affinity/METIS path.
The current behavior should remain the default unless a new strategy is explicitly selected.

The two new strategies to add are:

1. **Global training-graph METIS assignment**
2. **Hybrid global-partition-aware local METIS**

These should be implemented as **new selectable modes** in the existing partitioning pipeline.

---

## Current behavior summary

The current implementation does **not** partition the full training graph directly.
Instead, for each candidate edge set:

1. Build a **candidate-level affinity graph** whose nodes are candidate edges.
2. Compute affinity weights using structural information from the original training graph.
3. Run METIS on that candidate-level graph.
4. Compute cluster-wise influence and sum it.

This behavior must stay intact and remain the default.

---

## Required high-level outcome

Add a new top-level strategy selector, for example:

- `candidate_local_affinity`  ← current behavior, default
- `global_training_graph_assignment`
- `hybrid_training_graph_masked_local`

You may choose a slightly different flag name if it fits the codebase better, but the semantics should match exactly.

Suggested CLI flag:

```bash
--partition_strategy {candidate_local_affinity,global_training_graph_assignment,hybrid_training_graph_masked_local}
```

Default must be:

```bash
--partition_strategy candidate_local_affinity
```

Do not break existing configs that only specify `--metric_mode partition` and `--partition_method metis`.

---

## New strategy 1: Global training-graph METIS assignment

## Intuition

Partition the **full training graph nodes once** with METIS.
Then, for each candidate edge set, cluster candidate edges based on the precomputed training-graph partition membership of their endpoints.

This is intended as a reusable global baseline.

---

## Desired behavior

### Step 1. Partition the full training graph nodes

- Build a METIS input graph from the **full training graph**.
- Use the training graph as an **undirected graph**.
- Run `pymetis.part_graph(...)` on the training graph nodes.
- The result should be a node-to-partition assignment vector:

```python
node_part[node_id] -> int in [0, num_parts-1]
```

### Step 2. Cache the result

The full-graph partition must **not** be recomputed for every candidate.
Cache it per requested partition count.

Suggested cache key:

```python
(num_parts, graph_identity)
```

If graph identity is inconvenient, use a run-local cache keyed by `num_parts` and assume the training graph is fixed for the run.

### Step 3. Map each candidate edge to an owner partition

For a candidate edge `(u, v)`:

- if `node_part[u] == node_part[v]`, assign the edge to that partition
- otherwise, assign the edge using a deterministic cross-partition ownership rule

For the **first implementation**, use this simple deterministic baseline:

```python
owner_part(u, v) = min(node_part[u], node_part[v])
```

This is intentionally simple and acceptable as a baseline.
Do not over-engineer this first version.

Optional: if trivial to support, add a future-proof enum for cross-edge policy, but only if it does not complicate the patch too much.

### Step 4. Candidate clustering

Given one candidate edge set:

- group all candidate edges by `owner_part`
- return the non-empty groups as clusters

Important:

- the **effective** number of clusters may be smaller than the requested `num_clusters`
- do **not** create empty clusters just to force the count
- keep cluster ordering stable and deterministic (e.g. sort by owner partition id)

### Step 5. K=1 behavior

If the requested `num_clusters == 1`, global mode should reduce to one cluster containing the full candidate edge set.

---

## New strategy 2: Hybrid global-partition-aware local METIS

## Intuition

Use a full-training-graph partition as a **global locality prior**, but still preserve the current candidate-local affinity graph and local METIS step.

The easiest robust version is:

1. compute global training-graph node partitions once
2. assign each candidate edge an owner partition (same rule as strategy 1)
3. build the current candidate-local affinity graph as before
4. **suppress or downweight affinity across different owner partitions**
5. run the existing local METIS partitioning on the modified candidate affinity graph

This strategy combines:

- global reusable locality structure
- current candidate-specific structural affinity

---

## Desired behavior

### Step 1. Reuse the cached full-graph node partition

Reuse the same cached full-training-graph METIS result as in strategy 1.
Do not recompute it per candidate.

### Step 2. Compute candidate edge owner partitions

Use the same owner rule as strategy 1:

```python
owner_part(u, v) = node_part[u] if node_part[u] == node_part[v] else min(node_part[u], node_part[v])
```

### Step 3. Build the current candidate affinity matrix exactly as before

Do **not** change the current affinity computation logic for the default mode.
In hybrid mode, start from the existing candidate affinity matrix, then post-process it.

### Step 4. Mask or downweight cross-owner affinities

For candidate edges `e_i` and `e_j`:

- if `owner_part(e_i) == owner_part(e_j)`, keep the affinity weight unchanged
- otherwise multiply the affinity by a new scale parameter

Suggested CLI flag:

```bash
--hybrid_cross_owner_affinity_scale 0.0
```

Default: `0.0`

That means by default, hybrid mode completely removes affinity between candidate edges that belong to different global owner partitions.

Pseudo-code:

```python
if owner_part[i] != owner_part[j]:
    affinity[i, j] *= hybrid_cross_owner_affinity_scale
```

Then run the existing local METIS partitioning on this modified affinity matrix.

### Step 5. K=1 behavior

If `num_clusters == 1`, return the full candidate as one cluster, same as the current path.

---

## Backward-compatibility requirements

1. **Current behavior must remain default**.
2. Existing configs/scripts should continue to work unchanged.
3. Numerical behavior of the current default path should remain unchanged.
4. Only new flags/modes should activate the new strategies.

---

## Suggested code organization

Please adapt to the current codebase structure, but keep the patch clean and readable.
A reasonable organization would be:

### Existing files likely involved

- `main.py`
- `candidate_partition.py`
- `candidate_partition_graph.py`
- `partition_methods.py`
- `src/utils.py`

### Suggested additions

Add helper utilities/modules as needed, for example:

- `training_graph_partition.py`
  - build full-graph METIS input
  - run/cache node partitioning
  - edge owner assignment helpers

- `candidate_partition_global.py`
  - global assignment clustering logic

- optionally a small shared helper module for strategy dispatch

You do **not** have to use these exact filenames, but keep the separation of concerns clear.

---

## Recommended implementation plan

### 1. Add a new strategy selector

Thread a new parameter through the partitioning pipeline.
Keep the current path as the default.

### 2. Implement reusable full-training-graph METIS preprocessing

Create a helper that:

- accepts the training graph
- builds an undirected adjacency
- runs METIS for a requested `num_parts`
- caches and returns `node_part`

### 3. Implement global candidate clustering

Given `node_part` and a candidate edge tensor:

- compute `owner_part` for each candidate edge
- bucket edge indices by owner partition
- convert buckets back into the cluster tensor/list format expected downstream

### 4. Implement hybrid masking

Given:

- current candidate affinity matrix
- owner partition per candidate edge

apply the cross-owner mask/downweight, then reuse the existing local partitioning code.

### 5. Keep downstream influence code unchanged if possible

Prefer adapting the clusterer output so that downstream influence code can remain unchanged.
This will reduce regression risk.

---

## Detailed functional requirements

### A. Full-graph METIS preprocessing helper

Implement a reusable helper roughly like:

```python
def get_or_build_training_graph_metis_partition(
    train_edge_index,
    num_parts,
    cache,
    *,
    num_nodes=None,
):
    ...
    return node_part
```

Requirements:

- use undirected adjacency
- be deterministic given the same inputs/seed
- cache results per `num_parts`
- avoid repeated recomputation across candidates

### B. Global assignment clusterer

Implement a clusterer roughly like:

```python
def cluster_candidate_edges_by_global_partition(
    candidate_edges,
    node_part,
):
    ...
    return clusters
```

Requirements:

- deterministic ordering
- non-empty clusters only
- preserve original candidate edge values exactly
- support `num_clusters == 1`

### C. Hybrid masked-local clusterer

Implement a clusterer roughly like:

```python
def cluster_candidate_edges_hybrid_masked_local(
    candidate_edges,
    train_graph_context,
    node_part,
    num_clusters,
    ...existing affinity params...,
    hybrid_cross_owner_affinity_scale=0.0,
):
    ...
    return clusters
```

Requirements:

- reuse existing affinity builder
- only modify the affinity matrix after it is built
- then reuse the current METIS-on-affinity-graph logic
- preserve current path untouched when not in hybrid mode

---

## Logging / debug info

Add lightweight logging or returned metadata where practical.
At minimum, make it easy to inspect:

### Global mode

- requested `num_clusters`
- effective number of non-empty clusters
- number of candidate edges assigned to each owner partition
- number of cross-partition candidate edges

### Hybrid mode

- requested `num_clusters`
- owner partition histogram
- number of affinity entries removed/downweighted by hybrid masking
- effective number of clusters returned

Do not spam logs by default; use existing verbosity/debug patterns if present.

---

## Edge cases to handle

### 1. Candidate edges all in one owner partition

- Global mode: return one non-empty cluster
- Hybrid mode: should reduce to the current local path (because no cross-owner masking matters)

### 2. Candidate edges all cross partitions

- Global mode: ownership rule still must produce deterministic clusters
- Hybrid mode: owner partitions still must be well-defined for masking

### 3. Requested `num_clusters` larger than candidate size

- Do not crash
- behave consistently with current conventions

### 4. Empty or degenerate affinity graph

- Reuse current fallback behavior where possible
- Do not introduce a different failure mode for the new strategies

### 5. K-sweep experiments

If the code already sweeps multiple values of `num_clusters`, the full-graph METIS partition should be computed at most once per unique K and then reused.

---

## Minimal CLI/API additions

Please add only what is necessary.
A good minimal set is:

```bash
--partition_strategy {candidate_local_affinity,global_training_graph_assignment,hybrid_training_graph_masked_local}
--hybrid_cross_owner_affinity_scale 0.0
```

Optional but nice if low-effort:

```bash
--global_cross_edge_owner_policy {min_partition}
```

For now, `min_partition` is enough.
Do not add many extra policies unless implementation is truly trivial.

---

## Testing requirements

Add lightweight tests or at least a reproducible smoke-test path.

### Test 1. Backward compatibility

- Run an existing config using the default strategy
- Confirm outputs match the pre-change behavior

### Test 2. Global mode basic functionality

- Run one small example with `global_training_graph_assignment`
- Verify the full graph is partitioned once
- Verify candidate edges are bucketed by owner partition
- Verify downstream influence code runs end-to-end

### Test 3. Hybrid mode basic functionality

- Run one small example with `hybrid_training_graph_masked_local`
- Verify candidate affinity is built
- Verify cross-owner affinity entries are masked/downweighted
- Verify local METIS still runs afterward

### Test 4. Caching behavior

- For multiple candidates with the same K, confirm full-training-graph METIS is reused rather than recomputed each time

### Test 5. K=1

- Global mode should return the whole candidate as one cluster
- Hybrid mode should also collapse to one cluster

---

## Acceptance criteria

The patch is successful if all of the following are true:

1. The current candidate-local strategy remains the default and unchanged.
2. A new global training-graph assignment mode works end-to-end.
3. A new hybrid global-aware masked-local mode works end-to-end.
4. Full-training-graph METIS is cached and reused.
5. Existing experiments do not break.
6. The code is clean, readable, and minimally invasive.

---

## Important non-goals for this patch

Do **not** do the following in this patch:

- do not remove the current local candidate-affinity partitioning implementation
- do not redesign the downstream influence API unless absolutely necessary
- do not add many sophisticated cross-edge ownership heuristics
- do not over-optimize prematurely

This patch is for adding **clean baselines / alternatives** with minimal disruption.

---

## Final note

Please inspect the existing implementation carefully and fit these additions into the current abstractions rather than duplicating large blocks of code.
Refactor only where it clearly reduces duplication **without changing current behavior**.
