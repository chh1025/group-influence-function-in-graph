# Groupwise Metric Implementation Notes

This document describes the **actual implementation** that was added on top of the original repository in response to [groupwise_metric_codex_instructions.md](/data_seoul/undergrad_hh/influence-function-for-edge-edit/groupwise_metric_codex_instructions.md).

The intended use of this document is:
- to give a future Codex run enough context to understand what was implemented,
- to make it possible to port the same logic into other influence experiment settings in this repository,
- to clarify where the implementation matches the original instructions and where it intentionally deviates.

This document should be read together with:
- [groupwise_metric_codex_instructions.md](/data_seoul/undergrad_hh/influence-function-for-edge-edit/groupwise_metric_codex_instructions.md)

## 1. High-level Summary

The implementation adds a **groupwise partition search layer** on top of the existing grouped influence pipeline.

It does **not** replace the original influence calculation core.

Current semantics:
- Existing method: `calculate_influence_total`
  - one candidate edge-group is evaluated as a single global edit
- New method: `clusterwise_fixed_theta_total`
  - the same candidate edge-group is partitioned into subgroups using the new groupwise metric search
  - each subgroup is evaluated separately under fixed theta
  - subgroup influences are summed
- Ground truth proxy: `pbrf_total`

In code terms, the groupwise implementation is currently a **candidate-internal clusterer**.

It plugs into the existing `candidate_clusterer_fn` dispatch, so the main influence pipeline still works the same way after the partition is produced.

## 2. Current Scope

What is implemented:
- `metric_mode = "groupwise"` support
- edge-level representation extraction for candidate edges
- edge-level curvature proxy extraction using existing HVP code
- groupwise partition objective
- split / merge / reassignment refinement loop
- global baseline objective tracking
- integration into grouped influence experiments
- diagnostics export into `candidate_results.csv`

What is **not** implemented yet:
- direct support for `edge_insertion` in groupwise partition search
- a new standalone influence estimator outside the existing clusterwise fixed-theta path
- a representation-space Hessian or exact local metric
- a new step-by-step groupwise influence path

Current hard restriction:
- `metric_mode="groupwise"` is only active for `edge_removal`
- for `edge_insertion`, the code explicitly falls back to the original behavior

## 3. Core Files Added

New package:

```text
groupwise_metric/
    __init__.py
    proxies.py
    chains.py
    metrics.py
    objective.py
    refine.py
    search.py
    experiment.py
```

Responsibilities:

- `groupwise_metric/proxies.py`
  - probe vector generation
  - proxy normalization
  - group proxy center / dispersion

- `groupwise_metric/chains.py`
  - group chain construction
  - group radius computation

- `groupwise_metric/metrics.py`
  - consistency surrogate `F_g`
  - effective radius `d_g`

- `groupwise_metric/objective.py`
  - per-group stats
  - partition objective `J`
  - global baseline objective

- `groupwise_metric/refine.py`
  - initialization
  - split / merge / reassignment proposals
  - k-means and min-group-size handling

- `groupwise_metric/search.py`
  - monotone descent loop
  - config object construction
  - iteration logging

- `groupwise_metric/experiment.py`
  - public integration point for existing grouped influence experiments
  - precompute edge features/proxies for candidate edges
  - build a `candidate_clusterer_fn` compatible with the old pipeline

## 4. Existing Files Modified

Main integration points:

- [calculate_influence.py](/data_seoul/undergrad_hh/influence-function-for-edge-edit/calculate_influence.py)
  - added edge representation helpers
  - added curvature probe generation
  - added edge curvature proxy extraction
  - added local train-graph construction from influenced nodes
  - added config defaults for groupwise mode

- [src/utils.py](/data_seoul/undergrad_hh/influence-function-for-edge-edit/src/utils.py)
  - `metric_mode` dispatch
  - groupwise partition search injection via `candidate_clusterer_fn`
  - export of groupwise diagnostics into result CSV

- [main.py](/data_seoul/undergrad_hh/influence-function-for-edge-edit/main.py)
  - argparse and Hydra config support for groupwise fields
  - result summary uses `clusterwise_fixed_theta` when `metric_mode=groupwise`

- experiment configs in `conf/experiment/*.yaml`
  - added `metric_mode`
  - added `groupwise_*` config fields where needed

## 5. Actual Data Flow

The implemented data flow is:

1. The experiment creates grouped edge-removal candidates as before.
2. If `metric_mode != "groupwise"`, the old partition logic is used.
3. If `metric_mode == "groupwise"` and `influence_type == "edge_removal"`:
   - `prepare_groupwise_candidate_clusterer(...)` is called.
   - For every unique edge appearing in the candidate tensor:
     - compute edge representation `h_i`
     - compute curvature proxy `Q_i`
   - For every candidate:
     - run groupwise partition search on that candidate’s edges
     - cache the discovered partition as a list of edge clusters
   - inject a callable into `args.candidate_clusterer_fn`
4. Existing clusterwise fixed-theta influence code then uses those cached clusters.
5. `clusterwise_fixed_theta_total` becomes the “new method” output.

Important consequence:
- the groupwise code currently changes **how the candidate is partitioned**
- it does **not** change the underlying influence formula inside each cluster

## 6. Mathematical Objects as Implemented

This section documents the concrete implementation, not the idealized version.

### 6.1 Sample Unit

The sample unit is **one edge inside one grouped candidate**.

For one candidate with shape `[num_group_elem, 2]`, each edge is treated as one local sample `i`.

### 6.2 Representation `h_i`

The implementation does not use hidden-layer hooks.

Instead, it uses model output representations at the edge endpoints:

- compute node output tensor `z = model(graph)`
- for edge `(u, v)`, define edge representation as

```python
h_i = concat(
    0.5 * (z_u + z_v),
    abs(z_u - z_v),
)
```

Why:
- stable across current models
- requires no model refactor
- deterministic and easy to store

This is implemented in:
- `GraphInfluenceModule.get_edge_representations(...)`

### 6.3 Curvature Proxy `Q_i`

The implementation follows the instruction in spirit:
- use fixed random probe vectors
- use scalar probe responses `u^T H_i u`

Concrete behavior:

1. For edge `i`, compute the train nodes influenced by that edge using the same neighborhood logic already used in the influence code.
2. Create a graph clone with:
   - same graph structure
   - same model
   - `train_mask` replaced by a mask containing only the influenced train nodes
3. Use existing full-batch HVP machinery from `torch_influence`:

```python
module._hvp_graph(local_graph, flat_params, vec=probe, gnh=module.gnh)
```

4. For each probe vector:

```python
q_j = probe dot hvp
```

5. Stack those scalars into `Q_i`

This is implemented in:
- `GraphInfluenceModule.get_train_influenced_nodes(...)`
- `GraphInfluenceModule.build_local_train_graph(...)`
- `GraphInfluenceModule.get_curvature_probe_vectors(...)`
- `GraphInfluenceModule.compute_edge_curvature_proxies(...)`

Caching:
- probe vectors are cached by `(probe_dim, seed, flat_dim, device, dtype)`
- edge proxies are cached by canonical undirected edge key

Normalization:
- proxy matrix can be normalized column-wise by mean/std
- controlled by `groupwise_normalize_proxies`

### 6.4 Group Center `M_g`

Implemented exactly as:

```python
M_g = mean(Q_i for i in group)
```

File:
- `groupwise_metric/proxies.py`

### 6.5 Dispersion `tau_g`

Implemented as:

```python
tau_g = sqrt(mean(||Q_i - M_g||^2))
```

File:
- `groupwise_metric/proxies.py`

### 6.6 Chain and Radius `r_g`

Implemented options:
- `nearest_neighbor`
- `center_distance`

Default:
- choose anchor nearest to group center
- greedily append nearest unvisited point

Then:

```python
r_g = max ||h_i - h_{i+1}||
```

Files:
- `groupwise_metric/chains.py`

### 6.7 Consistency `F_g`

Implemented as the scalar-proxy surrogate described in the instruction:

```python
w_g = mean(relu(M_g)) + damping
per_link_energy = 0.5 * w_g * ||delta_h||^2
F_g = max(2 * per_link_energy / H_op_g)
H_op_g = w_g
```

With this surrogate, `F_g` is effectively on the same scale as maximum link squared distance.

File:
- `groupwise_metric/metrics.py`

### 6.8 Effective Radius `d_g`

Implemented as:

```python
d_g = max(r_g, sqrt(F_g))
```

Optional damping through:

```python
d_g <- (1-rho) * prev_d_g + rho * raw_d_g
```

File:
- `groupwise_metric/metrics.py`

### 6.9 Objective `J`

Implemented exactly as:

```python
J = sum_g tau_g * d_g^2
```

File:
- `groupwise_metric/objective.py`

### 6.10 Global Baseline

Implemented as a single-group partition:

```python
groups = [all_indices]
```

The search starts from:
- global baseline if initialization is worse
- initialized partition if initialization already improves `J`

File:
- `groupwise_metric/objective.py`
- `groupwise_metric/search.py`

## 7. Search / Refinement Behavior

### 7.1 Initialization

Initialization uses joint feature:

```python
z_i = [sqrt(alpha) * h_i ; sqrt(beta) * Q_i]
```

and a lightweight in-package k-means implementation.

This is implemented in:
- `initialize_groups(...)`
- `_run_kmeans(...)`

### 7.2 Split

The current code tries at most one accepted split per outer iteration:
- rank groups by current objective contribution
- propose 2-way k-means split in joint `(h, Q)` space
- accept if new partition reduces `J` by more than `split_threshold`

### 7.3 Merge

Merge candidates:
- nearest group centers in joint `(h, Q)` space
- top `merge_topk` pairs only

Accept if `J` decreases by more than `merge_threshold`

### 7.4 Reassignment

Current reassignment is local and heuristic:
- source points = farthest from source group center
- target groups = nearest group centers
- accept only if `J` decreases by more than `move_threshold`

### 7.5 Termination

The loop stops when one full outer iteration accepts:
- no split
- no merge
- no move

Implementation file:
- `groupwise_metric/search.py`

## 8. Important Practical Safeguards Added

These were not optional in practice.

### 8.1 `min_group_size`

Without this, the objective tends to collapse into singleton groups because:
- `tau_g` and radius terms can become artificially small
- the monotone descent loop keeps splitting

Current safeguard:
- `groupwise_min_group_size`
- splits that create too-small groups are rejected
- initialization merges undersized groups into nearest groups

This is implemented in:
- `groupwise_metric/refine.py`

### 8.2 Fallback to Global Baseline

Search starts from the global baseline unless initialization is already better.

This preserves the “no worse than global at the search-objective level” invariant.

### 8.3 Current `edge_removal`-only Scope

The groupwise partitioner currently raises `NotImplementedError` for:
- `edge_insertion`

Reason:
- the current local-proxy logic is defined around removal-side influenced nodes
- insertion support should be implemented explicitly rather than silently guessed

## 9. Config Fields

The implementation currently understands these fields:

```yaml
experiment:
  metric_mode: global | groupwise
  groupwise_num_groups_init: 3
  groupwise_alpha_repr: 1.0
  groupwise_beta_proxy: 1.0
  groupwise_probe_dim: 8
  groupwise_probe_seed: 0
  groupwise_normalize_proxies: 1
  groupwise_damping: 1e-3
  groupwise_split_threshold: 1e-6
  groupwise_merge_threshold: 1e-6
  groupwise_move_threshold: 1e-6
  groupwise_radius_update_rho: 1.0
  groupwise_max_outer_iters: 50
  groupwise_chain_method: nearest_neighbor | center_distance
  groupwise_metric_energy_mode: scalar_proxy
  groupwise_merge_topk: 10
  groupwise_max_reassign_candidates_per_group: 5
  groupwise_reassign_target_topk: 2
  groupwise_min_group_size: 2
```

These are parsed in:
- `main.py`
- `calculate_influence.py`
- Hydra experiment config YAMLs

## 10. Current Integration Semantics

This is the most important section for future extension.

### 10.1 Where the new method enters

The new method enters in:
- `src/utils.calculate_grouped_influence(...)`

Logic:
- instantiate the usual `GraphInfluenceModule`
- if `metric_mode == "groupwise"` and `influence_type == "edge_removal"`:
  - call `prepare_groupwise_candidate_clusterer(...)`
  - set `args.candidate_clusterer_fn = generated_clusterer`
- then call the existing clusterwise influence code

So the new method currently **only replaces the partitioner**.

### 10.2 What “new method” actually means in experiment outputs

In result CSVs:

- `calculate_influence_total`
  - old/global method
- `clusterwise_fixed_theta_total`
  - current groupwise method
- `pbrf_total`
  - proxy ground truth used for evaluation

### 10.3 Additional diagnostics exported

The CSV export includes:
- `groupwise_num_groups`
- `groupwise_objective_final`
- `groupwise_objective_global`
- `groupwise_objective_ratio`
- `groupwise_start_label`
- `groupwise_clusters`

Export path:
- `src/utils.save_candidate_result_tables(...)`

## 11. Deviations from the Original Instruction

Future Codex runs should know these explicitly.

### 11.1 The implementation is candidate-internal, not a new global experiment family

Instruction intent:
- a clean groupwise metric option on top of group influence experiments

Implemented form:
- partition search inside each grouped candidate
- reuse of existing clusterwise fixed-theta influence logic

This is a pragmatic but narrower interpretation.

### 11.2 Representation is based on output logits, not a hidden layer

Reason:
- no stable hidden-feature API existed across all current GNN models

### 11.3 `Q_i` is defined for edges using influenced-node local train masks

This is a valid local proxy, but it is not the only possible interpretation of “local second-order operator associated with sample `i`”.

### 11.4 `edge_insertion` is not implemented

This must be added explicitly if needed in future experiments.

### 11.5 The search objective can improve without improving GT alignment

This is already visible in experiments.

Future work should not assume that lowering `J` is sufficient for better influence estimation.

## 12. How to Reuse This in Other Influence Settings

If future work wants to apply the same groupwise method to another influence experiment family, the minimum required steps are:

### 12.1 Confirm the sample unit

You must decide what one local sample `i` is:
- edge
- node
- edge-edit cluster member
- something else

Current code assumes:
- `i = one edge inside one grouped candidate`

### 12.2 Implement representation extraction for that sample unit

You need an analog of:

```python
GraphInfluenceModule.get_edge_representations(...)
```

If the new experiment unit is not an edge, this function must change.

### 12.3 Implement local curvature proxy extraction for that sample unit

You need an analog of:

```python
GraphInfluenceModule.compute_edge_curvature_proxies(...)
```

The current logic depends on:
- determining affected train nodes
- building a local train-mask graph
- calling existing `_hvp_graph(...)`

For a new experiment type, the only safe way to reuse this is:
- define the local train subset for one sample
- define how to build the local graph/objective context
- keep probe vectors fixed per run

### 12.4 Adapt `prepare_groupwise_candidate_clusterer(...)`

That function currently assumes:
- candidates are `[num_candidates, num_group_elem, 2]`
- each row is an undirected edge

If the candidate tensor format changes, this file must be updated.

### 12.5 Keep the partitioner separate from the influence core

This current implementation is modular because:
- the partitioner returns clusters
- existing influence code consumes those clusters

That separation should be preserved.

## 13. Recommended Future Refactor Points

If this feature is going to be used more broadly, these refactors are worth doing:

1. Add a general sample-unit abstraction instead of edge-specific logic.
2. Expose model hidden representations in a stable model API.
3. Move local HVP/proxy logic out of `GraphInfluenceModule` into a reusable utility.
4. Add explicit support for `edge_insertion`.
5. Add a dedicated groupwise result summary script.
6. Consider adding a hybrid search objective that combines `J` with direct influence-alignment signals.

## 14. Checklist for Future Codex Runs

When adapting this implementation, future Codex should verify:

1. Is `metric_mode=groupwise` entering the pipeline at the correct experiment point?
2. Is the sample unit correctly defined for the new experiment?
3. Are `h_i` and `Q_i` computed once per local sample and reused across partition search?
4. Does the candidate format still match the clusterer assumptions?
5. Are `groupwise_*` config keys present in the relevant Hydra experiment YAML?
6. Is there a safeguard against singleton-group collapse?
7. Are `clusterwise_fixed_theta_total` and `calculate_influence_total` still interpreted correctly in result analysis?
8. Is the new method being evaluated against `pbrf_total` consistently?

## 15. Minimal Mental Model

If a future Codex needs a short mental model, use this:

- The old code already knew how to score a whole edge-group and how to score a list of subgroups.
- The new code teaches the repo how to choose those subgroups using a curvature-aware objective.
- The curvature-aware choice is done by:
  - building edge features `h_i`
  - building edge curvature proxies `Q_i`
  - searching over partitions to reduce `J = sum tau_g d_g^2`
- The final influence estimate is still obtained by running the old clusterwise fixed-theta influence calculation on the discovered partition.
