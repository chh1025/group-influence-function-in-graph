# Codex Implementation Guide: Group-wise Local Metric Search for Group Influence Function

## Goal

Add a **group-wise local metric / curvature-aware clustering module** on top of the existing group influence experiment codebase.

The codebase already contains:
- group influence experiments,
- iHVP / inverse-HVP utilities,
- training / evaluation pipelines,
- representation extraction or enough model hooks to obtain them.

This implementation should **reuse existing iHVP / HVP / influence infrastructure** as much as possible, and only add the following new functionality:

1. curvature proxy extraction,
2. group-wise clustering,
3. per-group local radius and consistency computation,
4. split / merge / reassignment refinement,
5. optional no-worse-to-global baseline bookkeeping,
6. a clean experimental entrypoint.

---

# 1. High-level design

We are **not** replacing the existing global group influence implementation. Instead, we are adding a new option:

- `metric_mode = "global"` : existing behavior
- `metric_mode = "groupwise"` : new behavior

The new method should:
- cluster examples into groups,
- compute a group-wise curvature proxy representative `M_g`,
- define a group-wise damped metric `H_tilde_g = M_g + lambda * I`,
- compute a local effective radius
  
  \[
  d_g = \max\{r_g, \sqrt{F_g}\}
  \]

- optimize / refine groups using the objective

  \[
  J(\Pi) = \sum_g \tau_g d_g^2
  \]

where:
- `r_g` = max local link norm inside group/chain,
- `F_g` = consistency quantity induced by the group-wise metric,
- `tau_g` = average within-group curvature proxy dispersion.

---

# 2. Key mathematical objects to implement

## 2.1 Representation

For each sample / graph `i`, extract a representation

\[
 h_i
\]

Use whatever representation is already natural in the codebase for the existing group influence experiments.

Preferred order:
1. reuse the exact representation already used in the current influence code,
2. if unavailable, use the same hidden feature used for the existing local linearization / influence calculation,
3. do **not** invent a new representation unless necessary.

Store as a flat vector tensor for clustering and distance computation.

---

## 2.2 Curvature proxy `Q_i`

We do **not** need the full Hessian.

Implement a curvature proxy from existing HVP / iHVP infrastructure.

### Preferred default proxy
Use **probe-vector scalar curvature responses**:

\[
Q_i = (u_1^\top H_i u_1, \dots, u_m^\top H_i u_m)
\]

where:
- `u_1, ..., u_m` are fixed probe vectors,
- `H_i` is the local second-order operator associated with sample `i`.

### Why this choice
- easy to compute with existing HVP code,
- low-dimensional,
- easy to store and cluster,
- no need to materialize any Hessian.

### Implementation details
Add a function like:

```python
compute_curvature_proxy(sample, model, probe_vecs, hvp_fn, damping, normalize=True) -> Tensor[m]
```

Expected behavior:
- for each probe vector `u_j`, compute `hvp = H_i u_j` using existing HVP utilities,
- compute scalar response `q_j = u_j^T hvp`,
- stack all `q_j` into a vector `Q_i`.

### Notes
- Probe vectors should be fixed across the entire experiment.
- They should be generated once per run with fixed seed.
- Start with small `m` (e.g. 8 or 16).
- Optionally normalize `Q_i` across the dataset before clustering.

---

## 2.3 Group representative `M_g`

For each group `g`, define

\[
M_g = \frac{1}{|\mathcal G_g|} \sum_{i \in \mathcal G_g} Q_i
\]

Implement this as the simple mean of the proxy vectors inside the group.

This is important because it is the minimizer of within-group squared proxy error.

Add helper:

```python
compute_group_proxy_center(Q, group_indices) -> Tensor[m]
```

---

## 2.4 Group proxy dispersion `tau_g`

For each group `g`, define

\[
\tau_g^2 = \frac{1}{|\mathcal G_g|} \sum_{i \in \mathcal G_g} \|Q_i - M_g\|^2
\]

Implement:

```python
compute_group_proxy_dispersion(Q, group_indices, M_g) -> float
```

Return `tau_g` (not `tau_g^2`) for convenience.

---

## 2.5 Local link radius `r_g`

Inside each group, define a chain ordering and compute

\[
r_g = \max_{(i,i+1) \subset \mathcal G_g} \|h_i - h_{i+1}\|
\]

Important: each group must have an internal order / chain.

### Default chain construction
Use a simple and deterministic heuristic:
- choose anchor = point closest to representation center,
- greedily build a nearest-neighbor chain inside the group.

Alternative acceptable if simpler:
- sort by distance to group center,
- connect consecutive points in that order.

Keep it simple and deterministic.

Implement:

```python
build_group_chain(h, group_indices, method="nearest_neighbor") -> List[int]
compute_group_radius(h, ordered_group_indices) -> float
```

---

## 2.6 Consistency quantity `F_g`

For each consecutive link in group `g`, define

\[
D_{\phi,g}(h_i,h_{i+1}) = \frac12 (h_i-h_{i+1})^\top \tilde H_g (h_i-h_{i+1})
\]

In practice, since we do not form a full matrix in representation space, implement this using the proxy metric induced by `M_g`.

### Important simplification for implementation
We do **not** need to literally instantiate a huge representation-space matrix.

Instead, define a metric-energy surrogate using the group proxy center magnitude. Use one of the following:

#### Default practical choice
Treat `M_g` as a diagonal or scalarized curvature descriptor and define

\[
D_{\phi,g}(h_i,h_{i+1}) \approx \frac12 w_g \|h_i-h_{i+1}\|^2
\]

where `w_g` is a positive scalar derived from `M_g`, e.g.

```python
w_g = mean(relu(M_g)) + damping
```

This is a practical surrogate that matches the theory direction without requiring explicit representation Hessians.

Then

\[
F_g = \sup_{(i,i+1) \subset \mathcal G_g} \frac{2D_{\phi,g}(h_i,h_{i+1})}{\|\tilde H_g\|_{op}}
\]

can be implemented as

```python
F_g = max_link_energy / H_op_g
```

with
- `max_link_energy = max( w_g * ||h_i - h_{i+1}||^2 )`
- `H_op_g = w_g`

which reduces to approximately

\[
F_g \approx \max \|h_i-h_{i+1}\|^2
\]

This is okay for the first implementation.

### Why this is acceptable
The purpose here is not to perfectly recover the exact representation Hessian, but to implement the **search logic** and group-wise refinement algorithm in a way that is compatible with the codebase and existing HVP tools.

If later needed, this can be replaced by a richer representation-space metric.

Implement:

```python
compute_group_consistency(h, ordered_group_indices, M_g, damping) -> dict
```

Return at least:
- `F_g`
- `c_g = sqrt(F_g)`
- per-link energies
- `H_op_g` surrogate

---

## 2.7 Effective radius `d_g`

Define

\[
d_g = \max\{r_g, \sqrt{F_g}\}
\]

Implement:

```python
compute_effective_radius(r_g, F_g) -> float
```

This is the main active radius used in the objective.

---

# 3. Global objective

Use the single theory-driven objective

\[
J(\Pi) = \sum_g \tau_g d_g^2
\]

Implement:

```python
compute_partition_objective(groups, h, Q, chain_cache=None) -> dict
```

Return:
- per-group stats:
  - `tau_g`
  - `r_g`
  - `F_g`
  - `d_g`
  - chain ordering
  - group size
- total objective `J`

This function is central and should be reused by split / merge / reassignment.

---

# 4. Initialization

## 4.1 Initial grouping

Start with a simple clustering over the joint feature

\[
z_i = [\sqrt{\alpha} h_i ; \sqrt{\beta} Q_i]
\]

using `k-means` or `agglomerative clustering`.

Prefer:
- `sklearn.cluster.KMeans` if available,
- otherwise simple PyTorch k-means implementation.

Initial number of groups can be:
- user-specified,
- or chosen from a small set.

Implement:

```python
initialize_groups(h, Q, num_groups, alpha, beta, seed) -> List[List[int]]
```

---

# 5. Refinement operations

All refinement operations must be evaluated using the same objective `J`.

## 5.1 Split

### Candidate selection
Compute per-group score

\[
S_g^{\text{split}} = \tau_g d_g^2
\]

Pick groups with largest score as split candidates.

### Proposed split
Within a selected group, run a 2-way clustering on

\[
z_i = [\sqrt{\alpha} h_i ; \sqrt{\beta} Q_i]
\]

restricted to that group.

### Acceptance criterion
Let current objective contribution be

\[
J_g = \tau_g d_g^2
\]

and split objective contribution be

\[
J_{g_1}+J_{g_2}
\]

Accept if

\[
\Delta_{\text{split}} = J_g - (J_{g_1}+J_{g_2}) > \eta_{\text{split}}.
\]

Implement:

```python
propose_split(group_indices, h, Q, alpha, beta, seed) -> (group1, group2)
try_split(groups, group_id, ...) -> (accepted, new_groups, diagnostics)
```

---

## 5.2 Merge

### Candidate pairs
Consider only:
- adjacent groups in chain order if defined globally, or
- nearest group centers in joint `(h,Q)` space.

### Acceptance criterion
Let

\[
\Delta_{\text{merge}} = (J_g + J_h) - J_{g\cup h}.
\]

Accept if

\[
\Delta_{\text{merge}} > \eta_{\text{merge}}.
\]

Implement:

```python
candidate_merge_pairs(groups, h, Q, topk=10) -> List[Tuple[g, h]]
try_merge(groups, g1, g2, ...) -> (accepted, new_groups, diagnostics)
```

---

## 5.3 Reassignment

Only allow **boundary reassignment** or **small local reassignment**, not arbitrary full search.

### Candidate points
For each group, identify points near the boundary in representation space or chain order.

### Candidate target groups
Only neighboring / nearby groups.

### Acceptance criterion
Let

\[
\Delta_{i:g\to h} = (J_g + J_h) - (J_g^{(-i)} + J_h^{(+i)}).
\]

Accept if

\[
\Delta_{i:g\to h} > \eta_{\text{move}}.
\]

Implement:

```python
candidate_reassignments(groups, h, Q, max_candidates_per_group=5) -> List[move]
try_reassignment(groups, move, ...) -> (accepted, new_groups, diagnostics)
```

---

# 6. Optimization loop

Use a monotone descent loop.

## Main refinement loop

At each outer iteration:
1. recompute partition objective and per-group stats,
2. attempt splits,
3. attempt merges,
4. attempt reassignments,
5. accept only moves that reduce `J` by more than threshold,
6. stop when no move is accepted.

### Important
For every candidate move, recompute:
- `tau_g`
- `r_g`
- `F_g`
- `d_g`

under the updated partition.

Implement:

```python
refine_groups(groups, h, Q, config) -> (final_groups, history, final_stats)
```

### Finite termination guarantee by construction
Because every accepted move satisfies

\[
J^{(t+1)} \le J^{(t)} - \eta
\]

for some user-defined threshold `eta > 0`, and because `J >= 0`, the algorithm must terminate after finitely many accepted moves.

We do **not** need to prove global optimality.

Codex should preserve this monotone descent invariant.

---

# 7. How to use the consistency condition actively

This is important.

The consistency inequality is **not only** a final check. It should actively inform the radius.

For each group `g`, after computing:
- `r_g` = max local link distance,
- `F_g`,

set the effective radius

\[
d_g = \max\{r_g, \sqrt{F_g}\}
\]

Optionally use a damped update:

\[
d_g^{(t+1)} = (1-\rho)d_g^{(t)} + \rho \max\{r_g^{(t)}, \sqrt{F_g^{(t)}}\}
\]

for `rho in (0, 1]`.

### Required behavior
Codex should ensure that:
- `F_g` is computed every refinement round,
- `sqrt(F_g)` enters the radius update,
- `d_g` is the quantity used inside the objective,
- not the raw candidate `delta_g`.

---

# 8. Global baseline comparison

We want the final method to be **no worse than the global shared-metric baseline at the level of the search objective**.

So the code should include:
- a baseline partition with a single global group,
- computation of

\[
J_{\text{glob}} = \tau_{\text{glob}} d_{\text{glob}}^2
\]

- optional assertion / logging that final `J <= J_glob` if the refinement procedure is started from the global baseline and only accepts improving moves.

Implement:

```python
compute_global_baseline_objective(h, Q, config) -> dict
```

and log:
- `J_final`
- `J_global`
- ratio `J_final / J_global`

This is important for analysis.

---

# 9. Suggested code structure

Create a new module, e.g.

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

## File responsibilities

### `proxies.py`
- probe vector generation
- curvature proxy computation
- group proxy center computation
- dispersion computation

### `chains.py`
- chain construction inside groups
- local link distances

### `metrics.py`
- local metric energy surrogate
- consistency quantity `F_g`
- effective radius `d_g`

### `objective.py`
- compute per-group stats
- compute total objective `J`
- global baseline objective

### `refine.py`
- split / merge / reassignment proposals
- acceptance tests

### `search.py`
- full monotone descent refinement loop

### `experiment.py`
- public entrypoint that plugs into existing group influence code

---

# 10. Required config fields

Add a config block like:

```yaml
groupwise_metric:
  enabled: true
  num_groups_init: 8
  alpha_repr: 1.0
  beta_proxy: 1.0
  probe_dim: 8
  probe_seed: 0
  damping: 1e-3
  split_threshold: 1e-6
  merge_threshold: 1e-6
  move_threshold: 1e-6
  radius_update_rho: 1.0
  max_outer_iters: 50
  chain_method: nearest_neighbor
  metric_energy_mode: scalar_proxy
```

---

# 11. Logging / diagnostics

At every outer iteration, log:
- total objective `J`
- number of groups
- per-group `tau_g`
- per-group `r_g`
- per-group `F_g`
- per-group `d_g`
- number of accepted splits / merges / reassignments
- baseline comparison `J / J_global`

This is required for debugging and analysis.

---

# 12. Minimal correctness requirements

Codex should ensure the following are true:

1. `M_g` is always the mean of `Q_i` inside the group.
2. `tau_g` is computed from group proxy dispersion.
3. `r_g` is computed from consecutive links in the group chain.
4. `F_g` is recomputed after every accepted partition update.
5. `d_g = max(r_g, sqrt(F_g))` or its damped update version.
6. The objective used for split / merge / reassignment is
   \[
   J(\Pi)=\sum_g \tau_g d_g^2.
   \]
7. Accepted refinement moves must strictly decrease `J` by more than threshold.
8. The global single-group baseline must be computable and logged.

---

# 13. What not to do

- Do **not** materialize full Hessians unless already cheaply available.
- Do **not** introduce a completely separate influence pipeline.
- Do **not** replace the existing iHVP code.
- Do **not** over-engineer the chain construction.
- Do **not** claim global optimality or exact convergence to the best partition.
- Do **not** use passive `delta_g` checks only; `F_g` must actively affect `d_g`.

---

# 14. Final deliverable expectation

Implement a working first version of the algorithm with the following priorities:

1. correctness of the group-wise search logic,
2. reuse of existing HVP / iHVP utilities,
3. monotone descent refinement,
4. clear diagnostics,
5. clean integration with current group influence experiments.

Perfection of the curvature metric is **not** the first priority. Start with a simple, stable proxy-based implementation that makes the entire search loop operational.
