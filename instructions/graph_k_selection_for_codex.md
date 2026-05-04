# Auto-selecting `k` for graph clustering in our pipeline

## Goal

Add **automatic `k` selection** for the new **k-means-based clustering path**, while **preserving the existing manual `k` path** and the current METIS / existing partition code.

This note is written for our current setting:

- We currently build a **candidate-level affinity graph** per candidate.
- Nodes of that graph are **candidate edges**.
- Edge weights come from structural relations in the original training graph.
- Clustering is therefore performed **per candidate affinity graph**, not on the full training graph by default.

So the new `k`-selection logic should be implemented to work **naturally on a weighted graph / affinity matrix**.

---

## Context from the current pipeline

Our current code does **not** partition the full training graph. For each candidate edge set, it builds a candidate-level affinity graph whose nodes are candidate edges, with affinities derived from structural relations in the original training graph, and then partitions that graph. Keep this behavior intact as the default existing path; only add a new clustering path with automatic `k` selection.
Reference description: current partitioning is done per candidate affinity graph, not on the full training graph. fileciteturn1file0

---

## High-level recommendation

For graph data, especially our **weighted candidate affinity graphs**, the best practical strategy is:

1. **Primary graph-native proposal:** `eigengap` on the normalized graph Laplacian.
2. **Primary validation / tie-breaker:** `silhouette` on the spectral embedding.
3. **Optional robustness check:** `stability` across random seeds / small graph perturbations.
4. **Always enforce feasibility bounds:** minimum cluster size, maximum cluster size, and candidate size.

In short:

- **Default recommendation:** `auto_k_method = eigengap_silhouette_hybrid`
- **Cheap mode:** `eigengap`
- **More robust but slower mode:** `silhouette` or `stability`

---

## Which `k`-selection methods are worth implementing

### 1. Eigengap heuristic (**recommended as primary graph-native method**)

### Idea
Given affinity matrix `W` for one candidate graph:

- Compute degree matrix `D`
- Compute normalized Laplacian, e.g.
  - `L_sym = I - D^{-1/2} W D^{-1/2}`
- Compute the smallest eigenvalues `λ_1 <= λ_2 <= ... <= λ_r`
- Choose `k` using the **largest gap** `λ_{i+1} - λ_i` within a feasible range

For graph clustering, this is the most natural method because it uses the spectrum of the graph itself.

### Why it fits our problem
- Our data is already a **weighted affinity graph**.
- The candidate graph may have latent cluster structure that is better reflected in the Laplacian spectrum than in raw Euclidean heuristics.
- If the new clustering method is spectral clustering + k-means on spectral embeddings, eigengap is especially natural.

### Pros
- Graph-native
- Cheap compared with repeated full clustering sweeps
- Easy to implement once spectral embedding is already available

### Cons
- Can be unstable if the spectrum is flat / noisy
- Needs tie-breaking when several gaps are similar
- Can overestimate or underestimate `k` on tiny or weakly-structured graphs

### Implementation note
Use this as a **proposal method**, not the final authority when ambiguity is high.

---

### 2. Silhouette sweep on spectral embedding (**recommended validator / alternative default**)

### Idea
For each candidate graph:

1. Compute spectral embedding from the graph affinity matrix
2. For each `k` in a feasible range:
   - run k-means on the embedding (multiple seeds)
   - compute average silhouette score
3. Choose the `k` with the best mean silhouette

Silhouette is a standard validation criterion for choosing the number of clusters. The uploaded reference notes it as a useful criterion: values near 1 mean a point is well matched to its cluster, while values near -1 suggest poor assignment. fileciteturn4file0

### Why it fits our problem
- Works after graph -> spectral embedding
- Gives an interpretable cluster-separation score
- More directly tied to the resulting partition quality than pure eigengap alone

### Pros
- Practical and easy to interpret
- Often more robust than elbow-style visual heuristics
- Good tie-breaker for graph-based k-means

### Cons
- Requires repeated clustering for many `k`
- Can favor overly separated tiny clusters unless size constraints are enforced
- Depends on the embedding quality

### Implementation note
Run multiple k-means restarts per `k` and average the silhouette. Also reject candidates where the best silhouette is weak and fall back to a conservative `k` (often 1 or small `k`).

---

### 3. Stability-based selection (**recommended if we care about scientific robustness**)

### Idea
Choose the `k` whose clustering is most stable under:

- different k-means seeds
- small noise on affinity weights
- weak-edge dropout / threshold perturbation
- optional subsampling for larger candidate graphs

Measure stability with ARI / NMI between repeated runs.

### Why it fits our problem
Our research question is not just “can we partition?” but “can we get a decomposition that is robust enough to support clusterwise influence summation?” A stable clustering is therefore more meaningful than a visually nice elbow.

### Pros
- Best aligned with research reliability
- Helps distinguish real structure from noise
- Good for ablations and paper-quality diagnostics

### Cons
- Most expensive
- More moving parts and hyperparameters
- Overkill for a very fast default path

### Implementation note
This is best as an **optional evaluation mode** or a **tie-breaker** for ambiguous cases, not necessarily the default production path.

---

### 4. Feasibility-bound selection by cluster size (**must implement as a hard constraint**)

### Idea
Even before scoring possible `k`, define a feasible range using the candidate size `m` and desired cluster-size policy.

For example:

- if `min_cluster_size = s_min`, then require `k <= floor(m / s_min)`
- if `max_cluster_size = s_max`, then require `k >= ceil(m / s_max)`

This is not a scoring rule by itself, but it is extremely important in our setting.

### Why it fits our problem
We are clustering **candidate edges in order to compute clusterwise influence**. So `k` should not be chosen only by abstract cluster quality; it must also respect the intended perturbation scale.

### Pros
- Directly aligned with our research objective
- Prevents pathological tiny or huge clusters
- Very cheap and easy to implement

### Cons
- Does not determine `k` alone
- Needs a principled choice of size bounds

### Implementation note
Always intersect the search range from the scoring method with this feasible range.

---

### 5. Connected-components / thresholded-components heuristic (**good auxiliary graph hint**)

### Idea
Threshold the affinity graph at some weak-edge cutoff and inspect connected components.

Possible uses:

- component count as a hint for `k`
- if the graph breaks into clear disconnected blocks, prefer `k` near that count
- if the graph is one dense component, avoid large `k`

### Why it fits our problem
Our affinity graph may already encode strong locality. In that case, simple connectivity structure can be informative.

### Pros
- Very cheap
- Graph-native
- Easy to explain and debug

### Cons
- Highly sensitive to threshold choice
- Can explode into many singleton components
- Should not be the sole decision rule

### Implementation note
Use only as a soft prior / diagnostic, not the default selection rule.

---

## Methods we should **not** rely on as the primary default

### A. Elbow method
The uploaded reference explicitly notes that the elbow is often ambiguous, subjective, and unreliable; it can even appear on uniform random data. fileciteturn4file0

So:

- do **not** use elbow as the default automatic selector
- if implemented at all, keep it as a debug plot only

### B. Gap statistic
The uploaded reference describes the gap statistic as comparing observed within-cluster dispersion to a null reference distribution, but also notes that reliability depends strongly on how plausible the null is. fileciteturn4file0

For graph affinity data, choosing a good null reference is awkward. This makes gap less attractive as a first implementation.

### C. Plain AIC/BIC / X-means as the default
The reference notes that information criteria and X-means are available for choosing `k`, and that X-means extends k-means by repeated subdivision with an information criterion such as AIC/BIC. fileciteturn4file0

These are reasonable on Euclidean mixture-like data, but for our graph setting they are less direct unless we first embed into a vector space and accept Gaussian-ish assumptions in that embedding.

So:

- they are acceptable **optional baselines**
- they should **not** be our first default in the graph pipeline

---

## Recommended policy for **our** project

## Default policy: `eigengap_silhouette_hybrid`

For each candidate affinity graph with `m` nodes:

1. Build feasible range `[k_min_feasible, k_max_feasible]`
   - include `k=1`
   - enforce `min_cluster_size`
   - enforce `max_cluster_size` if used
   - cap `k_max` by `m`
2. If `m` is tiny, return conservative `k`
   - e.g. if `m <= tiny_graph_threshold`, set `k=1` or search a tiny range only
3. Compute normalized Laplacian spectrum
4. Use eigengap to propose `k_eig`
5. Evaluate a small neighborhood around `k_eig` with silhouette on spectral embeddings
   - e.g. candidates in `{k_eig-1, k_eig, k_eig+1}` intersected with feasible range
6. Choose best silhouette among those candidates
7. If silhouette is weak / ambiguous, back off conservatively
   - prefer smaller `k`
   - optionally choose `k=1`

This gives a graph-native, efficient, and less brittle selection rule.

---

## Alternative policies worth supporting

### `auto_k_method = eigengap`
Fastest graph-native method. Good default for cheap sweeps.

### `auto_k_method = silhouette`
Brute-force but reliable baseline. Evaluate all feasible `k` in a range.

### `auto_k_method = stability`
Best for robustness studies / final experiments.

### `auto_k_method = bic_xmeans_like`
Optional embedding-based baseline only.

---

## Concrete implementation instructions for Codex

## 1. Do **not** remove existing behavior

Keep all existing paths intact:

- current METIS / partition path
- current manual `k` behavior for the new method (if already present)

Only add a **new auto-`k` layer**.

---

## 2. Add CLI / config options

Add new config arguments similar to the following:

```text
--auto_k_method {none,eigengap,silhouette,eigengap_silhouette_hybrid,stability,bic_xmeans_like}
--auto_k_min 1
--auto_k_max 8
--auto_k_max_ratio 1.0
--auto_k_min_cluster_size 1
--auto_k_max_cluster_size 0        # 0 means disabled
--auto_k_tiny_graph_threshold 4
--auto_k_num_restarts 10
--auto_k_random_seed 0
--auto_k_silhouette_metric euclidean
--auto_k_stability_trials 10
--auto_k_stability_edge_dropout 0.05
--auto_k_weak_silhouette_threshold 0.05
--auto_k_prefer_smaller_k true
```

Notes:

- `none` means manual `k` only
- `auto_k_max` should be further clipped by candidate size
- size-based feasibility should always be enforced

---

## 3. Implement a single reusable selector API

Create a new utility module, e.g.

```text
cluster_k_selection.py
```

with an interface like:

```python
def choose_num_clusters_for_affinity_graph(
    affinity: np.ndarray,
    method: str,
    k_min: int,
    k_max: int,
    min_cluster_size: int = 1,
    max_cluster_size: Optional[int] = None,
    tiny_graph_threshold: int = 4,
    num_restarts: int = 10,
    random_state: int = 0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Returns:
        {
            "k": int,
            "scores": {k: ...},
            "diagnostics": {...},
            "method": method,
        }
    """
```

The selector should return both the chosen `k` and diagnostics for logging/debugging.

---

## 4. Define the feasible `k` range carefully

Given candidate graph size `m`:

```python
k_lo = max(1, auto_k_min)
k_hi = min(auto_k_max, m)

if min_cluster_size > 1:
    k_hi = min(k_hi, m // min_cluster_size)

if max_cluster_size and max_cluster_size > 0:
    k_lo = max(k_lo, math.ceil(m / max_cluster_size))
```

If `k_lo > k_hi`, fall back conservatively:

- either `k = 1`
- or relax one of the bounds in a clearly logged way

For tiny graphs, prefer smaller `k`.

---

## 5. Implement graph spectral utilities

Implement reusable helpers:

```python
def normalized_laplacian_from_affinity(W): ...
def spectral_embedding_from_affinity(W, dim): ...
def eigengap_scores(W, k_min, k_max): ...
```

Behavior notes:

- Ensure symmetry of `W`
- Handle zero-degree nodes safely
- Use deterministic ordering of eigenpairs
- For tiny graphs, use dense eigendecomposition; for larger ones, sparse methods are fine

---

## 6. Implement the actual methods

### A. `eigengap`

- Compute the smallest eigenvalues up to `k_max + 1`
- Score each feasible `k` by `gap(k) = λ_{k+1} - λ_k`
- Pick the best `k`
- Prefer smaller `k` on ties

### B. `silhouette`

- For each feasible `k`:
  - compute spectral embedding
  - run k-means with multiple restarts
  - compute mean silhouette
- choose best `k`
- prefer smaller `k` on ties / weak improvements

### C. `eigengap_silhouette_hybrid`

- Use eigengap to propose `k_eig`
- Evaluate only a local neighborhood around `k_eig`
- pick best silhouette in that neighborhood
- if silhouette is too weak, shrink toward smaller `k`

### D. `stability`

- For each feasible `k`:
  - repeat clustering under seed / mild graph perturbation changes
  - compute mean pairwise ARI or NMI across runs
- choose most stable `k`
- optionally combine stability with silhouette

### E. `bic_xmeans_like` (optional)

- Embed graph spectrally into vectors
- Fit k-means / GMM variants across `k`
- choose by BIC-like score
- keep this optional, not default

---

## 7. Integrate into the new clustering method only

Wherever the new k-means-based clustering path is called:

- if manual `num_clusters` is explicitly provided and `auto_k_method == none`, keep old behavior
- if `auto_k_method != none`, call the selector first
- then run the clustering method with the selected `k`

Pseudo-flow:

```python
if auto_k_method != "none":
    selection = choose_num_clusters_for_affinity_graph(...)
    k = selection["k"]
else:
    k = manual_num_clusters

labels = run_new_kmeans_based_graph_clustering(affinity, k, ...)
```

---

## 8. Logging and diagnostics are mandatory

For each candidate, optionally log:

- candidate size `m`
- feasible `k` range
- chosen `k`
- eigengap scores
- silhouette scores
- stability scores if used
- whether fallback was triggered

This is important for research debugging.

---

## 9. Recommended defaults

Use these defaults initially:

```text
auto_k_method = eigengap_silhouette_hybrid
auto_k_min = 1
auto_k_max = 8
auto_k_min_cluster_size = 1
auto_k_max_cluster_size = 0
auto_k_num_restarts = 10
auto_k_tiny_graph_threshold = 4
auto_k_weak_silhouette_threshold = 0.05
auto_k_prefer_smaller_k = true
```

If we later want stronger perturbation-size control, introduce:

```text
auto_k_max_cluster_size > 0
```

because this directly matches the “avoid too-large perturbation cluster” research goal.

---

## 10. Experimental recommendation

Please implement the following comparison modes so we can evaluate them later:

1. `manual_k`
2. `eigengap`
3. `silhouette`
4. `eigengap_silhouette_hybrid`
5. `stability` (optional if time allows)

For evaluation, save per-candidate:

- chosen `k`
- cluster size distribution
- runtime
- downstream influence approximation metrics

---

## Final recommendation to Codex

Implement **automatic `k` selection for the new k-means-based graph clustering path** using a **graph-first design**:

- keep current code untouched
- add a reusable `choose_num_clusters_for_affinity_graph(...)`
- make `eigengap_silhouette_hybrid` the default
- keep `silhouette` and `eigengap` as independent baselines
- optionally add `stability` and `bic_xmeans_like`
- do **not** rely on elbow as the main automatic method

The elbow criterion is known to be ambiguous and unreliable, while silhouette, information-criterion approaches, X-means, and the gap statistic are standard alternatives discussed in the uploaded reference. For our graph setting, however, eigengap + silhouette is the most natural starting point. fileciteturn4file0
