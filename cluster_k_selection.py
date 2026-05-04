import math
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


SUPPORTED_AUTO_K_METHODS = {
    "none",
    "eigengap",
    "silhouette",
    "eigengap_silhouette_hybrid",
    "stability",
    "bic_xmeans_like",
}


def choose_num_clusters_for_affinity_graph(
    affinity,
    method,
    k_min,
    k_max,
    max_ratio=1.0,
    min_cluster_size=1,
    max_cluster_size=None,
    tiny_graph_threshold=4,
    num_restarts=10,
    random_state=0,
    silhouette_metric="euclidean",
    stability_trials=10,
    stability_edge_dropout=0.05,
    weak_silhouette_threshold=0.05,
    prefer_smaller_k=True,
):
    """
    Select k for a weighted affinity graph and return diagnostics.

    The selector is intentionally graph-first: eigengap is computed on the
    normalized Laplacian and silhouette/BIC/stability operate on spectral
    embeddings derived from the same affinity graph.
    """
    method = str(method).strip().lower()
    if method not in SUPPORTED_AUTO_K_METHODS:
        raise ValueError(f"Unknown auto_k_method={method}. Choose from {sorted(SUPPORTED_AUTO_K_METHODS)}.")

    W = _symmetrize_affinity(affinity)
    num_nodes = int(W.shape[0])
    feasible = _feasible_k_values(
        num_nodes=num_nodes,
        k_min=k_min,
        k_max=k_max,
        max_ratio=max_ratio,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
    )
    diagnostics = {
        "method": method,
        "num_nodes": num_nodes,
        "feasible_k_min": int(feasible[0]) if feasible else None,
        "feasible_k_max": int(feasible[-1]) if feasible else None,
        "feasible_k_values": [int(k) for k in feasible],
        "fallback_reason": None,
        "eigengap_scores": {},
        "silhouette_scores": {},
        "stability_scores": {},
        "bic_scores": {},
        "eigenvalues": [],
    }

    if method == "none":
        selected = max(1, min(int(k_max), max(1, num_nodes)))
        diagnostics["fallback_reason"] = "manual_k_path"
        return {"k": int(selected), "scores": {}, "diagnostics": diagnostics, "method": method}

    if num_nodes <= 0:
        diagnostics["fallback_reason"] = "empty_graph"
        return {"k": 1, "scores": {}, "diagnostics": diagnostics, "method": method}

    if not feasible:
        diagnostics["fallback_reason"] = "empty_feasible_range"
        return {"k": 1, "scores": {}, "diagnostics": diagnostics, "method": method}

    if num_nodes <= int(tiny_graph_threshold):
        selected = int(feasible[0]) if bool(prefer_smaller_k) else int(feasible[-1])
        diagnostics["fallback_reason"] = "tiny_graph"
        return {"k": int(selected), "scores": {}, "diagnostics": diagnostics, "method": method}

    if len(feasible) == 1:
        return {"k": int(feasible[0]), "scores": {}, "diagnostics": diagnostics, "method": method}

    max_k = int(feasible[-1])
    spectral = _spectral_cache(W, max_k=max_k)
    diagnostics["eigenvalues"] = [float(v) for v in spectral["eigenvalues"][: max_k + 1]]

    if method == "eigengap":
        eig_scores = eigengap_scores_from_eigenvalues(spectral["eigenvalues"], feasible)
        selected = _select_best_score(eig_scores, prefer_smaller_k=prefer_smaller_k)
        diagnostics["eigengap_scores"] = _stringify_scores(eig_scores)
        return {"k": int(selected), "scores": eig_scores, "diagnostics": diagnostics, "method": method}

    if method == "silhouette":
        sil_scores = silhouette_scores_for_k(
            spectral["embedding"],
            feasible,
            num_restarts=num_restarts,
            random_state=random_state,
            metric=silhouette_metric,
        )
        selected = _select_with_weak_silhouette_fallback(
            scores=sil_scores,
            feasible=feasible,
            weak_threshold=weak_silhouette_threshold,
            prefer_smaller_k=prefer_smaller_k,
        )
        diagnostics["silhouette_scores"] = _stringify_scores(sil_scores)
        return {"k": int(selected), "scores": sil_scores, "diagnostics": diagnostics, "method": method}

    if method == "eigengap_silhouette_hybrid":
        eig_scores = eigengap_scores_from_eigenvalues(spectral["eigenvalues"], feasible)
        eig_k = _select_best_score(eig_scores, prefer_smaller_k=prefer_smaller_k)
        neighborhood = [k for k in feasible if abs(int(k) - int(eig_k)) <= 1]
        if 1 in feasible and 1 not in neighborhood:
            neighborhood = [1] + neighborhood
        sil_scores = silhouette_scores_for_k(
            spectral["embedding"],
            neighborhood,
            num_restarts=num_restarts,
            random_state=random_state,
            metric=silhouette_metric,
        )
        selected = _select_with_weak_silhouette_fallback(
            scores=sil_scores,
            feasible=neighborhood,
            weak_threshold=weak_silhouette_threshold,
            prefer_smaller_k=prefer_smaller_k,
        )
        diagnostics["eigengap_selected_k"] = int(eig_k)
        diagnostics["eigengap_scores"] = _stringify_scores(eig_scores)
        diagnostics["silhouette_scores"] = _stringify_scores(sil_scores)
        return {"k": int(selected), "scores": sil_scores, "diagnostics": diagnostics, "method": method}

    if method == "stability":
        stability_scores = stability_scores_for_k(
            W,
            feasible,
            num_restarts=num_restarts,
            random_state=random_state,
            trials=stability_trials,
            edge_dropout=stability_edge_dropout,
        )
        selected = _select_best_score(stability_scores, prefer_smaller_k=prefer_smaller_k)
        diagnostics["stability_scores"] = _stringify_scores(stability_scores)
        return {"k": int(selected), "scores": stability_scores, "diagnostics": diagnostics, "method": method}

    if method == "bic_xmeans_like":
        bic_scores = bic_xmeans_like_scores_for_k(
            spectral["embedding"],
            feasible,
            num_restarts=num_restarts,
            random_state=random_state,
        )
        selected = _select_best_score(bic_scores, prefer_smaller_k=prefer_smaller_k, higher_is_better=False)
        diagnostics["bic_scores"] = _stringify_scores(bic_scores)
        return {"k": int(selected), "scores": bic_scores, "diagnostics": diagnostics, "method": method}

    raise AssertionError(f"Unhandled auto_k_method={method}")


def normalized_laplacian_from_affinity(affinity):
    W = _symmetrize_affinity(affinity)
    if sparse.issparse(W):
        return csgraph.laplacian(W, normed=True).astype(np.float64)

    degree = W.sum(axis=1)
    inv_sqrt = np.zeros_like(degree, dtype=np.float64)
    nonzero = degree > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    normalized_adj = inv_sqrt[:, None] * W * inv_sqrt[None, :]
    return np.eye(W.shape[0], dtype=np.float64) - normalized_adj


def spectral_embedding_from_affinity(affinity, dim):
    W = _symmetrize_affinity(affinity)
    dim = max(1, min(int(dim), max(1, int(W.shape[0]) - 1)))
    spectral = _spectral_cache(W, max_k=dim)
    return spectral["embedding"][:, :dim]


def eigengap_scores(affinity, k_min, k_max):
    W = _symmetrize_affinity(affinity)
    feasible = list(range(max(1, int(k_min)), min(int(k_max), int(W.shape[0])) + 1))
    spectral = _spectral_cache(W, max_k=max(feasible))
    return eigengap_scores_from_eigenvalues(spectral["eigenvalues"], feasible)


def eigengap_scores_from_eigenvalues(eigenvalues, feasible):
    values = np.asarray(eigenvalues, dtype=np.float64)
    scores = {}
    for k in feasible:
        k = int(k)
        if k <= 0 or k >= values.shape[0]:
            scores[k] = float("-inf")
            continue
        gap = float(values[k] - values[k - 1])
        scores[k] = gap if math.isfinite(gap) else float("-inf")
    return scores


def silhouette_scores_for_k(embedding, feasible, num_restarts, random_state, metric):
    embedding = np.asarray(embedding, dtype=np.float64)
    scores = {}
    for k in feasible:
        k = int(k)
        if k <= 1:
            scores[k] = 0.0
            continue
        if k >= embedding.shape[0]:
            scores[k] = float("-inf")
            continue

        labels = _kmeans_labels(
            embedding[:, : min(k, embedding.shape[1])],
            k,
            num_restarts=num_restarts,
            random_state=random_state + k,
        )
        if len(set(labels.tolist())) <= 1:
            scores[k] = float("-inf")
            continue
        try:
            scores[k] = float(silhouette_score(embedding[:, : min(k, embedding.shape[1])], labels, metric=metric))
        except ValueError:
            scores[k] = float("-inf")
    return scores


def stability_scores_for_k(affinity, feasible, num_restarts, random_state, trials, edge_dropout):
    W = _symmetrize_affinity(affinity)
    rng = np.random.default_rng(int(random_state))
    scores = {}
    for k in feasible:
        k = int(k)
        if k <= 1:
            scores[k] = 0.0
            continue

        labelings = []
        for trial_idx in range(max(2, int(trials))):
            perturbed = _drop_affinity_edges(W, rng=rng, edge_dropout=edge_dropout)
            spectral = _spectral_cache(perturbed, max_k=k)
            labels = _kmeans_labels(
                spectral["embedding"][:, : min(k, spectral["embedding"].shape[1])],
                k,
                num_restarts=num_restarts,
                random_state=random_state + 1009 * trial_idx + k,
            )
            labelings.append(labels)

        pair_scores = []
        for i in range(len(labelings)):
            for j in range(i + 1, len(labelings)):
                pair_scores.append(float(adjusted_rand_score(labelings[i], labelings[j])))
        scores[k] = float(np.mean(pair_scores)) if pair_scores else float("-inf")
    return scores


def bic_xmeans_like_scores_for_k(embedding, feasible, num_restarts, random_state):
    embedding = np.asarray(embedding, dtype=np.float64)
    n, dim = embedding.shape
    scores = {}
    for k in feasible:
        k = int(k)
        if k <= 1:
            centered = embedding - embedding.mean(axis=0, keepdims=True)
            inertia = float(np.square(centered).sum())
        else:
            estimator = KMeans(
                n_clusters=k,
                n_init=max(1, int(num_restarts)),
                random_state=int(random_state) + k,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(embedding[:, : min(k, dim)])
            inertia = float(estimator.inertia_)
        variance = max(inertia / max(1, n * max(1, min(k, dim))), 1e-12)
        params = k * (min(k, dim) + 1)
        scores[k] = float(n * math.log(variance) + params * math.log(max(2, n)))
    return scores


def _feasible_k_values(num_nodes, k_min, k_max, max_ratio, min_cluster_size, max_cluster_size):
    if num_nodes <= 0:
        return []

    ratio_cap = int(math.floor(float(max_ratio) * float(num_nodes))) if float(max_ratio) > 0 else int(num_nodes)
    k_lo = max(1, int(k_min))
    k_hi = min(int(k_max), max(1, ratio_cap), int(num_nodes))

    min_cluster_size = max(1, int(min_cluster_size))
    if min_cluster_size > 1:
        k_hi = min(k_hi, int(num_nodes) // min_cluster_size)

    if max_cluster_size is not None and int(max_cluster_size) > 0:
        k_lo = max(k_lo, int(math.ceil(float(num_nodes) / float(max_cluster_size))))

    if k_lo > k_hi:
        return []
    return list(range(k_lo, k_hi + 1))


def _symmetrize_affinity(affinity):
    if sparse.issparse(affinity):
        W = affinity.astype(np.float64).tocsr()
        W = 0.5 * (W + W.T)
        W.setdiag(0.0)
        W.eliminate_zeros()
        return W

    W = np.asarray(affinity, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("affinity must be a square matrix.")
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    W = np.maximum(W, 0.0)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def _spectral_cache(affinity, max_k):
    W = _symmetrize_affinity(affinity)
    num_nodes = int(W.shape[0])
    eig_count = max(1, min(num_nodes, int(max_k) + 1))
    W_for_laplacian = W
    if not sparse.issparse(W_for_laplacian) and num_nodes > 256:
        W_for_laplacian = sparse.csr_matrix(W_for_laplacian)
        W_for_laplacian.eliminate_zeros()
    L = normalized_laplacian_from_affinity(W_for_laplacian)

    if num_nodes <= 2 or eig_count >= num_nodes or num_nodes <= 256:
        dense_L = L.toarray() if sparse.issparse(L) else np.asarray(L, dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(dense_L)
    else:
        try:
            eigenvalues, eigenvectors = eigsh(
                L,
                k=eig_count,
                which="SM",
                tol=1e-4,
                maxiter=max(1000, 20 * num_nodes),
            )
        except Exception:
            if num_nodes > 1500:
                raise
            dense_L = L.toarray() if sparse.issparse(L) else np.asarray(L, dtype=np.float64)
            eigenvalues, eigenvectors = np.linalg.eigh(dense_L)

    order = np.argsort(eigenvalues)
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    embedding = eigenvectors[:, : max(1, min(int(max_k), eigenvectors.shape[1]))]
    embedding = _row_normalize(embedding)
    return {"eigenvalues": eigenvalues, "embedding": embedding}


def _row_normalize(values):
    values = np.asarray(values, dtype=np.float64)
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    denom[denom <= 1e-12] = 1.0
    return values / denom


def _kmeans_labels(values, k, num_restarts, random_state):
    estimator = KMeans(
        n_clusters=int(k),
        n_init=max(1, int(num_restarts)),
        random_state=int(random_state),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit_predict(np.asarray(values, dtype=np.float64))


def _select_best_score(scores, prefer_smaller_k=True, higher_is_better=True):
    finite_items = [(int(k), float(v)) for k, v in scores.items() if math.isfinite(float(v))]
    if not finite_items:
        return min(int(k) for k in scores.keys()) if scores else 1
    if higher_is_better:
        best_value = max(value for _, value in finite_items)
    else:
        best_value = min(value for _, value in finite_items)
    tolerance = 1e-9
    tied = [k for k, value in finite_items if abs(value - best_value) <= tolerance]
    return min(tied) if bool(prefer_smaller_k) else max(tied)


def _select_with_weak_silhouette_fallback(scores, feasible, weak_threshold, prefer_smaller_k):
    selected = _select_best_score(scores, prefer_smaller_k=prefer_smaller_k)
    best_score = float(scores.get(int(selected), float("-inf")))
    if best_score < float(weak_threshold):
        return int(min(feasible) if bool(prefer_smaller_k) else max(feasible))
    return int(selected)


def _stringify_scores(scores):
    return {str(int(k)): (float(v) if math.isfinite(float(v)) else None) for k, v in scores.items()}


def _drop_affinity_edges(affinity, rng, edge_dropout):
    W = _symmetrize_affinity(affinity)
    dropout = max(0.0, min(1.0, float(edge_dropout)))
    if dropout <= 0.0:
        return W

    if sparse.issparse(W):
        coo = sparse.triu(W, k=1).tocoo()
        keep = rng.random(coo.data.shape[0]) >= dropout
        upper = sparse.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=W.shape).tocsr()
        return upper + upper.T

    upper = np.triu(W, k=1)
    mask = rng.random(upper.shape) >= dropout
    upper = upper * np.triu(mask, k=1)
    return upper + upper.T
