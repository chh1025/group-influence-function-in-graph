import networkx as nx
import numpy as np
import pymetis
from sklearn.cluster import SpectralClustering


def partition_affinity_graph(affinity, method, num_clusters, seed=0):
    affinity = np.asarray(affinity, dtype=np.float64)
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError("affinity must be a square 2D matrix.")

    num_nodes = int(affinity.shape[0])
    if num_nodes == 0:
        raise ValueError("affinity graph must contain at least one node.")

    method = str(method).strip().lower()
    effective_k = max(1, min(int(num_clusters), num_nodes))

    if effective_k == 1:
        groups = [list(range(num_nodes))]
        return groups, _build_partition_stats(affinity, groups, method, runtime_meta={})

    if method == "metis":
        groups, runtime_meta = _partition_with_metis(affinity=affinity, num_clusters=effective_k)
    elif method == "spectral":
        groups, runtime_meta = _partition_with_spectral(affinity=affinity, num_clusters=effective_k, seed=seed)
    elif method == "local_ppr":
        groups, runtime_meta = _partition_with_local_ppr(affinity=affinity, num_clusters=effective_k, seed=seed)
    else:
        raise ValueError(f"Unknown partition method: {method}")

    return groups, _build_partition_stats(affinity, groups, method, runtime_meta=runtime_meta)


def _partition_with_metis(affinity, num_clusters):
    if np.count_nonzero(np.triu(affinity > 0, k=1)) == 0:
        return _balanced_partition(num_nodes=affinity.shape[0], num_clusters=num_clusters), {"metis_cutcount": 0}

    xadj = [0]
    adjncy = []
    eweights = []
    for row_idx in range(affinity.shape[0]):
        neighbors = np.where(affinity[row_idx] > 0)[0].tolist()
        neighbors.sort()
        for col_idx in neighbors:
            adjncy.append(int(col_idx))
            eweights.append(max(1, int(round(float(affinity[row_idx, col_idx]) * 1000.0))))
        xadj.append(len(adjncy))

    cutcount, membership = pymetis.part_graph(
        num_clusters,
        xadj=xadj,
        adjncy=adjncy,
        eweights=eweights,
        contiguous=False,
    )
    groups = _membership_to_groups(membership, num_clusters)
    return groups, {"metis_cutcount": int(cutcount)}


def _partition_with_spectral(affinity, num_clusters, seed):
    if np.count_nonzero(np.triu(affinity > 0, k=1)) == 0:
        return _balanced_partition(num_nodes=affinity.shape[0], num_clusters=num_clusters), {}

    spectral_affinity = affinity.copy()
    np.fill_diagonal(spectral_affinity, 1.0)
    estimator = SpectralClustering(
        n_clusters=num_clusters,
        random_state=int(seed),
        affinity="precomputed",
        assign_labels="kmeans",
        n_init=10,
    )
    labels = estimator.fit_predict(spectral_affinity)
    return _membership_to_groups(labels, num_clusters), {}


def _partition_with_local_ppr(affinity, num_clusters, seed):
    if np.count_nonzero(np.triu(affinity > 0, k=1)) == 0:
        return _balanced_partition(num_nodes=affinity.shape[0], num_clusters=num_clusters), {}

    graph = nx.from_numpy_array(affinity)
    remaining = set(range(affinity.shape[0]))
    target_sizes = _balanced_cluster_sizes(num_nodes=affinity.shape[0], num_clusters=num_clusters)
    groups = []
    seed = int(seed)

    for cluster_idx, target_size in enumerate(target_sizes):
        if not remaining:
            break
        if cluster_idx == len(target_sizes) - 1:
            groups.append(sorted(remaining))
            remaining.clear()
            break

        sub_nodes = sorted(remaining)
        subgraph = graph.subgraph(sub_nodes).copy()
        if subgraph.number_of_edges() == 0:
            selected = sub_nodes[:target_size]
        else:
            local_seed = _select_local_seed(subgraph=subgraph, seed=seed + cluster_idx)
            personalization = {node: 0.0 for node in subgraph.nodes}
            personalization[local_seed] = 1.0
            pagerank_scores = nx.pagerank(subgraph, alpha=0.85, personalization=personalization, weight="weight")
            weighted_degrees = dict(subgraph.degree(weight="weight"))
            selected = sorted(
                sub_nodes,
                key=lambda node: (
                    float(pagerank_scores.get(node, 0.0)),
                    float(weighted_degrees.get(node, 0.0)),
                    -int(node),
                ),
                reverse=True,
            )[:target_size]

        selected = sorted(selected)
        groups.append(selected)
        remaining.difference_update(selected)

    if remaining:
        groups.append(sorted(remaining))

    return groups, {}


def _select_local_seed(subgraph, seed):
    weighted_degrees = dict(subgraph.degree(weight="weight"))
    candidates = sorted(
        subgraph.nodes,
        key=lambda node: (
            float(weighted_degrees.get(node, 0.0)),
            -int(node),
        ),
        reverse=True,
    )
    if not candidates:
        raise ValueError("subgraph must contain at least one node.")

    if len(candidates) == 1:
        return candidates[0]

    seed_offset = int(seed) % len(candidates)
    return candidates[seed_offset]


def _membership_to_groups(membership, num_clusters):
    groups = [[] for _ in range(int(num_clusters))]
    for node_idx, cluster_idx in enumerate(membership):
        groups[int(cluster_idx)].append(int(node_idx))
    groups = [sorted(group) for group in groups if len(group) > 0]
    return groups


def _balanced_partition(num_nodes, num_clusters):
    target_sizes = _balanced_cluster_sizes(num_nodes=num_nodes, num_clusters=num_clusters)
    groups = []
    start = 0
    for target_size in target_sizes:
        end = min(num_nodes, start + target_size)
        groups.append(list(range(start, end)))
        start = end
    return [group for group in groups if len(group) > 0]


def _balanced_cluster_sizes(num_nodes, num_clusters):
    base_size = int(num_nodes) // int(num_clusters)
    remainder = int(num_nodes) % int(num_clusters)
    sizes = []
    for idx in range(int(num_clusters)):
        sizes.append(base_size + (1 if idx < remainder else 0))
    return [size for size in sizes if size > 0]


def _build_partition_stats(affinity, groups, method, runtime_meta):
    group_lookup = {}
    for group_idx, group in enumerate(groups):
        for node_idx in group:
            group_lookup[int(node_idx)] = int(group_idx)

    weighted_cut = 0.0
    for i in range(affinity.shape[0]):
        for j in range(i + 1, affinity.shape[0]):
            if group_lookup.get(i) != group_lookup.get(j):
                weighted_cut += float(affinity[i, j])

    return {
        "partition_method": str(method),
        "num_groups": int(len(groups)),
        "cluster_sizes": [int(len(group)) for group in groups],
        "weighted_cut": float(weighted_cut),
        "num_affinity_nodes": int(affinity.shape[0]),
        "num_affinity_edges": int(np.count_nonzero(np.triu(affinity > 0, k=1))),
        "affinity_density": _affinity_density(affinity.shape[0], np.count_nonzero(np.triu(affinity > 0, k=1))),
        "affinity_weight_sum": float(np.triu(affinity, k=1).sum()),
        **runtime_meta,
    }


def _affinity_density(num_nodes, num_edges):
    if int(num_nodes) <= 1:
        return 0.0
    return float((2.0 * float(num_edges)) / float(int(num_nodes) * (int(num_nodes) - 1)))
