from collections import deque

import numpy as np
import torch


def build_candidate_affinity_graph(candidate, graph, args):
    candidate_cpu = candidate.detach().cpu().to(torch.long)
    if candidate_cpu.dim() != 2 or candidate_cpu.shape[1] != 2:
        raise ValueError("candidate must have shape [num_edges, 2].")

    num_candidates = int(candidate_cpu.shape[0])
    if num_candidates == 0:
        raise ValueError("candidate must contain at least one edge.")

    shared_endpoint_bonus = float(getattr(args, "partition_shared_endpoint_bonus", 1.0))
    distance_scale = float(getattr(args, "partition_distance_scale", 1.0))
    max_hops = int(getattr(args, "partition_distance_max_hops", 3))
    min_weight = float(getattr(args, "partition_min_weight", 1e-8))

    adjacency = _build_undirected_adjacency(graph)
    endpoint_nodes = sorted({int(v) for v in candidate_cpu.reshape(-1).tolist()})
    distance_cache = {
        node: _bounded_bfs_distances(adjacency, start=node, max_hops=max_hops)
        for node in endpoint_nodes
    }

    affinity = np.zeros((num_candidates, num_candidates), dtype=np.float64)
    for i in range(num_candidates):
        edge_i = tuple(int(v) for v in candidate_cpu[i].tolist())
        for j in range(i + 1, num_candidates):
            edge_j = tuple(int(v) for v in candidate_cpu[j].tolist())
            weight = _pair_affinity(
                edge_i=edge_i,
                edge_j=edge_j,
                distance_cache=distance_cache,
                shared_endpoint_bonus=shared_endpoint_bonus,
                distance_scale=distance_scale,
            )
            if weight >= min_weight:
                affinity[i, j] = weight
                affinity[j, i] = weight

    positive_pairs = int(np.count_nonzero(np.triu(affinity > 0, k=1)))
    density = 0.0
    if num_candidates > 1:
        density = float((2.0 * positive_pairs) / float(num_candidates * (num_candidates - 1)))

    return {
        "affinity": affinity,
        "num_nodes": num_candidates,
        "num_edges": positive_pairs,
        "density": density,
        "weight_sum": float(np.triu(affinity, k=1).sum()),
        "shared_endpoint_bonus": shared_endpoint_bonus,
        "distance_scale": distance_scale,
        "distance_max_hops": max_hops,
    }


def _build_undirected_adjacency(graph):
    num_nodes = int(graph.num_nodes)
    adjacency = [set() for _ in range(num_nodes)]
    edge_index = graph.edge_index.detach().cpu().to(torch.long)
    for u, v in edge_index.t().tolist():
        u = int(u)
        v = int(v)
        if u == v:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)
    return [sorted(neighbors) for neighbors in adjacency]


def _bounded_bfs_distances(adjacency, start, max_hops):
    distances = {int(start): 0}
    queue = deque([int(start)])

    while queue:
        node = queue.popleft()
        current_dist = distances[node]
        if current_dist >= max_hops:
            continue
        for neighbor in adjacency[node]:
            if neighbor in distances:
                continue
            distances[neighbor] = current_dist + 1
            queue.append(neighbor)

    return distances


def _pair_affinity(edge_i, edge_j, distance_cache, shared_endpoint_bonus, distance_scale):
    endpoints_i = {int(edge_i[0]), int(edge_i[1])}
    endpoints_j = {int(edge_j[0]), int(edge_j[1])}

    shared_count = len(endpoints_i.intersection(endpoints_j))
    weight = float(shared_count) * shared_endpoint_bonus

    min_dist = None
    for src in endpoints_i:
        src_distances = distance_cache.get(src, {})
        for dst in endpoints_j:
            if dst not in src_distances:
                continue
            candidate_dist = int(src_distances[dst])
            if min_dist is None or candidate_dist < min_dist:
                min_dist = candidate_dist

    if min_dist is not None:
        weight += distance_scale / float(1 + min_dist)

    return float(weight)
