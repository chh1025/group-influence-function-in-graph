import numpy as np
import pymetis
import torch


def get_or_build_training_graph_metis_partition(graph, num_parts, cache=None):
    num_nodes = int(graph.num_nodes)
    effective_parts = max(1, min(int(num_parts), num_nodes))

    cache_store = cache if cache is not None else getattr(graph, "_training_graph_metis_partition_cache", None)
    if cache_store is None:
        cache_store = {}

    if effective_parts in cache_store:
        return np.asarray(cache_store[effective_parts], dtype=np.int64), True

    if effective_parts == 1:
        node_part = np.zeros((num_nodes,), dtype=np.int64)
    else:
        xadj, adjncy = _build_undirected_metis_adjacency(graph)
        _, membership = pymetis.part_graph(
            effective_parts,
            xadj=xadj,
            adjncy=adjncy,
            contiguous=False,
        )
        node_part = np.asarray(membership, dtype=np.int64)

    cache_store[effective_parts] = node_part
    if cache is None:
        graph._training_graph_metis_partition_cache = cache_store
    return node_part, False


def assign_candidate_edge_owner_partitions(candidate_edges, node_part):
    edge_tensor = _as_cpu_edge_tensor(candidate_edges)
    owners = []
    cross_partition_edges = 0

    for u, v in edge_tensor.tolist():
        owner_part, is_cross_partition = assign_edge_owner_partition(int(u), int(v), node_part)
        owners.append(int(owner_part))
        if is_cross_partition:
            cross_partition_edges += 1

    return owners, int(cross_partition_edges)


def assign_edge_owner_partition(u, v, node_part):
    part_u = int(node_part[int(u)])
    part_v = int(node_part[int(v)])
    if part_u == part_v:
        return part_u, False
    return min(part_u, part_v), True


def owner_partition_histogram(owner_partitions):
    histogram = {}
    for owner in owner_partitions:
        owner = int(owner)
        histogram[owner] = histogram.get(owner, 0) + 1
    return dict(sorted(histogram.items()))


def _build_undirected_metis_adjacency(graph):
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

    xadj = [0]
    adjncy = []
    for node in range(num_nodes):
        neighbors = sorted(adjacency[node])
        adjncy.extend(neighbors)
        xadj.append(len(adjncy))

    return xadj, adjncy


def _as_cpu_edge_tensor(candidate_edges):
    edge_tensor = candidate_edges.detach().cpu().to(torch.long)
    if edge_tensor.dim() == 1 and edge_tensor.numel() == 2:
        edge_tensor = edge_tensor.view(1, 2)
    if edge_tensor.dim() != 2 or edge_tensor.shape[1] != 2:
        raise ValueError("candidate_edges must have shape [num_edges, 2].")
    return edge_tensor
