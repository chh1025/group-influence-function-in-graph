import sys
import torch
import numpy as np
import itertools
import csv
import hashlib
from matplotlib import pyplot as plt
import os
from os import path as osp
import json
from collections import deque
from pathlib import Path
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
import shutil
from src.graph_utils import find_k_hop_neighbors_bfs, remove_edge, add_edge

def get_eval_node_idxs(graph, eval_metric, seed):
    set_seed(seed)
    if eval_metric in ['feature_ablation']:
        eval_node_idxs = torch.randperm(graph.x.shape[0])[:50].tolist()
    else:
        eval_node_idxs = [i for i in range(graph.x.shape[0])]

    return eval_node_idxs


def lower_is_better(args):
    if args.eval_metric in ['mean_validation_loss']:
        return True
    elif args.eval_metric in ['feature_ablation', 'dirichlet_energy']:
        return False
    else:
        raise ValueError


def get_edge_removal_candidates(graph, num_candidates):
    # Set all edges as removal candidates
    if graph.edge_index.numel()/4 > num_candidates:
        edges = graph.edge_index.T
        sorted_edges = torch.sort(edges, dim=1)[0]
        unique_edges = torch.unique(sorted_edges, dim=0)
        random_idxs = torch.randperm(unique_edges.shape[0])[:num_candidates]
        unique_edges = unique_edges[random_idxs]
    else:
        edges = graph.edge_index.T
        sorted_edges = torch.sort(edges, dim=1)[0]
        unique_edges = torch.unique(sorted_edges, dim=0)

    return unique_edges


def get_edge_insertion_candidates(graph, num_candidates):
    # Set {num_candidates} random edges as insertion candidates
    device = "cuda" if torch.cuda.is_available() else "cpu"

    candidate_nodes = torch.arange(0, graph.num_nodes).to(device)
    all_edges = torch.combinations(candidate_nodes, r=2)
    random_idxs = torch.randperm(all_edges.shape[0])[:num_candidates]
    candidate_edges = all_edges[random_idxs].T

    if candidate_edges.numel() == 0:
        return candidate_edges
    
    existing_edges = graph.edge_index

    existing_edges = existing_edges.sort(dim=0)[0]
    candidate_edges = candidate_edges.sort(dim=0)[0]

    mask = (candidate_edges[:, :, None] == existing_edges[:, None, :]).all(dim=0).any(dim=1)
    
    edge_insertion_candidates = candidate_edges[:, ~mask]

    return edge_insertion_candidates.T


def _get_unique_undirected_edges(graph):
    edges = graph.edge_index.T
    sorted_edges = torch.sort(edges, dim=1)[0]
    unique_edges = torch.unique(sorted_edges, dim=0)
    return unique_edges


def _build_k_hop_reachability_sets(graph, k, device="cpu"):
    """Build k-hop reachability sets per node on the specified device.

    Default device is CPU to avoid holding large neighbor tensors on GPU.
    """
    if k < 0:
        raise ValueError("k must be non-negative.")

    if k == 0:
        return [set([i]) for i in range(graph.num_nodes)]

    k_hop_neighbors = find_k_hop_neighbors_bfs(graph, k, device=device)
    reachability_sets = []
    for node_idx in range(graph.num_nodes):
        neighbors = k_hop_neighbors[node_idx].detach().cpu().tolist()
        reachability_sets.append(set(neighbors))

    return reachability_sets


def _edge_pair_is_within_distance(edge_a, edge_b, reachability_sets):
    a0, a1 = int(edge_a[0].item()), int(edge_a[1].item())
    b0, b1 = int(edge_b[0].item()), int(edge_b[1].item())

    a0_reach = reachability_sets[a0]
    a1_reach = reachability_sets[a1]
    return (b0 in a0_reach) or (b1 in a0_reach) or (b0 in a1_reach) or (b1 in a1_reach)


def _sample_connected_edge_group(unique_edges_cpu, num_group_elem):
    """Sample a connected edge group by growing from a seed edge.

    This is used for `group_neighbor` with `removal_neighbor_dist == 1`.
    The intended constraint for distance 1 is edge connectivity through a
    shared endpoint, not pairwise all-to-all adjacency among all sampled edges.
    """
    num_edges = unique_edges_cpu.shape[0]
    if num_group_elem <= 0:
        raise ValueError("num_group_elem must be positive.")
    if num_group_elem > num_edges:
        return None

    edge_pairs = [(int(edge[0].item()), int(edge[1].item())) for edge in unique_edges_cpu]
    node_to_edge_idxs = {}
    for edge_idx, (src, dst) in enumerate(edge_pairs):
        node_to_edge_idxs.setdefault(src, []).append(edge_idx)
        node_to_edge_idxs.setdefault(dst, []).append(edge_idx)

    seed_edge_idx = torch.randint(0, num_edges, (1,)).item()
    selected_idxs = {seed_edge_idx}
    frontier_idxs = set()

    for node in edge_pairs[seed_edge_idx]:
        frontier_idxs.update(node_to_edge_idxs[node])
    frontier_idxs.discard(seed_edge_idx)

    while len(selected_idxs) < num_group_elem:
        if not frontier_idxs:
            return None

        frontier_list = tuple(frontier_idxs)
        next_pos = torch.randint(0, len(frontier_list), (1,)).item()
        next_edge_idx = frontier_list[next_pos]
        frontier_idxs.remove(next_edge_idx)
        if next_edge_idx in selected_idxs:
            continue

        selected_idxs.add(next_edge_idx)
        for node in edge_pairs[next_edge_idx]:
            frontier_idxs.update(node_to_edge_idxs[node])
        frontier_idxs.difference_update(selected_idxs)

    ordered_idxs = sorted(selected_idxs)
    return unique_edges_cpu[ordered_idxs]


def _edge_is_valid_against_previous_clusters(edge, built_clusters, inter_forbidden_reachability_sets):
    if inter_forbidden_reachability_sets is None:
        return True

    for prev_cluster in built_clusters:
        for prev_edge in prev_cluster:
            if _edge_pair_is_within_distance(edge, prev_edge, inter_forbidden_reachability_sets):
                return False
    return True


def _flag_is_enabled(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _collect_pool_edge_indices_for_nodes(node_indices, incident_edge_idxs_by_node, allowed_edge_idxs=None):
    pool_edge_idxs = set()
    for node_idx in node_indices:
        for edge_idx in incident_edge_idxs_by_node[node_idx]:
            if allowed_edge_idxs is not None and edge_idx not in allowed_edge_idxs:
                continue
            pool_edge_idxs.add(edge_idx)
    return pool_edge_idxs


def _build_cluster_cache_metadata(graph, args):
    unique_edges = _get_unique_undirected_edges(graph).detach().cpu().to(torch.long)
    payload = unique_edges.reshape(-1).numpy().tobytes()
    graph_hash = hashlib.sha1(payload).hexdigest()[:16]
    return {
        "dataset": str(getattr(args, "dataset", "unknown")).lower(),
        "num_nodes": int(graph.num_nodes),
        "num_unique_edges": int(unique_edges.shape[0]),
        "graph_hash": graph_hash,
        "num_of_clusters": int(getattr(args, "num_of_clusters", 1)),
        "edges_per_cluster": int(getattr(args, "edges_per_cluster", -1)),
        "cluster_ratio_percent": int(getattr(args, "cluster_ratio_percent", -1)),
        "intra_cluster_dist": int(getattr(args, "intra_cluster_dist", -1)),
        "inter_cluster_dist": int(getattr(args, "inter_cluster_dist", -1)),
        "element_type": str(getattr(args, "element_type", "edge_removal")),
    }


def _cluster_candidate_cache_path(graph, args):
    metadata = _build_cluster_cache_metadata(graph, args)
    cache_root = str(getattr(args, "cluster_candidate_cache_root", "candidate_cache"))
    file_name = (
        f"clusters_nc{metadata['num_of_clusters']}"
        f"_epc{metadata['edges_per_cluster']}"
        f"_r{metadata['cluster_ratio_percent']}"
        f"_in{metadata['intra_cluster_dist']}"
        f"_out{metadata['inter_cluster_dist']}"
        f"_{metadata['graph_hash']}.pt"
    )
    return osp.join(
        cache_root,
        "cluster_edge_removal",
        metadata["dataset"],
        file_name,
    )


def _canonicalize_cluster_edge_indices(cluster_edge_idxs):
    return tuple(sorted(int(edge_idx) for edge_idx in cluster_edge_idxs))


def _canonicalize_candidate_cluster_indices(cluster_edge_idx_groups):
    normalized = [_canonicalize_cluster_edge_indices(cluster) for cluster in cluster_edge_idx_groups]
    normalized.sort()
    return tuple(normalized)


def _build_cluster_search_context(graph, intra_cluster_dist, inter_cluster_dist):
    unique_edges_cpu = _get_unique_undirected_edges(graph).detach().cpu().to(torch.long)
    edge_pairs = [tuple(int(v) for v in edge.tolist()) for edge in unique_edges_cpu]
    incident_edge_idxs_by_node = [[] for _ in range(int(graph.num_nodes))]
    for edge_idx, (src, dst) in enumerate(edge_pairs):
        incident_edge_idxs_by_node[src].append(edge_idx)
        incident_edge_idxs_by_node[dst].append(edge_idx)
    incident_edge_idxs_by_node = [tuple(sorted(edge_idxs)) for edge_idxs in incident_edge_idxs_by_node]

    one_hop_nodes = _build_k_hop_reachability_sets(graph, 1, device="cpu")
    one_hop_nodes = [tuple(sorted(nodes)) for nodes in one_hop_nodes]

    inter_forbidden_reachability_sets = None
    if inter_cluster_dist > 0:
        inter_forbidden_reachability_sets = _build_k_hop_reachability_sets(
            graph,
            inter_cluster_dist - 1,
            device="cpu",
        )

    edge_forbidden_nodes = []
    if inter_forbidden_reachability_sets is None:
        edge_forbidden_nodes = [set() for _ in edge_pairs]
    else:
        for src, dst in edge_pairs:
            forbidden_nodes = set(inter_forbidden_reachability_sets[src])
            forbidden_nodes.update(inter_forbidden_reachability_sets[dst])
            edge_forbidden_nodes.append(forbidden_nodes)

    context = {
        "unique_edges_cpu": unique_edges_cpu,
        "edge_pairs": edge_pairs,
        "incident_edge_idxs_by_node": incident_edge_idxs_by_node,
        "one_hop_nodes": one_hop_nodes,
        "inter_forbidden_reachability_sets": inter_forbidden_reachability_sets,
        "edge_forbidden_nodes": edge_forbidden_nodes,
        "num_edges": int(unique_edges_cpu.shape[0]),
        "all_edge_idxs": set(range(int(unique_edges_cpu.shape[0]))),
        "intra_cluster_dist": int(intra_cluster_dist),
        "inter_cluster_dist": int(inter_cluster_dist),
    }

    if intra_cluster_dist == 2:
        center_node_infos = []
        for node_idx, node_bucket in enumerate(one_hop_nodes):
            pool_edge_idxs = tuple(sorted(_collect_pool_edge_indices_for_nodes(node_bucket, incident_edge_idxs_by_node)))
            pool_size = len(pool_edge_idxs)
            if pool_size <= 0:
                continue
            center_node_infos.append(
                {
                    "center_node_idx": node_idx,
                    "center_nodes": set(node_bucket),
                    "pool_edge_idxs": pool_edge_idxs,
                    "pool_size": pool_size,
                }
            )
        center_node_infos.sort(key=lambda item: (-item["pool_size"], item["center_node_idx"]))
        context["center_node_infos"] = center_node_infos
    elif intra_cluster_dist >= 3:
        center_edge_infos = []
        for center_edge_idx, (src, dst) in enumerate(edge_pairs):
            center_nodes = set(one_hop_nodes[src])
            center_nodes.update(one_hop_nodes[dst])
            pool_edge_idxs = tuple(
                sorted(_collect_pool_edge_indices_for_nodes(center_nodes, incident_edge_idxs_by_node))
            )
            pool_size = len(pool_edge_idxs)
            if pool_size <= 0:
                continue
            center_edge_infos.append(
                {
                    "center_edge_idx": center_edge_idx,
                    "center_nodes": center_nodes,
                    "pool_edge_idxs": pool_edge_idxs,
                    "pool_size": pool_size,
                }
            )
        center_edge_infos.sort(key=lambda item: (-item["pool_size"], item["center_edge_idx"]))
        context["center_edge_infos"] = center_edge_infos

    return context


def _build_blocked_edge_idxs(used_edge_idxs, forbidden_nodes, incident_edge_idxs_by_node):
    blocked_edge_idxs = set(int(edge_idx) for edge_idx in used_edge_idxs)
    for node_idx in forbidden_nodes:
        blocked_edge_idxs.update(incident_edge_idxs_by_node[node_idx])
    return blocked_edge_idxs


def _component_seed_edge_indices(component_edge_idxs, max_seed_count):
    if len(component_edge_idxs) <= max_seed_count:
        return list(component_edge_idxs)

    last_pos = len(component_edge_idxs) - 1
    positions = {
        int(round(last_pos * (seed_idx / max(1, max_seed_count - 1))))
        for seed_idx in range(max_seed_count)
    }
    return [component_edge_idxs[pos] for pos in sorted(positions)]


def _connected_cluster_from_seed(seed_edge_idx, target_size, allowed_edge_idxs, edge_pairs, incident_edge_idxs_by_node):
    visited = {int(seed_edge_idx)}
    queue = deque([int(seed_edge_idx)])
    selected_edge_idxs = []

    while queue and len(selected_edge_idxs) < target_size:
        current_edge_idx = queue.popleft()
        selected_edge_idxs.append(current_edge_idx)

        src, dst = edge_pairs[current_edge_idx]
        neighbor_edge_idxs = set(incident_edge_idxs_by_node[src])
        neighbor_edge_idxs.update(incident_edge_idxs_by_node[dst])
        for neighbor_edge_idx in sorted(neighbor_edge_idxs):
            if neighbor_edge_idx not in allowed_edge_idxs or neighbor_edge_idx in visited:
                continue
            visited.add(neighbor_edge_idx)
            queue.append(neighbor_edge_idx)

    if len(selected_edge_idxs) != target_size:
        return None

    return _canonicalize_cluster_edge_indices(selected_edge_idxs)


def _window_start_positions(pool_size, target_size, max_variants):
    if pool_size <= target_size or max_variants <= 1:
        return [0]

    max_start = pool_size - target_size
    positions = {0, max_start}
    if max_variants >= 3:
        positions.add(max_start // 2)
    if max_variants >= 4:
        positions.add(max_start // 3)
        positions.add((2 * max_start) // 3)

    return sorted(positions)[:max_variants]


def _pick_compact_cluster_edges(
    pool_edge_idxs,
    edges_per_cluster,
    context,
    center_nodes,
    anchor_edge_idx=None,
):
    if len(pool_edge_idxs) < edges_per_cluster:
        return None

    edge_pairs = context["edge_pairs"]
    edge_forbidden_nodes = context["edge_forbidden_nodes"]

    remaining_edge_idxs = list(pool_edge_idxs)
    if anchor_edge_idx is None or anchor_edge_idx not in pool_edge_idxs:
        anchor_edge_idx = min(
            remaining_edge_idxs,
            key=lambda edge_idx: (
                len(edge_forbidden_nodes[edge_idx]),
                0 if edge_pairs[edge_idx][0] in center_nodes or edge_pairs[edge_idx][1] in center_nodes else 1,
                edge_idx,
            ),
        )

    selected_edge_idxs = [anchor_edge_idx]
    selected_edge_set = {anchor_edge_idx}
    selected_endpoints = set(edge_pairs[anchor_edge_idx])
    current_forbidden_nodes = set(edge_forbidden_nodes[anchor_edge_idx])

    while len(selected_edge_idxs) < edges_per_cluster:
        best_edge_idx = None
        best_score = None
        for edge_idx in remaining_edge_idxs:
            if edge_idx in selected_edge_set:
                continue

            src, dst = edge_pairs[edge_idx]
            endpoints = {src, dst}
            new_forbidden_nodes = edge_forbidden_nodes[edge_idx]
            delta_forbidden = len(new_forbidden_nodes.difference(current_forbidden_nodes))
            shared_selected = len(endpoints.intersection(selected_endpoints))
            shared_center = len(endpoints.intersection(center_nodes))
            score = (
                delta_forbidden,
                0 if shared_selected > 0 else 1,
                -shared_center,
                len(new_forbidden_nodes),
                edge_idx,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_edge_idx = edge_idx

        if best_edge_idx is None:
            return None

        selected_edge_idxs.append(best_edge_idx)
        selected_edge_set.add(best_edge_idx)
        selected_endpoints.update(edge_pairs[best_edge_idx])
        current_forbidden_nodes.update(edge_forbidden_nodes[best_edge_idx])

    return {
        "edge_indices": _canonicalize_cluster_edge_indices(selected_edge_idxs),
        "forbidden_nodes": current_forbidden_nodes,
    }


def _select_anchor_edge_candidates(pool_edge_idxs, center_nodes, context, max_anchor_count):
    edge_pairs = context["edge_pairs"]
    edge_forbidden_nodes = context["edge_forbidden_nodes"]

    ranked_edge_idxs = sorted(
        pool_edge_idxs,
        key=lambda edge_idx: (
            len(edge_forbidden_nodes[edge_idx]),
            0 if edge_pairs[edge_idx][0] in center_nodes or edge_pairs[edge_idx][1] in center_nodes else 1,
            edge_idx,
        ),
    )

    anchor_edge_idxs = []
    for edge_idx in ranked_edge_idxs[:max_anchor_count]:
        anchor_edge_idxs.append(edge_idx)

    if len(ranked_edge_idxs) > 0:
        anchor_edge_idxs.append(ranked_edge_idxs[len(ranked_edge_idxs) // 2])
        anchor_edge_idxs.append(ranked_edge_idxs[-1])

    deduped_edge_idxs = []
    seen_edge_idxs = set()
    for edge_idx in anchor_edge_idxs:
        if edge_idx in seen_edge_idxs:
            continue
        deduped_edge_idxs.append(edge_idx)
        seen_edge_idxs.add(edge_idx)

    return deduped_edge_idxs[: max(1, max_anchor_count)]


def _cluster_forbidden_nodes_from_edge_indices(cluster_edge_idxs, edge_pairs, inter_forbidden_reachability_sets):
    if inter_forbidden_reachability_sets is None:
        return set()

    forbidden_nodes = set()
    for edge_idx in cluster_edge_idxs:
        src, dst = edge_pairs[edge_idx]
        forbidden_nodes.update(inter_forbidden_reachability_sets[src])
        forbidden_nodes.update(inter_forbidden_reachability_sets[dst])
    return forbidden_nodes


def _candidate_signature_to_tensor(candidate_signature, unique_edges_cpu):
    clusters = [unique_edges_cpu[list(cluster_edge_idxs)] for cluster_edge_idxs in candidate_signature]
    return torch.cat(clusters, dim=0)


def _candidate_tensor_to_signature(candidate_tensor, num_clusters, edges_per_cluster):
    candidate_cpu = candidate_tensor.detach().cpu().to(torch.long)
    if candidate_cpu.dim() != 2 or candidate_cpu.shape[1] != 2:
        raise ValueError("candidate_tensor must have shape [num_edges, 2].")

    clusters = []
    start = 0
    for _ in range(num_clusters):
        end = min(start + edges_per_cluster, candidate_cpu.shape[0])
        if start >= end:
            break
        cluster_edges = []
        for edge in candidate_cpu[start:end]:
            src, dst = int(edge[0].item()), int(edge[1].item())
            cluster_edges.append(tuple(sorted((src, dst))))
        cluster_edges.sort()
        clusters.append(tuple(cluster_edges))
        start = end

    if start < candidate_cpu.shape[0] and len(clusters) > 0:
        cluster_edges = list(clusters[-1])
        for edge in candidate_cpu[start:]:
            src, dst = int(edge[0].item()), int(edge[1].item())
            cluster_edges.append(tuple(sorted((src, dst))))
        cluster_edges.sort()
        clusters[-1] = tuple(cluster_edges)

    clusters.sort()
    return tuple(clusters)


def _generate_intra1_cluster_proposals(context, allowed_edge_idxs, edges_per_cluster, proposal_limit):
    edge_pairs = context["edge_pairs"]
    incident_edge_idxs_by_node = context["incident_edge_idxs_by_node"]
    inter_forbidden_reachability_sets = context["inter_forbidden_reachability_sets"]

    remaining_edge_idxs = set(allowed_edge_idxs)
    components = []
    while remaining_edge_idxs:
        seed_edge_idx = min(remaining_edge_idxs)
        queue = deque([seed_edge_idx])
        remaining_edge_idxs.remove(seed_edge_idx)
        component_edge_idxs = []

        while queue:
            current_edge_idx = queue.popleft()
            component_edge_idxs.append(current_edge_idx)
            src, dst = edge_pairs[current_edge_idx]
            neighbor_edge_idxs = set(incident_edge_idxs_by_node[src])
            neighbor_edge_idxs.update(incident_edge_idxs_by_node[dst])
            for neighbor_edge_idx in sorted(neighbor_edge_idxs):
                if neighbor_edge_idx not in remaining_edge_idxs:
                    continue
                remaining_edge_idxs.remove(neighbor_edge_idx)
                queue.append(neighbor_edge_idx)

        if len(component_edge_idxs) >= edges_per_cluster:
            components.append(sorted(component_edge_idxs))

    components.sort(key=lambda component_edge_idxs: (-len(component_edge_idxs), component_edge_idxs[0]))

    proposals = {}
    for component_edge_idxs in components:
        seed_edge_idxs = _component_seed_edge_indices(component_edge_idxs, max_seed_count=12)
        for seed_edge_idx in seed_edge_idxs:
            cluster_edge_idxs = _connected_cluster_from_seed(
                seed_edge_idx,
                edges_per_cluster,
                allowed_edge_idxs,
                edge_pairs,
                incident_edge_idxs_by_node,
            )
            if cluster_edge_idxs is None or cluster_edge_idxs in proposals:
                continue
            proposals[cluster_edge_idxs] = {
                "edge_indices": cluster_edge_idxs,
                "forbidden_nodes": _cluster_forbidden_nodes_from_edge_indices(
                    cluster_edge_idxs,
                    edge_pairs,
                    inter_forbidden_reachability_sets,
                ),
                "score": len(component_edge_idxs),
                "center_key": seed_edge_idx,
            }
            if len(proposals) >= proposal_limit:
                break
        if len(proposals) >= proposal_limit:
            break

    result = list(proposals.values())
    result.sort(key=lambda item: (-item["score"], len(item["forbidden_nodes"]), item["center_key"], item["edge_indices"]))
    return result[:proposal_limit]


def _generate_intra2_cluster_proposals(context, allowed_edge_idxs, edges_per_cluster, proposal_limit):
    center_node_infos = context.get("center_node_infos", [])

    proposals = {}
    for center_info in center_node_infos:
        center_node_idx = center_info["center_node_idx"]
        center_nodes = center_info["center_nodes"]
        pool_edge_idxs = [edge_idx for edge_idx in center_info["pool_edge_idxs"] if edge_idx in allowed_edge_idxs]
        if len(pool_edge_idxs) < edges_per_cluster:
            continue

        anchor_edge_idxs = _select_anchor_edge_candidates(
            pool_edge_idxs=pool_edge_idxs,
            center_nodes=center_nodes,
            context=context,
            max_anchor_count=5,
        )
        for anchor_edge_idx in anchor_edge_idxs:
            cluster = _pick_compact_cluster_edges(
                pool_edge_idxs=pool_edge_idxs,
                edges_per_cluster=edges_per_cluster,
                context=context,
                center_nodes=center_nodes,
                anchor_edge_idx=anchor_edge_idx,
            )
            if cluster is None:
                continue
            cluster_edge_idxs = cluster["edge_indices"]
            if cluster_edge_idxs in proposals:
                continue
            proposals[cluster_edge_idxs] = {
                "edge_indices": cluster_edge_idxs,
                "forbidden_nodes": cluster["forbidden_nodes"],
                "score": (len(pool_edge_idxs), -len(cluster["forbidden_nodes"])),
                "center_key": center_node_idx,
            }
            if len(proposals) >= proposal_limit:
                break
        if len(proposals) >= proposal_limit:
            break

    result = list(proposals.values())
    result.sort(
        key=lambda item: (
            -item["score"][0],
            item["score"][1],
            len(item["forbidden_nodes"]),
            item["center_key"],
            item["edge_indices"],
        )
    )
    return result[:proposal_limit]


def _generate_intra3_cluster_proposals(context, allowed_edge_idxs, edges_per_cluster, proposal_limit):
    center_edge_infos = context.get("center_edge_infos", [])

    proposals = {}
    for center_info in center_edge_infos:
        center_edge_idx = center_info["center_edge_idx"]
        center_nodes = center_info["center_nodes"]
        pool_edge_idxs = [edge_idx for edge_idx in center_info["pool_edge_idxs"] if edge_idx in allowed_edge_idxs]
        if len(pool_edge_idxs) < edges_per_cluster:
            continue

        anchor_edge_idxs = _select_anchor_edge_candidates(
            pool_edge_idxs=pool_edge_idxs,
            center_nodes=center_nodes,
            context=context,
            max_anchor_count=4,
        )
        for anchor_edge_idx in anchor_edge_idxs:
            cluster = _pick_compact_cluster_edges(
                pool_edge_idxs=pool_edge_idxs,
                edges_per_cluster=edges_per_cluster,
                context=context,
                center_nodes=center_nodes,
                anchor_edge_idx=anchor_edge_idx,
            )
            if cluster is None:
                continue
            cluster_edge_idxs = cluster["edge_indices"]
            if cluster_edge_idxs in proposals:
                continue
            proposals[cluster_edge_idxs] = {
                "edge_indices": cluster_edge_idxs,
                "forbidden_nodes": cluster["forbidden_nodes"],
                "score": (len(pool_edge_idxs), -len(cluster["forbidden_nodes"])),
                "center_key": center_edge_idx,
            }
            if len(proposals) >= proposal_limit:
                break
        if len(proposals) >= proposal_limit:
            break

    result = list(proposals.values())
    result.sort(
        key=lambda item: (
            -item["score"][0],
            item["score"][1],
            len(item["forbidden_nodes"]),
            item["center_key"],
            item["edge_indices"],
        )
    )
    return result[:proposal_limit]


def _generate_cluster_proposals(context, used_edge_idxs, forbidden_nodes, edges_per_cluster, proposal_limit):
    blocked_edge_idxs = _build_blocked_edge_idxs(
        used_edge_idxs,
        forbidden_nodes,
        context["incident_edge_idxs_by_node"],
    )
    if len(blocked_edge_idxs) >= context["num_edges"]:
        return [], blocked_edge_idxs

    allowed_edge_idxs = context["all_edge_idxs"].difference(blocked_edge_idxs)
    if len(allowed_edge_idxs) < edges_per_cluster:
        return [], blocked_edge_idxs

    intra_cluster_dist = context["intra_cluster_dist"]
    if intra_cluster_dist == 1:
        proposals = _generate_intra1_cluster_proposals(
            context=context,
            allowed_edge_idxs=allowed_edge_idxs,
            edges_per_cluster=edges_per_cluster,
            proposal_limit=proposal_limit,
        )
    elif intra_cluster_dist == 2:
        proposals = _generate_intra2_cluster_proposals(
            context=context,
            allowed_edge_idxs=allowed_edge_idxs,
            edges_per_cluster=edges_per_cluster,
            proposal_limit=proposal_limit,
        )
    else:
        proposals = _generate_intra3_cluster_proposals(
            context=context,
            allowed_edge_idxs=allowed_edge_idxs,
            edges_per_cluster=edges_per_cluster,
            proposal_limit=proposal_limit,
        )

    return proposals, blocked_edge_idxs


def _find_cluster_candidate_signature(
    context,
    num_clusters,
    edges_per_cluster,
    used_candidate_signatures,
    candidate_index,
):
    proposal_limit = 256

    def _dfs(depth, selected_cluster_edge_idxs, used_edge_idxs, forbidden_nodes):
        blocked_edge_idxs = _build_blocked_edge_idxs(
            used_edge_idxs,
            forbidden_nodes,
            context["incident_edge_idxs_by_node"],
        )
        remaining_clusters = num_clusters - depth
        if context["num_edges"] - len(blocked_edge_idxs) < remaining_clusters * edges_per_cluster:
            return None

        if depth == num_clusters:
            candidate_signature = _canonicalize_candidate_cluster_indices(selected_cluster_edge_idxs)
            if candidate_signature in used_candidate_signatures:
                return None
            return candidate_signature

        proposals, _ = _generate_cluster_proposals(
            context=context,
            used_edge_idxs=used_edge_idxs,
            forbidden_nodes=forbidden_nodes,
            edges_per_cluster=edges_per_cluster,
            proposal_limit=proposal_limit,
        )
        if len(proposals) == 0:
            return None

        rotation = 0
        if len(proposals) > 1:
            rotation = (candidate_index * 7 + depth * 3) % len(proposals)
        ordered_proposals = proposals[rotation:] + proposals[:rotation]
        branch_limit = 128 if depth == 0 else 64
        ordered_proposals = ordered_proposals[:branch_limit]

        for proposal in ordered_proposals:
            next_used_edge_idxs = set(used_edge_idxs)
            next_used_edge_idxs.update(proposal["edge_indices"])

            next_forbidden_nodes = set(forbidden_nodes)
            next_forbidden_nodes.update(proposal["forbidden_nodes"])

            result = _dfs(
                depth + 1,
                selected_cluster_edge_idxs + [proposal["edge_indices"]],
                next_used_edge_idxs,
                next_forbidden_nodes,
            )
            if result is not None:
                return result

        return None

    return _dfs(depth=0, selected_cluster_edge_idxs=[], used_edge_idxs=set(), forbidden_nodes=set())


def _build_structured_clustered_edge_removal_candidate_list(
    graph,
    num_groups,
    num_clusters,
    edges_per_cluster,
    intra_cluster_dist,
    inter_cluster_dist,
    existing_signatures=None,
):
    context = _build_cluster_search_context(
        graph=graph,
        intra_cluster_dist=intra_cluster_dist,
        inter_cluster_dist=inter_cluster_dist,
    )

    used_candidate_signatures = set(existing_signatures or set())
    candidates = []
    for candidate_index in range(num_groups):
        candidate_signature = _find_cluster_candidate_signature(
            context=context,
            num_clusters=num_clusters,
            edges_per_cluster=edges_per_cluster,
            used_candidate_signatures=used_candidate_signatures,
            candidate_index=candidate_index,
        )
        if candidate_signature is None:
            break
        used_candidate_signatures.add(candidate_signature)
        candidates.append(_candidate_signature_to_tensor(candidate_signature, context["unique_edges_cpu"]))

    return candidates, used_candidate_signatures


def _build_randomized_clustered_edge_removal_candidate_list(
    graph,
    num_groups,
    num_clusters,
    edges_per_cluster,
    intra_cluster_dist,
    inter_cluster_dist,
    existing_signatures=None,
    max_candidate_restarts=3000,
    max_generation_attempts=24,
):
    unique_edges_cpu = _get_unique_undirected_edges(graph).detach().cpu()
    intra_reachability_sets = _build_k_hop_reachability_sets(graph, intra_cluster_dist, device="cpu")
    inter_forbidden_reachability_sets = None
    if inter_cluster_dist > 0:
        inter_forbidden_reachability_sets = _build_k_hop_reachability_sets(
            graph,
            inter_cluster_dist - 1,
            device="cpu",
        )

    max_draws_per_cluster = max(1000, unique_edges_cpu.shape[0] * 20)
    candidate_signatures = set(existing_signatures or set())
    candidates = []
    generation_attempts = 0

    while len(candidates) < num_groups and generation_attempts < max_generation_attempts:
        generation_attempts += 1
        candidate = _sample_clustered_edge_candidate(
            unique_edges_cpu=unique_edges_cpu,
            num_clusters=num_clusters,
            edges_per_cluster=edges_per_cluster,
            intra_cluster_dist=intra_cluster_dist,
            intra_reachability_sets=intra_reachability_sets,
            inter_forbidden_reachability_sets=inter_forbidden_reachability_sets,
            max_candidate_restarts=max_candidate_restarts,
            max_draws_per_cluster=max_draws_per_cluster,
        )
        if candidate is None:
            continue

        candidate_signature = _candidate_tensor_to_signature(
            candidate_tensor=candidate,
            num_clusters=num_clusters,
            edges_per_cluster=edges_per_cluster,
        )
        if candidate_signature in candidate_signatures:
            continue

        candidate_signatures.add(candidate_signature)
        candidates.append(candidate)

    return candidates, candidate_signatures


def _build_structured_clustered_edge_removal_candidates(
    graph,
    num_groups,
    num_clusters,
    edges_per_cluster,
    intra_cluster_dist,
    inter_cluster_dist,
):
    if intra_cluster_dist == 1:
        structured_candidates, _ = _build_structured_clustered_edge_removal_candidate_list(
            graph=graph,
            num_groups=num_groups,
            num_clusters=num_clusters,
            edges_per_cluster=edges_per_cluster,
            intra_cluster_dist=intra_cluster_dist,
            inter_cluster_dist=inter_cluster_dist,
        )
        if len(structured_candidates) != num_groups:
            raise RuntimeError(
                "Failed to sample a valid edge group with the current constraints. "
                "Unable to construct enough connected-cluster candidates."
            )
        return torch.stack(structured_candidates, dim=0).to(graph.edge_index.device)

    randomized_candidates, candidate_signatures = _build_randomized_clustered_edge_removal_candidate_list(
        graph=graph,
        num_groups=num_groups,
        num_clusters=num_clusters,
        edges_per_cluster=edges_per_cluster,
        intra_cluster_dist=intra_cluster_dist,
        inter_cluster_dist=inter_cluster_dist,
        max_candidate_restarts=6000,
        max_generation_attempts=max(64, num_groups * 16),
    )

    if len(randomized_candidates) < num_groups:
        remaining_needed = num_groups - len(randomized_candidates)
        print(
            f"[CANDIDATE-CACHE] randomized pairwise sampler found {len(randomized_candidates)}/{num_groups}; "
            f"falling back to structured proposals for remaining {remaining_needed}"
        )
        fallback_candidates, _ = _build_structured_clustered_edge_removal_candidate_list(
            graph=graph,
            num_groups=remaining_needed,
            num_clusters=num_clusters,
            edges_per_cluster=edges_per_cluster,
            intra_cluster_dist=intra_cluster_dist,
            inter_cluster_dist=inter_cluster_dist,
            existing_signatures=candidate_signatures,
        )
        randomized_candidates.extend(fallback_candidates)

    if len(randomized_candidates) != num_groups:
        raise RuntimeError(
            "Failed to sample a valid edge group with the current constraints. "
            "Unable to construct enough cached cluster candidates with the configured hybrid sampler."
        )

    return torch.stack(randomized_candidates, dim=0).to(graph.edge_index.device)


def _load_cluster_candidate_cache(cache_path, expected_metadata, min_num_candidates):
    if not osp.isfile(cache_path):
        return None

    payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict):
        return None

    metadata = payload.get("metadata", {})
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            return None

    candidates = payload.get("candidates", None)
    if not torch.is_tensor(candidates) or candidates.dim() != 3:
        return None
    if int(candidates.shape[0]) < int(min_num_candidates):
        return None

    return candidates.detach().cpu().to(torch.long)


def build_or_load_clustered_edge_removal_candidates(graph, args, force_rebuild=False):
    cache_path = _cluster_candidate_cache_path(graph, args)
    cache_metadata = _build_cluster_cache_metadata(graph, args)
    num_groups = int(getattr(args, "num_removal_candidates", 50))
    if not force_rebuild:
        cached_candidates = _load_cluster_candidate_cache(
            cache_path=cache_path,
            expected_metadata=cache_metadata,
            min_num_candidates=num_groups,
        )
        if cached_candidates is not None:
            print(
                f"[CANDIDATE-CACHE] loaded {num_groups} clustered removal candidates from {cache_path}"
            )
            return cached_candidates[:num_groups].to(graph.edge_index.device), cache_path, False

    print(
        f"[CANDIDATE-CACHE] building clustered removal candidates: "
        f"dataset={cache_metadata['dataset']}, num_groups={num_groups}, "
        f"num_clusters={cache_metadata['num_of_clusters']}, edges_per_cluster={cache_metadata['edges_per_cluster']}, "
        f"intra={cache_metadata['intra_cluster_dist']}, inter={cache_metadata['inter_cluster_dist']}, "
        f"ratio={cache_metadata['cluster_ratio_percent']}"
    )
    candidates = _build_structured_clustered_edge_removal_candidates(
        graph=graph,
        num_groups=num_groups,
        num_clusters=cache_metadata["num_of_clusters"],
        edges_per_cluster=cache_metadata["edges_per_cluster"],
        intra_cluster_dist=cache_metadata["intra_cluster_dist"],
        inter_cluster_dist=cache_metadata["inter_cluster_dist"],
    ).detach().cpu()

    os.makedirs(osp.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "version": 1,
            "metadata": cache_metadata,
            "candidates": candidates,
        },
        cache_path,
    )
    print(
        f"[CANDIDATE-CACHE] saved {int(candidates.shape[0])} clustered removal candidates to {cache_path}"
    )

    return candidates.to(graph.edge_index.device), cache_path, True


def _normalize_dataset_alias(dataset_name):
    normalized = str(dataset_name).lower()
    alias_map = {
        "cora": "cora_public",
        "citeseer": "citeseer_public",
        "pubmed": "pubmed_public",
    }
    return alias_map.get(normalized, normalized)


def _cluster_history_key(dataset_name, cluster_ratio_percent, num_of_clusters, intra_cluster_dist, inter_cluster_dist):
    return (
        _normalize_dataset_alias(dataset_name),
        int(cluster_ratio_percent),
        int(num_of_clusters),
        int(intra_cluster_dist),
        int(inter_cluster_dist),
    )


def _cluster_history_key_from_config(config):
    try:
        experiment_name = config.get("experiment_name")
        if experiment_name != "clusters":
            return None
        return _cluster_history_key(
            dataset_name=config.get("dataset"),
            cluster_ratio_percent=config.get("cluster_ratio_percent"),
            num_of_clusters=config.get("num_of_clusters", 3),
            intra_cluster_dist=config.get("intra_cluster_dist"),
            inter_cluster_dist=config.get("inter_cluster_dist"),
        )
    except Exception:
        return None


def _cluster_history_key_from_overrides(path):
    if not path.exists():
        return None

    parsed = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("- ") or "=" not in line:
            continue
        key, value = line[2:].split("=", 1)
        parsed[key] = value

    if parsed.get("experiment") != "clusters":
        return None

    cluster_ratio_percent = parsed.get("experiment.cluster_ratio_percent", "10")
    return _cluster_history_key(
        dataset_name=parsed.get("dataset"),
        cluster_ratio_percent=cluster_ratio_percent,
        num_of_clusters=parsed.get("experiment.num_of_clusters", 3),
        intra_cluster_dist=parsed.get("experiment.intra_cluster_dist"),
        inter_cluster_dist=parsed.get("experiment.inter_cluster_dist"),
    )


def build_cluster_history_index(repo_root="."):
    repo_root = Path(repo_root)
    history_index = {
        "candidate_results": {},
        "done": {},
        "constraint_unsat": {},
    }

    for candidate_csv in repo_root.rglob("candidate_results.csv"):
        config_path = candidate_csv.parent / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        history_key = _cluster_history_key_from_config(config)
        if history_key is None:
            continue
        history_index["candidate_results"].setdefault(history_key, str(candidate_csv))

    for done_marker in repo_root.rglob("DONE"):
        overrides_path = done_marker.parent / ".hydra" / "overrides.yaml"
        if overrides_path.exists():
            history_key = _cluster_history_key_from_overrides(overrides_path)
            if history_key is not None:
                history_index["done"].setdefault(history_key, str(done_marker.parent))
                continue

        config_path = done_marker.parent / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        history_key = _cluster_history_key_from_config(config)
        if history_key is not None:
            history_index["done"].setdefault(history_key, str(done_marker.parent))

    for unsat_marker in repo_root.rglob("CONSTRAINT_UNSAT.txt"):
        overrides_path = unsat_marker.parent / ".hydra" / "overrides.yaml"
        if overrides_path.exists():
            history_key = _cluster_history_key_from_overrides(overrides_path)
            if history_key is not None:
                history_index["constraint_unsat"].setdefault(history_key, []).append(str(unsat_marker))

    return history_index


def _connected_edge_component_sizes(edge_pairs, incident_edge_idxs_by_node):
    remaining_edge_idxs = set(range(len(edge_pairs)))
    component_sizes = []
    while remaining_edge_idxs:
        seed_edge_idx = min(remaining_edge_idxs)
        queue = deque([seed_edge_idx])
        remaining_edge_idxs.remove(seed_edge_idx)
        component_size = 0

        while queue:
            current_edge_idx = queue.popleft()
            component_size += 1
            src, dst = edge_pairs[current_edge_idx]
            neighbor_edge_idxs = set(incident_edge_idxs_by_node[src])
            neighbor_edge_idxs.update(incident_edge_idxs_by_node[dst])
            for neighbor_edge_idx in neighbor_edge_idxs:
                if neighbor_edge_idx not in remaining_edge_idxs:
                    continue
                remaining_edge_idxs.remove(neighbor_edge_idx)
                queue.append(neighbor_edge_idx)

        component_sizes.append(component_size)

    component_sizes.sort(reverse=True)
    return component_sizes


def label_cluster_feasibility(graph, args, history_index=None, repo_root="."):
    if history_index is None:
        history_index = build_cluster_history_index(repo_root=repo_root)

    history_key = _cluster_history_key(
        dataset_name=getattr(args, "dataset", "unknown"),
        cluster_ratio_percent=getattr(args, "cluster_ratio_percent", -1),
        num_of_clusters=getattr(args, "num_of_clusters", 3),
        intra_cluster_dist=getattr(args, "intra_cluster_dist", -1),
        inter_cluster_dist=getattr(args, "inter_cluster_dist", -1),
    )

    cache_path = _cluster_candidate_cache_path(graph, args)
    cache_exists = osp.isfile(cache_path)
    candidate_results_path = history_index["candidate_results"].get(history_key)
    done_path = history_index["done"].get(history_key)
    constraint_unsat_paths = history_index["constraint_unsat"].get(history_key, [])

    context = _build_cluster_search_context(
        graph=graph,
        intra_cluster_dist=int(getattr(args, "intra_cluster_dist", 1)),
        inter_cluster_dist=int(getattr(args, "inter_cluster_dist", 1)),
    )
    edges_per_cluster = int(getattr(args, "edges_per_cluster", -1))
    local_pool_count = 0
    max_pool_size = 0
    pool_kind = "none"

    if int(getattr(args, "intra_cluster_dist", 1)) == 1:
        component_sizes = _connected_edge_component_sizes(
            edge_pairs=context["edge_pairs"],
            incident_edge_idxs_by_node=context["incident_edge_idxs_by_node"],
        )
        local_pool_count = sum(1 for size in component_sizes if size >= edges_per_cluster)
        max_pool_size = component_sizes[0] if len(component_sizes) > 0 else 0
        pool_kind = "connected_component"
    elif int(getattr(args, "intra_cluster_dist", 1)) == 2:
        center_node_infos = context.get("center_node_infos", [])
        local_pool_count = sum(1 for info in center_node_infos if info["pool_size"] >= edges_per_cluster)
        max_pool_size = max((info["pool_size"] for info in center_node_infos), default=0)
        pool_kind = "center_node_pool"
    else:
        center_edge_infos = context.get("center_edge_infos", [])
        local_pool_count = sum(1 for info in center_edge_infos if info["pool_size"] >= edges_per_cluster)
        max_pool_size = max((info["pool_size"] for info in center_edge_infos), default=0)
        pool_kind = "center_edge_pool"

    label = "UNKNOWN"
    reason = "no witness or explicit failure history"
    evidence_path = None

    if cache_exists:
        label = "FEASIBLE"
        reason = "cluster candidate cache exists"
        evidence_path = cache_path
    elif candidate_results_path is not None:
        label = "FEASIBLE"
        reason = "historical candidate_results.csv exists"
        evidence_path = candidate_results_path
    elif done_path is not None:
        label = "FEASIBLE"
        reason = "historical DONE artifact exists"
        evidence_path = done_path
    elif max_pool_size < edges_per_cluster:
        label = "LIKELY_UNSAT"
        reason = f"no single {pool_kind} reaches edges_per_cluster"
    elif len(constraint_unsat_paths) > 0:
        label = "LIKELY_UNSAT"
        reason = "historical constraint-unsat artifacts exist without witness"
        evidence_path = constraint_unsat_paths[0]

    return {
        "label": label,
        "reason": reason,
        "evidence_path": evidence_path,
        "cache_exists": cache_exists,
        "candidate_results_path": candidate_results_path,
        "done_path": done_path,
        "constraint_unsat_count": len(constraint_unsat_paths),
        "pool_kind": pool_kind,
        "local_pool_count": int(local_pool_count),
        "max_pool_size": int(max_pool_size),
        "edges_per_cluster": int(edges_per_cluster),
        "history_key": history_key,
    }


def _sample_connected_cluster_edges(
    unique_edges_cpu,
    edges_per_cluster,
    selected_idxs,
    built_clusters,
    inter_forbidden_reachability_sets,
    max_draws_per_cluster,
):
    """Grow one cluster by adding edges that stay connected via shared endpoints.

    This is used when `intra_cluster_dist == 1`. In that setting, requiring all
    edge pairs inside a cluster to be adjacent is too strong; what we want is a
    connected cluster that can be grown one edge at a time.
    """
    edge_pairs = [(int(edge[0].item()), int(edge[1].item())) for edge in unique_edges_cpu]
    node_to_edge_idxs = {}
    for edge_idx, (src, dst) in enumerate(edge_pairs):
        node_to_edge_idxs.setdefault(src, []).append(edge_idx)
        node_to_edge_idxs.setdefault(dst, []).append(edge_idx)

    max_seed_restarts = max(50, min(max_draws_per_cluster, unique_edges_cpu.shape[0] * 2))

    for _ in range(max_seed_restarts):
        seed_edge_idx = torch.randint(0, unique_edges_cpu.shape[0], (1,)).item()
        if seed_edge_idx in selected_idxs:
            continue

        seed_edge = unique_edges_cpu[seed_edge_idx]
        if not _edge_is_valid_against_previous_clusters(
            seed_edge, built_clusters, inter_forbidden_reachability_sets
        ):
            continue

        cluster_edge_idxs = [seed_edge_idx]
        cluster_edge_idx_set = {seed_edge_idx}
        frontier_idxs = set()
        for node in edge_pairs[seed_edge_idx]:
            frontier_idxs.update(node_to_edge_idxs[node])
        frontier_idxs.difference_update(selected_idxs)
        frontier_idxs.discard(seed_edge_idx)

        draws = 0
        while len(cluster_edge_idxs) < edges_per_cluster and draws < max_draws_per_cluster:
            if not frontier_idxs:
                break

            draws += 1
            frontier_list = tuple(frontier_idxs)
            next_pos = torch.randint(0, len(frontier_list), (1,)).item()
            next_edge_idx = frontier_list[next_pos]
            frontier_idxs.remove(next_edge_idx)
            if next_edge_idx in selected_idxs or next_edge_idx in cluster_edge_idx_set:
                continue

            edge = unique_edges_cpu[next_edge_idx]
            if not _edge_is_valid_against_previous_clusters(
                edge, built_clusters, inter_forbidden_reachability_sets
            ):
                continue

            cluster_edge_idxs.append(next_edge_idx)
            cluster_edge_idx_set.add(next_edge_idx)
            for node in edge_pairs[next_edge_idx]:
                frontier_idxs.update(node_to_edge_idxs[node])
            frontier_idxs.difference_update(selected_idxs)
            frontier_idxs.difference_update(cluster_edge_idx_set)

        if len(cluster_edge_idxs) == edges_per_cluster:
            return cluster_edge_idxs

    return None


def _sample_single_edge_group(unique_edges_cpu, num_group_elem, sampler, reachability_sets, max_draws):
    selected_edges = []
    selected_idxs = set()
    draws = 0

    while len(selected_edges) < num_group_elem and draws < max_draws:
        draws += 1
        edge_idx = torch.randint(0, unique_edges_cpu.shape[0], (1,)).item()
        if edge_idx in selected_idxs:
            continue

        edge = unique_edges_cpu[edge_idx]
        is_valid = True
        if sampler in ["group_neighbor", "group_non_neighbor"]:
            for selected_edge in selected_edges:
                is_neighbor = _edge_pair_is_within_distance(edge, selected_edge, reachability_sets)
                if sampler == "group_neighbor" and not is_neighbor:
                    is_valid = False
                    break
                if sampler == "group_non_neighbor" and is_neighbor:
                    is_valid = False
                    break

        if not is_valid:
            continue

        selected_idxs.add(edge_idx)
        selected_edges.append(edge)

    if len(selected_edges) != num_group_elem:
        return None

    return torch.stack(selected_edges, dim=0)


def sample_grouped_edge_removal_candidates(
    graph,
    num_groups,
    num_group_elem,
    sampler="uniform",
    removal_neighbor_dist=1,
    max_group_restarts=500,
):
    """
    Sample grouped edge-removal candidates.
    - Sampling is with replacement across groups.
    - Sampling is without replacement within each group.
    """
    unique_edges = _get_unique_undirected_edges(graph)
    if unique_edges.shape[0] < num_group_elem:
        raise ValueError(
            f"Cannot sample {num_group_elem} unique edges inside a group from only {unique_edges.shape[0]} unique edges."
        )

    unique_edges_cpu = unique_edges.detach().cpu()
    reachability_sets = None
    if sampler in ["group_neighbor", "group_non_neighbor"] and removal_neighbor_dist > 1:
        reachability_sets = _build_k_hop_reachability_sets(graph, removal_neighbor_dist, device="cpu")

    groups = []
    max_draws = max(1000, unique_edges_cpu.shape[0] * 20)

    for _ in range(num_groups):
        group = None
        for _ in range(max_group_restarts):
            if sampler == "uniform":
                idxs = torch.randperm(unique_edges_cpu.shape[0])[:num_group_elem]
                group = unique_edges_cpu[idxs]
            elif sampler == "group_neighbor" and removal_neighbor_dist == 1:
                group = _sample_connected_edge_group(unique_edges_cpu, num_group_elem)
            else:
                group = _sample_single_edge_group(
                    unique_edges_cpu,
                    num_group_elem,
                    sampler,
                    reachability_sets,
                    max_draws=max_draws,
                )
            if group is not None:
                break

        if group is None:
            raise RuntimeError(
                "Failed to sample a valid edge group with the current constraints. "
                "Try lowering num_group_elem or adjusting removal_neighbor_dist."
            )
        groups.append(group)

    grouped_candidates = torch.stack(groups, dim=0).to(graph.edge_index.device)
    return grouped_candidates


def _sample_clustered_edge_candidate(
    unique_edges_cpu,
    num_clusters,
    edges_per_cluster,
    intra_cluster_dist,
    intra_reachability_sets,
    inter_forbidden_reachability_sets,
    max_candidate_restarts,
    max_draws_per_cluster,
):
    for _ in range(max_candidate_restarts):
        selected_idxs = set()
        built_clusters = []
        candidate_failed = False

        for _cluster_idx in range(num_clusters):
            if intra_cluster_dist == 1:
                cluster_edge_idxs = _sample_connected_cluster_edges(
                    unique_edges_cpu=unique_edges_cpu,
                    edges_per_cluster=edges_per_cluster,
                    selected_idxs=selected_idxs,
                    built_clusters=built_clusters,
                    inter_forbidden_reachability_sets=inter_forbidden_reachability_sets,
                    max_draws_per_cluster=max_draws_per_cluster,
                )
                if cluster_edge_idxs is None:
                    candidate_failed = True
                    break

                selected_idxs.update(cluster_edge_idxs)
                built_clusters.append(unique_edges_cpu[cluster_edge_idxs])
                continue

            cluster_edges = []
            draws = 0

            while len(cluster_edges) < edges_per_cluster and draws < max_draws_per_cluster:
                draws += 1
                edge_idx = torch.randint(0, unique_edges_cpu.shape[0], (1,)).item()
                if edge_idx in selected_idxs:
                    continue

                edge = unique_edges_cpu[edge_idx]
                is_valid = True

                # Intra-cluster constraint: all edge pairs in same cluster must be within intra distance.
                for selected_edge in cluster_edges:
                    if not _edge_pair_is_within_distance(edge, selected_edge, intra_reachability_sets):
                        is_valid = False
                        break
                if not is_valid:
                    continue

                # Inter-cluster constraint: all edge pairs across clusters must be farther than (inter_dist - 1).
                if not _edge_is_valid_against_previous_clusters(
                    edge, built_clusters, inter_forbidden_reachability_sets
                ):
                    is_valid = False
                if not is_valid:
                    continue

                selected_idxs.add(edge_idx)
                cluster_edges.append(edge)

            if len(cluster_edges) != edges_per_cluster:
                candidate_failed = True
                break

            built_clusters.append(torch.stack(cluster_edges, dim=0))

        if not candidate_failed:
            return torch.cat(built_clusters, dim=0)

    return None


def sample_clustered_edge_removal_candidates(
    graph,
    num_groups,
    num_clusters,
    edges_per_cluster,
    intra_cluster_dist,
    inter_cluster_dist,
    max_candidate_restarts=500,
):
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive.")
    if edges_per_cluster <= 0:
        raise ValueError("edges_per_cluster must be positive.")
    if intra_cluster_dist < 0:
        raise ValueError("intra_cluster_dist must be non-negative.")
    if inter_cluster_dist < 0:
        raise ValueError("inter_cluster_dist must be non-negative.")

    unique_edges = _get_unique_undirected_edges(graph)
    unique_edges_cpu = unique_edges.detach().cpu()
    num_group_elem = int(num_clusters * edges_per_cluster)
    if unique_edges_cpu.shape[0] < num_group_elem:
        raise ValueError(
            f"Cannot sample {num_group_elem} unique edges inside a candidate from only {unique_edges_cpu.shape[0]} unique edges."
        )
    return _build_structured_clustered_edge_removal_candidates(
        graph=graph,
        num_groups=num_groups,
        num_clusters=num_clusters,
        edges_per_cluster=edges_per_cluster,
        intra_cluster_dist=intra_cluster_dist,
        inter_cluster_dist=inter_cluster_dist,
    )


def clustered_edge_removal_candidates(graph, args):
    candidates, _, _ = build_or_load_clustered_edge_removal_candidates(
        graph=graph,
        args=args,
        force_rebuild=_flag_is_enabled(getattr(args, "cluster_candidate_force_rebuild", False)),
    )
    return candidates


def neighbor_edge_removal_candidates(graph, args):
    removal_neighbor_dist = getattr(args, "removal_neighbor_dist", 1)
    return sample_grouped_edge_removal_candidates(
        graph=graph,
        num_groups=args.num_removal_candidates,
        num_group_elem=args.num_group_elem,
        sampler="group_neighbor",
        removal_neighbor_dist=removal_neighbor_dist,
    )


def non_neighbor_edge_removal_candidates(graph, args):
    removal_neighbor_dist = getattr(args, "removal_neighbor_dist", 1)
    return sample_grouped_edge_removal_candidates(
        graph=graph,
        num_groups=args.num_removal_candidates,
        num_group_elem=args.num_group_elem,
        sampler="group_non_neighbor",
        removal_neighbor_dist=removal_neighbor_dist,
    )


def non_neighbor_edge_removal_cadidates(graph, args):
    # Backward-compatible alias for typo in experimental scripts.
    return non_neighbor_edge_removal_candidates(graph, args)


def get_grouped_edge_removal_candidates(graph, args):
    if getattr(args, "experiment_name", "none") == "clusters":
        return clustered_edge_removal_candidates(graph, args)

    sampler = getattr(args, "removal_candidate_sampler", "uniform")
    removal_neighbor_dist = getattr(args, "removal_neighbor_dist", 1)
    if sampler == "group_non_neighbor":
        return non_neighbor_edge_removal_candidates(graph, args)
    if sampler == "group_neighbor":
        return neighbor_edge_removal_candidates(graph, args)
    if sampler == "uniform":
        return sample_grouped_edge_removal_candidates(
            graph=graph,
            num_groups=args.num_removal_candidates,
            num_group_elem=args.num_group_elem,
            sampler="uniform",
            removal_neighbor_dist=removal_neighbor_dist,
        )
    raise ValueError(
        f"Unknown removal_candidate_sampler: {sampler}. "
        "Choose from ['uniform', 'group_non_neighbor', 'group_neighbor']."
    )


def _normalize_influence_mode(mode):
    if mode == "fixed_theta":
        return "calculate_influence"
    return mode


def _normalize_metric_mode(mode):
    if mode is None:
        return "global"
    normalized = str(mode).strip().lower()
    if normalized == "candidate_partition":
        return "partition"
    return normalized


def calculate_influence(influence_module, candidates, influence_type):
    total_inf, retrain_inf, perturb_inf, module_scale, inv_hvp_norm, num_ins = influence_module.calculate_influence(
        candidates, influence_type
    )
    return {
        "total_inf": total_inf,
        "retrain_inf": retrain_inf,
        "perturb_inf": perturb_inf,
        "module_scale": module_scale,
        "inv_hvp_norm": inv_hvp_norm,
        "avg_num_influenced_nodes": num_ins,
    }


def calculate_fixed_theta_influence(influence_module, candidates, influence_type):
    # Backward-compatible alias.
    return calculate_influence(influence_module, candidates, influence_type)


def _split_candidate_into_clusters_contiguous(candidate, num_clusters, edges_per_cluster):
    if num_clusters <= 1:
        return [candidate]

    if edges_per_cluster <= 0:
        edges_per_cluster = max(1, candidate.shape[0] // num_clusters)

    clusters = []
    start = 0
    for _ in range(num_clusters):
        end = min(start + edges_per_cluster, candidate.shape[0])
        if start >= end:
            break
        clusters.append(candidate[start:end])
        start = end

    if len(clusters) == 0:
        return [candidate]

    # If any remainder exists, attach it to the last cluster.
    if start < candidate.shape[0]:
        clusters[-1] = torch.cat([clusters[-1], candidate[start:]], dim=0)

    return clusters


def _split_candidate_into_clusters_round_robin(candidate, num_clusters):
    if num_clusters <= 1:
        return [candidate]

    buckets = [[] for _ in range(num_clusters)]
    for idx, edge in enumerate(candidate):
        buckets[idx % num_clusters].append(edge)

    clusters = []
    for bucket in buckets:
        if len(bucket) > 0:
            clusters.append(torch.stack(bucket, dim=0))

    if len(clusters) == 0:
        return [candidate]

    return clusters


def _split_candidate_into_clusters(candidate, args, graph=None):
    """
    Split one candidate ([num_group_elem, 2]) into cluster list.
    Default strategy uses contiguous partitioning, but dispatch is strategy-based
    so future random-candidate clustering logics can be plugged in.
    """
    if candidate.dim() != 2 or candidate.shape[1] != 2:
        raise ValueError("candidate must have shape [num_edges, 2].")

    num_clusters = int(getattr(args, "num_of_clusters", 1))
    edges_per_cluster = int(getattr(args, "edges_per_cluster", -1))
    strategy = getattr(args, "cluster_partition_strategy", "contiguous")

    custom_clusterer = getattr(args, "candidate_clusterer_fn", None)
    if callable(custom_clusterer):
        clusters = custom_clusterer(candidate, args, graph)
    elif strategy == "contiguous":
        clusters = _split_candidate_into_clusters_contiguous(candidate, num_clusters, edges_per_cluster)
    elif strategy == "round_robin":
        clusters = _split_candidate_into_clusters_round_robin(candidate, num_clusters)
    else:
        raise ValueError(
            f"Unknown cluster_partition_strategy: {strategy}. "
            "Choose from ['contiguous', 'round_robin'] or pass callable args.candidate_clusterer_fn."
        )

    if not isinstance(clusters, (list, tuple)) or len(clusters) == 0:
        raise ValueError("Clustering function must return a non-empty list of [num_edges, 2] tensors.")
    for cluster in clusters:
        if not torch.is_tensor(cluster) or cluster.dim() != 2 or cluster.shape[1] != 2:
            raise ValueError("Each cluster must be a tensor with shape [num_edges_in_cluster, 2].")

    return list(clusters)


def calculate_clusterwise_fixed_theta_influence(influence_module, candidates, args, influence_type):
    if candidates.dim() != 3:
        raise ValueError("candidates must be a 3D tensor with shape [num_candidates, num_group_elem, 2].")

    sum_total_list = []
    sum_retrain_list = []
    sum_perturb_list = []
    avg_num_influenced_nodes_list = []
    per_candidate_details = []
    module_scale = None
    inv_hvp_norm = None

    graph_for_clustering = getattr(influence_module, "graph", None)
    for candidate_idx, candidate in enumerate(candidates):
        clusters = _split_candidate_into_clusters(candidate, args, graph=graph_for_clustering)
        cluster_total_list = []
        cluster_retrain_list = []
        cluster_perturb_list = []
        cluster_avg_num_influenced_nodes = []

        for cluster in clusters:
            cluster_candidate = cluster.unsqueeze(0)
            c_total, c_retrain, c_perturb, c_module_scale, c_inv_hvp_norm, c_num_ins = influence_module.calculate_influence(
                cluster_candidate, influence_type
            )

            cluster_total = c_total.squeeze(0).detach().cpu()
            cluster_retrain = c_retrain.squeeze(0).detach().cpu()
            cluster_perturb = c_perturb.squeeze(0).detach().cpu()

            cluster_total_list.append(cluster_total)
            cluster_retrain_list.append(cluster_retrain)
            cluster_perturb_list.append(cluster_perturb)
            cluster_avg_num_influenced_nodes.append(float(c_num_ins))

            module_scale = c_module_scale
            inv_hvp_norm = c_inv_hvp_norm

        stacked_cluster_total = torch.stack(cluster_total_list, dim=0)
        stacked_cluster_retrain = torch.stack(cluster_retrain_list, dim=0)
        stacked_cluster_perturb = torch.stack(cluster_perturb_list, dim=0)

        sum_total = stacked_cluster_total.sum(dim=0)
        sum_retrain = stacked_cluster_retrain.sum(dim=0)
        sum_perturb = stacked_cluster_perturb.sum(dim=0)
        avg_influenced = float(np.mean(cluster_avg_num_influenced_nodes)) if len(cluster_avg_num_influenced_nodes) > 0 else 0.0

        sum_total_list.append(sum_total)
        sum_retrain_list.append(sum_retrain)
        sum_perturb_list.append(sum_perturb)
        avg_num_influenced_nodes_list.append(avg_influenced)

        per_candidate_details.append(
            {
                "candidate_idx": candidate_idx,
                "candidate_edges": candidate.detach().cpu(),
                "clusters": [cluster.detach().cpu() for cluster in clusters],
                "cluster_total_inf": stacked_cluster_total,
                "cluster_retrain_inf": stacked_cluster_retrain,
                "cluster_perturb_inf": stacked_cluster_perturb,
                "sum_total_inf": sum_total,
                "sum_retrain_inf": sum_retrain,
                "sum_perturb_inf": sum_perturb,
                "avg_num_influenced_nodes_per_cluster": cluster_avg_num_influenced_nodes,
            }
        )

    return {
        "total_inf": torch.stack(sum_total_list, dim=0),
        "retrain_inf": torch.stack(sum_retrain_list, dim=0),
        "perturb_inf": torch.stack(sum_perturb_list, dim=0),
        "module_scale": module_scale,
        "inv_hvp_norm": inv_hvp_norm,
        "avg_num_influenced_nodes": float(np.mean(avg_num_influenced_nodes_list)) if len(avg_num_influenced_nodes_list) > 0 else 0.0,
        "per_candidate": per_candidate_details,
    }


def calculate_clusterwise_step_by_step_influence(
    model,
    graph,
    args,
    eval_node_idxs,
    metric_fn,
    candidates,
    influence_type,
    influence_module_cls,
):
    if candidates.dim() != 3:
        raise ValueError("candidates must be a 3D tensor with shape [num_candidates, num_group_elem, 2].")

    if influence_type not in ["edge_removal", "edge_insertion"]:
        raise ValueError("influence_type must be one of ['edge_removal', 'edge_insertion'].")

    per_candidate_results = []

    for candidate_idx, candidate in enumerate(candidates):
        clusters = _split_candidate_into_clusters(candidate, args, graph=graph)
        num_clusters = len(clusters)
        candidate_permutations = []
        cluster_orderings = itertools.permutations(range(num_clusters))

        for ordering in cluster_orderings:
            ordered_clusters = [clusters[i] for i in ordering]
            step_total_list = []
            step_retrain_list = []
            step_perturb_list = []
            step_module_scale = []
            step_inv_hvp_norm = []
            step_avg_num_influenced_nodes = []

            running_graph = graph.clone()
            for cluster in ordered_clusters:
                step_module = influence_module_cls(
                    model,
                    running_graph,
                    args,
                    args.eval_metric,
                    1,
                    eval_node_idxs,
                    metric_fn,
                )
                cluster_candidate = cluster.unsqueeze(0)
                step_total, step_retrain, step_perturb, module_scale, inv_hvp_norm, num_ins = (
                    step_module.calculate_influence(cluster_candidate, influence_type)
                )

                step_total_list.append(step_total.squeeze(0).detach().cpu())
                step_retrain_list.append(step_retrain.squeeze(0).detach().cpu())
                step_perturb_list.append(step_perturb.squeeze(0).detach().cpu())
                step_module_scale.append(module_scale)
                step_inv_hvp_norm.append(inv_hvp_norm)
                step_avg_num_influenced_nodes.append(num_ins)

                if influence_type == "edge_removal":
                    for edge in cluster:
                        running_graph = remove_edge(running_graph, edge)
                else:
                    for edge in cluster:
                        running_graph = add_edge(running_graph, edge)

            total_steps = torch.stack(step_total_list)
            retrain_steps = torch.stack(step_retrain_list)
            perturb_steps = torch.stack(step_perturb_list)

            candidate_permutations.append(
                {
                    "cluster_order_indices": torch.tensor(ordering, dtype=torch.long),
                    "ordered_clusters": [cluster.detach().cpu() for cluster in ordered_clusters],
                    "step_total_inf": total_steps,
                    "step_retrain_inf": retrain_steps,
                    "step_perturb_inf": perturb_steps,
                    "sum_total_inf": total_steps.sum(dim=0),
                    "sum_retrain_inf": retrain_steps.sum(dim=0),
                    "sum_perturb_inf": perturb_steps.sum(dim=0),
                    "module_scales": step_module_scale,
                    "inv_hvp_norms": step_inv_hvp_norm,
                    "avg_num_influenced_nodes_per_step": step_avg_num_influenced_nodes,
                }
            )

        per_candidate_results.append(
            {
                "candidate_idx": candidate_idx,
                "candidate_edges": candidate.detach().cpu(),
                "clusters": [cluster.detach().cpu() for cluster in clusters],
                "permutations": candidate_permutations,
            }
        )

    return {"per_candidate": per_candidate_results}


def calculate_grouped_influence(
    model,
    graph,
    args,
    eval_node_idxs,
    metric_fn,
    candidates,
    influence_type,
    influence_module_cls,
):
    influence_module = influence_module_cls(model, graph, args, args.eval_metric, 1, eval_node_idxs, metric_fn)
    metric_mode = _normalize_metric_mode(getattr(args, "metric_mode", "global"))

    original_clusterer = getattr(args, "candidate_clusterer_fn", None)
    partition_search_result = None
    if metric_mode == "groupwise" and influence_type == "edge_removal":
        from groupwise_metric import prepare_groupwise_candidate_clusterer

        groupwise_clusterer, partition_search_result = prepare_groupwise_candidate_clusterer(
            candidates=candidates,
            influence_module=influence_module,
            args=args,
            influence_type=influence_type,
        )
        args.candidate_clusterer_fn = groupwise_clusterer
    elif metric_mode == "groupwise":
        print(
            f"[groupwise-metric] Skip metric_mode=groupwise for influence_type={influence_type}; "
            "fall back to global partitioning."
        )
    elif metric_mode == "partition":
        from candidate_partition import prepare_candidate_partition_clusterer

        partition_clusterer, partition_search_result = prepare_candidate_partition_clusterer(
            candidates=candidates,
            graph=graph,
            args=args,
            influence_type=influence_type,
        )
        args.candidate_clusterer_fn = partition_clusterer

    basic_calculate_influence_result = calculate_influence(influence_module, candidates, influence_type)

    mode = _normalize_influence_mode(getattr(args, "influence_calculation_mode", "both"))
    try:
        if mode in ["calculate_influence", "both", "clusterwise_step_by_step"]:
            clusterwise_fixed_theta_result = calculate_clusterwise_fixed_theta_influence(
                influence_module=influence_module,
                candidates=candidates,
                args=args,
                influence_type=influence_type,
            )
        else:
            clusterwise_fixed_theta_result = None

        if mode in ["clusterwise_step_by_step", "both"]:
            step_by_step_result = calculate_clusterwise_step_by_step_influence(
                model=model,
                graph=graph,
                args=args,
                eval_node_idxs=eval_node_idxs,
                metric_fn=metric_fn,
                candidates=candidates,
                influence_type=influence_type,
                influence_module_cls=influence_module_cls,
            )
        else:
            step_by_step_result = None
    finally:
        if original_clusterer is None and hasattr(args, "candidate_clusterer_fn"):
            delattr(args, "candidate_clusterer_fn")
        else:
            args.candidate_clusterer_fn = original_clusterer

    return {
        "calculate_influence": basic_calculate_influence_result,
        "clusterwise_fixed_theta": clusterwise_fixed_theta_result,
        "clusterwise_step_by_step": step_by_step_result,
        "partition_search": partition_search_result,
    }


def _flatten_numeric_values(value):
    if value is None:
        return []

    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()

    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()

    if isinstance(value, (list, tuple)):
        flattened = []
        for elem in value:
            flattened.extend(_flatten_numeric_values(elem))
        return flattened

    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _safe_mean(value):
    flattened = _flatten_numeric_values(value)
    if len(flattened) == 0:
        return None
    return float(np.mean(flattened))


def _edge_group_to_string(edge_group):
    if not torch.is_tensor(edge_group):
        return str(edge_group)

    edge_tensor = edge_group.detach().cpu().to(torch.long)
    if edge_tensor.dim() == 1 and edge_tensor.numel() == 2:
        edge_tensor = edge_tensor.view(1, 2)
    if edge_tensor.dim() != 2 or edge_tensor.shape[1] != 2:
        return str(edge_tensor.tolist())

    return ";".join(f"{int(edge[0])}-{int(edge[1])}" for edge in edge_tensor.tolist())


def _edge_group_list_to_string(edge_groups):
    if edge_groups is None:
        return None
    if not isinstance(edge_groups, (list, tuple)):
        return str(edge_groups)
    return "|".join(_edge_group_to_string(edge_group) for edge_group in edge_groups)


def _int_list_to_string(values):
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        return str(values)
    return ",".join(str(int(v)) for v in values)


def _json_to_string(value):
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _ordering_to_label(ordering):
    if torch.is_tensor(ordering):
        values = [int(v) for v in ordering.detach().cpu().reshape(-1).tolist()]
    else:
        values = [int(v) for v in list(ordering)]
    return "_".join(str(v) for v in values)


def save_candidate_result_tables(
    result_dir,
    file_stem,
    candidates,
    influence_results,
    total_pbrf=None,
    parameter_shift_pbrf=None,
    message_propagation_pbrf=None,
    loo=None,
):
    os.makedirs(result_dir, exist_ok=True)

    calculate_result = influence_results.get("calculate_influence", None)
    cluster_fixed_result = influence_results.get("clusterwise_fixed_theta", None)
    cluster_step_result = influence_results.get("clusterwise_step_by_step", None)
    partition_result = influence_results.get("partition_search", None)

    num_candidates = int(candidates.shape[0])
    partition_values_by_candidate = {}
    if partition_result is not None:
        for candidate_entry in partition_result.get("candidate_partitions", []):
            candidate_idx = int(candidate_entry.get("candidate_idx", -1))
            partition_values_by_candidate[candidate_idx] = {
                "partition_method": candidate_entry.get("partition_method", partition_result.get("partition_method", None)),
                "partition_strategy": candidate_entry.get("partition_strategy", partition_result.get("partition_strategy", None)),
                "partition_num_groups": int(candidate_entry.get("num_groups", 0)),
                "partition_cluster_sizes": _int_list_to_string(candidate_entry.get("cluster_sizes", [])),
                "partition_weighted_cut": candidate_entry.get("weighted_cut", None),
                "partition_runtime_sec": candidate_entry.get("partition_runtime_sec", None),
                "partition_affinity_num_nodes": candidate_entry.get("affinity_num_nodes", None),
                "partition_affinity_num_edges": candidate_entry.get("affinity_num_edges", None),
                "partition_affinity_density": candidate_entry.get("affinity_density", None),
                "partition_affinity_weight_sum": candidate_entry.get("affinity_weight_sum", None),
                "partition_metis_cutcount": candidate_entry.get("metis_cutcount", None),
                "partition_clusters": _edge_group_list_to_string(candidate_entry.get("clusters", [])),
                "partition_owner_partition_histogram": _json_to_string(candidate_entry.get("owner_partition_histogram", None)),
                "partition_cross_owner_candidate_edges": candidate_entry.get("cross_partition_candidate_edges", None),
                "partition_masked_affinity_entries": candidate_entry.get("masked_affinity_entries", None),
                "partition_auto_k_method": candidate_entry.get("auto_k_method", None),
                "partition_auto_k_selected": candidate_entry.get("auto_k_selected", None),
                "partition_auto_k_feasible_k_min": candidate_entry.get("auto_k_feasible_k_min", None),
                "partition_auto_k_feasible_k_max": candidate_entry.get("auto_k_feasible_k_max", None),
                "partition_auto_k_fallback_reason": candidate_entry.get("auto_k_fallback_reason", None),
                "partition_auto_k_eigengap_selected_k": candidate_entry.get("auto_k_eigengap_selected_k", None),
                "partition_auto_k_eigengap_scores": candidate_entry.get("auto_k_eigengap_scores", None),
                "partition_auto_k_silhouette_scores": candidate_entry.get("auto_k_silhouette_scores", None),
                "partition_auto_k_stability_scores": candidate_entry.get("auto_k_stability_scores", None),
                "partition_auto_k_bic_scores": candidate_entry.get("auto_k_bic_scores", None),
                "partition_auto_k_eigenvalues": candidate_entry.get("auto_k_eigenvalues", None),
                "partition_coco_cache_hit": candidate_entry.get("coco_cache_hit", None),
                "partition_coco_line_graph_num_nodes": candidate_entry.get("coco_line_graph_num_nodes", None),
                "partition_coco_line_graph_num_edges": candidate_entry.get("coco_line_graph_num_edges", None),
                "partition_coco_line_graph_density": candidate_entry.get("coco_line_graph_density", None),
                "partition_coco_train_runtime_sec": candidate_entry.get("coco_train_runtime_sec", None),
                "partition_coco_epochs": candidate_entry.get("coco_epochs", None),
                "partition_coco_effective_clusters": candidate_entry.get("coco_effective_clusters", None),
                "partition_coco_best_loss": candidate_entry.get("coco_best_loss", None),
                "partition_coco_device": candidate_entry.get("coco_device", None),
                "partition_coco_full_assignment_missing_edges": candidate_entry.get("coco_full_assignment_missing_edges", None),
                "partition_coco_fallback_to_candidate_line_graph": candidate_entry.get(
                    "coco_fallback_to_candidate_line_graph", None
                ),
                "groupwise_num_groups": int(candidate_entry.get("num_groups", 0)),
                "groupwise_objective_final": candidate_entry.get("objective_final", None),
                "groupwise_objective_global": candidate_entry.get("objective_global", None),
                "groupwise_objective_ratio": candidate_entry.get("objective_ratio_to_global", None),
                "groupwise_start_label": candidate_entry.get("start_label", None),
                "groupwise_clusters": _edge_group_list_to_string(candidate_entry.get("clusters", [])),
            }

    step_perm_values_by_candidate = {}
    all_step_perm_labels = []
    if cluster_step_result is not None:
        for candidate_entry in cluster_step_result.get("per_candidate", []):
            candidate_idx = int(candidate_entry.get("candidate_idx", -1))
            permutations = candidate_entry.get("permutations", [])
            perm_dict = {}
            for perm in permutations:
                perm_label = _ordering_to_label(perm.get("cluster_order_indices", []))
                perm_dict[perm_label] = {
                    "sum_total_inf": _safe_mean(perm.get("sum_total_inf")),
                    "sum_retrain_inf": _safe_mean(perm.get("sum_retrain_inf")),
                    "sum_perturb_inf": _safe_mean(perm.get("sum_perturb_inf")),
                }
                all_step_perm_labels.append(perm_label)
            step_perm_values_by_candidate[candidate_idx] = perm_dict

    all_step_perm_labels = sorted(set(all_step_perm_labels))

    base_fields = [
        "candidate_idx",
        "candidate_edges",
        "num_edges",
        "calculate_influence_total",
        "calculate_influence_parameter_shift",
        "calculate_influence_message_propagation",
        "clusterwise_fixed_theta_total",
        "clusterwise_fixed_theta_parameter_shift",
        "clusterwise_fixed_theta_message_propagation",
        "clusterwise_step_by_step_mean_total",
        "clusterwise_step_by_step_mean_parameter_shift",
        "clusterwise_step_by_step_mean_message_propagation",
        "clusterwise_step_by_step_num_permutations",
        "partition_method",
        "partition_strategy",
        "partition_num_groups",
        "partition_cluster_sizes",
        "partition_weighted_cut",
        "partition_runtime_sec",
        "partition_affinity_num_nodes",
        "partition_affinity_num_edges",
        "partition_affinity_density",
        "partition_affinity_weight_sum",
        "partition_metis_cutcount",
        "partition_clusters",
        "partition_owner_partition_histogram",
        "partition_cross_owner_candidate_edges",
        "partition_masked_affinity_entries",
        "partition_auto_k_method",
        "partition_auto_k_selected",
        "partition_auto_k_feasible_k_min",
        "partition_auto_k_feasible_k_max",
        "partition_auto_k_fallback_reason",
        "partition_auto_k_eigengap_selected_k",
        "partition_auto_k_eigengap_scores",
        "partition_auto_k_silhouette_scores",
        "partition_auto_k_stability_scores",
        "partition_auto_k_bic_scores",
        "partition_auto_k_eigenvalues",
        "partition_coco_cache_hit",
        "partition_coco_line_graph_num_nodes",
        "partition_coco_line_graph_num_edges",
        "partition_coco_line_graph_density",
        "partition_coco_train_runtime_sec",
        "partition_coco_epochs",
        "partition_coco_effective_clusters",
        "partition_coco_best_loss",
        "partition_coco_device",
        "partition_coco_full_assignment_missing_edges",
        "partition_coco_fallback_to_candidate_line_graph",
        "groupwise_num_groups",
        "groupwise_objective_final",
        "groupwise_objective_global",
        "groupwise_objective_ratio",
        "groupwise_start_label",
        "groupwise_clusters",
        "pbrf_total",
        "pbrf_parameter_shift",
        "pbrf_message_propagation",
        "leave_k_out",
    ]
    step_fields = []
    for perm_label in all_step_perm_labels:
        step_fields.extend(
            [
                f"clusterwise_step_by_step_total_perm_{perm_label}",
                f"clusterwise_step_by_step_parameter_shift_perm_{perm_label}",
                f"clusterwise_step_by_step_message_propagation_perm_{perm_label}",
            ]
        )
    fieldnames = base_fields + step_fields

    csv_path = osp.join(result_dir, f"{file_stem}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for candidate_idx in range(num_candidates):
            candidate_edges = candidates[candidate_idx]
            if torch.is_tensor(candidate_edges):
                if candidate_edges.dim() == 1 and candidate_edges.numel() == 2:
                    num_edges = 1
                elif candidate_edges.dim() == 2 and candidate_edges.shape[1] == 2:
                    num_edges = int(candidate_edges.shape[0])
                else:
                    num_edges = int(candidate_edges.numel())
            else:
                num_edges = None

            row = {
                "candidate_idx": candidate_idx,
                "candidate_edges": _edge_group_to_string(candidate_edges),
                "num_edges": num_edges,
                "calculate_influence_total": _safe_mean(calculate_result["total_inf"][candidate_idx]) if calculate_result is not None else None,
                "calculate_influence_parameter_shift": _safe_mean(calculate_result["retrain_inf"][candidate_idx]) if calculate_result is not None else None,
                "calculate_influence_message_propagation": _safe_mean(calculate_result["perturb_inf"][candidate_idx]) if calculate_result is not None else None,
                "clusterwise_fixed_theta_total": _safe_mean(cluster_fixed_result["total_inf"][candidate_idx]) if cluster_fixed_result is not None else None,
                "clusterwise_fixed_theta_parameter_shift": _safe_mean(cluster_fixed_result["retrain_inf"][candidate_idx]) if cluster_fixed_result is not None else None,
                "clusterwise_fixed_theta_message_propagation": _safe_mean(cluster_fixed_result["perturb_inf"][candidate_idx]) if cluster_fixed_result is not None else None,
                "pbrf_total": _safe_mean(total_pbrf[candidate_idx]) if total_pbrf is not None else None,
                "pbrf_parameter_shift": _safe_mean(parameter_shift_pbrf[candidate_idx]) if parameter_shift_pbrf is not None else None,
                "pbrf_message_propagation": _safe_mean(message_propagation_pbrf[candidate_idx]) if message_propagation_pbrf is not None else None,
                "leave_k_out": _safe_mean(loo[candidate_idx]) if loo is not None else None,
            }

            partition_values = partition_values_by_candidate.get(candidate_idx, {})
            row["partition_method"] = partition_values.get("partition_method", None)
            row["partition_strategy"] = partition_values.get("partition_strategy", None)
            row["partition_num_groups"] = partition_values.get("partition_num_groups", None)
            row["partition_cluster_sizes"] = partition_values.get("partition_cluster_sizes", None)
            row["partition_weighted_cut"] = partition_values.get("partition_weighted_cut", None)
            row["partition_runtime_sec"] = partition_values.get("partition_runtime_sec", None)
            row["partition_affinity_num_nodes"] = partition_values.get("partition_affinity_num_nodes", None)
            row["partition_affinity_num_edges"] = partition_values.get("partition_affinity_num_edges", None)
            row["partition_affinity_density"] = partition_values.get("partition_affinity_density", None)
            row["partition_affinity_weight_sum"] = partition_values.get("partition_affinity_weight_sum", None)
            row["partition_metis_cutcount"] = partition_values.get("partition_metis_cutcount", None)
            row["partition_clusters"] = partition_values.get("partition_clusters", None)
            row["partition_owner_partition_histogram"] = partition_values.get("partition_owner_partition_histogram", None)
            row["partition_cross_owner_candidate_edges"] = partition_values.get("partition_cross_owner_candidate_edges", None)
            row["partition_masked_affinity_entries"] = partition_values.get("partition_masked_affinity_entries", None)
            row["partition_auto_k_method"] = partition_values.get("partition_auto_k_method", None)
            row["partition_auto_k_selected"] = partition_values.get("partition_auto_k_selected", None)
            row["partition_auto_k_feasible_k_min"] = partition_values.get("partition_auto_k_feasible_k_min", None)
            row["partition_auto_k_feasible_k_max"] = partition_values.get("partition_auto_k_feasible_k_max", None)
            row["partition_auto_k_fallback_reason"] = partition_values.get("partition_auto_k_fallback_reason", None)
            row["partition_auto_k_eigengap_selected_k"] = partition_values.get(
                "partition_auto_k_eigengap_selected_k", None
            )
            row["partition_auto_k_eigengap_scores"] = partition_values.get("partition_auto_k_eigengap_scores", None)
            row["partition_auto_k_silhouette_scores"] = partition_values.get("partition_auto_k_silhouette_scores", None)
            row["partition_auto_k_stability_scores"] = partition_values.get("partition_auto_k_stability_scores", None)
            row["partition_auto_k_bic_scores"] = partition_values.get("partition_auto_k_bic_scores", None)
            row["partition_auto_k_eigenvalues"] = partition_values.get("partition_auto_k_eigenvalues", None)
            row["partition_coco_cache_hit"] = partition_values.get("partition_coco_cache_hit", None)
            row["partition_coco_line_graph_num_nodes"] = partition_values.get("partition_coco_line_graph_num_nodes", None)
            row["partition_coco_line_graph_num_edges"] = partition_values.get("partition_coco_line_graph_num_edges", None)
            row["partition_coco_line_graph_density"] = partition_values.get("partition_coco_line_graph_density", None)
            row["partition_coco_train_runtime_sec"] = partition_values.get("partition_coco_train_runtime_sec", None)
            row["partition_coco_epochs"] = partition_values.get("partition_coco_epochs", None)
            row["partition_coco_effective_clusters"] = partition_values.get("partition_coco_effective_clusters", None)
            row["partition_coco_best_loss"] = partition_values.get("partition_coco_best_loss", None)
            row["partition_coco_device"] = partition_values.get("partition_coco_device", None)
            row["partition_coco_full_assignment_missing_edges"] = partition_values.get(
                "partition_coco_full_assignment_missing_edges", None
            )
            row["partition_coco_fallback_to_candidate_line_graph"] = partition_values.get(
                "partition_coco_fallback_to_candidate_line_graph", None
            )
            row["groupwise_num_groups"] = partition_values.get("groupwise_num_groups", None)
            row["groupwise_objective_final"] = partition_values.get("groupwise_objective_final", None)
            row["groupwise_objective_global"] = partition_values.get("groupwise_objective_global", None)
            row["groupwise_objective_ratio"] = partition_values.get("groupwise_objective_ratio", None)
            row["groupwise_start_label"] = partition_values.get("groupwise_start_label", None)
            row["groupwise_clusters"] = partition_values.get("groupwise_clusters", None)

            candidate_perm_values = step_perm_values_by_candidate.get(candidate_idx, {})
            if len(candidate_perm_values) > 0:
                row["clusterwise_step_by_step_mean_total"] = _safe_mean(
                    [v["sum_total_inf"] for v in candidate_perm_values.values() if v["sum_total_inf"] is not None]
                )
                row["clusterwise_step_by_step_mean_parameter_shift"] = _safe_mean(
                    [v["sum_retrain_inf"] for v in candidate_perm_values.values() if v["sum_retrain_inf"] is not None]
                )
                row["clusterwise_step_by_step_mean_message_propagation"] = _safe_mean(
                    [v["sum_perturb_inf"] for v in candidate_perm_values.values() if v["sum_perturb_inf"] is not None]
                )
                row["clusterwise_step_by_step_num_permutations"] = len(candidate_perm_values)
            else:
                row["clusterwise_step_by_step_mean_total"] = None
                row["clusterwise_step_by_step_mean_parameter_shift"] = None
                row["clusterwise_step_by_step_mean_message_propagation"] = None
                row["clusterwise_step_by_step_num_permutations"] = 0

            for perm_label in all_step_perm_labels:
                perm_values = candidate_perm_values.get(perm_label, {})
                row[f"clusterwise_step_by_step_total_perm_{perm_label}"] = perm_values.get("sum_total_inf", None)
                row[f"clusterwise_step_by_step_parameter_shift_perm_{perm_label}"] = perm_values.get("sum_retrain_inf", None)
                row[f"clusterwise_step_by_step_message_propagation_perm_{perm_label}"] = perm_values.get("sum_perturb_inf", None)

            writer.writerow(row)

    # Save full step-by-step trajectory as a separate file.
    if cluster_step_result is not None:
        step_csv_path = osp.join(result_dir, f"{file_stem}_step_by_step.csv")
        step_fields = [
            "candidate_idx",
            "permutation_idx",
            "permutation_order",
            "step_idx",
            "cluster_edges",
            "step_total_inf",
            "step_parameter_shift_inf",
            "step_message_propagation_inf",
            "sum_total_inf",
            "sum_parameter_shift_inf",
            "sum_message_propagation_inf",
        ]
        with open(step_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=step_fields)
            writer.writeheader()
            for candidate_entry in cluster_step_result.get("per_candidate", []):
                candidate_idx = int(candidate_entry.get("candidate_idx", -1))
                permutations = candidate_entry.get("permutations", [])
                for perm_idx, perm in enumerate(permutations):
                    order_label = _ordering_to_label(perm.get("cluster_order_indices", []))
                    ordered_clusters = perm.get("ordered_clusters", [])
                    step_total_inf = perm.get("step_total_inf", None)
                    step_retrain_inf = perm.get("step_retrain_inf", None)
                    step_perturb_inf = perm.get("step_perturb_inf", None)
                    num_steps = len(ordered_clusters)

                    for step_idx in range(num_steps):
                        row = {
                            "candidate_idx": candidate_idx,
                            "permutation_idx": perm_idx,
                            "permutation_order": order_label,
                            "step_idx": step_idx,
                            "cluster_edges": _edge_group_to_string(ordered_clusters[step_idx]),
                            "step_total_inf": _safe_mean(step_total_inf[step_idx]) if step_total_inf is not None else None,
                            "step_parameter_shift_inf": _safe_mean(step_retrain_inf[step_idx]) if step_retrain_inf is not None else None,
                            "step_message_propagation_inf": _safe_mean(step_perturb_inf[step_idx]) if step_perturb_inf is not None else None,
                            "sum_total_inf": _safe_mean(perm.get("sum_total_inf")),
                            "sum_parameter_shift_inf": _safe_mean(perm.get("sum_retrain_inf")),
                            "sum_message_propagation_inf": _safe_mean(perm.get("sum_perturb_inf")),
                        }
                        writer.writerow(row)


def save_config(args, filename, dirs):
    if args.json_config == 'none':
        with open(filename, 'w') as f:
            json.dump(vars(args), f, indent=4)

def get_save_id(save_dir):
    if os.path.exists(save_dir):
        file_list = os.listdir(save_dir)
        ids = []
        for file_name in file_list:
            if file_name == 'pbrf_checkpoints':
                continue
            suffix = file_name.split('_')[-1]
            if suffix.isdigit():
                ids.append(int(suffix))

        if len(ids) == 0:
            return 0

        return max(ids) + 1
    else:
        raise ValueError


def reserve_result_dir(result_root):
    os.makedirs(result_root, exist_ok=True)
    save_id = get_save_id(result_root)

    while True:
        candidate_dir = osp.join(result_root, f"{save_id}")
        try:
            # Atomic directory reservation across concurrent processes.
            os.makedirs(candidate_dir, exist_ok=False)
            return save_id, candidate_dir
        except FileExistsError:
            save_id += 1

def get_edge_weight(graph, edge=None, node=None):
    if edge is not None and node is not None:
        raise ValueError
    
    if edge is not None:
        edge_index = graph.edge_index
        if edge.dim() != 1 or edge.numel() != 2:
            raise ValueError("edge must be a 1D tensor with 2 node indices.")

        # Match both directions and keep index tensor 1D even when one match exists.
        u, v = edge[0], edge[1]
        match_mask = torch.logical_or(
            torch.logical_and(edge_index[0] == u, edge_index[1] == v),
            torch.logical_and(edge_index[0] == v, edge_index[1] == u),
        )
        edge_idx = match_mask.nonzero(as_tuple=False).view(-1)

        return graph.edge_weight[edge_idx], edge_idx
    elif node is not None:
        tmp_graph = graph.clone()
        edges = tmp_graph.edge_index.T
        edge_weights = tmp_graph.edge_weight

        mask = (graph.edge_index == node).max(dim=0)[0]

        return None, mask
    else:
        raise ValueError

def add_gradients(grad1, grad2):
    res = []
    if grad1 is None:
        return grad2
    else:
        for grad1_elem, grad2_elem in zip(grad1, grad2):
            if grad1_elem is None and grad2_elem is None:
                res.append(None)
            else:
                res.append(grad1_elem + grad2_elem)
        return res
    
def scale_gradients(grad1, scale):
    res = []
    for grad_elem in grad1:
        if grad_elem is None:
            res.append(None)
        else:
            res.append(grad_elem * scale)

    return res


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def reshape_like_params(vec, params):
    pointer = 0
    split_tensors = []
    params_shape = tuple(p.shape for p in params)
    for dim in params_shape:
        num_param = dim.numel()
        split_tensors.append(vec[pointer: pointer + num_param].view(dim))
        pointer += num_param
    return tuple(split_tensors)


def flatten_parameters(model):
    flatten_params = []
    for p in model.parameters():
        flatten_params.append(p.view(-1))
    return torch.cat(flatten_params)

def flatten_params_like(params_like, param):
    vec = []
    for idx, p in enumerate(params_like):
        if p is None:
            vec.append(torch.zeros_like(param[idx]).view(-1))
        else:
            vec.append(p.view(-1))
    return torch.cat(vec)

def is_within_2std(x: torch.Tensor, k: float=2) -> torch.Tensor:
    """
    Check if each value in the tensor is within mean ± 2 * std.

    Parameters:
    -----------
    x : torch.Tensor
        Input tensor of any shape.

    Returns:
    --------
    torch.Tensor
        Boolean tensor indicating whether each element is within the range.
    """
    finite_mask = torch.isfinite(x)
    if not finite_mask.any():
        return torch.zeros_like(x, dtype=torch.bool)

    finite_x = x[finite_mask]
    mean = finite_x.mean()
    std = finite_x.std(unbiased=False)  # Set to False for population std
    lower = mean - k * std
    upper = mean + k * std
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[finite_mask] = (finite_x >= lower) & (finite_x <= upper)
    return mask

class FixedSciFormatter(ScalarFormatter):
    def _set_format(self):
        self.format = '%1.1f'

def plot_influence_loss(influence, loo, save_dir, save_name, args, margin_rate=0.1, xlabel="Estimated Influence", ylabel="Actual Influence", title=None, mask=None, r_size=None):
    os.makedirs(save_dir,exist_ok=True)
    plt.figure(figsize=(8, 6))

    influence = torch.as_tensor(influence).reshape(-1)
    loo_tensor = torch.as_tensor(loo, device=influence.device, dtype=influence.dtype).reshape(-1)
    common_n = min(influence.numel(), loo_tensor.numel())
    influence = influence[:common_n]
    loo_tensor = loo_tensor[:common_n]

    valid_mask = torch.isfinite(influence) & torch.isfinite(loo_tensor)

    r_num = None
    if args.element_type == 'edge_edit' and mask is not None and r_size is not None:
        r_num = int(mask[:r_size].sum().item())
        r_num = min(r_num, common_n)
        r_num = int(valid_mask[:r_num].sum().item())

    influence = influence[valid_mask]
    loo_tensor = loo_tensor[valid_mask]

    if args.element_type == 'edge_edit' and r_num is not None:
        plt.scatter(influence[:r_num], loo_tensor[:r_num], color='red', alpha=0.7, s=50, marker='x', linewidths=4, label='Deletion')
        plt.scatter(influence[r_num:], loo_tensor[r_num:], color='blue', alpha=0.7, s=50, marker='o', label='Insertion')
        handles, labels = plt.gca().get_legend_handles_labels()
        #legend = plt.legend(handles, labels, fontsize=23, loc='best', framealpha=0.6, markerscale=2.0)
    else:
        #plt.scatter(influence, loo, color='red', alpha=0.7, s=50, marker='x', linewidths=4)
        plt.scatter(influence, loo_tensor, color='blue', alpha=0.7, s=50)
    
    if influence.numel() < 2 or loo_tensor.numel() < 2:
        correlation = torch.tensor(float("nan"), device=influence.device, dtype=influence.dtype)
    else:
        correlation = torch.corrcoef(torch.stack((influence, loo_tensor)))[0, 1]

    plt.xlabel(xlabel, fontsize=26)
    plt.ylabel(ylabel, fontsize=26)
    
    if title is not None:
        if title == "mean_validation_loss":
            plt.title("Validation Loss", fontsize=30)
        elif title == "feature_ablation":
            plt.title("Over-squashing", fontsize=30)
        elif title == "dirichlet_energy":
            plt.title("Dirichlet Energy", fontsize=30)
        elif title == "Citeseer":
            plt.title("CiteSeer", fontsize=30)
        elif title == "Pubmed":
            plt.title("PubMed", fontsize=30)
        else:
            plt.title(title, fontsize=30)

    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    ax = plt.gca()

    x_formatter = ScalarFormatter(useMathText=True)
    x_formatter.set_scientific(True)
    x_formatter.set_powerlimits((0, 0))
    x_formatter.set_useOffset(True)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.offsetText.set_fontsize(24)

    y_formatter = FixedSciFormatter(useMathText=True)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((0, 0))
    y_formatter.set_useOffset(True)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.yaxis.offsetText.set_fontsize(24)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    # Display the plot
    plt.grid(alpha=0.3)

    if influence.numel() == 0 or loo_tensor.numel() == 0:
        ax.text(
            0.5,
            0.5,
            "No valid points after filtering",
            transform=ax.transAxes,
            fontsize=16,
            ha='center',
            va='center',
        )
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{save_name}.png', bbox_inches='tight')
        plt.savefig(f'{save_dir}/{save_name}.pdf', bbox_inches='tight')
        plt.clf()
        return

    min_val = torch.minimum(influence, loo_tensor).min().item()
    max_val = torch.maximum(influence, loo_tensor).max().item()
    margin = (max_val - min_val)  * margin_rate

    ax.text(0.98, 0.02, f"Correlation: {correlation:.2f}",
        transform=ax.transAxes,  
        fontsize=22, ha='right', va='bottom',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], color='red', linestyle='--', linewidth=5)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{save_name}.png', bbox_inches='tight')
    plt.savefig(f'{save_dir}/{save_name}.pdf', bbox_inches='tight')
    plt.clf()


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, seed=12134):
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]
    #print(test_idx)

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    
    return data

def display_progress(text, current_step, last_step, enabled=True,
                     fix_zero_start=True):
    """Draws a progress indicator on the screen with the text preceeding the
    progress

    Arguments:
        test: str, text displayed to describe the task being executed
        current_step: int, current step of the iteration
        last_step: int, last possible step of the iteration
        enabled: bool, if false this function will not execute. This is
            for running silently without stdout output.
        fix_zero_start: bool, if true adds 1 to each current step so that the
            display starts at 1 instead of 0, which it would for most loops
            otherwise.
    """
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text)
                                   + len(str(current_step))
                                   + len(str(last_step))
                                   + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step-1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")

    sys.stdout.flush()


def make_dirs(args):
    model_hparams = osp.join(args.model, f'linear_{args.linear}_bias_{args.bias}', args.dataset, f'layer_{args.num_layers}')
    learning_hparams = f"{args.lr}_{args.hidden_dim}_{args.epochs}_{args.weight_decay}"
    calculate_hparams = osp.join(args.element_type, f'{args.num_group_elem}edges')

    vanilla_model_dir = osp.join('checkpoints', "vanilla", model_hparams, learning_hparams)

    result_root = osp.join('results', args.hessian_type, args.eval_metric, model_hparams, calculate_hparams)
    os.makedirs(vanilla_model_dir, exist_ok=True)
    os.makedirs(result_root, exist_ok=True)
    
    loo_model_root = osp.join('checkpoints', 'loo_checkpoints', model_hparams, calculate_hparams)
    loo_model_dir = osp.join(loo_model_root, learning_hparams)
    
    pbrf_model_root = osp.join('checkpoints', 'pbrf_checkpoints', model_hparams, calculate_hparams)
    pbrf_model_dir = osp.join(pbrf_model_root, learning_hparams, f'{args.damp}_{args.pbrf_epochs}_{args.pbrf_weight_decay}')
    
    if args.json_config != "none":
        result_id = None
        result_dir = osp.join("configs", "results", f"{args.json_config[:-5]}")
    else:
        result_id, result_dir = reserve_result_dir(result_root)

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(loo_model_dir, exist_ok=True)
    os.makedirs(pbrf_model_dir, exist_ok=True)

    if args.fig_title == "none":
        fig_title = args.eval_metric
    else:
        fig_title = args.fig_title

    dirs = dict()
    dirs = {'vanilla': vanilla_model_dir, 
            'result': result_dir,
            'result_root': result_root,
            "result_id": result_id,
            "loo_model": loo_model_dir,
            "fig_title": fig_title,
            "pbrf_model": pbrf_model_dir
            }

    return dirs


def rename_result_dir(args, retrain_inf, retrain_pbrf, perturb_inf, perturb_pbrf, dirs):
    if args.json_config == "none":
        if not osp.isdir(dirs['result']):
            print(f"[rename_result_dir] Skip: source dir missing: {dirs['result']}")
            return
        # Keep all vectors 1D to avoid scalar-vs-vector stack errors when num_candidates == 1.
        torch_retrain_inf = retrain_inf.detach().clone().reshape(-1).to(torch.float32).cpu()
        torch_retrain_pbrf = torch.as_tensor(retrain_pbrf, dtype=torch.float32).reshape(-1).cpu()
        torch_perturb_inf = perturb_inf.detach().clone().reshape(-1).to(torch.float32).cpu()
        torch_perturb_pbrf = torch.as_tensor(perturb_pbrf, dtype=torch.float32).reshape(-1).cpu()

        common_n = min(
            torch_retrain_inf.numel(),
            torch_retrain_pbrf.numel(),
            torch_perturb_inf.numel(),
            torch_perturb_pbrf.numel(),
        )
        if common_n == 0:
            print("[rename_result_dir] Skip: empty influence arrays.")
            return

        torch_retrain_inf = torch_retrain_inf[:common_n]
        torch_retrain_pbrf = torch_retrain_pbrf[:common_n]
        torch_perturb_inf = torch_perturb_inf[:common_n]
        torch_perturb_pbrf = torch_perturb_pbrf[:common_n]

        if common_n < 2:
            retrain_corr = torch.tensor(float("nan"))
            perturb_corr = torch.tensor(float("nan"))
        else:
            retrain_corr = torch.corrcoef(torch.stack((torch_retrain_inf, torch_retrain_pbrf)))[0, 1]
            perturb_corr = torch.corrcoef(torch.stack((torch_perturb_inf, torch_perturb_pbrf)))[0, 1]

        l2_error = torch.norm(torch_retrain_inf - torch_retrain_pbrf, p=2).item()
        new_save_dir = osp.join(dirs['result_root'], f"{retrain_corr:.2f}_{l2_error:.4f}_{perturb_corr:.2f}_{dirs['result_id']}")
        if osp.exists(new_save_dir):
            print(f"[rename_result_dir] Destination exists, skipping move: {new_save_dir}")
            return
        try:
            shutil.move(dirs['result'], new_save_dir)
            dirs['result'] = new_save_dir
        except FileNotFoundError:
            # Another concurrent run may have moved/removed this path.
            print(f"[rename_result_dir] Skip: source dir missing during move: {dirs['result']}")

    return
