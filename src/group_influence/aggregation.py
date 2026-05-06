from dataclasses import replace

import torch

from src.graph_utils import add_edge, remove_edge
from src.group_influence.baseline_api import build_influence_module, compute_heo_oneshot


def compute_independent_cluster_sum(
    candidate_edges,
    cluster_labels,
    state,
    influence_type="edge_removal",
    influence_module=None,
):
    edge_tensor = _as_edge_tensor(candidate_edges).to(state.data.edge_index.device)
    labels = torch.as_tensor(cluster_labels, dtype=torch.long, device=edge_tensor.device)
    if labels.dim() != 1 or labels.shape[0] != edge_tensor.shape[0]:
        raise ValueError("cluster_labels must have shape [num_edges].")

    module = influence_module or build_influence_module(state)
    cluster_rows = []
    total = 0.0
    for cluster_id in sorted(torch.unique(labels).detach().cpu().tolist()):
        cluster_id = int(cluster_id)
        cluster_edges = edge_tensor[labels == cluster_id]
        result = compute_heo_oneshot(
            cluster_edges,
            state,
            influence_type=influence_type,
            influence_module=module,
        )
        total += float(result["total"])
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": int(cluster_edges.shape[0]),
                "cluster_prediction": float(result["total"]),
                "cluster_parameter_shift": float(result["parameter_shift"]),
                "cluster_message_passing": float(result["message_passing"]),
            }
        )

    return {
        "total": float(total),
        "cluster_rows": cluster_rows,
    }


def compute_cluster_sequential_graph_only(
    candidate_edges,
    cluster_labels,
    state,
    influence_type="edge_removal",
    order_policy="cluster_id",
    seed=0,
    independent_cluster_rows=None,
):
    edge_tensor = _as_edge_tensor(candidate_edges).to(state.data.edge_index.device)
    labels = torch.as_tensor(cluster_labels, dtype=torch.long, device=edge_tensor.device)
    if labels.dim() != 1 or labels.shape[0] != edge_tensor.shape[0]:
        raise ValueError("cluster_labels must have shape [num_edges].")

    clusters = _cluster_edge_tensors(edge_tensor, labels)
    order = _cluster_order(
        clusters=clusters,
        order_policy=order_policy,
        seed=seed,
        independent_cluster_rows=independent_cluster_rows,
    )

    current_data = state.data.clone()
    total = 0.0
    step_rows = []
    for step_idx, cluster_id in enumerate(order):
        cluster_edges = clusters[int(cluster_id)]
        step_state = replace(state, data=current_data)
        result = compute_heo_oneshot(
            cluster_edges,
            step_state,
            influence_type=influence_type,
            influence_module=None,
        )
        total += float(result["total"])
        step_rows.append(
            {
                "step_idx": int(step_idx),
                "cluster_id": int(cluster_id),
                "cluster_size": int(cluster_edges.shape[0]),
                "step_prediction": float(result["total"]),
                "step_parameter_shift": float(result["parameter_shift"]),
                "step_message_passing": float(result["message_passing"]),
                "running_total": float(total),
            }
        )
        current_data = _apply_edge_set(current_data, cluster_edges, influence_type)

    return {
        "total": float(total),
        "order_policy": order_policy,
        "order": [int(cluster_id) for cluster_id in order],
        "step_rows": step_rows,
    }


def _as_edge_tensor(candidate_edges):
    edge_tensor = torch.as_tensor(candidate_edges, dtype=torch.long)
    if edge_tensor.dim() == 1:
        edge_tensor = edge_tensor.view(1, 2)
    if edge_tensor.dim() == 3 and edge_tensor.shape[0] == 1:
        edge_tensor = edge_tensor.squeeze(0)
    if edge_tensor.dim() != 2 or edge_tensor.shape[1] != 2:
        raise ValueError("candidate_edges must have shape [num_edges, 2] or [1, num_edges, 2].")
    return edge_tensor


def _cluster_edge_tensors(edge_tensor, labels):
    clusters = {}
    for cluster_id in sorted(torch.unique(labels).detach().cpu().tolist()):
        cluster_id = int(cluster_id)
        clusters[cluster_id] = edge_tensor[labels == cluster_id]
    return clusters


def _cluster_order(clusters, order_policy, seed=0, independent_cluster_rows=None):
    order_policy = str(order_policy).strip().lower()
    cluster_ids = sorted(clusters.keys())
    if order_policy == "cluster_id":
        return cluster_ids
    if order_policy == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        perm = torch.randperm(len(cluster_ids), generator=generator).tolist()
        return [cluster_ids[idx] for idx in perm]
    if order_policy == "cluster_size_asc":
        return sorted(cluster_ids, key=lambda cluster_id: (int(clusters[cluster_id].shape[0]), cluster_id))

    influence_by_cluster = _cluster_influence_lookup(independent_cluster_rows)
    if order_policy == "small_abs_cluster_influence_first":
        return sorted(cluster_ids, key=lambda cluster_id: (abs(influence_by_cluster.get(cluster_id, 0.0)), cluster_id))
    if order_policy == "large_abs_cluster_influence_first":
        return sorted(cluster_ids, key=lambda cluster_id: (-abs(influence_by_cluster.get(cluster_id, 0.0)), cluster_id))

    raise ValueError(f"Unsupported order_policy: {order_policy}")


def _cluster_influence_lookup(independent_cluster_rows):
    if independent_cluster_rows is None:
        return {}
    lookup = {}
    for row in independent_cluster_rows:
        lookup[int(row["cluster_id"])] = float(row["cluster_prediction"])
    return lookup


def _apply_edge_set(data, edge_set, influence_type):
    current = data
    for edge in edge_set:
        if influence_type == "edge_removal":
            current = remove_edge(current, edge)
        elif influence_type == "edge_insertion":
            current = add_edge(current, edge)
        else:
            raise ValueError(f"Unsupported influence_type: {influence_type}")
    return current
