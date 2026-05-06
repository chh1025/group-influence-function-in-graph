import torch

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


def _as_edge_tensor(candidate_edges):
    edge_tensor = torch.as_tensor(candidate_edges, dtype=torch.long)
    if edge_tensor.dim() == 1:
        edge_tensor = edge_tensor.view(1, 2)
    if edge_tensor.dim() == 3 and edge_tensor.shape[0] == 1:
        edge_tensor = edge_tensor.squeeze(0)
    if edge_tensor.dim() != 2 or edge_tensor.shape[1] != 2:
        raise ValueError("candidate_edges must have shape [num_edges, 2] or [1, num_edges, 2].")
    return edge_tensor
