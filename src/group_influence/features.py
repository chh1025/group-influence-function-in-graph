import torch
import torch.nn.functional as F

from src.group_influence.baseline_api import compute_single_edge_sum


def build_cheap_edge_features(
    state,
    candidate_edges,
    score_rows=None,
    influence_type="edge_removal",
    influence_module=None,
):
    edge_tensor = _as_edge_tensor(candidate_edges).detach().cpu()
    aligned_scores = align_score_rows(
        candidate_edges=edge_tensor,
        score_rows=score_rows,
        state=state,
        influence_type=influence_type,
        influence_module=influence_module,
    )

    degrees = _directed_degrees(state.data.edge_index, state.data.num_nodes).cpu()
    representations = _node_representations(state).cpu()

    u = edge_tensor[:, 0]
    v = edge_tensor[:, 1]
    u_rep = representations[u]
    v_rep = representations[v]
    cosine = F.cosine_similarity(u_rep, v_rep, dim=1, eps=1e-12)
    l2_distance = torch.linalg.vector_norm(u_rep - v_rep, dim=1)

    influence = torch.tensor(
        [_float_or_zero(row.get("single_edge_influence")) for row in aligned_scores],
        dtype=torch.float32,
    )
    parameter_shift = torch.tensor(
        [_float_or_zero(row.get("single_edge_parameter_shift")) for row in aligned_scores],
        dtype=torch.float32,
    )
    message_passing = torch.tensor(
        [_float_or_zero(row.get("single_edge_message_passing")) for row in aligned_scores],
        dtype=torch.float32,
    )
    deg_u = degrees[u].float()
    deg_v = degrees[v].float()

    feature_names = [
        "single_edge_influence",
        "abs_single_edge_influence",
        "single_edge_parameter_shift",
        "single_edge_message_passing",
        "degree_u",
        "degree_v",
        "endpoint_logit_cosine",
        "endpoint_logit_l2",
    ]
    features = torch.stack(
        [
            influence,
            influence.abs(),
            parameter_shift,
            message_passing,
            deg_u,
            deg_v,
            cosine,
            l2_distance,
        ],
        dim=1,
    )

    feature_rows = []
    for idx, edge in enumerate(edge_tensor.tolist()):
        row = {
            "candidate_edge_id": idx,
            "u": int(edge[0]),
            "v": int(edge[1]),
        }
        for name, value in zip(feature_names, features[idx].tolist()):
            row[name] = float(value)
        feature_rows.append(row)

    return {
        "feature_type": "cheap",
        "feature_names": feature_names,
        "features": features,
        "feature_rows": feature_rows,
        "score_rows": aligned_scores,
    }


def standardize_features(features, eps=1e-12):
    feature_tensor = torch.as_tensor(features, dtype=torch.float32).detach().cpu()
    mean = feature_tensor.mean(dim=0)
    std = feature_tensor.std(dim=0, unbiased=False)
    safe_std = torch.where(std < eps, torch.ones_like(std), std)
    normalized = (feature_tensor - mean) / safe_std
    return {
        "features": normalized,
        "mean": mean,
        "std": safe_std,
    }


def align_score_rows(candidate_edges, score_rows, state, influence_type="edge_removal", influence_module=None):
    edge_tensor = _as_edge_tensor(candidate_edges)
    if score_rows is None:
        score_result = compute_single_edge_sum(
            edge_tensor.to(state.data.edge_index.device),
            state,
            influence_type=influence_type,
            influence_module=influence_module,
        )
        score_rows = score_result["edge_rows"]

    rows = _coerce_score_rows(score_rows)
    by_candidate_id = {
        int(row["candidate_edge_id"]): row
        for row in rows
        if _has_value(row.get("candidate_edge_id"))
    }
    if len(by_candidate_id) == int(edge_tensor.shape[0]):
        return [_normalized_score_row(idx, edge_tensor[idx], by_candidate_id[idx]) for idx in range(edge_tensor.shape[0])]

    by_edge = {}
    for row in rows:
        if _has_value(row.get("u")) and _has_value(row.get("v")):
            by_edge[(int(row["u"]), int(row["v"]))] = row
            by_edge[(int(row["v"]), int(row["u"]))] = row

    aligned = []
    missing = []
    missing_indices = []
    for idx, edge in enumerate(edge_tensor.tolist()):
        row = by_edge.get((int(edge[0]), int(edge[1])))
        if row is None:
            missing.append(edge)
            missing_indices.append(idx)
        else:
            aligned.append(_normalized_score_row(idx, edge, row))

    if len(missing) == 0:
        return aligned

    scored_missing = compute_single_edge_sum(
        torch.tensor(missing, device=state.data.edge_index.device, dtype=torch.long),
        state,
        influence_type=influence_type,
        influence_module=influence_module,
    )["edge_rows"]
    missing_by_index = {
        missing_idx: _normalized_score_row(missing_idx, edge_tensor[missing_idx], row)
        for missing_idx, row in zip(missing_indices, scored_missing)
    }

    completed = []
    aligned_iter = iter(aligned)
    for idx in range(edge_tensor.shape[0]):
        if idx in missing_by_index:
            completed.append(missing_by_index[idx])
        else:
            completed.append(next(aligned_iter))
    return completed


def _node_representations(state):
    was_training = state.model.training
    state.model.eval()
    with torch.no_grad():
        reps = state.model(state.data).detach()
    if was_training:
        state.model.train()
    return reps


def _directed_degrees(edge_index, num_nodes):
    flat_nodes = edge_index.detach().cpu().reshape(-1)
    return torch.bincount(flat_nodes, minlength=int(num_nodes)).float()


def _as_edge_tensor(candidate_edges):
    edge_tensor = torch.as_tensor(candidate_edges, dtype=torch.long)
    if edge_tensor.dim() == 1:
        edge_tensor = edge_tensor.view(1, 2)
    if edge_tensor.dim() == 3 and edge_tensor.shape[0] == 1:
        edge_tensor = edge_tensor.squeeze(0)
    if edge_tensor.dim() != 2 or edge_tensor.shape[1] != 2:
        raise ValueError("candidate_edges must have shape [num_edges, 2] or [1, num_edges, 2].")
    return edge_tensor


def _coerce_score_rows(score_rows):
    rows = []
    for row in score_rows:
        rows.append({str(key): value for key, value in dict(row).items()})
    return rows


def _normalized_score_row(candidate_edge_id, edge, row):
    return {
        "candidate_edge_id": int(candidate_edge_id),
        "u": int(edge[0]),
        "v": int(edge[1]),
        "single_edge_influence": _float_or_zero(row.get("single_edge_influence")),
        "single_edge_parameter_shift": _float_or_zero(row.get("single_edge_parameter_shift")),
        "single_edge_message_passing": _float_or_zero(row.get("single_edge_message_passing")),
    }


def _has_value(value):
    return value is not None and value != ""


def _float_or_zero(value):
    if value is None or value == "":
        return 0.0
    return float(value)
