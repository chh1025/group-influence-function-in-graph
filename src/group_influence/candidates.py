import torch

from src.group_influence.baseline_api import compute_single_edge_sum
from src.utils import get_edge_insertion_candidates, get_edge_removal_candidates, set_seed


def build_candidate_set(
    state,
    candidate_type,
    num_candidates,
    influence_type="edge_removal",
    pool_size=None,
    mixed_positive_count=None,
    mixed_negative_count=None,
):
    candidate_type = str(candidate_type).strip().lower()
    num_candidates = int(num_candidates)
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive.")

    requires_pool_scoring = candidate_type in {"top_abs", "mixed", "top_positive_negative"}
    pool_size = int(pool_size) if pool_size is not None else num_candidates
    if requires_pool_scoring:
        pool_size = max(pool_size, num_candidates)

    edge_pool = build_edge_pool(
        state=state,
        influence_type=influence_type,
        pool_size=pool_size,
        min_edges=num_candidates,
    )

    if candidate_type == "random":
        selected_edges = edge_pool[:num_candidates]
        pool_rows = []
    else:
        score_result = compute_single_edge_sum(edge_pool, state, influence_type=influence_type)
        pool_rows = _attach_selection_columns(score_result["edge_rows"])
        selected_indices = select_edge_indices(
            pool_rows=pool_rows,
            candidate_type=candidate_type,
            num_candidates=num_candidates,
            mixed_positive_count=mixed_positive_count,
            mixed_negative_count=mixed_negative_count,
        )
        selected_index_set = set(selected_indices)
        for row in pool_rows:
            row["selected"] = int(row["pool_index"] in selected_index_set)
        selected_edges = edge_pool[torch.tensor(selected_indices, device=edge_pool.device, dtype=torch.long)]

    selected_score_rows = _score_selected_edges(
        selected_edges=selected_edges,
        state=state,
        influence_type=influence_type,
        existing_pool_rows=pool_rows,
    )

    candidate_set_name = candidate_set_label(
        candidate_type=candidate_type,
        num_candidates=num_candidates,
        mixed_positive_count=mixed_positive_count,
        mixed_negative_count=mixed_negative_count,
    )
    return {
        "candidate_set_name": candidate_set_name,
        "candidate_type": candidate_type,
        "edge_pool": edge_pool.detach().cpu(),
        "candidate_edges": selected_edges.detach().cpu(),
        "pool_edge_scores": pool_rows,
        "candidate_edge_scores": selected_score_rows,
    }


def build_edge_pool(state, influence_type="edge_removal", pool_size=100, min_edges=None):
    set_seed(state.seed)
    pool_size = int(pool_size)
    if pool_size <= 0:
        raise ValueError("pool_size must be positive.")

    if influence_type == "edge_removal":
        edges = get_edge_removal_candidates(state.data, pool_size)
    elif influence_type == "edge_insertion":
        edges = get_edge_insertion_candidates(state.data, max(pool_size * 2, pool_size))
    else:
        raise ValueError(f"Unsupported influence_type: {influence_type}")

    available_edges = int(edges.shape[0])
    min_edges = pool_size if min_edges is None else int(min_edges)
    if available_edges < min_edges:
        raise ValueError(f"Only {available_edges} candidate edges are available; requested at least {min_edges}.")
    return edges[: min(pool_size, available_edges)].detach().clone().to(device=state.data.edge_index.device, dtype=torch.long)


def select_edge_indices(
    pool_rows,
    candidate_type,
    num_candidates,
    mixed_positive_count=None,
    mixed_negative_count=None,
):
    candidate_type = str(candidate_type).strip().lower()
    if candidate_type == "top_abs":
        ranked = sorted(
            pool_rows,
            key=lambda row: (-abs(float(row["single_edge_influence"])), int(row["pool_index"])),
        )
        return [int(row["pool_index"]) for row in ranked[:num_candidates]]

    if candidate_type in {"mixed", "top_positive_negative"}:
        pos_count = int(mixed_positive_count) if mixed_positive_count is not None else num_candidates // 2
        neg_count = int(mixed_negative_count) if mixed_negative_count is not None else num_candidates - pos_count
        if pos_count + neg_count != num_candidates:
            raise ValueError("mixed_positive_count + mixed_negative_count must equal num_candidates.")

        positives = sorted(
            [row for row in pool_rows if float(row["single_edge_influence"]) > 0],
            key=lambda row: (-float(row["single_edge_influence"]), int(row["pool_index"])),
        )
        negatives = sorted(
            [row for row in pool_rows if float(row["single_edge_influence"]) < 0],
            key=lambda row: (float(row["single_edge_influence"]), int(row["pool_index"])),
        )

        selected = [int(row["pool_index"]) for row in positives[:pos_count]]
        selected.extend(int(row["pool_index"]) for row in negatives[:neg_count])

        if len(selected) < num_candidates:
            selected_set = set(selected)
            fallback = sorted(
                [row for row in pool_rows if int(row["pool_index"]) not in selected_set],
                key=lambda row: (-abs(float(row["single_edge_influence"])), int(row["pool_index"])),
            )
            selected.extend(int(row["pool_index"]) for row in fallback[: num_candidates - len(selected)])
        return selected[:num_candidates]

    raise ValueError(f"Unsupported candidate_type: {candidate_type}")


def candidate_set_label(candidate_type, num_candidates, mixed_positive_count=None, mixed_negative_count=None):
    candidate_type = str(candidate_type).strip().lower()
    if candidate_type in {"mixed", "top_positive_negative"}:
        pos_count = int(mixed_positive_count) if mixed_positive_count is not None else int(num_candidates) // 2
        neg_count = int(mixed_negative_count) if mixed_negative_count is not None else int(num_candidates) - pos_count
        return f"top_positive_{pos_count}_top_negative_{neg_count}"
    return f"{candidate_type}_{int(num_candidates)}"


def _attach_selection_columns(edge_rows):
    rows = []
    for row in edge_rows:
        copied = dict(row)
        copied["pool_index"] = int(row["edge_id"])
        copied["selected"] = 0
        rows.append(copied)
    return rows


def _score_selected_edges(selected_edges, state, influence_type, existing_pool_rows):
    by_edge = {}
    for row in existing_pool_rows:
        by_edge[(int(row["u"]), int(row["v"]))] = row

    selected_rows = []
    missing_edges = []
    for selected_idx, edge in enumerate(selected_edges.detach().cpu().tolist()):
        key = (int(edge[0]), int(edge[1]))
        row = by_edge.get(key)
        if row is None:
            missing_edges.append(edge)
            continue
        copied = dict(row)
        copied["candidate_edge_id"] = selected_idx
        selected_rows.append(copied)

    if len(missing_edges) == 0:
        return selected_rows

    score_result = compute_single_edge_sum(
        torch.tensor(missing_edges, device=state.data.edge_index.device, dtype=torch.long),
        state,
        influence_type=influence_type,
    )
    offset = len(selected_rows)
    for idx, row in enumerate(score_result["edge_rows"]):
        copied = dict(row)
        copied["pool_index"] = None
        copied["selected"] = 1
        copied["candidate_edge_id"] = offset + idx
        selected_rows.append(copied)
    return selected_rows
