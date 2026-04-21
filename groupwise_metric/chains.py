from typing import List

import torch


def build_group_chain(h, group_indices, method="nearest_neighbor") -> List[int]:
    group_indices = _to_index_list(group_indices)
    if len(group_indices) <= 1:
        return group_indices

    if method not in {"nearest_neighbor", "center_distance"}:
        raise ValueError(f"Unknown chain method: {method}")

    group_h = h[group_indices]
    center = group_h.mean(dim=0, keepdim=True)

    if method == "center_distance":
        center_dists = torch.norm(group_h - center, dim=1)
        ordering = torch.argsort(center_dists, stable=True).tolist()
        return [group_indices[idx] for idx in ordering]

    center_dists = torch.norm(group_h - center, dim=1)
    anchor_local = int(torch.argmin(center_dists).item())
    remaining = set(range(len(group_indices)))
    remaining.remove(anchor_local)
    order_local = [anchor_local]

    while remaining:
        current = order_local[-1]
        current_vec = group_h[current].unsqueeze(0)
        candidate_locals = sorted(remaining)
        candidate_tensor = torch.tensor(candidate_locals, device=group_h.device, dtype=torch.long)
        candidate_h = group_h[candidate_tensor]
        dists = torch.norm(candidate_h - current_vec, dim=1)
        next_local = candidate_locals[int(torch.argmin(dists).item())]
        order_local.append(next_local)
        remaining.remove(next_local)

    return [group_indices[idx] for idx in order_local]


def compute_group_radius(h, ordered_group_indices) -> float:
    ordered_group_indices = _to_index_list(ordered_group_indices)
    if len(ordered_group_indices) <= 1:
        return 0.0

    idx = torch.tensor(ordered_group_indices, device=h.device, dtype=torch.long)
    diffs = h[idx[1:]] - h[idx[:-1]]
    if diffs.numel() == 0:
        return 0.0
    link_norms = torch.norm(diffs, dim=1)
    return float(link_norms.max().item())


def _to_index_list(group_indices):
    if torch.is_tensor(group_indices):
        return [int(v) for v in group_indices.detach().cpu().reshape(-1).tolist()]
    return [int(v) for v in group_indices]
