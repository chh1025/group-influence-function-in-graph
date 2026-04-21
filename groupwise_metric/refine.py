import math

import torch

from .objective import compute_partition_objective


def initialize_groups(h, Q, num_groups, alpha, beta, seed, min_group_size=2):
    num_points = int(h.shape[0])
    if num_points == 0:
        return []

    max_groups = max(1, num_points // max(1, int(min_group_size)))
    num_groups = max(1, min(int(num_groups), max_groups))
    if num_groups == 1:
        return [list(range(num_points))]

    z = _joint_features(h, Q, alpha=alpha, beta=beta)
    labels = _run_kmeans(z, num_clusters=num_groups, seed=seed)
    groups = []
    for cluster_id in range(num_groups):
        group = torch.nonzero(labels == cluster_id, as_tuple=False).reshape(-1)
        if group.numel() > 0:
            groups.append(sorted(int(v) for v in group.tolist()))

    return _normalize_groups(groups, x=z, num_points=num_points, min_group_size=min_group_size)


def propose_split(group_indices, h, Q, alpha, beta, seed, min_group_size=2):
    group_indices = sorted(int(v) for v in group_indices)
    if len(group_indices) < max(2, 2 * int(min_group_size)):
        return None

    z = _joint_features(h[group_indices], Q[group_indices], alpha=alpha, beta=beta)
    labels = _run_kmeans(z, num_clusters=2, seed=seed)
    group1 = [group_indices[idx] for idx in range(len(group_indices)) if int(labels[idx].item()) == 0]
    group2 = [group_indices[idx] for idx in range(len(group_indices)) if int(labels[idx].item()) == 1]
    if len(group1) == 0 or len(group2) == 0:
        half = len(group_indices) // 2
        group1 = group_indices[:half]
        group2 = group_indices[half:]
    if len(group1) < int(min_group_size) or len(group2) < int(min_group_size):
        return None
    return sorted(group1), sorted(group2)


def try_split(groups, group_id, h, Q, config, current_stats):
    target_group = groups[group_id]
    proposal = propose_split(
        target_group,
        h=h,
        Q=Q,
        alpha=config.alpha_repr,
        beta=config.beta_proxy,
        seed=config.seed + group_id,
        min_group_size=config.min_group_size,
    )
    if proposal is None:
        return False, groups, {"delta": 0.0, "reason": "unsplittable"}

    group1, group2 = proposal
    candidate_groups = [group for idx, group in enumerate(groups) if idx != group_id] + [group1, group2]
    candidate_stats = compute_partition_objective(
        candidate_groups,
        h=h,
        Q=Q,
        chain_method=config.chain_method,
        damping=config.damping,
        radius_update_rho=config.radius_update_rho,
        metric_energy_mode=config.metric_energy_mode,
        prev_group_stats=current_stats.get("group_stats_by_key", None),
    )
    delta = float(current_stats["J"] - candidate_stats["J"])
    accepted = delta > float(config.split_threshold)
    return accepted, candidate_stats["groups"] if accepted else groups, {
        "delta": delta,
        "candidate_stats": candidate_stats,
    }


def candidate_merge_pairs(groups, h, Q, alpha, beta, topk=10):
    if len(groups) <= 1:
        return []

    centers = []
    for group in groups:
        z = _joint_features(h[group], Q[group], alpha=alpha, beta=beta)
        centers.append(z.mean(dim=0))
    centers = torch.stack(centers, dim=0)

    pair_distances = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            dist = torch.norm(centers[i] - centers[j]).item()
            pair_distances.append((dist, i, j))
    pair_distances.sort(key=lambda item: (item[0], item[1], item[2]))
    return [(i, j) for _, i, j in pair_distances[: max(1, int(topk))]]


def try_merge(groups, g1, g2, h, Q, config, current_stats):
    if g1 == g2:
        return False, groups, {"delta": 0.0, "reason": "same_group"}

    merged_group = sorted(set(groups[g1]) | set(groups[g2]))
    candidate_groups = []
    for idx, group in enumerate(groups):
        if idx in {g1, g2}:
            continue
        candidate_groups.append(sorted(group))
    candidate_groups.append(merged_group)

    candidate_stats = compute_partition_objective(
        candidate_groups,
        h=h,
        Q=Q,
        chain_method=config.chain_method,
        damping=config.damping,
        radius_update_rho=config.radius_update_rho,
        metric_energy_mode=config.metric_energy_mode,
        prev_group_stats=current_stats.get("group_stats_by_key", None),
    )
    delta = float(current_stats["J"] - candidate_stats["J"])
    accepted = delta > float(config.merge_threshold)
    return accepted, candidate_stats["groups"] if accepted else groups, {
        "delta": delta,
        "candidate_stats": candidate_stats,
    }


def candidate_reassignments(
    groups,
    h,
    Q,
    alpha,
    beta,
    max_candidates_per_group=5,
    target_topk=2,
    min_group_size=2,
):
    if len(groups) <= 1:
        return []

    z = _joint_features(h, Q, alpha=alpha, beta=beta)
    group_centers = []
    for group in groups:
        group_centers.append(z[group].mean(dim=0))
    group_centers = torch.stack(group_centers, dim=0)

    moves = []
    for source_group_id, group in enumerate(groups):
        if len(group) <= max(1, int(min_group_size)):
            continue

        group_tensor = torch.tensor(group, device=z.device, dtype=torch.long)
        center = group_centers[source_group_id].unsqueeze(0)
        dists = torch.norm(z[group_tensor] - center, dim=1)
        ordering = torch.argsort(dists, descending=True, stable=True)
        candidate_points = [int(group_tensor[idx].item()) for idx in ordering[: max_candidates_per_group].tolist()]

        target_dists = torch.norm(group_centers - group_centers[source_group_id].unsqueeze(0), dim=1)
        target_order = torch.argsort(target_dists, stable=True).tolist()
        target_ids = [int(group_id) for group_id in target_order if int(group_id) != source_group_id][: max(1, int(target_topk))]

        for point_idx in candidate_points:
            for target_group_id in target_ids:
                moves.append(
                    {
                        "point_idx": int(point_idx),
                        "source_group_id": int(source_group_id),
                        "target_group_id": int(target_group_id),
                    }
                )

    return moves


def try_reassignment(groups, move, h, Q, config, current_stats):
    point_idx = int(move["point_idx"])
    source_group_id = int(move["source_group_id"])
    target_group_id = int(move["target_group_id"])

    source_group = [idx for idx in groups[source_group_id] if int(idx) != point_idx]
    if len(source_group) < int(config.min_group_size):
        return False, groups, {"delta": 0.0, "reason": "empty_source"}

    target_group = sorted(set(groups[target_group_id] + [point_idx]))
    candidate_groups = []
    for idx, group in enumerate(groups):
        if idx == source_group_id:
            candidate_groups.append(sorted(source_group))
        elif idx == target_group_id:
            candidate_groups.append(sorted(target_group))
        else:
            candidate_groups.append(sorted(group))
    candidate_groups = [group for group in candidate_groups if len(group) > 0]

    candidate_stats = compute_partition_objective(
        candidate_groups,
        h=h,
        Q=Q,
        chain_method=config.chain_method,
        damping=config.damping,
        radius_update_rho=config.radius_update_rho,
        metric_energy_mode=config.metric_energy_mode,
        prev_group_stats=current_stats.get("group_stats_by_key", None),
    )
    delta = float(current_stats["J"] - candidate_stats["J"])
    accepted = delta > float(config.move_threshold)
    return accepted, candidate_stats["groups"] if accepted else groups, {
        "delta": delta,
        "candidate_stats": candidate_stats,
    }


def _joint_features(h, Q, alpha, beta):
    alpha = float(alpha)
    beta = float(beta)
    alpha = max(alpha, 0.0)
    beta = max(beta, 0.0)
    h_scale = math.sqrt(alpha) if alpha > 0.0 else 0.0
    q_scale = math.sqrt(beta) if beta > 0.0 else 0.0
    return torch.cat([h * h_scale, Q * q_scale], dim=1)


def _run_kmeans(x, num_clusters, seed, max_iters=50):
    num_points = int(x.shape[0])
    if num_clusters <= 1 or num_points <= 1:
        return torch.zeros(num_points, dtype=torch.long, device=x.device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_points, generator=generator)[:num_clusters]
    centers = x[perm.to(device=x.device)]
    labels = torch.zeros(num_points, dtype=torch.long, device=x.device)

    for _ in range(max_iters):
        distances = torch.cdist(x, centers)
        next_labels = torch.argmin(distances, dim=1)
        if torch.equal(next_labels, labels):
            break
        labels = next_labels
        next_centers = []
        for cluster_id in range(num_clusters):
            members = x[labels == cluster_id]
            if members.numel() == 0:
                replacement_idx = int(torch.randint(0, num_points, (1,), generator=generator).item())
                next_centers.append(x[replacement_idx])
            else:
                next_centers.append(members.mean(dim=0))
        centers = torch.stack(next_centers, dim=0)

    return labels


def _normalize_groups(groups, x, num_points=None, min_group_size=2):
    normalized = []
    for group in groups:
        group = sorted(set(int(v) for v in group))
        if len(group) > 0:
            normalized.append(group)

    min_group_size = max(1, int(min_group_size))
    if min_group_size > 1 and len(normalized) > 1:
        normalized = _merge_small_groups(normalized, x=x, min_group_size=min_group_size)

    if len(normalized) == 0 and num_points is not None and num_points > 0:
        normalized = [list(range(num_points))]
    return normalized


def _merge_small_groups(groups, x, min_group_size):
    groups = [sorted(group) for group in groups if len(group) > 0]
    while True:
        small_group_ids = [idx for idx, group in enumerate(groups) if len(group) < min_group_size]
        if len(small_group_ids) == 0 or len(groups) <= 1:
            break

        source_id = small_group_ids[0]
        source_group = groups[source_id]
        source_center = x[source_group].mean(dim=0, keepdim=True)
        candidate_ids = [idx for idx in range(len(groups)) if idx != source_id]
        candidate_distances = []
        for target_id in candidate_ids:
            target_center = x[groups[target_id]].mean(dim=0, keepdim=True)
            candidate_distances.append((torch.norm(source_center - target_center).item(), target_id))
        candidate_distances.sort(key=lambda item: (item[0], item[1]))
        _, target_id = candidate_distances[0]
        groups[target_id] = sorted(set(groups[target_id] + source_group))
        del groups[source_id]

    return groups
