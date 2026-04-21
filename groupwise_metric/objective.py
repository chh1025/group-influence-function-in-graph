from typing import Dict, Iterable, List, Optional

import torch

from .chains import build_group_chain, compute_group_radius
from .metrics import compute_effective_radius, compute_group_consistency
from .proxies import compute_group_proxy_center, compute_group_proxy_dispersion


def compute_partition_objective(
    groups,
    h,
    Q,
    chain_method="nearest_neighbor",
    damping=1e-3,
    radius_update_rho=1.0,
    metric_energy_mode="scalar_proxy",
    prev_group_stats=None,
):
    total_objective = 0.0
    per_group_stats = []
    prev_group_stats = prev_group_stats or {}

    for group_id, group_indices in enumerate(groups):
        group_indices = _sorted_group(group_indices)
        if len(group_indices) == 0:
            continue

        M_g = compute_group_proxy_center(Q, group_indices)
        tau_g = compute_group_proxy_dispersion(Q, group_indices, M_g=M_g)
        ordered_group_indices = build_group_chain(h, group_indices, method=chain_method)
        r_g = compute_group_radius(h, ordered_group_indices)
        consistency = compute_group_consistency(
            h,
            ordered_group_indices,
            M_g=M_g,
            damping=damping,
            metric_energy_mode=metric_energy_mode,
        )

        group_key = tuple(group_indices)
        prev_d_g = None
        if group_key in prev_group_stats:
            prev_d_g = prev_group_stats[group_key].get("d_g", None)
        d_g = compute_effective_radius(r_g, consistency["F_g"], prev_d_g=prev_d_g, rho=radius_update_rho)
        objective_contrib = float(tau_g) * float(d_g ** 2)
        total_objective += objective_contrib

        per_group_stats.append(
            {
                "group_id": group_id,
                "group_indices": group_indices,
                "group_key": group_key,
                "size": len(group_indices),
                "M_g": M_g.detach().cpu(),
                "tau_g": float(tau_g),
                "r_g": float(r_g),
                "F_g": float(consistency["F_g"]),
                "d_g": float(d_g),
                "objective_contrib": float(objective_contrib),
                "chain": ordered_group_indices,
                "c_g": float(consistency["c_g"]),
                "per_link_energies": consistency["per_link_energies"],
                "per_link_sq_dists": consistency["per_link_sq_dists"],
                "H_op_g": float(consistency["H_op_g"]),
                "w_g": float(consistency["w_g"]),
            }
        )

    return {
        "J": float(total_objective),
        "groups": [_sorted_group(group) for group in groups if len(group) > 0],
        "per_group": per_group_stats,
        "group_stats_by_key": {entry["group_key"]: entry for entry in per_group_stats},
    }


def compute_global_baseline_objective(h, Q, config):
    return compute_partition_objective(
        groups=[list(range(int(h.shape[0])))],
        h=h,
        Q=Q,
        chain_method=config.chain_method,
        damping=config.damping,
        radius_update_rho=config.radius_update_rho,
        metric_energy_mode=config.metric_energy_mode,
    )


def _sorted_group(group_indices):
    if torch.is_tensor(group_indices):
        values = [int(v) for v in group_indices.detach().cpu().reshape(-1).tolist()]
    else:
        values = [int(v) for v in group_indices]
    return sorted(set(values))
