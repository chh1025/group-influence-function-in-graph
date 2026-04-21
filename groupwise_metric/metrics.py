import math

import torch


def compute_group_consistency(h, ordered_group_indices, M_g, damping, metric_energy_mode="scalar_proxy"):
    ordered_group_indices = _to_index_list(ordered_group_indices)
    if len(ordered_group_indices) <= 1:
        return {
            "F_g": 0.0,
            "c_g": 0.0,
            "per_link_energies": [],
            "per_link_sq_dists": [],
            "H_op_g": float(damping),
            "w_g": float(damping),
        }

    if metric_energy_mode != "scalar_proxy":
        raise ValueError(f"Unknown metric_energy_mode: {metric_energy_mode}")

    w_g = float(torch.relu(M_g).mean().item() + float(damping))
    H_op_g = max(w_g, 1e-12)

    idx = torch.tensor(ordered_group_indices, device=h.device, dtype=torch.long)
    diffs = h[idx[1:]] - h[idx[:-1]]
    link_sq_dists = diffs.pow(2).sum(dim=1)
    per_link_energies = 0.5 * w_g * link_sq_dists
    F_g = float((2.0 * per_link_energies / H_op_g).max().item()) if per_link_energies.numel() > 0 else 0.0

    return {
        "F_g": F_g,
        "c_g": math.sqrt(max(F_g, 0.0)),
        "per_link_energies": [float(v) for v in per_link_energies.detach().cpu().tolist()],
        "per_link_sq_dists": [float(v) for v in link_sq_dists.detach().cpu().tolist()],
        "H_op_g": float(H_op_g),
        "w_g": float(w_g),
    }


def compute_effective_radius(r_g, F_g, prev_d_g=None, rho=1.0):
    raw_radius = max(float(r_g), math.sqrt(max(float(F_g), 0.0)))
    if prev_d_g is None:
        return raw_radius
    rho = float(rho)
    rho = min(max(rho, 0.0), 1.0)
    return (1.0 - rho) * float(prev_d_g) + rho * raw_radius


def _to_index_list(group_indices):
    if torch.is_tensor(group_indices):
        return [int(v) for v in group_indices.detach().cpu().reshape(-1).tolist()]
    return [int(v) for v in group_indices]
