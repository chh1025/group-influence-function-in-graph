from typing import Iterable

import torch


def generate_probe_vectors(dim, probe_dim, seed, device, dtype=torch.float32, normalize=True):
    if dim <= 0:
        raise ValueError("dim must be positive.")
    if probe_dim <= 0:
        raise ValueError("probe_dim must be positive.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    probes = torch.randn((probe_dim, dim), generator=generator, dtype=dtype)
    probes = probes.to(device=device, dtype=dtype)

    if normalize:
        norms = probes.norm(dim=1, keepdim=True).clamp_min(1e-12)
        probes = probes / norms

    return probes


def normalize_proxy_matrix(Q):
    if Q.numel() == 0:
        return Q

    mean = Q.mean(dim=0, keepdim=True)
    std = Q.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-12)
    return (Q - mean) / std


def compute_group_proxy_center(Q, group_indices):
    index_tensor = _as_index_tensor(group_indices, device=Q.device)
    return Q[index_tensor].mean(dim=0)


def compute_group_proxy_dispersion(Q, group_indices, M_g=None):
    index_tensor = _as_index_tensor(group_indices, device=Q.device)
    if index_tensor.numel() == 0:
        return 0.0
    if M_g is None:
        M_g = compute_group_proxy_center(Q, index_tensor)
    centered = Q[index_tensor] - M_g
    tau_sq = centered.pow(2).sum(dim=1).mean()
    return float(torch.sqrt(torch.clamp(tau_sq, min=0.0)).item())


def _as_index_tensor(group_indices, device):
    if torch.is_tensor(group_indices):
        return group_indices.to(device=device, dtype=torch.long).reshape(-1)
    if isinstance(group_indices, Iterable):
        return torch.tensor(list(group_indices), device=device, dtype=torch.long)
    raise TypeError(f"Unsupported group_indices type: {type(group_indices)!r}")
