import torch

from .search import build_groupwise_config, search_groupwise_partition


def prepare_groupwise_candidate_clusterer(candidates, influence_module, args, influence_type):
    if candidates.dim() != 3:
        raise ValueError("candidates must be a 3D tensor with shape [num_candidates, num_group_elem, 2].")
    if influence_type != "edge_removal":
        raise NotImplementedError("groupwise metric clustering currently supports only edge_removal candidates.")

    device = influence_module.graph.edge_index.device
    edge_to_repr, edge_to_proxy, proxy_meta = _compute_edge_feature_caches(
        candidates=candidates,
        influence_module=influence_module,
        args=args,
    )
    config = build_groupwise_config(args, default_num_groups=getattr(args, "num_of_clusters", 1))

    partition_lookup = {}
    candidate_diagnostics = []

    for candidate_idx, candidate in enumerate(candidates):
        edge_keys = [_canonical_edge(edge) for edge in candidate]
        h = torch.stack([edge_to_repr[key] for key in edge_keys], dim=0).to(device=device)
        Q = torch.stack([edge_to_proxy[key] for key in edge_keys], dim=0).to(device=device)
        search_result = search_groupwise_partition(h=h, Q=Q, config=config)

        clusters = []
        for group in search_result["groups"]:
            group_tensor = candidate[torch.tensor(group, device=candidate.device, dtype=torch.long)]
            clusters.append(group_tensor.detach().clone())
        if len(clusters) == 0:
            clusters = [candidate.detach().clone()]

        partition_lookup[_candidate_key(candidate)] = clusters
        candidate_diagnostics.append(
            {
                "candidate_idx": int(candidate_idx),
                "candidate_edges": candidate.detach().cpu(),
                "clusters": [cluster.detach().cpu() for cluster in clusters],
                "objective_final": float(search_result["final_stats"]["J"]),
                "objective_global": float(search_result["global_baseline"]["J"]),
                "objective_initialized": float(search_result["initialized_stats"]["J"]),
                "objective_ratio_to_global": float(search_result["objective_ratio_to_global"]),
                "num_groups": len(search_result["groups"]),
                "start_label": search_result["start_label"],
                "history": search_result["history"],
                "per_group": [
                    {
                        "group_id": int(entry["group_id"]),
                        "group_indices": list(entry["group_indices"]),
                        "size": int(entry["size"]),
                        "tau_g": float(entry["tau_g"]),
                        "r_g": float(entry["r_g"]),
                        "F_g": float(entry["F_g"]),
                        "d_g": float(entry["d_g"]),
                        "objective_contrib": float(entry["objective_contrib"]),
                    }
                    for entry in search_result["final_stats"]["per_group"]
                ],
            }
        )

        print(
            f"[groupwise-metric] candidate={candidate_idx} groups={len(search_result['groups'])} "
            f"J_final={search_result['final_stats']['J']:.6f} "
            f"J_global={search_result['global_baseline']['J']:.6f} "
            f"ratio={search_result['objective_ratio_to_global']:.6f}"
        )

    def _clusterer(candidate, _args, _graph):
        key = _candidate_key(candidate)
        if key not in partition_lookup:
            raise KeyError("Candidate was not precomputed for groupwise clustering.")
        return [cluster.detach().clone().to(candidate.device) for cluster in partition_lookup[key]]

    diagnostics = {
        "metric_mode": "groupwise",
        "probe_dim": int(getattr(args, "groupwise_probe_dim", 8)),
        "probe_seed": int(getattr(args, "groupwise_probe_seed", 0)),
        "candidate_partitions": candidate_diagnostics,
        "num_unique_edges": len(edge_to_repr),
        "normalize_proxies": bool(getattr(args, "groupwise_normalize_proxies", 1)),
        "proxy_meta": proxy_meta,
    }
    return _clusterer, diagnostics


def _compute_edge_feature_caches(candidates, influence_module, args):
    unique_edge_keys = []
    seen = set()
    for edge in candidates.reshape(-1, 2):
        key = _canonical_edge(edge)
        if key not in seen:
            seen.add(key)
            unique_edge_keys.append(key)

    unique_edges = torch.tensor(unique_edge_keys, device=influence_module.graph.edge_index.device, dtype=torch.long)
    representations = influence_module.get_edge_representations(unique_edges)

    probe_dim = int(getattr(args, "groupwise_probe_dim", 8))
    probe_seed = int(getattr(args, "groupwise_probe_seed", 0))
    normalize = bool(getattr(args, "groupwise_normalize_proxies", 1))
    probe_vecs = influence_module.get_curvature_probe_vectors(probe_dim=probe_dim, probe_seed=probe_seed)
    proxies = influence_module.compute_edge_curvature_proxies(unique_edges, probe_vecs=probe_vecs, normalize=normalize)

    edge_to_repr = {key: representations[idx].detach().cpu() for idx, key in enumerate(unique_edge_keys)}
    edge_to_proxy = {key: proxies[idx].detach().cpu() for idx, key in enumerate(unique_edge_keys)}
    proxy_meta = {
        "probe_dim": probe_dim,
        "probe_seed": probe_seed,
        "normalize": normalize,
    }
    return edge_to_repr, edge_to_proxy, proxy_meta


def _canonical_edge(edge):
    if torch.is_tensor(edge):
        u, v = [int(x) for x in edge.detach().cpu().reshape(-1).tolist()]
    else:
        u, v = [int(x) for x in edge]
    return (u, v) if u <= v else (v, u)


def _candidate_key(candidate):
    edge_keys = sorted(_canonical_edge(edge) for edge in candidate)
    return tuple(edge_keys)
