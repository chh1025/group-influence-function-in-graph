import time

import numpy as np
import torch

from candidate_partition_graph import build_candidate_affinity_graph
from cluster_k_selection import choose_num_clusters_for_affinity_graph
from line_graph_coco import (
    build_line_graph_affinity_for_edges,
    canonical_undirected_edges_from_graph,
    cluster_candidate_by_full_line_graph,
    cluster_candidate_with_coco_line_graph,
    get_or_build_full_line_graph_coco_partition,
)
from partition_methods import partition_affinity_graph
from training_graph_partition import (
    assign_candidate_edge_owner_partitions,
    get_or_build_training_graph_metis_partition,
    owner_partition_histogram,
)


PARTITION_STRATEGIES = {
    "candidate_local_affinity",
    "global_training_graph_assignment",
    "hybrid_training_graph_masked_local",
    "coco_full_line_graph_assignment",
    "coco_candidate_line_graph",
}

METIS_GLOBAL_STRATEGIES = {
    "global_training_graph_assignment",
    "hybrid_training_graph_masked_local",
}

COCO_STRATEGIES = {
    "coco_full_line_graph_assignment",
    "coco_candidate_line_graph",
}


def prepare_candidate_partition_clusterer(candidates, graph, args, influence_type):
    if candidates.dim() != 3:
        raise ValueError("candidates must have shape [num_candidates, num_group_elem, 2].")

    partition_method = str(getattr(args, "partition_method", "metis")).strip().lower()
    partition_strategy = str(getattr(args, "partition_strategy", "candidate_local_affinity")).strip().lower()
    num_clusters = int(getattr(args, "num_of_clusters", 1))
    seed = int(getattr(args, "seed", 0))
    hybrid_cross_owner_affinity_scale = float(getattr(args, "hybrid_cross_owner_affinity_scale", 0.0))
    auto_k_method = _auto_k_method(args)

    if partition_strategy not in PARTITION_STRATEGIES:
        raise ValueError(
            f"Unknown partition_strategy: {partition_strategy}. "
            f"Choose from {sorted(PARTITION_STRATEGIES)}."
        )

    if partition_strategy in METIS_GLOBAL_STRATEGIES and partition_method != "metis":
        raise ValueError(
            f"partition_strategy={partition_strategy} currently requires partition_method=metis "
            f"(received: {partition_method})."
        )
    if partition_strategy in COCO_STRATEGIES and partition_method != "coco":
        raise ValueError(
            f"partition_strategy={partition_strategy} requires partition_method=coco "
            f"(received: {partition_method})."
        )
    if partition_method == "coco" and partition_strategy not in COCO_STRATEGIES:
        raise ValueError(
            f"partition_method=coco requires one of {sorted(COCO_STRATEGIES)} "
            f"(received strategy={partition_strategy})."
        )

    partition_lookup = {}
    candidate_diagnostics = []

    training_partition_cache = {}
    training_node_part = None
    training_partition_cache_hit = False
    training_partition_runtime_sec = 0.0
    if partition_strategy in {"global_training_graph_assignment", "hybrid_training_graph_masked_local"} and num_clusters > 1:
        preprocessing_start = time.perf_counter()
        training_node_part, training_partition_cache_hit = get_or_build_training_graph_metis_partition(
            graph=graph,
            num_parts=num_clusters,
            cache=training_partition_cache,
        )
        training_partition_runtime_sec = float(time.perf_counter() - preprocessing_start)

    coco_full_partition_cache = {}
    coco_full_partition = None
    coco_full_partition_cache_hit = False
    coco_full_partition_runtime_sec = 0.0
    coco_full_partition_metadata = {}
    coco_full_auto_k_selection = None
    coco_full_num_clusters = num_clusters
    if partition_strategy == "coco_full_line_graph_assignment" and auto_k_method != "none":
        full_edge_tensor = canonical_undirected_edges_from_graph(graph)
        full_affinity = build_line_graph_affinity_for_edges(full_edge_tensor)
        coco_full_auto_k_selection = _select_auto_k_for_affinity(full_affinity, args)
        coco_full_num_clusters = int(coco_full_auto_k_selection["k"])

    if partition_strategy == "coco_full_line_graph_assignment" and coco_full_num_clusters > 1:
        preprocessing_start = time.perf_counter()
        coco_full_partition, coco_full_partition_cache_hit = get_or_build_full_line_graph_coco_partition(
            graph=graph,
            args=args,
            num_clusters=coco_full_num_clusters,
            cache=coco_full_partition_cache,
        )
        coco_full_partition_runtime_sec = float(time.perf_counter() - preprocessing_start)
        coco_full_partition_metadata = dict(coco_full_partition.metadata)
        if coco_full_auto_k_selection is not None:
            coco_full_partition_metadata["auto_k_selection"] = _sanitize_auto_k_selection(coco_full_auto_k_selection)

    for candidate_idx, candidate in enumerate(candidates):
        start_time = time.perf_counter()

        if partition_strategy == "candidate_local_affinity":
            groups, clusters, partition_stats, extra_diagnostics = _cluster_candidate_local_affinity(
                candidate=candidate,
                graph=graph,
                args=args,
                partition_method=partition_method,
                num_clusters=num_clusters,
                seed=seed + candidate_idx,
            )
        elif partition_strategy == "global_training_graph_assignment":
            groups, clusters, partition_stats, extra_diagnostics = _cluster_candidate_global_assignment(
                candidate=candidate,
                num_clusters=num_clusters,
                node_part=training_node_part,
                partition_method=partition_method,
            )
        elif partition_strategy == "hybrid_training_graph_masked_local":
            groups, clusters, partition_stats, extra_diagnostics = _cluster_candidate_hybrid_masked_local(
                candidate=candidate,
                graph=graph,
                args=args,
                partition_method=partition_method,
                num_clusters=num_clusters,
                seed=seed + candidate_idx,
                node_part=training_node_part,
                hybrid_cross_owner_affinity_scale=hybrid_cross_owner_affinity_scale,
            )
        elif partition_strategy == "coco_full_line_graph_assignment":
            groups, clusters, partition_stats, extra_diagnostics = _cluster_candidate_coco_full_line_graph_assignment(
                candidate=candidate,
                graph=graph,
                args=args,
                partition_method=partition_method,
                num_clusters=coco_full_num_clusters,
                seed=seed + candidate_idx,
                full_partition=coco_full_partition,
                full_partition_cache_hit=coco_full_partition_cache_hit,
                full_partition_metadata=coco_full_partition_metadata,
                auto_k_selection=coco_full_auto_k_selection,
            )
        elif partition_strategy == "coco_candidate_line_graph":
            groups, clusters, partition_stats, extra_diagnostics = _cluster_candidate_coco_line_graph(
                candidate=candidate,
                graph=graph,
                args=args,
                partition_method=partition_method,
                num_clusters=num_clusters,
                seed=seed + candidate_idx,
            )
        else:
            raise AssertionError(f"Unhandled partition_strategy: {partition_strategy}")

        runtime_sec = float(time.perf_counter() - start_time)
        partition_lookup[_candidate_key(candidate)] = clusters

        candidate_diagnostics.append(
            {
                "candidate_idx": int(candidate_idx),
                "candidate_edges": candidate.detach().cpu(),
                "clusters": [cluster.detach().cpu() for cluster in clusters],
                "partition_method": partition_stats["partition_method"],
                "partition_strategy": partition_strategy,
                "num_groups": int(partition_stats["num_groups"]),
                "cluster_sizes": list(partition_stats["cluster_sizes"]),
                "weighted_cut": float(partition_stats["weighted_cut"]) if partition_stats["weighted_cut"] is not None else None,
                "partition_runtime_sec": runtime_sec,
                "affinity_num_nodes": partition_stats.get("num_affinity_nodes", None),
                "affinity_num_edges": partition_stats.get("num_affinity_edges", None),
                "affinity_density": partition_stats.get("affinity_density", None),
                "affinity_weight_sum": partition_stats.get("affinity_weight_sum", None),
                "metis_cutcount": partition_stats.get("metis_cutcount", None),
                **extra_diagnostics,
            }
        )
        print(
            f"[candidate-partition] candidate={candidate_idx} strategy={partition_strategy} method={partition_method} "
            f"groups={partition_stats['num_groups']} weighted_cut={float(partition_stats['weighted_cut']):.6f} "
            f"runtime={runtime_sec:.4f}s"
        )

    def _clusterer(candidate, _args, _graph):
        key = _candidate_key(candidate)
        if key not in partition_lookup:
            raise KeyError("Candidate was not precomputed for partition clustering.")
        return [cluster.detach().clone().to(candidate.device) for cluster in partition_lookup[key]]

    diagnostics = {
        "metric_mode": "partition",
        "partition_method": partition_method,
        "partition_strategy": partition_strategy,
        "influence_type": str(influence_type),
        "num_requested_clusters": num_clusters,
        "num_effective_clusters": int(coco_full_num_clusters if partition_strategy == "coco_full_line_graph_assignment" else num_clusters),
        "auto_k_method": auto_k_method,
        "auto_k_global_selection": _sanitize_auto_k_selection(coco_full_auto_k_selection),
        "hybrid_cross_owner_affinity_scale": hybrid_cross_owner_affinity_scale,
        "training_graph_partition_cache_hit": bool(training_partition_cache_hit),
        "training_graph_partition_runtime_sec": float(training_partition_runtime_sec),
        "coco_full_line_graph_cache_hit": bool(coco_full_partition_cache_hit),
        "coco_full_line_graph_runtime_sec": float(coco_full_partition_runtime_sec),
        "coco_full_line_graph_metadata": coco_full_partition_metadata,
        "candidate_partitions": candidate_diagnostics,
    }
    return _clusterer, diagnostics


def _cluster_candidate_local_affinity(candidate, graph, args, partition_method, num_clusters, seed):
    graph_info = build_candidate_affinity_graph(candidate=candidate, graph=graph, args=args)
    auto_k_selection = None
    if _auto_k_method(args) != "none":
        auto_k_selection = _select_auto_k_for_affinity(graph_info["affinity"], args)
        num_clusters = int(auto_k_selection["k"])
    groups, partition_stats = partition_affinity_graph(
        affinity=graph_info["affinity"],
        method=partition_method,
        num_clusters=num_clusters,
        seed=seed,
    )
    clusters = _groups_to_clusters(candidate, groups)
    extra = _empty_global_partition_diagnostics()
    extra.update(_auto_k_partition_diagnostics(auto_k_selection))
    return groups, clusters, partition_stats, extra


def _cluster_candidate_global_assignment(candidate, num_clusters, node_part, partition_method):
    if num_clusters <= 1:
        groups = [list(range(int(candidate.shape[0])))]
        clusters = _groups_to_clusters(candidate, groups)
        return groups, clusters, _single_group_partition_stats(candidate), _empty_global_partition_diagnostics()

    owner_partitions, cross_partition_candidate_edges = assign_candidate_edge_owner_partitions(candidate, node_part)
    groups = _groups_from_owner_partitions(owner_partitions)
    clusters = _groups_to_clusters(candidate, groups)
    partition_stats = _global_assignment_partition_stats(candidate, groups, partition_method)
    extra = {
        "owner_partition_histogram": owner_partition_histogram(owner_partitions),
        "cross_partition_candidate_edges": int(cross_partition_candidate_edges),
        "masked_affinity_entries": None,
    }
    return groups, clusters, partition_stats, extra


def _cluster_candidate_hybrid_masked_local(
    candidate,
    graph,
    args,
    partition_method,
    num_clusters,
    seed,
    node_part,
    hybrid_cross_owner_affinity_scale,
):
    if num_clusters <= 1:
        groups = [list(range(int(candidate.shape[0])))]
        clusters = _groups_to_clusters(candidate, groups)
        return groups, clusters, _single_group_partition_stats(candidate), _empty_global_partition_diagnostics()

    owner_partitions, cross_partition_candidate_edges = assign_candidate_edge_owner_partitions(candidate, node_part)
    graph_info = build_candidate_affinity_graph(candidate=candidate, graph=graph, args=args)
    affinity, masked_affinity_entries = _apply_cross_owner_affinity_scale(
        affinity=graph_info["affinity"],
        owner_partitions=owner_partitions,
        scale=hybrid_cross_owner_affinity_scale,
    )
    auto_k_selection = None
    if _auto_k_method(args) != "none":
        auto_k_selection = _select_auto_k_for_affinity(affinity, args)
        num_clusters = int(auto_k_selection["k"])
    groups, partition_stats = partition_affinity_graph(
        affinity=affinity,
        method=partition_method,
        num_clusters=num_clusters,
        seed=seed,
    )
    clusters = _groups_to_clusters(candidate, groups)
    extra = {
        "owner_partition_histogram": owner_partition_histogram(owner_partitions),
        "cross_partition_candidate_edges": int(cross_partition_candidate_edges),
        "masked_affinity_entries": int(masked_affinity_entries),
    }
    extra.update(_auto_k_partition_diagnostics(auto_k_selection))
    return groups, clusters, partition_stats, extra


def _cluster_candidate_coco_full_line_graph_assignment(
    candidate,
    graph,
    args,
    partition_method,
    num_clusters,
    seed,
    full_partition,
    full_partition_cache_hit,
    full_partition_metadata,
    auto_k_selection=None,
):
    if num_clusters <= 1:
        groups = [list(range(int(candidate.shape[0])))]
        clusters = _groups_to_clusters(candidate, groups)
        partition_stats = _coco_partition_stats(candidate, groups, partition_method)
        extra = _coco_partition_diagnostics(
            metadata={},
            cache_hit=None,
            full_assignment_missing_edges=0,
            fallback_to_candidate_line_graph=False,
            auto_k_selection=auto_k_selection,
        )
        return groups, clusters, partition_stats, extra

    groups, missing_edges = cluster_candidate_by_full_line_graph(candidate, full_partition)
    fallback_to_candidate_line_graph = groups is None
    if fallback_to_candidate_line_graph:
        groups, metadata = cluster_candidate_with_coco_line_graph(
            candidate=candidate,
            graph=graph,
            args=args,
            num_clusters=num_clusters,
            seed=seed,
        )
    else:
        metadata = dict(full_partition_metadata)

    clusters = _groups_to_clusters(candidate, groups)
    partition_stats = _coco_partition_stats(candidate, groups, partition_method)
    extra = _coco_partition_diagnostics(
        metadata=metadata,
        cache_hit=full_partition_cache_hit if not fallback_to_candidate_line_graph else None,
        full_assignment_missing_edges=missing_edges,
        fallback_to_candidate_line_graph=fallback_to_candidate_line_graph,
        auto_k_selection=auto_k_selection,
    )
    return groups, clusters, partition_stats, extra


def _cluster_candidate_coco_line_graph(candidate, graph, args, partition_method, num_clusters, seed):
    auto_k_selection = None
    if _auto_k_method(args) != "none":
        affinity = build_line_graph_affinity_for_edges(candidate)
        auto_k_selection = _select_auto_k_for_affinity(affinity, args)
        num_clusters = int(auto_k_selection["k"])

    if num_clusters <= 1:
        groups = [list(range(int(candidate.shape[0])))]
        clusters = _groups_to_clusters(candidate, groups)
        partition_stats = _coco_partition_stats(candidate, groups, partition_method)
        return groups, clusters, partition_stats, _coco_partition_diagnostics(
            metadata={},
            cache_hit=None,
            full_assignment_missing_edges=None,
            fallback_to_candidate_line_graph=False,
            auto_k_selection=auto_k_selection,
        )

    groups, metadata = cluster_candidate_with_coco_line_graph(
        candidate=candidate,
        graph=graph,
        args=args,
        num_clusters=num_clusters,
        seed=seed,
    )
    clusters = _groups_to_clusters(candidate, groups)
    partition_stats = _coco_partition_stats(candidate, groups, partition_method)
    extra = _coco_partition_diagnostics(
        metadata=metadata,
        cache_hit=None,
        full_assignment_missing_edges=None,
        fallback_to_candidate_line_graph=False,
        auto_k_selection=auto_k_selection,
    )
    return groups, clusters, partition_stats, extra


def _groups_to_clusters(candidate, groups):
    clusters = []
    for group in groups:
        if len(group) == 0:
            continue
        group_tensor = candidate[torch.tensor(group, device=candidate.device, dtype=torch.long)]
        clusters.append(group_tensor.detach().clone())

    if len(clusters) == 0:
        return [candidate.detach().clone()]
    return clusters


def _groups_from_owner_partitions(owner_partitions):
    grouped = {}
    for edge_idx, owner in enumerate(owner_partitions):
        grouped.setdefault(int(owner), []).append(int(edge_idx))
    return [grouped[owner] for owner in sorted(grouped.keys()) if len(grouped[owner]) > 0]


def _apply_cross_owner_affinity_scale(affinity, owner_partitions, scale):
    masked_affinity = affinity.copy()
    masked_entries = 0
    num_nodes = int(masked_affinity.shape[0])

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if int(owner_partitions[i]) == int(owner_partitions[j]):
                continue
            if float(masked_affinity[i, j]) == 0.0:
                continue
            masked_affinity[i, j] *= float(scale)
            masked_affinity[j, i] *= float(scale)
            masked_entries += 1

    return masked_affinity, int(masked_entries)


def _single_group_partition_stats(candidate):
    num_edges = int(candidate.shape[0])
    return {
        "partition_method": "metis",
        "num_groups": 1,
        "cluster_sizes": [num_edges],
        "weighted_cut": 0.0,
        "num_affinity_nodes": num_edges,
        "num_affinity_edges": 0,
        "affinity_density": 0.0,
        "affinity_weight_sum": 0.0,
        "metis_cutcount": 0,
    }


def _global_assignment_partition_stats(candidate, groups, partition_method):
    num_edges = int(candidate.shape[0])
    return {
        "partition_method": str(partition_method),
        "num_groups": int(len(groups)),
        "cluster_sizes": [int(len(group)) for group in groups],
        "weighted_cut": 0.0,
        "num_affinity_nodes": num_edges,
        "num_affinity_edges": 0,
        "affinity_density": 0.0,
        "affinity_weight_sum": 0.0,
        "metis_cutcount": None,
    }


def _coco_partition_stats(candidate, groups, partition_method):
    affinity = build_line_graph_affinity_for_edges(candidate)
    group_lookup = {}
    for group_idx, group in enumerate(groups):
        for edge_idx in group:
            group_lookup[int(edge_idx)] = int(group_idx)

    weighted_cut = 0.0
    for i in range(affinity.shape[0]):
        for j in range(i + 1, affinity.shape[0]):
            if group_lookup.get(i) != group_lookup.get(j):
                weighted_cut += float(affinity[i, j])

    positive_pairs = int(np.count_nonzero(np.triu(affinity > 0, k=1)))
    num_edges = int(candidate.shape[0])
    return {
        "partition_method": str(partition_method),
        "num_groups": int(len(groups)),
        "cluster_sizes": [int(len(group)) for group in groups],
        "weighted_cut": float(weighted_cut),
        "num_affinity_nodes": num_edges,
        "num_affinity_edges": positive_pairs,
        "affinity_density": _affinity_density(num_edges, positive_pairs),
        "affinity_weight_sum": float(np.triu(affinity, k=1).sum()),
        "metis_cutcount": None,
    }


def _coco_partition_diagnostics(
    metadata,
    cache_hit,
    full_assignment_missing_edges,
    fallback_to_candidate_line_graph,
    auto_k_selection=None,
):
    metadata = metadata or {}
    if auto_k_selection is None:
        auto_k_selection = metadata.get("auto_k_selection", None)
    return {
        "owner_partition_histogram": None,
        "cross_partition_candidate_edges": None,
        "masked_affinity_entries": None,
        "coco_cache_hit": cache_hit,
        "coco_line_graph_num_nodes": metadata.get("line_graph_num_nodes", None),
        "coco_line_graph_num_edges": metadata.get("line_graph_num_edges", None),
        "coco_line_graph_density": metadata.get("line_graph_density", None),
        "coco_train_runtime_sec": metadata.get("coco_train_runtime_sec", None),
        "coco_epochs": metadata.get("coco_epochs", None),
        "coco_effective_clusters": metadata.get("coco_effective_clusters", None),
        "coco_best_loss": metadata.get("coco_best_loss", None),
        "coco_device": metadata.get("coco_device", None),
        "coco_full_assignment_missing_edges": full_assignment_missing_edges,
        "coco_fallback_to_candidate_line_graph": bool(fallback_to_candidate_line_graph),
        **_auto_k_partition_diagnostics(auto_k_selection),
    }


def _empty_global_partition_diagnostics():
    return {
        "owner_partition_histogram": None,
        "cross_partition_candidate_edges": None,
        "masked_affinity_entries": None,
        **_auto_k_partition_diagnostics(None),
    }


def _auto_k_method(args):
    return str(getattr(args, "auto_k_method", "none")).strip().lower()


def _select_auto_k_for_affinity(affinity, args):
    return choose_num_clusters_for_affinity_graph(
        affinity=affinity,
        method=_auto_k_method(args),
        k_min=int(getattr(args, "auto_k_min", 1)),
        k_max=int(getattr(args, "auto_k_max", 8)),
        max_ratio=float(getattr(args, "auto_k_max_ratio", 1.0)),
        min_cluster_size=int(getattr(args, "auto_k_min_cluster_size", 1)),
        max_cluster_size=int(getattr(args, "auto_k_max_cluster_size", 0)),
        tiny_graph_threshold=int(getattr(args, "auto_k_tiny_graph_threshold", 4)),
        num_restarts=int(getattr(args, "auto_k_num_restarts", 10)),
        random_state=int(getattr(args, "auto_k_random_seed", getattr(args, "seed", 0))),
        silhouette_metric=str(getattr(args, "auto_k_silhouette_metric", "euclidean")),
        stability_trials=int(getattr(args, "auto_k_stability_trials", 10)),
        stability_edge_dropout=float(getattr(args, "auto_k_stability_edge_dropout", 0.05)),
        weak_silhouette_threshold=float(getattr(args, "auto_k_weak_silhouette_threshold", 0.05)),
        prefer_smaller_k=_truthy(getattr(args, "auto_k_prefer_smaller_k", True)),
    )


def _auto_k_partition_diagnostics(selection):
    selection = _sanitize_auto_k_selection(selection)
    diagnostics = selection.get("diagnostics", {}) if selection is not None else {}
    return {
        "auto_k_method": selection.get("method", None) if selection is not None else None,
        "auto_k_selected": selection.get("k", None) if selection is not None else None,
        "auto_k_feasible_k_min": diagnostics.get("feasible_k_min", None),
        "auto_k_feasible_k_max": diagnostics.get("feasible_k_max", None),
        "auto_k_fallback_reason": diagnostics.get("fallback_reason", None),
        "auto_k_eigengap_selected_k": diagnostics.get("eigengap_selected_k", None),
        "auto_k_eigengap_scores": _json_string_or_none(diagnostics.get("eigengap_scores", None)),
        "auto_k_silhouette_scores": _json_string_or_none(diagnostics.get("silhouette_scores", None)),
        "auto_k_stability_scores": _json_string_or_none(diagnostics.get("stability_scores", None)),
        "auto_k_bic_scores": _json_string_or_none(diagnostics.get("bic_scores", None)),
        "auto_k_eigenvalues": _json_string_or_none(diagnostics.get("eigenvalues", None)),
    }


def _sanitize_auto_k_selection(selection):
    if selection is None:
        return None
    diagnostics = dict(selection.get("diagnostics", {}))
    sanitized = {
        "k": int(selection.get("k", 1)),
        "method": str(selection.get("method", diagnostics.get("method", "none"))),
        "diagnostics": _sanitize_jsonable(diagnostics),
    }
    return sanitized


def _sanitize_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _sanitize_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _json_string_or_none(value):
    if value is None:
        return None
    import json

    return json.dumps(_sanitize_jsonable(value), sort_keys=True, separators=(",", ":"))


def _truthy(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _canonical_edge(edge):
    if torch.is_tensor(edge):
        u, v = [int(v) for v in edge.detach().cpu().reshape(-1).tolist()]
    else:
        u, v = [int(v) for v in edge]
    return (u, v) if u <= v else (v, u)


def _candidate_key(candidate):
    edge_keys = sorted(_canonical_edge(edge) for edge in candidate)
    return tuple(edge_keys)


def _affinity_density(num_nodes, num_edges):
    if int(num_nodes) <= 1:
        return 0.0
    return float((2.0 * float(num_edges)) / float(int(num_nodes) * (int(num_nodes) - 1)))
