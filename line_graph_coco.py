import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from CoCo.consistency_loss import Consist_net
from CoCo.kmeans_gpu import kmeans
from CoCo.model import compact_net


@dataclass
class CocoLineGraphPartition:
    edge_keys: list
    labels: np.ndarray
    edge_to_label: dict
    metadata: dict


def get_or_build_full_line_graph_coco_partition(graph, args, num_clusters, cache=None):
    edge_tensor = canonical_undirected_edges_from_graph(graph)
    num_edges = int(edge_tensor.shape[0])
    effective_k = max(1, min(int(num_clusters), num_edges))
    config = _coco_config_from_args(args, effective_k)
    cache_key = _full_partition_cache_key(graph=graph, edge_tensor=edge_tensor, config=config)

    cache_store = cache if cache is not None else getattr(graph, "_coco_line_graph_partition_cache", None)
    if cache_store is None:
        cache_store = {}

    if cache_key in cache_store:
        return cache_store[cache_key], True

    cache_path = _disk_cache_path(config, cache_key)
    if cache_path is not None and cache_path.is_file() and not bool(config["force_retrain"]):
        partition = _load_partition_cache(cache_path)
        cache_store[cache_key] = partition
        if cache is None:
            graph._coco_line_graph_partition_cache = cache_store
        return partition, True

    start_time = time.perf_counter()
    features, adjacency, edge_keys = build_line_graph_arrays(
        graph=graph,
        edge_tensor=edge_tensor,
        max_nodes=int(config["max_line_graph_nodes"]),
    )
    labels, train_metadata = run_coco_clustering(
        features=features,
        adjacency=adjacency,
        num_clusters=effective_k,
        config=config,
    )
    metadata = {
        **train_metadata,
        "line_graph_num_nodes": int(features.shape[0]),
        "line_graph_num_edges": int(np.count_nonzero(np.triu(adjacency > 0, k=1))),
        "line_graph_density": _affinity_density(
            int(features.shape[0]),
            int(np.count_nonzero(np.triu(adjacency > 0, k=1))),
        ),
        "cache_key": cache_key,
        "cache_path": str(cache_path) if cache_path is not None else None,
        "preprocess_runtime_sec": float(time.perf_counter() - start_time),
    }
    partition = _build_partition(edge_keys=edge_keys, labels=labels, metadata=metadata)

    if cache_path is not None:
        _save_partition_cache(cache_path, partition)

    cache_store[cache_key] = partition
    if cache is None:
        graph._coco_line_graph_partition_cache = cache_store
    return partition, False


def cluster_candidate_by_full_line_graph(candidate, partition):
    edge_tensor = _as_cpu_edge_tensor(candidate)
    labels = []
    missing = []
    for edge in edge_tensor.tolist():
        key = canonical_edge_key(edge)
        if key not in partition.edge_to_label:
            missing.append(key)
            continue
        labels.append(int(partition.edge_to_label[key]))

    if missing:
        return None, int(len(missing))
    return groups_from_labels(labels), 0


def cluster_candidate_with_coco_line_graph(candidate, graph, args, num_clusters, seed):
    edge_tensor = _as_cpu_edge_tensor(candidate)
    num_edges = int(edge_tensor.shape[0])
    effective_k = max(1, min(int(num_clusters), num_edges))
    if effective_k == 1:
        adjacency = build_line_graph_affinity_for_edges(edge_tensor)
        return [list(range(num_edges))], {
            "line_graph_num_nodes": num_edges,
            "line_graph_num_edges": int(np.count_nonzero(np.triu(adjacency > 0, k=1))),
            "line_graph_density": _affinity_density(
                num_edges,
                int(np.count_nonzero(np.triu(adjacency > 0, k=1))),
            ),
            "coco_cache_hit": None,
            "coco_train_runtime_sec": 0.0,
        }

    config = _coco_config_from_args(args, effective_k)
    config["seed"] = int(seed)
    features, adjacency, _ = build_line_graph_arrays(
        graph=graph,
        edge_tensor=edge_tensor,
        max_nodes=int(config["max_line_graph_nodes"]),
    )
    labels, train_metadata = run_coco_clustering(
        features=features,
        adjacency=adjacency,
        num_clusters=effective_k,
        config=config,
    )
    metadata = {
        **train_metadata,
        "line_graph_num_nodes": int(features.shape[0]),
        "line_graph_num_edges": int(np.count_nonzero(np.triu(adjacency > 0, k=1))),
        "line_graph_density": _affinity_density(
            int(features.shape[0]),
            int(np.count_nonzero(np.triu(adjacency > 0, k=1))),
        ),
        "coco_cache_hit": None,
    }
    return groups_from_labels(labels), metadata


def build_line_graph_arrays(graph, edge_tensor, max_nodes):
    edge_tensor = _as_cpu_edge_tensor(edge_tensor)
    if int(edge_tensor.shape[0]) > int(max_nodes):
        raise ValueError(
            f"CoCo line graph has {int(edge_tensor.shape[0])} nodes, exceeding "
            f"coco_max_line_graph_nodes={int(max_nodes)}. Increase the limit or use a smaller dataset."
        )

    x_cpu = graph.x.detach().cpu().float()
    src = edge_tensor[:, 0]
    dst = edge_tensor[:, 1]
    features = (0.5 * (x_cpu[src] + x_cpu[dst])).numpy().astype(np.float32, copy=False)
    adjacency = build_line_graph_affinity_for_edges(edge_tensor)
    edge_keys = [canonical_edge_key(edge) for edge in edge_tensor.tolist()]
    return features, adjacency, edge_keys


def build_line_graph_affinity_for_edges(edge_tensor):
    edge_tensor = _as_cpu_edge_tensor(edge_tensor)
    num_edges = int(edge_tensor.shape[0])
    adjacency = np.zeros((num_edges, num_edges), dtype=np.float32)
    incident = {}
    for edge_idx, (u, v) in enumerate(edge_tensor.tolist()):
        incident.setdefault(int(u), []).append(edge_idx)
        if int(v) != int(u):
            incident.setdefault(int(v), []).append(edge_idx)

    for edge_indices in incident.values():
        if len(edge_indices) <= 1:
            continue
        for pos, i in enumerate(edge_indices):
            for j in edge_indices[pos + 1 :]:
                adjacency[int(i), int(j)] = 1.0
                adjacency[int(j), int(i)] = 1.0
    return adjacency


def canonical_undirected_edges_from_graph(graph):
    edge_index = graph.edge_index.detach().cpu().to(torch.long)
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError("graph.edge_index must have shape [2, num_edges].")
    edges = edge_index.t()
    edges = torch.sort(edges, dim=1)[0]
    if edges.numel() == 0:
        raise ValueError("graph contains no edges.")
    edges = torch.unique(edges, dim=0)
    order = torch.argsort(edges[:, 0] * int(graph.num_nodes) + edges[:, 1])
    return edges[order].contiguous()


def canonical_edge_key(edge):
    u, v = [int(x) for x in edge]
    return (u, v) if u <= v else (v, u)


def groups_from_labels(labels):
    grouped = {}
    for edge_idx, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(int(edge_idx))
    return [grouped[label] for label in sorted(grouped.keys()) if len(grouped[label]) > 0]


def run_coco_clustering(features, adjacency, num_clusters, config):
    features = np.asarray(features, dtype=np.float32)
    adjacency = np.asarray(adjacency, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must be a 2D array.")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square 2D array.")
    if adjacency.shape[0] != features.shape[0]:
        raise ValueError("adjacency and features must have the same number of nodes.")

    num_nodes = int(features.shape[0])
    effective_k = max(1, min(int(num_clusters), num_nodes))
    if effective_k == 1:
        return np.zeros((num_nodes,), dtype=np.int64), {
            "coco_train_runtime_sec": 0.0,
            "coco_epochs": 0,
            "coco_effective_clusters": 1,
        }

    _set_seed(int(config["seed"]))
    device = _resolve_device(config.get("device", "auto"))
    train_start = time.perf_counter()

    features = _maybe_apply_pca(features, int(config["pca_dim"]), int(config["seed"]))
    x = torch.tensor(features, dtype=torch.float32)
    a = torch.tensor(adjacency, dtype=torch.float32)
    if bool(config["use_diffusion"]):
        ad = torch.tensor(_diffusion_adj(adjacency, alpha=float(config["alpha"])), dtype=torch.float32)
    else:
        ad = a.clone()

    x_l = _laplacian_filtering(a, x, int(config["filter_steps"]))
    x_g = _laplacian_filtering(ad, x, int(config["filter_steps"]))

    args = SimpleNamespace(
        dims=int(config["hidden_dim"]),
        consistency_memory_size=_resolve_memory_size(num_nodes, config),
        consistency_t=float(config["consistency_t"]),
        device=str(device),
    )
    consistency = Consist_net(args)
    model = compact_net(
        input_dim=int(x.shape[1]),
        hidden_dim=int(config["hidden_dim"]),
        act=str(config["activation"]),
        k=int(config["compact_k"]),
        stage_num=int(config["stage_num"]),
        beta=float(config["beta"]),
        cluster_num=effective_k,
    )
    model = model.to(device)
    consistency = consistency.to(device)
    x_l = x_l.to(device)
    x_g = x_g.to(device)
    a = a.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))

    best_z = None
    best_loss = None
    epochs = int(config["epochs"])
    for _ in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        z1, z2 = model(x_l, x_g, a)
        loss = 0.5 * (consistency(z1, z2) + consistency(z2, z1))
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        if math.isfinite(loss_value) and (best_loss is None or loss_value < best_loss):
            best_loss = loss_value
            with torch.no_grad():
                model.eval()
                z1_eval, z2_eval = model(x_l, x_g, a)
                best_z = ((z1_eval + z2_eval) / 2.0).detach().cpu()

    if best_z is None:
        with torch.no_grad():
            model.eval()
            z1_eval, z2_eval = model(x_l, x_g, a)
            best_z = ((z1_eval + z2_eval) / 2.0).detach().cpu()

    if not torch.isfinite(best_z).all():
        best_z = torch.where(torch.isfinite(best_z), best_z, torch.zeros_like(best_z))
        if torch.allclose(best_z, torch.zeros_like(best_z)):
            best_z = x_l.detach().cpu()

    labels, _ = kmeans(
        X=best_z,
        num_clusters=effective_k,
        distance=str(config["kmeans_distance"]),
        device=device,
    )
    labels = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    return labels, {
        "coco_train_runtime_sec": float(time.perf_counter() - train_start),
        "coco_epochs": epochs,
        "coco_effective_clusters": int(len(set(int(v) for v in labels.tolist()))),
        "coco_requested_clusters": effective_k,
        "coco_best_loss": best_loss,
        "coco_device": str(device),
    }


def _coco_config_from_args(args, num_clusters):
    return {
        "num_clusters": int(num_clusters),
        "seed": int(getattr(args, "seed", 0)),
        "epochs": int(getattr(args, "coco_epochs", 200)),
        "lr": float(getattr(args, "coco_lr", 1e-3)),
        "hidden_dim": int(getattr(args, "coco_hidden_dim", 256)),
        "activation": str(getattr(args, "coco_activation", "ident")),
        "compact_k": int(getattr(args, "coco_compact_k", 64)),
        "stage_num": int(getattr(args, "coco_stage_num", 10)),
        "beta": float(getattr(args, "coco_beta", 1.0)),
        "filter_steps": int(getattr(args, "coco_filter_steps", 2)),
        "alpha": float(getattr(args, "coco_alpha", 0.2)),
        "consistency_t": float(getattr(args, "coco_consistency_t", 0.02)),
        "memory_size": int(getattr(args, "coco_memory_size", 0)),
        "memory_multiplier": int(getattr(args, "coco_memory_multiplier", 10)),
        "pca_dim": int(getattr(args, "coco_pca_dim", -1)),
        "use_diffusion": int(getattr(args, "coco_use_diffusion", 1)),
        "kmeans_distance": str(getattr(args, "coco_kmeans_distance", "euclidean")),
        "device": str(getattr(args, "coco_device", "auto")),
        "cache_dir": str(getattr(args, "coco_cache_dir", "results/coco_line_graph_cache")),
        "force_retrain": int(getattr(args, "coco_force_retrain", 0)),
        "max_line_graph_nodes": int(getattr(args, "coco_max_line_graph_nodes", 6000)),
    }


def _resolve_memory_size(num_nodes, config):
    explicit_size = int(config.get("memory_size", 0))
    if explicit_size > 0:
        return explicit_size
    return max(1, int(num_nodes) * max(1, int(config.get("memory_multiplier", 10))))


def _resolve_device(device_name):
    device_name = str(device_name).strip().lower()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _maybe_apply_pca(features, pca_dim, seed):
    if int(pca_dim) <= 0:
        return features
    if features.shape[1] <= int(pca_dim) or features.shape[0] <= int(pca_dim):
        return features
    from sklearn.decomposition import PCA

    pca = PCA(n_components=int(pca_dim), random_state=int(seed))
    return pca.fit_transform(features).astype(np.float32, copy=False)


def _laplacian_filtering(adjacency, features, steps):
    adjacency = adjacency - torch.diag_embed(torch.diag(adjacency))
    normalized = _normalize_adj_torch(adjacency, self_loop=True, symmetry=True)
    out = features.float()
    for _ in range(max(0, int(steps))):
        out = normalized @ out
    return out.float()


def _normalize_adj_torch(adjacency, self_loop=True, symmetry=False):
    if self_loop:
        adjacency = adjacency + torch.eye(adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device)
    degree = adjacency.sum(dim=0)
    inv_degree = torch.zeros_like(degree)
    nonzero = degree > 0
    inv_degree[nonzero] = 1.0 / degree[nonzero]
    if symmetry:
        inv_sqrt = torch.sqrt(inv_degree)
        return inv_sqrt.view(-1, 1) * adjacency * inv_sqrt.view(1, -1)
    return inv_degree.view(1, -1) * adjacency


def _diffusion_adj(adjacency, alpha):
    adjacency = np.asarray(adjacency, dtype=np.float32)
    num_nodes = int(adjacency.shape[0])
    adj_tmp = adjacency + np.eye(num_nodes, dtype=np.float32)
    degree = adj_tmp.sum(axis=0)
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    norm_adj = inv_sqrt[:, None] * adj_tmp * inv_sqrt[None, :]
    eye = np.eye(num_nodes, dtype=np.float32)
    return (float(alpha) * np.linalg.inv(eye - (1.0 - float(alpha)) * norm_adj)).astype(np.float32, copy=False)


def _set_seed(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _build_partition(edge_keys, labels, metadata):
    labels = np.asarray(labels, dtype=np.int64)
    edge_to_label = {canonical_edge_key(edge): int(label) for edge, label in zip(edge_keys, labels.tolist())}
    return CocoLineGraphPartition(
        edge_keys=[canonical_edge_key(edge) for edge in edge_keys],
        labels=labels,
        edge_to_label=edge_to_label,
        metadata=dict(metadata),
    )


def _full_partition_cache_key(graph, edge_tensor, config):
    hasher = hashlib.sha1()
    hasher.update(np.asarray(edge_tensor.numpy(), dtype=np.int64).tobytes())
    x = graph.x.detach().cpu().float().contiguous().numpy()
    hasher.update(str(tuple(x.shape)).encode("utf-8"))
    hasher.update(x.tobytes())
    hasher.update(json.dumps(_cacheable_config(config), sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


def _cacheable_config(config):
    ignored = {"cache_dir", "force_retrain", "device"}
    return {key: value for key, value in config.items() if key not in ignored}


def _disk_cache_path(config, cache_key):
    cache_dir = str(config.get("cache_dir", "")).strip()
    if cache_dir == "" or cache_dir.lower() == "none":
        return None
    return Path(cache_dir) / f"{cache_key}.npz"


def _save_partition_cache(cache_path, partition):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(
            handle,
            edges=np.asarray(partition.edge_keys, dtype=np.int64),
            labels=np.asarray(partition.labels, dtype=np.int64),
            metadata=np.asarray(json.dumps(partition.metadata, sort_keys=True), dtype=object),
        )
    os.replace(tmp_path, cache_path)


def _load_partition_cache(cache_path):
    loaded = np.load(cache_path, allow_pickle=True)
    edge_keys = [canonical_edge_key(edge) for edge in loaded["edges"].tolist()]
    labels = loaded["labels"].astype(np.int64, copy=False)
    metadata_raw = loaded["metadata"].item()
    metadata = json.loads(str(metadata_raw))
    metadata["cache_path"] = str(cache_path)
    return _build_partition(edge_keys=edge_keys, labels=labels, metadata=metadata)


def _as_cpu_edge_tensor(edges):
    if torch.is_tensor(edges):
        edge_tensor = edges.detach().cpu().to(torch.long)
    else:
        edge_tensor = torch.tensor(edges, dtype=torch.long)
    if edge_tensor.dim() == 1 and edge_tensor.numel() == 2:
        edge_tensor = edge_tensor.view(1, 2)
    if edge_tensor.dim() != 2 or edge_tensor.shape[1] != 2:
        raise ValueError("edges must have shape [num_edges, 2].")
    sorted_edges = torch.sort(edge_tensor, dim=1)[0]
    return sorted_edges.contiguous()


def _affinity_density(num_nodes, num_edges):
    if int(num_nodes) <= 1:
        return 0.0
    return float((2.0 * float(num_edges)) / float(int(num_nodes) * (int(num_nodes) - 1)))
