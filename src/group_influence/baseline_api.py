from dataclasses import dataclass
import copy
import os.path as osp

import torch
import torch.optim as optim

from calculate_influence import GraphInfluenceModule, calculate_pbrf
from main import _apply_experiment_runtime_overrides, _ensure_optional_defaults
from src import DataLoader, GNN, make_metric_fns, train
from src.graph_utils import find_k_hop_neighborhoods
from src.utils import (
    get_edge_insertion_candidates,
    get_edge_removal_candidates,
    get_eval_node_idxs,
    make_dirs,
    random_planetoid_splits,
    set_seed,
)


@dataclass
class BaselineState:
    args: object
    dataset: object
    data: object
    model: object
    eval_node_idxs: list
    metric_fn: object
    dirs: dict
    device: str
    seed: int


def build_state(args):
    args = copy.copy(args)
    args = _ensure_optional_defaults(args)
    args.linear = bool(args.linear)
    args.bias = bool(args.bias)

    if args.hessian_type == "hessian":
        args.damp = args.weight_decay

    dataset = DataLoader(args.dataset, root="datasets")
    args.num_classes = dataset.num_classes
    data = dataset[0]
    data.edge_weight = torch.ones((data.edge_index.shape[1],))

    seed = int(args.seed)
    eval_node_idxs = get_eval_node_idxs(data, args.eval_metric, seed)

    if "public" not in str(args.dataset).lower():
        percls_trn = int(round(0.6 * len(data.y) / dataset.num_classes))
        val_lb = int(round(0.2 * len(data.y)))
        data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb, seed)

    args = _apply_experiment_runtime_overrides(args, data, torch)
    dirs = make_dirs(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.edge_weight = data.edge_weight.to(device)
    data.y = data.y.to(device)

    model = _load_or_train_model(args=args, dataset=dataset, data=data, dirs=dirs, device=device, seed=seed)

    exact_k_hop_neighbors = find_k_hop_neighborhoods(data, args.num_layers) if args.eval_metric == "feature_ablation" else None
    metric_fns = make_metric_fns(eval_node_idxs, exact_k_hop_neighbors, data.edge_index)
    metric_fn = metric_fns[args.eval_metric]

    return BaselineState(
        args=args,
        dataset=dataset,
        data=data,
        model=model,
        eval_node_idxs=eval_node_idxs,
        metric_fn=metric_fn,
        dirs=dirs,
        device=device,
        seed=seed,
    )


def _load_or_train_model(args, dataset, data, dirs, device, seed):
    vanilla_path = osp.join(dirs["vanilla"], f"{seed}.pth")
    set_seed(seed)
    model = GNN(
        name=args.model,
        in_dim=dataset.num_node_features,
        hidden_dim=args.hidden_dim,
        num_classes=dataset.num_classes,
        num_layers=args.num_layers,
        linear=args.linear,
        bias=args.bias,
        num_heads=args.num_heads,
    )
    model = model.to(device)

    if osp.isfile(vanilla_path):
        state_dict = torch.load(vanilla_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(1, int(args.epochs) + 1):
        train(data, model, optimizer, device)
    torch.save({k: v.clone().detach() for k, v in model.state_dict().items()}, vanilla_path)
    model.eval()
    return model


def build_influence_module(state):
    return GraphInfluenceModule(
        state.model,
        state.data,
        state.args,
        state.args.eval_metric,
        1,
        state.eval_node_idxs,
        state.metric_fn,
    )


def build_random_edge_set(state, num_edges, influence_type="edge_removal"):
    set_seed(state.seed)
    num_edges = int(num_edges)
    if influence_type == "edge_removal":
        edges = get_edge_removal_candidates(state.data, num_edges)
    elif influence_type == "edge_insertion":
        edges = get_edge_insertion_candidates(state.data, max(num_edges * 2, num_edges))
    else:
        raise ValueError(f"Unsupported influence_type for random edge set: {influence_type}")

    if int(edges.shape[0]) < num_edges:
        raise ValueError(f"Only {int(edges.shape[0])} candidate edges are available; requested {num_edges}.")
    return edges[:num_edges].detach().clone().to(device=state.data.edge_index.device, dtype=torch.long)


def compute_heo_oneshot(edge_set, state, influence_type="edge_removal", influence_module=None):
    module = influence_module or build_influence_module(state)
    candidates = _as_candidate_batch(edge_set, state.data.edge_index.device)
    total_inf, parameter_shift_inf, message_passing_inf, module_scale, inv_hvp_norm, avg_num_influenced = module.calculate_influence(
        candidates,
        influence_type,
    )
    return {
        "total": _scalar(total_inf[0]),
        "parameter_shift": _scalar(parameter_shift_inf[0]),
        "message_passing": _scalar(message_passing_inf[0]),
        "module_scale": _scalar(module_scale),
        "inv_hvp_norm": _scalar(inv_hvp_norm),
        "avg_num_influenced_nodes": _scalar(avg_num_influenced),
    }


def compute_single_edge_sum(edge_set, state, influence_type="edge_removal", influence_module=None):
    module = influence_module or build_influence_module(state)
    edge_tensor = _as_edge_tensor(edge_set, state.data.edge_index.device)
    single_candidates = edge_tensor.view(edge_tensor.shape[0], 1, 2)
    total_inf, parameter_shift_inf, message_passing_inf, module_scale, inv_hvp_norm, avg_num_influenced = module.calculate_influence(
        single_candidates,
        influence_type,
    )

    edge_rows = []
    for idx, edge in enumerate(edge_tensor.detach().cpu().tolist()):
        edge_rows.append(
            {
                "edge_id": idx,
                "u": int(edge[0]),
                "v": int(edge[1]),
                "single_edge_influence": _scalar(total_inf[idx]),
                "single_edge_parameter_shift": _scalar(parameter_shift_inf[idx]),
                "single_edge_message_passing": _scalar(message_passing_inf[idx]),
            }
        )

    return {
        "total": float(sum(row["single_edge_influence"] for row in edge_rows)),
        "parameter_shift": float(sum(row["single_edge_parameter_shift"] for row in edge_rows)),
        "message_passing": float(sum(row["single_edge_message_passing"] for row in edge_rows)),
        "module_scale": _scalar(module_scale),
        "inv_hvp_norm": _scalar(inv_hvp_norm),
        "avg_num_influenced_nodes": _scalar(avg_num_influenced),
        "edge_rows": edge_rows,
    }


def compute_actual_pbrf(edge_set, state, influence_type="edge_removal"):
    candidates = _as_candidate_batch(edge_set, state.data.edge_index.device)
    total_pbrf, parameter_shift_pbrf, message_passing_pbrf = calculate_pbrf(
        state.model,
        state.data,
        candidates,
        state.args,
        state.seed,
        state.dirs["pbrf_model"],
        state.metric_fn,
        influence_type,
    )
    return {
        "total": float(total_pbrf[0]),
        "parameter_shift": float(parameter_shift_pbrf[0]),
        "message_passing": float(message_passing_pbrf[0]),
    }


def _as_edge_tensor(edge_set, device):
    edge_tensor = torch.as_tensor(edge_set, device=device, dtype=torch.long)
    if edge_tensor.dim() == 1:
        edge_tensor = edge_tensor.view(1, 2)
    if edge_tensor.dim() == 3 and edge_tensor.shape[0] == 1:
        edge_tensor = edge_tensor.squeeze(0)
    if edge_tensor.dim() != 2 or edge_tensor.shape[1] != 2:
        raise ValueError("edge_set must have shape [num_edges, 2] or [1, num_edges, 2].")
    return edge_tensor


def _as_candidate_batch(edge_set, device):
    edge_tensor = _as_edge_tensor(edge_set, device)
    return edge_tensor.view(1, edge_tensor.shape[0], 2)


def _scalar(value):
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0.0
        return float(torch.as_tensor(value, dtype=torch.float32).detach().cpu().reshape(-1).mean().item())
    return float(torch.as_tensor(value).detach().cpu().reshape(-1).mean().item())
