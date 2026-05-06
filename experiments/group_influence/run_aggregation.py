#!/usr/bin/env python
import argparse
import json
import os
import sys

import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from main import _create_parser
from src.group_influence import (
    build_influence_module,
    build_state,
    compute_actual_pbrf,
    compute_heo_oneshot,
    compute_single_edge_sum,
)
from src.group_influence.aggregation import compute_independent_cluster_sum
from src.group_influence.cache import make_run_dir, save_csv, save_json, save_tensor


def main():
    cli = _parse_args()
    candidate_metadata = _load_metadata(cli.candidate_dir)
    clustering_metadata = _load_metadata(cli.clustering_dir)
    _fill_defaults_from_metadata(cli, candidate_metadata)

    candidate_edges_path = _candidate_edges_path(cli)
    cluster_labels_path = _cluster_labels_path(cli)
    loaded_candidate_edges = torch.load(candidate_edges_path, map_location="cpu", weights_only=True)
    cli.num_edges = int(_as_edge_tensor(loaded_candidate_edges).shape[0])

    experiment_args = _build_main_args(cli)
    config = _config_dict(
        cli=cli,
        experiment_args=experiment_args,
        candidate_metadata=candidate_metadata,
        clustering_metadata=clustering_metadata,
        candidate_edges_path=candidate_edges_path,
        cluster_labels_path=cluster_labels_path,
    )
    run_dir, config_hash = make_run_dir(cli.cache_root, "aggregation", config, run_id=cli.run_id)

    state = build_state(experiment_args)
    candidate_edges = torch.as_tensor(loaded_candidate_edges, device=state.data.edge_index.device, dtype=torch.long)
    cluster_labels = torch.load(cluster_labels_path, map_location=state.data.edge_index.device, weights_only=True)
    cluster_labels = torch.as_tensor(cluster_labels, device=state.data.edge_index.device, dtype=torch.long)

    influence_module = build_influence_module(state)
    heo = compute_heo_oneshot(
        candidate_edges,
        state,
        influence_type=cli.element_type,
        influence_module=influence_module,
    )
    single_sum = compute_single_edge_sum(
        candidate_edges,
        state,
        influence_type=cli.element_type,
        influence_module=influence_module,
    )
    independent = compute_independent_cluster_sum(
        candidate_edges=candidate_edges,
        cluster_labels=cluster_labels,
        state=state,
        influence_type=cli.element_type,
        influence_module=influence_module,
    )
    actual = None if cli.skip_pbrf else compute_actual_pbrf(candidate_edges, state, influence_type=cli.element_type)
    summary = _build_summary(actual=actual, heo=heo, single_sum=single_sum, independent=independent)

    metadata = {
        "config": config,
        "config_hash": config_hash,
        "run_dir": str(run_dir),
        "candidate_metadata": candidate_metadata,
        "clustering_metadata": clustering_metadata,
        "model_checkpoint_dir": state.dirs["vanilla"],
        "pbrf_checkpoint_dir": state.dirs["pbrf_model"],
        "legacy_result_dir": state.dirs["result"],
    }
    save_json(str(run_dir / "metadata.json"), metadata)
    save_json(
        str(run_dir / "aggregation_result.json"),
        {
            "summary": summary,
            "heo": heo,
            "single_edge_sum": single_sum,
            "independent_cluster_sum": independent,
            "actual_pbrf": actual,
        },
    )
    save_csv(str(run_dir / "aggregation_result.csv"), [summary], fieldnames=list(summary.keys()))
    save_csv(
        str(run_dir / "independent_cluster_rows.csv"),
        independent["cluster_rows"],
        fieldnames=[
            "cluster_id",
            "cluster_size",
            "cluster_prediction",
            "cluster_parameter_shift",
            "cluster_message_passing",
        ],
    )
    save_tensor(str(run_dir / "candidate_edges.pt"), candidate_edges)
    save_tensor(str(run_dir / "cluster_labels.pt"), cluster_labels)

    print(f"[group-influence-aggregation] run_dir={run_dir}")
    print(
        "A={actual} H={heo:.8g} E={single:.8g} C_ind={cluster:.8g} "
        "|A-C_ind|={err}".format(
            actual="skipped" if actual is None else f"{actual['total']:.8g}",
            heo=summary["H"],
            single=summary["E"],
            cluster=summary["C_ind"],
            err="n/a" if summary["abs_error_C_ind"] is None else f"{summary['abs_error_C_ind']:.8g}",
        )
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=str, default=None)
    parser.add_argument("--candidate-edges-path", type=str, default=None)
    parser.add_argument("--clustering-dir", type=str, default=None)
    parser.add_argument("--cluster-labels-path", type=str, default=None)
    parser.add_argument("--aggregation-method", type=str, default="independent_cluster_sum", choices=["independent_cluster_sum"])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--element-type", type=str, default=None, choices=["edge_removal", "edge_insertion"])
    parser.add_argument("--eval-metric", type=str, default=None)
    parser.add_argument("--hessian-type", type=str, default=None, choices=["hessian", "GNH"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--damp", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--lissa-iter", type=int, default=None)
    parser.add_argument("--pbrf-epochs", type=int, default=None)
    parser.add_argument("--pbrf-weight-decay", type=float, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--cache-root", type=str, default="results/group_influence")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--skip-pbrf", action="store_true")
    return parser.parse_args()


def _build_main_args(cli):
    args = _create_parser().parse_args([])
    args.dataset = cli.dataset
    args.model = cli.model
    args.num_layers = int(cli.num_layers)
    args.hidden_dim = int(cli.hidden_dim)
    args.seed = int(cli.seed)
    args.element_type = cli.element_type
    args.eval_metric = cli.eval_metric
    args.hessian_type = cli.hessian_type
    args.lr = float(cli.lr)
    args.epochs = int(cli.epochs)
    args.weight_decay = float(cli.weight_decay)
    args.damp = float(cli.damp)
    args.scale = float(cli.scale)
    args.lissa_iter = int(cli.lissa_iter)
    args.pbrf_epochs = int(cli.pbrf_epochs)
    args.pbrf_weight_decay = float(cli.pbrf_weight_decay)
    args.num_heads = int(cli.num_heads)
    args.num_group_elem = int(cli.num_edges)
    args.num_removal_candidates = 1
    args.num_insertion_candidates = 1
    args.metric_mode = "global"
    args.influence_calculation_mode = "calculate_influence"
    args.influence_mode = "calculate_influence"
    args.experiment_name = "none"
    args.fig_title = "group_influence_aggregation"
    return args


def _config_dict(
    cli,
    experiment_args,
    candidate_metadata,
    clustering_metadata,
    candidate_edges_path,
    cluster_labels_path,
):
    return {
        "phase": "aggregation",
        "dataset": cli.dataset,
        "model": cli.model,
        "num_layers": int(cli.num_layers),
        "hidden_dim": int(cli.hidden_dim),
        "seed": int(cli.seed),
        "num_edges": int(cli.num_edges),
        "element_type": cli.element_type,
        "aggregation_method": cli.aggregation_method,
        "candidate_dir": cli.candidate_dir,
        "candidate_edges_path": candidate_edges_path,
        "candidate_config_hash": candidate_metadata.get("config_hash") if candidate_metadata else None,
        "clustering_dir": cli.clustering_dir,
        "cluster_labels_path": cluster_labels_path,
        "clustering_config_hash": clustering_metadata.get("config_hash") if clustering_metadata else None,
        "eval_metric": cli.eval_metric,
        "hessian_type": cli.hessian_type,
        "lissa_iter": int(cli.lissa_iter),
        "scale": float(cli.scale),
        "pbrf_epochs": int(cli.pbrf_epochs),
        "skip_pbrf": bool(cli.skip_pbrf),
        "main_args": vars(experiment_args),
    }


def _load_metadata(path):
    if path is None:
        return {}
    metadata_path = os.path.join(path, "metadata.json")
    if not os.path.isfile(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fill_defaults_from_metadata(cli, metadata):
    config = metadata.get("config", {}) if metadata else {}
    defaults = {
        "dataset": "cora_public",
        "model": "GCN",
        "num_layers": 2,
        "hidden_dim": 16,
        "seed": 0,
        "element_type": "edge_removal",
        "eval_metric": "mean_validation_loss",
        "hessian_type": "GNH",
        "lr": 0.01,
        "epochs": 1000,
        "weight_decay": 0.001,
        "damp": 0.1,
        "scale": 32.0,
        "lissa_iter": 100,
        "pbrf_epochs": 5,
        "pbrf_weight_decay": 0.0,
        "num_heads": 8,
    }
    for name, fallback in defaults.items():
        if getattr(cli, name) is None:
            setattr(cli, name, config.get(name, fallback))


def _candidate_edges_path(cli):
    if cli.candidate_edges_path is not None:
        return cli.candidate_edges_path
    if cli.candidate_dir is None:
        raise ValueError("--candidate-dir or --candidate-edges-path is required.")
    return os.path.join(cli.candidate_dir, "candidate_edges.pt")


def _cluster_labels_path(cli):
    if cli.cluster_labels_path is not None:
        return cli.cluster_labels_path
    if cli.clustering_dir is None:
        raise ValueError("--clustering-dir or --cluster-labels-path is required.")
    return os.path.join(cli.clustering_dir, "cluster_labels.pt")


def _build_summary(actual, heo, single_sum, independent):
    actual_total = None if actual is None else actual["total"]
    summary = {
        "A": actual_total,
        "H": heo["total"],
        "E": single_sum["total"],
        "C_ind": independent["total"],
        "H_parameter_shift": heo["parameter_shift"],
        "H_message_passing": heo["message_passing"],
        "E_parameter_shift": single_sum["parameter_shift"],
        "E_message_passing": single_sum["message_passing"],
        "abs_error_H": None,
        "abs_error_E": None,
        "abs_error_C_ind": None,
        "relative_error_H": None,
        "relative_error_E": None,
        "relative_error_C_ind": None,
        "sign_match_H": None,
        "sign_match_E": None,
        "sign_match_C_ind": None,
    }
    if actual_total is not None:
        denom = max(abs(actual_total), 1e-12)
        for key, pred in [("H", heo["total"]), ("E", single_sum["total"]), ("C_ind", independent["total"])]:
            abs_error = abs(actual_total - pred)
            summary[f"abs_error_{key}"] = abs_error
            summary[f"relative_error_{key}"] = abs_error / denom
            summary[f"sign_match_{key}"] = _sign(actual_total) == _sign(pred)
    return summary


def _sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _as_edge_tensor(edge_set):
    edge_tensor = torch.as_tensor(edge_set, dtype=torch.long)
    if edge_tensor.dim() == 1:
        edge_tensor = edge_tensor.view(1, 2)
    if edge_tensor.dim() == 3 and edge_tensor.shape[0] == 1:
        edge_tensor = edge_tensor.squeeze(0)
    if edge_tensor.dim() != 2 or edge_tensor.shape[1] != 2:
        raise ValueError("edge_set must have shape [num_edges, 2] or [1, num_edges, 2].")
    return edge_tensor


if __name__ == "__main__":
    main()
