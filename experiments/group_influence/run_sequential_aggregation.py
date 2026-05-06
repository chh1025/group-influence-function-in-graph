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
    compute_independent_cluster_sum,
    compute_single_edge_sum,
)
from src.group_influence.aggregation import compute_cluster_sequential_graph_only
from src.group_influence.cache import make_run_dir, save_csv, save_json, save_tensor


def main():
    cli = _parse_args()
    candidate_metadata = _load_metadata(cli.candidate_dir)
    clustering_metadata = _load_metadata(cli.clustering_dir)
    baseline_result = _load_baseline_result(cli.baseline_aggregation_dir)
    _fill_defaults_from_metadata(cli, candidate_metadata)

    candidate_edges_path = _candidate_edges_path(cli)
    cluster_labels_path = _cluster_labels_path(cli)
    loaded_candidate_edges = torch.load(candidate_edges_path, map_location="cpu", weights_only=True)
    cli.num_edges = int(_as_edge_tensor(loaded_candidate_edges).shape[0])

    experiment_args = _build_main_args(cli)
    order_policies = _split_csv(cli.order_policies)
    config = _config_dict(
        cli=cli,
        experiment_args=experiment_args,
        candidate_metadata=candidate_metadata,
        clustering_metadata=clustering_metadata,
        candidate_edges_path=candidate_edges_path,
        cluster_labels_path=cluster_labels_path,
        order_policies=order_policies,
    )
    run_dir, config_hash = make_run_dir(cli.cache_root, "sequential_aggregation", config, run_id=cli.run_id)

    state = build_state(experiment_args)
    candidate_edges = torch.as_tensor(loaded_candidate_edges, device=state.data.edge_index.device, dtype=torch.long)
    cluster_labels = torch.load(cluster_labels_path, map_location=state.data.edge_index.device, weights_only=True)
    cluster_labels = torch.as_tensor(cluster_labels, device=state.data.edge_index.device, dtype=torch.long)

    baseline = _baseline_or_compute(
        baseline_result=baseline_result,
        candidate_edges=candidate_edges,
        cluster_labels=cluster_labels,
        state=state,
        influence_type=cli.element_type,
        skip_pbrf=cli.skip_pbrf,
    )

    sequential_results = {}
    for order_policy in order_policies:
        sequential = compute_cluster_sequential_graph_only(
            candidate_edges=candidate_edges,
            cluster_labels=cluster_labels,
            state=state,
            influence_type=cli.element_type,
            order_policy=order_policy,
            seed=cli.seed,
            independent_cluster_rows=baseline["independent_cluster_sum"]["cluster_rows"],
        )
        sequential_results[order_policy] = sequential
        save_csv(
            str(run_dir / f"sequential_steps_{order_policy}.csv"),
            sequential["step_rows"],
            fieldnames=[
                "step_idx",
                "cluster_id",
                "cluster_size",
                "step_prediction",
                "step_parameter_shift",
                "step_message_passing",
                "running_total",
            ],
        )

    summary = _build_summary(
        baseline_summary=baseline["summary"],
        sequential_results=sequential_results,
    )
    metadata = {
        "config": config,
        "config_hash": config_hash,
        "run_dir": str(run_dir),
        "candidate_metadata": candidate_metadata,
        "clustering_metadata": clustering_metadata,
        "baseline_aggregation_dir": cli.baseline_aggregation_dir,
        "model_checkpoint_dir": state.dirs["vanilla"],
        "legacy_result_dir": state.dirs["result"],
    }

    save_json(str(run_dir / "metadata.json"), metadata)
    save_json(
        str(run_dir / "sequential_result.json"),
        {
            "summary": summary,
            "baseline": baseline,
            "sequential": sequential_results,
        },
    )
    save_csv(str(run_dir / "sequential_result.csv"), [summary], fieldnames=list(summary.keys()))
    save_tensor(str(run_dir / "candidate_edges.pt"), candidate_edges)
    save_tensor(str(run_dir / "cluster_labels.pt"), cluster_labels)

    best_seq_key = min(order_policies, key=lambda policy: summary.get(f"abs_error_C_seq_{policy}", float("inf")))
    print(f"[group-influence-sequential-aggregation] run_dir={run_dir}")
    print(
        "A={actual} H={h:.8g} E={e:.8g} C_ind={cind:.8g} best_seq={policy}:{seq:.8g} "
        "|A-C_seq|={err}".format(
            actual="skipped" if summary["A"] is None else f"{summary['A']:.8g}",
            h=summary["H"],
            e=summary["E"],
            cind=summary["C_ind"],
            policy=best_seq_key,
            seq=summary[f"C_seq_{best_seq_key}"],
            err="n/a"
            if summary[f"abs_error_C_seq_{best_seq_key}"] is None
            else f"{summary[f'abs_error_C_seq_{best_seq_key}']:.8g}",
        )
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=str, required=True)
    parser.add_argument("--clustering-dir", type=str, required=True)
    parser.add_argument("--baseline-aggregation-dir", type=str, default=None)
    parser.add_argument("--candidate-edges-path", type=str, default=None)
    parser.add_argument("--cluster-labels-path", type=str, default=None)
    parser.add_argument(
        "--order-policies",
        type=str,
        default="cluster_id",
        help="Comma-separated policies: cluster_id,random,cluster_size_asc,small_abs_cluster_influence_first,large_abs_cluster_influence_first",
    )
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
    args.fig_title = "group_influence_sequential_aggregation"
    return args


def _config_dict(
    cli,
    experiment_args,
    candidate_metadata,
    clustering_metadata,
    candidate_edges_path,
    cluster_labels_path,
    order_policies,
):
    return {
        "phase": "sequential_aggregation",
        "dataset": cli.dataset,
        "model": cli.model,
        "num_layers": int(cli.num_layers),
        "hidden_dim": int(cli.hidden_dim),
        "seed": int(cli.seed),
        "num_edges": int(cli.num_edges),
        "element_type": cli.element_type,
        "order_policies": order_policies,
        "candidate_dir": cli.candidate_dir,
        "candidate_edges_path": candidate_edges_path,
        "candidate_config_hash": candidate_metadata.get("config_hash") if candidate_metadata else None,
        "clustering_dir": cli.clustering_dir,
        "cluster_labels_path": cluster_labels_path,
        "clustering_config_hash": clustering_metadata.get("config_hash") if clustering_metadata else None,
        "baseline_aggregation_dir": cli.baseline_aggregation_dir,
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


def _load_baseline_result(path):
    if path is None:
        return None
    result_path = os.path.join(path, "aggregation_result.json")
    if not os.path.isfile(result_path):
        return None
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _baseline_or_compute(baseline_result, candidate_edges, cluster_labels, state, influence_type, skip_pbrf):
    if baseline_result is not None:
        return {
            "summary": baseline_result["summary"],
            "heo": baseline_result["heo"],
            "single_edge_sum": baseline_result["single_edge_sum"],
            "independent_cluster_sum": baseline_result["independent_cluster_sum"],
            "actual_pbrf": baseline_result["actual_pbrf"],
        }

    influence_module = build_influence_module(state)
    heo = compute_heo_oneshot(candidate_edges, state, influence_type=influence_type, influence_module=influence_module)
    single_sum = compute_single_edge_sum(candidate_edges, state, influence_type=influence_type, influence_module=influence_module)
    independent = compute_independent_cluster_sum(
        candidate_edges=candidate_edges,
        cluster_labels=cluster_labels,
        state=state,
        influence_type=influence_type,
        influence_module=influence_module,
    )
    actual = None if skip_pbrf else compute_actual_pbrf(candidate_edges, state, influence_type=influence_type)
    return {
        "summary": _baseline_summary(actual=actual, heo=heo, single_sum=single_sum, independent=independent),
        "heo": heo,
        "single_edge_sum": single_sum,
        "independent_cluster_sum": independent,
        "actual_pbrf": actual,
    }


def _baseline_summary(actual, heo, single_sum, independent):
    actual_total = None if actual is None else actual["total"]
    summary = {
        "A": actual_total,
        "H": heo["total"],
        "E": single_sum["total"],
        "C_ind": independent["total"],
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


def _build_summary(baseline_summary, sequential_results):
    summary = dict(baseline_summary)
    actual_total = summary.get("A")
    for order_policy, result in sequential_results.items():
        key = f"C_seq_{order_policy}"
        summary[key] = result["total"]
        summary[f"abs_error_{key}"] = None
        summary[f"relative_error_{key}"] = None
        summary[f"sign_match_{key}"] = None
        if actual_total is not None:
            abs_error = abs(float(actual_total) - float(result["total"]))
            summary[f"abs_error_{key}"] = abs_error
            summary[f"relative_error_{key}"] = abs_error / max(abs(float(actual_total)), 1e-12)
            summary[f"sign_match_{key}"] = _sign(float(actual_total)) == _sign(float(result["total"]))
    return summary


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
    return os.path.join(cli.candidate_dir, "candidate_edges.pt")


def _cluster_labels_path(cli):
    if cli.cluster_labels_path is not None:
        return cli.cluster_labels_path
    return os.path.join(cli.clustering_dir, "cluster_labels.pt")


def _split_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


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
