#!/usr/bin/env python
import argparse
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
    build_random_edge_set,
    build_state,
    compute_actual_pbrf,
    compute_heo_oneshot,
    compute_single_edge_sum,
)
from src.group_influence.cache import make_run_dir, save_csv, save_json, save_tensor


def main():
    cli = _parse_args()
    experiment_args = _build_main_args(cli)
    config = _config_dict(cli, experiment_args)
    run_dir, config_hash = make_run_dir(cli.cache_root, "baselines", config, run_id=cli.run_id)

    state = build_state(experiment_args)
    if cli.candidate_edges_path is None:
        edge_set = build_random_edge_set(state, num_edges=cli.num_edges, influence_type=cli.element_type)
    else:
        edge_set = torch.load(cli.candidate_edges_path, map_location=state.data.edge_index.device, weights_only=True)
        edge_set = torch.as_tensor(edge_set, device=state.data.edge_index.device, dtype=torch.long)
    influence_module = build_influence_module(state)

    heo = compute_heo_oneshot(edge_set, state, influence_type=cli.element_type, influence_module=influence_module)
    single_sum = compute_single_edge_sum(edge_set, state, influence_type=cli.element_type, influence_module=influence_module)
    actual = None if cli.skip_pbrf else compute_actual_pbrf(edge_set, state, influence_type=cli.element_type)

    summary = _build_summary(actual=actual, heo=heo, single_sum=single_sum)
    metadata = {
        "config": config,
        "config_hash": config_hash,
        "run_dir": str(run_dir),
        "model_checkpoint_dir": state.dirs["vanilla"],
        "pbrf_checkpoint_dir": state.dirs["pbrf_model"],
        "legacy_result_dir": state.dirs["result"],
    }

    save_tensor(str(run_dir / "candidate_edges.pt"), edge_set)
    save_json(str(run_dir / "metadata.json"), metadata)
    save_json(str(run_dir / "baseline_result.json"), {"summary": summary, "heo": heo, "single_edge_sum": single_sum, "actual_pbrf": actual})
    save_csv(str(run_dir / "baseline_result.csv"), [summary], fieldnames=list(summary.keys()))
    save_csv(
        str(run_dir / "single_edge_influences.csv"),
        single_sum["edge_rows"],
        fieldnames=[
            "edge_id",
            "u",
            "v",
            "single_edge_influence",
            "single_edge_parameter_shift",
            "single_edge_message_passing",
        ],
    )

    print(f"[group-influence-baseline] run_dir={run_dir}")
    print(
        "A(S)={actual} H(S)={heo_total:.8g} E(S)={single_total:.8g} "
        "|A-H|={err_h} |A-E|={err_e}".format(
            actual="skipped" if actual is None else f"{actual['total']:.8g}",
            heo_total=heo["total"],
            single_total=single_sum["total"],
            err_h="n/a" if summary["abs_error_H"] is None else f"{summary['abs_error_H']:.8g}",
            err_e="n/a" if summary["abs_error_E"] is None else f"{summary['abs_error_E']:.8g}",
        )
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora_public")
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--element-type", type=str, default="edge_removal", choices=["edge_removal", "edge_insertion"])
    parser.add_argument(
        "--candidate-type",
        type=str,
        default="random",
        choices=["random", "top_abs", "mixed", "top_positive_negative"],
    )
    parser.add_argument("--candidate-edges-path", type=str, default=None)
    parser.add_argument("--num-edges", type=int, default=2)
    parser.add_argument("--eval-metric", type=str, default="mean_validation_loss")
    parser.add_argument("--hessian-type", type=str, default="GNH", choices=["hessian", "GNH"])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--damp", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=32.0)
    parser.add_argument("--lissa-iter", type=int, default=100)
    parser.add_argument("--pbrf-epochs", type=int, default=5)
    parser.add_argument("--pbrf-weight-decay", type=float, default=0.0)
    parser.add_argument("--num-heads", type=int, default=8)
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
    args.fig_title = "group_influence_baseline"
    return args


def _config_dict(cli, experiment_args):
    return {
        "phase": "baseline",
        "dataset": cli.dataset,
        "model": cli.model,
        "num_layers": int(cli.num_layers),
        "hidden_dim": int(cli.hidden_dim),
        "seed": int(cli.seed),
        "element_type": cli.element_type,
        "candidate_type": cli.candidate_type,
        "candidate_edges_path": cli.candidate_edges_path,
        "num_edges": int(cli.num_edges),
        "eval_metric": cli.eval_metric,
        "hessian_type": cli.hessian_type,
        "lissa_iter": int(cli.lissa_iter),
        "scale": float(cli.scale),
        "pbrf_epochs": int(cli.pbrf_epochs),
        "skip_pbrf": bool(cli.skip_pbrf),
        "main_args": vars(experiment_args),
    }


def _build_summary(actual, heo, single_sum):
    actual_total = None if actual is None else actual["total"]
    summary = {
        "A": actual_total,
        "H": heo["total"],
        "E": single_sum["total"],
        "H_parameter_shift": heo["parameter_shift"],
        "H_message_passing": heo["message_passing"],
        "E_parameter_shift": single_sum["parameter_shift"],
        "E_message_passing": single_sum["message_passing"],
        "abs_error_H": None,
        "abs_error_E": None,
        "relative_error_H": None,
        "relative_error_E": None,
        "sign_match_H": None,
        "sign_match_E": None,
    }
    if actual_total is not None:
        summary["abs_error_H"] = abs(actual_total - heo["total"])
        summary["abs_error_E"] = abs(actual_total - single_sum["total"])
        denom = max(abs(actual_total), 1e-12)
        summary["relative_error_H"] = summary["abs_error_H"] / denom
        summary["relative_error_E"] = summary["abs_error_E"] / denom
        summary["sign_match_H"] = _sign(actual_total) == _sign(heo["total"])
        summary["sign_match_E"] = _sign(actual_total) == _sign(single_sum["total"])
    return summary


def _sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


if __name__ == "__main__":
    main()
