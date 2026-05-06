#!/usr/bin/env python
import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from main import _create_parser
from src.group_influence import build_candidate_set, build_state
from src.group_influence.cache import make_run_dir, save_csv, save_json, save_tensor


def main():
    cli = _parse_args()
    experiment_args = _build_main_args(cli)
    config = _config_dict(cli, experiment_args)
    run_dir, config_hash = make_run_dir(cli.cache_root, "candidate_edges", config, run_id=cli.run_id)

    state = build_state(experiment_args)
    candidate_result = build_candidate_set(
        state=state,
        candidate_type=cli.candidate_type,
        num_candidates=cli.num_candidates,
        influence_type=cli.element_type,
        pool_size=cli.pool_size,
        mixed_positive_count=cli.mixed_positive_count,
        mixed_negative_count=cli.mixed_negative_count,
    )

    metadata = {
        "config": config,
        "config_hash": config_hash,
        "run_dir": str(run_dir),
        "candidate_set_name": candidate_result["candidate_set_name"],
        "num_pool_edges": int(candidate_result["edge_pool"].shape[0]),
        "num_candidate_edges": int(candidate_result["candidate_edges"].shape[0]),
        "model_checkpoint_dir": state.dirs["vanilla"],
        "legacy_result_dir": state.dirs["result"],
    }
    save_json(str(run_dir / "metadata.json"), metadata)
    save_tensor(str(run_dir / "edge_pool.pt"), candidate_result["edge_pool"])
    save_tensor(str(run_dir / "candidate_edges.pt"), candidate_result["candidate_edges"])
    save_csv(str(run_dir / "candidate_edge_scores.csv"), candidate_result["candidate_edge_scores"], fieldnames=_score_fields())
    if len(candidate_result["pool_edge_scores"]) > 0:
        save_csv(str(run_dir / "pool_edge_scores.csv"), candidate_result["pool_edge_scores"], fieldnames=_score_fields())

    print(f"[group-influence-candidates] run_dir={run_dir}")
    print(
        "candidate_set={name} candidate_edges={num_candidates} pool_edges={num_pool}".format(
            name=candidate_result["candidate_set_name"],
            num_candidates=int(candidate_result["candidate_edges"].shape[0]),
            num_pool=int(candidate_result["edge_pool"].shape[0]),
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
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--mixed-positive-count", type=int, default=None)
    parser.add_argument("--mixed-negative-count", type=int, default=None)
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
    args.num_group_elem = 1
    args.num_removal_candidates = int(cli.num_candidates)
    args.num_insertion_candidates = int(cli.num_candidates)
    args.metric_mode = "global"
    args.influence_calculation_mode = "calculate_influence"
    args.influence_mode = "calculate_influence"
    args.experiment_name = "none"
    args.fig_title = "group_influence_candidates"
    return args


def _config_dict(cli, experiment_args):
    return {
        "phase": "candidate_edges",
        "dataset": cli.dataset,
        "model": cli.model,
        "num_layers": int(cli.num_layers),
        "hidden_dim": int(cli.hidden_dim),
        "seed": int(cli.seed),
        "element_type": cli.element_type,
        "candidate_type": cli.candidate_type,
        "num_candidates": int(cli.num_candidates),
        "pool_size": cli.pool_size,
        "mixed_positive_count": cli.mixed_positive_count,
        "mixed_negative_count": cli.mixed_negative_count,
        "eval_metric": cli.eval_metric,
        "hessian_type": cli.hessian_type,
        "lissa_iter": int(cli.lissa_iter),
        "scale": float(cli.scale),
        "main_args": vars(experiment_args),
    }


def _score_fields():
    return [
        "candidate_edge_id",
        "pool_index",
        "edge_id",
        "u",
        "v",
        "selected",
        "single_edge_influence",
        "single_edge_parameter_shift",
        "single_edge_message_passing",
    ]


if __name__ == "__main__":
    main()
