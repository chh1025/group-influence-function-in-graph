#!/usr/bin/env python
import argparse
import csv
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
from src.group_influence import build_state
from src.group_influence.cache import make_run_dir, save_csv, save_json, save_tensor
from src.group_influence.clustering import cluster_features, summarize_clusters
from src.group_influence.features import build_cheap_edge_features, standardize_features


def main():
    cli = _parse_args()
    candidate_metadata = _load_candidate_metadata(cli.candidate_dir)
    _fill_defaults_from_candidate_metadata(cli, candidate_metadata)
    candidate_edges_path = _candidate_edges_path(cli)
    candidate_scores_path = _candidate_scores_path(cli)

    experiment_args = _build_main_args(cli)
    config = _config_dict(cli, experiment_args, candidate_metadata, candidate_edges_path, candidate_scores_path)
    run_dir, config_hash = make_run_dir(cli.cache_root, "feature_clustering", config, run_id=cli.run_id)

    state = build_state(experiment_args)
    candidate_edges = torch.load(candidate_edges_path, map_location="cpu", weights_only=True)
    score_rows = _read_csv(candidate_scores_path)

    feature_result = build_cheap_edge_features(
        state=state,
        candidate_edges=candidate_edges,
        score_rows=score_rows,
        influence_type=cli.element_type,
    )
    normalized = standardize_features(feature_result["features"])
    cluster_result = cluster_features(
        features=normalized["features"],
        method=cli.clustering_method,
        num_clusters=cli.num_clusters,
        seed=cli.seed,
        max_iter=cli.max_iter,
    )
    labels = cluster_result["labels"]
    influence_idx = feature_result["feature_names"].index("single_edge_influence")
    cluster_summary = summarize_clusters(
        labels=labels,
        features=normalized["features"],
        influences=feature_result["features"][:, influence_idx],
    )
    assignment_rows = _assignment_rows(feature_result["feature_rows"], labels)

    metadata = {
        "config": config,
        "config_hash": config_hash,
        "run_dir": str(run_dir),
        "candidate_edges_path": candidate_edges_path,
        "candidate_scores_path": candidate_scores_path,
        "candidate_metadata": candidate_metadata,
        "feature_names": feature_result["feature_names"],
        "clustering_method": cluster_result["method"],
        "num_iter": cluster_result["num_iter"],
    }
    save_json(str(run_dir / "metadata.json"), metadata)
    save_json(
        str(run_dir / "clustering_result.json"),
        {
            "summary": cluster_summary["summary"],
            "cluster_rows": cluster_summary["cluster_rows"],
        },
    )
    save_tensor(str(run_dir / "candidate_edges.pt"), torch.as_tensor(candidate_edges, dtype=torch.long))
    save_tensor(str(run_dir / "cheap_features.pt"), feature_result["features"])
    save_tensor(str(run_dir / "normalized_features.pt"), normalized["features"])
    save_tensor(str(run_dir / "cluster_labels.pt"), labels)
    save_tensor(str(run_dir / "cluster_centroids.pt"), cluster_result["centroids"])
    save_csv(
        str(run_dir / "cluster_assignments.csv"),
        assignment_rows,
        fieldnames=["candidate_edge_id", "u", "v", "cluster_id"] + feature_result["feature_names"],
    )
    save_csv(
        str(run_dir / "cluster_summary.csv"),
        cluster_summary["cluster_rows"],
        fieldnames=[
            "cluster_id",
            "cluster_size",
            "cluster_influence_sum",
            "cluster_abs_edge_influence_sum",
            "positive_count",
            "negative_count",
            "zero_count",
            "sign_purity",
            "within_cluster_feature_variance",
        ],
    )

    summary = cluster_summary["summary"]
    print(f"[group-influence-feature-clustering] run_dir={run_dir}")
    print(
        "method={method} K={k} total={total:.8g} cancel={cancel:.8g} "
        "sign_purity={purity:.4f} within_var={var:.8g}".format(
            method=cluster_result["method"],
            k=summary["num_clusters"],
            total=summary["total_influence_sum"],
            cancel=summary["cancellation_score"],
            purity=summary["weighted_sign_purity"],
            var=summary["mean_within_cluster_feature_variance"],
        )
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=str, default=None)
    parser.add_argument("--candidate-edges-path", type=str, default=None)
    parser.add_argument("--candidate-edge-scores-path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--element-type", type=str, default=None, choices=["edge_removal", "edge_insertion"])
    parser.add_argument("--feature-type", type=str, default="cheap", choices=["cheap"])
    parser.add_argument("--clustering-method", type=str, default="cheap_kmeans", choices=["cheap_kmeans", "kmeans", "random"])
    parser.add_argument("--num-clusters", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=100)
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
    args.num_removal_candidates = 1
    args.num_insertion_candidates = 1
    args.metric_mode = "global"
    args.influence_calculation_mode = "calculate_influence"
    args.influence_mode = "calculate_influence"
    args.experiment_name = "none"
    args.fig_title = "group_influence_feature_clustering"
    return args


def _config_dict(cli, experiment_args, candidate_metadata, candidate_edges_path, candidate_scores_path):
    return {
        "phase": "feature_clustering",
        "dataset": cli.dataset,
        "model": cli.model,
        "num_layers": int(cli.num_layers),
        "hidden_dim": int(cli.hidden_dim),
        "seed": int(cli.seed),
        "element_type": cli.element_type,
        "feature_type": cli.feature_type,
        "clustering_method": cli.clustering_method,
        "num_clusters": int(cli.num_clusters),
        "max_iter": int(cli.max_iter),
        "candidate_dir": cli.candidate_dir,
        "candidate_edges_path": candidate_edges_path,
        "candidate_scores_path": candidate_scores_path,
        "candidate_config_hash": candidate_metadata.get("config_hash") if candidate_metadata else None,
        "eval_metric": cli.eval_metric,
        "hessian_type": cli.hessian_type,
        "lissa_iter": int(cli.lissa_iter),
        "scale": float(cli.scale),
        "main_args": vars(experiment_args),
    }


def _load_candidate_metadata(candidate_dir):
    if candidate_dir is None:
        return {}
    metadata_path = os.path.join(candidate_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fill_defaults_from_candidate_metadata(cli, metadata):
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


def _candidate_scores_path(cli):
    if cli.candidate_edge_scores_path is not None:
        return cli.candidate_edge_scores_path
    if cli.candidate_dir is None:
        return None
    path = os.path.join(cli.candidate_dir, "candidate_edge_scores.csv")
    return path if os.path.isfile(path) else None


def _read_csv(path):
    if path is None or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _assignment_rows(feature_rows, labels):
    rows = []
    for row, label in zip(feature_rows, labels.detach().cpu().tolist()):
        copied = dict(row)
        copied["cluster_id"] = int(label)
        rows.append(copied)
    return rows


if __name__ == "__main__":
    main()
