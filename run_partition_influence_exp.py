#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

from main import _create_parser, run_experiment


PROFILE_GRID = {
    "small": {
        "models": ["GCN", "GAT"],
        "layers": [2, 4],
        "datasets": ["cora_public", "citeseer_public", "texas", "cornell"],
    },
    "full": {
        "models": ["SGC", "GCN", "GAT", "ChebNet"],
        "layers": [2, 4, 6, 12],
        "datasets": [
            "cora_public",
            "citeseer_public",
            "pubmed_public",
            "computers",
            "photo",
            "chameleon",
            "actor",
            "squirrel",
            "texas",
            "cornell",
        ],
    },
}

SUMMARY_FIELDS = [
    "status",
    "error_message",
    "profile",
    "run_index",
    "assigned_gpu",
    "run_log",
    "partition_method",
    "partition_strategy",
    "element_type",
    "auto_k_method",
    "num_clusters",
    "num_clusters_spec",
    "dataset",
    "model",
    "num_layers",
    "ratio_group_elem",
    "num_removal_candidates",
    "seed",
    "run_wallclock_sec",
    "result_dir",
    "candidate_results_csv",
    "num_candidates",
    "mean_num_edges",
    "baseline_pbrf_mae",
    "baseline_pbrf_pearson",
    "baseline_pbrf_spearman",
    "baseline_pbrf_sign_acc",
    "cluster_pbrf_mae",
    "cluster_pbrf_pearson",
    "cluster_pbrf_spearman",
    "cluster_pbrf_sign_acc",
    "baseline_cluster_mae",
    "baseline_cluster_pearson",
    "baseline_cluster_spearman",
    "mean_partition_num_groups",
    "mean_partition_weighted_cut",
    "mean_partition_runtime_sec",
    "mean_partition_affinity_density",
    "mean_partition_affinity_num_edges",
    "mean_partition_auto_k_selected",
    "mean_partition_coco_train_runtime_sec",
    "mean_partition_coco_line_graph_num_nodes",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default="small", choices=sorted(PROFILE_GRID.keys()))
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--ratios", type=str, default="1,5,10,30,50,80")
    parser.add_argument("--partition-methods", type=str, default="metis,spectral,local_ppr")
    parser.add_argument("--partition-strategy", type=str, default="candidate_local_affinity")
    parser.add_argument("--partition-strategies", type=str, default=None)
    parser.add_argument(
        "--element-type",
        type=str,
        default="edge_removal",
        choices=["edge_removal", "edge_insertion", "edge_edit"],
    )
    parser.add_argument("--num-clusters", type=int, default=3)
    parser.add_argument("--num-clusters-list", type=str, default=None)
    parser.add_argument("--num-removal-candidates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hessian-type", type=str, default="GNH", choices=["hessian", "GNH"])
    parser.add_argument("--eval-metric", type=str, default="mean_validation_loss")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--pbrf-epochs", type=int, default=1000)
    parser.add_argument("--lissa-iter", type=int, default=1000)
    parser.add_argument("--scale", type=float, default=32.0)
    parser.add_argument("--partition-shared-endpoint-bonus", type=float, default=1.0)
    parser.add_argument("--partition-distance-scale", type=float, default=1.0)
    parser.add_argument("--partition-distance-max-hops", type=int, default=3)
    parser.add_argument("--partition-min-weight", type=float, default=1e-8)
    parser.add_argument("--hybrid-cross-owner-affinity-scale", type=float, default=0.0)
    parser.add_argument("--coco-epochs", type=int, default=200)
    parser.add_argument("--coco-lr", type=float, default=1e-3)
    parser.add_argument("--coco-hidden-dim", type=int, default=256)
    parser.add_argument("--coco-activation", type=str, default="ident")
    parser.add_argument("--coco-compact-k", type=int, default=64)
    parser.add_argument("--coco-stage-num", type=int, default=10)
    parser.add_argument("--coco-beta", type=float, default=1.0)
    parser.add_argument("--coco-filter-steps", type=int, default=2)
    parser.add_argument("--coco-alpha", type=float, default=0.2)
    parser.add_argument("--coco-consistency-t", type=float, default=0.02)
    parser.add_argument("--coco-memory-size", type=int, default=0)
    parser.add_argument("--coco-memory-multiplier", type=int, default=10)
    parser.add_argument("--coco-pca-dim", type=int, default=-1)
    parser.add_argument("--coco-use-diffusion", type=int, default=1)
    parser.add_argument("--coco-kmeans-distance", type=str, default="euclidean")
    parser.add_argument("--coco-device", type=str, default="auto")
    parser.add_argument("--coco-cache-dir", type=str, default="results/coco_line_graph_cache")
    parser.add_argument("--coco-force-retrain", type=int, default=0)
    parser.add_argument("--coco-max-line-graph-nodes", type=int, default=6000)
    parser.add_argument(
        "--auto-k-method",
        type=str,
        default="none",
        choices=[
            "none",
            "eigengap",
            "silhouette",
            "eigengap_silhouette_hybrid",
            "stability",
            "bic_xmeans_like",
        ],
    )
    parser.add_argument("--auto-k-min", type=int, default=1)
    parser.add_argument("--auto-k-max", type=int, default=8)
    parser.add_argument("--auto-k-max-ratio", type=float, default=1.0)
    parser.add_argument("--auto-k-min-cluster-size", type=int, default=1)
    parser.add_argument("--auto-k-max-cluster-size", type=int, default=0)
    parser.add_argument("--auto-k-tiny-graph-threshold", type=int, default=4)
    parser.add_argument("--auto-k-num-restarts", type=int, default=10)
    parser.add_argument("--auto-k-random-seed", type=int, default=0)
    parser.add_argument("--auto-k-silhouette-metric", type=str, default="euclidean")
    parser.add_argument("--auto-k-stability-trials", type=int, default=10)
    parser.add_argument("--auto-k-stability-edge-dropout", type=float, default=0.05)
    parser.add_argument("--auto-k-weak-silhouette-threshold", type=float, default=0.05)
    parser.add_argument("--auto-k-prefer-smaller-k", type=int, default=1)
    parser.add_argument("--summary-csv", type=str, default=None)
    parser.add_argument("--job-dir", type=str, default=None)
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--poll-interval-sec", type=float, default=1.0)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--single-run-spec", type=str, default=None)
    parser.add_argument("--single-run-result", type=str, default=None)
    args = parser.parse_args()

    if args.single_run_spec is not None:
        raise SystemExit(_run_single_worker(args))

    _run_master_scheduler(args)


def _run_master_scheduler(args):
    gpu_ids = _parse_str_csv(args.gpu_ids)
    if len(gpu_ids) == 0:
        raise ValueError("At least one GPU id is required.")

    max_workers = len(gpu_ids) if args.max_workers is None else int(args.max_workers)
    max_workers = max(1, min(max_workers, len(gpu_ids)))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.summary_csv is None:
        args.summary_csv = os.path.join("results", f"partition_compare_summary_{timestamp}.csv")
    if args.job_dir is None:
        args.job_dir = os.path.join("results", "partition_compare_jobs", timestamp)

    os.makedirs(args.job_dir, exist_ok=True)

    planned_runs = deque(_build_planned_runs(args))
    if args.max_runs is not None:
        planned_runs = deque(list(planned_runs)[: max(0, int(args.max_runs))])

    total_runs = len(planned_runs)
    print(
        f"[partition-sweep] profile={args.profile} runs={total_runs} "
        f"summary_csv={args.summary_csv} gpu_ids={','.join(gpu_ids)} max_workers={max_workers}"
    )

    active_jobs = []
    completed_runs = 0
    while planned_runs or active_jobs:
        active_gpu_ids = {job["gpu_id"] for job in active_jobs}
        free_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id not in active_gpu_ids]

        while planned_runs and free_gpu_ids and len(active_jobs) < max_workers:
            run_spec = planned_runs.popleft()
            gpu_id = free_gpu_ids.pop(0)
            job = _launch_worker_job(run_spec=run_spec, args=args, gpu_id=gpu_id)
            active_jobs.append(job)

        completed_jobs = []
        for job in active_jobs:
            return_code = job["process"].poll()
            if return_code is None:
                continue

            job["log_handle"].close()
            summary_row = _collect_worker_result(job=job, args=args, return_code=return_code)
            _append_summary_row(args.summary_csv, summary_row)
            completed_jobs.append(job)
            completed_runs += 1
            print(
                f"[scheduler] completed {completed_runs}/{total_runs} "
                f"gpu={job['gpu_id']} run={job['run_index']} status={summary_row['status']}"
            )

        if completed_jobs:
            active_jobs = [job for job in active_jobs if job not in completed_jobs]
            continue

        if active_jobs:
            time.sleep(max(0.1, float(args.poll_interval_sec)))


def _launch_worker_job(run_spec, args, gpu_id):
    run_index = int(run_spec["run_index"])
    spec_path = Path(args.job_dir) / f"run_{run_index:04d}_spec.json"
    result_path = Path(args.job_dir) / f"run_{run_index:04d}_result.json"
    log_path = Path(args.job_dir) / f"run_{run_index:04d}.log"

    payload = {
        "run_spec": run_spec,
        "config": _extract_master_config(args),
    }
    spec_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    child_env["EIF_RESPECT_CUDA_VISIBLE_DEVICES"] = "1"
    child_env["PARTITION_ASSIGNED_GPU"] = str(gpu_id)
    child_env["PYTHONUNBUFFERED"] = "1"

    log_handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-u", __file__, "--single-run-spec", str(spec_path), "--single-run-result", str(result_path)],
        cwd=str(Path(__file__).resolve().parent),
        env=child_env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    print(
        f"[scheduler] launched gpu={gpu_id} run={run_index} "
        f"strategy={run_spec['partition_strategy']} method={run_spec['partition_method']} "
        f"k={run_spec['num_clusters']} spec={run_spec.get('num_clusters_spec', run_spec['num_clusters'])} "
        f"model={run_spec['model']} "
        f"layers={run_spec['num_layers']} dataset={run_spec['dataset']} ratio={run_spec['ratio']}"
    )
    return {
        "process": process,
        "gpu_id": str(gpu_id),
        "run_index": run_index,
        "spec_path": spec_path,
        "result_path": result_path,
        "log_path": log_path,
        "log_handle": log_handle,
        "run_spec": run_spec,
    }


def _collect_worker_result(job, args, return_code):
    summary_row = _init_summary_row(run_spec=job["run_spec"], config=_extract_master_config(args))
    summary_row["assigned_gpu"] = str(job["gpu_id"])
    summary_row["run_log"] = str(job["log_path"])

    if job["result_path"].is_file():
        try:
            loaded = json.loads(job["result_path"].read_text(encoding="utf-8"))
            for field in SUMMARY_FIELDS:
                if field in loaded:
                    summary_row[field] = loaded[field]
        except json.JSONDecodeError as exc:
            summary_row["status"] = "failed"
            summary_row["error_message"] = f"Invalid worker result JSON: {exc}"
    elif return_code != 0:
        summary_row["status"] = "failed"
        summary_row["error_message"] = f"Worker exited with code {return_code} and no result file."
    else:
        summary_row["status"] = "failed"
        summary_row["error_message"] = "Worker completed without producing a result file."

    summary_row["assigned_gpu"] = str(job["gpu_id"])
    summary_row["run_log"] = str(job["log_path"])
    return summary_row


def _run_single_worker(args):
    if args.single_run_result is None:
        raise ValueError("--single-run-result is required with --single-run-spec.")

    payload = json.loads(Path(args.single_run_spec).read_text(encoding="utf-8"))
    run_spec = payload["run_spec"]
    config = payload["config"]

    summary_row = _init_summary_row(run_spec=run_spec, config=config)
    summary_row["assigned_gpu"] = os.environ.get("PARTITION_ASSIGNED_GPU", os.environ.get("CUDA_VISIBLE_DEVICES"))

    try:
        start_time = time.perf_counter()
        experiment_args = build_experiment_args(
            dataset=run_spec["dataset"],
            model=run_spec["model"],
            num_layers=run_spec["num_layers"],
            ratio=run_spec["ratio"],
            partition_method=run_spec["partition_method"],
            partition_strategy=run_spec.get("partition_strategy", "candidate_local_affinity"),
            element_type=config["element_type"],
            num_clusters=run_spec["num_clusters"],
            num_removal_candidates=config["num_removal_candidates"],
            seed=config["seed"],
            hessian_type=config["hessian_type"],
            eval_metric=config["eval_metric"],
            lr=config["lr"],
            hidden_dim=config["hidden_dim"],
            epochs=config["epochs"],
            weight_decay=config["weight_decay"],
            pbrf_epochs=config["pbrf_epochs"],
            lissa_iter=config["lissa_iter"],
            scale=config["scale"],
            partition_shared_endpoint_bonus=config["partition_shared_endpoint_bonus"],
            partition_distance_scale=config["partition_distance_scale"],
            partition_distance_max_hops=config["partition_distance_max_hops"],
            partition_min_weight=config["partition_min_weight"],
            hybrid_cross_owner_affinity_scale=config["hybrid_cross_owner_affinity_scale"],
            coco_epochs=config["coco_epochs"],
            coco_lr=config["coco_lr"],
            coco_hidden_dim=config["coco_hidden_dim"],
            coco_activation=config["coco_activation"],
            coco_compact_k=config["coco_compact_k"],
            coco_stage_num=config["coco_stage_num"],
            coco_beta=config["coco_beta"],
            coco_filter_steps=config["coco_filter_steps"],
            coco_alpha=config["coco_alpha"],
            coco_consistency_t=config["coco_consistency_t"],
            coco_memory_size=config["coco_memory_size"],
            coco_memory_multiplier=config["coco_memory_multiplier"],
            coco_pca_dim=config["coco_pca_dim"],
            coco_use_diffusion=config["coco_use_diffusion"],
            coco_kmeans_distance=config["coco_kmeans_distance"],
            coco_device=config["coco_device"],
            coco_cache_dir=config["coco_cache_dir"],
            coco_force_retrain=config["coco_force_retrain"],
            coco_max_line_graph_nodes=config["coco_max_line_graph_nodes"],
            auto_k_method=config["auto_k_method"],
            auto_k_min=config["auto_k_min"],
            auto_k_max=config["auto_k_max"],
            auto_k_max_ratio=config["auto_k_max_ratio"],
            auto_k_min_cluster_size=config["auto_k_min_cluster_size"],
            auto_k_max_cluster_size=config["auto_k_max_cluster_size"],
            auto_k_tiny_graph_threshold=config["auto_k_tiny_graph_threshold"],
            auto_k_num_restarts=config["auto_k_num_restarts"],
            auto_k_random_seed=config["auto_k_random_seed"],
            auto_k_silhouette_metric=config["auto_k_silhouette_metric"],
            auto_k_stability_trials=config["auto_k_stability_trials"],
            auto_k_stability_edge_dropout=config["auto_k_stability_edge_dropout"],
            auto_k_weak_silhouette_threshold=config["auto_k_weak_silhouette_threshold"],
            auto_k_prefer_smaller_k=config["auto_k_prefer_smaller_k"],
        )
        dirs = run_experiment(experiment_args)
        summary_row["run_wallclock_sec"] = float(time.perf_counter() - start_time)
        candidate_csv = os.path.join(dirs["result"], "candidate_results.csv")
        candidate_csvs = [candidate_csv]
        if config["element_type"] == "edge_edit" and not os.path.isfile(candidate_csv):
            candidate_csvs = [
                os.path.join(dirs["result"], "removal_candidate_results.csv"),
                os.path.join(dirs["result"], "insertion_candidate_results.csv"),
            ]
        summary_row["result_dir"] = dirs["result"]
        summary_row["candidate_results_csv"] = "|".join(candidate_csvs)
        summary_row.update(_summarize_candidate_results(candidate_csvs))
        summary_row["status"] = "ok"
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error_message"] = f"{type(exc).__name__}: {exc}"
        print(traceback.format_exc(), flush=True)

    Path(args.single_run_result).write_text(json.dumps(summary_row, indent=2), encoding="utf-8")
    return 0 if summary_row["status"] == "ok" else 1


def _build_planned_runs(args):
    ratios = _parse_int_csv(args.ratios)
    partition_methods = _parse_str_csv(args.partition_methods)
    partition_strategies = (
        [str(args.partition_strategy)]
        if args.partition_strategies is None
        else _parse_str_csv(args.partition_strategies)
    )
    num_clusters_specs = [str(args.num_clusters)] if args.num_clusters_list is None else _parse_str_csv(args.num_clusters_list)
    _validate_partition_strategy_methods(partition_methods=partition_methods, partition_strategies=partition_strategies)
    grid = _resolve_grid(args)
    graph_stats_cache = {}

    planned_runs = []
    run_index = 1
    for partition_strategy in partition_strategies:
        for partition_method in partition_methods:
            for model in grid["models"]:
                for num_layers in grid["layers"]:
                    for dataset in grid["datasets"]:
                        for ratio in ratios:
                            resolved_num_clusters = _resolve_num_clusters_specs(
                                specs=num_clusters_specs,
                                dataset_name=dataset,
                                ratio=ratio,
                                graph_stats_cache=graph_stats_cache,
                            )
                            for cluster_entry in resolved_num_clusters:
                                planned_runs.append(
                                    {
                                        "run_index": run_index,
                                        "partition_method": partition_method,
                                        "partition_strategy": str(partition_strategy),
                                        "num_clusters": int(cluster_entry["num_clusters"]),
                                        "num_clusters_spec": str(cluster_entry["spec"]),
                                        "num_clusters_uncapped": int(cluster_entry["uncapped"]),
                                        "num_clusters_cap": int(cluster_entry["cap"]),
                                        "model": model,
                                        "num_layers": num_layers,
                                        "dataset": dataset,
                                        "ratio": ratio,
                                    }
                                )
                                run_index += 1
    return planned_runs


def _extract_master_config(args):
    return {
        "profile": args.profile,
        "element_type": args.element_type,
        "num_clusters": int(args.num_clusters),
        "num_clusters_list": args.num_clusters_list,
        "num_removal_candidates": int(args.num_removal_candidates),
        "seed": int(args.seed),
        "hessian_type": args.hessian_type,
        "eval_metric": args.eval_metric,
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "epochs": int(args.epochs),
        "weight_decay": float(args.weight_decay),
        "pbrf_epochs": int(args.pbrf_epochs),
        "lissa_iter": int(args.lissa_iter),
        "scale": float(args.scale),
        "partition_shared_endpoint_bonus": float(args.partition_shared_endpoint_bonus),
        "partition_distance_scale": float(args.partition_distance_scale),
        "partition_distance_max_hops": int(args.partition_distance_max_hops),
        "partition_min_weight": float(args.partition_min_weight),
        "hybrid_cross_owner_affinity_scale": float(args.hybrid_cross_owner_affinity_scale),
        "coco_epochs": int(args.coco_epochs),
        "coco_lr": float(args.coco_lr),
        "coco_hidden_dim": int(args.coco_hidden_dim),
        "coco_activation": str(args.coco_activation),
        "coco_compact_k": int(args.coco_compact_k),
        "coco_stage_num": int(args.coco_stage_num),
        "coco_beta": float(args.coco_beta),
        "coco_filter_steps": int(args.coco_filter_steps),
        "coco_alpha": float(args.coco_alpha),
        "coco_consistency_t": float(args.coco_consistency_t),
        "coco_memory_size": int(args.coco_memory_size),
        "coco_memory_multiplier": int(args.coco_memory_multiplier),
        "coco_pca_dim": int(args.coco_pca_dim),
        "coco_use_diffusion": int(args.coco_use_diffusion),
        "coco_kmeans_distance": str(args.coco_kmeans_distance),
        "coco_device": str(args.coco_device),
        "coco_cache_dir": str(args.coco_cache_dir),
        "coco_force_retrain": int(args.coco_force_retrain),
        "coco_max_line_graph_nodes": int(args.coco_max_line_graph_nodes),
        "auto_k_method": str(args.auto_k_method),
        "auto_k_min": int(args.auto_k_min),
        "auto_k_max": int(args.auto_k_max),
        "auto_k_max_ratio": float(args.auto_k_max_ratio),
        "auto_k_min_cluster_size": int(args.auto_k_min_cluster_size),
        "auto_k_max_cluster_size": int(args.auto_k_max_cluster_size),
        "auto_k_tiny_graph_threshold": int(args.auto_k_tiny_graph_threshold),
        "auto_k_num_restarts": int(args.auto_k_num_restarts),
        "auto_k_random_seed": int(args.auto_k_random_seed),
        "auto_k_silhouette_metric": str(args.auto_k_silhouette_metric),
        "auto_k_stability_trials": int(args.auto_k_stability_trials),
        "auto_k_stability_edge_dropout": float(args.auto_k_stability_edge_dropout),
        "auto_k_weak_silhouette_threshold": float(args.auto_k_weak_silhouette_threshold),
        "auto_k_prefer_smaller_k": int(args.auto_k_prefer_smaller_k),
    }


def _init_summary_row(run_spec, config):
    return {
        "status": "failed",
        "error_message": None,
        "profile": config["profile"],
        "run_index": int(run_spec["run_index"]),
        "assigned_gpu": None,
        "run_log": None,
        "partition_method": run_spec["partition_method"],
        "partition_strategy": run_spec.get("partition_strategy", "candidate_local_affinity"),
        "element_type": config["element_type"],
        "auto_k_method": config["auto_k_method"],
        "num_clusters": int(run_spec.get("num_clusters", config["num_clusters"])),
        "num_clusters_spec": run_spec.get("num_clusters_spec", run_spec.get("num_clusters", config["num_clusters"])),
        "dataset": run_spec["dataset"],
        "model": run_spec["model"],
        "num_layers": int(run_spec["num_layers"]),
        "ratio_group_elem": int(run_spec["ratio"]),
        "num_removal_candidates": int(config["num_removal_candidates"]),
        "seed": int(config["seed"]),
        "run_wallclock_sec": None,
        "result_dir": None,
        "candidate_results_csv": None,
        "num_candidates": None,
        "mean_num_edges": None,
        "baseline_pbrf_mae": None,
        "baseline_pbrf_pearson": None,
        "baseline_pbrf_spearman": None,
        "baseline_pbrf_sign_acc": None,
        "cluster_pbrf_mae": None,
        "cluster_pbrf_pearson": None,
        "cluster_pbrf_spearman": None,
        "cluster_pbrf_sign_acc": None,
        "baseline_cluster_mae": None,
        "baseline_cluster_pearson": None,
        "baseline_cluster_spearman": None,
        "mean_partition_num_groups": None,
        "mean_partition_weighted_cut": None,
        "mean_partition_runtime_sec": None,
        "mean_partition_affinity_density": None,
        "mean_partition_affinity_num_edges": None,
        "mean_partition_auto_k_selected": None,
        "mean_partition_coco_train_runtime_sec": None,
        "mean_partition_coco_line_graph_num_nodes": None,
    }


def build_experiment_args(
    dataset,
    model,
    num_layers,
    ratio,
    partition_method,
    partition_strategy,
    element_type,
    num_clusters,
    num_removal_candidates,
    seed,
    hessian_type,
    eval_metric,
    lr,
    hidden_dim,
    epochs,
    weight_decay,
    pbrf_epochs,
    lissa_iter,
    scale,
    partition_shared_endpoint_bonus,
    partition_distance_scale,
    partition_distance_max_hops,
    partition_min_weight,
    hybrid_cross_owner_affinity_scale,
    coco_epochs,
    coco_lr,
    coco_hidden_dim,
    coco_activation,
    coco_compact_k,
    coco_stage_num,
    coco_beta,
    coco_filter_steps,
    coco_alpha,
    coco_consistency_t,
    coco_memory_size,
    coco_memory_multiplier,
    coco_pca_dim,
    coco_use_diffusion,
    coco_kmeans_distance,
    coco_device,
    coco_cache_dir,
    coco_force_retrain,
    coco_max_line_graph_nodes,
    auto_k_method,
    auto_k_min,
    auto_k_max,
    auto_k_max_ratio,
    auto_k_min_cluster_size,
    auto_k_max_cluster_size,
    auto_k_tiny_graph_threshold,
    auto_k_num_restarts,
    auto_k_random_seed,
    auto_k_silhouette_metric,
    auto_k_stability_trials,
    auto_k_stability_edge_dropout,
    auto_k_weak_silhouette_threshold,
    auto_k_prefer_smaller_k,
):
    experiment_args = _create_parser().parse_args([])
    experiment_args.dataset = str(dataset)
    experiment_args.model = str(model)
    experiment_args.num_layers = int(num_layers)
    experiment_args.seed = int(seed)
    experiment_args.hessian_type = str(hessian_type)
    experiment_args.eval_metric = str(eval_metric)
    experiment_args.lr = float(lr)
    experiment_args.hidden_dim = int(hidden_dim)
    experiment_args.epochs = int(epochs)
    experiment_args.weight_decay = float(weight_decay)
    experiment_args.pbrf_epochs = int(pbrf_epochs)
    experiment_args.lissa_iter = int(lissa_iter)
    experiment_args.scale = float(scale)
    experiment_args.element_type = str(element_type)
    experiment_args.experiment_name = "large_drop_influence"
    experiment_args.ratio_group_elem = int(ratio)
    experiment_args.num_removal_candidates = int(num_removal_candidates)
    experiment_args.num_insertion_candidates = int(num_removal_candidates)
    experiment_args.metric_mode = "partition"
    experiment_args.influence_mode = "calculate_influence"
    experiment_args.influence_calculation_mode = "calculate_influence"
    experiment_args.num_of_clusters = int(num_clusters)
    experiment_args.partition_method = str(partition_method)
    experiment_args.partition_strategy = str(partition_strategy)
    experiment_args.partition_shared_endpoint_bonus = float(partition_shared_endpoint_bonus)
    experiment_args.partition_distance_scale = float(partition_distance_scale)
    experiment_args.partition_distance_max_hops = int(partition_distance_max_hops)
    experiment_args.partition_min_weight = float(partition_min_weight)
    experiment_args.hybrid_cross_owner_affinity_scale = float(hybrid_cross_owner_affinity_scale)
    experiment_args.coco_epochs = int(coco_epochs)
    experiment_args.coco_lr = float(coco_lr)
    experiment_args.coco_hidden_dim = int(coco_hidden_dim)
    experiment_args.coco_activation = str(coco_activation)
    experiment_args.coco_compact_k = int(coco_compact_k)
    experiment_args.coco_stage_num = int(coco_stage_num)
    experiment_args.coco_beta = float(coco_beta)
    experiment_args.coco_filter_steps = int(coco_filter_steps)
    experiment_args.coco_alpha = float(coco_alpha)
    experiment_args.coco_consistency_t = float(coco_consistency_t)
    experiment_args.coco_memory_size = int(coco_memory_size)
    experiment_args.coco_memory_multiplier = int(coco_memory_multiplier)
    experiment_args.coco_pca_dim = int(coco_pca_dim)
    experiment_args.coco_use_diffusion = int(coco_use_diffusion)
    experiment_args.coco_kmeans_distance = str(coco_kmeans_distance)
    experiment_args.coco_device = str(coco_device)
    experiment_args.coco_cache_dir = str(coco_cache_dir)
    experiment_args.coco_force_retrain = int(coco_force_retrain)
    experiment_args.coco_max_line_graph_nodes = int(coco_max_line_graph_nodes)
    experiment_args.auto_k_method = str(auto_k_method)
    experiment_args.auto_k_min = int(auto_k_min)
    experiment_args.auto_k_max = int(auto_k_max)
    experiment_args.auto_k_max_ratio = float(auto_k_max_ratio)
    experiment_args.auto_k_min_cluster_size = int(auto_k_min_cluster_size)
    experiment_args.auto_k_max_cluster_size = int(auto_k_max_cluster_size)
    experiment_args.auto_k_tiny_graph_threshold = int(auto_k_tiny_graph_threshold)
    experiment_args.auto_k_num_restarts = int(auto_k_num_restarts)
    experiment_args.auto_k_random_seed = int(auto_k_random_seed)
    experiment_args.auto_k_silhouette_metric = str(auto_k_silhouette_metric)
    experiment_args.auto_k_stability_trials = int(auto_k_stability_trials)
    experiment_args.auto_k_stability_edge_dropout = float(auto_k_stability_edge_dropout)
    experiment_args.auto_k_weak_silhouette_threshold = float(auto_k_weak_silhouette_threshold)
    experiment_args.auto_k_prefer_smaller_k = int(auto_k_prefer_smaller_k)
    experiment_args.json_config = "none"
    experiment_args.fig_title = "none"
    return experiment_args


def _summarize_candidate_results(candidate_csv):
    candidate_csvs = candidate_csv if isinstance(candidate_csv, (list, tuple)) else [candidate_csv]
    for path in candidate_csvs:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"candidate results not found: {path}")

    rows = []
    for path in candidate_csvs:
        with open(path, "r", encoding="utf-8") as handle:
            rows.extend(list(csv.DictReader(handle)))

    baseline = []
    cluster = []
    pbrf = []
    num_edges = []
    partition_num_groups = []
    partition_weighted_cut = []
    partition_runtime = []
    partition_density = []
    partition_num_edges = []
    partition_auto_k_selected = []
    partition_coco_train_runtime = []
    partition_coco_line_graph_nodes = []

    for row in rows:
        _append_if_number(num_edges, row.get("num_edges"))
        _append_if_number(partition_num_groups, row.get("partition_num_groups"))
        _append_if_number(partition_weighted_cut, row.get("partition_weighted_cut"))
        _append_if_number(partition_runtime, row.get("partition_runtime_sec"))
        _append_if_number(partition_density, row.get("partition_affinity_density"))
        _append_if_number(partition_num_edges, row.get("partition_affinity_num_edges"))
        _append_if_number(partition_auto_k_selected, row.get("partition_auto_k_selected"))
        _append_if_number(partition_coco_train_runtime, row.get("partition_coco_train_runtime_sec"))
        _append_if_number(partition_coco_line_graph_nodes, row.get("partition_coco_line_graph_num_nodes"))

        baseline_val = _parse_float(row.get("calculate_influence_total"))
        cluster_val = _parse_float(row.get("clusterwise_fixed_theta_total"))
        pbrf_val = _parse_float(row.get("pbrf_total"))

        baseline.append(baseline_val)
        cluster.append(cluster_val)
        pbrf.append(pbrf_val)

    summary = {
        "num_candidates": int(len(rows)),
        "mean_num_edges": _safe_mean(num_edges),
        "mean_partition_num_groups": _safe_mean(partition_num_groups),
        "mean_partition_weighted_cut": _safe_mean(partition_weighted_cut),
        "mean_partition_runtime_sec": _safe_mean(partition_runtime),
        "mean_partition_affinity_density": _safe_mean(partition_density),
        "mean_partition_affinity_num_edges": _safe_mean(partition_num_edges),
        "mean_partition_auto_k_selected": _safe_mean(partition_auto_k_selected),
        "mean_partition_coco_train_runtime_sec": _safe_mean(partition_coco_train_runtime),
        "mean_partition_coco_line_graph_num_nodes": _safe_mean(partition_coco_line_graph_nodes),
    }

    summary.update(_pair_metrics(baseline, pbrf, prefix="baseline_pbrf"))
    summary.update(_pair_metrics(cluster, pbrf, prefix="cluster_pbrf"))
    summary.update(_pair_metrics(baseline, cluster, prefix="baseline_cluster", include_sign_acc=False))
    return summary


def _pair_metrics(pred_values, target_values, prefix, include_sign_acc=True):
    pairs = _paired_numeric_values(pred_values, target_values)
    if len(pairs) == 0:
        metrics = {
            f"{prefix}_mae": None,
            f"{prefix}_pearson": None,
            f"{prefix}_spearman": None,
        }
        if include_sign_acc:
            metrics[f"{prefix}_sign_acc"] = None
        return metrics

    pred = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    target = np.asarray([pair[1] for pair in pairs], dtype=np.float64)

    metrics = {
        f"{prefix}_mae": float(np.mean(np.abs(pred - target))),
        f"{prefix}_pearson": _safe_corrcoef(pred, target),
        f"{prefix}_spearman": _safe_spearman(pred, target),
    }
    if include_sign_acc:
        metrics[f"{prefix}_sign_acc"] = float(np.mean(np.sign(pred) == np.sign(target)))
    return metrics


def _paired_numeric_values(pred_values, target_values):
    pairs = []
    for pred, target in zip(pred_values, target_values):
        pred_num = _parse_float(pred)
        target_num = _parse_float(target)
        if pred_num is None or target_num is None:
            continue
        pairs.append((pred_num, target_num))
    return pairs


def _safe_corrcoef(lhs, rhs):
    if lhs.size < 2 or rhs.size < 2:
        return None
    if np.allclose(lhs, lhs[0]) or np.allclose(rhs, rhs[0]):
        return None
    value = float(np.corrcoef(lhs, rhs)[0, 1])
    if not math.isfinite(value):
        return None
    return value


def _safe_spearman(lhs, rhs):
    if lhs.size < 2 or rhs.size < 2:
        return None
    correlation = spearmanr(lhs, rhs).correlation
    if correlation is None:
        return None
    correlation = float(correlation)
    if not math.isfinite(correlation):
        return None
    return correlation


def _append_if_number(values, raw_value):
    parsed = _parse_float(raw_value)
    if parsed is not None:
        values.append(parsed)


def _safe_mean(values):
    if len(values) == 0:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _parse_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None
        return float(value)

    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _parse_int_csv(text):
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if token == "":
            continue
        values.append(int(token))
    return values


def _resolve_num_clusters_specs(specs, dataset_name, ratio, graph_stats_cache):
    stats = None
    if any(_cluster_spec_needs_graph_stats(spec) for spec in specs):
        stats = _get_dataset_graph_stats(dataset_name, graph_stats_cache)
    candidate_edges = None
    if stats is not None:
        candidate_edges = max(1, int(round(float(stats["num_edges"]) * (float(ratio) / 100.0))))
    resolved = []
    seen = set()
    for raw_spec in specs:
        spec = str(raw_spec).strip()
        if spec == "":
            continue
        entry = _resolve_num_clusters_spec(spec, stats, candidate_edges=candidate_edges)
        if stats is not None:
            cap = max(1, min(int(stats["num_edges"]), int(candidate_edges)))
        else:
            cap = max(1, int(entry["uncapped"]))
        num_clusters = max(1, min(int(entry["uncapped"]), int(cap)))
        key = int(num_clusters)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(
            {
                "num_clusters": int(num_clusters),
                "spec": entry["spec"],
                "uncapped": int(entry["uncapped"]),
                "cap": int(cap),
            }
        )

    if len(resolved) == 0:
        raise ValueError("At least one num-clusters spec is required.")
    return resolved


def _cluster_spec_needs_graph_stats(spec):
    normalized = str(spec).strip().lower()
    if _is_int_token(normalized):
        return False
    return True


def _resolve_num_clusters_spec(spec, stats, candidate_edges=None):
    normalized = str(spec).strip().lower()
    if _is_int_token(normalized):
        value = int(normalized)
        return {"num_clusters": value, "uncapped": value, "spec": normalized}

    if stats is None:
        raise ValueError(f"num_clusters spec requires graph stats: {spec}")

    if _is_float_token(normalized):
        fraction = float(normalized)
        if fraction <= 0:
            raise ValueError(f"num_clusters edge ratio must be positive: {spec}")
        value = int(round(float(stats["num_edges"]) * fraction))
        return {"num_clusters": value, "uncapped": value, "spec": f"edge_ratio:{fraction:g}"}

    if ":" in normalized:
        kind, raw_value = normalized.split(":", 1)
        kind = kind.strip()
        raw_value = raw_value.strip()
        value, explicit_percent = _parse_value_with_percent(raw_value)
        if kind in {"edge_ratio", "edge_frac", "edge_fraction"}:
            resolved = int(round(float(stats["num_edges"]) * value))
        elif kind in {"edge_pct", "edge_percent"}:
            fraction = value if explicit_percent else value / 100.0
            resolved = int(round(float(stats["num_edges"]) * fraction))
        elif kind in {"node_ratio", "node_frac", "node_fraction"}:
            resolved = int(round(float(stats["num_nodes"]) * value))
        elif kind in {"node_pct", "node_percent"}:
            fraction = value if explicit_percent else value / 100.0
            resolved = int(round(float(stats["num_nodes"]) * fraction))
        elif kind in {"edge_cluster_size", "full_edge_cluster_size", "edges_per_cluster"}:
            if value <= 0:
                raise ValueError(f"edge_cluster_size must be positive: {spec}")
            resolved = int(math.ceil(float(stats["num_edges"]) / value))
        elif kind in {
            "candidate_cluster_size",
            "candidate_edges_per_cluster",
            "cand_cluster_size",
            "cand_edges_per_cluster",
        }:
            if candidate_edges is None:
                raise ValueError(f"candidate cluster-size spec requires candidate edge count: {spec}")
            if value <= 0:
                raise ValueError(f"candidate_cluster_size must be positive: {spec}")
            resolved = int(math.ceil(float(candidate_edges) / value))
        elif kind in {"candidate_ratio", "candidate_frac", "candidate_fraction", "cand_ratio", "cand_frac"}:
            if candidate_edges is None:
                raise ValueError(f"candidate ratio spec requires candidate edge count: {spec}")
            resolved = int(round(float(candidate_edges) * value))
        elif kind in {"candidate_pct", "candidate_percent", "cand_pct", "cand_percent"}:
            if candidate_edges is None:
                raise ValueError(f"candidate percent spec requires candidate edge count: {spec}")
            fraction = value if explicit_percent else value / 100.0
            resolved = int(round(float(candidate_edges) * fraction))
        elif kind in {"avg_degree", "average_degree"}:
            resolved = int(round(float(stats["avg_degree"]) * value))
        elif kind in {"sqrt_edges"}:
            resolved = int(round(math.sqrt(float(stats["num_edges"])) * value))
        elif kind in {"sqrt_nodes"}:
            resolved = int(round(math.sqrt(float(stats["num_nodes"])) * value))
        elif kind in {"sqrt_candidate_edges", "sqrt_candidate", "sqrt_cand_edges"}:
            if candidate_edges is None:
                raise ValueError(f"sqrt_candidate_edges spec requires candidate edge count: {spec}")
            resolved = int(round(math.sqrt(float(candidate_edges)) * value))
        elif kind in {"log2_candidate_edges", "log2_candidate", "log2_cand_edges"}:
            if candidate_edges is None:
                raise ValueError(f"log2_candidate_edges spec requires candidate edge count: {spec}")
            resolved = int(round(math.log2(max(2.0, float(candidate_edges))) * value))
        elif kind in {"log2_edges"}:
            resolved = int(round(math.log2(max(2.0, float(stats["num_edges"]))) * value))
        elif kind in {"log2_nodes"}:
            resolved = int(round(math.log2(max(2.0, float(stats["num_nodes"]))) * value))
        else:
            raise ValueError(f"Unknown num_clusters spec: {spec}")
        return {"num_clusters": resolved, "uncapped": resolved, "spec": normalized}

    if normalized == "sqrt_edges":
        value = int(round(math.sqrt(float(stats["num_edges"]))))
    elif normalized == "sqrt_nodes":
        value = int(round(math.sqrt(float(stats["num_nodes"]))))
    elif normalized == "log2_edges":
        value = int(round(math.log2(max(2.0, float(stats["num_edges"])))))
    elif normalized == "log2_nodes":
        value = int(round(math.log2(max(2.0, float(stats["num_nodes"])))))
    elif normalized in {"avg_degree", "average_degree"}:
        value = int(round(float(stats["avg_degree"])))
    elif normalized in {"sqrt_candidate_edges", "sqrt_candidate", "sqrt_cand_edges"}:
        if candidate_edges is None:
            raise ValueError(f"sqrt_candidate_edges spec requires candidate edge count: {spec}")
        value = int(round(math.sqrt(float(candidate_edges))))
    elif normalized in {"log2_candidate_edges", "log2_candidate", "log2_cand_edges"}:
        if candidate_edges is None:
            raise ValueError(f"log2_candidate_edges spec requires candidate edge count: {spec}")
        value = int(round(math.log2(max(2.0, float(candidate_edges)))))
    else:
        raise ValueError(
            f"Unknown num_clusters spec: {spec}. Supported examples: "
            "edge_cluster_size:32, candidate_cluster_size:16, edge_pct:1, "
            "sqrt_edges, sqrt_candidate_edges, log2_edges, avg_degree."
        )
    return {"num_clusters": value, "uncapped": value, "spec": normalized}


def _parse_value_with_percent(raw_value):
    text = str(raw_value).strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100.0, True
    return float(text), False


def _is_int_token(text):
    try:
        int(str(text).strip())
    except ValueError:
        return False
    return str(text).strip().lstrip("-").isdigit()


def _is_float_token(text):
    try:
        float(str(text).strip())
    except ValueError:
        return False
    return not _is_int_token(text)


def _get_dataset_graph_stats(dataset_name, graph_stats_cache):
    dataset_key = str(dataset_name).strip().lower()
    if dataset_key in graph_stats_cache:
        return graph_stats_cache[dataset_key]

    from src import DataLoader

    dataset = DataLoader(dataset_key, root="datasets")
    data = dataset[0]
    num_nodes = int(getattr(data, "num_nodes", 0) or data.x.shape[0])
    edge_index = data.edge_index.detach().cpu().long().numpy()
    if edge_index.shape[0] != 2:
        raise ValueError(f"Dataset {dataset_name} has invalid edge_index shape: {tuple(edge_index.shape)}")

    edges = np.asarray(edge_index.T, dtype=np.int64)
    edges = np.sort(edges, axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if edges.size == 0:
        num_edges = 0
    else:
        num_edges = int(np.unique(edges, axis=0).shape[0])
    if num_edges <= 0:
        raise ValueError(f"Dataset {dataset_name} has no non-self-loop edges.")

    avg_degree = float(2.0 * num_edges / max(1, num_nodes))
    density = 0.0
    if num_nodes > 1:
        density = float(2.0 * num_edges / float(num_nodes * (num_nodes - 1)))
    stats = {
        "dataset": dataset_key,
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "avg_degree": avg_degree,
        "density": density,
    }
    graph_stats_cache[dataset_key] = stats
    print(
        f"[cluster-spec] dataset={dataset_key} nodes={num_nodes} "
        f"undirected_edges={num_edges} avg_degree={avg_degree:.3f} density={density:.6f}"
    )
    return stats


def _parse_str_csv(text):
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if token == "":
            continue
        values.append(token)
    return values


def _validate_partition_strategy_methods(partition_methods, partition_strategies):
    normalized_methods = [str(method).strip().lower() for method in partition_methods]
    normalized_strategies = [str(strategy).strip().lower() for strategy in partition_strategies]

    metis_global_strategies = {"global_training_graph_assignment", "hybrid_training_graph_masked_local"}
    coco_strategies = {"coco_full_line_graph_assignment", "coco_candidate_line_graph"}

    if any(strategy in metis_global_strategies for strategy in normalized_strategies):
        invalid_methods = [method for method in normalized_methods if method != "metis"]
        if invalid_methods:
            raise ValueError(
                "partition_strategies that use global training-graph METIS currently require "
                f"partition_methods=metis only. Invalid methods: {invalid_methods}"
            )

    if any(strategy in coco_strategies for strategy in normalized_strategies):
        invalid_methods = [method for method in normalized_methods if method != "coco"]
        if invalid_methods:
            raise ValueError(
                "CoCo line-graph partition strategies require partition_methods=coco only. "
                f"Invalid methods: {invalid_methods}"
            )

    if "coco" in normalized_methods:
        invalid_strategies = [strategy for strategy in normalized_strategies if strategy not in coco_strategies]
        if invalid_strategies:
            raise ValueError(
                "partition_methods=coco requires a CoCo partition strategy. "
                f"Invalid strategies: {invalid_strategies}"
            )


def _resolve_grid(args):
    profile_grid = PROFILE_GRID[args.profile]
    return {
        "models": _resolve_grid_selection(
            raw_value=args.models,
            defaults=profile_grid["models"],
            field_name="models",
            parser=_parse_str_csv,
        ),
        "layers": _resolve_grid_selection(
            raw_value=args.layers,
            defaults=profile_grid["layers"],
            field_name="layers",
            parser=_parse_int_csv,
        ),
        "datasets": _resolve_grid_selection(
            raw_value=args.datasets,
            defaults=profile_grid["datasets"],
            field_name="datasets",
            parser=_parse_str_csv,
        ),
    }


def _resolve_grid_selection(raw_value, defaults, field_name, parser):
    if raw_value is None:
        return list(defaults)

    selected = parser(raw_value)
    if len(selected) == 0:
        raise ValueError(f"At least one value is required for {field_name}.")

    default_set = set(defaults)
    invalid = [value for value in selected if value not in default_set]
    if invalid:
        raise ValueError(
            f"Unknown {field_name}: {invalid}. Available values for this profile: {list(defaults)}"
        )
    return selected


def _append_summary_row(summary_csv, row):
    summary_dir = os.path.dirname(summary_csv)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    file_exists = os.path.isfile(summary_csv)
    normalized_row = {field: row.get(field, None) for field in SUMMARY_FIELDS}
    with open(summary_csv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(normalized_row)


if __name__ == "__main__":
    main()
