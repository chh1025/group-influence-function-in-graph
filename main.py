import argparse
import errno
import json
import math
import os
import os.path as osp
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

DEFAULT_SEED = 1941488137
NUM_GPUS = 4
CONSTRAINT_UNSAT_MARKER = "CONSTRAINT_UNSAT"
EDGE_GROUP_CONSTRAINT_ERROR_TEXT = "Failed to sample a valid edge group with the current constraints."


def _ensure_repo_on_pythonpath():
    repo_root = str(Path(__file__).resolve().parent)
    current = os.environ.get("PYTHONPATH", "")
    paths = [p for p in current.split(os.pathsep) if p]
    if repo_root not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_root] + paths) if paths else repo_root


def _normalize_influence_mode(mode):
    if mode == "fixed_theta":
        return "calculate_influence"
    return mode


def _patch_submitit_entrypoint_compat():
    """
    Compatibility patch for environments where submitit plugin discovery may raise
    KeyError when metadata.entry_points() does not contain 'submitit' group.
    """
    try:
        from importlib import metadata
        import submitit.core.plugins as submitit_plugins
    except Exception:
        return

    def _safe_iter_submitit_entrypoints():
        eps = metadata.entry_points()
        if hasattr(eps, "select"):
            return eps.select(group="submitit")

        try:
            return metadata.entry_points()["submitit"]
        except (TypeError, KeyError):
            pass

        if hasattr(eps, "get"):
            return eps.get("submitit", [])

        return [ep for ep in eps if getattr(ep, "group", None) == "submitit"]

    submitit_plugins._iter_submitit_entrypoints = _safe_iter_submitit_entrypoints

    for cache_fn_name in ["_get_plugins", "get_executors", "get_job_environments"]:
        cache_fn = getattr(submitit_plugins, cache_fn_name, None)
        if cache_fn is not None and hasattr(cache_fn, "cache_clear"):
            cache_fn.cache_clear()


def _patch_submitit_move_file_compat():
    """
    Handle EXDEV cross-device rename failures observed in some filesystem setups
    by falling back to copy+unlink.
    """
    try:
        import submitit.core.utils as submitit_utils
    except Exception:
        return

    original_move = submitit_utils.JobPaths.move_temporary_file
    if getattr(original_move, "_eif_patched", False):
        return

    def _safe_move_temporary_file(self, tmp_path, name, keep_as_symlink=False):
        self.folder.mkdir(parents=True, exist_ok=True)
        src = Path(tmp_path)
        dst = getattr(self, name)
        try:
            src.rename(dst)
        except OSError as e:
            if e.errno not in (errno.EXDEV, 18):
                raise
            shutil.copy2(src, dst)
            src.unlink(missing_ok=True)

        if keep_as_symlink:
            src.unlink(missing_ok=True)
            src.symlink_to(dst)

    _safe_move_temporary_file._eif_patched = True
    submitit_utils.JobPaths.move_temporary_file = _safe_move_temporary_file


def _extract_job_num_from_argv(argv):
    for token in argv:
        if token.startswith("hydra.job.num="):
            return token.split("=", 1)[1]
    return None


def _resolve_job_num(cfg=None):
    job_num = os.environ.get("HYDRA_JOB_NUM")
    if job_num is not None:
        return job_num

    if cfg is not None:
        try:
            hydra_cfg = cfg.get("hydra", None)
            if hydra_cfg is not None:
                job_cfg = hydra_cfg.get("job", None)
                if job_cfg is not None and job_cfg.get("num", None) is not None:
                    return str(job_cfg.get("num"))
        except Exception:
            pass

    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            return str(HydraConfig.get().job.num)
    except Exception:
        pass

    argv_job_num = _extract_job_num_from_argv(sys.argv[1:])
    if argv_job_num is not None:
        return argv_job_num

    return "0"


def _bind_single_gpu(cfg=None, num_gpus=NUM_GPUS, force=False):
    if not force and os.environ.get("CUDA_VISIBLE_DEVICES"):
        return

    job_num = _resolve_job_num(cfg)
    gpu_id = int(job_num) % num_gpus

    os.environ["HYDRA_JOB_NUM"] = str(job_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU-BIND] HYDRA_JOB_NUM={job_num}, CUDA_VISIBLE_DEVICES={gpu_id}")


def _ensure_optional_defaults(args):
    if not hasattr(args, "fig_title"):
        args.fig_title = "none"
    if not hasattr(args, "json_config"):
        args.json_config = "none"
    if not hasattr(args, "removal_candidate_sampler"):
        args.removal_candidate_sampler = "uniform"
    if not hasattr(args, "removal_neighbor_dist"):
        args.removal_neighbor_dist = 1
    if not hasattr(args, "influence_calculation_mode"):
        args.influence_calculation_mode = "both"
    if not hasattr(args, "influence_mode"):
        args.influence_mode = "none"
    if not hasattr(args, "experiment_name"):
        args.experiment_name = "none"
    if not hasattr(args, "ratio_group_elem"):
        args.ratio_group_elem = None
    if not hasattr(args, "num_of_clusters"):
        args.num_of_clusters = 3
    if not hasattr(args, "edges_per_cluster"):
        args.edges_per_cluster = -1
    if not hasattr(args, "removal_cluster_dist"):
        args.removal_cluster_dist = 1
    if not hasattr(args, "cluster_ratio_percent"):
        args.cluster_ratio_percent = 10
    if not hasattr(args, "cluster_partition_strategy"):
        args.cluster_partition_strategy = "contiguous"
    if not hasattr(args, "intra_cluster_dist"):
        args.intra_cluster_dist = 1
    if not hasattr(args, "inter_cluster_dist"):
        args.inter_cluster_dist = 1
    if not hasattr(args, "metric_mode"):
        args.metric_mode = "global"
    if not hasattr(args, "groupwise_num_groups_init"):
        args.groupwise_num_groups_init = None
    if not hasattr(args, "groupwise_alpha_repr"):
        args.groupwise_alpha_repr = 1.0
    if not hasattr(args, "groupwise_beta_proxy"):
        args.groupwise_beta_proxy = 1.0
    if not hasattr(args, "groupwise_probe_dim"):
        args.groupwise_probe_dim = 8
    if not hasattr(args, "groupwise_probe_seed"):
        args.groupwise_probe_seed = 0
    if not hasattr(args, "groupwise_normalize_proxies"):
        args.groupwise_normalize_proxies = 1
    if not hasattr(args, "groupwise_damping"):
        args.groupwise_damping = 1e-3
    if not hasattr(args, "groupwise_split_threshold"):
        args.groupwise_split_threshold = 1e-6
    if not hasattr(args, "groupwise_merge_threshold"):
        args.groupwise_merge_threshold = 1e-6
    if not hasattr(args, "groupwise_move_threshold"):
        args.groupwise_move_threshold = 1e-6
    if not hasattr(args, "groupwise_radius_update_rho"):
        args.groupwise_radius_update_rho = 1.0
    if not hasattr(args, "groupwise_max_outer_iters"):
        args.groupwise_max_outer_iters = 50
    if not hasattr(args, "groupwise_chain_method"):
        args.groupwise_chain_method = "nearest_neighbor"
    if not hasattr(args, "groupwise_metric_energy_mode"):
        args.groupwise_metric_energy_mode = "scalar_proxy"
    if not hasattr(args, "groupwise_merge_topk"):
        args.groupwise_merge_topk = 10
    if not hasattr(args, "groupwise_max_reassign_candidates_per_group"):
        args.groupwise_max_reassign_candidates_per_group = 5
    if not hasattr(args, "groupwise_reassign_target_topk"):
        args.groupwise_reassign_target_topk = 2
    if not hasattr(args, "groupwise_min_group_size"):
        args.groupwise_min_group_size = 2
    if not hasattr(args, "seed"):
        args.seed = DEFAULT_SEED

    args.influence_calculation_mode = _normalize_influence_mode(args.influence_calculation_mode)
    args.influence_mode = _normalize_influence_mode(args.influence_mode)

    return args


def _load_json_config_if_needed(args):
    if args.json_config == "none":
        return args

    with open(args.json_config, "r") as f:
        loaded = json.load(f)

    merged = vars(args).copy()
    merged.update(loaded)
    merged["json_config"] = args.json_config

    args = SimpleNamespace(**merged)
    return _ensure_optional_defaults(args)


def _count_unique_undirected_edges(data, torch):
    edges = data.edge_index.T
    sorted_edges = torch.sort(edges, dim=1)[0]
    unique_edges = torch.unique(sorted_edges, dim=0)
    return int(unique_edges.shape[0])


def _apply_experiment_runtime_overrides(args, data, torch):
    experiment_name = getattr(args, "experiment_name", "none")
    total_edges = _count_unique_undirected_edges(data, torch)

    if experiment_name == "large_drop_influence":
        if args.ratio_group_elem is None:
            raise ValueError("experiment.ratio_group_elem is required for large_drop_influence.")

        ratio = float(args.ratio_group_elem)
        # Requested conversion: k = max(1, round(total_edges * ratio/100)).
        k = max(1, round(total_edges * (ratio / 100.0)))
        args.num_group_elem = int(k)
        print(
            f"[EXPERIMENT] large_drop_influence: total_edges={total_edges}, "
            f"ratio={ratio}%, num_group_elem(k)={args.num_group_elem}"
        )

    elif experiment_name == "non_neighbor_edges":
        ratio = float(args.cluster_ratio_percent)
        k = max(1, round(total_edges * (ratio / 100.0)))
        args.num_group_elem = int(k)
        print(
            f"[EXPERIMENT] non_neighbor_edges: total_edges={total_edges}, "
            f"ratio={ratio}%, num_group_elem(k)={args.num_group_elem}, "
            f"sampler={args.removal_candidate_sampler}, removal_neighbor_dist={args.removal_neighbor_dist}"
        )

    elif experiment_name == "clusters":
        if args.num_of_clusters <= 0:
            raise ValueError("experiment.num_of_clusters must be positive.")
        if args.intra_cluster_dist <= 0:
            raise ValueError("experiment.intra_cluster_dist must be positive.")
        if args.inter_cluster_dist <= 0:
            raise ValueError("experiment.inter_cluster_dist must be positive.")

        target_total_edges_to_remove = round(total_edges * (float(args.cluster_ratio_percent) / 100.0))

        # Requested default distribution rule:
        # edges_per_cluster = max(1, floor(target_total_edges_to_remove / num_of_clusters))
        # remainder is dropped by default.
        if args.edges_per_cluster is None or int(args.edges_per_cluster) <= 0:
            args.edges_per_cluster = max(1, math.floor(target_total_edges_to_remove / args.num_of_clusters))

        args.edges_per_cluster = int(args.edges_per_cluster)
        args.num_group_elem = int(args.num_of_clusters * args.edges_per_cluster)
        remainder = target_total_edges_to_remove - args.num_group_elem

        print(
            f"[EXPERIMENT] clusters: total_edges={total_edges}, target_total={target_total_edges_to_remove}, "
            f"num_of_clusters={args.num_of_clusters}, edges_per_cluster={args.edges_per_cluster}, "
            f"num_group_elem={args.num_group_elem}, dropped_remainder={remainder}, "
            f"intra_cluster_dist={args.intra_cluster_dist}, inter_cluster_dist={args.inter_cluster_dist}"
        )

    if args.influence_mode != "none":
        args.influence_calculation_mode = _normalize_influence_mode(args.influence_mode)

    # LiSSA dominates runtime for grouped-edge experiments; cap default iterations
    # unless the user explicitly sets a lower value.
    if args.hessian_type == "GNH" and experiment_name in {"non_neighbor_edges", "clusters"}:
        raw_cap = os.environ.get("EIF_MAX_LISSA_ITER_GROUPED", "5000")
        try:
            lissa_cap = max(1, int(raw_cap))
        except ValueError:
            lissa_cap = 5000
        if int(args.lissa_iter) > lissa_cap:
            original_lissa_iter = int(args.lissa_iter)
            args.lissa_iter = lissa_cap
            print(
                f"[LiSSA] Auto-adjust lissa_iter for {experiment_name}: "
                f"{original_lissa_iter} -> {int(args.lissa_iter)} (hessian_type={args.hessian_type})"
            )

    return args


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora_public")
    parser.add_argument("--model", type=str, default="GCN", choices=["SGC", "GCN", "GAT", "ChebNet"])
    parser.add_argument("--hessian_type", type=str, default="GNH", choices=["hessian", "GNH"])
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--damp", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--lissa_iter", type=int, default=10000)
    parser.add_argument(
        "--eval_metric",
        type=str,
        default="mean_validation_loss",
        choices=["dirichlet_energy", "feature_ablation", "mean_validation_loss"],
    )
    parser.add_argument("--linear", type=int, default=0)
    parser.add_argument("--bias", type=int, default=0)
    parser.add_argument("--pbrf_epochs", type=int, default=1000)
    parser.add_argument("--pbrf_weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--element_type",
        type=str,
        default="edge_edit",
        choices=["edge_removal", "edge_insertion", "edge_edit"],
    )
    parser.add_argument("--num_insertion_candidates", type=int, default=50)
    parser.add_argument("--num_removal_candidates", type=int, default=50)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--check_runtime", type=int, default=0)
    parser.add_argument("--json_config", type=str, default="none")
    parser.add_argument("--fig_title", type=str, default="none")
    parser.add_argument("--num_group_elem", type=int, default=1)
    parser.add_argument(
        "--removal_candidate_sampler",
        type=str,
        default="uniform",
        choices=["uniform", "group_non_neighbor", "group_neighbor"],
    )
    parser.add_argument("--removal_neighbor_dist", type=int, default=1)
    parser.add_argument(
        "--influence_calculation_mode",
        type=str,
        default="both",
        choices=["calculate_influence", "clusterwise_step_by_step", "both", "fixed_theta"],
    )

    # Experiment-level controls for Hydra-compatible behavior in argparse mode.
    parser.add_argument("--experiment_name", type=str, default="none")
    parser.add_argument("--ratio_group_elem", type=int, default=None)
    parser.add_argument("--num_of_clusters", type=int, default=3)
    parser.add_argument("--edges_per_cluster", type=int, default=-1)
    parser.add_argument("--removal_cluster_dist", type=int, default=1)
    parser.add_argument("--cluster_ratio_percent", type=int, default=10)
    parser.add_argument("--intra_cluster_dist", type=int, default=1)
    parser.add_argument("--inter_cluster_dist", type=int, default=1)
    parser.add_argument(
        "--cluster_partition_strategy",
        type=str,
        default="contiguous",
        choices=["contiguous", "round_robin"],
    )
    parser.add_argument("--metric_mode", type=str, default="global", choices=["global", "groupwise"])
    parser.add_argument("--groupwise_num_groups_init", type=int, default=None)
    parser.add_argument("--groupwise_alpha_repr", type=float, default=1.0)
    parser.add_argument("--groupwise_beta_proxy", type=float, default=1.0)
    parser.add_argument("--groupwise_probe_dim", type=int, default=8)
    parser.add_argument("--groupwise_probe_seed", type=int, default=0)
    parser.add_argument("--groupwise_normalize_proxies", type=int, default=1)
    parser.add_argument("--groupwise_damping", type=float, default=1e-3)
    parser.add_argument("--groupwise_split_threshold", type=float, default=1e-6)
    parser.add_argument("--groupwise_merge_threshold", type=float, default=1e-6)
    parser.add_argument("--groupwise_move_threshold", type=float, default=1e-6)
    parser.add_argument("--groupwise_radius_update_rho", type=float, default=1.0)
    parser.add_argument("--groupwise_max_outer_iters", type=int, default=50)
    parser.add_argument(
        "--groupwise_chain_method",
        type=str,
        default="nearest_neighbor",
        choices=["nearest_neighbor", "center_distance"],
    )
    parser.add_argument(
        "--groupwise_metric_energy_mode",
        type=str,
        default="scalar_proxy",
        choices=["scalar_proxy"],
    )
    parser.add_argument("--groupwise_merge_topk", type=int, default=10)
    parser.add_argument("--groupwise_max_reassign_candidates_per_group", type=int, default=5)
    parser.add_argument("--groupwise_reassign_target_topk", type=int, default=2)
    parser.add_argument("--groupwise_min_group_size", type=int, default=2)
    parser.add_argument(
        "--influence_mode",
        type=str,
        default="none",
        choices=["calculate_influence", "clusterwise_step_by_step", "both", "fixed_theta", "none"],
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    return parser


def _args_from_hydra_cfg(cfg):
    from omegaconf import OmegaConf

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    model_cfg = cfg_dict.get("model", {})
    dataset_cfg = cfg_dict.get("dataset", {})
    exp_cfg = cfg_dict.get("experiment", {})
    train_cfg = cfg_dict.get("train", {})
    influence_cfg = cfg_dict.get("influence", {})
    run_cfg = cfg_dict.get("run", {})

    args = SimpleNamespace(
        dataset=dataset_cfg.get("name", "cora_public"),
        model=model_cfg.get("name", "GCN"),
        hessian_type=influence_cfg.get("hessian_type", "GNH"),
        num_layers=int(model_cfg.get("num_layers", 2)),
        lr=float(train_cfg.get("lr", 0.01)),
        hidden_dim=int(model_cfg.get("hidden_dim", 16)),
        epochs=int(train_cfg.get("epochs", 1000)),
        weight_decay=float(train_cfg.get("weight_decay", 0.001)),
        damp=float(influence_cfg.get("damp", 0.1)),
        scale=float(influence_cfg.get("scale", 1.0)),
        lissa_iter=int(influence_cfg.get("lissa_iter", 10000)),
        eval_metric=influence_cfg.get("eval_metric", "mean_validation_loss"),
        linear=bool(model_cfg.get("linear", False)),
        bias=bool(model_cfg.get("bias", False)),
        pbrf_epochs=int(train_cfg.get("pbrf_epochs", 1000)),
        pbrf_weight_decay=float(train_cfg.get("pbrf_weight_decay", 0.0)),
        element_type=run_cfg.get("element_type", "edge_removal"),
        num_insertion_candidates=int(run_cfg.get("num_insertion_candidates", 50)),
        num_removal_candidates=int(run_cfg.get("num_removal_candidates", 50)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        check_runtime=int(influence_cfg.get("check_runtime", 0)),
        json_config=run_cfg.get("json_config", "none"),
        fig_title=run_cfg.get("fig_title", "none"),
        num_group_elem=int(exp_cfg.get("num_group_elem", run_cfg.get("num_group_elem", 1))),
        removal_candidate_sampler=exp_cfg.get("removal_candidate_sampler", "uniform"),
        removal_neighbor_dist=int(exp_cfg.get("removal_neighbor_dist", 1)),
        influence_calculation_mode=exp_cfg.get("influence_mode", "calculate_influence"),
        experiment_name=exp_cfg.get("name", "none"),
        ratio_group_elem=exp_cfg.get("ratio_group_elem", None),
        num_of_clusters=int(exp_cfg.get("num_of_clusters", 3)),
        edges_per_cluster=exp_cfg.get("edges_per_cluster", -1),
        removal_cluster_dist=int(exp_cfg.get("removal_cluster_dist", 1)),
        cluster_ratio_percent=int(exp_cfg.get("cluster_ratio_percent", 10)),
        intra_cluster_dist=int(exp_cfg.get("intra_cluster_dist", 1)),
        inter_cluster_dist=int(exp_cfg.get("inter_cluster_dist", 1)),
        cluster_partition_strategy=exp_cfg.get("cluster_partition_strategy", "contiguous"),
        metric_mode=exp_cfg.get("metric_mode", "global"),
        groupwise_num_groups_init=exp_cfg.get("groupwise_num_groups_init", None),
        groupwise_alpha_repr=float(exp_cfg.get("groupwise_alpha_repr", 1.0)),
        groupwise_beta_proxy=float(exp_cfg.get("groupwise_beta_proxy", 1.0)),
        groupwise_probe_dim=int(exp_cfg.get("groupwise_probe_dim", 8)),
        groupwise_probe_seed=int(exp_cfg.get("groupwise_probe_seed", 0)),
        groupwise_normalize_proxies=int(exp_cfg.get("groupwise_normalize_proxies", 1)),
        groupwise_damping=float(exp_cfg.get("groupwise_damping", 1e-3)),
        groupwise_split_threshold=float(exp_cfg.get("groupwise_split_threshold", 1e-6)),
        groupwise_merge_threshold=float(exp_cfg.get("groupwise_merge_threshold", 1e-6)),
        groupwise_move_threshold=float(exp_cfg.get("groupwise_move_threshold", 1e-6)),
        groupwise_radius_update_rho=float(exp_cfg.get("groupwise_radius_update_rho", 1.0)),
        groupwise_max_outer_iters=int(exp_cfg.get("groupwise_max_outer_iters", 50)),
        groupwise_chain_method=exp_cfg.get("groupwise_chain_method", "nearest_neighbor"),
        groupwise_metric_energy_mode=exp_cfg.get("groupwise_metric_energy_mode", "scalar_proxy"),
        groupwise_merge_topk=int(exp_cfg.get("groupwise_merge_topk", 10)),
        groupwise_max_reassign_candidates_per_group=int(
            exp_cfg.get("groupwise_max_reassign_candidates_per_group", 5)
        ),
        groupwise_reassign_target_topk=int(exp_cfg.get("groupwise_reassign_target_topk", 2)),
        groupwise_min_group_size=int(exp_cfg.get("groupwise_min_group_size", 2)),
        influence_mode=exp_cfg.get("influence_mode", "calculate_influence"),
        seed=int(cfg_dict.get("seed", DEFAULT_SEED)),
    )

    if args.edges_per_cluster is None:
        args.edges_per_cluster = -1
    else:
        args.edges_per_cluster = int(args.edges_per_cluster)

    return _ensure_optional_defaults(args)


def _write_done_markers(dirs):
    try:
        done_path = Path(dirs["result"]) / "DONE"
        done_path.touch()
    except Exception:
        pass

    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            hydra_output_dir = HydraConfig.get().runtime.output_dir
            Path(hydra_output_dir, "DONE").touch()
    except Exception:
        pass


def _is_edge_group_constraint_error(exc):
    return isinstance(exc, RuntimeError) and EDGE_GROUP_CONSTRAINT_ERROR_TEXT in str(exc)


def _write_constraint_unsat_marker(reason):
    marker_written = False
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
            hydra_output_dir.mkdir(parents=True, exist_ok=True)
            (hydra_output_dir / CONSTRAINT_UNSAT_MARKER).touch()
            (hydra_output_dir / f"{CONSTRAINT_UNSAT_MARKER}.txt").write_text(f"{reason}\n", encoding="utf-8")
            marker_written = True
    except Exception:
        pass

    if marker_written:
        print(f"[EXPERIMENT-STATUS] {CONSTRAINT_UNSAT_MARKER}: {reason}")
    else:
        print(f"[EXPERIMENT-STATUS] constraint-unsat marker write skipped: {reason}")


def run_experiment(args):
    import time

    import torch
    import torch.optim as optim

    from calculate_influence import GraphInfluenceModule, calculate_loo, get_pbrf
    from src import DataLoader, GNN, make_metric_fns, train
    from src.graph_utils import find_k_hop_neighborhoods
    from src.utils import (
        calculate_grouped_influence,
        get_edge_insertion_candidates,
        get_eval_node_idxs,
        get_grouped_edge_removal_candidates,
        is_within_2std,
        make_dirs,
        plot_influence_loss,
        random_planetoid_splits,
        rename_result_dir,
        save_candidate_result_tables,
        save_config,
        set_seed,
    )

    args = _ensure_optional_defaults(args)
    args = _load_json_config_if_needed(args)

    args.linear = bool(args.linear)
    args.bias = bool(args.bias)
    print(args)

    if args.hessian_type == "hessian":
        print("Warning. args.damp should be the same with args.weight_decay when args.hessian_type is hessian.")
        print(f"Original damp: {args.damp}, adjusted damp: {args.weight_decay}")
        args.damp = args.weight_decay

    dataset = DataLoader(args.dataset, root="datasets")
    args.num_classes = dataset.num_classes
    data = dataset[0]
    data.edge_weight = torch.ones((data.edge_index.shape[1],))

    seed = int(args.seed)

    eval_node_idxs = get_eval_node_idxs(data, args.eval_metric, seed)

    if "public" not in args.dataset:
        percls_trn = int(round(0.6 * len(data.y) / dataset.num_classes))
        val_lb = int(round(0.2 * len(data.y)))
        data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb, seed)

    args = _apply_experiment_runtime_overrides(args, data, torch)

    dirs = make_dirs(args)
    save_config(args, osp.join(dirs["result"], "config.json"), dirs)

    vanilla_dir = dirs["vanilla"]
    vanilla_path = osp.join(vanilla_dir, f"{seed}.pth")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.edge_weight = data.edge_weight.to(device)
    data.y = data.y.to(device)

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
    if osp.isfile(vanilla_path):
        model_state_dict = torch.load(vanilla_path, map_location=device, weights_only=True)
        model.load_state_dict(model_state_dict)
        model = model.to(device)
    else:
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(1, args.epochs + 1):
            train_loss, val_loss, test_loss, train_acc, val_acc, test_acc = train(data, model, optimizer, device)
            if epoch % 100 == 0:
                print("-----------------------------------------------")
                print(
                    f"Epoch: {epoch}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, "
                    f"test_loss: {test_loss:.4f}"
                )
                print(
                    f"Train acc: {train_acc*100:.2f}%, val acc: {val_acc*100:.2f}%, "
                    f"test_acc: {test_acc*100:.2f}%"
                )
                print("-----------------------------------------------")
        torch.save({k: v.clone().detach() for k, v in model.state_dict().items()}, vanilla_path)

    save_name = "influence_vs_pbrf"
    save_name_parameter_shift = "parameter_shift_effect"
    save_name_ps_mp = "parameter_shift_vs_message_propagation"

    set_seed(seed)
    if args.eval_metric == "feature_ablation":
        exact_k_hop_neighbors = find_k_hop_neighborhoods(data, args.num_layers)
    else:
        exact_k_hop_neighbors = None

    metric_fns = make_metric_fns(eval_node_idxs, exact_k_hop_neighbors, data.edge_index)
    metric_fn = metric_fns[args.eval_metric]
    # Backward-compatible fallback:
    # older calculate_influence code paths may reference module-level `metric_fn`.
    import calculate_influence as _calculate_influence_module

    _calculate_influence_module.metric_fn = metric_fn

    if args.element_type in ["edge_removal", "edge_edit"]:
        set_seed(seed)
        removal_candidates = get_grouped_edge_removal_candidates(data, args)
        candidates = removal_candidates

    if args.element_type in ["edge_insertion", "edge_edit"]:
        set_seed(seed)
        num_candidates = args.num_insertion_candidates * args.num_group_elem
        candidates = get_edge_insertion_candidates(data, num_candidates * 2)[:num_candidates]
        candidates = candidates.view(args.num_insertion_candidates, args.num_group_elem, 2)
        insertion_candidates = candidates

    print(f"Calculate the Influence of {args.element_type}...")
    start_time = time.time()
    table_exports = []
    use_groupwise_summary = str(getattr(args, "metric_mode", "global")).lower() == "groupwise"
    if args.element_type == "edge_edit":
        removal_influence_results = calculate_grouped_influence(
            model=model,
            graph=data,
            args=args,
            eval_node_idxs=eval_node_idxs,
            metric_fn=metric_fn,
            candidates=removal_candidates,
            influence_type="edge_removal",
            influence_module_cls=GraphInfluenceModule,
        )
        insertion_influence_results = calculate_grouped_influence(
            model=model,
            graph=data,
            args=args,
            eval_node_idxs=eval_node_idxs,
            metric_fn=metric_fn,
            candidates=insertion_candidates,
            influence_type="edge_insertion",
            influence_module_cls=GraphInfluenceModule,
        )

        r_calculate = removal_influence_results["calculate_influence"]
        i_calculate = insertion_influence_results["calculate_influence"]
        r_cluster_fixed = removal_influence_results["clusterwise_fixed_theta"]
        i_cluster_fixed = insertion_influence_results["clusterwise_fixed_theta"]
        r_cluster_step = removal_influence_results["clusterwise_step_by_step"]
        i_cluster_step = insertion_influence_results["clusterwise_step_by_step"]
        r_summary = r_cluster_fixed if use_groupwise_summary and r_cluster_fixed is not None else r_calculate
        i_summary = i_cluster_fixed if use_groupwise_summary and i_cluster_fixed is not None else i_calculate
        r_total_inf = r_summary["total_inf"]
        r_parameter_shift_inf = r_summary["retrain_inf"]
        r_message_propagation_inf = r_summary["perturb_inf"]
        i_total_inf = i_summary["total_inf"]
        i_parameter_shift_inf = i_summary["retrain_inf"]
        i_message_propagation_inf = i_summary["perturb_inf"]

        total_inf = torch.cat((r_total_inf, i_total_inf), dim=0)
        parameter_shift_inf = torch.cat((r_parameter_shift_inf, i_parameter_shift_inf), dim=0)
        message_propagation_inf = torch.cat((r_message_propagation_inf, i_message_propagation_inf), dim=0)
        table_exports = [
            {
                "file_stem": "removal_candidate_results",
                "candidates": removal_candidates,
                "influence_results": removal_influence_results,
            },
            {
                "file_stem": "insertion_candidate_results",
                "candidates": insertion_candidates,
                "influence_results": insertion_influence_results,
            },
        ]
    else:
        influence_results = calculate_grouped_influence(
            model=model,
            graph=data,
            args=args,
            eval_node_idxs=eval_node_idxs,
            metric_fn=metric_fn,
            candidates=candidates,
            influence_type=args.element_type,
            influence_module_cls=GraphInfluenceModule,
        )
        calculate_inf = influence_results["calculate_influence"]
        cluster_fixed = influence_results["clusterwise_fixed_theta"]
        cluster_step = influence_results["clusterwise_step_by_step"]
        summary_result = cluster_fixed if use_groupwise_summary and cluster_fixed is not None else calculate_inf
        total_inf = summary_result["total_inf"]
        parameter_shift_inf = summary_result["retrain_inf"]
        message_propagation_inf = summary_result["perturb_inf"]
        table_exports = [
            {
                "file_stem": "candidate_results",
                "candidates": candidates,
                "influence_results": influence_results,
            }
        ]
    print(f"Consumed time: {time.time()-start_time:.2f}s")

    loo_values = None
    if args.hessian_type == "hessian":
        loo = calculate_loo(model, data, candidates, args, seed, dirs["loo_model"], metric_fn, args.element_type)
        loo_values = loo

        parameter_shift_vec = parameter_shift_inf.detach().reshape(-1).to(torch.float32).cpu()
        loo_vec = torch.as_tensor(loo, dtype=torch.float32).reshape(-1).cpu()
        common_n = min(parameter_shift_vec.numel(), loo_vec.numel())
        if common_n > 0:
            parameter_shift_vec = parameter_shift_vec[:common_n]
            loo_vec = loo_vec[:common_n]
            mask = torch.logical_and(is_within_2std(parameter_shift_vec), is_within_2std(loo_vec))
            plot_influence_loss(
                parameter_shift_vec[mask],
                loo_vec[mask],
                dirs["result"],
                save_name_parameter_shift,
                args,
                title=dirs["fig_title"],
            )

    elif args.hessian_type == "GNH":
        if args.element_type == "edge_edit":
            r_total_pbrf, r_parameter_shift_pbrf, r_message_propagation_pbrf = get_pbrf(
                args, model, data, removal_candidates, seed, dirs, "edge_removal", metric_fn=metric_fn
            )
            i_total_pbrf, i_parameter_shift_pbrf, i_message_propagation_pbrf = get_pbrf(
                args, model, data, insertion_candidates, seed, dirs, "edge_insertion", metric_fn=metric_fn
            )

            total_pbrf = r_total_pbrf + i_total_pbrf
            parameter_shift_pbrf = r_parameter_shift_pbrf + i_parameter_shift_pbrf
            message_propagation_pbrf = r_message_propagation_pbrf + i_message_propagation_pbrf
            r_size = len(r_total_pbrf)

            table_exports[0]["total_pbrf"] = r_total_pbrf
            table_exports[0]["parameter_shift_pbrf"] = r_parameter_shift_pbrf
            table_exports[0]["message_propagation_pbrf"] = r_message_propagation_pbrf
            table_exports[1]["total_pbrf"] = i_total_pbrf
            table_exports[1]["parameter_shift_pbrf"] = i_parameter_shift_pbrf
            table_exports[1]["message_propagation_pbrf"] = i_message_propagation_pbrf
        else:
            total_pbrf, parameter_shift_pbrf, message_propagation_pbrf = get_pbrf(
                args, model, data, candidates, seed, dirs, args.element_type, metric_fn=metric_fn
            )
            r_size = None
            table_exports[0]["total_pbrf"] = total_pbrf
            table_exports[0]["parameter_shift_pbrf"] = parameter_shift_pbrf
            table_exports[0]["message_propagation_pbrf"] = message_propagation_pbrf

        rename_result_dir(
            args,
            parameter_shift_inf,
            parameter_shift_pbrf,
            message_propagation_inf,
            message_propagation_pbrf,
            dirs,
        )
        k = 2
        total_inf_vec = total_inf.detach().reshape(-1).to(torch.float32).cpu()
        total_pbrf_vec = torch.as_tensor(total_pbrf, dtype=torch.float32).reshape(-1).cpu()
        parameter_shift_vec = parameter_shift_inf.detach().reshape(-1).to(torch.float32).cpu()
        message_propagation_vec = message_propagation_inf.detach().reshape(-1).to(torch.float32).cpu()
        common_n = min(
            total_inf_vec.numel(),
            total_pbrf_vec.numel(),
            parameter_shift_vec.numel(),
            message_propagation_vec.numel(),
        )
        if common_n > 0:
            total_inf_vec = total_inf_vec[:common_n]
            total_pbrf_vec = total_pbrf_vec[:common_n]
            parameter_shift_vec = parameter_shift_vec[:common_n]
            message_propagation_vec = message_propagation_vec[:common_n]
            mask = torch.logical_and(is_within_2std(total_inf_vec, k), is_within_2std(total_pbrf_vec, k))
            plot_influence_loss(
                total_inf_vec[mask],
                total_pbrf_vec[mask],
                dirs["result"],
                save_name,
                args,
                title=dirs["fig_title"],
                mask=mask,
                r_size=r_size,
            )
            plot_influence_loss(
                parameter_shift_vec[mask],
                message_propagation_vec[mask],
                dirs["result"],
                save_name_ps_mp,
                args,
                xlabel="Parameter Shift Effect",
                ylabel="Message Propagation Effect",
                title=dirs["fig_title"],
                mask=mask,
                r_size=r_size,
            )

    if loo_values is not None and len(table_exports) == 1:
        table_exports[0]["loo"] = loo_values

    for table_export in table_exports:
        save_candidate_result_tables(
            result_dir=dirs["result"],
            file_stem=table_export["file_stem"],
            candidates=table_export["candidates"],
            influence_results=table_export["influence_results"],
            total_pbrf=table_export.get("total_pbrf", None),
            parameter_shift_pbrf=table_export.get("parameter_shift_pbrf", None),
            message_propagation_pbrf=table_export.get("message_propagation_pbrf", None),
            loo=table_export.get("loo", None),
        )

    _write_done_markers(dirs)


def _should_use_hydra(argv):
    hydra_flags = {"-m", "--multirun", "--config-name", "--config-path", "--hydra-help", "--cfg"}
    if any(token in hydra_flags for token in argv):
        return True

    for idx, token in enumerate(argv):
        if token.startswith("+") or token.startswith("~"):
            return True
        if "=" in token and not token.startswith("--"):
            prev_token = argv[idx - 1] if idx > 0 else ""
            if prev_token.startswith("--"):
                continue
            return True

    return False


def _run_with_hydra():
    _ensure_repo_on_pythonpath()
    _patch_submitit_entrypoint_compat()
    _patch_submitit_move_file_compat()

    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="conf", config_name="config", version_base=None)
    def _hydra_entry(cfg: DictConfig):
        respect_cuda_visible_devices = os.environ.get("EIF_RESPECT_CUDA_VISIBLE_DEVICES", "0") == "1"
        _bind_single_gpu(cfg=cfg, force=not respect_cuda_visible_devices)
        args = _args_from_hydra_cfg(cfg)
        try:
            run_experiment(args)
        except RuntimeError as exc:
            if _is_edge_group_constraint_error(exc):
                _write_constraint_unsat_marker(str(exc))
                return
            raise

    _hydra_entry()


def main():
    _ensure_repo_on_pythonpath()
    if _should_use_hydra(sys.argv[1:]):
        _run_with_hydra()
        return

    parser = _create_parser()
    args = parser.parse_args()
    _bind_single_gpu(force=False)
    run_experiment(args)


if __name__ == "__main__":
    main()
