from types import SimpleNamespace

from .objective import compute_global_baseline_objective, compute_partition_objective
from .refine import (
    candidate_merge_pairs,
    candidate_reassignments,
    initialize_groups,
    try_merge,
    try_reassignment,
    try_split,
)


def refine_groups(groups, h, Q, config):
    current_stats = compute_partition_objective(
        groups,
        h=h,
        Q=Q,
        chain_method=config.chain_method,
        damping=config.damping,
        radius_update_rho=config.radius_update_rho,
        metric_energy_mode=config.metric_energy_mode,
    )
    history = [_history_entry(iteration=0, stats=current_stats, accepted={"split": 0, "merge": 0, "move": 0})]
    _print_history_entry(history[-1])

    for outer_iter in range(1, int(config.max_outer_iters) + 1):
        accepted_counts = {"split": 0, "merge": 0, "move": 0}

        group_scores = sorted(
            (
                (entry["objective_contrib"], entry["group_id"])
                for entry in current_stats["per_group"]
                if entry["size"] >= 2
            ),
            reverse=True,
        )
        for _, group_id in group_scores:
            accepted, new_groups, diag = try_split(
                current_stats["groups"],
                group_id=group_id,
                h=h,
                Q=Q,
                config=config,
                current_stats=current_stats,
            )
            if accepted:
                current_stats = diag["candidate_stats"]
                accepted_counts["split"] += 1
                break

        merge_pairs = candidate_merge_pairs(
            current_stats["groups"],
            h=h,
            Q=Q,
            alpha=config.alpha_repr,
            beta=config.beta_proxy,
            topk=config.merge_topk,
        )
        for g1, g2 in merge_pairs:
            accepted, new_groups, diag = try_merge(
                current_stats["groups"],
                g1=g1,
                g2=g2,
                h=h,
                Q=Q,
                config=config,
                current_stats=current_stats,
            )
            if accepted:
                current_stats = diag["candidate_stats"]
                accepted_counts["merge"] += 1
                break

        moves = candidate_reassignments(
            current_stats["groups"],
            h=h,
            Q=Q,
            alpha=config.alpha_repr,
            beta=config.beta_proxy,
            max_candidates_per_group=config.max_reassign_candidates_per_group,
            target_topk=config.reassign_target_topk,
            min_group_size=config.min_group_size,
        )
        for move in moves:
            accepted, new_groups, diag = try_reassignment(
                current_stats["groups"],
                move=move,
                h=h,
                Q=Q,
                config=config,
                current_stats=current_stats,
            )
            if accepted:
                current_stats = diag["candidate_stats"]
                accepted_counts["move"] += 1
                break

        history.append(_history_entry(iteration=outer_iter, stats=current_stats, accepted=accepted_counts))
        _print_history_entry(history[-1])
        if sum(accepted_counts.values()) == 0:
            break

    return current_stats["groups"], history, current_stats


def search_groupwise_partition(h, Q, config):
    global_stats = compute_global_baseline_objective(h, Q, config)
    init_groups = initialize_groups(
        h=h,
        Q=Q,
        num_groups=config.num_groups_init,
        alpha=config.alpha_repr,
        beta=config.beta_proxy,
        seed=config.seed,
        min_group_size=config.min_group_size,
    )
    init_stats = compute_partition_objective(
        init_groups,
        h=h,
        Q=Q,
        chain_method=config.chain_method,
        damping=config.damping,
        radius_update_rho=config.radius_update_rho,
        metric_energy_mode=config.metric_energy_mode,
    )

    start_groups = global_stats["groups"]
    start_stats = global_stats
    start_label = "global"
    if init_stats["J"] + 1e-12 < global_stats["J"]:
        start_groups = init_stats["groups"]
        start_stats = init_stats
        start_label = "initialized"

    print(
        f"[groupwise-search] start={start_label} J_start={start_stats['J']:.6f} "
        f"J_global={global_stats['J']:.6f} groups={len(start_groups)}"
    )

    final_groups, history, final_stats = refine_groups(start_groups, h=h, Q=Q, config=config)
    final_stats = compute_partition_objective(
        final_groups,
        h=h,
        Q=Q,
        chain_method=config.chain_method,
        damping=config.damping,
        radius_update_rho=config.radius_update_rho,
        metric_energy_mode=config.metric_energy_mode,
    )

    return {
        "groups": final_stats["groups"],
        "history": history,
        "final_stats": final_stats,
        "global_baseline": global_stats,
        "initialized_stats": init_stats,
        "start_label": start_label,
        "start_stats": start_stats,
        "objective_ratio_to_global": (
            float(final_stats["J"] / global_stats["J"]) if global_stats["J"] > 0 else 1.0
        ),
    }


def build_groupwise_config(args, default_num_groups=None):
    if default_num_groups is None:
        default_num_groups = getattr(args, "num_of_clusters", 1)
    raw_num_groups_init = getattr(args, "groupwise_num_groups_init", None)
    if raw_num_groups_init is None:
        raw_num_groups_init = default_num_groups

    return SimpleNamespace(
        num_groups_init=int(raw_num_groups_init),
        alpha_repr=float(getattr(args, "groupwise_alpha_repr", 1.0)),
        beta_proxy=float(getattr(args, "groupwise_beta_proxy", 1.0)),
        damping=float(getattr(args, "groupwise_damping", 1e-3)),
        split_threshold=float(getattr(args, "groupwise_split_threshold", 1e-6)),
        merge_threshold=float(getattr(args, "groupwise_merge_threshold", 1e-6)),
        move_threshold=float(getattr(args, "groupwise_move_threshold", 1e-6)),
        radius_update_rho=float(getattr(args, "groupwise_radius_update_rho", 1.0)),
        max_outer_iters=int(getattr(args, "groupwise_max_outer_iters", 50)),
        chain_method=str(getattr(args, "groupwise_chain_method", "nearest_neighbor")),
        metric_energy_mode=str(getattr(args, "groupwise_metric_energy_mode", "scalar_proxy")),
        merge_topk=int(getattr(args, "groupwise_merge_topk", 10)),
        max_reassign_candidates_per_group=int(
            getattr(args, "groupwise_max_reassign_candidates_per_group", 5)
        ),
        reassign_target_topk=int(getattr(args, "groupwise_reassign_target_topk", 2)),
        min_group_size=int(getattr(args, "groupwise_min_group_size", 2)),
        seed=int(getattr(args, "seed", 0)),
    )


def _history_entry(iteration, stats, accepted):
    return {
        "iteration": int(iteration),
        "J": float(stats["J"]),
        "num_groups": len(stats["groups"]),
        "accepted_splits": int(accepted["split"]),
        "accepted_merges": int(accepted["merge"]),
        "accepted_moves": int(accepted["move"]),
        "per_group": [
            {
                "group_id": int(entry["group_id"]),
                "size": int(entry["size"]),
                "tau_g": float(entry["tau_g"]),
                "r_g": float(entry["r_g"]),
                "F_g": float(entry["F_g"]),
                "d_g": float(entry["d_g"]),
                "objective_contrib": float(entry["objective_contrib"]),
            }
            for entry in stats["per_group"]
        ],
    }


def _print_history_entry(entry):
    print(
        f"[groupwise-search] iter={entry['iteration']} J={entry['J']:.6f} "
        f"groups={entry['num_groups']} "
        f"splits={entry['accepted_splits']} merges={entry['accepted_merges']} moves={entry['accepted_moves']}"
    )
