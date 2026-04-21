import torch
import time
import torch.nn as nn
import os.path as osp
import hashlib
import math
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import grad
from tqdm import tqdm
from torch_influence import BaseObjective, LiSSAInfluenceModule
from src import train, train_pbrf, mean_validation_loss, DataLoader, GNN, make_metric_fns
from src.graph_utils import *
from src.utils import *
import argparse

from groupwise_metric.proxies import generate_probe_vectors, normalize_proxy_matrix


def _candidate_edge_checkpoint_name(candidate_edge):
    edge_tensor = candidate_edge.detach().cpu().to(torch.long)

    if edge_tensor.dim() == 1:
        edge_tensor = edge_tensor.view(1, 2)
    elif edge_tensor.dim() != 2 or edge_tensor.shape[1] != 2:
        raise ValueError("candidate_edge must have shape [2] or [num_edges, 2].")

    # Canonicalize undirected edge pairs for deterministic cache keys.
    edge_pairs = [tuple(sorted((int(edge[0]), int(edge[1])))) for edge in edge_tensor]
    edge_pairs.sort()
    payload = ";".join(f"{u}-{v}" for u, v in edge_pairs)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]
    return f"cand_e{len(edge_pairs)}_{digest}.pth"


def _ensure_optional_experiment_defaults(args):
    if not hasattr(args, "experiment_name"):
        args.experiment_name = "none"
    if not hasattr(args, "num_of_clusters"):
        args.num_of_clusters = 3
    if not hasattr(args, "edges_per_cluster"):
        args.edges_per_cluster = -1
    if not hasattr(args, "cluster_ratio_percent"):
        args.cluster_ratio_percent = 10
    if not hasattr(args, "intra_cluster_dist"):
        args.intra_cluster_dist = 1
    if not hasattr(args, "inter_cluster_dist"):
        args.inter_cluster_dist = 1
    if not hasattr(args, "removal_candidate_sampler"):
        args.removal_candidate_sampler = "uniform"
    if not hasattr(args, "removal_neighbor_dist"):
        args.removal_neighbor_dist = 1
    if not hasattr(args, "cluster_candidate_init_only"):
        args.cluster_candidate_init_only = 0
    if not hasattr(args, "cluster_candidate_force_rebuild"):
        args.cluster_candidate_force_rebuild = 0
    if not hasattr(args, "cluster_candidate_cache_root"):
        args.cluster_candidate_cache_root = "candidate_cache"
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

    args.num_of_clusters = int(args.num_of_clusters)
    args.edges_per_cluster = int(args.edges_per_cluster)
    args.cluster_ratio_percent = int(args.cluster_ratio_percent)
    args.intra_cluster_dist = int(args.intra_cluster_dist)
    args.inter_cluster_dist = int(args.inter_cluster_dist)
    args.removal_neighbor_dist = int(args.removal_neighbor_dist)
    args.cluster_candidate_init_only = int(args.cluster_candidate_init_only)
    args.cluster_candidate_force_rebuild = int(args.cluster_candidate_force_rebuild)
    return args


def _count_unique_undirected_edges(data):
    edges = data.edge_index.T
    sorted_edges = torch.sort(edges, dim=1)[0]
    unique_edges = torch.unique(sorted_edges, dim=0)
    return int(unique_edges.shape[0])


def _apply_experiment_runtime_overrides(args, data):
    if getattr(args, "experiment_name", "none") != "clusters":
        return args

    if args.num_of_clusters <= 0:
        raise ValueError("experiment.num_of_clusters must be positive.")
    if args.intra_cluster_dist <= 0:
        raise ValueError("experiment.intra_cluster_dist must be positive.")
    if args.inter_cluster_dist <= 0:
        raise ValueError("experiment.inter_cluster_dist must be positive.")

    total_edges = _count_unique_undirected_edges(data)
    target_total_edges_to_remove = round(total_edges * (float(args.cluster_ratio_percent) / 100.0))
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
    return args


class CrossEntropyObjective(BaseObjective):
    def __init__(self, args):
        self.pbrf_wd = args.pbrf_weight_decay

    def train_outputs(self, model, batch):
        return model(batch)[batch.train_mask]

    def train_loss_on_outputs(self, outputs, batch):
        return F.cross_entropy(outputs, batch.y[batch.train_mask])  # mean reduction required

    def train_regularization(self, params):
        return self.pbrf_wd/2 * torch.square(params.norm())
    
    def train_loss_without_reg(self, model, batch):
        outputs = self.train_outputs(model, batch)
        return self.train_loss_on_outputs(outputs, batch)

    def test_loss(self, model, params, batch):
        val_output = model(batch)[batch.val_mask]
        return F.cross_entropy(val_output, batch.y[batch.val_mask])  # no regularization in test loss
    
    def indiv_train_loss(self, model, params, batch, idx):
        train_output = model(batch)[batch.train_mask]
        train_y = batch.y[batch.train_mask]
        train_loss = F.cross_entropy(train_output[idx], train_y[idx])
        return train_loss + self.train_regularization(params)
    

class GraphInfluenceModule:
    def __init__(self, model, graph, args, eval_metric, num_folds, eval_node_idxs, metric_fn):
        self.model = model
        self.graph = graph
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.inv_hvp = None
        self.nodes_within_km1_hop = None
        self.nodes_within_k_hop = None
        self.validation_splits = None
        self.eval_metric = eval_metric
        self.num_folds = num_folds
        self.metric_fn = metric_fn
        self.node_output_representations = None
        self.curvature_probe_cache = {}
        self.edge_curvature_proxy_cache = {}

        self.eval_node_idxs = eval_node_idxs
        self.exact_k_hop_neighbors = self._load_exact_k_hop_neighbors()
        self.get_validation_splits()
    
    def get_validation_splits(self):
        if self.validation_splits is None:
            num_vals = self.graph.val_mask.sum()
            val_idxs = self.graph.val_mask.nonzero().squeeze()
            num_per_split = int(num_vals/self.num_folds)
            shuffled_val_idxs = val_idxs[torch.randperm(num_vals)]

            validation_splits = []
            for i in range(self.num_folds):
                if i == self.num_folds - 1:
                    validation_splits.append(shuffled_val_idxs)
                else:
                    validation_splits.append(shuffled_val_idxs[:num_per_split])
                    shuffled_val_idxs = shuffled_val_idxs[num_per_split:]
                

            self.validation_splits = validation_splits

        return self.validation_splits
        
    def get_parameter_shifting_influence(self, targets, influence_type, params, candidate_idx=None):
        """
        target: the target to estimate influence
        influence_type: the type of graph element. Choices: {'edge_removal', 'edge_insertion'}
        """
        origin_logit = self.model(self.graph)

        if influence_type == 'edge_removal':
            perturbed_logit = self.get_perturbed_logit(self.model, self.graph, removed_edge=targets)
        elif influence_type == 'edge_insertion':
            perturbed_logit = self.get_perturbed_logit(self.model, self.graph, added_edge=targets)

        train_influenced_nodes = self.get_train_influenced_nodes(targets)
        
        if train_influenced_nodes.numel() == 0:
            return [0 for i in range(self.num_folds)], 0
        else:
            # Memory-efficient equivalent of summing per-node gradients:
            # grad(sum_i loss_i) == sum_i grad(loss_i)
            target_labels = self.graph.y[train_influenced_nodes]
            origin_loss = F.cross_entropy(origin_logit[train_influenced_nodes], target_labels, reduction="sum")
            perturbed_loss = F.cross_entropy(perturbed_logit[train_influenced_nodes], target_labels, reduction="sum")
            origin_grad = grad(origin_loss, params, retain_graph=False)
            perturbed_grad = grad(perturbed_loss, params, retain_graph=False)
            origin_grad = [g.detach() for g in origin_grad]
            perturbed_grad = [g.detach() for g in perturbed_grad]

            k_fold_edge_influence = []
            k_fold_mean_grad_cos_sim = []
            for i in range(self.num_folds):
                edge_influence = 0
                param_cos_sims = []
                for inv_hvp_elem, origin_grad_elem, perturbed_grad_elem in zip(
                    self.inv_hvp[i], origin_grad, perturbed_grad
                ):
                    elem_influence = inv_hvp_elem * (origin_grad_elem - perturbed_grad_elem)

                    grad_cos_sim = F.cosine_similarity(
                        origin_grad_elem.reshape(1, -1),
                        perturbed_grad_elem.reshape(1, -1),
                        dim=1,
                        eps=1e-12,
                    ).item()
                    param_cos_sims.append(grad_cos_sim)

                    edge_influence += elem_influence.sum()
                mean_grad_cos_sim = sum(param_cos_sims) / len(param_cos_sims) if param_cos_sims else float("nan")
                k_fold_mean_grad_cos_sim.append(mean_grad_cos_sim)
                edge_influence = edge_influence / self.graph.train_mask.sum()
                k_fold_edge_influence.append(edge_influence.item())

            candidate_label = "candidate" if candidate_idx is None else f"candidate {candidate_idx}"
            num_target_edges = 1 if targets.dim() == 1 else int(targets.shape[0])
            overall_mean_grad_cos_sim = (
                sum(k_fold_mean_grad_cos_sim) / len(k_fold_mean_grad_cos_sim)
                if k_fold_mean_grad_cos_sim else float("nan")
            )
            fold_summary = ", ".join(f"{v:.6f}" for v in k_fold_mean_grad_cos_sim)
            tqdm.write(
                f"[grad-cos] {candidate_label}: mean={overall_mean_grad_cos_sim:.6f}, "
                f"folds=[{fold_summary}], target_edges={num_target_edges}, "
                f"influenced_train_nodes={train_influenced_nodes.numel()}"
            )

            return k_fold_edge_influence, train_influenced_nodes.numel()

    @staticmethod
    def _ensure_2d_edge_tensor(targets, device=None):
        if not torch.is_tensor(targets):
            targets = torch.tensor(targets, dtype=torch.long, device=device)
        if device is not None:
            targets = targets.to(device=device)
        targets = targets.to(dtype=torch.long)
        if targets.dim() == 1:
            if targets.numel() != 2:
                raise ValueError("targets must contain 2 values for a single edge.")
            targets = targets.view(1, 2)
        if targets.dim() != 2 or targets.shape[1] != 2:
            raise ValueError("targets must have shape [2] or [num_edges, 2].")
        return targets

    @staticmethod
    def _canonical_edge_key(edge):
        if torch.is_tensor(edge):
            u, v = [int(x) for x in edge.detach().cpu().reshape(-1).tolist()]
        else:
            u, v = [int(x) for x in edge]
        return (u, v) if u <= v else (v, u)

    def get_train_influenced_nodes(self, targets):
        targets = self._ensure_2d_edge_tensor(targets, device=self.graph.edge_index.device)
        self.get_nodes_within_km1_hop()

        influenced_nodes = []
        for target in targets:
            inf_nodes = torch.unique(
                torch.cat(
                    [
                        self.nodes_within_km1_hop[target[0].item()],
                        self.nodes_within_km1_hop[target[1].item()],
                    ]
                )
            ).to(torch.long)
            influenced_nodes.append(inf_nodes)

        if len(influenced_nodes) == 0:
            return torch.empty(0, dtype=torch.long, device=self.graph.train_mask.device)

        influenced_nodes = torch.unique(torch.cat(influenced_nodes, dim=-1))
        influenced_mask = torch.zeros_like(self.graph.train_mask, dtype=torch.bool)
        influenced_mask[influenced_nodes] = True
        train_influenced_mask = torch.logical_and(influenced_mask, self.graph.train_mask)
        return train_influenced_mask.nonzero().squeeze(1)

    def build_local_train_graph(self, targets):
        targets = self._ensure_2d_edge_tensor(targets, device=self.graph.edge_index.device)
        train_influenced_nodes = self.get_train_influenced_nodes(targets)
        local_graph = self.graph.clone()
        local_train_mask = torch.zeros_like(self.graph.train_mask, dtype=torch.bool)
        if train_influenced_nodes.numel() > 0:
            local_train_mask[train_influenced_nodes] = True
        local_graph.train_mask = local_train_mask
        return local_graph, train_influenced_nodes

    def get_node_output_representations(self):
        if self.node_output_representations is None:
            self.model.eval()
            with torch.no_grad():
                self.node_output_representations = self.model(self.graph).detach()
        return self.node_output_representations

    def get_edge_representations(self, targets):
        targets = self._ensure_2d_edge_tensor(targets, device=self.graph.edge_index.device)
        node_repr = self.get_node_output_representations()
        src = targets[:, 0]
        dst = targets[:, 1]
        mean_repr = 0.5 * (node_repr[src] + node_repr[dst])
        diff_repr = torch.abs(node_repr[src] - node_repr[dst])
        return torch.cat([mean_repr, diff_repr], dim=1).detach()

    def get_curvature_probe_vectors(self, probe_dim=None, probe_seed=None):
        if probe_dim is None:
            probe_dim = int(getattr(self.args, "groupwise_probe_dim", 8))
        if probe_seed is None:
            probe_seed = int(getattr(self.args, "groupwise_probe_seed", 0))

        params = [p for p in self.model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise ValueError("Model has no trainable parameters.")
        flat_dim = int(sum(p.numel() for p in params))
        dtype = params[0].dtype
        cache_key = (int(probe_dim), int(probe_seed), flat_dim, str(self.device), str(dtype))
        if cache_key not in self.curvature_probe_cache:
            self.curvature_probe_cache[cache_key] = generate_probe_vectors(
                dim=flat_dim,
                probe_dim=int(probe_dim),
                seed=int(probe_seed),
                device=self.device,
                dtype=dtype,
                normalize=True,
            )
        return self.curvature_probe_cache[cache_key]

    def compute_edge_curvature_proxies(self, targets, probe_vecs=None, normalize=True):
        targets = self._ensure_2d_edge_tensor(targets, device=self.graph.edge_index.device)
        if probe_vecs is None:
            probe_vecs = self.get_curvature_probe_vectors()
        probe_vecs = probe_vecs.to(device=self.device)

        target_keys = [self._canonical_edge_key(edge) for edge in targets]
        missing = []
        missing_keys = []
        for edge, key in zip(targets, target_keys):
            if key not in self.edge_curvature_proxy_cache:
                missing.append(edge.detach().clone())
                missing_keys.append(key)

        if len(missing) > 0:
            module = self._create_lissa_module()
            params = module._model_make_functional()
            flat_params = module._flatten_params_like(params)

            try:
                for edge, key in zip(missing, missing_keys):
                    local_graph, train_influenced_nodes = self.build_local_train_graph(edge)
                    if train_influenced_nodes.numel() == 0:
                        proxy = torch.zeros(probe_vecs.shape[0], device=self.device, dtype=probe_vecs.dtype)
                    else:
                        responses = []
                        for probe in probe_vecs:
                            module.vjp_func = None
                            hvp = module._hvp_graph(local_graph, flat_params, vec=probe, gnh=module.gnh)
                            scalar = torch.dot(probe, hvp).detach()
                            if not torch.isfinite(scalar):
                                scalar = torch.zeros((), device=probe.device, dtype=probe.dtype)
                            responses.append(scalar)
                        proxy = torch.stack(responses, dim=0)
                    self.edge_curvature_proxy_cache[key] = proxy.detach().cpu()
            finally:
                with torch.no_grad():
                    module._model_reinsert_params(module._reshape_like_params(flat_params), register=True)

        proxy_matrix = torch.stack(
            [self.edge_curvature_proxy_cache[key].to(device=self.device, dtype=probe_vecs.dtype) for key in target_keys],
            dim=0,
        )
        if normalize:
            proxy_matrix = normalize_proxy_matrix(proxy_matrix)
        return proxy_matrix.detach()
    
    def get_message_passing_influence(self, targets, influence_type):
        """
        target: the target to estimate influence
        influence_type: the type of graph element. Choices: {'edge_removal', 'edge_insertion'}
        """
        if influence_type == 'edge_removal':
            removed_edge_idx = []
            for target in targets:
                _, r_edge_idx = get_edge_weight(self.graph, target)
                removed_edge_idx.append(r_edge_idx)
            removed_edge_idx = torch.cat(removed_edge_idx, dim=-1)

            k_fold_message_passing_effect = []
            for i in range(self.num_folds):
                eval_grad = self.weight_grad[i][removed_edge_idx]
                message_passing_effect = eval_grad.sum() * -1
                k_fold_message_passing_effect.append(message_passing_effect.item())
        elif influence_type == 'edge_insertion':
            added_edge_idx = []
            for target in targets:
                _, a_edge_idx = get_edge_weight(self.graph_with_dummy_edges, target)
                added_edge_idx.append(a_edge_idx)
            added_edge_idx = torch.cat(added_edge_idx)
            
            k_fold_message_passing_effect = []
            for i in range(self.num_folds):
                eval_grad = self.weight_grad_with_dummy_edges[i][added_edge_idx]
                message_passing_effect = eval_grad.sum()
                k_fold_message_passing_effect.append(message_passing_effect.item())

        return k_fold_message_passing_effect

    def calculate_influence(self, candidates, influence_type):
        """
        candidates: list containing the targets to estimate the influence
        influence_type: the type of graph element. Choices: {'edge_removal', 'edge_insertion'}
        """
        self.get_inv_hvp()

        if "edge" in influence_type:
            self.get_nodes_within_km1_hop()
        elif "node" in influence_type:
            self.get_nodes_within_k_hop()
        
        if influence_type in ["edge_insertion"]:
            self.get_weight_grad_with_dummy_edges(candidates.view(-1,2))

        params = [p for p in self.model.parameters() if p.requires_grad]

        total_inf_list = []
        parameter_shift_inf_list = []
        message_passing_inf_list = []
        total_num_influenced_nodes = 0

        for candidate_idx, target in enumerate(tqdm(candidates), start=1):
            parameter_shift_inf, num_influenced_nodes = self.get_parameter_shifting_influence(
                target,
                influence_type,
                params,
                candidate_idx=candidate_idx,
            )
            parameter_shift_inf = torch.tensor(parameter_shift_inf)
            total_num_influenced_nodes += num_influenced_nodes
            parameter_shift_inf_list.append(parameter_shift_inf)

            message_passing_inf = self.get_message_passing_influence(target, influence_type)
            message_passing_inf = torch.tensor(message_passing_inf)
            message_passing_inf_list.append(message_passing_inf)

            total_inf = parameter_shift_inf + message_passing_inf
            total_inf_list.append(total_inf)

        parameter_shift_inf_list = torch.stack(parameter_shift_inf_list)
        message_passing_inf_list = torch.stack(message_passing_inf_list)
        total_inf_list = torch.stack(total_inf_list)
        
        avg_num_influenced_nodes = total_num_influenced_nodes / candidates.shape[0]
        return (
            total_inf_list,
            parameter_shift_inf_list,
            message_passing_inf_list,
            self.module.scale,
            self.inv_hvp_norm,
            avg_num_influenced_nodes,
        )
    
    def _load_exact_k_hop_neighbors(self):
        if self.eval_metric == 'feature_ablation':
            return find_k_hop_neighborhoods(self.graph, self.args.num_layers)
        else:
            return None

    def _create_lissa_module(self):
        lissa_scale = float(self.args.scale)
        exp_name = getattr(self.args, "experiment_name", "none")
        auto_scale_by_experiment = {
            "non_neighbor_edges": 32.0,  # exp3
            "clusters": 32.0,            # exp5 (and cluster-style runs)
        }
        if (
            self.args.hessian_type == "GNH"
            and exp_name in auto_scale_by_experiment
            and lissa_scale <= 1.0
        ):
            # Cluster/grouped edge edits commonly diverge with scale=1.0, causing many restarts.
            # Keep user-provided scales intact and only stabilize the default.
            lissa_scale = auto_scale_by_experiment[exp_name]
            print(
                f"[LiSSA] Auto-adjust scale for {exp_name}: {float(self.args.scale):.2f} -> {lissa_scale:.2f} "
                f"(hessian_type={self.args.hessian_type})"
            )

        return LiSSAInfluenceModule(
            graph=self.graph,
            model=self.model,
            objective=CrossEntropyObjective(self.args),
            train_loader=None,
            test_loader=None,
            device=self.device,
            damp=self.args.damp,
            repeat=1,
            lissa_iter = self.args.lissa_iter,
            scale=lissa_scale,
            depth=None,
            gnh=True if self.args.hessian_type=='GNH' else False,
            full_batch=True
        )

    def get_inv_hvp(self):
        if self.inv_hvp is None:
            self.module = self._create_lissa_module()
            eval_result, weight_grad, inv_hvp, inv_hvp_norm = self.approximate_inv_hvp(
                self.model, self.graph, self.module, self.eval_metric, self.num_folds, self.validation_splits
            )

            params = [p for p in self.model.parameters() if p.requires_grad]
            
            reshaped_inv_hvp = []
            for i in range(self.num_folds):
                reshaped_inv_hvp.append(reshape_like_params(inv_hvp[i], params))
            self.inv_hvp = reshaped_inv_hvp
            self.weight_grad = weight_grad
            self.inv_hvp_norm = inv_hvp_norm
    
    def get_nodes_within_k_hop(self):
        if self.nodes_within_k_hop is None:
            self.nodes_within_k_hop = find_nodes_within_k_hop(self.graph, self.args.num_layers)
    
    def get_nodes_within_km1_hop(self):
        if self.args.dataset == "Squirrel":
            # To do: Integrate across all datasets.
            self.nodes_within_km1_hop = find_k_hop_neighbors_bfs(self.graph, self.args.num_layers-1, device="cpu")
        if self.nodes_within_km1_hop is None:
            self.nodes_within_km1_hop = find_nodes_within_k_hop(self.graph, self.args.num_layers-1)

    @staticmethod
    def _set_leaf_edge_weight_requires_grad(graph):
        # Some graph edits (clone/slicing/cat) can produce non-leaf edge_weight.
        # Autograd flags can be toggled only on leaf tensors.
        graph.edge_weight = graph.edge_weight.detach().clone().requires_grad_(True)

    def get_weight_grad_with_dummy_edges(self, insertion_candidates):
        self.graph_with_dummy_edges = add_zero_weight_edges(self.graph, insertion_candidates)
        self._set_leaf_edge_weight_requires_grad(self.graph_with_dummy_edges)
        
        weight_grads = []
        if self.eval_metric == "mean_validation_loss":
            
            for i in range(self.num_folds):
                valid_idxs = self.validation_splits[i]
                eval_result = mean_validation_loss(self.model, self.graph_with_dummy_edges, valid_idxs)
                weight_grad = grad(eval_result, self.graph_with_dummy_edges.edge_weight)[0]
                weight_grads.append(weight_grad)
        else:
            eval_result = self.get_eval_result(self.model, self.graph_with_dummy_edges)
            weight_grad = grad(eval_result, self.graph_with_dummy_edges.edge_weight)[0]
            weight_grads.append(weight_grad)
        
        self.weight_grad_with_dummy_edges = weight_grads
    
    def get_perturbed_logit(self, model, graph, removed_edge=None, removed_node=None, added_edge=None):
        perturbed_graph = graph.clone()
        if removed_edge is not None:
            for edge in removed_edge:
                perturbed_graph = remove_edge(perturbed_graph, edge)
            perturbed_logit = model(perturbed_graph)
        elif added_edge is not None:
            for edge in added_edge:
                perturbed_graph = add_edge(perturbed_graph, edge)
            perturbed_logit = model(perturbed_graph)
        
        return perturbed_logit

    def get_eval_result(self, model, graph):
        self._set_leaf_edge_weight_requires_grad(graph)
        eval_result = self.metric_fn(model, graph)

        return eval_result

    def approximate_inv_hvp(self, model, graph, module, eval_metric, num_folds, validation_splits):
        eval_results = []
        weight_grads = []
        inv_hvps = []
        inv_hvp_norms = []
        if eval_metric == 'mean_validation_loss':
            self._set_leaf_edge_weight_requires_grad(graph)
            
            for i in range(num_folds):
                params = list(model.parameters())
                valid_idxs = validation_splits[i]
                eval_result = mean_validation_loss(model, graph, valid_idxs)
                param_grad = grad(eval_result, params, retain_graph=True)
                flatten_vec = flatten_params_like(param_grad, params)
                weight_grad = grad(eval_result, graph.edge_weight)[0]
                inv_hvp, inv_hvp_norm = module.stest(grad_eval=flatten_vec)

                eval_results.append(eval_result)
                weight_grads.append(weight_grad)
                inv_hvps.append(inv_hvp)
                inv_hvp_norms.append(inv_hvp_norm)
        elif eval_metric in ['feature_ablation','dirichlet_energy']:
            self._set_leaf_edge_weight_requires_grad(graph)
            params = list(model.parameters())
            eval_result = self.metric_fn(model, graph)
            param_grad = grad(eval_result, params, retain_graph=True)
            flatten_vec = flatten_params_like(param_grad, params)
            weight_grad = grad(eval_result, graph.edge_weight)[0]
            inv_hvp, inv_hvp_norm = module.stest(grad_eval=flatten_vec)

            eval_results.append(eval_result)
            weight_grads.append(weight_grad)
            inv_hvps.append(inv_hvp)
            inv_hvp_norms.append(inv_hvp_norm)
        else:
            raise ValueError
        
        return eval_results, weight_grads, inv_hvps, inv_hvp_norms

    def get_indiv_grad(self, logits, targets, params):
        criterion = nn.CrossEntropyLoss()
        results = [[] for _ in range(len(params))]

        for i in range(targets.numel()):
            indiv_loss = criterion(logits[i], targets[i])
            indiv_grad = grad(indiv_loss, params, retain_graph=True)

            indiv_grad_detached = [g.detach() for g in indiv_grad]

            for j, paramwise_grad in enumerate(indiv_grad_detached):
                results[j].append(paramwise_grad)

        tensor_results = []
        for result in results:
            tensor_results.append(torch.stack(result))

        return tensor_results

def calculate_loo(model, graph, candidate_edges, args, seed, model_save_dir, metric_fn, element_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = int(graph.x.shape[-1])
    num_classes = int(torch.max(graph.y).item()) + 1

    evaluation_result = metric_fn(model, graph)

    loo_results = []
    for candidate_edge in tqdm(candidate_edges):
        if element_type == 'edge_removal':
            perturbed_graph = remove_edge(graph, candidate_edge)
            perturbed_graph.edge_weight = perturbed_graph.edge_weight.detach()
        elif element_type == 'edge_insertion':
            perturbed_graph = add_edge(graph, candidate_edge)
            perturbed_graph.edge_weight = perturbed_graph.edge_weight.detach()
        else:
            raise ValueError

        set_seed(seed)
        new_model = GNN(
                name=args.model,
                in_dim=in_dim,
                hidden_dim=args.hidden_dim, 
                num_classes=num_classes,
                num_layers=args.num_layers,
                linear=args.linear,
                bias=args.bias,
                num_heads=args.num_heads,
            )
        
        edge_perturb_model_path = osp.join(model_save_dir, _candidate_edge_checkpoint_name(candidate_edge))
        if osp.isfile(edge_perturb_model_path):
            edge_perturb_state_dict = torch.load(edge_perturb_model_path, map_location=device, weights_only=True)
            new_model.load_state_dict(edge_perturb_state_dict)
            new_model = new_model.to(device)
        else:
            new_model = new_model.to(device)
            new_optimizer = optim.SGD(new_model.parameters(), lr=args.lr, weight_decay=args.damp)

            new_model.train()
            for _ in range(args.epochs):
                train_loss, _, _, _, _, _ = train(perturbed_graph, new_model, new_optimizer, device)
            torch.save(new_model.state_dict(), edge_perturb_model_path)

        new_model.eval()
        perturbed_result = metric_fn(new_model, perturbed_graph)

        loo_result = perturbed_result-evaluation_result
        loo_results.append(loo_result.item())

    return loo_results

def calculate_pbrf(model, graph, candidate_edges, args, seed, model_dir, metric_fn, element_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = int(graph.x.shape[-1])
    num_classes = int(torch.max(graph.y).item()) + 1
    eval_result = metric_fn(model, graph)

    km1_hop_neighbors = find_k_hop_neighbors_bfs(graph, args.num_layers-1, device="cpu")
    y_s = model(graph)
    theta_s = flatten_parameters(model).detach()
    loss_func = nn.CrossEntropyLoss()

    train_y_s = y_s[graph.train_mask]
    train_target = graph.y[graph.train_mask]

    bregman_grad = grad(loss_func(train_y_s, train_target), train_y_s)[0]
    y_s = y_s.detach()

    pbrf_results = []
    pbrf_nip_results = []
    pbrf_nrt_results = []
    for edge_idx, candidate_edge in enumerate(tqdm(candidate_edges)):
        graph.x.requires_grad = False
        graph.edge_weight.requires_grad = False

        if candidate_edge.dim() == 2:
            influenced_nodes = []
            for target in candidate_edge:
                neighbors = torch.unique(torch.cat([
                    km1_hop_neighbors[target[0].item()],
                    km1_hop_neighbors[target[1].item()],
                ])).to(torch.long)
                i_nodes = neighbors.to(graph.train_mask.device)
                influenced_nodes.append(i_nodes)
            influenced_nodes = torch.unique(torch.cat(influenced_nodes, dim=-1))
        else:
            neighbors = torch.unique(torch.cat([
                km1_hop_neighbors[candidate_edge[0].item()],
                km1_hop_neighbors[candidate_edge[1].item()],
            ])).to(torch.long)
            influenced_nodes = neighbors.to(graph.train_mask.device)
        influenced_mask = torch.zeros_like(graph.train_mask)
        influenced_mask[influenced_nodes] = 1
        train_influenced_mask = torch.logical_and(influenced_mask, graph.train_mask)
        train_influenced_nodes = train_influenced_mask.nonzero().squeeze(1)

        if element_type == 'edge_removal':
            perturbed_graph = graph.clone()
            for edge in candidate_edge:
                perturbed_graph = remove_edge(perturbed_graph, edge)
        elif element_type == 'edge_insertion':
            perturbed_graph = graph.clone()
            for edge in candidate_edge:
                perturbed_graph = add_edge(perturbed_graph, edge)
        else:
            raise ValueError
        
        if train_influenced_nodes.numel() == 0:
            model.eval()
            perturbed_result = metric_fn(model, perturbed_graph)
            perturbed_result_nip = metric_fn(model, graph)
            perturbed_result_nrt = metric_fn(model, perturbed_graph)
        else:
            set_seed(seed)
            new_model = GNN(
                    name=args.model,
                    in_dim=in_dim,
                    hidden_dim=args.hidden_dim, 
                    num_classes=num_classes,
                    num_layers=args.num_layers,
                    linear=args.linear,
                    bias=args.bias,
                    num_heads=args.num_heads
                )
            
            edge_perturb_model_path = osp.join(model_dir, _candidate_edge_checkpoint_name(candidate_edge))
            if osp.isfile(edge_perturb_model_path):
                edge_perturb_state_dict = torch.load(edge_perturb_model_path, map_location=device, weights_only=True)
                new_model.load_state_dict(edge_perturb_state_dict)
                new_model = new_model.to(device)
            else:
                new_model.load_state_dict(model.state_dict())
                new_model = new_model.to(device)

                new_optimizer = optim.SGD(new_model.parameters(), lr=args.lr, weight_decay=args.pbrf_weight_decay)

                new_model.train()
                for epoch in range(args.pbrf_epochs):
                    train_loss, remove_loss, add_loss, train_acc, val_acc, test_acc = train_pbrf(train_influenced_nodes, graph, perturbed_graph, new_model, new_optimizer, device, y_s, theta_s, bregman_grad, args)

                torch.save(new_model.state_dict(), edge_perturb_model_path)

            new_model.eval()
            perturbed_result = metric_fn(new_model, perturbed_graph)
            perturbed_result_nip = metric_fn(new_model, graph)
            perturbed_result_nrt = metric_fn(model, perturbed_graph)

        pbrf_result = perturbed_result-eval_result
        pbrf_results.append(pbrf_result.item())
        pbrf_nip_results.append((perturbed_result_nip-eval_result).item())
        pbrf_nrt_results.append((perturbed_result_nrt-eval_result).item())

    return pbrf_results, pbrf_nip_results, pbrf_nrt_results


def get_pbrf(args, model, data, candidate_edges, seed, dirs, element_type, metric_fn=None):
    print('Calculate PBRF...')
    start_time = time.time()
    if metric_fn is None:
        metric_fn = globals().get("metric_fn")
        if metric_fn is None:
            raise ValueError("metric_fn must be provided to get_pbrf.")
    edge_pbrf, act_nip, act_nrt = calculate_pbrf(model, data, candidate_edges, args, seed, dirs["pbrf_model"], metric_fn, element_type)
    print(f'Consumed time: {time.time()-start_time:.2f}s')

    return edge_pbrf, act_nip, act_nrt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora_public')
    parser.add_argument('--model', type=str, default='GCN', choices=['SGC', 'GCN', 'GAT', 'ChebNet'])
    parser.add_argument('--hessian_type', type=str, default='GNH', choices=['hessian', 'GNH'])
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--damp', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--lissa_iter', type=int, default=10000)
    parser.add_argument('--eval_metric', type=str, default='mean_validation_loss', choices=['dirichlet_energy', 'feature_ablation', 'mean_validation_loss'])
    parser.add_argument('--linear', type=int, default=0)
    parser.add_argument('--bias', type=int, default=0)
    parser.add_argument('--pbrf_epochs', type=int, default=1000)
    parser.add_argument('--pbrf_weight_decay', type=float, default=0.0)
    parser.add_argument("--element_type", type=str, default='edge_edit', choices=['edge_removal', 'edge_insertion', 'edge_edit'])
    parser.add_argument("--num_insertion_candidates", type=int, default=50)
    parser.add_argument("--num_removal_candidates", type=int, default=50)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--check_runtime", type=int, default=0)
    parser.add_argument("--json_config", type=str, default="none")
    parser.add_argument("--fig_title", type=str, default="none")
    parser.add_argument("--num_group_elem", type=int, default=1)
    parser.add_argument("--experiment_name", type=str, default="none")
    parser.add_argument("--num_of_clusters", type=int, default=3)
    parser.add_argument("--edges_per_cluster", type=int, default=-1)
    parser.add_argument("--cluster_ratio_percent", type=int, default=10)
    parser.add_argument("--intra_cluster_dist", type=int, default=1)
    parser.add_argument("--inter_cluster_dist", type=int, default=1)
    parser.add_argument("--removal_cluster_dist", type=int, default=1)
    parser.add_argument(
        "--removal_candidate_sampler",
        type=str,
        default="uniform",
        choices=["uniform", "group_non_neighbor", "group_neighbor"],
    )
    parser.add_argument("--removal_neighbor_dist", type=int, default=1)
    parser.add_argument("--cluster_candidate_init_only", type=int, default=0)
    parser.add_argument("--cluster_candidate_force_rebuild", type=int, default=0)
    parser.add_argument("--cluster_candidate_cache_root", type=str, default="candidate_cache")
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

    args = parser.parse_args()
    args.linear = bool(args.linear)
    args.bias = bool(args.bias)
    print(args)

    if args.json_config != "none":
        import json
        from types import SimpleNamespace

        json_config = args.json_config
        with open(args.json_config, 'r') as f:
            args = json.load(f)
        args = SimpleNamespace(**args)     
        args.json_config = json_config 
        if "fig_title" not in vars(args).keys():  
            args.fig_title = args.eval_metric
        print(args)

    args = _ensure_optional_experiment_defaults(args)

    WD = args.weight_decay
    PBRF_WD = args.pbrf_weight_decay
    if args.hessian_type == 'hessian':
        print('Warning. args.damp should be the same with args.weight_decay when args.hessian_type is hessian.')
        print(f'Original damp: {args.damp}, adjusted damp: {args.weight_decay}')
        args.damp = args.weight_decay

    dataset = DataLoader(args.dataset, root='datasets')
    args.num_classes = dataset.num_classes
    data = dataset[0]
    data.edge_weight = torch.ones((data.edge_index.shape[1], ))
    args = _apply_experiment_runtime_overrides(args, data)

    if (
        args.cluster_candidate_init_only
        and args.experiment_name == "clusters"
        and args.element_type in ["edge_removal", "edge_edit"]
    ):
        candidates, cache_path, was_built = build_or_load_clustered_edge_removal_candidates(
            graph=data,
            args=args,
            force_rebuild=bool(args.cluster_candidate_force_rebuild),
        )
        status = "built" if was_built else "loaded"
        print(
            f"[CANDIDATE-CACHE] init-only {status}: path={cache_path}, "
            f"shape={tuple(candidates.shape)}"
        )
        raise SystemExit(0)

    dirs = make_dirs(args)
    save_config(args, osp.join(dirs['result'], 'config.json'), dirs)

    SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]
    seed = SEEDS[0]

    vanilla_dir = dirs["vanilla"]
    vanilla_path = osp.join(vanilla_dir, f"{seed}.pth")
    
    eval_node_idxs = get_eval_node_idxs(data, args.eval_metric, seed)

    if 'public' not in args.dataset:
        percls_trn = int(round(0.6*len(data.y)/dataset.num_classes))
        val_lb = int(round(0.2*len(data.y)))
        data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb, seed)

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
                num_heads=args.num_heads
            )
    if osp.isfile(vanilla_path):
        model_state_dict = torch.load(vanilla_path, map_location=device, weights_only=True)
        model.load_state_dict(model_state_dict)
        model = model.to(device)
    else:
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(1,args.epochs+1):
            train_loss, val_loss, test_loss, train_acc, val_acc, test_acc = train(data, model, optimizer, device)
            if epoch % 100 == 0:
                print("-----------------------------------------------")
                print(f"Epoch: {epoch}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, test_loss: {test_loss:.4f}")
                print(f"Train acc: {train_acc*100:.2f}%, val acc: {val_acc*100:.2f}%, test_acc: {test_acc*100:.2f}%")
                print("-----------------------------------------------")
        torch.save({k: v.clone().detach() for k, v in model.state_dict().items()}, vanilla_path)

    save_name = f'influence_vs_pbrf'
    save_name_nip = f'parameter_shifting_effect'
    save_name_nrt = f'message_passing_effect'
    save_name_rtpt = f'parameter_shifting_vs_message_passing'

    set_seed(seed)
    if args.eval_metric == "feature_ablation":
        exact_k_hop_neighbors = find_k_hop_neighborhoods(data, args.num_layers)
    else:
        exact_k_hop_neighbors = None

    metric_fns = make_metric_fns(eval_node_idxs, exact_k_hop_neighbors, data.edge_index)
    metric_fn = metric_fns[args.eval_metric]

    if args.element_type in ['edge_removal', 'edge_edit']:
        set_seed(seed)
        if args.experiment_name == "clusters":
            removal_candidates = get_grouped_edge_removal_candidates(data, args)
            candidates = removal_candidates
        else:
            num_candidates = args.num_removal_candidates * args.num_group_elem
            candidates = get_edge_removal_candidates(data, num_candidates)
            candidates = candidates.view(args.num_removal_candidates, args.num_group_elem, 2)
            removal_candidates = candidates
    if args.element_type in ['edge_insertion', 'edge_edit']:
        set_seed(seed)
        num_candidates = args.num_insertion_candidates * args.num_group_elem
        candidates = get_edge_insertion_candidates(data, num_candidates*2)[:num_candidates]
        candidates = candidates.view(args.num_insertion_candidates, args.num_group_elem, 2)

        num_insertion_candidates = num_candidates
        insertion_candidates = candidates
        
    print(f'Calculate the Influence of {args.element_type}...')
    start_time = time.time()
    influence_module = GraphInfluenceModule(model, data, args, args.eval_metric, 1, eval_node_idxs, metric_fn)
    if args.element_type == 'edge_edit':
        r_total_inf, r_parameter_shift_inf, r_message_passing_inf, module_scale, inv_hvp_norm, num_ins = influence_module.calculate_influence(removal_candidates, 'edge_removal')
        i_total_inf, i_parameter_shift_inf, i_message_passing_inf, module_scale, inv_hvp_norm, num_ins = influence_module.calculate_influence(insertion_candidates, 'edge_insertion')
        
        total_inf = torch.cat((r_total_inf, i_total_inf), dim=0)
        parameter_shift_inf = torch.cat((r_parameter_shift_inf, i_parameter_shift_inf), dim=0)
        message_passing_inf = torch.cat((r_message_passing_inf, i_message_passing_inf), dim=0)
    else:
        total_inf, parameter_shift_inf, message_passing_inf, module_scale, inv_hvp_norm, num_ins = influence_module.calculate_influence(candidates, args.element_type)
    print(f'Consumed time: {time.time()-start_time:.2f}s')

    if args.hessian_type == 'hessian':
        loo = calculate_loo(model, data, candidates, args, seed, dirs['loo_model'], metric_fn, args.element_type)

        parameter_shift_vec = parameter_shift_inf.detach().reshape(-1).to(torch.float32).cpu()
        loo_vec = torch.as_tensor(loo, dtype=torch.float32).reshape(-1).cpu()
        common_n = min(parameter_shift_vec.numel(), loo_vec.numel())
        if common_n > 0:
            parameter_shift_vec = parameter_shift_vec[:common_n]
            loo_vec = loo_vec[:common_n]
            mask = torch.logical_and(is_within_2std(parameter_shift_vec), is_within_2std(loo_vec))
            plot_influence_loss(parameter_shift_vec[mask], loo_vec[mask], dirs['result'], save_name_nip, args, title=dirs['fig_title'])

    elif args.hessian_type == 'GNH':
        if args.element_type == "edge_edit":
            r_total_pbrf, r_parameter_shift_pbrf, r_message_passing_pbrf = get_pbrf(args, model, data, removal_candidates, seed, dirs, 'edge_removal')
            i_total_pbrf, i_parameter_shift_pbrf, i_message_passing_pbrf = get_pbrf(args, model, data, insertion_candidates, seed, dirs, 'edge_insertion')
            
            total_pbrf = r_total_pbrf + i_total_pbrf
            parameter_shift_pbrf = r_parameter_shift_pbrf + i_parameter_shift_pbrf
            message_passing_pbrf = r_message_passing_pbrf + i_message_passing_pbrf
            r_size = len(r_total_pbrf)
        else:
            total_pbrf, parameter_shift_pbrf, message_passing_pbrf = get_pbrf(args, model, data, candidates, seed, dirs, args.element_type)
            r_size = None
        
        rename_result_dir(args, parameter_shift_inf, parameter_shift_pbrf, message_passing_inf, message_passing_pbrf, dirs)
        k=2
        total_inf_vec = total_inf.detach().reshape(-1).to(torch.float32).cpu()
        total_pbrf_vec = torch.as_tensor(total_pbrf, dtype=torch.float32).reshape(-1).cpu()
        parameter_shift_vec = parameter_shift_inf.detach().reshape(-1).to(torch.float32).cpu()
        message_passing_vec = message_passing_inf.detach().reshape(-1).to(torch.float32).cpu()
        common_n = min(
            total_inf_vec.numel(),
            total_pbrf_vec.numel(),
            parameter_shift_vec.numel(),
            message_passing_vec.numel(),
        )
        if common_n > 0:
            total_inf_vec = total_inf_vec[:common_n]
            total_pbrf_vec = total_pbrf_vec[:common_n]
            parameter_shift_vec = parameter_shift_vec[:common_n]
            message_passing_vec = message_passing_vec[:common_n]
            mask = torch.logical_and(is_within_2std(total_inf_vec, k), is_within_2std(total_pbrf_vec, k))
            plot_influence_loss(total_inf_vec[mask], total_pbrf_vec[mask], dirs['result'], save_name, args, title=dirs['fig_title'], mask=mask, r_size=r_size)
            plot_influence_loss(parameter_shift_vec[mask], message_passing_vec[mask], dirs['result'], save_name_rtpt, args, xlabel="Parameter Shift Effect", ylabel="Propagation Effect", title=dirs['fig_title'], mask=mask, r_size=r_size)
