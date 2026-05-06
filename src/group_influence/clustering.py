import torch


def cluster_features(features, method, num_clusters, seed=0, max_iter=100):
    feature_tensor = torch.as_tensor(features, dtype=torch.float32).detach().cpu()
    method = str(method).strip().lower()
    num_clusters = int(num_clusters)
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive.")
    if num_clusters > int(feature_tensor.shape[0]):
        raise ValueError("num_clusters cannot exceed the number of candidate edges.")

    if method in {"kmeans", "cheap_kmeans"}:
        labels, centroids, num_iter = _kmeans(feature_tensor, num_clusters, seed=seed, max_iter=max_iter)
        normalized_method = "cheap_kmeans"
    elif method == "random":
        labels = _random_labels(feature_tensor.shape[0], num_clusters, seed=seed)
        centroids = _centroids_for_labels(feature_tensor, labels, num_clusters)
        num_iter = 0
        normalized_method = "random"
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    return {
        "method": normalized_method,
        "labels": labels,
        "centroids": centroids,
        "num_iter": int(num_iter),
    }


def summarize_clusters(labels, features, influences):
    label_tensor = torch.as_tensor(labels, dtype=torch.long).detach().cpu()
    feature_tensor = torch.as_tensor(features, dtype=torch.float32).detach().cpu()
    influence_tensor = torch.as_tensor(influences, dtype=torch.float32).detach().cpu()
    num_clusters = int(label_tensor.max().item()) + 1 if label_tensor.numel() > 0 else 0

    rows = []
    for cluster_id in range(num_clusters):
        mask = label_tensor == cluster_id
        cluster_features = feature_tensor[mask]
        cluster_influences = influence_tensor[mask]
        cluster_size = int(mask.sum().item())
        if cluster_size == 0:
            continue
        influence_sum = float(cluster_influences.sum().item())
        positive_count = int((cluster_influences > 0).sum().item())
        negative_count = int((cluster_influences < 0).sum().item())
        zero_count = int((cluster_influences == 0).sum().item())
        nonzero_count = max(positive_count + negative_count, 1)
        sign_purity = float(max(positive_count, negative_count) / nonzero_count)
        centroid = cluster_features.mean(dim=0)
        within_variance = float(((cluster_features - centroid) ** 2).sum(dim=1).mean().item())
        rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": cluster_size,
                "cluster_influence_sum": influence_sum,
                "cluster_abs_edge_influence_sum": float(cluster_influences.abs().sum().item()),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "zero_count": zero_count,
                "sign_purity": sign_purity,
                "within_cluster_feature_variance": within_variance,
            }
        )

    total_influence = float(influence_tensor.sum().item())
    sum_abs_cluster_influence = float(sum(abs(row["cluster_influence_sum"]) for row in rows))
    cancellation = float(sum_abs_cluster_influence - abs(total_influence))
    denom = sum_abs_cluster_influence + 1e-12
    summary = {
        "num_edges": int(label_tensor.shape[0]),
        "num_clusters": int(len(rows)),
        "total_influence_sum": total_influence,
        "sum_abs_cluster_influence": sum_abs_cluster_influence,
        "cancellation_score": cancellation,
        "normalized_cancellation_score": float(cancellation / denom),
        "mean_sign_purity": _mean([row["sign_purity"] for row in rows]),
        "weighted_sign_purity": _weighted_mean(
            [row["sign_purity"] for row in rows],
            [row["cluster_size"] for row in rows],
        ),
        "mean_within_cluster_feature_variance": _mean(
            [row["within_cluster_feature_variance"] for row in rows]
        ),
    }
    return {
        "summary": summary,
        "cluster_rows": rows,
    }


def _kmeans(features, num_clusters, seed=0, max_iter=100):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(features.shape[0], generator=generator)
    centroids = features[perm[:num_clusters]].clone()
    previous_labels = None

    for iteration in range(1, int(max_iter) + 1):
        distances = torch.cdist(features, centroids)
        labels = distances.argmin(dim=1)
        if previous_labels is not None and torch.equal(labels, previous_labels):
            return labels, centroids, iteration

        new_centroids = []
        min_distances = distances.min(dim=1).values
        for cluster_id in range(num_clusters):
            mask = labels == cluster_id
            if mask.any():
                new_centroids.append(features[mask].mean(dim=0))
            else:
                replacement_idx = int(min_distances.argmax().item())
                new_centroids.append(features[replacement_idx])
                min_distances[replacement_idx] = -1
        centroids = torch.stack(new_centroids, dim=0)
        previous_labels = labels

    return previous_labels, centroids, int(max_iter)


def _random_labels(num_items, num_clusters, seed=0):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(int(num_items), generator=generator)
    labels = torch.empty(int(num_items), dtype=torch.long)
    for offset, item_idx in enumerate(perm.tolist()):
        labels[item_idx] = int(offset % num_clusters)
    return labels


def _centroids_for_labels(features, labels, num_clusters):
    centroids = []
    for cluster_id in range(num_clusters):
        mask = labels == cluster_id
        if mask.any():
            centroids.append(features[mask].mean(dim=0))
        else:
            centroids.append(torch.zeros(features.shape[1], dtype=features.dtype))
    return torch.stack(centroids, dim=0)


def _mean(values):
    if len(values) == 0:
        return 0.0
    return float(sum(values) / len(values))


def _weighted_mean(values, weights):
    if len(values) == 0:
        return 0.0
    denom = float(sum(weights))
    if denom == 0:
        return 0.0
    return float(sum(value * weight for value, weight in zip(values, weights)) / denom)
