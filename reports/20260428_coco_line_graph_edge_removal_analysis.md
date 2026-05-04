# CoCo Line-Graph Edge-Removal Analysis

Analyzed artifacts:

- Summary: `results/partition_compare_summary_20260428_223356.csv`
- Job dir: `results/partition_compare_jobs/20260428_223356`
- Implementation note: `reports/20260428_coco_line_graph_partition.md`

## 1. What Was Run

This run tested CoCo clustering on the full training-graph line graph, then assigned candidate removal edges by their learned line-graph cluster labels.

- partition method: `coco`
- partition strategy: `coco_full_line_graph_assignment`
- element type: `edge_removal`
- datasets: `texas`, `cornell`
- models: `GCN`, `GAT`
- layers: `2`
- ratios: `30`, `50`, `80`
- candidate count: `20`
- K specs:
  - `edge_cluster_size:16`
  - `edge_cluster_size:32`
  - `edge_cluster_size:64`
  - `sqrt_edges`
  - `avg_degree`
- total runs: `60`
- successful runs: `60 / 60`

For both `texas` and `cornell`, the resolved K values were:

| K spec | Resolved K |
|---|---:|
| `avg_degree` | 3 |
| `edge_cluster_size:64` | 5 |
| `edge_cluster_size:32` | 9 |
| `sqrt_edges` | 17 |
| `edge_cluster_size:16` | 18 |

## 2. Topline

Overall, CoCo line-graph partitioning did **not** improve the existing whole-group influence estimate in this pilot.

| Metric | Baseline | CoCo cluster-sum | Delta |
|---|---:|---:|---:|
| MAE vs PBRF | 0.3340 | 0.3702 | +0.0362 |
| Spearman vs PBRF | 0.3514 | 0.2871 | -0.0643 |
| Pearson vs PBRF | 0.3711 | 0.3040 | -0.0671 |
| Sign accuracy vs PBRF | 0.4292 | 0.4075 | -0.0217 |

Improvement counts across 60 runs:

| Metric | Improved runs |
|---|---:|
| MAE | 23 / 60 |
| Spearman | 26 / 60 |
| Sign accuracy | 20 / 60 |

The current result is therefore negative as a general-purpose replacement for whole-group influence.

## 3. K-Spec Comparison

| K spec | K | CoCo MAE | MAE delta | CoCo Spearman | Spearman delta | CoCo sign | Sign delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `avg_degree` | 3.00 | 0.3429 | +0.0089 | 0.3158 | -0.0356 | 0.4125 | -0.0167 |
| `edge_cluster_size:64` | 5.00 | 0.3598 | +0.0258 | 0.3524 | +0.0010 | 0.4083 | -0.0208 |
| `edge_cluster_size:32` | 9.00 | 0.3606 | +0.0266 | 0.2658 | -0.0856 | 0.4208 | -0.0083 |
| `edge_cluster_size:16` | 17.75 | 0.3929 | +0.0589 | 0.2870 | -0.0644 | 0.3958 | -0.0333 |
| `sqrt_edges` | 16.87 | 0.3949 | +0.0609 | 0.2145 | -0.1368 | 0.4000 | -0.0292 |

Readout:

- The best overall K spec is `avg_degree`, but it still loses to baseline on every aggregate metric.
- `edge_cluster_size:64` is the only spec that roughly preserves Spearman.
- Fine partitions (`edge_cluster_size:16`, `sqrt_edges`) are clearly too aggressive in this setup.

## 4. Dataset Split

| Dataset | Baseline MAE | CoCo MAE | MAE delta | Baseline Spearman | CoCo Spearman | Spearman delta | Baseline sign | CoCo sign | Sign delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cornell` | 0.2259 | 0.2214 | -0.0045 | 0.4040 | 0.4138 | +0.0098 | 0.5917 | 0.6017 | +0.0100 |
| `texas` | 0.4421 | 0.5190 | +0.0769 | 0.2987 | 0.1604 | -0.1384 | 0.2667 | 0.2133 | -0.0533 |

This is the clearest split in the run:

- `cornell` shows a small positive signal.
- `texas` is strongly negative.

So the CoCo line-graph approach is not uniformly bad, but it is dataset-sensitive and currently unreliable.

## 5. Dataset By K Spec

| Dataset | Best-ish K spec | Readout |
|---|---|---|
| `cornell` | `edge_cluster_size:16` for MAE, `edge_cluster_size:64` for Spearman | Cornell benefits slightly from CoCo under several specs. |
| `texas` | `avg_degree` is least bad | Every K spec hurts Texas; finer K values hurt most. |

Concrete dataset/spec averages:

| Dataset | K spec | CoCo MAE | MAE delta | CoCo Spearman | Spearman delta |
|---|---|---:|---:|---:|---:|
| `cornell` | `edge_cluster_size:16` | 0.2136 | -0.0122 | 0.4546 | +0.0506 |
| `cornell` | `edge_cluster_size:64` | 0.2225 | -0.0034 | 0.4920 | +0.0880 |
| `texas` | `avg_degree` | 0.4652 | +0.0231 | 0.2241 | -0.0747 |
| `texas` | `edge_cluster_size:16` | 0.5721 | +0.1300 | 0.1193 | -0.1794 |

## 6. Ratio Split

| Ratio | CoCo MAE | MAE delta | CoCo Spearman | Spearman delta | CoCo sign | Sign delta |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0.2506 | +0.0091 | 0.3698 | -0.0602 | 0.5100 | +0.0475 |
| 50 | 0.3653 | +0.0226 | 0.3257 | +0.0009 | 0.4050 | -0.0325 |
| 80 | 0.4947 | +0.0770 | 0.1657 | -0.1335 | 0.3075 | -0.0800 |

CoCo gets worse as ratio increases. The ratio-80 setting is especially damaging.

## 7. Model Split

| Model | CoCo MAE | MAE delta | CoCo Spearman | Spearman delta | CoCo sign | Sign delta |
|---|---:|---:|---:|---:|---:|---:|
| `GCN` | 0.1617 | +0.0261 | 0.5500 | -0.1305 | 0.4850 | -0.0067 |
| `GAT` | 0.5788 | +0.0464 | 0.0242 | +0.0019 | 0.3300 | -0.0367 |

The GCN ranking drop is large. GAT Spearman is near zero in both baseline and CoCo, so the tiny positive delta should not be over-interpreted.

## 8. Runtime

Average runtime:

- end-to-end wall-clock: `1.50 min/run`
- mean candidate partition runtime: `2.52 sec`
- mean CoCo train runtime recorded per candidate diagnostics: `2.79 sec`

This is computationally manageable on the tested small graph setting.

Important caveat: this run used only `texas` and `cornell`; larger datasets may be dominated by dense line-graph adjacency and CoCo diffusion cost unless `COCO_USE_DIFFUSION=0` and cache reuse are used carefully.

## 9. Practical Conclusion

This pilot does not justify moving CoCo line-graph clustering into the main influence pipeline as a default.

The most defensible next steps are:

1. Keep CoCo as an ablation, not a replacement for METIS local/global.
2. Use only coarse K specs if continuing:
   - `avg_degree`
   - `edge_cluster_size:64`
3. Do not prioritize fine K specs:
   - `edge_cluster_size:16`
   - `sqrt_edges`
4. If pursuing the positive Cornell signal, run a narrow validation:
   - `cornell`
   - `edge_cluster_size:16`
   - `edge_cluster_size:64`
   - ratios `30,50`
   - seeds beyond `0`

Bottom line:

> CoCo line-graph clustering is technically runnable and fast enough on this small pilot, but it is not currently a reliable improvement over whole-group influence. Its only encouraging signal is Cornell-specific; Texas is clearly negative.
