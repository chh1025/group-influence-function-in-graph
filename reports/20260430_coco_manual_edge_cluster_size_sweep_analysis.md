# CoCo Manual Edge-Cluster-Size Sweep Analysis

Analyzed artifact:

- Summary: `results/partition_compare_summary_20260429_020401.csv`

## 1. Run Setup

- profile: `small`
- partition method: `coco`
- partition strategy: `coco_full_line_graph_assignment`
- element type: `edge_removal`
- auto-k method: `none`
- cluster specs: `edge_cluster_size:{8,16,32,64,128,256,512}`
- datasets: `cora_public`, `citeseer_public`, `texas`, `cornell`
- models: `GCN`, `GAT`
- layers: `2`, `4`
- ratios: `1`, `10`, `30`, `50`, `80`
- removal candidates per run: `100`
- seed: `0`
- total runs: `504`
- successful runs: `503 / 504`

One run failed:

| Run | Model | Dataset | Layer | Ratio | Spec | K | Error |
| ---: | --- | --- | ---: | ---: | --- | ---: | --- |
| 296 | `GAT` | `citeseer_public` | 2 | 30 | `edge_cluster_size:8` | 569 | `EOFError` while loading a PBRF model checkpoint |

The analysis below excludes that failed row.

## 2. Cluster-Size Spec Semantics

`edge_cluster_size:S` is not a balanced cluster-size constraint. The code resolves it as:

```text
K = ceil(full_graph_num_edges / S)
K = min(K, full_graph_num_edges, candidate_edge_count_for_ratio)
```

Then CoCo k-means uses that `K`. It does not enforce equal group sizes. At small ratios such as `1` and `10`, the candidate-edge cap can make the effective `K` smaller than `ceil(full_edges / S)`.

## 3. Topline

Run-level averages over 503 successful rows:

| Metric | Baseline | CoCo | Delta | Improved Runs |
| --- | ---: | ---: | ---: | ---: |
| MAE vs PBRF | 2.6894 | 2.6876 | -0.0019 | 173 / 503 |
| Spearman vs PBRF | 0.5315 | 0.4902 | -0.0413 | 108 / 503 |
| Pearson vs PBRF | 0.5298 | 0.4915 | -0.0383 | 133 / 503 |
| Sign accuracy vs PBRF | 0.7218 | 0.7166 | -0.0052 | 151 / 503 |

Interpretation:

- The average MAE improvement is tiny and not broad: only `34.4%` of successful runs improve MAE.
- Rank quality drops clearly: Spearman `-0.0413`, Pearson `-0.0383`.
- Sign accuracy is slightly worse on average.
- This sweep does not support CoCo line-graph clustering as a general default for edge-removal influence approximation.

## 4. Model Split

| Model | Runs | Base MAE | CoCo MAE | MAE d | Base Sp | CoCo Sp | Sp d | Base Sign | CoCo Sign | Sign d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GAT | 251 | 5.3113 | 5.3064 | -0.0049 | 0.4625 | 0.3977 | -0.0648 | 0.7323 | 0.7320 | -0.0003 |
| GCN | 252 | 0.0780 | 0.0791 | +0.0011 | 0.6003 | 0.5823 | -0.0179 | 0.7114 | 0.7013 | -0.0101 |

GAT receives most of the small aggregate MAE gain, but its rank degradation is much larger. GCN is more stable, but still loses rank and sign accuracy.

## 5. Dataset Split

| Dataset | Runs | Base MAE | CoCo MAE | MAE d | Base Sp | CoCo Sp | Sp d | Base Sign | CoCo Sign | Sign d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `citeseer_public` | 127 | 0.1002 | 0.1037 | +0.0035 | 0.6113 | 0.5780 | -0.0333 | 0.8805 | 0.8780 | -0.0025 |
| `cora_public` | 128 | 0.0531 | 0.0673 | +0.0142 | 0.7289 | 0.6505 | -0.0784 | 0.8648 | 0.8128 | -0.0520 |
| `cornell` | 124 | 5.3094 | 5.3012 | -0.0082 | 0.4195 | 0.3878 | -0.0316 | 0.5632 | 0.5998 | +0.0365 |
| `texas` | 124 | 5.4427 | 5.4250 | -0.0177 | 0.3581 | 0.3373 | -0.0208 | 0.5704 | 0.5690 | -0.0014 |

Dataset read:

- `cora_public` is the clearest negative case. MAE, rank, and sign accuracy all degrade.
- `citeseer_public` is also mildly negative overall.
- `cornell` improves MAE and sign accuracy, but loses rank quality.
- `texas` improves MAE, but rank still drops and sign is nearly unchanged.

## 6. Layer and Ratio Split

Layer split:

| Layer | Runs | Base MAE | CoCo MAE | MAE d | Base Sp | CoCo Sp | Sp d | Base Sign | CoCo Sign | Sign d |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 251 | 0.1631 | 0.1801 | +0.0170 | 0.5868 | 0.5417 | -0.0452 | 0.7062 | 0.6974 | -0.0088 |
| 4 | 252 | 5.2058 | 5.1851 | -0.0207 | 0.4764 | 0.4390 | -0.0374 | 0.7374 | 0.7358 | -0.0016 |

Ratio split:

| Ratio | Runs | Base MAE | CoCo MAE | MAE d | Base Sp | CoCo Sp | Sp d | Base Sign | CoCo Sign | Sign d |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 56 | 0.0112 | 0.0115 | +0.0003 | 0.5702 | 0.5646 | -0.0056 | 0.7341 | 0.7266 | -0.0075 |
| 10 | 112 | 0.0616 | 0.0622 | +0.0006 | 0.6296 | 0.6037 | -0.0259 | 0.7467 | 0.7196 | -0.0271 |
| 30 | 111 | 0.3938 | 0.3879 | -0.0059 | 0.5707 | 0.5339 | -0.0369 | 0.7330 | 0.7112 | -0.0217 |
| 50 | 112 | 5.1784 | 5.1662 | -0.0123 | 0.5096 | 0.4679 | -0.0417 | 0.7252 | 0.7172 | -0.0079 |
| 80 | 112 | 6.4425 | 6.4514 | +0.0089 | 0.3971 | 0.3186 | -0.0785 | 0.6765 | 0.7135 | +0.0369 |

Layer/ratio read:

- Layer 2 is worse on all tracked metrics.
- Layer 4 has an MAE gain, mostly from GAT layer 4, but still loses rank quality.
- Ratio 80 is risky for rank: Spearman drops by `-0.0785`.
- Ratio 30 and 50 improve MAE slightly, but the improvement comes with rank loss.

## 7. Cluster-Size Spec Split

| Edge Cluster Size | Runs | Mean K | Mean Groups | Base MAE | CoCo MAE | MAE d | Base Sp | CoCo Sp | Sp d | Base Sign | CoCo Sign | Sign d | Mean Wall Sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 79 | 248.04 | 199.15 | 2.3744 | 2.3716 | -0.0029 | 0.5350 | 0.4738 | -0.0612 | 0.7196 | 0.7011 | -0.0186 | 1480.58 |
| 16 | 64 | 162.75 | 142.27 | 1.0758 | 1.0651 | -0.0107 | 0.5260 | 0.4684 | -0.0576 | 0.7208 | 0.7098 | -0.0110 | 1329.90 |
| 32 | 64 | 81.50 | 75.32 | 3.0396 | 3.0397 | +0.0000 | 0.5259 | 0.4758 | -0.0501 | 0.7212 | 0.7118 | -0.0094 | 1316.49 |
| 64 | 64 | 41.25 | 39.55 | 5.1204 | 5.1100 | -0.0104 | 0.5251 | 0.4816 | -0.0435 | 0.7200 | 0.7266 | +0.0067 | 1295.13 |
| 128 | 72 | 23.00 | 20.20 | 2.6562 | 2.6551 | -0.0011 | 0.5380 | 0.5030 | -0.0351 | 0.7231 | 0.7159 | -0.0072 | 381.53 |
| 256 | 80 | 10.75 | 9.94 | 2.3929 | 2.3994 | +0.0065 | 0.5339 | 0.5072 | -0.0266 | 0.7238 | 0.7236 | -0.0002 | 391.39 |
| 512 | 80 | 5.50 | 5.23 | 2.3929 | 2.3953 | +0.0024 | 0.5338 | 0.5138 | -0.0200 | 0.7238 | 0.7271 | +0.0032 | 375.84 |

Cluster-size read:

- Fine partitions (`8`, `16`) are expensive and hurt rank quality the most.
- `edge_cluster_size:64` gives the best average MAE among cluster sizes and slightly improves sign accuracy, but still loses Spearman by `-0.0435`.
- Coarse partitions (`256`, `512`) preserve rank better and are much faster.
- `edge_cluster_size:512` is the safest if the priority is rank preservation, but it is close to a very coarse grouping and can become `K=1` on small graphs.

Win rates by cluster size:

| Edge Cluster Size | MAE Win Rate | Spearman Win Rate | Sign Win Rate |
| ---: | ---: | ---: | ---: |
| 8 | 35.44% | 24.05% | 35.44% |
| 16 | 39.06% | 25.00% | 32.81% |
| 32 | 31.25% | 25.00% | 32.81% |
| 64 | 42.19% | 29.69% | 37.50% |
| 128 | 34.72% | 20.83% | 26.39% |
| 256 | 26.25% | 21.25% | 32.50% |
| 512 | 33.75% | 7.50% | 15.00% |

The low Spearman win rate for `512` means it rarely improves rank, but its average rank damage is smallest because it mostly stays close to baseline.

## 8. Model-Specific Cluster Size

| Model | Edge Cluster Size | MAE d | Spearman d | Sign d | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| GAT | 16 | -0.0227 | -0.0950 | +0.0127 | Best GAT MAE, but severe rank loss |
| GAT | 64 | -0.0209 | -0.0715 | +0.0124 | Similar MAE/sign benefit with less rank loss than 16 |
| GAT | 512 | +0.0051 | -0.0322 | -0.0040 | Best GAT rank preservation, but no MAE/sign gain |
| GCN | 512 | -0.0003 | -0.0079 | +0.0105 | Best GCN tradeoff |
| GCN | 256 | +0.0000 | -0.0141 | +0.0085 | Similar, slightly worse rank |
| GCN | 64 | +0.0001 | -0.0156 | +0.0009 | Neutral MAE/sign, modest rank loss |

If we continue using manual cluster sizes, GCN should use coarse specs. GAT only benefits in MAE at smaller specs, but that comes with substantial rank degradation.

## 9. Dataset-Specific Cluster Size

Best cluster sizes by average MAE delta:

| Dataset | Best Spec | MAE d | Spearman d | Sign d |
| --- | ---: | ---: | ---: | ---: |
| `citeseer_public` | 512 | +0.0022 | -0.0230 | +0.0075 |
| `cora_public` | 256 | +0.0063 | -0.0654 | -0.0025 |
| `cornell` | 8 | -0.0419 | -0.0454 | +0.0948 |
| `texas` | 32 | -0.0349 | -0.0254 | -0.0021 |

Best cluster sizes by average Spearman delta:

| Dataset | Best Spec | MAE d | Spearman d | Sign d |
| --- | ---: | ---: | ---: | ---: |
| `citeseer_public` | 256 | +0.0031 | -0.0212 | -0.0005 |
| `cora_public` | 512 | +0.0074 | -0.0571 | +0.0055 |
| `cornell` | 512 | +0.0000 | +0.0000 | +0.0000 |
| `texas` | 256 | -0.0180 | +0.0015 | +0.0031 |

Dataset read:

- `cora_public` has no good manual spec in this sweep.
- `citeseer_public` is also not improved, but coarse specs limit the damage.
- `cornell` benefits in MAE/sign from fine specs, but rank worsens. `512` is effectively neutral because it becomes very coarse.
- `texas` has a usable rank-preserving point at `256`, with small MAE/sign gains.

## 10. Runtime

Mean over successful rows:

| Field | Mean |
| --- | ---: |
| run wallclock sec | 910.6794 |
| partition runtime sec | 0.5352 |
| CoCo train runtime sec | 7.4595 |
| line-graph nodes | 2836.9071 |

Runtime is dominated by the full experiment pipeline, not the partition function itself. However, fine cluster-size specs still noticeably increase end-to-end wall time because they create many groups and downstream influence computations.

## 11. Conclusion

This completed sweep is mostly negative for rank-based influence quality.

Practical conclusions:

1. Do not use CoCo line-graph clustering as a general default for edge-removal influence approximation from this evidence.
2. If the objective is rank preservation, use coarse specs only: `edge_cluster_size:256` or `edge_cluster_size:512`.
3. If the objective is MAE on heterophilous datasets, smaller specs can help, but they are not rank-safe.
4. Avoid `edge_cluster_size:8` as a default. It is expensive and has the largest average rank loss.
5. `cora_public` is the key blocker. The method degrades all major metrics there.

Recommended next sweep:

- Primary rank-preserving grid: `edge_cluster_size:256,edge_cluster_size:512`
- Optional tradeoff grid: `edge_cluster_size:64,edge_cluster_size:128,edge_cluster_size:256,edge_cluster_size:512`
- Treat `GCN` and `GAT` separately:
  - `GCN`: prefer `512`
  - `GAT`: compare `64` vs `512` depending on whether MAE or rank is the target
- Do not aggregate only by global average. Always report at least dataset, model, layer, ratio, and cluster-size slices.
