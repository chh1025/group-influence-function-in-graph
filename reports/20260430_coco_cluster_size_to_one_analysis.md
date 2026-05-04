# CoCo Cluster-Size-to-One Continuation Analysis

Analyzed artifacts:

- New summary: `results/partition_compare_summary_coco_cluster_size_to_one_20260430_110841.csv`
- Previous summary for comparison: `results/partition_compare_summary_20260429_020401.csv`
- Launcher: `run_coco_cluster_size_to_one_small.sh`

## 1. Run Setup

This run continued the previous manual `edge_cluster_size` sweep beyond `512`.

- datasets: `cora_public`, `citeseer_public`
- models: `GCN`, `GAT`
- layers: `2`, `4`
- ratios: `1`, `10`, `30`, `50`, `80`
- cluster specs: `edge_cluster_size:{1024,2048,4096,8192}`
- partition method: `coco`
- partition strategy: `coco_full_line_graph_assignment`
- element type: `edge_removal`
- auto-k method: `none`
- removal candidates per run: `100`
- total runs: `160`
- successful runs: `160 / 160`

Effective `K`:

| Dataset | Full Edges | 512 | 1024 | 2048 | 4096 | 8192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cora_public` | 5278 | 11 | 6 | 3 | 2 | 1 |
| `citeseer_public` | 4552 | 9 | 5 | 3 | 2 | 1 |

So this continuation did reach the one-cluster condition for both remaining datasets.

## 2. New-Run Topline

Successful-run averages over the 160 new rows:

| Metric | Baseline | CoCo | Delta |
| --- | ---: | ---: | ---: |
| MAE vs PBRF | 0.0702 | 0.0718 | +0.0016 |
| Spearman vs PBRF | 0.6661 | 0.6554 | -0.0106 |
| Pearson vs PBRF | 0.6910 | 0.6813 | -0.0098 |
| Sign accuracy vs PBRF | 0.8615 | 0.8609 | -0.0006 |

The new coarse range is much less harmful than the earlier fine/coarse range up to `512`, but it still does not improve the aggregate metrics.

## 3. By Cluster Size

| Edge Cluster Size | Runs | Mean K | Mean Groups | MAE d | Spearman d | Pearson d | Sign d | Mean Wall Sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 40 | 5.50 | 5.4590 | +0.0041 | -0.0345 | -0.0306 | -0.0032 | 472.39 |
| 2048 | 40 | 3.00 | 2.9870 | +0.0012 | -0.0017 | -0.0035 | -0.0005 | 440.12 |
| 4096 | 40 | 2.00 | 1.9990 | +0.0010 | -0.0063 | -0.0050 | +0.0013 | 433.93 |
| 8192 | 40 | 1.00 | 1.0000 | +0.0000 | +0.0000 | -0.0000 | +0.0000 | 412.44 |

Interpretation:

- `edge_cluster_size:8192` is exactly the `K=1` control. It produces zero deltas because it reduces to a single group.
- `edge_cluster_size:1024` is still too many clusters for this coarse region, especially on Cora.
- `edge_cluster_size:2048` is the best nontrivial coarse setting by rank preservation.
- `edge_cluster_size:4096` is also close to neutral, with slightly better sign accuracy but worse Spearman than `2048`.

## 4. By Dataset

| Dataset | Runs | MAE d | Spearman d | Pearson d | Sign d | Mean K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `citeseer_public` | 80 | +0.0015 | -0.0057 | -0.0061 | -0.0026 | 2.75 |
| `cora_public` | 80 | +0.0016 | -0.0155 | -0.0135 | +0.0014 | 3.00 |

Cora still loses more rank than Citeseer, but the damage is much smaller than at `512` and below.

## 5. Previous 512 Plus New Coarse Sweep

This combines the previous `edge_cluster_size:512` rows with the new `1024,2048,4096,8192` rows.

| Dataset | Size | K | MAE d | Spearman d | Pearson d | Sign d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `citeseer_public` | 512 | 9 | +0.0022 | -0.0230 | -0.0221 | +0.0075 |
| `citeseer_public` | 1024 | 5 | +0.0033 | -0.0112 | -0.0104 | -0.0025 |
| `citeseer_public` | 2048 | 3 | +0.0012 | +0.0003 | -0.0037 | -0.0100 |
| `citeseer_public` | 4096 | 2 | +0.0016 | -0.0120 | -0.0102 | +0.0020 |
| `citeseer_public` | 8192 | 1 | +0.0000 | +0.0000 | -0.0000 | +0.0000 |
| `cora_public` | 512 | 11 | +0.0074 | -0.0571 | -0.0535 | +0.0055 |
| `cora_public` | 1024 | 6 | +0.0050 | -0.0578 | -0.0508 | -0.0040 |
| `cora_public` | 2048 | 3 | +0.0011 | -0.0036 | -0.0033 | +0.0090 |
| `cora_public` | 4096 | 2 | +0.0004 | -0.0006 | +0.0002 | +0.0005 |
| `cora_public` | 8192 | 1 | -0.0000 | +0.0000 | +0.0000 | +0.0000 |

The important change is Cora:

- `512` and `1024` are bad for rank.
- `2048` and `4096` almost remove the rank loss.
- `8192` is neutral because `K=1`.

Citeseer is more mixed:

- `2048` gives the best average Spearman delta, but sign drops.
- `4096` gives a small sign gain, but Spearman drops.
- `8192` is neutral.

## 6. Ratio-Specific Reads

### Citeseer Spearman Delta, 512+

| Ratio | 512 | 1024 | 2048 | 4096 | 8192 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.0046 | -0.0033 | -0.0015 | +0.0007 | +0.0000 |
| 10 | +0.0002 | -0.0117 | -0.0011 | -0.0007 | +0.0000 |
| 30 | -0.0016 | -0.0107 | -0.0034 | -0.0071 | +0.0000 |
| 50 | -0.0140 | -0.0019 | -0.0106 | -0.0158 | +0.0000 |
| 80 | -0.0952 | -0.0284 | +0.0181 | -0.0372 | +0.0000 |

### Cora Spearman Delta, 512+

| Ratio | 512 | 1024 | 2048 | 4096 | 8192 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.0065 | -0.0076 | -0.0020 | -0.0010 | +0.0000 |
| 10 | -0.0211 | -0.0259 | +0.0068 | +0.0044 | +0.0000 |
| 30 | -0.0512 | -0.0668 | -0.0204 | -0.0029 | +0.0000 |
| 50 | -0.0785 | -0.0652 | -0.0072 | -0.0005 | +0.0000 |
| 80 | -0.1283 | -0.1236 | +0.0046 | -0.0031 | +0.0000 |

Ratio read:

- High ratios were where the previous `512` result was most damaging.
- Moving to `2048` or `4096` fixes most of the Cora rank degradation.
- Citeseer ratio `80` has a real positive Spearman point at `2048`, but with MAE increase and sign drop.
- `8192` is always neutral, so it is not a useful improving condition.

## 7. Conclusion

The missing only-one-cluster range has now been covered.

Main conclusions:

1. `K=1` is a neutral control, not a useful CoCo setting. It produces zero deltas.
2. For Cora/Citeseer, the useful coarse nontrivial range is around `K=2` or `K=3`.
3. `edge_cluster_size:1024` should not be used; it preserves too much of the earlier rank loss.
4. If continuing with coarse manual specs:
   - `cora_public`: use `2048` or `4096`
   - `citeseer_public`: use `2048` if rank is the priority, `4096` if sign stability is the priority
5. The overall read remains conservative: CoCo line-graph clustering does not clearly improve Cora/Citeseer, but very coarse `K=2/3` avoids most of the damage.

Recommended next grid for Cora/Citeseer only:

```text
edge_cluster_size:2048,edge_cluster_size:4096
```

Do not include `8192` in the main comparison unless a no-op/control baseline is explicitly needed.
