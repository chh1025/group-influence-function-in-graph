# METIS K-Sweep Analysis

Analyzed artifacts:
- Summary: `results/partition_compare_summary_metis_k_sweep_high_ratio_gcn_gat4_20260422_131508.csv`
- Log note: `codex_log/20260423_metis_k_sweep_high_ratio_gcn_gat4.md`
- Table snippets: `reports/20260423_metis_k_sweep_high_ratio_gcn_gat4_tables.tex`
- Expanded table snippets: `reports/20260423_metis_k_sweep_high_ratio_gcn_gat4_big_tables.tex`

## 1. What Was Run

This follow-up experiment isolates `METIS` and sweeps the number of clusters `K`.

- Partition method: `metis`
- `K`: `1, 2, 3, 4, 6, 8`
- Ratios: `30, 50, 80`
- Models: `GCN`, `GAT`
- Layers: `4`
- Datasets: `cora_public`, `citeseer_public`, `texas`, `cornell`
- Total runs: `144`
- Successful runs: `144 / 144`

## 2. Topline

This sweep gives a clean trade-off:

- Larger `K` improves magnitude-style metrics:
  - MAE improves monotonically
  - MAPE improves monotonically
  - sign accuracy improves monotonically
- Larger `K` hurts ranking-style metrics:
  - Spearman drops overall
  - Pearson drops overall

So the right `K` depends on the objective:

- ranking-oriented: keep `K` small
- error-oriented: use larger `K`

## 3. Overall K Summary

| K | Baseline MAE | Cluster MAE | Baseline Spearman | Cluster Spearman | Baseline Pearson | Cluster Pearson | Baseline Sign | Cluster Sign | Baseline MAPE | Cluster MAPE | Runtime (min) | Partition sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.1914 | 4.1914 | 0.4101 | 0.4101 | 0.3885 | 0.3885 | 0.7481 | 0.7481 | 2.8578 | 2.8578 | 13.73 | 12.803 |
| 2 | 4.1914 | 4.1795 | 0.4101 | 0.4017 | 0.3885 | 0.3845 | 0.7481 | 0.7494 | 2.8580 | 2.5913 | 13.82 | 12.981 |
| 3 | 4.1914 | 4.1726 | 0.4101 | 0.3976 | 0.3885 | 0.3782 | 0.7481 | 0.7591 | 2.8578 | 2.5067 | 14.02 | 13.072 |
| 4 | 4.1914 | 4.1677 | 0.4101 | 0.3680 | 0.3885 | 0.3633 | 0.7481 | 0.7635 | 2.8580 | 2.4316 | 14.32 | 13.429 |
| 6 | 4.1914 | 4.1571 | 0.4101 | 0.3719 | 0.3885 | 0.3547 | 0.7481 | 0.7645 | 2.8579 | 2.3448 | 14.24 | 13.353 |
| 8 | 4.1914 | 4.1477 | 0.4101 | 0.3632 | 0.3885 | 0.3559 | 0.7481 | 0.7826 | 2.8580 | 2.2910 | 14.47 | 13.558 |

### Readout

- `K=1` reproduces the whole-group baseline exactly, which is the expected sanity check.
- Best partitioned Spearman: `K=2`
- Best partitioned MAE: `K=8`
- Best partitioned MAPE: `K=8`
- Best partitioned sign accuracy: `K=8`

There is no single dominant `K`.

## 4. Model-Wise Tables

### GCN-4

| K | Baseline MAE | Cluster MAE | Baseline Spearman | Cluster Spearman | Baseline Sign | Cluster Sign | Baseline MAPE | Cluster MAPE | Runtime (min) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1483 | 0.1483 | 0.3695 | 0.3695 | 0.6850 | 0.6850 | 3.3537 | 3.3537 | 13.55 |
| 2 | 0.1483 | 0.1505 | 0.3695 | 0.3747 | 0.6850 | 0.6750 | 3.3537 | 3.2879 | 13.30 |
| 3 | 0.1483 | 0.1431 | 0.3695 | 0.3758 | 0.6850 | 0.6917 | 3.3537 | 3.0639 | 13.73 |
| 4 | 0.1483 | 0.1484 | 0.3695 | 0.3582 | 0.6850 | 0.7000 | 3.3536 | 3.1330 | 14.26 |
| 6 | 0.1483 | 0.1457 | 0.3695 | 0.3547 | 0.6850 | 0.6800 | 3.3535 | 3.0645 | 13.98 |
| 8 | 0.1483 | 0.1459 | 0.3695 | 0.3566 | 0.6850 | 0.7000 | 3.3537 | 3.0805 | 15.24 |

GCN is relatively well-behaved under moderate partitioning:

- `K=3` is the strongest compromise for GCN
- `K=2` and `K=3` slightly improve Spearman
- pushing to large `K` no longer helps ranking

### GAT-4

| K | Baseline MAE | Cluster MAE | Baseline Spearman | Cluster Spearman | Baseline Sign | Cluster Sign | Baseline MAPE | Cluster MAPE | Runtime (min) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.2346 | 8.2346 | 0.4507 | 0.4507 | 0.8113 | 0.8113 | 2.3618 | 2.3618 | 13.91 |
| 2 | 8.2346 | 8.2085 | 0.4507 | 0.4287 | 0.8113 | 0.8238 | 2.3624 | 1.8948 | 14.33 |
| 3 | 8.2346 | 8.2020 | 0.4507 | 0.4193 | 0.8113 | 0.8265 | 2.3618 | 1.9495 | 14.30 |
| 4 | 8.2346 | 8.1871 | 0.4507 | 0.3777 | 0.8113 | 0.8270 | 2.3624 | 1.7302 | 14.38 |
| 6 | 8.2346 | 8.1686 | 0.4507 | 0.3891 | 0.8113 | 0.8489 | 2.3624 | 1.6250 | 14.51 |
| 8 | 8.2346 | 8.1495 | 0.4507 | 0.3699 | 0.8113 | 0.8652 | 2.3624 | 1.5014 | 13.70 |

GAT shows a much sharper trade-off:

- MAE, MAPE, and sign accuracy all improve as `K` grows
- Spearman degrades for every `K > 1`
- for ranking, GAT prefers `K=1`
- for error minimization, GAT prefers `K=8`

## 5. Ratio-Wise Tables

| Ratio | K | Baseline MAE | Cluster MAE | Baseline Spearman | Cluster Spearman | Partition sec |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 1 | 0.3557 | 0.3557 | 0.4621 | 0.4621 | 3.993 |
| 30 | 2 | 0.3557 | 0.3483 | 0.4621 | 0.4464 | 3.767 |
| 30 | 3 | 0.3557 | 0.3442 | 0.4621 | 0.4355 | 4.123 |
| 30 | 4 | 0.3557 | 0.3406 | 0.4621 | 0.4297 | 4.371 |
| 30 | 6 | 0.3557 | 0.3336 | 0.4621 | 0.4591 | 4.810 |
| 30 | 8 | 0.3557 | 0.3300 | 0.4621 | 0.4552 | 4.136 |
| 50 | 1 | 3.3982 | 3.3982 | 0.4336 | 0.4336 | 10.336 |
| 50 | 2 | 3.3982 | 3.3864 | 0.4336 | 0.4395 | 11.411 |
| 50 | 3 | 3.3982 | 3.3730 | 0.4336 | 0.4336 | 9.760 |
| 50 | 4 | 3.3982 | 3.3720 | 0.4336 | 0.4125 | 11.287 |
| 50 | 6 | 3.3982 | 3.3630 | 0.4336 | 0.3990 | 9.628 |
| 50 | 8 | 3.3982 | 3.3529 | 0.4336 | 0.3893 | 9.572 |
| 80 | 1 | 8.8203 | 8.8203 | 0.3346 | 0.3346 | 24.080 |
| 80 | 2 | 8.8203 | 8.8037 | 0.3346 | 0.3192 | 23.764 |
| 80 | 3 | 8.8203 | 8.8005 | 0.3346 | 0.3236 | 25.333 |
| 80 | 4 | 8.8203 | 8.7906 | 0.3346 | 0.2617 | 24.628 |
| 80 | 6 | 8.8203 | 8.7747 | 0.3346 | 0.2575 | 25.621 |
| 80 | 8 | 8.8203 | 8.7602 | 0.3346 | 0.2452 | 26.965 |

### Readout

- Ratio `30`: larger `K` helps MAE and only mildly hurts Spearman
- Ratio `50`: `K=2` is the safest compromise
- Ratio `80`: larger `K` helps MAE, but Spearman collapses most strongly

## 6. Dataset-Wise Tables

| Dataset | K | Baseline MAE | Cluster MAE | Baseline Spearman | Cluster Spearman | Runtime (min) |
|---|---:|---:|---:|---:|---:|---:|
| cora_public | 1 | 0.0976 | 0.0976 | 0.6736 | 0.6736 | 30.96 |
| cora_public | 2 | 0.0976 | 0.0995 | 0.6736 | 0.6564 | 30.37 |
| cora_public | 3 | 0.0976 | 0.1024 | 0.6736 | 0.6539 | 31.40 |
| cora_public | 4 | 0.0976 | 0.0986 | 0.6736 | 0.6496 | 31.98 |
| cora_public | 6 | 0.0976 | 0.1073 | 0.6736 | 0.6426 | 31.72 |
| cora_public | 8 | 0.0976 | 0.1032 | 0.6736 | 0.6236 | 30.17 |
| citeseer_public | 1 | 0.1770 | 0.1770 | 0.4824 | 0.4824 | 21.72 |
| citeseer_public | 2 | 0.1770 | 0.1766 | 0.4824 | 0.4892 | 22.58 |
| citeseer_public | 3 | 0.1770 | 0.1779 | 0.4824 | 0.4822 | 22.18 |
| citeseer_public | 4 | 0.1770 | 0.1740 | 0.4824 | 0.4702 | 22.75 |
| citeseer_public | 6 | 0.1770 | 0.1757 | 0.4824 | 0.4763 | 22.80 |
| citeseer_public | 8 | 0.1770 | 0.1736 | 0.4824 | 0.4288 | 25.03 |
| texas | 1 | 14.8752 | 14.8752 | 0.2332 | 0.2332 | 1.16 |
| texas | 2 | 14.8752 | 14.8403 | 0.2332 | 0.2072 | 1.16 |
| texas | 3 | 14.8752 | 14.7644 | 0.2332 | 0.1924 | 1.26 |
| texas | 4 | 14.8752 | 14.7961 | 0.2332 | 0.1453 | 1.30 |
| texas | 6 | 14.8752 | 14.7585 | 0.2332 | 0.1585 | 1.21 |
| texas | 8 | 14.8752 | 14.7512 | 0.2332 | 0.1660 | 1.40 |
| cornell | 1 | 1.6158 | 1.6158 | 0.2513 | 0.2513 | 1.08 |
| cornell | 2 | 1.6158 | 1.6015 | 0.2513 | 0.2540 | 1.15 |
| cornell | 3 | 1.6158 | 1.6456 | 0.2513 | 0.2618 | 1.23 |
| cornell | 4 | 1.6158 | 1.6022 | 0.2513 | 0.2068 | 1.24 |
| cornell | 6 | 1.6158 | 1.5869 | 0.2513 | 0.2102 | 1.25 |
| cornell | 8 | 1.6158 | 1.5627 | 0.2513 | 0.2345 | 1.27 |

### Dataset Best-K Summary

| Dataset | Best K by MAE | Best K by Spearman | Best K by Sign Accuracy |
|---|---:|---:|---:|
| Cora | 1 | 1 | 4 |
| CiteSeer | 8 | 2 | 4 |
| Texas | 8 | 1 | 3 |
| Cornell | 8 | 3 | 8 |

This is the clearest evidence that no universal `K` exists:

- `Cora` prefers no partitioning
- `CiteSeer` likes small `K` for ranking, large `K` for error
- `Texas` is strongly error-vs-ranking split
- `Cornell` likes intermediate `K` for ranking

## 7. Best-K Frequency Table

Across the 24 `(model, dataset, ratio)` combinations:

| Metric | K=1 | K=2 | K=3 | K=4 | K=6 | K=8 |
|---|---:|---:|---:|---:|---:|---:|
| Best MAE | 5 | 2 | 2 | 3 | 2 | 10 |
| Best Spearman | 8 | 6 | 4 | 1 | 3 | 2 |
| Best Sign Accuracy | 11 | 2 | 3 | 4 | 1 | 3 |
| Best MAPE | 2 | 1 | 3 | 6 | 2 | 10 |

This makes the objective split explicit:

- `K=8` dominates error minimization
- `K=1` or `K=2` dominate ranking preservation

## 8. Practical Conclusion

This sweep supports a branched recommendation rather than one fixed `K`.

If the main goal is ranking:

- keep `K=1`
- if partitioning must be used, choose `K=2`

If the main goal is scalar approximation error:

- choose `K=6` or `K=8`

If one shared compromise is needed:

- `K=2` is the safest global choice
- `K=3` is the best GCN-centered choice

## 9. Bottom Line

The `METIS` K-sweep clarified the earlier ambiguity:

- partitioning can improve error metrics,
- but the improvement comes from larger `K`,
- and larger `K` generally damages ranking quality.

So the next experiment should not ask “what is the best global `K`?” but instead:

- “which `K` is best for ranking?”
- and separately
- “which `K` is best for scalar approximation?”
