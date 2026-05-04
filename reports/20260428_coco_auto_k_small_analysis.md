# CoCo Auto-K Line-Graph Small Analysis

Analyzed artifacts:

- Summary: `results/partition_compare_summary_coco_auto_k_small_20260428_232419.csv`
- Job dir: `results/partition_compare_jobs/coco_auto_k_small_20260428_232419`
- Launcher: `run_coco_auto_k_small.sh`

## 1. Run Setup

- profile: `small`
- partition method: `coco`
- partition strategy: `coco_full_line_graph_assignment`
- element type: `edge_removal`
- auto-k method: `eigengap_silhouette_hybrid`
- auto-k search range: `1..8`
- datasets: `cora_public`, `citeseer_public`, `texas`, `cornell`
- models: `GCN`, `GAT`
- layers: `2`, `4`
- ratios: `30`, `50`, `80`
- candidates per run: `20`
- CoCo diffusion: disabled (`COCO_USE_DIFFUSION=0`)
- total runs: `48`
- successful runs: `48 / 48`

## 2. Topline

Run-level averages:

| Metric | Baseline | CoCo auto-k | Delta |
| --- | ---: | ---: | ---: |
| MAE vs PBRF | 0.4362 | 0.4322 | -0.0040 |
| Spearman vs PBRF | 0.4426 | 0.3975 | -0.0451 |
| Pearson vs PBRF | 0.4581 | 0.4080 | -0.0501 |
| Sign acc. vs PBRF | 0.7113 | 0.7257 | 0.0144 |

Improvement counts across 48 runs:

| Metric | Improved | Total |
| --- | ---: | ---: |
| MAE | 22 | 48 |
| Spearman | 21 | 48 |
| Pearson | 14 | 48 |
| Sign accuracy | 13 | 48 |

Interpretation:

- CoCo auto-k slightly improves average MAE and sign accuracy.
- Ranking quality drops clearly: Spearman `-0.0451`, Pearson `-0.0501`.
- The MAE gain is not broad: only `22 / 48` runs improve.
- The sign-accuracy average improves, but only `13 / 48` individual runs improve, so the gain is concentrated in a few slices.

## 3. Auto-K Choices

| Dataset | K | Fallback | Base MAE | CoCo MAE | MAE d | Base Sp | CoCo Sp | Sp d | Base Sign | CoCo Sign | Sign d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| citeseer_public | 6.0000 | 0.0000 | 0.1475 | 0.1482 | 0.0008 | 0.5818 | 0.5188 | -0.0630 | 0.9417 | 0.9500 | 0.0083 |
| cora_public | 8.0000 | 0.0000 | 0.0810 | 0.0868 | 0.0058 | 0.6665 | 0.5941 | -0.0724 | 0.9000 | 0.9417 | 0.0417 |
| cornell | 6.0000 | 0.8000 | 0.4439 | 0.4513 | 0.0073 | 0.2952 | 0.3023 | 0.0071 | 0.4847 | 0.4736 | -0.0111 |
| texas | 2.0000 | 1.0000 | 1.0724 | 1.0426 | -0.0298 | 0.2268 | 0.1748 | -0.0520 | 0.5187 | 0.5375 | 0.0187 |

Selected global line-graph `k`:

- `cora_public`: `8`
- `citeseer_public`: `6`
- `cornell`: `6`
- `texas`: `2`

Important caveat:

- `cora_public` and `citeseer_public` used full line-graph assignment cleanly.
- `cornell` fell back to candidate-level CoCo for `80%` of candidates.
- `texas` fell back to candidate-level CoCo for `100%` of candidates.

The cause was self-loop handling. `texas` has `16` unique self-loops and `cornell` has `3`; the full line graph excluded self-loops, while edge-removal candidates could still contain them. After this analysis, `line_graph_coco.py` was patched so full line-graph construction includes self-loops and line-graph affinity construction avoids self-diagonal artifacts.

Because of this caveat, the `texas/cornell` rows should not be read as pure full-line-graph assignment results from this completed run.

## 4. Full-Assignment vs Fallback-Heavy

| Mode | Base MAE | CoCo MAE | MAE d | Base Sp | CoCo Sp | Sp d | Base Sign | CoCo Sign | Sign d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full-assignment (`cora`, `citeseer`) | 0.1143 | 0.1175 | +0.0033 | 0.6242 | 0.5565 | -0.0677 | 0.9208 | 0.9458 | +0.0250 |
| fallback-heavy (`texas`, `cornell`) | 0.7582 | 0.7470 | -0.0112 | 0.2610 | 0.2385 | -0.0225 | 0.5017 | 0.5056 | +0.0038 |

The clean full-assignment subset is not encouraging for rank quality. It improves sign accuracy but loses MAE and Spearman.

## 5. Ratio Split

| Ratio | Base MAE | CoCo MAE | MAE d | Base Sp | CoCo Sp | Sp d | Base Sign | CoCo Sign | Sign d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 0.1683 | 0.1617 | -0.0065 | 0.5029 | 0.5031 | +0.0002 | 0.7312 | 0.7625 | +0.0312 |
| 50 | 0.3314 | 0.3264 | -0.0050 | 0.4787 | 0.4419 | -0.0367 | 0.7219 | 0.7250 | +0.0031 |
| 80 | 0.8090 | 0.8086 | -0.0004 | 0.3462 | 0.2474 | -0.0987 | 0.6807 | 0.6896 | +0.0089 |

The ratio-30 regime is the only setting where rank quality is approximately preserved. At ratio 80, Spearman drops sharply.

## 6. Model/Layer Split

| Model | Layer | Base MAE | CoCo MAE | MAE d | Base Sp | CoCo Sp | Sp d | Base Sign | CoCo Sign | Sign d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GAT | 2 | 0.3488 | 0.3660 | +0.0172 | 0.3506 | 0.2585 | -0.0921 | 0.6833 | 0.6500 | -0.0333 |
| GAT | 4 | 1.1721 | 1.1429 | -0.0292 | 0.4328 | 0.3456 | -0.0872 | 0.8118 | 0.8569 | +0.0451 |
| GCN | 2 | 0.0734 | 0.0751 | +0.0017 | 0.6659 | 0.6147 | -0.0513 | 0.6667 | 0.7042 | +0.0375 |
| GCN | 4 | 0.1506 | 0.1449 | -0.0057 | 0.3211 | 0.3712 | +0.0501 | 0.6833 | 0.6917 | +0.0083 |

Best slice:

- `GCN layer 4` is the cleanest slice: MAE, Spearman, and sign accuracy all improve.

Weak slice:

- `GAT layer 2` gets worse across MAE, Spearman, and sign accuracy.

## 7. Runtime

| Field | Mean |
| --- | ---: |
| run_wallclock_sec | 134.063 |
| mean_partition_runtime_sec | 1.935 |
| mean_partition_coco_train_runtime_sec | 77.438 |
| mean_partition_coco_line_graph_num_nodes | 2542.883 |

The mean end-to-end run time is about `2.23` minutes per run. With the 4-GPU scheduler this small profile is manageable.

## 8. Conclusion

The auto-k selector is technically working and produces reasonable dataset-scale values, but the current CoCo line-graph approximation should not be promoted to the main default.

Current read:

- It can slightly improve MAE and sign accuracy.
- It consistently risks degrading rank metrics.
- The only broadly positive model/layer slice is `GCN layer 4`.
- `ratio=80` is especially risky for Spearman.
- The completed `texas/cornell` results are mixed with candidate-level fallback due to the self-loop issue in the run code.

Recommended next step:

1. Rerun `texas/cornell` only after the self-loop patch if we want a clean full-line-graph auto-k comparison.
2. Keep `cora/citeseer` results as valid full-assignment evidence; those results do not support CoCo auto-k as a default because Spearman drops strongly.
3. If continuing, narrow the next validation to:
   - `GCN`
   - `layer=4`
   - ratios `30,50`
   - clean full-line-graph assignment after the self-loop patch
