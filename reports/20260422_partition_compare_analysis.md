# Partition Compare Analysis

Analyzed artifacts:
- Plan: `instructions/202604211451.md`
- Run script: `run_partition_compare.sh`
- Summary: `results/partition_compare_summary_20260421_173204.csv`
- Table snippets: `reports/20260422_partition_compare_metric_tables.tex`

## 1. What Was Actually Run

The original plan asked for a partition-then-sum group influence experiment across METIS, spectral clustering, and local PPR, ideally over multiple candidate sizes and multiple `K`.

The actual run made a few pragmatic simplifications:

- Profile: `small`
- Methods: `metis`, `spectral`, `local_ppr`
- `K`: fixed to `3` for all runs
- Candidate size control: ratios `1,5,10,30,50,80` instead of absolute group sizes like `10,20,50`
- Candidate count: fixed `50`
- Edit type: edge removal only
- Models: `GCN`, `GAT`
- Layers: `2`, `4`
- Datasets: `cora_public`, `citeseer_public`, `texas`, `cornell`
- Total runs: `288`
- Successful runs: `288 / 288`

So this is a clean, completed ablation over 3 partition methods, but not yet the full multi-`K` study described in the plan.

## 2. Main Quantitative Result

The main question from the plan was whether cluster-sum can approximate whole-group influence better than the one-shot whole-group estimate, using PBRF as the reference target.

Overall answer from this run:

- Cluster-sum does **not** clearly dominate the existing whole-group baseline.
- `METIS` is the strongest partition method in this experiment.
- `spectral` is roughly neutral.
- `local_ppr` is the weakest tradeoff: no accuracy gain overall and the worst correlation drop.

### Overall averages across 96 runs per method

| Method | Baseline vs PBRF MAE | Cluster vs PBRF MAE | MAE delta | Baseline Spearman | Cluster Spearman | Spearman delta | Baseline sign acc | Cluster sign acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metis | 1.1226 | 1.1185 | -0.0041 | 0.5415 | 0.5284 | -0.0131 | 0.7326 | 0.7349 |
| spectral | 1.1226 | 1.1223 | -0.0003 | 0.5414 | 0.5298 | -0.0115 | 0.7322 | 0.7305 |
| local_ppr | 1.1226 | 1.1257 | +0.0031 | 0.5414 | 0.5161 | -0.0253 | 0.7322 | 0.7295 |

Interpretation:

- `metis` gives the best average MAE and a tiny sign-accuracy gain, but ranking quality still drops.
- `spectral` is almost identical to the baseline in MAE, but again loses ranking quality.
- `local_ppr` is worse than the baseline in both MAE and Spearman on average.

Another important point: cluster-sum usually stayed close to the baseline estimate itself.

- Mean `baseline_cluster_mae`: `0.0319` for `metis`
- Mean `baseline_cluster_mae`: `0.0307` for `spectral`
- Mean `baseline_cluster_mae`: `0.0371` for `local_ppr`

This is much smaller than the average baseline-vs-PBRF error (`1.1226`), so partitioning changed the estimate only modestly in most cases.

## 3. When Partitioning Helps

The modest gains show up mostly at larger group sizes, and mostly in MAE rather than correlation.

### By ratio

- Ratios `1` and `5`: partitioning is basically unnecessary. MAE usually gets slightly worse or stays flat.
- Ratio `10`: almost a tie across methods.
- Ratios `30` and `50`: `metis` and `spectral` show small average MAE improvements over the whole-group baseline.
- Ratio `80`: only `metis` keeps a small average MAE gain; `spectral` and `local_ppr` become worse than the baseline.

Concrete examples:

- Best gains were concentrated in `texas`, especially `GAT`, `4-layer`, high-ratio cases.
- The largest MAE improvements for all three methods appeared at `texas + GAT + 4-layer + ratio 80`.
- The worst regressions tended to appear in `cora_public` or `cornell` high-ratio cases, especially for `local_ppr` and sometimes `spectral`.

So the observed benefit is not broad-based. It is concentrated in a subset of high-ratio, small-graph settings.

## 4. Correlation Story

This experiment does **not** support the claim that partitioning improves ranking fidelity.

- Baseline Spearman is around `0.541`
- All three partition methods reduce it
- `local_ppr` drops the most

This matters because if the downstream goal is to rank or choose groups by influence, a lower Spearman can be more damaging than a tiny MAE gain.

In short:

- For scalar approximation error, `metis` can help a little at large ratios.
- For ranking consistency, the original whole-group baseline is still safer.

## 5. Runtime Story

Runtime is where the methods separate more clearly.

### Average wall-clock per run

- `metis`: `26.05` min
- `spectral`: `9.65` min
- `local_ppr`: `10.47` min

### Mean partition time per candidate

At small ratios, partition cost is negligible:

- ratio `1`: `0.017s` metis, `0.047s` spectral, `0.029s` local_ppr

At large ratios, it becomes a major cost:

| Ratio | metis | spectral | local_ppr |
|---|---:|---:|---:|
| 30 | 3.65s | 4.56s | 5.58s |
| 50 | 9.38s | 11.17s | 15.22s |
| 80 | 24.47s | 38.65s | 39.50s |

Because each run uses 50 candidates, ratio `80` implies rough partition-only overheads of:

- `metis`: about `20.4` min/run
- `spectral`: about `32.2` min/run
- `local_ppr`: about `32.9` min/run

This matches the observed long-tail behavior during the run. Large-ratio `local_ppr` was especially expensive on `cora_public` and `citeseer_public`.

## 6. Practical Conclusion

Given the current setup, the best reading is:

1. Partition-then-sum is a viable approximation, but only marginally different from the existing whole-group baseline.
2. `METIS` is the only method with a clear accuracy-side signal under fixed `K=3`, although it is not the fastest end-to-end in wall-clock time.
3. `spectral` is acceptable as a neutral comparison point: faster overall than `METIS`, but without a clear accuracy win.
4. `local_ppr` is not justified in this configuration. It is slower at high ratios and worse on the most important aggregate metrics.

## 7. Recommended Next Experiments

To align better with the original plan and make the next result more decisive:

1. Keep `metis` as the primary partition method and treat `spectral` as the secondary comparator.
2. Drop `local_ppr` from the main grid, or keep it only as an ablation on a reduced subset.
3. Vary `K`, because the current study fixes `K=3` and therefore does not test the main design degree of freedom from the plan.
4. Focus the next sweep on the regime where partitioning has a chance to matter:
   - higher ratios
   - `GAT`
   - deeper models
   - `cora_public` and `citeseer_public` for stress-testing
5. If the downstream use case is ranking/selecting groups, prioritize Spearman and sign accuracy more heavily than MAE.

## 8. Bottom Line

This run shows that:

- cluster-sum with graph partitioning is technically stable and fully runnable,
- its estimates stay close to the original whole-group influence,
- but it does **not** yet provide a decisive improvement over the baseline,
- and among the tested methods, `METIS` is the only one with a clear aggregate accuracy benefit, while `spectral` is the main speed-oriented comparator.
