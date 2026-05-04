# CoCo Dataset x Ratio x Cluster-Size Analysis

Source:

- `results/partition_compare_summary_20260429_020401.csv`

Rows:

- total: `504`
- successful: `503`
- failed/excluded: `1`

The failed row is `GAT / citeseer_public / layer=2 / ratio=30 / edge_cluster_size:8`, caused by `EOFError` while loading a PBRF checkpoint.

This report groups successful rows by:

- dataset
- ratio
- `edge_cluster_size`

There are `126` observed groups, not `140`, because several specs collapse to the same effective `K` at small ratios and are deduplicated by the launcher.

## Best Cluster Size by MAE

Lower `MAE d` is better.

| Dataset | Ratio | Best Size | MAE d | Spearman d | Pearson d | Sign d | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `citeseer_public` | 1 | 8 | -0.0000 | -0.0045 | -0.0025 | +0.0000 | Neutral |
| `citeseer_public` | 10 | 32 | -0.0001 | -0.0146 | -0.0108 | +0.0000 | Tiny MAE gain, rank loss |
| `citeseer_public` | 30 | 512 | +0.0006 | -0.0016 | -0.0033 | +0.0075 | Coarse is safest |
| `citeseer_public` | 50 | 512 | +0.0018 | -0.0140 | -0.0145 | +0.0175 | Coarse is safest |
| `citeseer_public` | 80 | 512 | +0.0085 | -0.0952 | -0.0885 | +0.0075 | Still weak at high ratio |
| `cora_public` | 1 | 512 | +0.0001 | -0.0065 | -0.0017 | -0.0275 | Negative |
| `cora_public` | 10 | 512 | +0.0007 | -0.0211 | -0.0233 | -0.0300 | Negative |
| `cora_public` | 30 | 512 | +0.0021 | -0.0512 | -0.0419 | -0.0175 | Negative |
| `cora_public` | 50 | 256 | +0.0070 | -0.0894 | -0.0980 | +0.0125 | Negative rank |
| `cora_public` | 80 | 256 | +0.0201 | -0.1125 | -0.0956 | +0.0850 | Sign improves, rank poor |
| `cornell` | 1 | 512 | +0.0000 | +0.0000 | -0.0000 | +0.0000 | Neutral |
| `cornell` | 10 | 64 | -0.0062 | -0.0567 | -0.0423 | +0.0228 | MAE/sign gain, rank loss |
| `cornell` | 30 | 16 | -0.0405 | -0.0406 | -0.0205 | +0.0985 | Good MAE/sign, rank loss |
| `cornell` | 50 | 8 | -0.0521 | -0.0311 | +0.0350 | +0.1034 | Best tradeoff for MAE/sign |
| `cornell` | 80 | 8 | -0.1343 | -0.0619 | -0.0373 | +0.1851 | Large MAE/sign gain, rank loss |
| `texas` | 1 | 512 | +0.0000 | +0.0000 | -0.0000 | +0.0000 | Neutral |
| `texas` | 10 | 16 | -0.0023 | -0.0034 | -0.0053 | -0.0100 | Small MAE gain only |
| `texas` | 30 | 16 | -0.0246 | -0.0388 | -0.0204 | -0.0050 | MAE gain, rank/sign loss |
| `texas` | 50 | 16 | -0.0738 | -0.0336 | -0.0516 | -0.0149 | MAE gain, rank/sign loss |
| `texas` | 80 | 32 | -0.0445 | -0.0342 | -0.0198 | +0.0290 | MAE/sign gain, rank loss |

## Best Cluster Size by Spearman

Higher `Spearman d` is better.

| Dataset | Ratio | Best Size | MAE d | Spearman d | Pearson d | Sign d | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `citeseer_public` | 1 | 8 | -0.0000 | -0.0045 | -0.0025 | +0.0000 | No rank-improving spec |
| `citeseer_public` | 10 | 512 | +0.0000 | +0.0002 | -0.0023 | +0.0050 | Near neutral |
| `citeseer_public` | 30 | 512 | +0.0006 | -0.0016 | -0.0033 | +0.0075 | Near neutral |
| `citeseer_public` | 50 | 64 | +0.0024 | +0.0024 | +0.0030 | +0.0125 | Best local slice |
| `citeseer_public` | 80 | 256 | +0.0109 | -0.0559 | -0.0628 | -0.0175 | Rank still bad |
| `cora_public` | 1 | 128 | +0.0002 | -0.0015 | -0.0052 | -0.0400 | No useful improvement |
| `cora_public` | 10 | 512 | +0.0007 | -0.0211 | -0.0233 | -0.0300 | Negative |
| `cora_public` | 30 | 512 | +0.0021 | -0.0512 | -0.0419 | -0.0175 | Negative |
| `cora_public` | 50 | 128 | +0.0111 | -0.0638 | -0.0658 | -0.0175 | Negative |
| `cora_public` | 80 | 256 | +0.0201 | -0.1125 | -0.0956 | +0.0850 | Sign-only gain |
| `cornell` | 1 | 8 | +0.0018 | +0.0029 | +0.0054 | +0.0200 | Small rank/sign gain |
| `cornell` | 10 | 512 | +0.0000 | +0.0000 | -0.0000 | +0.0000 | Coarse neutral |
| `cornell` | 30 | 512 | -0.0000 | +0.0000 | +0.0000 | +0.0000 | Coarse neutral |
| `cornell` | 50 | 512 | -0.0000 | +0.0000 | +0.0000 | +0.0000 | Coarse neutral |
| `cornell` | 80 | 512 | +0.0000 | +0.0000 | -0.0000 | +0.0000 | Coarse neutral |
| `texas` | 1 | 512 | +0.0000 | +0.0000 | -0.0000 | +0.0000 | Neutral |
| `texas` | 10 | 512 | +0.0000 | +0.0000 | -0.0000 | +0.0000 | Neutral |
| `texas` | 30 | 512 | +0.0000 | +0.0000 | -0.0000 | +0.0000 | Neutral |
| `texas` | 50 | 256 | -0.0425 | +0.0149 | +0.0219 | -0.0149 | Rank/MAE useful |
| `texas` | 80 | 256 | -0.0383 | +0.0178 | -0.0291 | +0.0306 | Useful except Pearson |

## Positive-Spec Counts

Counts are out of the number of cluster-size specs actually observed for that dataset/ratio.

| Dataset | Ratio | MAE-Improving Specs | Spearman-Improving Specs | Sign-Improving Specs | Best MAE | Best Spearman | Best Sign |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `citeseer_public` | 1 | 2 | 0 | 1 | 8 | 8 | 256 |
| `citeseer_public` | 10 | 3 | 1 | 3 | 32 | 512 | 64 |
| `citeseer_public` | 30 | 0 | 0 | 5 | 512 | 512 | 32 |
| `citeseer_public` | 50 | 0 | 1 | 4 | 512 | 64 | 512 |
| `citeseer_public` | 80 | 0 | 0 | 3 | 512 | 256 | 64 |
| `cora_public` | 1 | 0 | 0 | 0 | 512 | 128 | 512 |
| `cora_public` | 10 | 0 | 0 | 0 | 512 | 512 | 512 |
| `cora_public` | 30 | 0 | 0 | 0 | 512 | 512 | 512 |
| `cora_public` | 50 | 0 | 0 | 2 | 256 | 128 | 512 |
| `cora_public` | 80 | 0 | 0 | 7 | 256 | 256 | 256 |
| `cornell` | 1 | 0 | 1 | 2 | 512 | 8 | 8 |
| `cornell` | 10 | 5 | 0 | 5 | 64 | 512 | 8 |
| `cornell` | 30 | 6 | 0 | 5 | 16 | 512 | 8 |
| `cornell` | 50 | 6 | 0 | 5 | 8 | 512 | 8 |
| `cornell` | 80 | 3 | 0 | 6 | 8 | 512 | 8 |
| `texas` | 1 | 0 | 0 | 2 | 512 | 512 | 256 |
| `texas` | 10 | 4 | 0 | 0 | 16 | 512 | 512 |
| `texas` | 30 | 6 | 0 | 1 | 16 | 512 | 256 |
| `texas` | 50 | 7 | 1 | 0 | 16 | 256 | 512 |
| `texas` | 80 | 5 | 1 | 6 | 32 | 256 | 64 |

## Dataset-Level Reads

### `citeseer_public`

- Mostly near-neutral at low ratios.
- High ratio `80` is weak for all specs.
- Coarse specs are safest:
  - MAE-safe: `512` for ratios `30`, `50`, `80`
  - rank-safe: `512` at ratio `10`, `512` at ratio `30`, `64` at ratio `50`, `256` at ratio `80`
- Fine specs do not provide enough benefit to justify their runtime.

### `cora_public`

- This is the clearest failure dataset.
- MAE never improves for any dataset/ratio group.
- Spearman never improves for any dataset/ratio group.
- Sign improves only at high ratios, especially `80`, but rank is still poor.
- If CoCo must be used on `cora_public`, choose coarse specs (`256` or `512`) only to minimize damage.

### `cornell`

- Fine specs help MAE and sign, especially at ratios `30`, `50`, `80`.
- `edge_cluster_size:8` is strongest for sign and high-ratio MAE:
  - ratio `50`: MAE d `-0.0521`, sign d `+0.1034`
  - ratio `80`: MAE d `-0.1343`, sign d `+0.1851`
- Rank is usually not improved. The rank-safe choice is `512`, but that is effectively a neutral coarse grouping on this small graph.
- Use `8` only if MAE/sign are the target; use `512` if rank preservation is the target.

### `texas`

- MAE improves for most nontrivial ratios.
- Best MAE:
  - ratio `30`: `16`
  - ratio `50`: `16`
  - ratio `80`: `32`
- Rank-safe choices are coarse:
  - ratio `50`: `256` gives Spearman d `+0.0149`
  - ratio `80`: `256` gives Spearman d `+0.0178`
- `512` is mostly neutral because it becomes extremely coarse.

## Practical Recommendation

If choosing a single rule:

- Avoid `edge_cluster_size:8` as a global default.
- Avoid CoCo on `cora_public` unless the goal is exploratory comparison only.
- Use `edge_cluster_size:256` or `512` for rank preservation.
- Use dataset-specific sizes only if the target metric is explicit:
  - `citeseer_public`: `512` generally, `64` only for ratio `50` rank
  - `cora_public`: `256` or `512`, damage-control only
  - `cornell`: `8` for MAE/sign, `512` for rank/neutral
  - `texas`: `16` or `32` for MAE, `256` for rank at high ratios

For the next sweep, the most defensible grid is:

```text
edge_cluster_size:64,edge_cluster_size:128,edge_cluster_size:256,edge_cluster_size:512
```

Then evaluate with dataset-specific conclusions instead of a single global winner.
