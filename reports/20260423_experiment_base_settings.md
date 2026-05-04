# Experiment Base Settings

Reference source:
- `run_experiments.sh`

This note keeps only the two base settings that are worth documenting repeatedly:

1. `small` / `full` profile definitions
2. evaluation metrics used in the recent analysis tables

## 1. Profile Definitions

### `full`

- models: `sgc`, `gcn`, `gat`, `chebnet`
- layers: `2`, `4`, `6`, `12`
- datasets: `cora`, `citeseer`, `pubmed`, `computers`, `photo`, `chameleon`, `actor`, `squirrel`, `texas`, `cornell`

### `small`

- models: `gcn`, `gat`
- layers: `2`, `4`
- datasets: `cora`, `citeseer`, `texas`, `cornell`

### Shared default

- default profile when omitted: `full`

## 2. Metrics Used

The main comparison metrics used in the recent partition / K-sweep analysis are:

### Spearman correlation

- used to measure ranking consistency
- larger is better

### Pearson correlation

- used to measure linear correlation
- larger is better

### MAE

- mean absolute error
- used to measure absolute prediction error
- smaller is better

### MAPE

- mean absolute percentage error
- used to measure relative prediction error
- smaller is better

For the recent reports, these four metrics are the main ones worth prioritizing when comparing settings.
