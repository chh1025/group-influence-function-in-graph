#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Usage:
#   ./run_experiments.sh exp1
#   ./run_experiments.sh exp3
#   ./run_experiments.sh exp4
#   ./run_experiments.sh exp5
#   ./run_experiments.sh all
#   ./run_experiments.sh exp1 small
#   ./run_experiments.sh all small
#   ./run_experiments.sh init-exp4-candidates [full|small]
#   ./run_experiments.sh init-exp5-candidates [full|small]
#   ./run_experiments.sh init-cluster-candidates [full|small]
#   ./run_experiments.sh label-cluster-feasibility [full|small] [output_csv]
#   ./run_experiments.sh witness-cluster-feasibility [full|small] [timeout_sec]
#   ./run_experiments.sh list-failed [search_root]
#   ./run_experiments.sh rerun-failed [search_root]
#   ./run_experiments.sh list-constraint-unsatisfied [search_root]
#   ./run_experiments.sh rerun-constraint-unsatisfied <exp1|exp3|exp4|exp5|all> [full|small] [search_root]
#   ./run_experiments.sh list-missing <exp1|exp3|exp4|exp5|all> [search_root]
#   ./run_experiments.sh rerun-missing <exp1|exp3|exp4|exp5|all> [search_root]
#   ./run_experiments.sh list-missing <exp1|exp3|exp4|exp5|all> small [search_root]
#   ./run_experiments.sh rerun-missing <exp1|exp3|exp4|exp5|all> small [search_root]
#   ./run_experiments.sh progress [search_root]
#   ./run_experiments.sh summary <exp1|exp3|exp4|exp5|all> [full|small] [search_root]
#   ./run_experiments.sh summary <exp1|exp3|exp4|exp5|all> small [search_root]
#   ./run_experiments.sh watch-progress [search_root] [interval_sec]
#
# Notes:
# - search_root defaults to multirun/
# - failed run = has .hydra/overrides.yaml but missing both DONE and CONSTRAINT_UNSAT in same run dir.

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "error: neither 'python' nor 'python3' is available. Set PYTHON_BIN explicitly." >&2
  exit 1
fi
SEARCH_ROOT_DEFAULT="multirun"
CONSTRAINT_UNSAT_MARKER_FILE="${CONSTRAINT_UNSAT_MARKER_FILE:-CONSTRAINT_UNSAT}"
CONSTRAINT_UNSAT_ERROR_TEXT="${CONSTRAINT_UNSAT_ERROR_TEXT:-Failed to sample a valid edge group with the current constraints.}"

# Keep OpenMP thread usage predictable for parallel multirun jobs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Ensure submitit worker processes inherit repository-level python customizations.
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${SCRIPT_DIR}"
fi

MODELS_FULL="sgc,gcn,gat,chebnet"
LAYERS_FULL="2,4,6,12"
DATASETS_FULL="cora,citeseer,pubmed,computers,photo,chameleon,actor,squirrel,texas,cornell"
MODELS_SMALL="gcn,gat"
LAYERS_SMALL="2,4"
DATASETS_SMALL="cora,citeseer,texas,cornell"
MODELS="$MODELS_FULL"
LAYERS="$LAYERS_FULL"
DATASETS="$DATASETS_FULL"
SEED_DEFAULT="0"
EXP3_CLUSTER_RATIO_PERCENT="${EXP3_CLUSTER_RATIO_PERCENT:-15}"
EXP3_RETRY_CLUSTER_RATIO_PERCENT="${EXP3_RETRY_CLUSTER_RATIO_PERCENT:-9}"
EXP3_RETRY_MARKER_FILE="${EXP3_RETRY_MARKER_FILE:-$SEARCH_ROOT_DEFAULT/EXP3_RETRY_MARKERS.log}"
# Default for exp4: 15% per run (= 5% per cluster when num_of_clusters=3).
EXP4_CLUSTER_RATIO_PERCENT="${EXP4_CLUSTER_RATIO_PERCENT:-15}"
EXP4_RETRY_CLUSTER_RATIO_PERCENT="${EXP4_RETRY_CLUSTER_RATIO_PERCENT:-$EXP4_CLUSTER_RATIO_PERCENT}"
EXP5_CLUSTER_RATIO_PERCENT="${EXP5_CLUSTER_RATIO_PERCENT:-5}"
EXP5_RETRY_CLUSTER_RATIO_PERCENT="${EXP5_RETRY_CLUSTER_RATIO_PERCENT:-$EXP5_CLUSTER_RATIO_PERCENT}"

apply_profile() {
  local profile="${1:-full}"
  case "$profile" in
    full)
      MODELS="$MODELS_FULL"
      LAYERS="$LAYERS_FULL"
      DATASETS="$DATASETS_FULL"
      ;;
    small)
      MODELS="$MODELS_SMALL"
      LAYERS="$LAYERS_SMALL"
      DATASETS="$DATASETS_SMALL"
      ;;
    *)
      echo "error: invalid profile '$profile' (use: full|small)"
      return 1
      ;;
  esac
}

resolve_dataset_loader_name() {
  local dataset_cfg="$1"
  case "$dataset_cfg" in
    cora)
      echo "cora_public"
      ;;
    citeseer)
      echo "citeseer_public"
      ;;
    pubmed)
      echo "pubmed_public"
      ;;
    *)
      echo "$dataset_cfg"
      ;;
  esac
}

init_cluster_candidate_cache() {
  local dataset_cfg="$1"
  local cluster_ratio_percent="$2"
  local intra_cluster_dist="$3"
  local inter_cluster_dist="$4"
  local force_rebuild="${5:-0}"
  local dataset_name

  dataset_name="$(resolve_dataset_loader_name "$dataset_cfg")"
  echo "[cluster-cache] dataset=$dataset_name ratio=$cluster_ratio_percent intra=$intra_cluster_dist inter=$inter_cluster_dist force=$force_rebuild"
  "$PYTHON_BIN" calculate_influence.py \
    --dataset "$dataset_name" \
    --model GCN \
    --num_layers 2 \
    --element_type edge_removal \
    --experiment_name clusters \
    --num_of_clusters 3 \
    --edges_per_cluster -1 \
    --cluster_ratio_percent "$cluster_ratio_percent" \
    --intra_cluster_dist "$intra_cluster_dist" \
    --inter_cluster_dist "$inter_cluster_dist" \
    --num_removal_candidates 50 \
    --cluster_candidate_init_only 1 \
    --cluster_candidate_force_rebuild "$force_rebuild"
}

init_exp4_candidates() {
  local force_rebuild="${1:-0}"
  local dataset_cfg

  IFS=',' read -r -a dataset_cfgs <<< "$DATASETS"
  for dataset_cfg in "${dataset_cfgs[@]}"; do
    for intra_cluster_dist in 1 2 3; do
      for inter_cluster_dist in 1 2 3; do
        init_cluster_candidate_cache \
          "$dataset_cfg" \
          "$EXP4_CLUSTER_RATIO_PERCENT" \
          "$intra_cluster_dist" \
          "$inter_cluster_dist" \
          "$force_rebuild"
      done
    done
  done
}

init_exp5_candidates() {
  local force_rebuild="${1:-0}"
  local dataset_cfg

  IFS=',' read -r -a dataset_cfgs <<< "$DATASETS"
  for dataset_cfg in "${dataset_cfgs[@]}"; do
    init_cluster_candidate_cache \
      "$dataset_cfg" \
      "$EXP5_CLUSTER_RATIO_PERCENT" \
      1 \
      2 \
      "$force_rebuild"
  done
}

label_cluster_feasibility() {
  local output_csv="${1:-candidate_cache/cluster_feasibility/cluster_feasibility.csv}"

  mkdir -p "$(dirname "$output_csv")"
  DATASETS="$DATASETS" \
    EXP4_CLUSTER_RATIO_PERCENT="$EXP4_CLUSTER_RATIO_PERCENT" \
    EXP5_CLUSTER_RATIO_PERCENT="$EXP5_CLUSTER_RATIO_PERCENT" \
    OUTPUT_CSV="$output_csv" \
    "$PYTHON_BIN" - <<'PY'
import csv
import math
import os
from pathlib import Path
from types import SimpleNamespace

from src import DataLoader
from src.utils import _get_unique_undirected_edges, build_cluster_history_index, label_cluster_feasibility


def resolve_dataset_name(dataset_cfg):
    mapping = {
        "cora": "cora_public",
        "citeseer": "citeseer_public",
        "pubmed": "pubmed_public",
    }
    dataset_cfg = dataset_cfg.strip().lower()
    return mapping.get(dataset_cfg, dataset_cfg)


dataset_cfgs = [x for x in os.environ.get("DATASETS", "").split(",") if x]
exp4_ratio = int(os.environ.get("EXP4_CLUSTER_RATIO_PERCENT", "15"))
exp5_ratio = int(os.environ.get("EXP5_CLUSTER_RATIO_PERCENT", "5"))
output_csv = os.environ["OUTPUT_CSV"]

history_index = build_cluster_history_index(repo_root=".")
rows = []

for dataset_cfg in dataset_cfgs:
    dataset_name = resolve_dataset_name(dataset_cfg)
    data = DataLoader(dataset_name, root="datasets")[0]
    total_edges = int(_get_unique_undirected_edges(data).shape[0])

    exp4_target_total = round(total_edges * (exp4_ratio / 100.0))
    exp4_edges_per_cluster = max(1, math.floor(exp4_target_total / 3))
    for intra_cluster_dist in [1, 2, 3]:
        for inter_cluster_dist in [1, 2, 3]:
            args = SimpleNamespace(
                dataset=dataset_name,
                cluster_ratio_percent=exp4_ratio,
                num_of_clusters=3,
                edges_per_cluster=exp4_edges_per_cluster,
                intra_cluster_dist=intra_cluster_dist,
                inter_cluster_dist=inter_cluster_dist,
                element_type="edge_removal",
                cluster_candidate_cache_root="candidate_cache",
            )
            label_info = label_cluster_feasibility(data, args, history_index=history_index, repo_root=".")
            rows.append(
                {
                    "experiment": "exp4",
                    "dataset": dataset_name,
                    "cluster_ratio_percent": exp4_ratio,
                    "num_of_clusters": 3,
                    "edges_per_cluster": exp4_edges_per_cluster,
                    "num_group_elem": 3 * exp4_edges_per_cluster,
                    "intra_cluster_dist": intra_cluster_dist,
                    "inter_cluster_dist": inter_cluster_dist,
                    **label_info,
                }
            )

    exp5_target_total = round(total_edges * (exp5_ratio / 100.0))
    exp5_edges_per_cluster = max(1, math.floor(exp5_target_total / 3))
    args = SimpleNamespace(
        dataset=dataset_name,
        cluster_ratio_percent=exp5_ratio,
        num_of_clusters=3,
        edges_per_cluster=exp5_edges_per_cluster,
        intra_cluster_dist=1,
        inter_cluster_dist=2,
        element_type="edge_removal",
        cluster_candidate_cache_root="candidate_cache",
    )
    label_info = label_cluster_feasibility(data, args, history_index=history_index, repo_root=".")
    rows.append(
        {
            "experiment": "exp5",
            "dataset": dataset_name,
            "cluster_ratio_percent": exp5_ratio,
            "num_of_clusters": 3,
            "edges_per_cluster": exp5_edges_per_cluster,
            "num_group_elem": 3 * exp5_edges_per_cluster,
            "intra_cluster_dist": 1,
            "inter_cluster_dist": 2,
            **label_info,
        }
    )

fieldnames = [
    "experiment",
    "dataset",
    "cluster_ratio_percent",
    "num_of_clusters",
    "edges_per_cluster",
    "num_group_elem",
    "intra_cluster_dist",
    "inter_cluster_dist",
    "label",
    "reason",
    "evidence_path",
    "cache_exists",
    "candidate_results_path",
    "done_path",
    "constraint_unsat_count",
    "pool_kind",
    "local_pool_count",
    "max_pool_size",
]

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fieldnames})

print(f"[cluster-feasibility] wrote {len(rows)} rows to {output_csv}")
PY
}

witness_cluster_feasibility() {
  local timeout_sec="${1:-90}"

  DATASETS="$DATASETS" \
    EXP4_CLUSTER_RATIO_PERCENT="$EXP4_CLUSTER_RATIO_PERCENT" \
    WITNESS_TIMEOUT_SEC="$timeout_sec" \
    "$PYTHON_BIN" - <<'PY'
import csv
import os
import subprocess
import sys
from pathlib import Path

timeout_sec = int(os.environ.get("WITNESS_TIMEOUT_SEC", "90"))
csv_path = Path("candidate_cache/cluster_feasibility/cluster_feasibility_small.csv")
if not csv_path.exists():
    raise SystemExit("cluster feasibility CSV is missing. Run label-cluster-feasibility first.")

rows = list(csv.DictReader(csv_path.open()))
targets = []
seen = set()
for row in rows:
    if row["experiment"] != "exp4" or row["label"] != "LIKELY_UNSAT":
        continue
    key = (
        row["dataset"],
        int(row["cluster_ratio_percent"]),
        int(row["intra_cluster_dist"]),
        int(row["inter_cluster_dist"]),
        int(row["edges_per_cluster"]),
    )
    if key in seen:
        continue
    seen.add(key)
    targets.append(key)

targets.sort(key=lambda item: (item[4], item[0], item[2], item[3]))
print(f"[witness-cluster] targets={len(targets)} timeout_sec={timeout_sec}")
for idx, (dataset, ratio, intra, inter, edges_per_cluster) in enumerate(targets, start=1):
    print(
        f"[witness-cluster] ({idx}/{len(targets)}) "
        f"dataset={dataset} ratio={ratio} intra={intra} inter={inter} epc={edges_per_cluster}",
        flush=True,
    )
    cmd = [
        sys.executable,
        "calculate_influence.py",
        "--dataset", dataset,
        "--model", "GCN",
        "--num_layers", "2",
        "--element_type", "edge_removal",
        "--experiment_name", "clusters",
        "--num_of_clusters", "3",
        "--edges_per_cluster", str(edges_per_cluster),
        "--cluster_ratio_percent", str(ratio),
        "--intra_cluster_dist", str(intra),
        "--inter_cluster_dist", str(inter),
        "--num_removal_candidates", "1",
        "--cluster_candidate_init_only", "1",
        "--cluster_candidate_force_rebuild", "1",
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_sec)
        print(proc.stdout, end="")
        if proc.returncode == 0:
            print(
                f"[witness-cluster] OK dataset={dataset} intra={intra} inter={inter}",
                flush=True,
            )
        else:
            print(proc.stderr, end="")
            print(
                f"[witness-cluster] FAILED dataset={dataset} intra={intra} inter={inter} rc={proc.returncode}",
                flush=True,
            )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        print(stdout, end="")
        print(stderr, end="")
        print(
            f"[witness-cluster] TIMEOUT dataset={dataset} intra={intra} inter={inter}",
            flush=True,
        )
PY
}

run_exp1() {
  "$PYTHON_BIN" main.py -m \
    experiment=large_drop_influence \
    model="$MODELS" \
    model.num_layers="$LAYERS" \
    dataset="$DATASETS" \
    experiment.ratio_group_elem=1,5,10,20,30,50,70,90 \
    seed=0
}

run_exp3_with_cluster_ratio() {
  local cluster_ratio_percent="$1"
  "$PYTHON_BIN" main.py -m \
    experiment=non_neighbor_edges \
    model="$MODELS" \
    model.num_layers="$LAYERS" \
    dataset="$DATASETS" \
    experiment.edges_per_cluster=-1 \
    experiment.cluster_ratio_percent="$cluster_ratio_percent" \
    experiment.removal_candidate_sampler=group_non_neighbor,group_neighbor \
    experiment.removal_neighbor_dist=1,2,3,4,5 \
    seed=0
}

run_exp3() {
  local initial_cluster_ratio="$EXP3_CLUSTER_RATIO_PERCENT"
  local retry_cluster_ratio="$EXP3_RETRY_CLUSTER_RATIO_PERCENT"
  local tmp_log
  local run_status
  tmp_log="$(mktemp)"

  run_status=0
  run_exp3_with_cluster_ratio "$initial_cluster_ratio" 2>&1 | tee "$tmp_log" || run_status=$?

  if ! grep -Fq "$CONSTRAINT_UNSAT_ERROR_TEXT" "$tmp_log"; then
    rm -f "$tmp_log"
    return "$run_status"
  fi

  mkdir -p "$(dirname "$EXP3_RETRY_MARKER_FILE")"
  {
    printf '[%s] exp3 constraint-unsat retry triggered: %s -> %s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" \
      "$initial_cluster_ratio" \
      "$retry_cluster_ratio"
  } >> "$EXP3_RETRY_MARKER_FILE"

  echo "[exp3] detected constraint-unsat error. Retrying with experiment.cluster_ratio_percent=$retry_cluster_ratio"
  if run_exp3_with_cluster_ratio "$retry_cluster_ratio"; then
    {
      printf '[%s] exp3 retry success\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    } >> "$EXP3_RETRY_MARKER_FILE"
    rm -f "$tmp_log"
    return 0
  else
    run_status=$?
  fi

  {
    printf '[%s] exp3 retry failed\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  } >> "$EXP3_RETRY_MARKER_FILE"
  rm -f "$tmp_log"
  return "$run_status"
}

run_exp4_with_cluster_ratio() {
  local cluster_ratio_percent="$1"
  "$PYTHON_BIN" main.py -m \
    experiment=clusters \
    model="$MODELS" \
    model.num_layers="$LAYERS" \
    dataset="$DATASETS" \
    experiment.num_of_clusters=3 \
    experiment.edges_per_cluster=-1 \
    experiment.intra_cluster_dist=1,2,3 \
    experiment.inter_cluster_dist=1,2,3 \
    experiment.cluster_ratio_percent="$cluster_ratio_percent" \
    seed=0
}

run_exp4() {
  run_exp4_with_cluster_ratio "$EXP4_CLUSTER_RATIO_PERCENT"
}

run_exp5_with_cluster_ratio() {
  local cluster_ratio_percent="$1"
  "$PYTHON_BIN" main.py -m \
    experiment=influence_calc \
    model="$MODELS" \
    model.num_layers="$LAYERS" \
    dataset="$DATASETS" \
    experiment.name=clusters \
    experiment.num_of_clusters=3 \
    experiment.edges_per_cluster=-1 \
    experiment.intra_cluster_dist=1 \
    experiment.inter_cluster_dist=2 \
    experiment.cluster_ratio_percent="$cluster_ratio_percent" \
    experiment.influence_mode=calculate_influence,clusterwise_step_by_step \
    seed=0
}

run_exp5() {
  run_exp5_with_cluster_ratio "$EXP5_CLUSTER_RATIO_PERCENT"
}

collect_missing_matrix_rows() {
  local target_exp="$1"
  local root="$2"

  MODELS="$MODELS" LAYERS="$LAYERS" DATASETS="$DATASETS" SEED_DEFAULT="$SEED_DEFAULT" \
    EXP3_CLUSTER_RATIO_PERCENT="$EXP3_CLUSTER_RATIO_PERCENT" \
    EXP3_RETRY_CLUSTER_RATIO_PERCENT="$EXP3_RETRY_CLUSTER_RATIO_PERCENT" \
    EXP4_CLUSTER_RATIO_PERCENT="$EXP4_CLUSTER_RATIO_PERCENT" \
    EXP4_RETRY_CLUSTER_RATIO_PERCENT="$EXP4_RETRY_CLUSTER_RATIO_PERCENT" \
    EXP5_CLUSTER_RATIO_PERCENT="$EXP5_CLUSTER_RATIO_PERCENT" \
    EXP5_RETRY_CLUSTER_RATIO_PERCENT="$EXP5_RETRY_CLUSTER_RATIO_PERCENT" \
    CONSTRAINT_UNSAT_MARKER="$CONSTRAINT_UNSAT_MARKER_FILE" \
    CONSTRAINT_UNSAT_ERROR_TEXT="$CONSTRAINT_UNSAT_ERROR_TEXT" \
    "$PYTHON_BIN" - "$target_exp" "$root" <<'PY'
import glob
import os
import sys
from itertools import product

target_exp = sys.argv[1]
root = sys.argv[2]

models = [x for x in os.environ.get("MODELS", "").split(",") if x]
layers = [x for x in os.environ.get("LAYERS", "").split(",") if x]
datasets = [x for x in os.environ.get("DATASETS", "").split(",") if x]
seed = os.environ.get("SEED_DEFAULT", "0")
exp3_cluster_ratio_percent = os.environ.get("EXP3_CLUSTER_RATIO_PERCENT", "15")
exp3_retry_cluster_ratio_percent = os.environ.get("EXP3_RETRY_CLUSTER_RATIO_PERCENT", "9")
exp4_cluster_ratio_percent = os.environ.get("EXP4_CLUSTER_RATIO_PERCENT", "15")
exp4_retry_cluster_ratio_percent = os.environ.get("EXP4_RETRY_CLUSTER_RATIO_PERCENT", "9")
exp5_cluster_ratio_percent = os.environ.get("EXP5_CLUSTER_RATIO_PERCENT", "5")
exp5_retry_cluster_ratio_percent = os.environ.get("EXP5_RETRY_CLUSTER_RATIO_PERCENT", exp5_cluster_ratio_percent)


def key_of(lines):
    return "\n".join(sorted(lines))


def parse_override_map(lines):
    parsed = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key] = value
    return parsed


def logical_key_of(lines):
    parsed = parse_override_map(lines)
    base_ratio = None
    valid_ratios = None

    if parsed.get("experiment") == "non_neighbor_edges":
        base_ratio = exp3_cluster_ratio_percent
        valid_ratios = {exp3_cluster_ratio_percent, exp3_retry_cluster_ratio_percent}
    elif parsed.get("experiment") == "clusters":
        base_ratio = exp4_cluster_ratio_percent
        valid_ratios = {exp4_cluster_ratio_percent, exp4_retry_cluster_ratio_percent}
    elif parsed.get("experiment") == "influence_calc" and parsed.get("experiment.name") == "clusters":
        base_ratio = exp5_cluster_ratio_percent
        valid_ratios = {exp5_cluster_ratio_percent, exp5_retry_cluster_ratio_percent}

    if base_ratio is None:
        return key_of(lines)

    normalized = []
    for line in lines:
        if line.startswith("experiment.cluster_ratio_percent="):
            current_ratio = line.split("=", 1)[1]
            if current_ratio in valid_ratios:
                line = f"experiment.cluster_ratio_percent={base_ratio}"
        normalized.append(line)
    return key_of(normalized)


def existing_index(search_root):
    attempted = {}
    done = {}
    constraint_unsat = {}
    if not os.path.isdir(search_root):
        return attempted, done, constraint_unsat

    pattern = os.path.join(search_root, "**", ".hydra", "overrides.yaml")
    marker_name = os.environ.get("CONSTRAINT_UNSAT_MARKER", "CONSTRAINT_UNSAT")
    error_text = os.environ.get(
        "CONSTRAINT_UNSAT_ERROR_TEXT",
        "Failed to sample a valid edge group with the current constraints.",
    )

    def is_constraint_unsat_run(run_dir):
        if os.path.isfile(os.path.join(run_dir, marker_name)):
            return True
        if os.path.isfile(os.path.join(run_dir, f"{marker_name}.txt")):
            return True
        log_path = os.path.join(run_dir, "main.log")
        if not os.path.isfile(log_path):
            return False
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as log_f:
                for line in log_f:
                    if error_text in line:
                        return True
        except OSError:
            return False
        return False

    for override_file in glob.glob(pattern, recursive=True):
        run_dir = os.path.dirname(os.path.dirname(override_file))
        with open(override_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for raw in f:
                line = raw.strip()
                if line.startswith("- "):
                    lines.append(line[2:])
        k = logical_key_of(lines)
        attempted.setdefault(k, []).append(run_dir)
        if os.path.isfile(os.path.join(run_dir, "DONE")):
            done[k] = True
        if is_constraint_unsat_run(run_dir):
            constraint_unsat[k] = True
    return attempted, done, constraint_unsat


def expected_overrides(exp_name):
    rows = []
    if exp_name in ("exp1", "all"):
        ratios = ["1", "5", "10", "20", "30", "50", "70", "90"]
        for model, layer, dataset, ratio in product(models, layers, datasets, ratios):
            rows.append(
                [
                    "experiment=large_drop_influence",
                    f"model={model}",
                    f"model.num_layers={layer}",
                    f"dataset={dataset}",
                    f"experiment.ratio_group_elem={ratio}",
                    f"seed={seed}",
                ]
            )

    if exp_name in ("exp3", "all"):
        samplers = ["group_non_neighbor", "group_neighbor"]
        dists = ["1", "2", "3", "4", "5"]
        for model, layer, dataset, sampler, dist in product(models, layers, datasets, samplers, dists):
            rows.append(
                [
                    "experiment=non_neighbor_edges",
                    f"model={model}",
                    f"model.num_layers={layer}",
                    f"dataset={dataset}",
                    "experiment.edges_per_cluster=-1",
                    f"experiment.cluster_ratio_percent={exp3_cluster_ratio_percent}",
                    f"experiment.removal_candidate_sampler={sampler}",
                    f"experiment.removal_neighbor_dist={dist}",
                    f"seed={seed}",
                ]
            )

    if exp_name in ("exp4", "all"):
        intra_dists = ["1", "2", "3"]
        inter_dists = ["1", "2", "3"]
        for model, layer, dataset, intra_dist, inter_dist in product(models, layers, datasets, intra_dists, inter_dists):
            rows.append(
                [
                    "experiment=clusters",
                    f"model={model}",
                    f"model.num_layers={layer}",
                    f"dataset={dataset}",
                    "experiment.num_of_clusters=3",
                    "experiment.edges_per_cluster=-1",
                    f"experiment.intra_cluster_dist={intra_dist}",
                    f"experiment.inter_cluster_dist={inter_dist}",
                    f"experiment.cluster_ratio_percent={exp4_cluster_ratio_percent}",
                    f"seed={seed}",
                ]
            )

    if exp_name in ("exp5", "all"):
        modes = ["calculate_influence", "clusterwise_step_by_step"]
        for model, layer, dataset, mode in product(models, layers, datasets, modes):
            rows.append(
                [
                    "experiment=influence_calc",
                    f"model={model}",
                    f"model.num_layers={layer}",
                    f"dataset={dataset}",
                    "experiment.name=clusters",
                    "experiment.num_of_clusters=3",
                    "experiment.edges_per_cluster=-1",
                    "experiment.intra_cluster_dist=1",
                    "experiment.inter_cluster_dist=2",
                    f"experiment.cluster_ratio_percent={exp5_cluster_ratio_percent}",
                    f"experiment.influence_mode={mode}",
                    f"seed={seed}",
                ]
            )

    return rows


if target_exp not in {"exp1", "exp3", "exp4", "exp5", "all"}:
    raise SystemExit(f"invalid experiment selector: {target_exp}")

attempted_idx, done_idx, constraint_unsat_idx = existing_index(root)
for overrides in expected_overrides(target_exp):
    k = logical_key_of(overrides)
    if done_idx.get(k, False) or constraint_unsat_idx.get(k, False):
        continue
    if k in attempted_idx:
        status = "failed"
        run_dir = sorted(attempted_idx[k])[-1]
    else:
        status = "unattempted"
        run_dir = "-"
    print("\t".join([status, run_dir] + overrides))
PY
}

collect_constraint_unsat_matrix_rows() {
  local target_exp="$1"
  local root="$2"

  MODELS="$MODELS" LAYERS="$LAYERS" DATASETS="$DATASETS" SEED_DEFAULT="$SEED_DEFAULT" \
    EXP3_CLUSTER_RATIO_PERCENT="$EXP3_CLUSTER_RATIO_PERCENT" \
    EXP3_RETRY_CLUSTER_RATIO_PERCENT="$EXP3_RETRY_CLUSTER_RATIO_PERCENT" \
    EXP4_CLUSTER_RATIO_PERCENT="$EXP4_CLUSTER_RATIO_PERCENT" \
    EXP4_RETRY_CLUSTER_RATIO_PERCENT="$EXP4_RETRY_CLUSTER_RATIO_PERCENT" \
    EXP5_CLUSTER_RATIO_PERCENT="$EXP5_CLUSTER_RATIO_PERCENT" \
    EXP5_RETRY_CLUSTER_RATIO_PERCENT="$EXP5_RETRY_CLUSTER_RATIO_PERCENT" \
    CONSTRAINT_UNSAT_MARKER="$CONSTRAINT_UNSAT_MARKER_FILE" \
    CONSTRAINT_UNSAT_ERROR_TEXT="$CONSTRAINT_UNSAT_ERROR_TEXT" \
    "$PYTHON_BIN" - "$target_exp" "$root" <<'PY'
import glob
import os
import sys
from itertools import product

target_exp = sys.argv[1]
root = sys.argv[2]

models = [x for x in os.environ.get("MODELS", "").split(",") if x]
layers = [x for x in os.environ.get("LAYERS", "").split(",") if x]
datasets = [x for x in os.environ.get("DATASETS", "").split(",") if x]
seed = os.environ.get("SEED_DEFAULT", "0")
exp3_cluster_ratio_percent = os.environ.get("EXP3_CLUSTER_RATIO_PERCENT", "15")
exp3_retry_cluster_ratio_percent = os.environ.get("EXP3_RETRY_CLUSTER_RATIO_PERCENT", "9")
exp4_cluster_ratio_percent = os.environ.get("EXP4_CLUSTER_RATIO_PERCENT", "15")
exp4_retry_cluster_ratio_percent = os.environ.get("EXP4_RETRY_CLUSTER_RATIO_PERCENT", "9")
exp5_cluster_ratio_percent = os.environ.get("EXP5_CLUSTER_RATIO_PERCENT", "5")
exp5_retry_cluster_ratio_percent = os.environ.get("EXP5_RETRY_CLUSTER_RATIO_PERCENT", exp5_cluster_ratio_percent)


def key_of(lines):
    return "\n".join(sorted(lines))


def parse_override_map(lines):
    parsed = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key] = value
    return parsed


def logical_key_of(lines):
    parsed = parse_override_map(lines)
    base_ratio = None
    valid_ratios = None

    if parsed.get("experiment") == "non_neighbor_edges":
        base_ratio = exp3_cluster_ratio_percent
        valid_ratios = {exp3_cluster_ratio_percent, exp3_retry_cluster_ratio_percent}
    elif parsed.get("experiment") == "clusters":
        base_ratio = exp4_cluster_ratio_percent
        valid_ratios = {exp4_cluster_ratio_percent, exp4_retry_cluster_ratio_percent}
    elif parsed.get("experiment") == "influence_calc" and parsed.get("experiment.name") == "clusters":
        base_ratio = exp5_cluster_ratio_percent
        valid_ratios = {exp5_cluster_ratio_percent, exp5_retry_cluster_ratio_percent}

    if base_ratio is None:
        return key_of(lines)

    normalized = []
    for line in lines:
        if line.startswith("experiment.cluster_ratio_percent="):
            current_ratio = line.split("=", 1)[1]
            if current_ratio in valid_ratios:
                line = f"experiment.cluster_ratio_percent={base_ratio}"
        normalized.append(line)
    return key_of(normalized)


def existing_index(search_root):
    done = {}
    constraint_unsat_dirs = {}
    if not os.path.isdir(search_root):
        return done, constraint_unsat_dirs

    pattern = os.path.join(search_root, "**", ".hydra", "overrides.yaml")
    marker_name = os.environ.get("CONSTRAINT_UNSAT_MARKER", "CONSTRAINT_UNSAT")
    error_text = os.environ.get(
        "CONSTRAINT_UNSAT_ERROR_TEXT",
        "Failed to sample a valid edge group with the current constraints.",
    )

    def is_constraint_unsat_run(run_dir):
        if os.path.isfile(os.path.join(run_dir, marker_name)):
            return True
        if os.path.isfile(os.path.join(run_dir, f"{marker_name}.txt")):
            return True
        log_path = os.path.join(run_dir, "main.log")
        if not os.path.isfile(log_path):
            return False
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as log_f:
                for line in log_f:
                    if error_text in line:
                        return True
        except OSError:
            return False
        return False

    for override_file in glob.glob(pattern, recursive=True):
        run_dir = os.path.dirname(os.path.dirname(override_file))
        with open(override_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for raw in f:
                line = raw.strip()
                if line.startswith("- "):
                    lines.append(line[2:])
        k = logical_key_of(lines)
        if os.path.isfile(os.path.join(run_dir, "DONE")):
            done[k] = True
        if is_constraint_unsat_run(run_dir):
            constraint_unsat_dirs.setdefault(k, []).append(run_dir)
    return done, constraint_unsat_dirs


def expected_overrides(exp_name):
    rows = []
    if exp_name in ("exp1", "all"):
        ratios = ["1", "5", "10", "20", "30", "50", "70", "90"]
        for model, layer, dataset, ratio in product(models, layers, datasets, ratios):
            rows.append(
                [
                    "experiment=large_drop_influence",
                    f"model={model}",
                    f"model.num_layers={layer}",
                    f"dataset={dataset}",
                    f"experiment.ratio_group_elem={ratio}",
                    f"seed={seed}",
                ]
            )

    if exp_name in ("exp3", "all"):
        samplers = ["group_non_neighbor", "group_neighbor"]
        dists = ["1", "2", "3", "4", "5"]
        for model, layer, dataset, sampler, dist in product(models, layers, datasets, samplers, dists):
            rows.append(
                [
                    "experiment=non_neighbor_edges",
                    f"model={model}",
                    f"model.num_layers={layer}",
                    f"dataset={dataset}",
                    "experiment.edges_per_cluster=-1",
                    f"experiment.cluster_ratio_percent={exp3_cluster_ratio_percent}",
                    f"experiment.removal_candidate_sampler={sampler}",
                    f"experiment.removal_neighbor_dist={dist}",
                    f"seed={seed}",
                ]
            )

    if exp_name in ("exp4", "all"):
        intra_dists = ["1", "2", "3"]
        inter_dists = ["1", "2", "3"]
        for model, layer, dataset, intra_dist, inter_dist in product(models, layers, datasets, intra_dists, inter_dists):
            rows.append(
                [
                    "experiment=clusters",
                    f"model={model}",
                    f"model.num_layers={layer}",
                    f"dataset={dataset}",
                    "experiment.num_of_clusters=3",
                    "experiment.edges_per_cluster=-1",
                    f"experiment.intra_cluster_dist={intra_dist}",
                    f"experiment.inter_cluster_dist={inter_dist}",
                    f"experiment.cluster_ratio_percent={exp4_cluster_ratio_percent}",
                    f"seed={seed}",
                ]
            )

    if exp_name in ("exp5", "all"):
        modes = ["calculate_influence", "clusterwise_step_by_step"]
        for model, layer, dataset, mode in product(models, layers, datasets, modes):
            rows.append(
                [
                    "experiment=influence_calc",
                    f"model={model}",
                    f"model.num_layers={layer}",
                    f"dataset={dataset}",
                    "experiment.name=clusters",
                    "experiment.num_of_clusters=3",
                    "experiment.edges_per_cluster=-1",
                    "experiment.intra_cluster_dist=1",
                    "experiment.inter_cluster_dist=2",
                    f"experiment.cluster_ratio_percent={exp5_cluster_ratio_percent}",
                    f"experiment.influence_mode={mode}",
                    f"seed={seed}",
                ]
            )

    return rows


if target_exp not in {"exp1", "exp3", "exp4", "exp5", "all"}:
    raise SystemExit(f"invalid experiment selector: {target_exp}")

done_idx, constraint_unsat_dirs_idx = existing_index(root)
for overrides in expected_overrides(target_exp):
    k = logical_key_of(overrides)
    if done_idx.get(k, False):
        continue
    run_dirs = constraint_unsat_dirs_idx.get(k, [])
    if not run_dirs:
        continue
    run_dir = sorted(run_dirs)[-1]
    print("\t".join(["constraint_unsatisfied", run_dir] + overrides))
PY
}

collect_failed_override_files() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    return 0
  fi

  find "$root" -type f -path '*/.hydra/overrides.yaml' | while IFS= read -r override_file; do
    local run_dir
    run_dir="$(dirname "$(dirname "$override_file")")"
    if [[ -f "$run_dir/DONE" ]]; then
      continue
    fi
    if is_constraint_unsat_run_dir "$run_dir"; then
      continue
    fi
    echo "$override_file"
  done
}

collect_constraint_unsat_override_files() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    return 0
  fi

  find "$root" -type f -path '*/.hydra/overrides.yaml' | while IFS= read -r override_file; do
    local run_dir
    run_dir="$(dirname "$(dirname "$override_file")")"
    if [[ -f "$run_dir/DONE" ]]; then
      continue
    fi
    if ! is_constraint_unsat_run_dir "$run_dir"; then
      continue
    fi
    echo "$override_file"
  done
}

collect_run_dirs() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    return 0
  fi

  find "$root" -type f -path '*/.hydra/overrides.yaml' | while IFS= read -r override_file; do
    dirname "$(dirname "$override_file")"
  done
}

is_constraint_unsat_run_dir() {
  local run_dir="$1"
  local log_file="$run_dir/main.log"

  if [[ -f "$run_dir/$CONSTRAINT_UNSAT_MARKER_FILE" ]]; then
    return 0
  fi

  if [[ -f "$run_dir/${CONSTRAINT_UNSAT_MARKER_FILE}.txt" ]]; then
    return 0
  fi

  if [[ -f "$log_file" ]] && grep -Fq "$CONSTRAINT_UNSAT_ERROR_TEXT" "$log_file"; then
    return 0
  fi

  return 1
}

prepare_run_dir_for_rerun() {
  local run_dir="$1"
  rm -f \
    "$run_dir/DONE" \
    "$run_dir/$CONSTRAINT_UNSAT_MARKER_FILE" \
    "$run_dir/${CONSTRAINT_UNSAT_MARKER_FILE}.txt" \
    "$run_dir/main.log" 2>/dev/null || true
}

detect_exp_from_overrides() {
  local has_influence_calc="0"
  local has_clusters_name="0"
  local override

  for override in "$@"; do
    case "$override" in
      experiment=non_neighbor_edges)
        echo "exp3"
        return 0
        ;;
      experiment=clusters)
        echo "exp4"
        return 0
        ;;
      experiment=influence_calc)
        has_influence_calc="1"
        ;;
      experiment.name=clusters)
        has_clusters_name="1"
        ;;
    esac
  done

  if [[ "$has_influence_calc" == "1" && "$has_clusters_name" == "1" ]]; then
    echo "exp5"
  fi
}

retry_cluster_ratio_for_exp() {
  local exp_id="$1"
  case "$exp_id" in
    exp3)
      echo "$EXP3_RETRY_CLUSTER_RATIO_PERCENT"
      ;;
    exp4)
      echo "$EXP4_RETRY_CLUSTER_RATIO_PERCENT"
      ;;
    exp5)
      echo "$EXP5_RETRY_CLUSTER_RATIO_PERCENT"
      ;;
  esac
}

get_override_value() {
  local key="$1"
  shift
  local override

  for override in "$@"; do
    case "$override" in
      "$key"=*)
        echo "${override#*=}"
        return 0
        ;;
    esac
  done

  return 1
}

set_override_value() {
  local key="$1"
  local value="$2"
  local array_name="$3"
  local -n override_ref="$array_name"
  local idx

  for idx in "${!override_ref[@]}"; do
    case "${override_ref[$idx]}" in
      "$key"=*)
        override_ref[$idx]="$key=$value"
        return 0
        ;;
    esac
  done

  override_ref+=("$key=$value")
}

list_failed() {
  local root="${1:-$SEARCH_ROOT_DEFAULT}"
  local count=0

  while IFS= read -r override_file; do
    [[ -z "$override_file" ]] && continue
    local run_dir
    run_dir="$(dirname "$(dirname "$override_file")")"
    echo "$run_dir"
    count=$((count + 1))
  done < <(collect_failed_override_files "$root")

  echo "failed_runs=$count"
}

list_constraint_unsatisfied() {
  local root="${1:-$SEARCH_ROOT_DEFAULT}"
  local count=0

  while IFS= read -r override_file; do
    [[ -z "$override_file" ]] && continue
    local run_dir
    run_dir="$(dirname "$(dirname "$override_file")")"
    echo "$run_dir"
    count=$((count + 1))
  done < <(collect_constraint_unsat_override_files "$root")

  echo "constraint_unsatisfied_runs=$count"
}

show_progress() {
  local root="${1:-$SEARCH_ROOT_DEFAULT}"
  local total=0
  local done=0
  local constraint_unsatisfied=0
  local remaining=0

  while IFS= read -r run_dir; do
    [[ -z "$run_dir" ]] && continue
    total=$((total + 1))
    if [[ -f "$run_dir/DONE" ]]; then
      done=$((done + 1))
    elif is_constraint_unsat_run_dir "$run_dir"; then
      constraint_unsatisfied=$((constraint_unsatisfied + 1))
    else
      remaining=$((remaining + 1))
    fi
  done < <(collect_run_dirs "$root")

  if [[ "$total" -eq 0 ]]; then
    echo "progress: total=0 done=0 constraint_unsatisfied=0 remaining=0 percent=0.00%"
    return 0
  fi

  local percent
  percent=$(awk -v d="$done" -v c="$constraint_unsatisfied" -v t="$total" 'BEGIN { printf "%.2f", (100.0*(d+c))/t }')
  echo "progress: total=$total done=$done constraint_unsatisfied=$constraint_unsatisfied remaining=$remaining percent=${percent}%"
}

watch_progress() {
  local root="${1:-$SEARCH_ROOT_DEFAULT}"
  local interval="${2:-10}"

  while true; do
    date '+[%Y-%m-%d %H:%M:%S]'
    show_progress "$root"
    sleep "$interval"
  done
}

run_single_from_override_file() {
  local override_file="$1"
  local run_dir
  run_dir="$(dirname "$(dirname "$override_file")")"

  local -a overrides=()
  mapfile -t overrides < <(sed -n 's/^\s*-\s*//p' "$override_file")
  if [[ "${#overrides[@]}" -eq 0 ]]; then
    echo "[skip] no overrides in $override_file"
    return 0
  fi

  echo "[rerun] $run_dir"
  "$PYTHON_BIN" main.py "${overrides[@]}" "hydra.run.dir=$run_dir"
}

rerun_failed() {
  local root="${1:-$SEARCH_ROOT_DEFAULT}"
  local rerun_num_gpus="${RERUN_NUM_GPUS:-4}"
  local rerun_max_parallel="${RERUN_MAX_PARALLEL:-$rerun_num_gpus}"
  local count=0
  local success=0
  local failed=0
  local constraint_unsatisfied=0
  local launched=0
  local -a pids=()
  local -a run_dirs=()

  if ! [[ "$rerun_num_gpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: RERUN_NUM_GPUS must be a positive integer (current: $rerun_num_gpus)"
    return 1
  fi
  if ! [[ "$rerun_max_parallel" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: RERUN_MAX_PARALLEL must be a positive integer (current: $rerun_max_parallel)"
    return 1
  fi

  while IFS= read -r override_file; do
    [[ -z "$override_file" ]] && continue
    local run_dir
    run_dir="$(dirname "$(dirname "$override_file")")"
    local -a overrides=()
    mapfile -t overrides < <(sed -n 's/^\s*-\s*//p' "$override_file")
    if [[ "${#overrides[@]}" -eq 0 ]]; then
      echo "[skip] no overrides in $override_file"
      continue
    fi

    prepare_run_dir_for_rerun "$run_dir"

    local gpu_id
    gpu_id=$((launched % rerun_num_gpus))
    echo "[rerun-failed gpu${gpu_id}] $run_dir"
    (
      export CUDA_VISIBLE_DEVICES="$gpu_id"
      export EIF_RESPECT_CUDA_VISIBLE_DEVICES=1
      "$PYTHON_BIN" main.py "${overrides[@]}" "hydra.run.dir=$run_dir"
    ) &
    pids+=("$!")
    run_dirs+=("$run_dir")
    launched=$((launched + 1))
    count=$((count + 1))

    if (( ${#pids[@]} >= rerun_max_parallel )); then
      if wait "${pids[0]}"; then
        if is_constraint_unsat_run_dir "${run_dirs[0]}"; then
          constraint_unsatisfied=$((constraint_unsatisfied + 1))
        else
          touch "${run_dirs[0]}/DONE" 2>/dev/null || true
          success=$((success + 1))
        fi
      else
        failed=$((failed + 1))
      fi
      pids=("${pids[@]:1}")
      run_dirs=("${run_dirs[@]:1}")
    fi
  done < <(collect_failed_override_files "$root")

  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      if is_constraint_unsat_run_dir "${run_dirs[$idx]}"; then
        constraint_unsatisfied=$((constraint_unsatisfied + 1))
      else
        touch "${run_dirs[$idx]}/DONE" 2>/dev/null || true
        success=$((success + 1))
      fi
    else
      failed=$((failed + 1))
    fi
  done

  echo "rerun_count=$count success=$success constraint_unsatisfied=$constraint_unsatisfied failed=$failed parallel=$rerun_max_parallel gpus=$rerun_num_gpus"
  if (( failed > 0 )); then
    return 1
  fi
}

rerun_constraint_unsatisfied() {
  local exp_selector="${1:-all}"
  local root="${2:-$SEARCH_ROOT_DEFAULT}"
  local rerun_cluster_ratio_override="${3:-}"
  local rerun_num_gpus="${RERUN_NUM_GPUS:-4}"
  local rerun_max_parallel="${RERUN_MAX_PARALLEL:-$rerun_num_gpus}"
  local count=0
  local success=0
  local failed=0
  local still_constraint_unsatisfied=0
  local target_constraint_unsatisfied=0
  local launched=0
  local -a pids=()
  local -a run_dirs=()

  if [[ ! "$exp_selector" =~ ^(exp1|exp3|exp4|exp5|all)$ ]]; then
    echo "error: rerun-constraint-unsatisfied requires experiment selector: exp1|exp3|exp4|exp5|all"
    return 1
  fi
  if [[ -n "$rerun_cluster_ratio_override" ]] && ! [[ "$rerun_cluster_ratio_override" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: rerun cluster ratio override must be a positive integer (current: $rerun_cluster_ratio_override)"
    return 1
  fi
  if ! [[ "$rerun_num_gpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: RERUN_NUM_GPUS must be a positive integer (current: $rerun_num_gpus)"
    return 1
  fi
  if ! [[ "$rerun_max_parallel" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: RERUN_MAX_PARALLEL must be a positive integer (current: $rerun_max_parallel)"
    return 1
  fi

  while IFS=$'\t' read -r -a row; do
    [[ "${#row[@]}" -lt 3 ]] && continue
    local run_dir="${row[1]}"
    local -a overrides=("${row[@]:2}")
    local exp_id
    local current_cluster_ratio=""
    local retry_cluster_ratio=""
    if [[ "$run_dir" == "-" ]]; then
      continue
    fi
    target_constraint_unsatisfied=$((target_constraint_unsatisfied + 1))

    exp_id="$(detect_exp_from_overrides "${overrides[@]}")"
    current_cluster_ratio="$(get_override_value "experiment.cluster_ratio_percent" "${overrides[@]}" 2>/dev/null || true)"
    if [[ -n "$rerun_cluster_ratio_override" && "$exp_id" =~ ^(exp3|exp4|exp5)$ ]]; then
      retry_cluster_ratio="$rerun_cluster_ratio_override"
    else
      retry_cluster_ratio="$(retry_cluster_ratio_for_exp "$exp_id")"
    fi
    if [[ -n "$retry_cluster_ratio" ]]; then
      set_override_value "experiment.cluster_ratio_percent" "$retry_cluster_ratio" overrides
    fi

    prepare_run_dir_for_rerun "$run_dir"

    local gpu_id
    gpu_id=$((launched % rerun_num_gpus))
    if [[ -n "$retry_cluster_ratio" && "$current_cluster_ratio" != "$retry_cluster_ratio" ]]; then
      echo "[rerun-constraint-unsatisfied gpu${gpu_id}] $run_dir :: cluster_ratio_percent ${current_cluster_ratio:-unknown} -> $retry_cluster_ratio"
    else
      echo "[rerun-constraint-unsatisfied gpu${gpu_id}] $run_dir"
    fi
    (
      export CUDA_VISIBLE_DEVICES="$gpu_id"
      export EIF_RESPECT_CUDA_VISIBLE_DEVICES=1
      "$PYTHON_BIN" main.py "${overrides[@]}" "hydra.run.dir=$run_dir"
    ) &
    pids+=("$!")
    run_dirs+=("$run_dir")
    launched=$((launched + 1))
    count=$((count + 1))

    if (( ${#pids[@]} >= rerun_max_parallel )); then
      if wait "${pids[0]}"; then
        if is_constraint_unsat_run_dir "${run_dirs[0]}"; then
          still_constraint_unsatisfied=$((still_constraint_unsatisfied + 1))
        else
          touch "${run_dirs[0]}/DONE" 2>/dev/null || true
          rm -f "${run_dirs[0]}/$CONSTRAINT_UNSAT_MARKER_FILE" "${run_dirs[0]}/${CONSTRAINT_UNSAT_MARKER_FILE}.txt" 2>/dev/null || true
          success=$((success + 1))
        fi
      else
        failed=$((failed + 1))
      fi
      pids=("${pids[@]:1}")
      run_dirs=("${run_dirs[@]:1}")
    fi
  done < <(collect_constraint_unsat_matrix_rows "$exp_selector" "$root")

  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      if is_constraint_unsat_run_dir "${run_dirs[$idx]}"; then
        still_constraint_unsatisfied=$((still_constraint_unsatisfied + 1))
      else
        touch "${run_dirs[$idx]}/DONE" 2>/dev/null || true
        rm -f "${run_dirs[$idx]}/$CONSTRAINT_UNSAT_MARKER_FILE" "${run_dirs[$idx]}/${CONSTRAINT_UNSAT_MARKER_FILE}.txt" 2>/dev/null || true
        success=$((success + 1))
      fi
    else
      failed=$((failed + 1))
    fi
  done

  echo "rerun_count=$count success=$success still_constraint_unsatisfied=$still_constraint_unsatisfied failed=$failed target_constraint_unsatisfied=$target_constraint_unsatisfied parallel=$rerun_max_parallel gpus=$rerun_num_gpus"
  if (( failed > 0 )); then
    return 1
  fi
}

list_missing() {
  local exp_selector="${1:-}"
  local root="${2:-$SEARCH_ROOT_DEFAULT}"
  local total=0
  local failed=0
  local unattempted=0

  if [[ -z "$exp_selector" ]]; then
    echo "error: list-missing requires experiment selector: exp1|exp3|exp4|exp5|all"
    return 1
  fi

  while IFS=$'\t' read -r -a row; do
    [[ "${#row[@]}" -lt 3 ]] && continue
    local status="${row[0]}"
    local run_dir="${row[1]}"
    local overrides=("${row[@]:2}")
    total=$((total + 1))

    if [[ "$status" == "failed" ]]; then
      failed=$((failed + 1))
      echo "[failed] $run_dir :: ${overrides[*]}"
    else
      unattempted=$((unattempted + 1))
      echo "[unattempted] ${overrides[*]}"
    fi
  done < <(collect_missing_matrix_rows "$exp_selector" "$root")

  echo "missing_total=$total failed=$failed unattempted=$unattempted"
}

show_summary() {
  local exp_selector="${1:-}"
  local root="${2:-$SEARCH_ROOT_DEFAULT}"

  if [[ -z "$exp_selector" ]]; then
    echo "error: summary requires experiment selector: exp1|exp3|exp4|exp5|all"
    return 1
  fi

  MODELS="$MODELS" LAYERS="$LAYERS" DATASETS="$DATASETS" SEED_DEFAULT="$SEED_DEFAULT" \
    EXP3_CLUSTER_RATIO_PERCENT="$EXP3_CLUSTER_RATIO_PERCENT" \
    EXP3_RETRY_CLUSTER_RATIO_PERCENT="$EXP3_RETRY_CLUSTER_RATIO_PERCENT" \
    EXP4_CLUSTER_RATIO_PERCENT="$EXP4_CLUSTER_RATIO_PERCENT" \
    EXP4_RETRY_CLUSTER_RATIO_PERCENT="$EXP4_RETRY_CLUSTER_RATIO_PERCENT" \
    EXP5_CLUSTER_RATIO_PERCENT="$EXP5_CLUSTER_RATIO_PERCENT" \
    EXP5_RETRY_CLUSTER_RATIO_PERCENT="$EXP5_RETRY_CLUSTER_RATIO_PERCENT" \
    CONSTRAINT_UNSAT_MARKER="$CONSTRAINT_UNSAT_MARKER_FILE" \
    CONSTRAINT_UNSAT_ERROR_TEXT="$CONSTRAINT_UNSAT_ERROR_TEXT" \
    "$PYTHON_BIN" - "$exp_selector" "$root" <<'PY'
import glob
import os
import sys
from itertools import product
from collections import defaultdict

exp_selector = sys.argv[1]
root = sys.argv[2]

models = [x for x in os.environ.get("MODELS", "").split(",") if x]
layers = [x for x in os.environ.get("LAYERS", "").split(",") if x]
datasets = [x for x in os.environ.get("DATASETS", "").split(",") if x]
seed = os.environ.get("SEED_DEFAULT", "0")
exp3_cluster_ratio_percent = os.environ.get("EXP3_CLUSTER_RATIO_PERCENT", "15")
exp3_retry_cluster_ratio_percent = os.environ.get("EXP3_RETRY_CLUSTER_RATIO_PERCENT", "9")
exp4_cluster_ratio_percent = os.environ.get("EXP4_CLUSTER_RATIO_PERCENT", "15")
exp4_retry_cluster_ratio_percent = os.environ.get("EXP4_RETRY_CLUSTER_RATIO_PERCENT", "9")
exp5_cluster_ratio_percent = os.environ.get("EXP5_CLUSTER_RATIO_PERCENT", "5")
exp5_retry_cluster_ratio_percent = os.environ.get("EXP5_RETRY_CLUSTER_RATIO_PERCENT", exp5_cluster_ratio_percent)


def key_of(lines):
    return "\n".join(sorted(lines))


def parse_override_map(lines):
    parsed = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key] = value
    return parsed


def logical_key_of(lines):
    parsed = parse_override_map(lines)
    base_ratio = None
    valid_ratios = None

    if parsed.get("experiment") == "non_neighbor_edges":
        base_ratio = exp3_cluster_ratio_percent
        valid_ratios = {exp3_cluster_ratio_percent, exp3_retry_cluster_ratio_percent}
    elif parsed.get("experiment") == "clusters":
        base_ratio = exp4_cluster_ratio_percent
        valid_ratios = {exp4_cluster_ratio_percent, exp4_retry_cluster_ratio_percent}
    elif parsed.get("experiment") == "influence_calc" and parsed.get("experiment.name") == "clusters":
        base_ratio = exp5_cluster_ratio_percent
        valid_ratios = {exp5_cluster_ratio_percent, exp5_retry_cluster_ratio_percent}

    if base_ratio is None:
        return key_of(lines)

    normalized = []
    for line in lines:
        if line.startswith("experiment.cluster_ratio_percent="):
            current_ratio = line.split("=", 1)[1]
            if current_ratio in valid_ratios:
                line = f"experiment.cluster_ratio_percent={base_ratio}"
        normalized.append(line)
    return key_of(normalized)


def existing_index(search_root):
    attempted = {}
    done = {}
    constraint_unsatisfied = {}
    if not os.path.isdir(search_root):
        return attempted, done, constraint_unsatisfied

    pattern = os.path.join(search_root, "**", ".hydra", "overrides.yaml")
    marker_name = os.environ.get("CONSTRAINT_UNSAT_MARKER", "CONSTRAINT_UNSAT")
    error_text = os.environ.get(
        "CONSTRAINT_UNSAT_ERROR_TEXT",
        "Failed to sample a valid edge group with the current constraints.",
    )

    def is_constraint_unsat_run(run_dir):
        if os.path.isfile(os.path.join(run_dir, marker_name)):
            return True
        if os.path.isfile(os.path.join(run_dir, f"{marker_name}.txt")):
            return True
        log_path = os.path.join(run_dir, "main.log")
        if not os.path.isfile(log_path):
            return False
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as log_f:
                for line in log_f:
                    if error_text in line:
                        return True
        except OSError:
            return False
        return False

    for override_file in glob.glob(pattern, recursive=True):
        run_dir = os.path.dirname(os.path.dirname(override_file))
        with open(override_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for raw in f:
                line = raw.strip()
                if line.startswith("- "):
                    lines.append(line[2:])
        k = logical_key_of(lines)
        attempted.setdefault(k, []).append(run_dir)
        if os.path.isfile(os.path.join(run_dir, "DONE")):
            done[k] = True
        if is_constraint_unsat_run(run_dir):
            constraint_unsatisfied[k] = True
    return attempted, done, constraint_unsatisfied


def expected_rows(exp_name):
    rows = []
    if exp_name in ("exp1", "all"):
        ratios = ["1", "5", "10", "20", "30", "50", "70", "90"]
        for model, layer, dataset, ratio in product(models, layers, datasets, ratios):
            rows.append(
                (
                    "exp1",
                    [
                        "experiment=large_drop_influence",
                        f"model={model}",
                        f"model.num_layers={layer}",
                        f"dataset={dataset}",
                        f"experiment.ratio_group_elem={ratio}",
                        f"seed={seed}",
                    ],
                )
            )

    if exp_name in ("exp3", "all"):
        samplers = ["group_non_neighbor", "group_neighbor"]
        dists = ["1", "2", "3", "4", "5"]
        for model, layer, dataset, sampler, dist in product(models, layers, datasets, samplers, dists):
            rows.append(
                (
                    "exp3",
                    [
                        "experiment=non_neighbor_edges",
                        f"model={model}",
                        f"model.num_layers={layer}",
                        f"dataset={dataset}",
                        "experiment.edges_per_cluster=-1",
                        f"experiment.cluster_ratio_percent={exp3_cluster_ratio_percent}",
                        f"experiment.removal_candidate_sampler={sampler}",
                        f"experiment.removal_neighbor_dist={dist}",
                        f"seed={seed}",
                    ],
                )
            )

    if exp_name in ("exp4", "all"):
        intra_dists = ["1", "2", "3"]
        inter_dists = ["1", "2", "3"]
        for model, layer, dataset, intra_dist, inter_dist in product(models, layers, datasets, intra_dists, inter_dists):
            rows.append(
                (
                    "exp4",
                    [
                        "experiment=clusters",
                        f"model={model}",
                        f"model.num_layers={layer}",
                        f"dataset={dataset}",
                        "experiment.num_of_clusters=3",
                        "experiment.edges_per_cluster=-1",
                        f"experiment.intra_cluster_dist={intra_dist}",
                        f"experiment.inter_cluster_dist={inter_dist}",
                        f"experiment.cluster_ratio_percent={exp4_cluster_ratio_percent}",
                        f"seed={seed}",
                    ],
                )
            )

    if exp_name in ("exp5", "all"):
        modes = ["calculate_influence", "clusterwise_step_by_step"]
        for model, layer, dataset, mode in product(models, layers, datasets, modes):
            rows.append(
                (
                    "exp5",
                    [
                        "experiment=influence_calc",
                        f"model={model}",
                        f"model.num_layers={layer}",
                        f"dataset={dataset}",
                        "experiment.name=clusters",
                        "experiment.num_of_clusters=3",
                        "experiment.edges_per_cluster=-1",
                        "experiment.intra_cluster_dist=1",
                        "experiment.inter_cluster_dist=2",
                        f"experiment.cluster_ratio_percent={exp5_cluster_ratio_percent}",
                        f"experiment.influence_mode={mode}",
                        f"seed={seed}",
                    ],
                )
            )

    return rows


if exp_selector not in {"exp1", "exp3", "exp4", "exp5", "all"}:
    raise SystemExit(f"invalid experiment selector: {exp_selector}")

expected = expected_rows(exp_selector)
attempted_idx, done_idx, constraint_unsat_idx = existing_index(root)

stats = defaultdict(lambda: {"total": 0, "done": 0, "constraint_unsatisfied": 0, "failed": 0, "unattempted": 0})

for exp_id, overrides in expected:
    k = logical_key_of(overrides)
    row = stats[exp_id]
    row["total"] += 1
    if done_idx.get(k, False):
        row["done"] += 1
    elif constraint_unsat_idx.get(k, False):
        row["constraint_unsatisfied"] += 1
    elif k in attempted_idx:
        row["failed"] += 1
    else:
        row["unattempted"] += 1


def print_row(label, row):
    total = row["total"]
    done = row["done"]
    constraint_unsatisfied = row["constraint_unsatisfied"]
    failed = row["failed"]
    unattempted = row["unattempted"]
    missing = failed + unattempted
    completed = done + constraint_unsatisfied
    percent = 0.0 if total == 0 else (100.0 * completed) / total
    print(
        f"[{label}] total={total} done={done} constraint_unsatisfied={constraint_unsatisfied} "
        f"failed={failed} missing={missing} "
        f"unattempted={unattempted} percent={percent:.2f}%"
    )


exp_order = ["exp1", "exp3", "exp4", "exp5"]
if exp_selector == "all":
    for exp_id in exp_order:
        if stats[exp_id]["total"] > 0:
            print_row(exp_id, stats[exp_id])
else:
    print_row(exp_selector, stats[exp_selector])

overall = {"total": 0, "done": 0, "constraint_unsatisfied": 0, "failed": 0, "unattempted": 0}
for row in stats.values():
    for k in overall:
        overall[k] += row[k]
print_row("overall", overall)
PY
}

rerun_missing() {
  local exp_selector="${1:-}"
  local root="${2:-$SEARCH_ROOT_DEFAULT}"
  local rerun_num_gpus="${RERUN_NUM_GPUS:-4}"
  local rerun_max_parallel="${RERUN_MAX_PARALLEL:-$rerun_num_gpus}"
  local count=0
  local failed=0
  local unattempted=0
  local success=0
  local launch_failed=0
  local constraint_unsatisfied=0
  local launched=0
  local -a pids=()
  local -a run_dirs=()
  local -a mark_done_flags=()

  if [[ -z "$exp_selector" ]]; then
    echo "error: rerun-missing requires experiment selector: exp1|exp3|exp4|exp5|all"
    return 1
  fi
  if ! [[ "$rerun_num_gpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: RERUN_NUM_GPUS must be a positive integer (current: $rerun_num_gpus)"
    return 1
  fi
  if ! [[ "$rerun_max_parallel" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: RERUN_MAX_PARALLEL must be a positive integer (current: $rerun_max_parallel)"
    return 1
  fi

  mkdir -p "$root/_rerun_missing/$exp_selector"

  while IFS=$'\t' read -r -a row; do
    [[ "${#row[@]}" -lt 3 ]] && continue
    local status="${row[0]}"
    local run_dir="${row[1]}"
    local overrides=("${row[@]:2}")
    local label="$status"
    local mark_done="0"
    local target_run_dir="$run_dir"

    if [[ "$status" == "failed" ]]; then
      failed=$((failed + 1))
      label="rerun-failed"
      mark_done="1"
    else
      unattempted=$((unattempted + 1))
      label="run-unattempted"
      target_run_dir="$root/_rerun_missing/$exp_selector/job$(date +%s%N)_$launched"
    fi

    prepare_run_dir_for_rerun "$target_run_dir"

    local gpu_id
    gpu_id=$((launched % rerun_num_gpus))
    echo "[$label gpu${gpu_id}] ${overrides[*]}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu_id"
      export EIF_RESPECT_CUDA_VISIBLE_DEVICES=1
      "$PYTHON_BIN" main.py "${overrides[@]}" "hydra.run.dir=$target_run_dir"
    ) &
    pids+=("$!")
    run_dirs+=("$target_run_dir")
    mark_done_flags+=("$mark_done")
    launched=$((launched + 1))
    count=$((count + 1))

    if (( ${#pids[@]} >= rerun_max_parallel )); then
      if wait "${pids[0]}"; then
        if is_constraint_unsat_run_dir "${run_dirs[0]}"; then
          constraint_unsatisfied=$((constraint_unsatisfied + 1))
        else
          if [[ "${mark_done_flags[0]}" == "1" && "${run_dirs[0]}" != "-" ]]; then
            touch "${run_dirs[0]}/DONE" 2>/dev/null || true
          fi
          success=$((success + 1))
        fi
      else
        launch_failed=$((launch_failed + 1))
      fi
      pids=("${pids[@]:1}")
      run_dirs=("${run_dirs[@]:1}")
      mark_done_flags=("${mark_done_flags[@]:1}")
    fi
  done < <(collect_missing_matrix_rows "$exp_selector" "$root")

  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      if is_constraint_unsat_run_dir "${run_dirs[$idx]}"; then
        constraint_unsatisfied=$((constraint_unsatisfied + 1))
      else
        if [[ "${mark_done_flags[$idx]}" == "1" && "${run_dirs[$idx]}" != "-" ]]; then
          touch "${run_dirs[$idx]}/DONE" 2>/dev/null || true
        fi
        success=$((success + 1))
      fi
    else
      launch_failed=$((launch_failed + 1))
    fi
  done

  echo "rerun_count=$count success=$success constraint_unsatisfied=$constraint_unsatisfied failed_runs=$launch_failed (target_failed=$failed, target_unattempted=$unattempted, parallel=$rerun_max_parallel, gpus=$rerun_num_gpus)"
  if (( launch_failed > 0 )); then
    return 1
  fi
}

usage() {
  cat <<'EOF'
Usage:
  ./run_experiments.sh exp1
  ./run_experiments.sh exp3
  ./run_experiments.sh exp4
  ./run_experiments.sh exp5
  ./run_experiments.sh all
  ./run_experiments.sh exp1 [full|small]
  ./run_experiments.sh exp3 [full|small]
  ./run_experiments.sh exp4 [full|small]
  ./run_experiments.sh exp5 [full|small]
  ./run_experiments.sh all [full|small]
  ./run_experiments.sh init-exp4-candidates [full|small] [force]
  ./run_experiments.sh init-exp5-candidates [full|small] [force]
  ./run_experiments.sh init-cluster-candidates [full|small] [force]
  ./run_experiments.sh label-cluster-feasibility [full|small] [output_csv]
  ./run_experiments.sh witness-cluster-feasibility [full|small] [timeout_sec]
  ./run_experiments.sh list-failed [search_root]
  ./run_experiments.sh rerun-failed [search_root]
  ./run_experiments.sh list-constraint-unsatisfied [search_root]
  ./run_experiments.sh rerun-constraint-unsatisfied <exp1|exp3|exp4|exp5|all> [full|small] [rerun_cluster_ratio] [search_root]
  ./run_experiments.sh list-missing <exp1|exp3|exp4|exp5|all> [full|small] [search_root]
  ./run_experiments.sh rerun-missing <exp1|exp3|exp4|exp5|all> [full|small] [search_root]
  ./run_experiments.sh progress [search_root]
  ./run_experiments.sh summary <exp1|exp3|exp4|exp5|all> [full|small] [search_root]
  ./run_experiments.sh watch-progress [search_root] [interval_sec]

Examples:
  ./run_experiments.sh exp1
  ./run_experiments.sh exp1 small
  ./run_experiments.sh all small
  ./run_experiments.sh init-exp4-candidates small
  ./run_experiments.sh init-exp5-candidates small force
  ./run_experiments.sh label-cluster-feasibility small
  ./run_experiments.sh witness-cluster-feasibility small 90
  ./run_experiments.sh list-failed
  ./run_experiments.sh rerun-failed multirun/large_drop_influence
  ./run_experiments.sh list-constraint-unsatisfied
  ./run_experiments.sh rerun-constraint-unsatisfied all
  ./run_experiments.sh rerun-constraint-unsatisfied exp4 small
  ./run_experiments.sh rerun-constraint-unsatisfied exp3 small 9
  ./run_experiments.sh rerun-constraint-unsatisfied exp4 small 15
  ./run_experiments.sh list-missing exp1
  ./run_experiments.sh list-missing exp1 small
  ./run_experiments.sh rerun-missing exp1
  ./run_experiments.sh rerun-missing exp1 small
  ./run_experiments.sh progress
  ./run_experiments.sh summary exp1 small
  ./run_experiments.sh watch-progress multirun 5
EOF
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    exp1)
      apply_profile "${2:-full}"
      run_exp1
      ;;
    exp3)
      apply_profile "${2:-full}"
      run_exp3
      ;;
    exp4)
      apply_profile "${2:-full}"
      run_exp4
      ;;
    exp5)
      apply_profile "${2:-full}"
      run_exp5
      ;;
    all)
      apply_profile "${2:-full}"
      run_exp1
      run_exp3
      run_exp4
      run_exp5
      ;;
    init-exp4-candidates)
      apply_profile "${2:-full}"
      if [[ "${3:-}" == "force" ]]; then
        init_exp4_candidates 1
      else
        init_exp4_candidates 0
      fi
      ;;
    init-exp5-candidates)
      apply_profile "${2:-full}"
      if [[ "${3:-}" == "force" ]]; then
        init_exp5_candidates 1
      else
        init_exp5_candidates 0
      fi
      ;;
    init-cluster-candidates)
      apply_profile "${2:-full}"
      if [[ "${3:-}" == "force" ]]; then
        init_exp4_candidates 1
        init_exp5_candidates 1
      else
        init_exp4_candidates 0
        init_exp5_candidates 0
      fi
      ;;
    label-cluster-feasibility)
      apply_profile "${2:-full}"
      label_cluster_feasibility "${3:-candidate_cache/cluster_feasibility/cluster_feasibility_${2:-full}.csv}"
      ;;
    witness-cluster-feasibility)
      apply_profile "${2:-full}"
      witness_cluster_feasibility "${3:-90}"
      ;;
    list-failed)
      list_failed "${2:-$SEARCH_ROOT_DEFAULT}"
      ;;
    rerun-failed)
      rerun_failed "${2:-$SEARCH_ROOT_DEFAULT}"
      ;;
    list-constraint-unsatisfied)
      list_constraint_unsatisfied "${2:-$SEARCH_ROOT_DEFAULT}"
      ;;
    rerun-constraint-unsatisfied)
      if [[ "${3:-}" == "small" || "${3:-}" == "full" ]]; then
        apply_profile "${3:-full}"
        if [[ "${4:-}" =~ ^[1-9][0-9]*$ ]]; then
          rerun_constraint_unsatisfied "${2:-all}" "${5:-$SEARCH_ROOT_DEFAULT}" "${4:-}"
        else
          rerun_constraint_unsatisfied "${2:-all}" "${4:-$SEARCH_ROOT_DEFAULT}"
        fi
      else
        apply_profile "full"
        if [[ "${3:-}" =~ ^[1-9][0-9]*$ ]]; then
          rerun_constraint_unsatisfied "${2:-all}" "${4:-$SEARCH_ROOT_DEFAULT}" "${3:-}"
        else
          rerun_constraint_unsatisfied "${2:-all}" "${3:-$SEARCH_ROOT_DEFAULT}"
        fi
      fi
      ;;
    list-missing)
      if [[ "${3:-}" == "small" || "${3:-}" == "full" ]]; then
        apply_profile "${3:-full}"
        list_missing "${2:-}" "${4:-$SEARCH_ROOT_DEFAULT}"
      else
        apply_profile "full"
        list_missing "${2:-}" "${3:-$SEARCH_ROOT_DEFAULT}"
      fi
      ;;
    rerun-missing)
      if [[ "${3:-}" == "small" || "${3:-}" == "full" ]]; then
        apply_profile "${3:-full}"
        rerun_missing "${2:-}" "${4:-$SEARCH_ROOT_DEFAULT}"
      else
        apply_profile "full"
        rerun_missing "${2:-}" "${3:-$SEARCH_ROOT_DEFAULT}"
      fi
      ;;
    summary)
      if [[ "${3:-}" == "small" || "${3:-}" == "full" ]]; then
        apply_profile "${3:-full}"
        show_summary "${2:-}" "${4:-$SEARCH_ROOT_DEFAULT}"
      else
        apply_profile "full"
        show_summary "${2:-}" "${3:-$SEARCH_ROOT_DEFAULT}"
      fi
      ;;
    progress)
      show_progress "${2:-$SEARCH_ROOT_DEFAULT}"
      ;;
    watch-progress)
      watch_progress "${2:-$SEARCH_ROOT_DEFAULT}" "${3:-10}"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
