
## Common setting
### Models
- SGC
	- 2 layers
	- 4 layers
	- 6 layers
	- 12 layers
- GCN
	- 2 layers
	- 4 layers
	- 6 layers
	- 12 layers
- GAT
	- 2 layers
	- 4 layers
	- 6 layers
	- 12 layers
- ChebNet
	- 2 layers
	- 4 layers
	- 6 layers
	- 12 layers

### Datasets

#### Homophilic
- Cora
- CiteSeer
- PubMed
- Computers
- Photo
#### Hetrophilic
- Chameleon
- Actor
- Squirrel
- Texas
- Cornell

모든 실험에 대해, 위 모든 경우의 수 실행.

## 실행 스코프 설정 (`all`, `small`)

`run_experiments.sh`에서 실험 선택자와 프로파일은 아래처럼 동작한다.

- `all`: exp1, exp3, exp4, exp5를 순서대로 모두 실행
- `small`: 축소된 모델/레이어/데이터셋 조합으로 실행
- 기본 프로파일: `full` (`small`을 지정하지 않으면 full)

예시:
- `./run_experiments.sh all`
- `./run_experiments.sh all small`
- `./run_experiments.sh exp3 small`

### `small` 프로파일 구성

- models: `gcn`, `gat` (2개)
- layers: `2`, `4` (2개)
- datasets: `cora`, `citeseer`, `texas`, `cornell` (4개)
- 기본 조합 수: `2 * 2 * 4 = 16`

### 조합 수 (small 기준)

- exp1: `16 * 8(ratio)` = `128`
- exp3: `16 * 2(sampler) * 5(dist)` = `160`
- exp4: `16 * 3(intra) * 3(inter)` = `144`
- exp5: `16 * 2(influence_mode)` = `32`
- `all small` 총합: `464`

참고:
- full 기준 기본 조합 수는 `4(models) * 4(layers) * 10(datasets) = 160`
- `all full` 총합은 `4640`

# 실험1. large drop influence
각 모델, 데이터셋에 대해

전체 edge의 1%, 5%, 10%, 20%, 30%, 50%, 70%, 90% Drop
args ratio_group_elem /int/로 제어

## 실험3. non neighbor edges

args
removal_candidate_sampler "group_non_neighbor" / "group_neighbor"
removal_neighbor_dist /int/ (1~5)

특정 조건을 만족하는 edges를 drop하는 실험.
그 조건은 edges의 endpoints 간 최소거리.
removal_neighbor_dist는 threshold
## 실험4. clusters

cluster:=edges들의 모임

num_of_clusters
edges_per_cluster
intra_cluster_dist : 한 cluster 내부 edges 간 최대 거리 (<= intra)
inter_cluster_dist : cluster 간 edges 최소 거리 (>= inter)

현재 exp4는 위 4개 인자로 구성하며, model/layer/dataset 조합과 함께 실행.

- sweep
  - intra_cluster_dist /int/ (1~3)
  - inter_cluster_dist /int/ (1~3)
  - num_of_clusters: 현재 3 (고정)
  - edges_per_cluster: 현재 -1 (자동 계산)

- edges_per_cluster 자동 계산 규칙 (`edges_per_cluster <= 0`일 때)
  - target_total_edges_to_remove = round(total_edges * cluster_ratio_percent / 100)
  - edges_per_cluster = max(1, floor(target_total_edges_to_remove / num_of_clusters))
  - num_group_elem = num_of_clusters * edges_per_cluster
  - remainder(target_total - num_group_elem)는 버림

- 제약 판정 단위
  - intra/inter 모두 "모든 edge쌍" 기준
  - intra: 같은 cluster 내부 edge쌍이 intra_cluster_dist 이내
  - inter: 서로 다른 cluster 간 edge쌍이 inter_cluster_dist 이상

참고: exp4는 기존 `removal_candidate_sampler` / `removal_neighbor_dist` sweep을 사용하지 않음.

### 실험5. influence 계산 방식 비교 (cluster candidate 고정)

exp5는 `experiment=influence_calc`를 사용하지만, candidate 생성은 아래 클러스터 조건으로 고정해서 수행.

- experiment.name = clusters
- num_of_clusters = 3
- edges_per_cluster = -1 (자동 계산)
- intra_cluster_dist = 1
- inter_cluster_dist = 2
- per_cluster_ratio_percent = 5
--> totla ratio: 15

`edges_per_cluster` 자동 계산은 exp4와 동일한 규칙을 사용.

influence 계산 방식은 아래 2가지를 sweep.

1. calculate_influence: 각 cluster influence를 계산해 합산 (기존 fixed_theta와 동일 의미)
2. clusterwise_step_by_step: parameter shift를 step별로 누적하며 influence 계산, 모든 순서 permutation 실험
