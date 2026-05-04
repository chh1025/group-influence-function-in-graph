# METIS Partitioning Process

Reference code:
- [main.py](/home/undergrad_hh/group-influence-function-in-graph/main.py:401)
- [candidate_partition.py](/home/undergrad_hh/group-influence-function-in-graph/candidate_partition.py:9)
- [candidate_partition_graph.py](/home/undergrad_hh/group-influence-function-in-graph/candidate_partition_graph.py:7)
- [partition_methods.py](/home/undergrad_hh/group-influence-function-in-graph/partition_methods.py:7)
- [src/utils.py](/home/undergrad_hh/group-influence-function-in-graph/src/utils.py:1690)

## 한 줄 요약

현재 `METIS`는 **training graph 전체를 직접 clustering하는 것이 아니라**,
**candidate 내부 edge들을 노드로 하는 candidate-level affinity graph**를 만든 뒤 그 그래프 위에서 돌립니다.

training graph는 partition 대상이 아니라, **candidate edge 사이 affinity를 계산하기 위한 구조적 context**로만 사용됩니다.

## 1. 어디서 켜지나

`partition` 기반 cluster-sum 경로는 `metric_mode=partition`일 때 활성화됩니다.

- `main.py`에서
  - `--metric_mode` 기본값은 `global`
  - `--partition_method` 기본값은 `metis`
  - 관련 인자는
    - `partition_shared_endpoint_bonus`
    - `partition_distance_scale`
    - `partition_distance_max_hops`
    - `partition_min_weight`
  로 정의돼 있습니다 ([main.py](/home/undergrad_hh/group-influence-function-in-graph/main.py:401)).

실제 clusterer 등록은 [src/utils.py](/home/undergrad_hh/group-influence-function-in-graph/src/utils.py:1893)에서 일어납니다.

## 2. Partition 대상은 무엇인가

`prepare_candidate_partition_clusterer(...)`는 입력 `candidates`가 아래 shape이라고 가정합니다.

- `[num_candidates, num_group_elem, 2]`

여기서:

- `num_candidates`: 평가할 candidate set 개수
- `num_group_elem`: candidate 하나 안에 들어 있는 edge 개수
- 마지막 `2`: edge endpoint pair

즉 **candidate 하나는 여러 edge로 이루어진 edge set**입니다.
이 함수는 [candidate_partition.py](/home/undergrad_hh/group-influence-function-in-graph/candidate_partition.py:19)에서 **candidate를 하나씩 순회**합니다.

따라서 partition은:

- 전체 training graph에 한 번 수행되는 것이 아니고
- **candidate 하나마다 따로 수행**됩니다.

## 3. Candidate-level affinity graph를 어떻게 만드나

`build_candidate_affinity_graph(...)`가 실제 affinity graph를 만듭니다 ([candidate_partition_graph.py](/home/undergrad_hh/group-influence-function-in-graph/candidate_partition_graph.py:7)).

### 3.1 노드

affinity graph의 노드는:

- training graph의 node가 아니라
- **candidate 안에 포함된 각 edge**

입니다.

즉 candidate에 edge가 `m`개 있으면 affinity matrix는 `m x m`입니다 ([candidate_partition_graph.py](/home/undergrad_hh/group-influence-function-in-graph/candidate_partition_graph.py:12), [candidate_partition_graph.py](/home/undergrad_hh/group-influence-function-in-graph/candidate_partition_graph.py:28)).

### 3.2 training graph는 어디에 쓰이나

training graph는 [candidate_partition_graph.py](/home/undergrad_hh/group-influence-function-in-graph/candidate_partition_graph.py:21)에서 undirected adjacency로 바뀝니다.

그 다음 candidate edge들의 endpoint node들에 대해 bounded BFS distance cache를 만듭니다 ([candidate_partition_graph.py](/home/undergrad_hh/group-influence-function-in-graph/candidate_partition_graph.py:22), [candidate_partition_graph.py](/home/undergrad_hh/group-influence-function-in-graph/candidate_partition_graph.py:75)).

즉 training graph는:

- partition할 대상이 아니라
- candidate edge pair 간의 구조적 거리를 계산하는 reference graph입니다.

### 3.3 edge pair affinity

candidate edge `edge_i`, `edge_j` 사이 affinity는 [candidate_partition_graph.py](/home/undergrad_hh/group-influence-function-in-graph/candidate_partition_graph.py:93)에서 계산됩니다.

현재 weight는 대략 아래 두 항의 합입니다.

1. **shared endpoint bonus**
- 두 edge가 endpoint를 공유하면 bonus 부여
- 크기는 `partition_shared_endpoint_bonus`

2. **endpoint distance bonus**
- 두 edge의 endpoint 쌍 사이 minimum hop distance를 training graph에서 찾음
- 거리 항은 대략 `distance_scale / (1 + min_dist)`

그리고 weight가 `partition_min_weight`보다 작으면 edge를 affinity graph에 넣지 않습니다 ([candidate_partition_graph.py](/home/undergrad_hh/group-influence-function-in-graph/candidate_partition_graph.py:40)).

## 4. METIS는 어디서 도나

candidate affinity matrix가 준비되면 [candidate_partition.py](/home/undergrad_hh/group-influence-function-in-graph/candidate_partition.py:21)에서 `partition_affinity_graph(...)`로 넘깁니다.

`method == "metis"`이면 [partition_methods.py](/home/undergrad_hh/group-influence-function-in-graph/partition_methods.py:23)에서 `_partition_with_metis(...)`가 호출됩니다.

그 안에서:

- affinity matrix의 positive entries를 adjacency list로 바꾸고
- weight를 정수 edge weight로 변환한 뒤
- `pymetis.part_graph(...)`를 호출합니다 ([partition_methods.py](/home/undergrad_hh/group-influence-function-in-graph/partition_methods.py:39), [partition_methods.py](/home/undergrad_hh/group-influence-function-in-graph/partition_methods.py:50)).

중요한 점:

- `pymetis`가 받는 그래프는 **training graph가 아님**
- **candidate edges를 node로 하는 affinity graph**임

## 5. K=1이면 어떻게 되나

`num_clusters == 1`이면 [partition_methods.py](/home/undergrad_hh/group-influence-function-in-graph/partition_methods.py:19)에서 바로:

- 전체 candidate edge index를 하나의 group으로 반환합니다.

그래서 `K=1`은 사실상 partition을 하지 않는 baseline과 동일해야 하고, 실제로 recent K-sweep 결과에서도 그렇게 나왔습니다.

## 6. Affinity edge가 하나도 없으면 어떻게 되나

METIS 경로에서는 positive affinity edge가 하나도 없으면 [partition_methods.py](/home/undergrad_hh/group-influence-function-in-graph/partition_methods.py:35)에서:

- `_balanced_partition(...)` fallback을 사용합니다.

즉 이 경우는 구조적 연결성이 없는 candidate edge들을 단순 균등 분할에 가깝게 나누게 됩니다.

## 7. Partition 결과는 어떻게 influence 계산으로 이어지나

`prepare_candidate_partition_clusterer(...)`는 group index를 실제 edge cluster tensor로 바꿉니다 ([candidate_partition.py](/home/undergrad_hh/group-influence-function-in-graph/candidate_partition.py:30)).

그 후 `calculate_clusterwise_fixed_theta_influence(...)`에서 candidate 하나마다:

1. cluster list를 얻고
2. 각 cluster를 개별 candidate처럼 `influence_module.calculate_influence(...)`에 넣고
3. cluster별 influence를 합산합니다 ([src/utils.py](/home/undergrad_hh/group-influence-function-in-graph/src/utils.py:1703)).

즉 pipeline은:

1. candidate 하나 선택
2. candidate edge들로 affinity graph 생성
3. METIS로 edge cluster 분할
4. cluster별 influence 계산
5. 합산

입니다.

## 8. 오해하기 쉬운 점

### 오해 1: training graph 전체를 METIS로 나눈다

아닙니다.

- training graph 전체 node/edge를 partition하지 않습니다.
- candidate 안의 edge들만 partition합니다.

### 오해 2: 모든 candidate를 한꺼번에 하나의 큰 graph로 partition한다

아닙니다.

- candidate마다 독립적으로 affinity graph를 만들고
- candidate마다 독립적으로 partition합니다.

### 오해 3: training graph가 무시된다

아닙니다.

- training graph는 affinity 계산의 핵심 입력입니다.
- endpoint 연결성과 hop distance는 training graph에서 옵니다.

## 9. 실제로 무엇을 말할 수 있나

현재 구현을 정확히 표현하면:

> We do not partition the full training graph.
> For each candidate edge set, we construct a candidate-level affinity graph whose nodes are candidate edges and whose weights are derived from structural relations in the original training graph, then run METIS on that candidate graph.

이 문장이 현재 코드와 가장 잘 맞습니다.
