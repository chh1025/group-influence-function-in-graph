# Hydra + Submitit Local 멀티런 세팅 요청서 (4GPU, GPU당 동시 1 job)

## 0) 목표
- 단일 서버 **GPU 4장**에서
- **DDP 없이**, Hydra multirun으로 많은 조합 실험을 수행한다.
- **GPU 한 장당 동시에 1개의 실험만 실행**되도록 하고, 어떤 GPU에서 실행이 끝나면 **큐에서 다음 실험이 자동 투입**되게 구성한다.
- 설정 계층은 아래 3단계로 운영한다:
  1) **최상위 default**(공통 설정)
  2) **실험별 default**(experiment group)
  3) **실행 시 CLI override + multirun sweep**(bash 단)

---

## 1) 실험 스펙 (첨부 문서 기반)
### 공통: Models × Layers
- Model: `SGC | GCN | GAT | ChebNet`
- num_layers: `2, 4, 6, 12`

### 공통: Datasets
Homophilic:
- `Cora, CiteSeer, PubMed, Computers, Photo`
Heterophilic:
- `Chameleon, Actor, Squirrel, Texas, Cornell`

모든 실험은 기본적으로 **모든 모델 × 모든 레이어 × 모든 데이터셋** 조합에 대해 실행 가능해야 한다.

---

## 2) 실험별 정의

### 실험1: large drop influence
- 각 모델/데이터셋에 대해 edge drop 비율 sweep
- drop 비율(%): `1, 5, 10, 20, 30, 50, 70, 90`
- 파라미터 이름(입력): `experiment.ratio_group_elem` (int, “퍼센트”)

#### 중요: 퍼센트 → 실제 edge 개수 변환 로직 필요
- 데이터셋마다 전체 edge 개수가 다르므로, 퍼센트 기반으로 제거할 edge 개수 `k`를 계산하는 로직이 필요하다.
- 구현 제안(명시):
  - `total_edges = <dataset edge count>`
  - `k = max(1, round(total_edges * (ratio_group_elem / 100.0)))`
  - `k`는 실제 샘플링/제거 로직에 사용
- config는 퍼센트만 받고(`ratio_group_elem`), 내부에서 `k`로 변환하는 방식으로 구현할 것.

---

### 실험3: non neighbor edges
- 파라미터:
  - `experiment.removal_candidate_sampler`: `"group_non_neighbor"` or `"group_neighbor"`
  - `experiment.removal_neighbor_dist`: int (1~5)

---

### 실험4: clusters
- cluster 관련 파라미터:
  - `experiment.num_of_clusters`
  - `experiment.edges_per_cluster`
  - `experiment.removal_cluster_dist`  (cluster 정의 시 edges 간 최대 거리)
  - `experiment.removal_candidate_sampler`: `"group_non_neighbor"` or `"group_neighbor"`
  - `experiment.removal_neighbor_dist`: int (1~5)

#### 실험4 기본값(요청사항 반영)
- (사용자 확정) **`num_of_clusters = 3`**
- “10%”는 **전체 edge 중 제거 대상 규모가 10%**임을 의미 (데이터셋마다 전체 edge 수가 다름)

##### 10% 제거 규모를 3개 클러스터로 분배하는 기본 해석(명시)
- `target_total_edges_to_remove = round(total_edges * 0.10)`
- `num_of_clusters = 3` (고정)
- `edges_per_cluster`는 기본적으로 아래로 계산(정수):
  - `edges_per_cluster = max(1, floor(target_total_edges_to_remove / num_of_clusters))`
- 남는 찌꺼지(`target_total_edges_to_remove - num_of_clusters*edges_per_cluster`)는
  - **기본 구현: 버림** (즉, 실제 제거 edge 수는 `num_of_clusters*edges_per_cluster`)
  - (옵션) 더 정확히 맞추고 싶다면: 마지막 클러스터에 remainder를 더하는 정책을 추가할 수 있음(요구사항에는 없음)

> NOTE: 만약 실험 코드가 `edges_per_cluster`를 반드시 config로 요구한다면, 위 계산값을 런타임에 주입하거나(코드에서 override), `experiment.edges_per_cluster`를 config에서 비워두고 코드에서 결정하도록 구현해도 된다.

- `removal_cluster_dist`의 기본값은 문서/코드 베이스에 맞춰 설정(불명확하면 TODO로 두고 CLI override로 조정 가능하게).

---

### 실험5: influence 계산 방식
- mode 2가지:
  1) `fixed_theta`: 각 cluster influence 계산 후 합산
  2) `clusterwise_step_by_step`: parameter shift를 명시적으로 누적, **모든 순서 permutation** 실험

- 파라미터:
  - `experiment.influence_mode`: `"fixed_theta"` or `"clusterwise_step_by_step"`

#### permutation 범위(명확화)
- `clusterwise_step_by_step`에서 permutation은 “모든 순서”를 의미하며,
- 기본값이 `num_of_clusters = 3`이므로 permutation 수는 `3! = 6`
- 즉, 이 모드에서는 **6가지 순서를 모두 실행**하도록 구현

---

## 3) 구현 요구사항: Hydra 설정

### 3.1) Config 디렉토리 구조(권장)
프로젝트 루트에 아래 구조로 생성:

```
conf/
  config.yaml
  hydra/
    launcher/
      submitit_local.yaml
  model/
    sgc.yaml
    gcn.yaml
    gat.yaml
    chebnet.yaml
  dataset/
    cora.yaml
    citeseer.yaml
    pubmed.yaml
    computers.yaml
    photo.yaml
    chameleon.yaml
    actor.yaml
    squirrel.yaml
    texas.yaml
    cornell.yaml
  experiment/
    large_drop_influence.yaml
    non_neighbor_edges.yaml
    clusters.yaml
    influence_calc.yaml
```

### 3.2) 최상위 config: `conf/config.yaml`
- `defaults:`로 아래 포함:
  - model group (기본값 1개)
  - dataset group (기본값 1개)
  - experiment group (기본값 1개)
  - `hydra/launcher: submitit_local`
- 공통 파라미터 예시:
  - `seed`
  - `train.*` (epochs, lr, weight_decay 등)
  - `model.*` (num_layers, hidden_dim, dropout, etc)
  - `dataset.*` (name, kind 등)
- multirun 결과 정리를 위해 `hydra.sweep.dir` 및 `hydra.sweep.subdir` 설정 필요

#### 출력 디렉토리(subdir) 요구사항
- “정말 필요하고 중요한 정보”만 포함해서 **짧고 알아보기 쉬운** 형태로 구성
- 최소 포함:
  - experiment.name
  - dataset.name
  - model.name
  - model.num_layers
  - experiment 핵심 sweep 변수(실험별 1~2개)

예시 subdir 포맷(권장):
- 실험1: `${experiment.name}/${dataset.name}/${model.name}/L${model.num_layers}/r${experiment.ratio_group_elem}/job${hydra.job.num}`
- 실험3: `${experiment.name}/${dataset.name}/${model.name}/L${model.num_layers}/${experiment.removal_candidate_sampler}-d${experiment.removal_neighbor_dist}/job${hydra.job.num}`
- 실험4: `${experiment.name}/${dataset.name}/${model.name}/L${model.num_layers}/c${experiment.num_of_clusters}/${experiment.removal_candidate_sampler}-d${experiment.removal_neighbor_dist}/job${hydra.job.num}`
- 실험5: `${experiment.name}/${dataset.name}/${model.name}/L${model.num_layers}/${experiment.influence_mode}/job${hydra.job.num}`

---

## 4) Submitit Local Launcher 요구사항 (4GPU 큐잉)
### `conf/hydra/launcher/submitit_local.yaml`
- Hydra Submitit Launcher 플러그인의 LocalLauncher 사용
- 동시에 실행되는 job 수를 **4로 제한**(GPU 4장)
- job 하나가 사용할 CPU는 적당히(예: 4)

예시(버전 차이가 있을 수 있으니, 동작하는 키로 맞춰 구현):

```yaml
_target_: hydra_plugins.hydra_submitit_launcher.submitit_launcher.LocalLauncher
tasks_per_node: 4
cpus_per_task: 4
timeout_min: 0
```

---

## 5) GPU 바인딩 요구사항 (중요)
각 multirun job이 서로 다른 GPU 하나만 보도록 강제해야 한다.
- 방식: `job_num % 4`로 GPU id 결정
- **torch import 이전**에 `CUDA_VISIBLE_DEVICES` 설정

요구 로직:
- `job_num = int(os.environ.get("HYDRA_JOB_NUM", "0"))`
- `gpu_id = job_num % 4`
- `os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)`

환경변수 미존재 시 fallback:
- HydraConfig 또는 cfg에서 job num을 읽어 대체(플러그인 버전에 따라 다를 수 있음)

---

## 6) 실행 커맨드 예시 (필수로 동작 확인)

### 실험1 전체 sweep
```bash
python main.py -m \
  experiment=large_drop_influence \
  model=sgc,gcn,gat,chebnet \
  model.num_layers=2,4,6,12 \
  dataset=cora,citeseer,pubmed,computers,photo,chameleon,actor,squirrel,texas,cornell \
  experiment.ratio_group_elem=1,5,10,20,30,50,70,90 \
  seed=0
```

### 실험3 전체 sweep
```bash
python main.py -m \
  experiment=non_neighbor_edges \
  model=sgc,gcn,gat,chebnet \
  model.num_layers=2,4,6,12 \
  dataset=cora,citeseer,pubmed,computers,photo,chameleon,actor,squirrel,texas,cornell \
  experiment.removal_candidate_sampler=group_non_neighbor,group_neighbor \
  experiment.removal_neighbor_dist=1,2,3,4,5 \
  seed=0
```

### (옵션) 실험4/5 예시 (형태만)
```bash
python main.py -m \
  experiment=clusters \
  model=sgc,gcn,gat,chebnet \
  model.num_layers=2,4,6,12 \
  dataset=cora,citeseer,pubmed,computers,photo,chameleon,actor,squirrel,texas,cornell \
  experiment.num_of_clusters=3 \
  experiment.removal_candidate_sampler=group_non_neighbor,group_neighbor \
  experiment.removal_neighbor_dist=1,2,3,4,5
```

```bash
python main.py -m \
  experiment=influence_calc \
  model=sgc,gcn,gat,chebnet \
  model.num_layers=2,4,6,12 \
  dataset=cora,citeseer,pubmed,computers,photo,chameleon,actor,squirrel,texas,cornell \
  experiment.influence_mode=fixed_theta,clusterwise_step_by_step
```

---

## 7) 실패 추적(권장)
- 각 run 디렉토리에 `.hydra/config.yaml`, `.hydra/overrides.yaml`가 생성되도록 유지(Hydra 기본)
- 가능하면 run별 `run.log` 파일로 stdout/stderr 로깅을 남길 것
- 성공 종료 시 `DONE` 파일 생성:
  - 중단/실패 후 “DONE 없는 것만 재실행”이 쉬움

---

## 8) Codex 산출물(요구)
1) 위 구조대로 `conf/` 파일 세트 생성
2) `main.py`(또는 entrypoint)에 GPU 바인딩 로직 포함
3) 작은 multirun(예: 8개)으로 “동시에 4개씩 실행되는지” 확인 가이드 제공
4) 실험1에서 ratio(%) → 실제 edge 개수 변환 로직을 코드에 명확히 포함
5) 실험4에서 “10% 제거 규모를 num_of_clusters=3으로 분배” 로직을 코드에 주석과 함께 구현
