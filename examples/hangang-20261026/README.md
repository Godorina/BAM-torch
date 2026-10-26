# hangang 2026-10-26 runs (KISTI hangang, GH200)

2026-10-26에 hangang에서 돌린 omat24 학습 잡(`omat24-fusion`)을 똑같이 재현하기 위한 폴더입니다.
학습 코드는 이 브랜치(`hangang-speedup`)의 `bam_torch/`이고, 기준 커밋은 myung-group `equivariant-mve` 브랜치의 `aa9d0c8`입니다.

| run | GPU | 데이터 | 모델 | 에폭 |
|---|---|---|---|---|
| `runs/omat24-fusion` | 64 nodes x 4 = 256 | omat24 (`train_mix` 5216 shards) | `128x0e+64x1o+32x2e+32x3o`, 5 layers, cutoff 6, features 64 | 10 |

설정: RACE 모델(`trainer: mp_v1`), DeNS 켜짐(확률 0.4), huber loss(delta 0.01, `enr_lambda 10`, `frc_lambda 1`, `str_lambda 10`), Adam lr 0.01, ReduceLROnPlateau, EMA, `nbatch 30`, `max_edges_per_batch 15000`, `shard_group 4`, `prefetch_next_shard`, `shard_frames_by_rank`, `train_save_every 50`, `valid_interval 80`, 2에폭마다 `model_epoch<N>.pkl` 저장(`epoch_save_interval 2`).

## 1. 환경

hangang(SLES 15 SP6, aarch64, driver 570.124.06)에서 쓴 환경입니다. 정확한 버전은 `env/`에 있습니다.

- `env/conda-packages.txt`: conda 패키지 (python 3.13.17, conda-forge, linux-aarch64)
- `env/requirements.lock.txt`: `pip freeze` 결과 (BAM-torch 자신은 뺐음)
- `env/versions.txt`: python, torch 2.11.0+cu128, CUDA 12.8, NCCL 2.28.9, e3nn 0.6.0, ase 3.28.0

```bash
conda create -p $HOME/prog/envs/bam -c conda-forge python=3.13.17
$HOME/prog/envs/bam/bin/pip install -r env/requirements.lock.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
git clone -b hangang-speedup https://github.com/Godorina/BAM-torch $HOME/prog/BAM-torch
$HOME/prog/envs/bam/bin/pip install -e $HOME/prog/BAM-torch --no-deps
```

잡 안에서는 `segment.sh`가 아래를 설정합니다 (hangang 모듈 이름 기준):

```bash
module load aws-ofi-nccl/1.20.0   # NCCL over Slingshot
module load cuda/12.8
export CC=gcc-13 CXX=g++-13 TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR=$HOME/prog/torch_extensions   # OpenEquivariance JIT 빌드 결과
```

OpenEquivariance 확장은 첫 실행 때 빌드되므로, 한 노드에서 먼저 한 번 import해 두면 잡이 빌드를 기다리지 않습니다.

## 2. 데이터

데이터는 크기 때문에 넣지 않았습니다. 같은 파일인지는 `data/manifest_omat24_*.tsv`(파일 이름, 바이트 크기)로 확인합니다.

| manifest | 파일 수 | 크기 |
|---|---|---|
| `manifest_omat24_train_mix.tsv` | 5216 | 305.6 GB |
| `manifest_omat24_val_mix.tsv` | 11 | 3.1 GB |

학습 순서는 파일 이름 순서이므로, 링크 폴더를 만드는 `data/build_links.py`도 같이 둡니다. omat24 11개 subset을 고르게 섞은 `$DATA_ROOT/omat24/train_mix/`, `val_mix/` 링크를 만듭니다(복사 없음, `DATA_ROOT` 기본값 `/scratch/$USER`).
원본은 `$DATA_ROOT/omat24/train/<subset>/*.extxyz`, `$DATA_ROOT/omat24/val/<subset>/*.extxyz`에 있어야 합니다.

E0(원소별 기준 에너지) 파일 `get_enr_avg_per_element_omat24.txt`는 run 폴더에 있습니다.

`data/`의 나머지 파일(`build_mix.py`, `build_odac_extxyz.py`, `fit_e0_split.py`, omol25/opoly26/odac25 manifest)은 같은 날 함께 돌린 다른 데이터셋용이고, 이 run에는 쓰지 않습니다.

## 3. 실행

계정마다 다른 경로는 `segment.sh`가 아래 변수로 받습니다. 기본값을 따르면 따로 설정할 것이 없습니다.

| 변수 | 기본값 | 의미 |
|---|---|---|
| `BAM_ENV` | `$HOME/prog/envs/bam` | BAM-torch를 설치한 conda 환경 |
| `DATA_ROOT` | `/scratch/$USER` | 데이터 링크 폴더(`omat24/train_mix`, `omat24/val_mix`)가 있는 곳 |
| `TORCH_EXTENSIONS_DIR` | `$HOME/prog/torch_extensions` | OpenEquivariance 빌드 결과 |

```bash
cp -r runs/omat24-fusion /scratch/$USER/omat24/calc
cd /scratch/$USER/omat24/calc
sbatch segment.sh                                    # 기본값 사용
DATA_ROOT=/scratch/other/data BAM_ENV=$HOME/envs/bam sbatch segment.sh   # 경로가 다를 때
```

`valid_data` 링크는 없으면 `segment.sh`가 `$DATA_ROOT` 아래로 만듭니다.

- 한 잡은 24시간 제한입니다. 끝나면 같은 폴더에서 `sbatch segment.sh`를 다시 하면 이어서 학습합니다.
  `prepare_segment.py`가 로그에서 마지막으로 시작한 shard를 찾아 `train_data/`를 그 shard부터 다시 만들고, 가장 최근 체크포인트를 `fname_pkl`로 올리고 `restart: true`로 바꿉니다.
- 자동으로 다음 잡을 거는 기능은 `CHAIN` 파일이 있을 때만 켜집니다 (hangang은 사용자당 잡 5개 제한).
- `shards_done.json`으로 끝낸 shard를 세고, `total_epochs`(10)에 닿으면 `STOP` 파일을 만들고 멈춥니다.
- `input.json`은 처음 시작하는 상태입니다(`restart: false`). 실행 중에는 `prepare_segment.py`와 `segment.sh`가 `restart`, `nepoch`를 고칩니다.

## 4. 측정된 속도 (shards/min)

| run | 이 브랜치 전 | gc 제거 후 | + shard_group/prefetch |
|---|---|---|---|
| omat24-fusion | 14.5 | 19.7 | 30.4 (1에폭 약 2.9시간) |

## 5. 비트 단위로 같지 않은 부분

- `deterministic: "warn_only"`: OpenEquivariance conv_fusion의 atomic scatter 같은 연산은 실행할 때마다 결과가 조금씩 다릅니다.
- 원래 잡은 2026-10-26에 중간에 여러 번 다시 시작했습니다 (코드 수정 적용). 이 폴더로 새로 돌리면 처음부터 이 브랜치 코드로 돕니다.
- 다른 클러스터에서는 `segment.sh`의 Slurm 옵션(`-p`, `--comment`, `--exclude`)과 모듈 이름을 바꿔야 합니다.
