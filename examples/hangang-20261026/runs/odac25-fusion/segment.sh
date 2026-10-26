#!/bin/bash
#SBATCH -J odac25-fusion
#SBATCH -p normal
#SBATCH -N 64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288   # else srun pins all ranks to CPU 0
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time 24:00:00
#SBATCH --comment="field=mse;appl=pytorch-ddp"
#SBATCH --exclude=node0736,node0319   # prolog failure / NCCL OFI domain failure
#SBATCH -o seg_%j.out
#SBATCH -e seg_%j.err
# One 24h segment of the odac25 run (32 nodes x 4 GH200 = 128 GPUs), chained:
#   prepare_segment.py rotates train_data / promotes the checkpoint / sets restart,
#   then this job queues its own successor (afterany) and trains.
# Stop the chain: `touch STOP` here (or lower MAX_SEGMENTS). Status: segment_state.json, chain.log.
cd "$SLURM_SUBMIT_DIR"
E=/home01/t510a03/prog/envs/bam
export PATH=$E/bin:$PATH
module load aws-ofi-nccl/1.20.0          # NCCL over Slingshot (libfabric/cxi)
# OpenEquivariance JIT extension prebuilt in TORCH_EXTENSIONS_DIR; ranks only load it.
module load cuda/12.8
export CC=gcc-13 CXX=g++-13 TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR=/home01/t510a03/prog/torch_extensions
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
log() { echo "$(date '+%F %T') job=$SLURM_JOB_ID $*" | tee -a chain.log; }

[ -e STOP ] && { log "STOP present, not training"; exit 0; }
SEG=$(python -c 'import json;print(json.load(open("segment_state.json"))["seg"])' 2>/dev/null || echo 0)
MAX=$(cat MAX_SEGMENTS 2>/dev/null || echo 1)
[ "$SEG" -ge "$MAX" ] && { log "segment $SEG >= MAX_SEGMENTS $MAX, chain done"; exit 0; }

# E0 (enr_avg_per_element) fitted once on the ODAC25 train split, spread over all nodes.
if [ ! -s e0_odac25_train.txt ]; then
  X=/scratch/t510a03/odac25/convert; G=/scratch/t510a03/envs/graphpkl/bin/python; rm -rf e0_parts
  srun --ntasks-per-node=1 bash -c "OPENBLAS_NUM_THREADS=1 $G $X/fit_e0_split.py part e0_parts /scratch/t510a03/odac25/raw/train --split \$SLURM_PROCID/\$SLURM_NTASKS -w 280" 2>&1 | grep -vi warn
  OPENBLAS_NUM_THREADS=1 $G $X/fit_e0_split.py solve e0_parts e0_odac25_train.txt 2>&1 | grep -vi warn | tee -a chain.log
  [ -s e0_odac25_train.txt ] || { log "E0 fit failed"; exit 1; }
fi

TRAIN_SRC=/scratch/t510a03/odac25/odac25_extxyz/train python prepare_segment.py || { log "prepare_segment failed"; exit 1; }
# Chaining is off unless a CHAIN file exists here (then queue the next 24h segment).
if [ -e CHAIN ]; then
# Queue the successor. The account allows only 5 queued+running jobs, so if that is
# full, keep retrying every 10 min in the background for as long as this job runs.
queue_next() { sbatch --parsable --dependency=afterany:$SLURM_JOB_ID segment.sh; }
if NEXT=$(queue_next); then
  log "segment $((SEG+1))/$MAX started, successor $NEXT queued"
else
  log "segment $((SEG+1))/$MAX started, successor not queued yet (job limit?), retrying every 10 min"
  ( until NEXT=$(queue_next); do sleep 600; done; log "successor $NEXT queued (retry)" ) &
fi
else
  log "segment $((SEG+1)) started (no chain: successor not queued)"
fi

HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
echo "host=$(hostname) head=$HEAD nodes=$SLURM_NNODES start=$(date)"
srun --ntasks-per-node=1 python -m torch.distributed.run \
    --nnodes="$SLURM_NNODES" --nproc-per-node=4 \
    --rdzv-backend=c10d --rdzv-endpoint="$HEAD:29500" --rdzv-id="$SLURM_JOB_ID" \
    main_gpu.py
log "training exit=$?"
