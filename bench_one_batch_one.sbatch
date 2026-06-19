#!/bin/bash
#SBATCH --job-name=bench-one-batch
#SBATCH --account=liuv-pennnetworks
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

source /vast/projects/liuv/pennnetworks/ljm_data/vortex_torch/cluster_env.sh
cd "$ROOT/vortex_torch"

hostname
nvidia-smi || true
echo ">>> python: $(which python)"
echo ">>> ALGO=${ALGO} TOPK_VAL=${TOPK_VAL:-NA} BS=${BS} IN_LEN=${IN_LEN} OUT_LEN=${OUT_LEN}"

bash examples/bench_one_batch_one.sh
