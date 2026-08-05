#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --job-name=ats-7b-moe
#SBATCH --output=%x-%j.log

# Submit with: sbatch scripts/slurm_submit.sh
# Edit the #SBATCH directives above and the ats-train arguments at the
# bottom of this file for your job.

export NUM_NODES=$SLURM_NNODES
export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export JOB_ID=$SLURM_JOB_ID

srun scripts/launch.sh --config configs/7b.yaml --use-moe --use-mla
