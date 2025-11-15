#!/bin/bash
#SBATCH --job-name=s0p128
#SBATCH --nodes=2                      
#SBATCH --ntasks-per-node=1            
#SBATCH --gres=gpu:a5000:1              
#SBATCH --time=480:00:00
#SBATCH --mem=64G
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --partition=youlab-gpu
#SBATCH --exclusive
#SBATCH --mail-type=END,FAIL               # 可选：任务结束或失败时通知
#SBATCH --mail-user=zhenjiao.du@duke.edu        # 可选：你的邮箱地址
source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate pubgo

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12345

# Use a single srun to launch one process per node.
# why do we need to call conda again, because the above conda and activated envs won't work in the new nodes
srun bash -c "source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh && \
conda activate pubgo && \
torchrun --nproc_per_node=1 \
        --nnodes=$SLURM_NNODES \
        --node_rank=\$SLURM_PROCID \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        Model_trainining_and_evaluation.py --backend=nccl --use_syn"