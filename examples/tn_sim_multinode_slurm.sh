#!/bin/bash

#SBATCH --job-name=tn_sim
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.out

# load the environment
export NVIDIA_TF32_OVERRIDE=0 # disable tensorcore

python -u problem_script_multinode.py -taskid $SLURM_ARRAY_TASK_ID -device 0

exit 0