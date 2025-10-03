#!/bin/bash -l

#SBATCH --job-name="rahmm224"           # Name that appears in queue
#SBATCH --partition=gpu
#SBATCH --nodes=1                   # Number of Nodes
#SBATCH --time=0-06:00:00           # Job duration in DD-hh:mm:ss
#SBATCH --mem=64GB                   # Requested memory
#SBATCH --mail-user=rahmm224@wfu.edu   # User email
#SBATCH --mail-type=FAIL  # Mail sent on begin, end/failure

#SBATCH --output="/home/rahmm224/AIinHealthProject/jobs_oe/job_name-%j.o"
#SBATCH --error="/home/rahmm224/AIinHealthProject/jobs_oe/job_name-%j.e"

# Request GPU resources
#SBATCH --gres=gpu:A100_80:1          # A100_80/A100_40/V100_32, gres=gpu:A100_80:1


# Load module
module load nvidia/cuda12/cuda/12.4.1
# Create virtual env
conda activate /home/rahmm224/csc790env
# run the program
dataset_name="har70+"
no_features=6
no_classes=7
window_size=128
step_size=64
batch_size=64
model_name="itransformer"
emb_dim=128
n_heads=8
e_layers=2
experiments=10
learning_rate=1e-3
weight_decay=0.01
epochs=100
CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=1
python /home/rahmm224/AIinHealthProject/code/train.py\
    --dataset_name $dataset_name\
    --no_features $no_features\
    --no_classes $no_classes\
    --window_size $window_size\
    --step_size $step_size\
    --batch_size $batch_size\
    --model_name $model_name\
    --emb_dim $emb_dim\
    --n_heads $n_heads\
    --e_layers $e_layers\
    --experiments $experiments\
    --learning_rate $learning_rate\
    --weight_decay $weight_decay\
    --epochs $epochs

conda deactivate
