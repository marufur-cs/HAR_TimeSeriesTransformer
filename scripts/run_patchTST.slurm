#!/bin/bash -l

#SBATCH --job-name="rahmm224"           # Name that appears in queue
#SBATCH --partition=gpu
#SBATCH --nodes=1                   # Number of Nodes
#SBATCH --time=0-01:00:00           # Job duration in DD-hh:mm:ss
#SBATCH --mem=16GB                   # Requested memory
#SBATCH --mail-user=rahmm224@wfu.edu   # User email
#SBATCH --mail-type=FAIL,end  # Mail sent on begin, end/failure

#SBATCH --output="/home/rahmm224/AIinHealthProject/jobs_oe/job_name-%j.o"
#SBATCH --error="/home/rahmm224/AIinHealthProject/jobs_oe/job_name-%j.e"

# Request GPU resources
#SBATCH --gres=gpu:1  # Request 1 GPU (adjust the number as needed)

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

model_name="patchtst"
n_heads=8
e_layers=2
pred_len=128
patch_len=16
stride=8

experiments=10
learning_rate=1e-3
weight_decay=0.01
epochs=100
python /home/rahmm224/AIinHealthProject/code/train.py\
    --dataset_name $dataset_name\
    --no_features $no_features\
    --no_classes $no_classes\
    --window_size $window_size\
    --step_size $step_size\
    --batch_size $batch_size\
    --model_name $model_name\
    --n_heads $n_heads\
    --e_layers $e_layers\
    --pred_len $pred_len\
    --patch_len $patch_len\
    --stride $stride\
    --experiments $experiments\
    --learning_rate $learning_rate\
    --weight_decay $weight_decay\
    --epochs $epochs

conda deactivate
