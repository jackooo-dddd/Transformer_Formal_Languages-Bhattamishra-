#!/bin/bash
#SBATCH --time=0-00:30:00
#SBATCH --account=def-vumaiha               # <== use your actual account
#SBATCH --mem=32000M
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Hello World"
nvidia-smi

module load python/3.12
module load cuda/12.6  # Explicit is better than implicit

# Activate your venv
source ~/envs/Formal_Languages-Bhattamishra/bin/activate

python - <<'EOF'
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Number of CUDA devices:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f" Device {i}: {torch.cuda.get_device_name(i)}")
EOF
cd ~/scratch/Transformer_Formal_Languages-Bhattamishra-/
python -m src.main \
  -mode train \
  -run_name Shuffle2_SAN \
  -dataset Shuffle-2 \
  -model_type SAN \
  -depth 2 \
  -d_model 32 \
  -heads 4 \
  -pos_encode \
  -batch_size 32 \
  -epochs 50 \
  -gpu 0





