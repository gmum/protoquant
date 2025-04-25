#!/bin/bash -l
#SBATCH -J cosine_codebook
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -A plgomenn-gpu-a100
#SBATCH -p plgrid-gpu-a100


module add CUDA/12.0.0
module add Miniconda3/23.3.1-0
module add GCC/10.3.0
module load Ninja/1.10.2
export CUDA_HOME=/net/software/v1/software/CUDA/12.0.0

nvidia-smi -L
source activate XXX
cd <path to repo>/codebook_train

OUTPUT_DIR="hydra_out"
export WANDB_DIR=${OUTPUT_DIR}

CODEBOOK=dim_reduction
CODEBOOK_ENTRIES=4096
TARGET_ENTRIES=3000
STEPS=8
CODEBOOK_EMBED_DIM=768

MODEL=convnext_tiny
BATCH_SIZE=128
CHECKPOINT_PATH="checkpoints/convnext_tiny_cub.pth"
CODEBOOK_PATH="checkpoints/pruned_4096_convnext_tiny_codebook_2025-04-24_13-37-14.pth"
SEED=42
DATASET=cub200
DATASET_PATH="data/cub"
WANDB_PROJECT=codebook_training

CODEBOOK_INPUT_DIM=768
CODEBOOK_IN_DIMS='[]'
CODEBOOK_OUT_DIMS='[]'
CODEBOOK_MAPPING='[]'

python prune_codebook.py \
hydra.run.dir=${OUTPUT_DIR} \
codebook=${CODEBOOK} \
codebook.num_entries=${CODEBOOK_ENTRIES} \
codebook.embedding_dim=${CODEBOOK_EMBED_DIM} \
codebook_path=${CODEBOOK_PATH} \
dataset=${DATASET} \
dataset._path=${DATASET_PATH} \
wandb.is_enabled=False \
wandb.project=${WANDB_PROJECT} \
model.name=${MODEL} \
model.checkpoint_path=${CHECKPOINT_PATH} \
seed=${SEED} \
steps=${STEPS} \
target_num_codes=${TARGET_ENTRIES} \
train_dataloader.batch_size=${BATCH_SIZE} \
val_dataloader.batch_size=${BATCH_SIZE} \
codebook.input_dim=${CODEBOOK_INPUT_DIM} \
"codebook.in_block_config=${CODEBOOK_IN_DIMS}" \
"codebook.out_block_config=${CODEBOOK_OUT_DIMS}" \
"codebook.mapping_dim_config=${CODEBOOK_MAPPING}" \