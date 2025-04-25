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

CODEBOOK=cosine
CODEBOOK_ENTRIES=2048
CODEBOOK_EMBED_DIM=768
ENABLE_SCHEDULERS=True

UNFREEZE_LAYERS=0

OPTIMIZER=adam
MODEL=convnext_tiny
LR=0.0005
BATCH_SIZE=128
EPOCHS=60
WARMUP_EPOCHS=10
CHECKPOINT_PATH="checkpoints/convnext_tiny_cub.pth"
CODEBOOK_CHECKPOINT_PATH=""
SEED=42
DATASET=cub200
DATASET_PATH="data/cub"
WANDB_PROJECT=codebook_training


python train_codebook.py \
hydra.run.dir=${OUTPUT_DIR} \
model.name=${MODEL} \
model.checkpoint_path=${CHECKPOINT_PATH} \
epochs=${EPOCHS} \
training.enable_schedulers=${ENABLE_SCHEDULERS} \
training.warmup_epochs=${WARMUP_EPOCHS} \
train_dataloader.batch_size=${BATCH_SIZE} \
val_dataloader.batch_size=${BATCH_SIZE} \
codebook=${CODEBOOK} \
codebook.embedding_dim=${CODEBOOK_EMBED_DIM} \
codebook.num_entries=${CODEBOOK_ENTRIES} \
codebook_optimizer=${OPTIMIZER} \
codebook_optimizer.lr=${LR} \
base_optimizer=${OPTIMIZER} \
base_optimizer.lr=${LR} \
dataset=${DATASET} \
dataset._path=${DATASET_PATH} \
wandb.is_enabled=True \
wandb.project=${WANDB_PROJECT} \
seed=${SEED} \
training.unfreeze_before=${UNFREEZE_LAYERS} \
codebook_path=${CODEBOOK_CHECKPOINT_PATH}
