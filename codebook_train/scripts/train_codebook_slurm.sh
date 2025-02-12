#!/bin/bash
#SBATCH --job-name=codebook 
#SBATCH --qos=normal
#SBATCH --mem=32G
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Go to the folder you want to run jupyter in
cd $HOME/source/codebook_playground/codebook_train
source activate omenn_localize

nvidia-smi -L

CODEBOOK=cosine
CODEBOOK_ENTRIES=256
CODEBOOK_EMBED_DIM=768

UNFREEZE_LAYERS=0

OPTIMIZER=adam
MODEL=convnext_tiny
LR=0.5
BATCH_SIZE=128
EPOCHS=60
CHECKPOINT_PATH="checkpoints/convnext_tiny_cub_2025-02-08_19-06-35.pth"
CODEBOOK_CHECKPOINT_PATH="checkpoints/convnext_tiny_codebook.pth"
SEED=42
DATASET=cub200
DATASET_PATH="data/"
WANDB_PROJECT=codebook_training


python train_codebook.py \
model.name=${MODEL} \
model.checkpoint_path=${CHECKPOINT_PATH} \
epochs=${EPOCHS} \
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
