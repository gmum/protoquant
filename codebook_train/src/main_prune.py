from pathlib import Path
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from src.codebook import create_codebook_wrapper, ConvNextCosineWrapper
from src.construct_model import construct_model
from src.datasets.construct_dataset import get_dataloaders
from src.utils import (
    validate_epoch,
    set_reproducibility,
    save_checkpoint,
    validate_epoch_cosine_codebook,
)
from datetime import datetime
import logging
import hydra
from src.config.pruning_config import PruningConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    import wandb
except ImportError:
    wandb = None
    logger.info("wandb is not available, skipping wandb.init")


@hydra.main(config_path="config", config_name="pruning_config", version_base="1.2")
def start_pruning(cfg: PruningConfig) -> None:
    """Main function for training the codebook

    Args:
        cfg (PruningConfig): Configuration object containing all the parameters for the pruning process.
    """
    logger.setLevel(cfg._logging_level)

    if cfg.wandb.is_enabled:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg),
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            job_type=cfg.wandb.job_type,
            tags=cfg.wandb.tags,
        )
    else:
        wandb_run = None
        logger.info("wandb is not enabled")

    logger.info(OmegaConf.to_yaml(cfg))
    hydra_output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    logger.info(f"Hydra output directory: {hydra_output_dir}")
    set_reproducibility(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepare_codebook_pruning(
        cfg=cfg,
        device=device,
        wandb_run=wandb_run,
    )


def prepare_codebook_pruning(
    cfg: PruningConfig, device: torch.device, wandb_run=None
) -> None:
    """Prepare the codebook pruning process by loading the model, dataloaders, and optimizers.

    Args:
        cfg (PruningConfig): Configuration object containing all the parameters for the pruning process.
        device (torch.device): The device to run the training on
        wandb_run (_type_, optional): Wandb object for logging. Defaults to None.
    """

    model = construct_model(cfg).to(device)

    _, val_dataloader = get_dataloaders(cfg)
    logger.info("Validate the base model")
    base_top1_acc, base_top5_acc = validate_epoch(
        model=model, val_dataloader=val_dataloader, device=device
    )

    logger.info(
        f"Base Validation top1-accuracy {base_top1_acc}, top5-accuracy {base_top5_acc}"
    )

    # create and insert the codebook into the model, set the requires_grad
    codebook = hydra.utils.instantiate(cfg.codebook).to(device)
    codebook.load_state_dict(torch.load(cfg.codebook_path))

    model_with_codebook = create_codebook_wrapper(
        model=model,
        codebook=codebook,
        model_name=cfg.model.name,
        unfreeze_before=cfg.training.unfreeze_before,
    )
    logger.info(f"Model with codebook: {model_with_codebook}")

    codebook_pruning(
        model=model_with_codebook,
        val_dataloader=val_dataloader,
        num_codes=codebook.num_entries,
        target_num_codes=cfg.target_num_codes,
        device=device,
        steps=cfg.steps,
    )

    if cfg.output_checkpoint_path is not None:
        hydra_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_codebook_path = (
            hydra_path
            / f"pruned_{cfg.target_num_codes}_{cfg.model.name}_codebook_{current_date}.pth"
        )
        save_checkpoint(
            model=codebook,
            path=out_codebook_path,
        )
        logger.info(f"Saved pruned codebook to {out_codebook_path}")


def codebook_pruning(
    model: ConvNextCosineWrapper,
    val_dataloader: torch.utils.data.DataLoader,
    num_codes: int,
    target_num_codes: int,
    device: torch.device,
    steps: int,
    wandb_run=None,
):
    codes_per_step = (num_codes - target_num_codes) // steps
    for step in range(steps):
        logger.info(f"Step {step + 1}/{steps}")

        val_statistics = validate_epoch_cosine_codebook(
            model=model, val_dataloader=val_dataloader, device=device
        )
        val_statistics = {f"val_{k}": v for k, v in val_statistics.items()}
        logger.info(f"Validation statistics: {val_statistics}")
        if wandb_run:
            wandb.log(
                {
                    "Step": step,
                    **val_statistics,
                }
            )

        # get the STEP least used codes to remove
        code_usage: torch.Tensor = val_statistics["val_code_usage"]
        codes_to_remove = code_usage.argsort()[:codes_per_step]

        # Handle different codebook types
        if hasattr(model.codebook, "codebook"):  # For DimReductionWrapper
            prune_codebook_embeddings(model.codebook.codebook, codes_to_remove)
        else:  # Direct codebook
            prune_codebook_embeddings(model.codebook, codes_to_remove)

        model.codebook.reset_statistics()

    # final validation
    logger.info("Final validation after pruning")
    val_statistics = validate_epoch_cosine_codebook(
        model=model, val_dataloader=val_dataloader, device=device
    )
    val_statistics = {f"val_{k}": v for k, v in val_statistics.items()}
    logger.info(f"Validation statistics: {val_statistics}")
    if wandb_run:
        wandb.log(
            {
                "Step": steps,
                **val_statistics,
            }
        )


def prune_codebook_embeddings(codebook, codes_to_remove):
    """Prune the codebook by removing specified codes and creating a smaller embedding matrix.

    Args:
        codebook: The codebook module containing the embeddings
        codes_to_remove: Indices of codes to remove
    """
    current_num_entries = codebook.num_entries
    new_num_entries = current_num_entries - len(codes_to_remove)
    embedding_dim = codebook.embeddings.weight.shape[1]

    # Get all code indices
    all_indices = torch.arange(current_num_entries, device=codes_to_remove.device)

    # Create mask for codes to keep
    keep_mask = torch.ones(
        current_num_entries, dtype=torch.bool, device=codes_to_remove.device
    )
    keep_mask[codes_to_remove] = False

    # Get indices of codes to keep
    codes_to_keep = all_indices[keep_mask]

    # Create new embedding weights with only the kept codes
    old_embeddings = codebook.embeddings.weight.data
    new_embeddings = old_embeddings[codes_to_keep]

    # Create new embedding layer with smaller size
    new_embedding_layer = nn.Embedding(new_num_entries, embedding_dim)
    new_embedding_layer.to(old_embeddings.device)
    new_embedding_layer.weight.data.copy_(new_embeddings)

    # Replace old embeddings with new ones
    codebook.embeddings = new_embedding_layer
    codebook.num_entries = new_num_entries

    # Update tracking buffers to the new size
    if hasattr(codebook, "code_usage"):
        new_code_usage = codebook.code_usage[codes_to_keep]
        codebook.register_buffer(
            "code_usage",
            torch.zeros(
                new_num_entries, dtype=torch.long, device=old_embeddings.device
            ),
        )
        codebook.code_usage.copy_(new_code_usage)

    if hasattr(codebook, "restarted_count"):
        new_restarted_count = codebook.restarted_count[codes_to_keep]
        codebook.register_buffer(
            "restarted_count",
            torch.zeros(
                new_num_entries, dtype=torch.long, device=old_embeddings.device
            ),
        )
        codebook.restarted_count.copy_(new_restarted_count)

    logger.info(
        f"Pruned codebook from {current_num_entries} to {new_num_entries} entries"
    )
