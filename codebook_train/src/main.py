from pathlib import Path
from omegaconf import OmegaConf
import torch
from src.main_train_supervised import prepare_codebook_training
from src.utils import (
    set_reproducibility,
)
from src.distributed_utils import setup_hydra_logging_for_rank
import logging
import hydra
import wandb
from src.config.main_config import MainConfig
import torch.distributed as dist


@hydra.main(config_path="config", config_name="main_config", version_base="1.2")
def start_training(cfg: MainConfig) -> None:
    """Main function for training the codebook

    Args:
        cfg (MainConfig): Hydra config object with all the settings. (Located in config/main_config.py)
    """

    main_logger = logging.getLogger(__name__)
    main_logger.setLevel(cfg._logging_level)

    main_logger.info(OmegaConf.to_yaml(cfg))
    hydra_output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    main_logger.info(f"Hydra output directory: {hydra_output_dir}")

    main_logger.info(
        f"Starting distributed training with {cfg.distributed.world_size} processes (GPU's) using {cfg.distributed.backend} backend."
    )

    torch.multiprocessing.spawn(
        fn=distributed_worker,
        args=(cfg, hydra_output_dir),
        nprocs=cfg.distributed.world_size,
        join=True,
    )


def distributed_worker(rank: int, cfg: MainConfig, hydra_output_dir: Path) -> None:
    """Worker function for distributed training.

    Args:
        rank (int):  The rank of the current process in distributed training.
        cfg (MainConfig): Main configuration object with all the settings.
        hydra_output_dir
    """

    setup_hydra_logging_for_rank(rank, cfg._logging_level, hydra_output_dir)

    if rank == 0 and cfg.wandb.is_enabled:
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

    # Initialize DDP
    dist.init_process_group(
        backend=cfg.distributed.backend,
        init_method=cfg.distributed.init_method,
        world_size=cfg.distributed.world_size,
        rank=rank,
    )
    device = torch.device(f"cuda:{rank}")

    # Set reproducibility with different seed per rank
    set_reproducibility(cfg.seed + rank)

    try:
        # Run training
        prepare_codebook_training(cfg, device, hydra_output_dir, wandb_run)
    finally:
        # Cleanup
        if wandb_run and rank == 0:
            wandb_run.finish()
        dist.destroy_process_group()
