import logging
from typing import Tuple
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
import sys
from pathlib import Path


def create_samplers(
    train_dataset: Dataset, val_dataset: Dataset, rank: int, world_size: int, seed: int
) -> Tuple[DistributedSampler, DistributedSampler]:
    """Create distributed samplers for training and validation datasets.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        seed (int): Random seed for shuffling.

    Returns:
        Tuple[DistributedSampler, DistributedSampler]: A tuple containing the training and validation samplers.
    """
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=seed,
    )

    return train_sampler, val_sampler


def setup_hydra_logging_for_rank(
    rank: int, log_level: int, hydra_output_dir: Path
) -> None:
    """Setup Hydra-style logging for distributed processes.

    Args:
        rank (int): Process rank
        log_level (str): Logging level
    """

    # Setup root logger
    root_logger = logging.getLogger()

    if rank != 0:
        log_level = logging.WARNING

    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter similar to Hydra's default
    formatter = logging.Formatter(
        f"[{rank}][%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Add file handler for each rank (similar to Hydra's file logging)
    if rank == 0:
        # Main log file for rank 0
        log_file = hydra_output_dir / "main.log"
    else:
        # Separate log files for other ranks
        log_file = hydra_output_dir / f"rank_{rank}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)

    logging.getLogger(__name__).info(
        f"Logging setup complete for rank {rank}, logging to {log_file}"
    )
