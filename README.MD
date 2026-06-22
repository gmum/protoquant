# ProtoQuant

Code for the paper ProtoQuant: Quantization of Prototypical Parts For General and Fine-Grained Image Classification.

- Paper: https://arxiv.org/abs/2602.06592
- PDF: https://arxiv.org/pdf/2602.06592

This repository contains the training, pruning, benchmarking, and analysis code used for ProtoQuant.

## Overview

The project is organized around three parts:

- ProtoQuant training and pruning pipelines in `codebook_train/src/`
- Benchmark runners for FunnyBirds, purity, and related evaluations in `codebook_train/run_*.py`
- Prototype analysis and visualization utilities in the top-level `codebook_train/` directory

## External repositories

This code depends on the following upstream projects:

- FunnyBirds framework: https://github.com/visinf/funnybirds-framework
- PIP-Net: https://github.com/M-Nauta/PIPNet

For reproducibility, trimmed benchmark-compatible copies are included in:

- `codebook_train/other_benchmarks/funnybirds_framework/`
- `codebook_train/other_benchmarks/purity_benchmark/`

If you want to reproduce the original benchmark protocols exactly, refer to the upstream repositories above.

## Requirements

Create a Python environment with PyTorch and install the project dependencies:

```bash
cd codebook_train
pip install -r ../requirements.txt
```

The code is designed for GPU execution, although several analysis utilities can run on CPU.

## Data preparation

The repository supports standard image-classification datasets as well as the paper benchmarks.

- CUB-200-2011 and Stanford Cars are supported by the bundled PIP-Net benchmark code.
- FunnyBirds uses the directory structure expected by the bundled FunnyBirds framework.
- Additional datasets can be added through the dataset helpers in `codebook_train/src/datasets/`.

Relevant helpers:

- `codebook_train/memfs.py` prepares FunnyBirds data for cluster-style storage layouts
- `codebook_train/src/datasets/construct_dataset.py` defines the main training datasets and loaders
- `codebook_train/other_benchmarks/purity_benchmark/util/data.py` defines the PIP-Net benchmark data paths

## Hydra configuration

The main training, pruning, and ProtoQuant entrypoints use Hydra for configuration.

- Top-level wrapper scripts such as `train_codebook.py`, `train_ema.py`, `prune_codebook.py`, and `train_protoquant.py` forward into Hydra-based modules under `codebook_train/src/`.
- Hydra overrides are passed on the command line using `key=value` syntax.
- Each run writes to a dedicated output directory controlled by `hydra.run.dir`.

Example override pattern:

```bash
python train_codebook.py \
	hydra.run.dir=outputs/example_run \
	dataset=cub200 \
	model.name=convnext_tiny \
	epochs=60
```

## Running experiments

Run the commands below from `codebook_train/`.

### Main codebook training

```bash
python train_codebook.py
```

This launches the distributed training flow implemented in `codebook_train/src/main.py`.

### Self-supervised / EMA training

```bash
python train_ema.py
```

This uses the self-supervised training pipeline in `codebook_train/src/main_train_ema.py`.

### Codebook pruning

```bash
python prune_codebook.py
```

This runs the pruning pipeline in `codebook_train/src/main_prune.py` and stores a pruned codebook checkpoint.

### ProtoQuant training

```bash
python train_protoquant.py
```

This trains the ProtoQuant head using `codebook_train/src/main_train_protoquant.py`.

### FunnyBirds benchmark

```bash
python run_funnybirds.py
```

For a debugging-oriented variant:

```bash
python run_funnybirds_debug.py
```

### Purity benchmark

```bash
python run_purity_benchmark.py
```

To execute the official PIP-Net benchmark path bundled with this repository:

```bash
python run_purity_official.py
```

### Analysis and visualization

Useful scripts include:

- `python analyze_local_size.py`
- `python analyze_proto_similarity.py`
- `python visualize_protoquant_prototypes.py`

These scripts help inspect prototype usage, codebook size, similarity structure, and local explanations.

## Slurm usage

Example Slurm submissions are provided in `codebook_train/slurm_scripts/`.

The training script `codebook_train/slurm_scripts/train_cosine_athena.sh` illustrates the typical pattern for a Hydra-based distributed job:

- request the desired GPU, CPU, and memory resources with `#SBATCH` directives
- load the required CUDA / Conda / compiler modules
- activate the Conda environment
- set a run-specific output directory, often under `$SCRATCH`
- pass experiment settings as Hydra overrides to `python train_codebook.py`

The pruning script `codebook_train/slurm_scripts/prune_codebook_athena.sh` shows the analogous pattern for pruning jobs.

Recommended workflow for Slurm:

1. Copy one of the provided scripts.
2. Update the environment activation line and repository path.
3. Adjust dataset, checkpoint, and output paths.
4. Submit with `sbatch <script-name>.sh`.

If you are running on a cluster, it is usually best to keep `hydra.run.dir` inside `$SCRATCH` or another writable job-local location.

## Notes

- `wandb` logging is optional and disabled by default.
- Some scripts require a pretrained backbone checkpoint or a codebook checkpoint, depending on the experiment.
- Run any script with `--help` if you want to inspect the full argument surface.

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{protoquant2026,
	title={ProtoQuant: Quantization of Prototypical Parts For General and Fine-Grained Image Classification},
	author={Janusz, Mikołaj and Wróbel, Adam and Zieliński, Bartosz and Rymarczyk, Dawid},
	year={2026},
	eprint={2602.06592},
	archivePrefix={arXiv},
	primaryClass={cs.CV}
}
```