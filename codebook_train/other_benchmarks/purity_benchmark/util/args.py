import os
import argparse
import pickle
import numpy as np
import random
import torch
import torch.optim

"""
    Utility functions for handling parsed arguments

"""


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train a PIP-Net")
    parser.add_argument(
        "--dataset",
        type=str,
        default="CUB-200-2011",
        help="Data set on PIP-Net should be trained",
    )
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.0,
        help="Split between training and validation set. Can be zero when there is a separate test or validation directory. Should be between 0 and 1. Used for partimagenet (e.g. 0.2)",
    )
    parser.add_argument(
        "--net",
        type=str,
        default="convnext_tiny_26",
        help="Base network used as backbone of PIP-Net. Default is convnext_tiny_26 with adapted strides to output 26x26 latent representations. Other option is convnext_tiny_13 that outputs 13x13 (smaller and faster to train, less fine-grained). convnext_tiny_7 uses the original torchvision ConvNeXt-Tiny stride (typically ~7x7 latent grid for 224x224). Pretrained network on iNaturalist is only available for resnet50_inat. Options are: resnet18, resnet34, resnet50, resnet50_inat, resnet101, resnet152, convnext_tiny_26, convnext_tiny_13, convnext_tiny_7.",
    )

    # Model selection
    parser.add_argument(
        "--model_type",
        type=str,
        default="pipnet",
        choices=["pipnet", "protoquant"],
        help="Which model implementation to use: the original PIP-Net (pipnet) or ProtoQuantNet (protoquant).",
    )

    # ProtoQuantNet evaluation-only options
    parser.add_argument(
        "--protoquant_checkpoint",
        type=str,
        default="",
        help='Path to a ProtoQuantNet checkpoint file (must contain a state_dict with a "codes" tensor). Required when --model_type protoquant.',
    )
    parser.add_argument(
        "--protoquant_backbone",
        type=str,
        default="resnet50",
        help="Backbone architecture name used by ProtoQuantNet (must exist in codebook_train/src/models_registry.py).",
    )
    parser.add_argument(
        "--protoquant_global_pool",
        type=str,
        default="",
        help='Optional global_pool setting for timm-style ViT backbones (e.g., "avg"). Leave empty for CNNs.',
    )
    parser.add_argument(
        "--protoquant_train_codebook",
        action="store_true",
        help="If set, loads ProtoQuantNet with trainable codebook (evaluation-only benchmark typically leaves this unset).",
    )
    parser.add_argument(
        "--protoquant_temperature",
        type=float,
        default=0.1,
        help="Temperature used inside ProtoQuantNet pooling (kept for completeness; adapter uses softmax over similarity maps).",
    )
    parser.add_argument(
        "--protoquant_classifier_sparsity_lambda",
        type=float,
        default=0.0,
        help="L1 regularization strength on ProtoQuant classifier weights (not used in eval-only).",
    )
    parser.add_argument(
        "--protoquant_limit_k",
        type=int,
        default=0,
        help=(
            "If > 0, masks ProtoQuantNet classifier weights to keep only the top-k "
            "prototypes per class (uses ProtoQuantNet.limit_prototypes). Useful to "
            "match PIPNet-style sparse per-class prototype usage."
        ),
    )
    parser.add_argument(
        "--protoquant_prune_after_limit",
        action="store_true",
        help=(
            "If set, physically prunes unused prototypes after applying --protoquant_limit_k, "
            "shrinking the codebook and classifier weight matrix for faster evaluation."
        ),
    )
    parser.add_argument(
        "--protoquant_prune_min_weight",
        type=float,
        default=0.0,
        help=(
            "Minimum positive classifier weight to consider a prototype active when pruning. "
            "Defaults to 0.0 (keep any prototype with weight > 0)."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size when training the model using minibatch gradient descent. Batch size is multiplied with number of available GPUs",
    )
    parser.add_argument(
        "--batch_size_pretrain",
        type=int,
        default=128,
        help="Batch size when pretraining the prototypes (first training stage)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="The number of epochs PIP-Net should be trained (second training stage)",
    )
    parser.add_argument(
        "--epochs_pretrain",
        type=int,
        default=10,
        help="Number of epochs to pre-train the prototypes (first training stage). Recommended to train at least until the align loss < 1",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="The optimizer that should be used when training PIP-Net",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="The optimizer learning rate for training the weights from prototypes to classes",
    )
    parser.add_argument(
        "--lr_block",
        type=float,
        default=0.0005,
        help="The optimizer learning rate for training the last conv layers of the backbone",
    )
    parser.add_argument(
        "--lr_net",
        type=float,
        default=0.0005,
        help="The optimizer learning rate for the backbone. Usually similar as lr_block.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay used in the optimizer",
    )
    parser.add_argument(
        "--disable_cuda",
        action="store_true",
        help="Flag that disables GPU usage if set",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./runs/run_pipnet",
        help="The directory in which train progress should be logged",
    )

    # Weights & Biases (optional)
    parser.add_argument(
        "--wandb_enabled",
        action="store_true",
        help="Enable Weights & Biases logging (scalars only; no checkpoint artifacts are uploaded).",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
        help="W&B mode. Use 'offline' on clusters without internet; default is 'disabled'.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="purity_benchmark",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="W&B entity/team (optional).",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default="",
        help="W&B group name (optional).",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help="W&B run name (optional).",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="Comma-separated list of W&B tags (optional).",
    )
    parser.add_argument(
        "--wandb_notes",
        type=str,
        default="",
        help="W&B notes (optional).",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=0,
        help="Number of prototypes. When zero (default) the number of prototypes is the number of output channels of backbone. If this value is set, then a 1x1 conv layer will be added. Recommended to keep 0, but can be increased when number of classes > num output channels in backbone.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input images will be resized to --image_size x --image_size (square). Code only tested with 224x224, so no guarantees that it works for different sizes.",
    )
    parser.add_argument(
        "--state_dict_dir_net",
        type=str,
        default="",
        help="The directory containing a state dict with a pretrained PIP-Net. E.g., ./runs/run_pipnet/checkpoints/net_pretrained",
    )
    parser.add_argument(
        "--cub_data_path", type=str, default="", help="Path to the CUB dataset root"
    )
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=10,
        help="Number of epochs where pretrained features_net will be frozen while training classification layer (and last layer(s) of backbone)",
    )

    parser.add_argument(
        "--backbone_train_policy",
        type=str,
        default="all",
        choices=["all", "none", "last_block", "last_layer"],
        help=(
            "Controls which backbone parameters are allowed to train during PIP-Net training. "
            "all (default) keeps the original benchmark behavior. "
            "none freezes the entire backbone for all epochs. "
            "last_block trains only the final block (ConvNeXt: features.7.2; ResNet: layer4.2). "
            "last_layer trains the entire last stage (ConvNeXt: features.7; ResNet: layer4)."
        ),
    )

    parser.add_argument(
        "--pipnet_head_only",
        action="store_true",
        help=(
            "PIP-Net only: train classifier head only during the second stage (no backbone fine-tuning and no add-on/prototype training). "
            "This is intended for fair comparisons against methods that keep the backbone frozen (e.g., ProtoQuant). "
            "If enabled, you should typically also set --epochs_pretrain 0 and optionally --backbone_train_policy none."
        ),
    )
    parser.add_argument(
        "--dir_for_saving_images",
        type=str,
        default="visualization_results",
        help="Directoy for saving the prototypes and explanations",
    )
    parser.add_argument(
        "--disable_pretrained",
        action="store_true",
        help="When set, the backbone network is initialized with random weights instead of being pretrained on another dataset).",
    )
    parser.add_argument(
        "--weighted_loss",
        action="store_true",
        help="Flag that weights the loss based on the class balance of the dataset. Recommended to use when data is imbalanced. ",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed. Note that there will still be differences between runs due to nondeterminism. See https://pytorch.org/docs/stable/notes/randomness.html",
    )
    parser.add_argument(
        "--gpu_ids", type=str, default="", help="ID of gpu. Can be separated with comma"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Num workers in dataloaders."
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        help="Flag that indicates whether to include a trainable bias in the linear classification layer.",
    )
    parser.add_argument(
        "--extra_test_image_folder",
        type=str,
        default="./experiments",
        help="Folder with images that PIP-Net will predict and explain, that are not in the training or test set. E.g. images with 2 objects or OOD image. Images should be in subfolder. E.g. images in ./experiments/images/, and argument --./experiments",
    )

    args = parser.parse_args()
    if len(args.log_dir.split("/")) > 2:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

    if args.model_type == "protoquant" and args.protoquant_checkpoint == "":
        raise ValueError(
            "When --model_type protoquant, you must provide --protoquant_checkpoint"
        )

    if args.protoquant_limit_k < 0:
        raise ValueError("--protoquant_limit_k must be >= 0")

    if args.protoquant_prune_min_weight < 0:
        raise ValueError("--protoquant_prune_min_weight must be >= 0")

    return args


def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + "/args.txt", "w") as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(
                val, str
            ):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write("{}: {}\n".format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + "/args.pickle", "wb") as f:
        pickle.dump(args, f)


def get_optimizer_nn(
    net, args: argparse.Namespace
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
]:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # create parameter groups
    params_to_freeze = []
    params_to_train = []
    params_backbone = []

    backbone_policy = getattr(args, "backbone_train_policy", "all")
    # set up optimizer
    if "resnet50" in args.net:
        # freeze resnet50 except last convolutional layer
        for name, param in net.module._net.named_parameters():
            if backbone_policy == "none":
                param.requires_grad = False
                params_backbone.append(param)  # All frozen params in backbone group
                continue
            if backbone_policy == "last_block":
                if "layer4.2" in name:
                    params_to_train.append(param)
                elif "layer4" in name or "layer3" in name:
                    params_to_freeze.append(param)  # Frozen but nearby layers
                else:
                    param.requires_grad = False
                    params_backbone.append(param)  # Earlier frozen layers
                continue
            if backbone_policy == "last_layer":
                if "layer4" in name:
                    params_to_train.append(param)
                elif "layer3" in name:
                    params_to_freeze.append(param)  # Frozen but nearby layer
                else:
                    param.requires_grad = False
                    params_backbone.append(param)  # Earlier frozen layers
                continue
            if "layer4.2" in name:
                params_to_train.append(param)
            elif "layer4" in name or "layer3" in name:
                params_to_freeze.append(param)
            elif "layer2" in name:
                params_backbone.append(param)
            else:  # such that model training fits on one gpu.
                param.requires_grad = False
                # params_backbone.append(param)

    elif "convnext" in args.net:
        print("chosen network is convnext", flush=True)
        for name, param in net.module._net.named_parameters():
            if backbone_policy == "none":
                param.requires_grad = False
                params_backbone.append(param)  # All frozen params in backbone group
                continue
            if backbone_policy == "last_block":
                # ConvNeXt last block only
                if "features.7.2" in name:
                    params_to_train.append(param)
                elif "features.7" in name or "features.6" in name:
                    params_to_freeze.append(param)  # Frozen but nearby stages
                else:
                    param.requires_grad = False
                    params_backbone.append(param)  # Earlier frozen stages
                continue
            if backbone_policy == "last_layer":
                # ConvNeXt entire last stage (features.7)
                if "features.7" in name:
                    params_to_train.append(param)
                elif "features.6" in name:
                    params_to_freeze.append(param)  # Frozen but nearby stage
                else:
                    param.requires_grad = False
                    params_backbone.append(param)  # Earlier frozen stages
                continue
            if "features.7.2" in name:
                params_to_train.append(param)
            elif "features.7" in name or "features.6" in name:
                params_to_freeze.append(param)
            # CUDA MEMORY ISSUES? COMMENT LINE 202-203 AND USE THE FOLLOWING LINES INSTEAD
            # elif 'features.5' in name or 'features.4' in name:
            #     params_backbone.append(param)
            # else:
            #     param.requires_grad = False
            else:
                params_backbone.append(param)
    else:
        print("Network is not ResNet or ConvNext.", flush=True)
    classification_weight = []
    classification_bias = []
    for name, param in net.module._classification.named_parameters():
        if "weight" in name:
            classification_weight.append(param)
        elif "multiplier" in name:
            param.requires_grad = False
        else:
            if args.bias:
                classification_bias.append(param)

    paramlist_net = [
        {
            "params": params_backbone,
            "lr": args.lr_net,
            "weight_decay_rate": args.weight_decay,
        },
        {
            "params": params_to_freeze,
            "lr": args.lr_block,
            "weight_decay_rate": args.weight_decay,
        },
        {
            "params": params_to_train,
            "lr": args.lr_block,
            "weight_decay_rate": args.weight_decay,
        },
        {
            "params": net.module._add_on.parameters(),
            "lr": args.lr_block * 10.0,
            "weight_decay_rate": args.weight_decay,
        },
    ]

    paramlist_classifier = [
        {
            "params": classification_weight,
            "lr": args.lr,
            "weight_decay_rate": args.weight_decay,
        },
        {"params": classification_bias, "lr": args.lr, "weight_decay_rate": 0},
    ]

    if args.optimizer == "Adam":
        optimizer_net = torch.optim.AdamW(
            paramlist_net, lr=args.lr, weight_decay=args.weight_decay
        )
        optimizer_classifier = torch.optim.AdamW(
            paramlist_classifier, lr=args.lr, weight_decay=args.weight_decay
        )
        return (
            optimizer_net,
            optimizer_classifier,
            params_to_freeze,
            params_to_train,
            params_backbone,
        )
    else:
        raise ValueError("this optimizer type is not implemented")
