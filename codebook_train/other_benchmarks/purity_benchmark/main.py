from .pipnet.pipnet import PIPNet, get_network
from .util.log import Log
import torch.nn as nn
from .util.args import get_args, save_args, get_optimizer_nn
from .util.data import get_dataloaders
from .util.func import init_weights_xavier
from .pipnet.train import train_pipnet
from .pipnet.test import eval_pipnet, get_thresholds, eval_ood
from .util.eval_cub_csv import (
    eval_prototypes_cub_parts_csv,
    get_topk_cub,
    get_proto_patches_cub,
)
import torch
from .util.vis_pipnet import visualize, visualize_topk
from .util.visualize_prediction import vis_pred, vis_pred_experiments
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import logging
from copy import deepcopy
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)


def _maybe_init_wandb(args) -> Any:
    """Initialize a W&B run if enabled; otherwise return None.

    This benchmark logs scalars only and does not upload checkpoints as artifacts.
    """

    enabled = bool(getattr(args, "wandb_enabled", False))
    mode = str(getattr(args, "wandb_mode", "disabled"))
    if not enabled or mode == "disabled":
        return None

    try:
        import wandb  # type: ignore
    except Exception as e:
        logger.warning(f"W&B requested but wandb import failed: {e}")
        return None

    tags_raw = str(getattr(args, "wandb_tags", "") or "")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

    entity = str(getattr(args, "wandb_entity", "") or "")
    group = str(getattr(args, "wandb_group", "") or "")
    name = str(getattr(args, "wandb_run_name", "") or "")
    notes = str(getattr(args, "wandb_notes", "") or "")

    init_kwargs: Dict[str, Any] = {
        "project": str(getattr(args, "wandb_project", "purity_benchmark")),
        "config": vars(args),
        "tags": tags or None,
        "mode": mode,
        "dir": str(getattr(args, "log_dir", ".")),
        "notes": notes or None,
    }
    if entity:
        init_kwargs["entity"] = entity
    if group:
        init_kwargs["group"] = group
    if name:
        init_kwargs["name"] = name

    run = wandb.init(**init_kwargs)
    logger.info(
        f"W&B enabled: project={init_kwargs['project']} mode={mode} run={getattr(run, 'name', '')}"
    )
    return run


def _wandb_log(
    run: Any, metrics: Dict[str, Any], *, step: Optional[int] = None
) -> None:
    if run is None:
        return
    try:
        import wandb  # type: ignore

        if step is None:
            wandb.log(metrics)
        else:
            wandb.log(metrics, step=step)
    except Exception as e:
        logger.debug(f"W&B log failed (ignored): {e}")


def run_pipnet(args=None, wandb_run=None):
    args = args or get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model_type = getattr(args, "model_type", "pipnet")
    backbone_policy = getattr(args, "backbone_train_policy", "all")
    pipnet_head_only = bool(getattr(args, "pipnet_head_only", False))
    assert args.batch_size > 1

    # Create a logger
    log = Log(args.log_dir)
    logger.info(f"Log dir: {args.log_dir}")
    # Log the run arguments
    save_args(args, log.metadata_dir)

    # Initialize W&B after log dir exists and args are saved locally
    if wandb_run is None:
        wandb_run = _maybe_init_wandb(args)
    _wandb_log(
        wandb_run,
        {
            "run/log_dir": args.log_dir,
            "run/dataset": args.dataset,
            "run/model_type": getattr(args, "model_type", "pipnet"),
            "run/net": getattr(args, "net", ""),
        },
        step=0,
    )

    gpu_list = args.gpu_ids.split(",")
    device_ids = []
    if args.gpu_ids != "":
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))

    global device
    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids) == 1:
            device = torch.device("cuda:{}".format(args.gpu_ids))
        elif len(device_ids) == 0:
            device = torch.device("cuda")
            logger.info("CUDA device set without id specification")
            device_ids.append(torch.cuda.current_device())
        else:
            logger.info(
                "This code should work with multiple GPU's but we didn't test that, so we recommend to use only 1 GPU."
            )
            device_str = ""
            for d in device_ids:
                device_str += str(d)
                device_str += ","
            device = torch.device("cuda:" + str(device_ids[0]))
    else:
        device = torch.device("cpu")

    # Log which device was actually used
    logger.info(f"Device used: {device} with id {device_ids}")
    _wandb_log(
        wandb_run,
        {
            "run/device": str(device),
            "run/device_ids": str(device_ids),
        },
        step=0,
    )

    # Obtain the dataset and dataloaders
    (
        trainloader,
        trainloader_pretraining,
        trainloader_normal,
        trainloader_normal_augment,
        projectloader,
        testloader,
        test_projectloader,
        classes,
    ) = get_dataloaders(args, device)
    if len(classes) <= 20:
        if args.validation_size == 0.0:
            logger.info(f"Classes: {testloader.dataset.class_to_idx}")
        else:
            logger.info(f"Classes: {classes}")

    if model_type == "pipnet" and getattr(args, "protoquant_checkpoint", ""):
        logger.warning(
            "Ignoring --protoquant_checkpoint because --model_type is pipnet; use --state_dict_dir_net to load a PIP-Net checkpoint."
        )

    if model_type == "pipnet" and pipnet_head_only:
        if int(getattr(args, "epochs_pretrain", 0)) != 0:
            logger.warning(
                "--pipnet_head_only enabled: forcing --epochs_pretrain=0 (prototype pretraining is disabled in classifier-only mode)."
            )
            args.epochs_pretrain = 0

    if model_type == "protoquant":
        if int(args.epochs_pretrain) != 0 or int(args.epochs) != 0:
            raise ValueError(
                "ProtoQuant mode is evaluation-only here. Run with --epochs_pretrain 0 --epochs 0."
            )
        if args.state_dict_dir_net != "":
            raise ValueError(
                "When --model_type protoquant, use --protoquant_checkpoint (not --state_dict_dir_net)."
            )

        from src.construct_model import construct_model_no_cfg, get_backbone
        from src.pipnet_utils import build_pipnet_model
        from .pipnet.protoquant_adapter import ProtoQuantAdapter

        base_model = construct_model_no_cfg(
            model_name=args.protoquant_backbone,
            num_classes=len(classes),
            device=device,
            checkpoint_path=None,
            global_pool=args.protoquant_global_pool,
        )
        backbone = get_backbone(base_model)
        protoquant = build_pipnet_model(
            backbone=backbone,
            num_classes=len(classes),
            device=device,
            pipnet_checkpoint_path=args.protoquant_checkpoint,
            codebook_path=None,
            train_codebook=args.protoquant_train_codebook,
            temperature=args.protoquant_temperature,
            classifier_sparsity_lambda=args.protoquant_classifier_sparsity_lambda,
            use_random_codes=False,
        )

        # Optional: limit classifier to top-k prototypes per class for interpretability.
        # This masks classifier weights; prototypes remain in the similarity map but may be unused.
        if getattr(args, "protoquant_limit_k", 0) and int(args.protoquant_limit_k) > 0:
            kept = protoquant.limit_prototypes(int(args.protoquant_limit_k))
            logger.info(
                "ProtoQuantNet limited to top-%d prototypes/class; %d unique prototypes remain active",
                int(args.protoquant_limit_k),
                int(kept),
            )

            # Optional: physically prune unused prototypes for faster similarity computation.
            if bool(getattr(args, "protoquant_prune_after_limit", False)):
                min_w = float(getattr(args, "protoquant_prune_min_weight", 0.0))
                pruned_p = protoquant.prune_inactive_prototypes(min_weight=min_w)
                logger.info(
                    "ProtoQuantNet codebook pruned to %d prototypes", int(pruned_p)
                )

        net = ProtoQuantAdapter(protoquant, num_classes=len(classes))
        num_prototypes = int(net._num_prototypes)
    else:
        # Create a convolutional network based on arguments and add 1x1 conv layer
        feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = (
            get_network(len(classes), args)
        )

        # Create a PIP-Net
        net = PIPNet(
            num_classes=len(classes),
            num_prototypes=num_prototypes,
            feature_net=feature_net,
            args=args,
            add_on_layers=add_on_layers,
            pool_layer=pool_layer,
            classification_layer=classification_layer,
        )

    net = net.to(device=device)
    net = nn.DataParallel(net, device_ids=device_ids)

    def _enforce_backbone_train_policy(*, finetune: bool) -> None:
        """Ensure backbone parameter training matches args.backbone_train_policy.

        - finetune=True keeps backbone fully frozen (classifier-only warmup).
        - policy=all keeps the original benchmark behavior.
        - policy=none freezes the entire backbone for all epochs.
        - policy=last_block trains only the final block (ConvNeXt: features.7.2; ResNet: layer4.2).
        - policy=last_layer trains the entire last stage (ConvNeXt: features.7; ResNet: layer4).
        """

        # Always keep backbone frozen during finetune (classifier-only) epochs.
        if finetune:
            for p in net.module._net.parameters():
                p.requires_grad = False
            return

        if backbone_policy == "all":
            return

        if backbone_policy == "none":
            for p in net.module._net.parameters():
                p.requires_grad = False
            return

        if backbone_policy == "last_block":
            for name, p in net.module._net.named_parameters():
                p.requires_grad = False
                if "features.7.2" in name or "layer4.2" in name:
                    p.requires_grad = True
            return

        if backbone_policy == "last_layer":
            for name, p in net.module._net.named_parameters():
                p.requires_grad = False
                if "features.7" in name or "layer4" in name:
                    p.requires_grad = True
            return

    # Eval-only shortcut for ProtoQuantNet: skip all optimizer/scheduler/training setup.
    if model_type == "protoquant":
        epoch = 0
        with torch.no_grad():
            xs1, _, _ = next(iter(trainloader))
            xs1 = xs1.to(device)
            proto_features, _, _ = net(xs1)
            args.wshape = proto_features.shape[-1]
            logger.info(f"Output shape: {proto_features.shape}")
            log.create_log(
                "log_epoch_overview",
                "epoch",
                "test_top1_acc",
                "test_top5_acc",
                "almost_sim_nonzeros",
                "local_size_all_classes",
                "almost_nonzeros_pooled",
                "num_nonzero_prototypes",
                "mean_train_acc",
                "mean_train_loss_during_epoch",
            )

        _run_postrun_evals(
            net=net,
            projectloader=projectloader,
            test_projectloader=test_projectloader,
            testloader=testloader,
            classes=classes,
            device=device,
            args=args,
            log=log,
            epoch=epoch,
            wandb_run=wandb_run,
        )
        logger.info("Done!")
        return

    # Eval-only shortcut for PIP-Net: load checkpoint and run postrun evals.
    # This matches the original upstream usage: --epochs_pretrain 0 --epochs 0 --state_dict_dir_net <ckpt>
    if int(args.epochs_pretrain) == 0 and int(args.epochs) == 0:
        if args.state_dict_dir_net == "":
            raise ValueError(
                "PIP-Net eval-only requested (--epochs_pretrain 0 --epochs 0), but --state_dict_dir_net is empty. "
                "Provide a trained PIP-Net checkpoint to evaluate purity."
            )

        epoch = 0
        checkpoint = torch.load(args.state_dict_dir_net, map_location=device)
        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain 'model_state_dict'. Provide a PIP-Net checkpoint saved by this benchmark/training code."
            )
        net.load_state_dict(checkpoint["model_state_dict"], strict=True)
        logger.info("Pretrained network loaded")

        with torch.no_grad():
            xs1, _, _ = next(iter(trainloader))
            xs1 = xs1.to(device)
            proto_features, _, _ = net(xs1)
            args.wshape = proto_features.shape[-1]
            logger.info(f"Output shape: {proto_features.shape}")
            log.create_log(
                "log_epoch_overview",
                "epoch",
                "test_top1_acc",
                "test_top5_acc",
                "almost_sim_nonzeros",
                "local_size_all_classes",
                "almost_nonzeros_pooled",
                "num_nonzero_prototypes",
                "mean_train_acc",
                "mean_train_loss_during_epoch",
            )

        _run_postrun_evals(
            net=net,
            projectloader=projectloader,
            test_projectloader=test_projectloader,
            testloader=testloader,
            classes=classes,
            device=device,
            args=args,
            log=log,
            epoch=epoch,
            wandb_run=wandb_run,
        )
        logger.info("Done!")
        return

    (
        optimizer_net,
        optimizer_classifier,
        params_to_freeze,
        params_to_train,
        params_backbone,
    ) = get_optimizer_nn(net, args)

    # Initialize or load model
    with torch.no_grad():
        if args.state_dict_dir_net != "":
            epoch = 0
            checkpoint = torch.load(args.state_dict_dir_net, map_location=device)
            net.load_state_dict(checkpoint["model_state_dict"], strict=True)
            logger.info("Pretrained network loaded")
            net.module._multiplier.requires_grad = False
            try:
                optimizer_net.load_state_dict(checkpoint["optimizer_net_state_dict"])
            except Exception:
                pass
            if (
                torch.mean(net.module._classification.weight).item() > 1.0
                and torch.mean(net.module._classification.weight).item() < 3.0
                and torch.count_nonzero(
                    torch.relu(net.module._classification.weight - 1e-5)
                )
                .float()
                .item()
                > 0.8 * (num_prototypes * len(classes))
            ):  # assume that the linear classification layer is not yet trained (e.g. when loading a pretrained backbone only)
                logger.info(
                    "We assume that the classification layer is not yet trained. Re-initializing it..."
                )
                torch.nn.init.normal_(
                    net.module._classification.weight, mean=1.0, std=0.1
                )
                torch.nn.init.constant_(net.module._multiplier, val=2.0)
                logger.info(
                    f"Classification layer initialized with mean {torch.mean(net.module._classification.weight).item()}"
                )
                if args.bias:
                    torch.nn.init.constant_(net.module._classification.bias, val=0.0)
            # else: #uncomment these lines if you want to load the optimizer too
            #     if 'optimizer_classifier_state_dict' in checkpoint.keys():
            #         optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])

        else:
            net.module._add_on.apply(init_weights_xavier)
            torch.nn.init.normal_(net.module._classification.weight, mean=1.0, std=0.1)
            if args.bias:
                torch.nn.init.constant_(net.module._classification.bias, val=0.0)
            torch.nn.init.constant_(net.module._multiplier, val=2.0)
            net.module._multiplier.requires_grad = False

            logger.info(
                f"Classification layer initialized with mean {torch.mean(net.module._classification.weight).item()}",
            )

    # Define classification loss function and scheduler
    criterion = nn.NLLLoss(reduction="mean").to(device)
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_net,
        T_max=len(trainloader_pretraining) * args.epochs_pretrain,
        eta_min=args.lr_block / 100.0,
        last_epoch=-1,
    )

    # Forward one batch through the backbone to get the latent output size
    with torch.no_grad():
        xs1, _, _ = next(iter(trainloader))
        xs1 = xs1.to(device)
        proto_features, _, _ = net(xs1)
        wshape = proto_features.shape[-1]
        args.wshape = wshape  # needed for calculating image patch size
        logger.info(f"Output shape: {proto_features.shape}")
        # Create a csv log for storing the test accuracy (top 1 and top 5), mean train accuracy and mean loss for each epoch
        log.create_log(
            "log_epoch_overview",
            "epoch",
            "test_top1_acc",
            "test_top5_acc",
            "almost_sim_nonzeros",
            "local_size_all_classes",
            "almost_nonzeros_pooled",
            "num_nonzero_prototypes",
            "mean_train_acc",
            "mean_train_loss_during_epoch",
        )

    lrs_pretrain_net = []
    # PRETRAINING PROTOTYPES PHASE
    for epoch in range(1, args.epochs_pretrain + 1):
        for param in params_to_train:
            param.requires_grad = True
        for param in net.module._add_on.parameters():
            param.requires_grad = True
        for param in net.module._classification.parameters():
            param.requires_grad = False
        for param in params_to_freeze:
            param.requires_grad = (
                True  # can be set to False when you want to freeze more layers
            )
        for param in params_backbone:
            param.requires_grad = False  # can be set to True when you want to train whole backbone (e.g. if dataset is very different from ImageNet)

        _enforce_backbone_train_policy(finetune=False)

        logger.info(
            f"Pretrain Epoch {epoch} with batch size {trainloader_pretraining.batch_size}"
        )

        # Pretrain prototypes
        train_info = train_pipnet(
            net,
            trainloader_pretraining,
            optimizer_net,
            optimizer_classifier,
            scheduler_net,
            None,
            criterion,
            epoch,
            args.epochs_pretrain,
            device,
            pretrain=True,
            finetune=False,
        )
        lrs_pretrain_net += train_info["lrs_net"]
        plt.clf()
        plt.plot(lrs_pretrain_net)
        plt.savefig(os.path.join(args.log_dir, "lr_pretrain_net.png"))
        log.log_values(
            "log_epoch_overview",
            epoch,
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            train_info["loss"],
        )
        pretrain_step = int(epoch)
        _wandb_log(
            wandb_run,
            {
                "pretrain/epoch": int(epoch),
                "pretrain/train_loss": float(train_info.get("loss", 0.0)),
                "pretrain/train_acc": float(train_info.get("train_accuracy", 0.0)),
                "pretrain/lr_net": float(optimizer_net.param_groups[0]["lr"]),
            },
            step=pretrain_step,
        )

    if args.state_dict_dir_net == "":
        net.eval()
        torch.save(
            {
                "model_state_dict": net.state_dict(),
                "optimizer_net_state_dict": optimizer_net.state_dict(),
            },
            os.path.join(os.path.join(args.log_dir, "checkpoints"), "net_pretrained"),
        )
        net.train()
    with torch.no_grad():
        if "convnext" in args.net and args.epochs_pretrain > 0:
            visualize_topk(
                net,
                projectloader,
                len(classes),
                device,
                "visualised_pretrained_prototypes_topk",
                args,
            )

    # SECOND TRAINING PHASE
    # re-initialize optimizers and schedulers for second training phase
    (
        optimizer_net,
        optimizer_classifier,
        params_to_freeze,
        params_to_train,
        params_backbone,
    ) = get_optimizer_nn(net, args)
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_net, T_max=len(trainloader) * args.epochs, eta_min=args.lr_net / 100.0
    )
    # scheduler for the classification layer is with restarts, such that the model can re-active zeroed-out prototypes. Hence an intuitive choice.
    if args.epochs <= 30:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_classifier, T_0=5, eta_min=0.001, T_mult=1
        )
    else:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_classifier, T_0=10, eta_min=0.001, T_mult=1
        )
    for param in net.module.parameters():
        param.requires_grad = False
    for param in net.module._classification.parameters():
        param.requires_grad = True

    frozen = True
    lrs_net = []
    lrs_classifier = []

    for epoch in range(1, args.epochs + 1):
        if model_type == "pipnet" and pipnet_head_only:
            # Strict classifier-only mode: never train backbone or add-on/prototypes.
            # This matches frozen-backbone baselines where prototypes are fixed and only the classifier weights are trained.
            finetune = True
            for param in net.module._add_on.parameters():
                param.requires_grad = False
            for param in params_to_train:
                param.requires_grad = False
            for param in params_to_freeze:
                param.requires_grad = False
            for param in params_backbone:
                param.requires_grad = False
            for param in net.module._net.parameters():
                param.requires_grad = False
            for param in net.module._classification.parameters():
                param.requires_grad = True
            frozen = True
        else:
            epochs_to_finetune = 3  # finetune = classifier-only warmup epochs
            if epoch <= epochs_to_finetune and (
                args.epochs_pretrain > 0 or args.state_dict_dir_net != ""
            ):
                for param in net.module._add_on.parameters():
                    param.requires_grad = False
                for param in params_to_train:
                    param.requires_grad = False
                for param in params_to_freeze:
                    param.requires_grad = False
                for param in params_backbone:
                    param.requires_grad = False
                finetune = True
            else:
                finetune = False
                if frozen:
                    # unfreeze backbone
                    if epoch > (args.freeze_epochs):
                        for param in net.module._add_on.parameters():
                            param.requires_grad = True
                        for param in params_to_freeze:
                            param.requires_grad = True
                        for param in params_to_train:
                            param.requires_grad = True
                        for param in params_backbone:
                            param.requires_grad = True
                        frozen = False
                    # freeze first layers of backbone, train rest
                    else:
                        for param in params_to_freeze:
                            param.requires_grad = True  # Can be set to False if you want to train fewer layers of backbone
                        for param in net.module._add_on.parameters():
                            param.requires_grad = True
                        for param in params_to_train:
                            param.requires_grad = True
                        for param in params_backbone:
                            param.requires_grad = False

        _enforce_backbone_train_policy(finetune=finetune)

        logger.info(f"Epoch {epoch} frozen={frozen}")
        if (epoch == args.epochs or epoch % 30 == 0) and args.epochs > 1:
            # SET SMALL WEIGHTS TO ZERO
            with torch.no_grad():
                torch.set_printoptions(profile="full")
                net.module._classification.weight.copy_(
                    torch.clamp(net.module._classification.weight.data - 0.001, min=0.0)
                )
                nonzero_w = net.module._classification.weight[
                    net.module._classification.weight.nonzero(as_tuple=True)
                ]
                logger.info(f"Classifier weights (nonzero): {nonzero_w}")
                logger.info(f"Classifier weights (nonzero) shape: {nonzero_w.shape}")
                if args.bias:
                    logger.info(f"Classifier bias: {net.module._classification.bias}")
                torch.set_printoptions(profile="default")

        train_info = train_pipnet(
            net,
            trainloader,
            optimizer_net,
            optimizer_classifier,
            scheduler_net,
            scheduler_classifier,
            criterion,
            epoch,
            args.epochs,
            device,
            pretrain=False,
            finetune=finetune,
        )
        lrs_net += train_info["lrs_net"]
        lrs_classifier += train_info["lrs_class"]
        # Evaluate model
        eval_info = eval_pipnet(net, testloader, epoch, device, log)
        log.log_values(
            "log_epoch_overview",
            epoch,
            eval_info["top1_accuracy"],
            eval_info["top5_accuracy"],
            eval_info["almost_sim_nonzeros"],
            eval_info["local_size_all_classes"],
            eval_info["almost_nonzeros"],
            eval_info["num non-zero prototypes"],
            train_info["train_accuracy"],
            train_info["loss"],
        )

        global_step = int(args.epochs_pretrain) + int(epoch)
        # Note: for binary classification, eval_pipnet stores F1 into top5_accuracy.
        _wandb_log(
            wandb_run,
            {
                "train/epoch": int(epoch),
                "train/finetune": bool(finetune),
                "train/train_loss": float(train_info.get("loss", 0.0)),
                "train/train_acc": float(train_info.get("train_accuracy", 0.0)),
                "train/lr_net": float(optimizer_net.param_groups[0]["lr"]),
                "train/lr_classifier": float(
                    optimizer_classifier.param_groups[0]["lr"]
                ),
                "eval/test_top1_acc": float(eval_info.get("top1_accuracy", 0.0)),
                "eval/test_top5_or_f1": float(eval_info.get("top5_accuracy", 0.0)),
                "eval/almost_sim_nonzeros": float(
                    eval_info.get("almost_sim_nonzeros", 0.0)
                ),
                "eval/local_size_all_classes": float(
                    eval_info.get("local_size_all_classes", 0.0)
                ),
                "eval/almost_nonzeros": float(eval_info.get("almost_nonzeros", 0.0)),
                "eval/num_nonzero_prototypes": float(
                    eval_info.get("num non-zero prototypes", 0.0)
                ),
            },
            step=global_step,
        )

        with torch.no_grad():
            net.eval()
            torch.save(
                {
                    "model_state_dict": net.state_dict(),
                    "optimizer_net_state_dict": optimizer_net.state_dict(),
                    "optimizer_classifier_state_dict": optimizer_classifier.state_dict(),
                },
                os.path.join(os.path.join(args.log_dir, "checkpoints"), "net_trained"),
            )

            if epoch % 30 == 0:
                net.eval()
                torch.save(
                    {
                        "model_state_dict": net.state_dict(),
                        "optimizer_net_state_dict": optimizer_net.state_dict(),
                        "optimizer_classifier_state_dict": optimizer_classifier.state_dict(),
                    },
                    os.path.join(
                        os.path.join(args.log_dir, "checkpoints"),
                        "net_trained_%s" % str(epoch),
                    ),
                )

            # save learning rate in figure
            plt.clf()
            plt.plot(lrs_net)
            plt.savefig(os.path.join(args.log_dir, "lr_net.png"))
            plt.clf()
            plt.plot(lrs_classifier)
            plt.savefig(os.path.join(args.log_dir, "lr_class.png"))

    net.eval()
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_net_state_dict": optimizer_net.state_dict(),
            "optimizer_classifier_state_dict": optimizer_classifier.state_dict(),
        },
        os.path.join(os.path.join(args.log_dir, "checkpoints"), "net_trained_last"),
    )

    # Post-run evaluations (includes purity computation for CUB and visualization).
    # This makes sure purity outputs exist for training runs too.
    _run_postrun_evals(
        net=net,
        projectloader=projectloader,
        test_projectloader=test_projectloader,
        testloader=testloader,
        classes=classes,
        device=device,
        args=args,
        log=log,
        epoch=args.epochs,
        wandb_run=wandb_run,
    )

    logger.info("Done!")


def _run_postrun_evals(
    net,
    projectloader,
    test_projectloader,
    testloader,
    classes,
    device,
    args,
    log,
    epoch,
    wandb_run=None,
) -> None:
    topks = visualize_topk(
        net, projectloader, len(classes), device, "visualised_prototypes_topk", args
    )
    # set weights of prototypes that are never really found in projection set to 0
    set_to_zero = []
    if topks:
        for prot in topks.keys():
            found = False
            for i_id, score in topks[prot]:
                if score > 0.1:
                    found = True
            if not found:
                torch.nn.init.zeros_(net.module._classification.weight[:, prot])
                set_to_zero.append(prot)
        logger.info(
            f"Weights of prototypes {set_to_zero} are set to zero because it is never detected with similarity>0.1 in the training set"
        )
        eval_info = eval_pipnet(
            net, testloader, "notused" + str(args.epochs), device, log
        )
        log.log_values(
            "log_epoch_overview",
            "notused" + str(args.epochs),
            eval_info["top1_accuracy"],
            eval_info["top5_accuracy"],
            eval_info["almost_sim_nonzeros"],
            eval_info["local_size_all_classes"],
            eval_info["almost_nonzeros"],
            eval_info["num non-zero prototypes"],
            "n.a.",
            "n.a.",
        )

    logger.info(f"Classifier weights: {net.module._classification.weight}")
    nonzero_w = net.module._classification.weight[
        net.module._classification.weight.nonzero(as_tuple=True)
    ]
    logger.info(f"Classifier weights nonzero: {nonzero_w}")
    logger.info(f"Classifier weights nonzero shape: {nonzero_w.shape}")
    logger.info(f"Classifier bias: {net.module._classification.bias}")
    # logger.info weights and relevant prototypes per class
    for c in range(net.module._classification.weight.shape[0]):
        relevant_ps = []
        proto_weights = net.module._classification.weight[c, :]
        for p in range(net.module._classification.weight.shape[1]):
            if proto_weights[p] > 1e-3:
                relevant_ps.append((p, proto_weights[p].item()))
        if args.validation_size == 0.0:
            class_name = None
            try:
                class_name = list(testloader.dataset.class_to_idx.keys())[  # type: ignore[attr-defined]
                    list(testloader.dataset.class_to_idx.values()).index(c)  # type: ignore[attr-defined]
                ]
            except Exception:
                class_name = str(c)
            logger.info(
                f"Class {c} ({class_name}): has {len(relevant_ps)} relevant prototypes."
            )

    # Evaluate prototype purity
    if args.dataset == "CUB-200-2011":
        project_path = args.cub_data_path
        parts_loc_path = os.path.join(project_path, "parts/part_locs.txt")
        parts_name_path = os.path.join(project_path, "parts/parts.txt")
        imgs_id_path = os.path.join(project_path, "images.txt")
        cubthreshold = 0.5

        net.eval()
        logger.info("\n\nEvaluating cub prototypes for training set")
        csvfile_topk = get_topk_cub(
            net, projectloader, 10, "train_" + str(epoch), device, args
        )
        purity_train_topk = eval_prototypes_cub_parts_csv(
            csvfile_topk,
            parts_loc_path,
            parts_name_path,
            imgs_id_path,
            "train_topk_" + str(epoch),
            args,
            log,
        )
        if purity_train_topk:
            _wandb_log(
                wandb_run,
                {f"purity/train_topk/{k}": v for k, v in purity_train_topk.items()},
            )

        csvfile_all = get_proto_patches_cub(
            net,
            projectloader,
            "train_all_" + str(epoch),
            device,
            args,
            threshold=cubthreshold,
        )
        purity_train_all = eval_prototypes_cub_parts_csv(
            csvfile_all,
            parts_loc_path,
            parts_name_path,
            imgs_id_path,
            "train_all_thres" + str(cubthreshold) + "_" + str(epoch),
            args,
            log,
        )
        if purity_train_all:
            _wandb_log(
                wandb_run,
                {f"purity/train_all/{k}": v for k, v in purity_train_all.items()},
            )

        logger.info("\n\nEvaluating cub prototypes for test set")
        csvfile_topk = get_topk_cub(
            net, test_projectloader, 10, "test_" + str(epoch), device, args
        )
        purity_test_topk = eval_prototypes_cub_parts_csv(
            csvfile_topk,
            parts_loc_path,
            parts_name_path,
            imgs_id_path,
            "test_topk_" + str(epoch),
            args,
            log,
        )
        if purity_test_topk:
            _wandb_log(
                wandb_run,
                {f"purity/test_topk/{k}": v for k, v in purity_test_topk.items()},
            )
        cubthreshold = 0.5
        csvfile_all = get_proto_patches_cub(
            net,
            test_projectloader,
            "test_" + str(epoch),
            device,
            args,
            threshold=cubthreshold,
        )
        purity_test_all = eval_prototypes_cub_parts_csv(
            csvfile_all,
            parts_loc_path,
            parts_name_path,
            imgs_id_path,
            "test_all_thres" + str(cubthreshold) + "_" + str(epoch),
            args,
            log,
        )
        if purity_test_all:
            _wandb_log(
                wandb_run,
                {f"purity/test_all/{k}": v for k, v in purity_test_all.items()},
            )

    # visualize predictions
    visualize(net, projectloader, len(classes), device, "visualised_prototypes", args)
    testset_img0_path = test_projectloader.dataset.samples[0][0]
    test_path = os.path.split(os.path.split(testset_img0_path)[0])[0]
    vis_pred(net, test_path, classes, device, args)
    if args.extra_test_image_folder != "":
        if os.path.exists(args.extra_test_image_folder):
            vis_pred_experiments(
                net, args.extra_test_image_folder, classes, device, args
            )

    # EVALUATE OOD DETECTION
    ood_datasets = ["CARS", "CUB-200-2011", "pets"]
    # SKIP
    for percent in []:
        logger.info(f"OOD Evaluation for epoch {epoch} with percent {percent}")
        _, _, _, class_thresholds = get_thresholds(
            net, testloader, epoch, device, percent, log
        )
        logger.info(f"Thresholds: {class_thresholds}")
        # Evaluate with in-distribution data
        id_fraction = eval_ood(net, testloader, epoch, device, class_thresholds)
        logger.info(
            f"ID class threshold ID fraction (TPR) with percent {percent}: {id_fraction}"
        )

        # Evaluate with out-of-distribution data
        for ood_dataset in ood_datasets:
            if ood_dataset != args.dataset:
                logger.info(f"OOD dataset: {ood_dataset}")
                ood_args = deepcopy(args)
                ood_args.dataset = ood_dataset
                _, _, _, _, _, ood_testloader, _, _ = get_dataloaders(ood_args, device)

                id_fraction = eval_ood(
                    net, ood_testloader, epoch, device, class_thresholds
                )
                logger.info(
                    f"{args.dataset} - OOD {ood_dataset} class threshold ID fraction (FPR) with percent {percent}: {id_fraction}"
                )

    logger.info("Done!")


def main_entry() -> None:
    """Wrapper entrypoint that reads CLI args and runs the benchmark.

    This function always parses arguments from the process' argv using
    ``get_args()``; it does not accept an argv parameter for simplicity.
    """
    # Parse args from the current process argv
    args = get_args()

    # Set deterministic seeds from args
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info(f"Parsed arguments: {args}")

    # Prepare log directory (no stdout/stderr redirection)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    # Configure logging for console; benchmark also writes structured logs via util.log
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info(f"Starting purity benchmark; log_dir={args.log_dir}")

    wandb_run = _maybe_init_wandb(args)
    try:
        run_pipnet(args, wandb_run=wandb_run)
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass

    logger.info("Benchmark finished")


if __name__ == "__main__":
    main_entry()
