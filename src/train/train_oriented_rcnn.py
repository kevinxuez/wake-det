"""
Step 2b – Oriented R-CNN Training (Paper-Based Approach)
Trains an Oriented R-CNN detector for SAR ship-wake detection, following the
architecture described in the OpenSARWake paper (Xu & Wang, 2024).

The paper's SWNet uses ConvNeXt-T backbone + HR-FPN* + Oriented R-CNN head,
achieving 49.0% mAP on the OpenSARWake test set.  This script implements the
closest available configuration using the MMRotate framework.

Requirements::

    pip install mmengine mmcv mmdet mmrotate

Usage::

    # Convert labels and train
    python src/train/train_oriented_rcnn.py \
        --config configs/train_config.yaml \
        --data-root data/splits

    # Convert labels only (no training)
    python src/train/train_oriented_rcnn.py \
        --data-root data/splits --convert-only
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label format conversion: YOLO-OBB → DOTA
# ---------------------------------------------------------------------------

def convert_yolo_obb_to_dota(
    data_root: Path,
    output_root: Path | None = None,
    image_size: int = 1024,
    class_names: list[str] | None = None,
) -> Path:
    """Convert YOLO-OBB labels (normalized polygon) to DOTA format.

    YOLO-OBB format (per line)::

        class_id  x1 y1  x2 y2  x3 y3  x4 y4   (normalised 0-1)

    DOTA format (per line)::

        x1 y1 x2 y2 x3 y3 x4 y4  class_name  difficulty

    Args:
        data_root: Path containing ``{train,val,test}/labels/*.txt``.
        output_root: Where to write DOTA-format labels.  Defaults to
            ``data_root / "dota_labels"``.
        image_size: Pixel dimension to denormalise coordinates (default 1024).
        class_names: Mapping from class ID to name.  Defaults to
            ``["ship_wake"]``.

    Returns:
        Path to the output root directory.
    """
    if class_names is None:
        class_names = ["ship_wake"]
    if output_root is None:
        output_root = data_root / "dota_labels"

    for split in ("train", "val", "test"):
        src_dir = data_root / split / "labels"
        dst_dir = output_root / split
        if not src_dir.exists():
            logger.warning("Skipping split '%s': %s not found", split, src_dir)
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        count = 0

        for txt_file in sorted(src_dir.glob("*.txt")):
            lines_out: list[str] = []
            content = txt_file.read_text().strip()
            if not content:
                # Negative sample — write empty DOTA label
                (dst_dir / txt_file.name).write_text("")
                count += 1
                continue

            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) != 9:
                    logger.warning("Malformed line in %s: %s", txt_file, line)
                    continue

                cls_id = int(parts[0])
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

                # Denormalise the 4 polygon corners
                coords = [float(v) * image_size for v in parts[1:]]
                coord_str = " ".join(f"{c:.1f}" for c in coords)
                lines_out.append(f"{coord_str} {cls_name} 0")

            (dst_dir / txt_file.name).write_text("\n".join(lines_out) + "\n")
            count += 1

        logger.info("Converted %d label files for split '%s' → %s", count, split, dst_dir)

    return output_root


# ---------------------------------------------------------------------------
# MMRotate config builder
# ---------------------------------------------------------------------------

def build_mmrotate_config(
    data_root: Path,
    dota_label_root: Path,
    backbone: str = "convnext_tiny",
    epochs: int = 600,
    learning_rate: float = 0.0001,
    batch_size: int = 8,
    image_size: int = 1024,
    device: str = "mps",
    save_dir: str = "runs/train_orcnn",
) -> dict:
    """Build a MMRotate-compatible config dict for Oriented R-CNN.

    This programmatically constructs a config that mirrors the paper:
    ConvNeXt-T backbone, FPN neck, Oriented R-CNN head, Adam optimizer
    with step LR decay at epochs 400 and 500.

    Args:
        data_root: Dataset root with ``{train,val,test}/images/``.
        dota_label_root: DOTA-format label root with ``{train,val,test}/``.
        backbone: Backbone name (default ``"convnext_tiny"``).
        epochs: Total training epochs (default 600, per paper).
        learning_rate: Initial LR (default 0.0001, per paper).
        batch_size: Batch size (default 8).
        image_size: Image size (default 1024).
        device: Compute device.
        save_dir: Output directory.

    Returns:
        Config dict for the Oriented R-CNN training run.
    """
    config = {
        "model": {
            "type": "OrientedRCNN",
            "backbone": {
                "type": "mmdet.ConvNeXt",
                "arch": "tiny",
                "out_indices": [0, 1, 2, 3],
                "drop_path_rate": 0.4,
                "layer_scale_init_value": 1.0,
                "gap_before_final_norm": False,
                "init_cfg": {
                    "type": "Pretrained",
                    "checkpoint": "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_32xb128-noema_in1k_20221114-b0802f26.pth",
                    "prefix": "backbone.",
                },
            },
            "neck": {
                "type": "FPN",
                "in_channels": [96, 192, 384, 768],
                "out_channels": 256,
                "num_outs": 5,
            },
            "rpn_head": {
                "type": "OrientedRPNHead",
                "in_channels": 256,
                "feat_channels": 256,
                "anchor_generator": {
                    "type": "AnchorGenerator",
                    "scales": [8],
                    "ratios": [0.5, 1.0, 2.0],
                    "strides": [4, 8, 16, 32, 64],
                },
                "bbox_coder": {
                    "type": "MidpointOffsetCoder",
                    "angle_range": "oc",
                },
                "loss_cls": {
                    "type": "mmdet.CrossEntropyLoss",
                    "use_sigmoid": True,
                    "loss_weight": 1.0,
                },
                "loss_bbox": {
                    "type": "mmdet.SmoothL1Loss",
                    "beta": 1.0 / 9.0,
                    "loss_weight": 1.0,
                },
            },
            "roi_head": {
                "type": "OrientedStandardRoIHead",
                "bbox_roi_extractor": {
                    "type": "RotatedSingleRoIExtractor",
                    "roi_layer": {"type": "RoIAlignRotated", "out_size": 7, "sample_num": 2},
                    "out_channels": 256,
                    "featmap_strides": [4, 8, 16, 32],
                },
                "bbox_head": {
                    "type": "RotatedShared2FCBBoxHead",
                    "in_channels": 256,
                    "fc_out_channels": 1024,
                    "roi_feat_size": 7,
                    "num_classes": 1,
                    "bbox_coder": {
                        "type": "DeltaXYWHTRBBoxCoder",
                        "angle_range": "oc",
                        "norm_factor": None,
                        "edge_swap": True,
                        "proj_xy": True,
                        "target_means": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "target_stds": [0.1, 0.1, 0.2, 0.2, 0.1],
                    },
                    "reg_class_agnostic": True,
                    "loss_cls": {
                        "type": "mmdet.CrossEntropyLoss",
                        "use_sigmoid": False,
                        "loss_weight": 1.0,
                    },
                    "loss_bbox": {
                        "type": "mmdet.SmoothL1Loss",
                        "beta": 1.0,
                        "loss_weight": 1.0,
                    },
                },
            },
        },
        "dataset": {
            "type": "DOTADataset",
            "data_root": str(data_root),
            "dota_label_root": str(dota_label_root),
            "image_size": image_size,
        },
        "optimizer": {
            "type": "Adam",
            "lr": learning_rate,
        },
        "lr_schedule": {
            "type": "step",
            "milestones": [400, 500],
            "gamma": 0.1,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "save_dir": save_dir,
        },
    }
    return config


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_root: Path,
    dota_label_root: Path,
    backbone: str = "convnext_tiny",
    epochs: int = 600,
    learning_rate: float = 0.0001,
    batch_size: int = 8,
    image_size: int = 1024,
    device: str = "mps",
    save_dir: str = "runs/train_orcnn",
) -> Path:
    """Train Oriented R-CNN on OpenSARWake using MMRotate.

    Args:
        data_root: Dataset root with ``{train,val,test}/images/``.
        dota_label_root: DOTA-format label root.
        backbone: Backbone name (default ``"convnext_tiny"``).
        epochs: Training epochs (default 600, per paper).
        learning_rate: Initial LR (default 0.0001, per paper).
        batch_size: Batch size (default 8).
        image_size: Image size (default 1024).
        device: Compute device.
        save_dir: Output directory for weights/logs.

    Returns:
        Path to the best checkpoint file.
    """
    try:
        from mmengine.runner import Runner
        from mmengine.config import Config as MmConfig
    except ImportError as exc:
        raise ImportError(
            "MMRotate stack is required. Install with:\n"
            "  pip install mmengine mmcv mmdet mmrotate\n"
            "See: https://mmrotate.readthedocs.io/en/1.x/get_started.html"
        ) from exc

    cfg = build_mmrotate_config(
        data_root=data_root,
        dota_label_root=dota_label_root,
        backbone=backbone,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        image_size=image_size,
        device=device,
        save_dir=save_dir,
    )

    # Write config to disk for reproducibility
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    config_out = save_path / "oriented_rcnn_config.yaml"
    with open(config_out, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False)
    logger.info("Config saved to %s", config_out)

    # Build MMRotate config from the model/dataset/scheduler spec
    mm_cfg = _build_mmrotate_runner_config(cfg, data_root, dota_label_root)

    logger.info("=" * 50)
    logger.info("Oriented R-CNN — Training Run (Paper Config)")
    logger.info("=" * 50)
    logger.info("Backbone: %s | Epochs: %d | LR: %s | Optimizer: Adam",
                backbone, epochs, learning_rate)
    logger.info("LR decay: ÷10 at epochs 400 and 500 (per paper)")
    logger.info("Image size: %d | Batch: %d | Device: %s", image_size, batch_size, device)

    t0 = time.time()
    runner = Runner.from_cfg(mm_cfg)
    runner.train()
    elapsed = time.time() - t0

    minutes, seconds = divmod(int(elapsed), 60)
    hours, minutes = divmod(minutes, 60)
    logger.info("Training complete in %dh %dm %ds", hours, minutes, seconds)

    # Find best checkpoint
    ckpt_dir = save_path / "checkpoints"
    best = ckpt_dir / "best.pth" if (ckpt_dir / "best.pth").exists() else None
    if best is None:
        pths = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
        best = pths[-1] if pths else save_path
    logger.info("Best checkpoint: %s", best)
    return best


def _build_mmrotate_runner_config(cfg: dict, data_root: Path, dota_label_root: Path):
    """Translate our config dict into a full MMEngine Runner config.

    This constructs the dataset pipelines, dataloaders, model, optimizer,
    and scheduler configs that MMEngine's Runner expects.
    """
    from mmengine.config import Config as MmConfig

    image_size = cfg["dataset"]["image_size"]
    train_cfg = cfg["training"]
    opt_cfg = cfg["optimizer"]
    lr_cfg = cfg["lr_schedule"]

    train_pipeline = [
        {"type": "mmdet.LoadImageFromFile"},
        {"type": "mmdet.LoadAnnotations", "with_bbox": True, "box_type": "qbox"},
        {"type": "ConvertBoxType", "box_type_mapping": {"gt_bboxes": "rbox"}},
        {"type": "mmdet.Resize", "scale": (image_size, image_size), "keep_ratio": True},
        {"type": "mmdet.RandomFlip", "prob": 0.5, "direction": ["horizontal", "vertical"]},
        {"type": "RandomRotate", "prob": 0.5, "angle_range": 90},
        {"type": "mmdet.PackDetInputs"},
    ]

    val_pipeline = [
        {"type": "mmdet.LoadImageFromFile"},
        {"type": "mmdet.LoadAnnotations", "with_bbox": True, "box_type": "qbox"},
        {"type": "ConvertBoxType", "box_type_mapping": {"gt_bboxes": "rbox"}},
        {"type": "mmdet.Resize", "scale": (image_size, image_size), "keep_ratio": True},
        {"type": "mmdet.PackDetInputs"},
    ]

    mm_dict = {
        "model": cfg["model"],
        "train_dataloader": {
            "batch_size": train_cfg["batch_size"],
            "num_workers": 4,
            "persistent_workers": True,
            "sampler": {"type": "DefaultSampler", "shuffle": True},
            "dataset": {
                "type": "DOTADataset",
                "data_root": str(data_root),
                "ann_file": str(dota_label_root / "train") + "/",
                "data_prefix": {"img_path": "train/images"},
                "filter_cfg": {"filter_empty_gt": False},
                "pipeline": train_pipeline,
            },
        },
        "val_dataloader": {
            "batch_size": 1,
            "num_workers": 2,
            "persistent_workers": True,
            "sampler": {"type": "DefaultSampler", "shuffle": False},
            "dataset": {
                "type": "DOTADataset",
                "data_root": str(data_root),
                "ann_file": str(dota_label_root / "val") + "/",
                "data_prefix": {"img_path": "val/images"},
                "filter_cfg": {"filter_empty_gt": False},
                "pipeline": val_pipeline,
            },
        },
        "val_evaluator": {
            "type": "DOTAMetric",
            "metric": "mAP",
        },
        "optim_wrapper": {
            "optimizer": {
                "type": opt_cfg["type"],
                "lr": opt_cfg["lr"],
            },
        },
        "param_scheduler": [
            {
                "type": "MultiStepLR",
                "begin": 0,
                "end": train_cfg["epochs"],
                "milestones": lr_cfg["milestones"],
                "gamma": lr_cfg["gamma"],
                "by_epoch": True,
            },
        ],
        "train_cfg": {
            "type": "EpochBasedTrainLoop",
            "max_epochs": train_cfg["epochs"],
            "val_interval": 10,
        },
        "val_cfg": {"type": "ValLoop"},
        "default_hooks": {
            "checkpoint": {
                "type": "CheckpointHook",
                "interval": 50,
                "save_best": "auto",
            },
            "logger": {
                "type": "LoggerHook",
                "interval": 50,
            },
        },
        "work_dir": train_cfg["save_dir"],
        "default_scope": "mmrotate",
    }

    return MmConfig(mm_dict)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Oriented R-CNN for SAR ship-wake detection (paper config)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_config.yaml"),
        help="Path to training config YAML (default: configs/train_config.yaml).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/splits"),
        help="Dataset root containing {train,val,test}/{images,labels}/.",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert YOLO-OBB labels to DOTA format, then exit.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    # Load config
    cfg_path: Path = args.config
    if cfg_path.exists():
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh) or {}
    else:
        logger.warning("Config %s not found; using defaults.", cfg_path)
        cfg = {}

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    orcnn_cfg = cfg.get("oriented_rcnn", {})

    data_root = args.data_root
    image_size = data_cfg.get("image_size", 1024)

    # Step 1: Convert labels
    logger.info("Converting YOLO-OBB labels to DOTA format...")
    dota_root = convert_yolo_obb_to_dota(
        data_root=data_root,
        image_size=image_size,
    )

    if args.convert_only:
        logger.info("Label conversion complete. Exiting (--convert-only).")
        return

    # Step 2: Train
    logger.info("=" * 50)
    logger.info("SAR Ship-Wake Detection — Oriented R-CNN")
    logger.info("=" * 50)

    train(
        data_root=data_root,
        dota_label_root=dota_root,
        backbone=orcnn_cfg.get("backbone", "convnext_tiny"),
        epochs=train_cfg.get("epochs", 600),
        learning_rate=train_cfg.get("learning_rate", 0.0001),
        batch_size=train_cfg.get("batch_size", 8),
        image_size=image_size,
        device=str(train_cfg.get("device", "mps")),
        save_dir="runs/train_orcnn",
    )


if __name__ == "__main__":
    main()
