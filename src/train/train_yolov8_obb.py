"""
Step 2 – YOLOv8-OBB Model Training
Trains a YOLOv8-OBB (Oriented Bounding Box) model for SAR ship-wake detection
using the OpenSARWake dataset (3,973 images, L/C/X-band, OBB annotations).

YOLOv8-OBB uses an anchor-free mechanism with rotation-invariant augmentations,
achieving ~86 % recognition accuracy on C-band SAR ship-wake imagery.

Usage::

    python src/train/train_yolov8_obb.py \
        --config configs/train_config.yaml \
        --data   data/splits/dataset.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset YAML helper
# ---------------------------------------------------------------------------

def build_dataset_yaml(
    data_root: Path,
    output_path: Path,
    class_names: list[str] | None = None,
) -> Path:
    """Generate a YOLOv8-compatible dataset YAML file.

    Args:
        data_root: Root directory that contains ``train/``, ``val/``, and
            optionally ``test/`` sub-directories of images.
        output_path: Where to write the YAML file.
        class_names: List of class names.  Defaults to ``["ship_wake"]``.

    Returns:
        Path to the written YAML file.
    """
    if class_names is None:
        class_names = ["ship_wake"]

    config: dict = {
        "path": str(data_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        yaml.dump(config, fh, default_flow_style=False)

    logger.info("Dataset YAML written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train(
    model_variant: str = "yolov8n-obb",
    data_yaml: str | Path = "data/dataset.yaml",
    image_size: int = 1024,
    epochs: int = 300,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    optimizer: str = "AdamW",
    device: str = "0",
    save_dir: str | Path = "runs/train",
    degrees: float = 45.0,
    mosaic: float = 1.0,
) -> Path:
    """Train a YOLOv8-OBB model for SAR ship-wake detection.

    Rotation-invariant augmentations (``degrees``, ``mosaic``) are key for
    detecting diagonal ship wakes at arbitrary angles.

    Hyperparameter choices informed by the OpenSARWake paper (Xu & Wang,
    2024): Adam-family optimizer, lower learning rate, and extended training.

    Args:
        model_variant: YOLOv8 OBB model variant (e.g. ``yolov8n-obb``).
        data_yaml: Path to the dataset YAML file.
        image_size: Input image size in pixels (default 1024).
        epochs: Number of training epochs (default 300).
        batch_size: Training batch size (default 8).
        learning_rate: Initial learning rate (default 0.001).
        optimizer: Optimizer name – ``"AdamW"``, ``"Adam"``, ``"SGD"``, etc.
            (default ``"AdamW"``; paper uses Adam).
        device: Compute device – GPU index, ``"mps"``, or ``"cpu"``.
        save_dir: Root directory for training artefacts.
        degrees: Maximum rotation augmentation in degrees (default 45).
        mosaic: Mosaic augmentation probability (default 1.0).

    Returns:
        Path to the best weights file (``best.pt``) produced by the run.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'ultralytics' package is required.  Install with: "
            "pip install ultralytics"
        ) from exc

    logger.info("Loading model: %s (pretrained weights)", model_variant)
    model = YOLO(model_variant)

    logger.info("Dataset YAML: %s", data_yaml)
    logger.info("Device: %s | Image size: %d | Batch: %d", device, image_size, batch_size)
    logger.info("Optimizer: %s | LR: %s", optimizer, learning_rate)
    logger.info("Augmentation: degrees=%.1f, mosaic=%.1f, flipud=0.5, fliplr=0.5",
                degrees, mosaic)
    logger.info("Starting training for %d epochs...", epochs)

    t0 = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        lr0=learning_rate,
        optimizer=optimizer,
        device=device,
        project=str(save_dir),
        degrees=degrees,
        mosaic=mosaic,
        flipud=0.5,
        fliplr=0.5,
    )
    elapsed = time.time() - t0

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    minutes, seconds = divmod(int(elapsed), 60)
    hours, minutes = divmod(minutes, 60)
    logger.info("Training complete in %dh %dm %ds", hours, minutes, seconds)
    logger.info("Best weights: %s", best_weights)
    logger.info("Results dir: %s", results.save_dir)
    return best_weights


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-OBB for SAR ship-wake detection."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_config.yaml"),
        help="Path to training configuration YAML (default: configs/train_config.yaml).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to dataset YAML.  Overrides the value in --config.",
    )
    parser.add_argument(
        "--generate-dataset-yaml",
        type=Path,
        default=None,
        metavar="DATA_ROOT",
        help=(
            "If provided, auto-generate a dataset YAML from the given data "
            "root directory and exit."
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    # Optionally just generate the dataset YAML and stop.
    if args.generate_dataset_yaml is not None:
        build_dataset_yaml(
            data_root=args.generate_dataset_yaml,
            output_path=args.generate_dataset_yaml / "dataset.yaml",
        )
        return

    # Load training config.
    cfg_path: Path = args.config
    if not cfg_path.exists():
        logger.warning(
            "Config file %s not found; using built-in defaults.", cfg_path
        )
        cfg: dict = {}
    else:
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh) or {}

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    aug_cfg = cfg.get("augmentation", {})
    out_cfg = cfg.get("output", {})

    data_yaml = args.data or Path(data_cfg.get("dataset_root", "data/splits/")) / "dataset.yaml"

    logger.info("=" * 50)
    logger.info("SAR Ship-Wake Detection — Training Run")
    logger.info("=" * 50)

    train(
        model_variant=model_cfg.get("architecture", "yolov8n-obb"),
        data_yaml=data_yaml,
        image_size=data_cfg.get("image_size", 1024),
        epochs=train_cfg.get("epochs", 300),
        batch_size=train_cfg.get("batch_size", 8),
        learning_rate=train_cfg.get("learning_rate", 0.001),
        optimizer=train_cfg.get("optimizer", "AdamW"),
        device=str(train_cfg.get("device", "0")),
        save_dir=out_cfg.get("save_dir", "runs/train"),
        degrees=aug_cfg.get("degrees", 45.0),
        mosaic=aug_cfg.get("mosaic", 1.0),
    )


if __name__ == "__main__":
    main()
