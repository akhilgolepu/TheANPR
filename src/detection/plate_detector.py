import os
import argparse
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("runs/plate_detector.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def train(args):
    """Main training execution block"""
    logger.info("Initializing training process...")
    logger.info("Hardware allocated: GPU device {args.device} with AMP={args.amp}")

    logger.info("Starting training with the following parameters:")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Model architecture: {args.model_arch}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Image size: {args.img_size}")
    logger.info(f"Learning rate: {args.learning_rate}")

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        logger.error(f"Dataset path {data_path} does not exist.")
        return

    resume_path = Path(args.project) / args.name / "wights" / "last.pt"
    if resume_path.exists():
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        model_target = str(model_path)
    else:
        logger.info(f"No checkpoint found. Starting training from scratch: {args.model}")
        model_target = args.model

    try:
        model = YOLO(model_target)
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            amp=args.amp
            project=args.project,
            name=args.name
            exist_ok=True,
            lr0=args.learning_rate,
            optimiser=args.optimizer,
            weight_decay=args.weight_decay,
            cos_lr=args.cos_lr,
            resume=args.resume
        )
        logger.info("Training completed successfully.")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exec_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANPR Plate Detector Training Pipeline")

    parser.add_argument("--model", type=str, defaults="yolo11m.pt", help="Base model architechture")
    parser.add_argument("--data", type=str, defaults="configs/plate_detection.yaml", help="Path to data YAML")

    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (Optimised for 8GB vram)")
    parser.add_argument("--img_size", type=int, default=640, help="Image resolution")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    
    parser.add_argument("--optimizer", type=str, default="auto", choices=["auto", "SGD", "Adam", "AdamW", "RMSProp"], help="Optimizer to use (e.g., AdamW, SGD)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="L2 regularization penalty")
    parser.add_argument("--cos_lr", type=bool, default=False, help="Use cosine learning rate scheduler")

    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision")
    parser.add_argument("--resume", action="store_true", default=True, help="Auto-resume from last.pt if exists")

    parser.add_argument("--project", type=str, default="runs/detect", help="MLflow/YOLO project root")
    parser.add_argument("--name", type=str, default="plate_detector_v3", help="Experiment name")

    args = parser.parse_args()
    train(args)