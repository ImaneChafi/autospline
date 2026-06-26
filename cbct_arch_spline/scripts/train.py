"""
Training script.

Usage:
    python scripts/train.py --cbct_dir /path/to/nifti_files --epochs 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ANNOTATIONS_DIR, MODELS_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from data.dataset import CBCTArchDataset
from models.unet import build_model
from training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CBCT arch spline detector")
    p.add_argument("--cbct_dir", type=str, required=True,
                   help="Directory with NIfTI CBCT files (*.nii.gz)")
    p.add_argument("--fcsv_dir", type=str, default=str(ANNOTATIONS_DIR),
                   help="Directory with .fcsv annotation files")
    p.add_argument("--save_dir", type=str, default=str(MODELS_DIR))
    p.add_argument("--experiment", type=str, default="arch_detector")
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--no_pretrained", action="store_true",
                   help="Use SimpleUNet instead of EfficientNet backbone")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading dataset from {args.cbct_dir}")
    dataset = CBCTArchDataset(
        cbct_dir=args.cbct_dir,
        fcsv_dir=args.fcsv_dir,
        augment=True,
    )
    print(f"Found {len(dataset)} samples")

    model = build_model(use_pretrained=not args.no_pretrained)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {type(model).__name__} — {n_params:,} trainable parameters")

    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        experiment_name=args.experiment,
    )
    history = trainer.train()

    best_val = min(history["val_loss"])
    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
