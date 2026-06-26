"""Training and validation loop."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    VAL_SPLIT,
    SEED,
    MODELS_DIR,
)
from training.losses import CombinedHeatmapLoss


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataset,
        device: str = "auto",
        batch_size: int = BATCH_SIZE,
        lr: float = LEARNING_RATE,
        num_epochs: int = NUM_EPOCHS,
        val_split: float = VAL_SPLIT,
        save_dir: str | Path = MODELS_DIR,
        experiment_name: str = "arch_detector",
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
            if device == "auto" else device
        )
        self.model = model.to(self.device)
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        # Split dataset
        n_val = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        generator = torch.Generator().manual_seed(SEED)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )

        self.criterion = CombinedHeatmapLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=lr * 0.01
        )

        self.best_val_loss = float("inf")
        self.history: dict[str, list] = {"train_loss": [], "val_loss": []}

    def train(self) -> dict:
        print(f"Training on {self.device} | {len(self.train_loader.dataset)} train, "
              f"{len(self.val_loader.dataset)} val samples")

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            print(f"Epoch {epoch:3d}/{self.num_epochs} | "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best.pth")

        self._save_checkpoint("last.pth")
        return self.history

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
                images = batch["image"].to(self.device)
                heatmaps = batch["heatmap"].to(self.device)

                preds = self.model(images)
                loss = self.criterion(preds, heatmaps)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item() * len(images)

        return total_loss / len(loader.dataset)

    def _save_checkpoint(self, filename: str) -> None:
        path = self.save_dir / f"{self.experiment_name}_{filename}"
        torch.save(
            {
                "epoch": self.num_epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "history": self.history,
            },
            path,
        )
        print(f"  Saved: {path}")
