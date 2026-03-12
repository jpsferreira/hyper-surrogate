from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hyper_surrogate.training.losses import EnergyStressLoss


@dataclass
class TrainingResult:
    model: nn.Module
    history: dict[str, list[float]] = field(default_factory=lambda: {"train_loss": [], "val_loss": []})
    best_epoch: int = 0


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: object,
        val_dataset: object,
        loss_fn: nn.Module,
        lr: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 1000,
        patience: int = 50,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.loss_fn = loss_fn
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=max(1, patience // 3))
        self._is_energy_loss = isinstance(loss_fn, EnergyStressLoss)
        self._best_state: dict | None = None
        self._best_val_loss = float("inf")
        self._best_epoch = 0
        self._patience_counter = 0

    def _compute_loss(self, batch: tuple) -> torch.Tensor:
        x, y = batch
        x = x.to(self.device)

        if self._is_energy_loss:
            x.requires_grad_(True)
            pred = self.model(x)
            if isinstance(y, (tuple, list)):
                w_true = y[0].to(self.device)
                s_true = y[1].to(self.device)
            else:
                w_true = y.to(self.device)
                s_true = torch.zeros_like(x)
            return self.loss_fn(pred, w_true, x, s_true)
        else:
            pred = self.model(x)
            y = y[0].to(self.device) if isinstance(y, (tuple, list)) else y.to(self.device)
            return self.loss_fn(pred, y)

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            loss = self._compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(batch[0])
            n += len(batch[0])
        return total_loss / n

    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad() if not self._is_energy_loss else torch.enable_grad():
            for batch in self.val_loader:
                loss = self._compute_loss(batch)
                total_loss += loss.item() * len(batch[0])
                n += len(batch[0])
        return total_loss / n

    def fit(self) -> TrainingResult:
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(self.max_epochs):
            train_loss = self._train_epoch()
            val_loss = self._val_epoch()
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            self.scheduler.step(val_loss)

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._best_state = copy.deepcopy(self.model.state_dict())
                self._best_epoch = epoch
                self._patience_counter = 0
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.patience:
                    break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        return TrainingResult(model=self.model, history=history, best_epoch=self._best_epoch)
