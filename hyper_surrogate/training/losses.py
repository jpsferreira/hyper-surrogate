from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StressLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class StressTangentLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred and target are both (N, 27): first 6 = stress, rest = tangent."""
        s_pred, c_pred = pred[:, :6], pred[:, 6:]
        s_true, c_true = target[:, :6], target[:, 6:]
        return self.alpha * F.mse_loss(s_pred, s_true) + self.beta * F.mse_loss(c_pred, c_true)


class EnergyStressLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        w_pred: torch.Tensor,
        w_true: torch.Tensor,
        inputs: torch.Tensor,
        s_true: torch.Tensor,
    ) -> torch.Tensor:
        energy_loss = F.mse_loss(w_pred, w_true)
        dw_di = torch.autograd.grad(w_pred.sum(), inputs, create_graph=True)[0]
        stress_loss = F.mse_loss(dw_di, s_true)
        return self.alpha * energy_loss + self.beta * stress_loss


class SparseLoss(nn.Module):
    """Energy-stress loss with L1 regularization on model weights for CANN model discovery.

    Encourages sparsity in CANN basis function weights, so surviving terms
    reveal the minimal constitutive law.
    """

    def __init__(self, base_loss: nn.Module, l1_lambda: float = 0.01, weight_param: str = "raw_weights") -> None:
        super().__init__()
        self.base_loss = base_loss
        self.l1_lambda = l1_lambda
        self.weight_param = weight_param

    def forward(self, *args: torch.Tensor, model: nn.Module | None = None, **kwargs: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = self.base_loss(*args, **kwargs)
        if model is not None:
            for name, param in model.named_parameters():
                if self.weight_param in name:
                    loss = loss + self.l1_lambda * torch.abs(param).sum()
        return loss
