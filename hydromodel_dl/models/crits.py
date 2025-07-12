from typing import Union
import torch
from torch import distributions as Tensor
from hydrodatautils.foundation.hydro_device import get_the_device
from torch import distributions as tdist

class GaussianLoss(torch.nn.Module):
    def __init__(self, mu=0, sigma=0):
        """Compute the negative log likelihood of Gaussian Distribution
        From https://arxiv.org/abs/1907.00235
        """
        super(GaussianLoss, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x: torch.Tensor):
        loss = -tdist.Normal(self.mu, self.sigma).log_prob(x)
        return torch.sum(loss) / (loss.size(0) * loss.size(1))


class MAELoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        """
        MAE loss which ignores NaN values and supports reduction.
        """
        super(MAELoss, self).__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mask = ~torch.isnan(output) & ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        abs_error = torch.abs(output[mask] - target[mask])
        if self.reduction == 'none':
            return abs_error
        elif self.reduction == 'mean':
            return torch.mean(abs_error)
        elif self.reduction == 'sum':
            return torch.sum(abs_error)
        else:
            raise ValueError(
                f"Unsupported reduction method of loss function: {self.reduction}. Use 'mean', 'sum' or 'none'."
            )

class MSELoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        """
        MSE loss which ignores NaN values and supports reduction.
        """
        super(MSELoss, self).__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mask = ~torch.isnan(output) & ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        sq_error = (output[mask] - target[mask]) ** 2
        if self.reduction == 'none':
            return sq_error
        elif self.reduction == 'mean':
            return torch.mean(sq_error)
        elif self.reduction == 'sum':
            return torch.sum(sq_error)
        else:
            raise ValueError(
                f"Unsupported reduction method of loss function: {self.reduction}. Use 'mean', 'sum' or 'none'."
            )

class RMSELoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        """
        RMSE loss which ignores NaN values and supports reduction.
        """
        super(RMSELoss, self).__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mask = ~torch.isnan(output) & ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        sq_error = (output[mask] - target[mask]) ** 2
        if self.reduction == 'none':
            return torch.sqrt(sq_error)
        elif self.reduction == 'mean':
            return torch.sqrt(torch.mean(sq_error))
        elif self.reduction == 'sum':
            return torch.sqrt(torch.sum(sq_error))
        else:
            raise ValueError(
                f"Unsupported reduction method of loss function: {self.reduction}. Use 'mean', 'sum' or 'none'."
            )

class PESLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        """
        PES Loss: MSE × sigmoid(MSE)
        """
        super(PESLoss, self).__init__()
        self.mse = MSELoss(reduction='none')
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mask = ~torch.isnan(output) & ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        output_masked = output[mask]
        target_masked = target[mask]
        mse_value = self.mse(output_masked, target_masked)
        sigmoid_mse = torch.sigmoid(mse_value)
        loss = mse_value * sigmoid_mse
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise ValueError(
                f"Unsupported reduction method of loss function: {self.reduction}. Use 'mean', 'sum' or 'none'."
            )

class HybridLoss(torch.nn.Module):
    def __init__(self, mae_weight=0.5, reduction='mean'):
        """
        Hybrid Loss: PES loss + mae_weight × MAE
        """
        super(HybridLoss, self).__init__()
        self.pes = PESLoss(reduction='none')
        self.mae = MAELoss(reduction='none')
        self.mae_weight = mae_weight
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mask = ~torch.isnan(output) & ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        output_masked = output[mask]
        target_masked = target[mask]
        pes = self.pes(output_masked, target_masked)
        mae = self.mae(output_masked, target_masked)
        loss = pes + self.mae_weight * mae
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise ValueError(
                f"Unsupported reduction method of loss function: {self.reduction}. Use 'mean', 'sum' or 'none'."
            )
