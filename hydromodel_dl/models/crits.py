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
    def __init__(self, reduction="mean"):
        """
        MAE loss which ignores NaN values and supports reduction.
        """
        super(MAELoss, self).__init__()
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mask = ~torch.isnan(output) & ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        abs_error = torch.abs(output[mask] - target[mask])
        if self.reduction == "none":
            return abs_error
        elif self.reduction == "mean":
            return torch.mean(abs_error)
        elif self.reduction == "sum":
            return torch.sum(abs_error)
        else:
            raise ValueError(
                f"Unsupported reduction method of loss function: {self.reduction}. Use 'mean', 'sum' or 'none'."
            )


class MSELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        """
        MSE loss which ignores NaN values and supports reduction.
        """
        super(MSELoss, self).__init__()
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mask = ~torch.isnan(output) & ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        sq_error = (output[mask] - target[mask]) ** 2
        if self.reduction == "none":
            return sq_error
        elif self.reduction == "mean":
            return torch.mean(sq_error)
        elif self.reduction == "sum":
            return torch.sum(sq_error)
        else:
            raise ValueError(
                f"Unsupported reduction method of loss function: {self.reduction}. Use 'mean', 'sum' or 'none'."
            )


class RMSELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        """
        RMSE loss which ignores NaN values and supports reduction.
        """
        super(RMSELoss, self).__init__()
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mask = ~torch.isnan(output) & ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        sq_error = (output[mask] - target[mask]) ** 2
        if self.reduction == "none":
            return torch.sqrt(sq_error)
        elif self.reduction == "mean":
            return torch.sqrt(torch.mean(sq_error))
        elif self.reduction == "sum":
            return torch.sqrt(torch.sum(sq_error))
        else:
            raise ValueError(
                f"Unsupported reduction method of loss function: {self.reduction}. Use 'mean', 'sum' or 'none'."
            )


class PESLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        """
        PES Loss: MSE × sigmoid(MSE)
        """
        super(PESLoss, self).__init__()
        self.mse = MSELoss(reduction="none")
        assert reduction in ("none", "mean", "sum")
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
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            raise ValueError(
                f"Unsupported reduction method of loss function: {self.reduction}. Use 'mean', 'sum' or 'none'."
            )


class HybridLoss(torch.nn.Module):
    def __init__(self, mae_weight=0.5, reduction="mean"):
        """
        Hybrid Loss: PES loss + mae_weight × MAE
        """
        super(HybridLoss, self).__init__()
        self.pes = PESLoss(reduction="none")
        self.mae = MAELoss(reduction="none")
        self.mae_weight = mae_weight
        assert reduction in ("none", "mean", "sum")
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
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            raise ValueError(
                f"Unsupported reduction method of loss function: {self.reduction}. Use 'mean', 'sum' or 'none'."
            )

class RMSEFloodLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        """
        RMSE Flood Loss: RMSE loss with flood weighting

        Applies Root Mean Square Error loss with flood event filtering.
        This class filters flood events first then calculates RMSE loss,
        focusing computation only on flood periods.

        Parameters
        ----------
        reduction : str
            Reduction method for RMSE loss, default is "mean"
        """
        super(RMSEFloodLoss, self).__init__()
        self.reduction = reduction

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, flood_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flood-aware RMSE loss.

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions [batch_size, seq_len, output_features]
        targets : torch.Tensor
            Target values [batch_size, seq_len, output_features]
        flood_mask : torch.Tensor
            Flood mask [batch_size, seq_len, 1] (1 for flood, 0 for normal)

        Returns
        -------
        torch.Tensor
            Computed RMSE loss value
        """
        boolean_mask = flood_mask.to(torch.bool)
        predictions = predictions[boolean_mask]
        targets = targets[boolean_mask]

        base_loss_func = RMSELoss(self.reduction)
        return base_loss_func(predictions, targets)


class PESFloodEvent(torch.nn.Module):
    def __init__(self):
        """
        PES Flood Event Loss: PES loss with flood event filtering

        Applies PES loss (MSE × sigmoid(MSE)) with flood event filtering.
        This class filters flood events first then calculates PES loss,
        focusing computation only on flood periods.

        The difference from standard PES loss is that this class filters flood events first,
        because PES does sigmoid on MSE, when we want to focus only on flood events,
        we need to filter them out first before applying the PES calculation.
        """
        super(PESFloodEvent, self).__init__()

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, flood_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flood-aware PES loss.

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions [batch_size, seq_len, output_features]
        targets : torch.Tensor
            Target values [batch_size, seq_len, output_features]
        flood_mask : torch.Tensor
            Flood mask [batch_size, seq_len, 1] (1 for flood, 0 for normal)

        Returns
        -------
        torch.Tensor
            Computed PES loss value
        """
        boolean_mask = flood_mask.to(torch.bool)
        predictions = predictions[boolean_mask]
        targets = targets[boolean_mask]

        base_loss_func = PESLoss()
        return base_loss_func(predictions, targets)


class HybridFloodLoss(torch.nn.Module):
    def __init__(self, mae_weight=0.5):
        """
        Hybrid Flood Loss: PES loss + mae_weight × MAE with flood weighting

        Combines PES loss (MSE × sigmoid(MSE)) with Mean Absolute Error,
        applying flood weighting to the loss.

        The difference from FloodLoss is that this class filter flood events first then calculate loss,
        because Hybrid does sigmoid on MSE, when the non-flood-weight is 0, which means we do not want to
        calculate loss on non-flood events, so we need to filter them out first.

        Parameters
        ----------
        mae_weight : float
            Weight for the MAE component, default is 0.5
        flood_weight : float
            Weight multiplier for flood events, default is 2.0
        """
        super(HybridFloodLoss, self).__init__()
        self.mae_weight = mae_weight

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, flood_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flood-aware loss using the specified strategy.

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions [batch_size, seq_len, output_features]
        targets : torch.Tensor
            Target values [batch_size, seq_len, output_features]
        flood_mask : torch.Tensor
            Flood mask [batch_size, seq_len, 1] (1 for flood, 0 for normal)

        Returns
        -------
        torch.Tensor
            Computed loss value
        """
        boolean_mask = flood_mask.to(torch.bool)
        predictions = predictions[boolean_mask]
        targets = targets[boolean_mask]

        base_loss_func = HybridLoss(self.mae_weight)
        return base_loss_func(predictions, targets)




































