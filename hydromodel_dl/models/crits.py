from typing import Union
import torch
from torch import distributions as tdist


class GaussianLoss(torch.nn.Module):
    """
    Compute the negative log likelihood of Gaussian Distribution.
    
    Reference: https://arxiv.org/abs/1907.00235
    """
    
    def __init__(self, mu=0, sigma=0):
        """
        Initialize GaussianLoss.
        
        Parameters
        ----------
        mu : float or torch.Tensor, default=0
            Mean of the Gaussian distribution
        sigma : float or torch.Tensor, default=0
            Standard deviation of the Gaussian distribution
        """
        super(GaussianLoss, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log likelihood loss.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Tensor
            Normalized negative log likelihood loss
        """
        # Ensure mu and sigma are on the same device and dtype as x
        if isinstance(self.mu, torch.Tensor):
            mu = self.mu.to(x.device)
        else:
            mu = torch.tensor(self.mu, device=x.device, dtype=x.dtype)
            
        if isinstance(self.sigma, torch.Tensor):
            sigma = self.sigma.to(x.device)
        else:
            sigma = torch.tensor(self.sigma, device=x.device, dtype=x.dtype)
        
        # Ensure sigma is positive to avoid numerical issues
        sigma = torch.clamp(sigma, min=1e-6)
        
        # Create Normal distribution - PyTorch will handle broadcasting automatically
        loss = -tdist.Normal(mu, sigma).log_prob(x)
        
        # Use numel() to get total number of elements for normalization (works for any shape)
        return torch.sum(loss) / loss.numel()


class RMSELoss(torch.nn.Module):
    """
    RMSE loss which ignores NaN values and supports reduction.
    """
    
    def __init__(self, reduction="mean"):
        """
        Initialize RMSELoss.
        
        Parameters
        ----------
        reduction : str, default="mean"
            Reduction method: "none", "mean", or "sum"
        """
        super(RMSELoss, self).__init__()
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute RMSE loss.
        
        Parameters
        ----------
        output : torch.Tensor
            Model predictions
        target : torch.Tensor
            Target values
            
        Returns
        -------
        torch.Tensor
            RMSE loss value
        """
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
                f"Unsupported reduction method of loss function: {self.reduction}. "
                f"Use 'mean', 'sum' or 'none'."
            )


class RMSEFloodSampleLoss(torch.nn.Module):
    """
    RMSE loss with flood event filtering (computed per sample).
    
    Applies Root Mean Square Error loss with flood event filtering.
    This class filters flood events first then calculates RMSE loss,
    focusing computation only on flood periods. Loss is computed independently
    for each sample in the batch.
    """
    
    def __init__(self, reduction="mean"):
        """
        Initialize RMSEFloodSampleLoss.
        
        Parameters
        ----------
        reduction : str, default="mean"
            Reduction method: "none", "mean", or "sum"
        """
        super(RMSEFloodSampleLoss, self).__init__()
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        flood_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flood-aware RMSE loss.

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions, shape: [batch_size, seq_len, output_features]
        targets : torch.Tensor
            Target values, shape: [batch_size, seq_len, output_features]
        flood_mask : torch.Tensor
            Flood mask, shape: [batch_size, seq_len, 1] (1 for flood, 0 for normal)

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


class RMSEFloodEventLoss(torch.nn.Module):
    """
    RMSE loss for flood events (computed per event).
    
    Implementation (following paper formula):
    - For each flood event i in the batch:
      1) Merge flood period data from all samples of this event (identified by event_id)
      2) Compute RMSE on the merged data for this event
    - Batch average loss: L_batch = 1/B ∑(i=1)^B RMSE_i, where B is the number of unique events
    
    Difference from RMSEFloodSampleLoss:
    - RMSEFloodSampleLoss: Each sample is computed independently (assumes each sample = one complete event),
      then average over all samples
    - RMSEFloodEventLoss: Samples of the same event are merged first, compute RMSE once per event,
      then average over all events. Suitable when one flood event is split into multiple samples.
    
    Design objectives:
    - Compute loss only during flood periods (specified by flood_mask)
    - Merge samples with the same event_id, then compute RMSE for that event
    - Average RMSE over all events
    """

    def __init__(self, reduction: str = "mean"):
        """
        Initialize RMSEFloodEventLoss.

        Parameters
        ----------
        reduction : str, default="mean"
            Reduction method: "none", "mean", or "sum"
        """
        super(RMSEFloodEventLoss, self).__init__()
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def _compute_event_rmse(
        self, 
        pred_flood: torch.Tensor, 
        targ_flood: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RMSE for a single event.
        
        Parameters
        ----------
        pred_flood : torch.Tensor
            Merged predictions for this event (flood period)
        targ_flood : torch.Tensor
            Merged targets for this event (flood period)
            
        Returns
        -------
        torch.Tensor
            RMSE value for this event
        """
        # Handle NaN values
        valid_mask = ~torch.isnan(pred_flood) & ~torch.isnan(targ_flood)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_flood.device, requires_grad=True)
        
        pred_flood = pred_flood[valid_mask]
        targ_flood = targ_flood[valid_mask]
        
        if len(pred_flood) == 0:
            return torch.tensor(0.0, device=pred_flood.device, requires_grad=True)

        # Compute overall RMSE during flood period
        sq_error = (pred_flood - targ_flood) ** 2
        event_rmse = torch.sqrt(torch.mean(sq_error))

        # Ensure scalar tensor
        if event_rmse.dim() > 0:
            event_rmse = event_rmse.squeeze()
        
        return event_rmse

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        flood_mask: torch.Tensor,
        event_ids: list[Union[str, int]],
    ) -> torch.Tensor:
        """
        Compute RMSE loss for flood events (computed per event).

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions, shape: [batch_size, seq_len, output_features] or [batch_size, seq_len]
        targets : torch.Tensor
            Target values, shape: [batch_size, seq_len, output_features] or [batch_size, seq_len]
        flood_mask : torch.Tensor
            Flood mask, shape: [batch_size, seq_len, 1] or [batch_size, seq_len], 1 indicates flood period
        event_ids : list[Union[str, int]]
            Event ID list, length equals batch_size, used to group samples

        Returns
        -------
        torch.Tensor
            Loss value(s) according to reduction method
        """
        # Handle dimensions: if 3D, take first feature (usually flow)
        if predictions.dim() == 3:
            predictions_2d = predictions[:, :, 0]  # [batch_size, seq_len]
            targets_2d = targets[:, :, 0]  # [batch_size, seq_len]
            if flood_mask.dim() == 3:
                flood_mask_2d = flood_mask[:, :, 0]  # [batch_size, seq_len]
            else:
                flood_mask_2d = flood_mask
        else:
            predictions_2d = predictions
            targets_2d = targets
            flood_mask_2d = flood_mask

        batch_size = predictions_2d.shape[0]
        
        if len(event_ids) != batch_size:
            raise ValueError(
                f"event_ids length ({len(event_ids)}) must match batch_size ({batch_size})"
            )

        # Group samples by event
        event_groups = {}
        for i, event_id in enumerate(event_ids):
            event_key = str(event_id)
            if event_key not in event_groups:
                event_groups[event_key] = []
            event_groups[event_key].append(i)

        # Compute RMSE for each event
        event_rmses: list[torch.Tensor] = []
        
        for event_key, sample_indices in event_groups.items():
            # Merge flood data from all samples of this event
            all_pred_flood = []
            all_targ_flood = []
            
            for idx in sample_indices:
                sample_predictions = predictions_2d[idx]  # [seq_len]
                sample_targets = targets_2d[idx]  # [seq_len]
                sample_flood_mask = flood_mask_2d[idx]  # [seq_len]
                
                # Extract flood period data for this sample
                boolean_mask = sample_flood_mask.to(torch.bool)
                if boolean_mask.sum() == 0:
                    continue  # Skip samples with no flood
                
                pred_flood = sample_predictions[boolean_mask]
                targ_flood = sample_targets[boolean_mask]
                
                all_pred_flood.append(pred_flood)
                all_targ_flood.append(targ_flood)
            
            # Skip if no valid flood data for this event
            if len(all_pred_flood) == 0:
                continue
            
            # Merge flood data from all samples of this event
            merged_pred = torch.cat(all_pred_flood)
            merged_targ = torch.cat(all_targ_flood)
            
            # Compute RMSE for this event
            event_rmse = self._compute_event_rmse(merged_pred, merged_targ)
            event_rmses.append(event_rmse)

        # Average RMSE over all events
        if len(event_rmses) == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        loss = torch.stack(event_rmses)

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            raise ValueError(
                f"Unsupported reduction method: {self.reduction}. "
                f"Use 'mean', 'sum' or 'none'."
            )


class PeakFocusedFloodSampleLoss(torch.nn.Module):
    """
    Peak area flood loss with peak value, high-value region, and RMSE (computed per sample).
    
    Design objectives:
    - Compute loss only during flood periods (specified by flood_mask)
    - Consider three components:
      1) Overall RMSE during flood periods (overall fitting)
      2) Peak value error (error of maximum flow)
      3) High-value region RMSE (>= target_peak * high_ratio) to strengthen
         fitting for high-flow segments near the peak
    - Does not consider peak timing (no time index constraints)
    """

    def __init__(
        self,
        peak_weight: float = 2.0,
        high_weight: float = 1.0,
        overall_weight: float = 0.5,
        high_ratio: float = 0.8,
        reduction: str = "mean",
    ):
        """
        Initialize PeakFocusedFloodSampleLoss.

        Parameters
        ----------
        peak_weight : float, default=2.0
            Weight for peak value error
        high_weight : float, default=1.0
            Weight for high-value region RMSE
        overall_weight : float, default=0.5
            Weight for overall RMSE during flood periods
        high_ratio : float, default=0.8
            High-value region threshold ratio (relative to target peak),
            time steps with values >= target_peak * high_ratio are considered high-value region
        reduction : str, default="mean"
            Reduction method: "none", "mean", or "sum"
        """
        super(PeakFocusedFloodSampleLoss, self).__init__()
        self.peak_weight = peak_weight
        self.high_weight = high_weight
        self.overall_weight = overall_weight
        self.high_ratio = high_ratio
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        flood_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute flood loss (only during flood periods, per sample).

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions, shape: [batch_size, seq_len, output_features] or [batch_size, seq_len]
        targets : torch.Tensor
            Target values, shape: [batch_size, seq_len, output_features] or [batch_size, seq_len]
        flood_mask : torch.Tensor
            Flood mask, shape: [batch_size, seq_len, 1] or [batch_size, seq_len], 1 indicates flood period

        Returns
        -------
        torch.Tensor
            Loss value(s) according to reduction method
        """
        # Handle dimensions: if 3D, take first feature (usually flow)
        if predictions.dim() == 3:
            predictions_2d = predictions[:, :, 0]  # [batch_size, seq_len]
            targets_2d = targets[:, :, 0]  # [batch_size, seq_len]
            if flood_mask.dim() == 3:
                flood_mask_2d = flood_mask[:, :, 0]  # [batch_size, seq_len]
            else:
                flood_mask_2d = flood_mask
        else:
            predictions_2d = predictions
            targets_2d = targets
            flood_mask_2d = flood_mask

        batch_size = predictions_2d.shape[0]
        sample_losses: list[torch.Tensor] = []

        for i in range(batch_size):
            sample_predictions = predictions_2d[i]  # [seq_len]
            sample_targets = targets_2d[i]  # [seq_len]
            sample_flood_mask = flood_mask_2d[i]  # [seq_len]

            # Extract flood period data for this sample
            boolean_mask = sample_flood_mask.to(torch.bool)
            if boolean_mask.sum() == 0:
                # No flood, loss is 0
                sample_losses.append(
                    torch.tensor(0.0, device=predictions.device, requires_grad=True)
                )
                continue

            pred_flood = sample_predictions[boolean_mask]
            targ_flood = sample_targets[boolean_mask]

            # Handle NaN values
            valid_mask = ~torch.isnan(pred_flood) & ~torch.isnan(targ_flood)
            if valid_mask.sum() == 0:
                sample_losses.append(
                    torch.tensor(0.0, device=predictions.device, requires_grad=True)
                )
                continue

            pred_flood = pred_flood[valid_mask]
            targ_flood = targ_flood[valid_mask]

            # Overall RMSE during flood period
            sq_error = (pred_flood - targ_flood) ** 2
            overall_rmse = torch.sqrt(torch.mean(sq_error))

            # Peak value error (using maximum value in flood period)
            target_peak = torch.max(targ_flood)
            pred_peak = torch.max(pred_flood)
            peak_error = torch.abs(pred_peak - target_peak)

            # High-value region RMSE (>= target_peak * high_ratio)
            # If target_peak <= 0, high-value region degenerates to entire flood period
            if target_peak <= 0:
                high_mask = torch.ones_like(targ_flood, dtype=torch.bool)
            else:
                threshold = target_peak * self.high_ratio
                high_mask = targ_flood >= threshold

            if high_mask.sum() > 0:
                pred_high = pred_flood[high_mask]
                targ_high = targ_flood[high_mask]
                high_sq_error = (pred_high - targ_high) ** 2
                high_rmse = torch.sqrt(torch.mean(high_sq_error))
            else:
                high_rmse = torch.tensor(
                    0.0, device=predictions.device, requires_grad=True
                )

            # Combine loss components for this sample
            sample_loss = (
                self.overall_weight * overall_rmse
                + self.peak_weight * peak_error
                + self.high_weight * high_rmse
            )

            # Ensure scalar tensor
            if sample_loss.dim() > 0:
                sample_loss = sample_loss.squeeze()
            sample_losses.append(sample_loss)

        if len(sample_losses) == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        loss = torch.stack(sample_losses)

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            raise ValueError(
                f"Unsupported reduction method: {self.reduction}. "
                f"Use 'mean', 'sum' or 'none'."
            )


class PeakFocusedFloodEventLoss(torch.nn.Module):
    """
    Peak area flood loss with peak value, high-value region, and RMSE (computed per event).
    
    Implementation (following paper formula):
    - For each flood event i in the batch:
      1) Merge flood period data from all samples of this event (identified by event_id)
      2) Compute loss L_i^PF on the merged data for this event (formulas 1-4)
    - Batch average loss: L_batch = 1/B ∑(i=1)^B L_i^PF (formula 5), where B is the number of unique events
    
    Difference from PeakFocusedFloodSampleLoss:
    - PeakFocusedFloodSampleLoss: Each sample is computed independently (assumes each sample = one complete event),
      then average over all samples
    - PeakFocusedFloodEventLoss: Samples of the same event are merged first, compute loss once per event,
      then average over all events. Suitable when one flood event is split into multiple samples.
    
    Design objectives:
    - Compute loss only during flood periods (specified by flood_mask)
    - Merge samples with the same event_id, then compute loss for that event
    - Consider three components:
      1) Overall RMSE during flood periods (overall fitting)
      2) Peak value error (error of maximum flow)
      3) High-value region RMSE (>= target_peak * high_ratio) to strengthen
         fitting for high-flow segments near the peak
    - Does not consider peak timing (no time index constraints)
    """

    def __init__(
        self,
        peak_weight: float = 2.0,
        high_weight: float = 1.0,
        overall_weight: float = 0.5,
        high_ratio: float = 0.8,
        reduction: str = "mean",
    ):
        """
        Initialize PeakFocusedFloodEventLoss.

        Parameters
        ----------
        peak_weight : float, default=2.0
            Weight for peak value error
        high_weight : float, default=1.0
            Weight for high-value region RMSE
        overall_weight : float, default=0.5
            Weight for overall RMSE during flood periods
        high_ratio : float, default=0.8
            High-value region threshold ratio (relative to target peak),
            time steps with values >= target_peak * high_ratio are considered high-value region
        reduction : str, default="mean"
            Reduction method: "none", "mean", or "sum"
        """
        super(PeakFocusedFloodEventLoss, self).__init__()
        self.peak_weight = peak_weight
        self.high_weight = high_weight
        self.overall_weight = overall_weight
        self.high_ratio = high_ratio
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def _compute_event_loss(
        self, 
        pred_flood: torch.Tensor, 
        targ_flood: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for a single event.
        
        Parameters
        ----------
        pred_flood : torch.Tensor
            Merged predictions for this event (flood period)
        targ_flood : torch.Tensor
            Merged targets for this event (flood period)
            
        Returns
        -------
        torch.Tensor
            Loss value for this event
        """
        # Handle NaN values
        valid_mask = ~torch.isnan(pred_flood) & ~torch.isnan(targ_flood)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_flood.device, requires_grad=True)
        
        pred_flood = pred_flood[valid_mask]
        targ_flood = targ_flood[valid_mask]
        
        if len(pred_flood) == 0:
            return torch.tensor(0.0, device=pred_flood.device, requires_grad=True)

        # Overall RMSE during flood period
        sq_error = (pred_flood - targ_flood) ** 2
        overall_rmse = torch.sqrt(torch.mean(sq_error))

        # Peak value error (using maximum value in flood period)
        target_peak = torch.max(targ_flood)
        pred_peak = torch.max(pred_flood)
        peak_error = torch.abs(pred_peak - target_peak)

        # High-value region RMSE (>= target_peak * high_ratio)
        if target_peak <= 0:
            high_mask = torch.ones_like(targ_flood, dtype=torch.bool)
        else:
            threshold = target_peak * self.high_ratio
            high_mask = targ_flood >= threshold

        if high_mask.sum() > 0:
            pred_high = pred_flood[high_mask]
            targ_high = targ_flood[high_mask]
            high_sq_error = (pred_high - targ_high) ** 2
            high_rmse = torch.sqrt(torch.mean(high_sq_error))
        else:
            high_rmse = torch.tensor(
                0.0, device=pred_flood.device, requires_grad=True
            )

        # Combine loss components for this event
        event_loss = (
            self.overall_weight * overall_rmse
            + self.peak_weight * peak_error
            + self.high_weight * high_rmse
        )

        # Ensure scalar tensor
        if event_loss.dim() > 0:
            event_loss = event_loss.squeeze()
        
        return event_loss

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        flood_mask: torch.Tensor,
        event_ids: list[Union[str, int]],
    ) -> torch.Tensor:
        """
        Compute flood loss (computed per event).

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions, shape: [batch_size, seq_len, output_features] or [batch_size, seq_len]
        targets : torch.Tensor
            Target values, shape: [batch_size, seq_len, output_features] or [batch_size, seq_len]
        flood_mask : torch.Tensor
            Flood mask, shape: [batch_size, seq_len, 1] or [batch_size, seq_len], 1 indicates flood period
        event_ids : list[Union[str, int]]
            Event ID list, length equals batch_size, used to group samples

        Returns
        -------
        torch.Tensor
            Loss value(s) according to reduction method
        """
        # Handle dimensions: if 3D, take first feature (usually flow)
        if predictions.dim() == 3:
            predictions_2d = predictions[:, :, 0]  # [batch_size, seq_len]
            targets_2d = targets[:, :, 0]  # [batch_size, seq_len]
            if flood_mask.dim() == 3:
                flood_mask_2d = flood_mask[:, :, 0]  # [batch_size, seq_len]
            else:
                flood_mask_2d = flood_mask
        else:
            predictions_2d = predictions
            targets_2d = targets
            flood_mask_2d = flood_mask

        batch_size = predictions_2d.shape[0]
        
        if len(event_ids) != batch_size:
            raise ValueError(
                f"event_ids length ({len(event_ids)}) must match batch_size ({batch_size})"
            )

        # Group samples by event
        event_groups = {}
        for i, event_id in enumerate(event_ids):
            event_key = str(event_id)
            if event_key not in event_groups:
                event_groups[event_key] = []
            event_groups[event_key].append(i)

        # Compute loss for each event
        event_losses: list[torch.Tensor] = []
        
        for event_key, sample_indices in event_groups.items():
            # Merge flood data from all samples of this event
            all_pred_flood = []
            all_targ_flood = []
            
            for idx in sample_indices:
                sample_predictions = predictions_2d[idx]  # [seq_len]
                sample_targets = targets_2d[idx]  # [seq_len]
                sample_flood_mask = flood_mask_2d[idx]  # [seq_len]
                
                # Extract flood period data for this sample
                boolean_mask = sample_flood_mask.to(torch.bool)
                if boolean_mask.sum() == 0:
                    continue  # Skip samples with no flood
                
                pred_flood = sample_predictions[boolean_mask]
                targ_flood = sample_targets[boolean_mask]
                
                all_pred_flood.append(pred_flood)
                all_targ_flood.append(targ_flood)
            
            # Skip if no valid flood data for this event
            if len(all_pred_flood) == 0:
                continue
            
            # Merge flood data from all samples of this event
            merged_pred = torch.cat(all_pred_flood)
            merged_targ = torch.cat(all_targ_flood)
            
            # Compute loss for this event
            event_loss = self._compute_event_loss(merged_pred, merged_targ)
            event_losses.append(event_loss)

        # Average loss over all events
        if len(event_losses) == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        loss = torch.stack(event_losses)

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            raise ValueError(
                f"Unsupported reduction method: {self.reduction}. "
                f"Use 'mean', 'sum' or 'none'."
            )
