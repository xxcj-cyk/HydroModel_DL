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
        # Convert to tensor if they are scalars, but keep as-is if already tensor
        # This allows flexibility for both scalar and tensor inputs
        self.mu = mu
        self.sigma = sigma

    def forward(self, x: torch.Tensor):
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
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mask = ~torch.isnan(output) & ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        output_masked = output[mask]
        target_masked = target[mask]
        # Directly compute MSE since data is already masked
        mse_value = (output_masked - target_masked) ** 2
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
        self.mae_weight = mae_weight
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mask = ~torch.isnan(output) & ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        output_masked = output[mask]
        target_masked = target[mask]
        # Directly compute PES and MAE since data is already masked
        # PES: MSE × sigmoid(MSE)
        mse_value = (output_masked - target_masked) ** 2
        sigmoid_mse = torch.sigmoid(mse_value)
        pes = mse_value * sigmoid_mse
        # MAE
        mae = torch.abs(output_masked - target_masked)
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
    def __init__(self, reduction="mean"):
        """
        PES Flood Event Loss: PES loss with flood event filtering

        Applies PES loss (MSE × sigmoid(MSE)) with flood event filtering.
        This class filters flood events first then calculates PES loss,
        focusing computation only on flood periods.

        The difference from standard PES loss is that this class filters flood events first,
        because PES does sigmoid on MSE, when we want to focus only on flood events,
        we need to filter them out first before applying the PES calculation.

        Parameters
        ----------
        reduction : str
            Reduction method for PES loss, default is "mean"
        """
        super(PESFloodEvent, self).__init__()
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

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

        base_loss_func = PESLoss(reduction=self.reduction)
        return base_loss_func(predictions, targets)


class HybridFloodLoss(torch.nn.Module):
    def __init__(self, mae_weight=0.5, reduction="mean"):
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
        reduction : str
            Reduction method for loss, default is "mean"
        """
        super(HybridFloodLoss, self).__init__()
        self.mae_weight = mae_weight
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

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

        base_loss_func = HybridLoss(self.mae_weight, reduction=self.reduction)
        return base_loss_func(predictions, targets)


class PeakFocusedLoss(torch.nn.Module):
    """
    针对长序列的洪峰和峰现时间优化的损失函数
    
    该损失函数用于长序列数据，对所有数据点计算损失，重点关注：
    1. 洪峰误差（Peak Flow Error）
    2. 峰现时间误差（Peak Time Error）
    3. 整体形状拟合（Overall RMSE）
    
    注意：此损失函数计算所有数据点的整体RMSE，适用于长序列数据。
    如果只需要对洪水期间计算损失，请使用 PeakFocusedFloodLoss。
    """
    
    def __init__(
        self,
        peak_weight=2.0,
        time_weight=1.0,
        overall_weight=0.5,
        peak_threshold_ratio=0.7,
        reduction="mean",
    ):
        """
        初始化PeakFocusedLoss（长序列版本）
        
        Parameters
        ----------
        peak_weight : float
            洪峰误差的权重，默认2.0（给予更高权重）
        time_weight : float
            峰现时间误差的权重，默认1.0
        overall_weight : float
            整体RMSE的权重，默认0.5
        peak_threshold_ratio : float
            识别洪峰的阈值比例（相对于序列最大值），默认0.7
            用于识别洪峰区域
        reduction : str
            损失归约方式，默认"mean"
        """
        super(PeakFocusedLoss, self).__init__()
        self.peak_weight = peak_weight
        self.time_weight = time_weight
        self.overall_weight = overall_weight
        self.peak_threshold_ratio = peak_threshold_ratio
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def _find_high_value_regions(self, sequence: torch.Tensor):
        """
        找到序列中的高值区域（>= threshold的区域）
        
        Parameters
        ----------
        sequence : torch.Tensor
            序列数据，shape: [batch_size, seq_len] 或 [seq_len]
            
        Returns
        -------
        torch.Tensor
            高值区域掩码，shape与sequence相同，True表示高值区域
        """
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, seq_len = sequence.shape
        high_value_masks = []
        
        for i in range(batch_size):
            seq = sequence[i]
            
            # 使用阈值方法识别高值区域
            max_val = seq.max()
            if max_val > 0:
                threshold = max_val * self.peak_threshold_ratio
                mask = seq >= threshold
            else:
                mask = torch.zeros(seq_len, dtype=torch.bool, device=seq.device)
            
            high_value_masks.append(mask)
        
        result = torch.stack(high_value_masks)
        
        if squeeze_output:
            result = result.squeeze(0)
            
        return result
    
    def _find_multiple_peaks(self, sequence: torch.Tensor, high_value_mask: torch.Tensor):
        """
        在高值区域中找到多个洪峰（每个连续的高值区域一个洪峰）
        
        Parameters
        ----------
        sequence : torch.Tensor
            序列数据，shape: [batch_size, seq_len] 或 [seq_len]
        high_value_mask : torch.Tensor
            高值区域掩码，shape与sequence相同
            
        Returns
        -------
        tuple
            (peak_values_list, peak_indices_list) 
            peak_values_list: 每个样本的洪峰值列表（可能多个）
            peak_indices_list: 每个样本的峰现时间索引列表（可能多个）
        """
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)
            high_value_mask = high_value_mask.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, seq_len = sequence.shape
        all_peak_values = []
        all_peak_indices = []
        
        for i in range(batch_size):
            seq = sequence[i]
            mask = high_value_mask[i]
            
            if mask.sum() == 0:
                # 没有高值区域
                all_peak_values.append(torch.tensor([], device=seq.device))
                all_peak_indices.append(torch.tensor([], device=seq.device, dtype=torch.long))
                continue
            
            # 找到所有连续的高值区域
            peak_values = []
            peak_indices = []
            
            # 找到所有True的索引
            high_indices = torch.where(mask)[0]
            
            if len(high_indices) == 0:
                all_peak_values.append(torch.tensor([], device=seq.device))
                all_peak_indices.append(torch.tensor([], device=seq.device, dtype=torch.long))
                continue
            
            # 找到连续区域的边界
            # 计算相邻索引的差值
            if len(high_indices) > 1:
                diff = high_indices[1:] - high_indices[:-1]
                # 差值大于1的地方就是区域边界
                boundaries = torch.where(diff > 1)[0] + 1
                
                # 分割成连续区域
                start_idx = 0
                for end_idx in boundaries:
                    region_indices = high_indices[start_idx:end_idx]
                    region_seq = seq[region_indices]
                    # 在这个区域内找最大值
                    peak_idx_in_region = torch.argmax(region_seq)
                    peak_idx_global = region_indices[peak_idx_in_region]
                    peak_values.append(region_seq[peak_idx_in_region])
                    peak_indices.append(peak_idx_global)
                    start_idx = end_idx
                
                # 最后一个区域
                region_indices = high_indices[start_idx:]
                region_seq = seq[region_indices]
                peak_idx_in_region = torch.argmax(region_seq)
                peak_idx_global = region_indices[peak_idx_in_region]
                peak_values.append(region_seq[peak_idx_in_region])
                peak_indices.append(peak_idx_global)
            else:
                # 只有一个高值点
                peak_values.append(seq[high_indices[0]])
                peak_indices.append(high_indices[0])
            
            if len(peak_values) > 0:
                all_peak_values.append(torch.stack(peak_values))
                all_peak_indices.append(torch.stack(peak_indices))
            else:
                all_peak_values.append(torch.tensor([], device=seq.device))
                all_peak_indices.append(torch.tensor([], device=seq.device, dtype=torch.long))
        
        if squeeze_output:
            # 单个样本，返回第一个元素（tensor）
            if len(all_peak_values) > 0:
                all_peak_values = all_peak_values[0]
                all_peak_indices = all_peak_indices[0]
            else:
                all_peak_values = torch.tensor([], device=sequence.device)
                all_peak_indices = torch.tensor([], device=sequence.device, dtype=torch.long)
            
        return all_peak_values, all_peak_indices

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算针对洪峰和峰现时间的损失（长序列版本，对所有数据点计算）
        
        Parameters
        ----------
        output : torch.Tensor
            模型预测值，shape: [batch_size, seq_len, features] 或 [batch_size, seq_len]
        target : torch.Tensor
            真实值，shape: [batch_size, seq_len, features] 或 [batch_size, seq_len]
            
        Returns
        -------
        torch.Tensor
            计算得到的损失值
        """
        # 处理维度：如果是3D，取第一个特征（通常是流量）
        if output.dim() == 3:
            output_2d = output[:, :, 0]  # [batch_size, seq_len]
            target_2d = target[:, :, 0]  # [batch_size, seq_len]
        else:
            output_2d = output
            target_2d = target
        
        # 处理NaN值
        mask = ~torch.isnan(output_2d) & ~torch.isnan(target_2d)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        
        # 应用NaN掩码：将NaN值替换为0（在计算洪峰时会忽略）
        output_2d = torch.where(mask, output_2d, torch.zeros_like(output_2d))
        target_2d = torch.where(mask, target_2d, torch.zeros_like(target_2d))
        
        batch_size = output_2d.shape[0]
        
        # 计算每个样本的损失
        sample_losses = []
        for i in range(batch_size):
            sample_output = output_2d[i]  # [seq_len]
            sample_target = target_2d[i]   # [seq_len]
            
            # 1. 计算该样本的整体RMSE损失
            sq_error = (sample_output - sample_target) ** 2
            overall_rmse = torch.sqrt(torch.mean(sq_error))
            
            # 2. 识别高值区域（基于真实值）
            true_high_mask = self._find_high_value_regions(sample_target.unsqueeze(0)).squeeze(0)
            
            # 3. 计算高值区域的加权RMSE（对高值区域给予更高权重）
            if true_high_mask.sum() > 0:
                # 高值区域的RMSE
                high_value_sq_error = sq_error[true_high_mask]
                high_value_rmse = torch.sqrt(torch.mean(high_value_sq_error))
                
                # 非高值区域的RMSE
                low_value_mask = ~true_high_mask
                if low_value_mask.sum() > 0:
                    low_value_sq_error = sq_error[low_value_mask]
                    low_value_rmse = torch.sqrt(torch.mean(low_value_sq_error))
                else:
                    low_value_rmse = torch.tensor(0.0, device=sample_output.device)
                
                # 加权组合：高值区域权重更高
                weighted_rmse = 0.7 * high_value_rmse + 0.3 * low_value_rmse
            else:
                # 没有高值区域，使用整体RMSE
                weighted_rmse = overall_rmse
            
            # 4. 计算多个洪峰的误差
            pred_high_mask = self._find_high_value_regions(sample_output.unsqueeze(0))
            pred_peaks_list, pred_peak_indices_list = self._find_multiple_peaks(
                sample_output.unsqueeze(0), 
                pred_high_mask
            )
            true_peaks_list, true_peak_indices_list = self._find_multiple_peaks(
                sample_target.unsqueeze(0),
                true_high_mask.unsqueeze(0)
            )
            
            # 提取当前样本的洪峰（_find_multiple_peaks对单个样本返回tensor）
            pred_peaks = pred_peaks_list
            pred_indices = pred_peak_indices_list.float() if isinstance(pred_peak_indices_list, torch.Tensor) else torch.tensor([], device=sample_output.device, dtype=torch.float32)
            true_peaks = true_peaks_list
            true_indices = true_peak_indices_list.float() if isinstance(true_peak_indices_list, torch.Tensor) else torch.tensor([], device=sample_output.device, dtype=torch.float32)
            
            # 计算洪峰误差（匹配最接近的洪峰对）
            peak_error = torch.tensor(0.0, device=sample_output.device)
            time_error = torch.tensor(0.0, device=sample_output.device)
            
            # 检查是否有洪峰数据
            if len(pred_peaks) > 0 and len(true_peaks) > 0:
                # 如果数量不同，取较小的数量进行匹配
                min_count = min(len(pred_peaks), len(true_peaks))
                if min_count > 0:
                    # 简单匹配：按大小排序后对应
                    pred_sorted, pred_sort_idx = torch.sort(pred_peaks, descending=True)
                    true_sorted, true_sort_idx = torch.sort(true_peaks, descending=True)
                    
                    # 计算前min_count个洪峰的误差
                    peak_errors = torch.abs(pred_sorted[:min_count] - true_sorted[:min_count])
                    peak_error = torch.mean(peak_errors)
                    
                    # 计算峰现时间误差
                    pred_time_sorted = pred_indices[pred_sort_idx[:min_count]]
                    true_time_sorted = true_indices[true_sort_idx[:min_count]]
                    time_errors = torch.abs(pred_time_sorted - true_time_sorted)
                    time_error = torch.mean(time_errors)
            
            # 5. 组合该样本的损失
            sample_loss = (
                self.overall_weight * overall_rmse
                + self.overall_weight * weighted_rmse  # 高值区域加权RMSE
                + self.peak_weight * peak_error
                + self.time_weight * time_error
            )
            sample_losses.append(sample_loss)
        
        # 堆叠所有样本的损失
        loss = torch.stack(sample_losses).squeeze(-1)  # [batch_size]
        
        # 根据reduction返回
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


class PeakFocusedFloodLoss(torch.nn.Module):
    """
    针对场次洪水的洪峰和峰现时间优化的损失函数
    
    该损失函数专门设计用于场次洪水预报，只对洪水期间的数据点计算损失，重点关注：
    1. 洪峰误差（Peak Flow Error）
    2. 峰现时间误差（Peak Time Error）
    3. 洪水期间的整体形状拟合（Overall RMSE）
    
    注意：此损失函数先过滤出洪水期间的数据，然后只对这些数据计算损失。
    适用于场次洪水数据，与RMSEFloodLoss、HybridFloodLoss的设计模式一致。
    """
    
    def __init__(
        self,
        peak_weight=2.0,
        time_weight=1.0,
        overall_weight=0.5,
        reduction="mean",
    ):
        """
        初始化PeakFocusedFloodLoss（场次洪水版本）
        
        Parameters
        ----------
        peak_weight : float
            洪峰误差的权重，默认2.0（给予更高权重）
        time_weight : float
            峰现时间误差的权重，默认1.0
        overall_weight : float
            洪水期间整体RMSE的权重，默认0.5
        reduction : str
            损失归约方式，默认"mean"
        """
        super(PeakFocusedFloodLoss, self).__init__()
        self.peak_weight = peak_weight
        self.time_weight = time_weight
        self.overall_weight = overall_weight
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def _find_peaks_in_flood(self, sequence: torch.Tensor, flood_mask: torch.Tensor):
        """
        在洪水期间找到洪峰值和峰现时间
        
        Parameters
        ----------
        sequence : torch.Tensor
            序列数据，shape: [batch_size, seq_len] 或 [seq_len]
        flood_mask : torch.Tensor
            洪水掩码，shape: [batch_size, seq_len] 或 [seq_len]
            
        Returns
        -------
        tuple
            (peak_values, peak_indices) 洪峰值和峰现时间索引
        """
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)
            flood_mask = flood_mask.unsqueeze(0) if flood_mask.dim() == 1 else flood_mask
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, seq_len = sequence.shape
        peak_values = []
        peak_indices = []
        
        for i in range(batch_size):
            seq = sequence[i]
            mask = flood_mask[i].bool()
            
            if mask.sum() == 0:
                # 没有洪水，返回0
                peak_values.append(torch.tensor(0.0, device=seq.device))
                peak_indices.append(torch.tensor(0, device=seq.device, dtype=torch.long))
            else:
                # 只在洪水期间查找洪峰
                flood_seq = seq[mask]
                flood_indices = torch.where(mask)[0]
                
                if len(flood_seq) == 0:
                    peak_values.append(torch.tensor(0.0, device=seq.device))
                    peak_indices.append(torch.tensor(0, device=seq.device, dtype=torch.long))
                else:
                    # 找到洪峰值和对应的索引
                    peak_idx_in_flood = torch.argmax(flood_seq)
                    peak_idx_global = flood_indices[peak_idx_in_flood]
                    peak_values.append(flood_seq[peak_idx_in_flood])
                    peak_indices.append(peak_idx_global)
        
        # 确保所有tensor都是标量（0维），然后堆叠
        # 这样无论squeeze_output如何，返回的维度都一致
        peak_values_tensors = []
        peak_indices_tensors = []
        for pv, pi in zip(peak_values, peak_indices):
            # 确保是标量tensor
            if isinstance(pv, torch.Tensor):
                if pv.dim() > 0:
                    pv = pv.flatten()[0] if pv.numel() > 0 else torch.tensor(0.0, device=pv.device)
                peak_values_tensors.append(pv)
            else:
                peak_values_tensors.append(torch.tensor(float(pv), device=peak_values[0].device if len(peak_values) > 0 else torch.device('cpu')))
            
            if isinstance(pi, torch.Tensor):
                if pi.dim() > 0:
                    pi = pi.flatten()[0] if pi.numel() > 0 else torch.tensor(0, device=pi.device, dtype=torch.long)
                peak_indices_tensors.append(pi)
            else:
                peak_indices_tensors.append(torch.tensor(int(pi), device=peak_indices[0].device if len(peak_indices) > 0 else torch.device('cpu'), dtype=torch.long))
        
        peak_values = torch.stack(peak_values_tensors)
        peak_indices = torch.stack(peak_indices_tensors)
        
        if squeeze_output:
            # 如果是单个样本，返回标量（0维）
            peak_values = peak_values.squeeze(0)
            peak_indices = peak_indices.squeeze(0)
            
        return peak_values, peak_indices

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        flood_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算针对洪峰和峰现时间的损失（场次洪水版本，只对洪水期间计算）
        
        Parameters
        ----------
        predictions : torch.Tensor
            模型预测值，shape: [batch_size, seq_len, output_features] 或 [batch_size, seq_len]
        targets : torch.Tensor
            真实值，shape: [batch_size, seq_len, output_features] 或 [batch_size, seq_len]
        flood_mask : torch.Tensor
            洪水掩码，shape: [batch_size, seq_len, 1] 或 [batch_size, seq_len]
            (1 for flood, 0 for normal)
            
        Returns
        -------
        torch.Tensor
            计算得到的损失值
        """
        # 处理维度：如果是3D，取第一个特征（通常是流量）
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
        
        # 计算每个样本的损失
        sample_losses = []
        for i in range(batch_size):
            sample_predictions = predictions_2d[i:i+1]  # 保持维度
            sample_targets = targets_2d[i:i+1]
            sample_flood_mask = flood_mask_2d[i:i+1]
            
            # 先过滤出该样本洪水期间的数据
            boolean_mask = sample_flood_mask.to(torch.bool)
            if boolean_mask.sum() == 0:
                # 没有洪水，损失为0
                sample_losses.append(torch.tensor(0.0, device=predictions.device, requires_grad=True))
                continue
            
            predictions_flood = sample_predictions[boolean_mask]
            targets_flood = sample_targets[boolean_mask]
            
            # 处理NaN值
            mask = ~torch.isnan(predictions_flood) & ~torch.isnan(targets_flood)
            if mask.sum() == 0:
                sample_losses.append(torch.tensor(0.0, device=predictions.device, requires_grad=True))
                continue
            
            predictions_flood = predictions_flood[mask]
            targets_flood = targets_flood[mask]
            
            # 1. 计算该样本洪水期间的整体RMSE损失
            sq_error = (predictions_flood - targets_flood) ** 2
            overall_rmse = torch.sqrt(torch.mean(sq_error))
            
            # 2. 计算该样本的洪峰误差（在洪水期间查找洪峰）
            pred_peaks, pred_peak_indices = self._find_peaks_in_flood(sample_predictions, sample_flood_mask)
            true_peaks, true_peak_indices = self._find_peaks_in_flood(sample_targets, sample_flood_mask)
            
            # 统一处理：提取标量值（_find_peaks_in_flood返回的可能是[1]或标量）
            # 由于sample_predictions是[1, seq_len]，返回的应该是[1]，需要提取第一个元素
            if isinstance(pred_peaks, torch.Tensor):
                pred_peak_val = pred_peaks[0].item() if pred_peaks.numel() > 0 and pred_peaks.dim() > 0 else pred_peaks.item()
            else:
                pred_peak_val = float(pred_peaks)
            
            if isinstance(true_peaks, torch.Tensor):
                true_peak_val = true_peaks[0].item() if true_peaks.numel() > 0 and true_peaks.dim() > 0 else true_peaks.item()
            else:
                true_peak_val = float(true_peaks)
            
            if isinstance(pred_peak_indices, torch.Tensor):
                pred_time_val = float(pred_peak_indices[0].item() if pred_peak_indices.numel() > 0 and pred_peak_indices.dim() > 0 else pred_peak_indices.item())
            else:
                pred_time_val = float(pred_peak_indices)
            
            if isinstance(true_peak_indices, torch.Tensor):
                true_time_val = float(true_peak_indices[0].item() if true_peak_indices.numel() > 0 and true_peak_indices.dim() > 0 else true_peak_indices.item())
            else:
                true_time_val = float(true_peak_indices)
            
            # 转换为标量tensor（0维）
            pred_peaks_t = torch.tensor(pred_peak_val, device=sample_predictions.device, dtype=torch.float32)
            true_peaks_t = torch.tensor(true_peak_val, device=sample_predictions.device, dtype=torch.float32)
            pred_time_t = torch.tensor(pred_time_val, device=sample_predictions.device, dtype=torch.float32)
            true_time_t = torch.tensor(true_time_val, device=sample_predictions.device, dtype=torch.float32)
            
            # 洪峰值误差
            peak_error = torch.abs(pred_peaks_t - true_peaks_t)
            
            # 3. 计算该样本的峰现时间误差
            time_error = torch.abs(pred_time_t - true_time_t)
            
            # 4. 组合该样本的损失（所有项都是0维标量）
            sample_loss = (
                self.overall_weight * overall_rmse
                + self.peak_weight * peak_error
                + self.time_weight * time_error
            )
            # 确保是0维标量tensor
            if sample_loss.dim() > 0:
                sample_loss = sample_loss.squeeze()
            if sample_loss.dim() > 0:
                sample_loss = torch.tensor(sample_loss.item(), device=sample_loss.device, requires_grad=True)
            sample_losses.append(sample_loss)
        
        # 堆叠所有样本的损失（确保所有tensor都是0维标量）
        # 统一处理：确保所有tensor都是0维
        processed_losses = []
        for loss in sample_losses:
            if loss.dim() == 0:
                processed_losses.append(loss)
            else:
                # 如果是多维，转换为0维
                loss_flat = loss.flatten()
                if loss_flat.numel() > 0:
                    processed_losses.append(torch.tensor(loss_flat[0].item(), device=loss.device, requires_grad=True))
                else:
                    processed_losses.append(torch.tensor(0.0, device=loss.device, requires_grad=True))
        
        loss = torch.stack(processed_losses)  # [batch_size]
        
        # 根据reduction返回
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


class PeakAreaFloodLoss(torch.nn.Module):
    """
    PeakAreaFloodLoss：针对场次洪水的洪峰极值 + 高值区域 + RMSE 的损失函数

    设计目标：
    - 只在洪水期间（由 flood_mask 指定）计算损失；
    - 同时考虑：
      1) 洪水期间整体 RMSE（整体拟合）；
      2) 洪峰极值误差（最大流量的误差）；
      3) 高值区域（>= 目标峰值 * high_ratio）的 RMSE，强化对洪峰附近高流量段的拟合。
    - 不考虑峰现时间（不对时间索引加约束）。
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
        初始化 PeakAreaFloodLoss

        Parameters
        ----------
        peak_weight : float
            洪峰极值误差的权重，默认 2.0
        high_weight : float
            高值区域 RMSE 的权重，默认 1.0
        overall_weight : float
            洪水期间整体 RMSE 的权重，默认 0.5
        high_ratio : float
            高值区域阈值比例（相对于真实峰值），默认 0.8，
            表示将所有 >= target_peak * high_ratio 的时段视为“高值区域”
        reduction : str
            损失归约方式，"none" | "mean" | "sum"，默认 "mean"
        """
        super(PeakAreaFloodLoss, self).__init__()
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
        计算针对场次洪水的损失（只在洪水期间计算）

        Parameters
        ----------
        predictions : torch.Tensor
            模型预测值，shape: [batch_size, seq_len, output_features] 或 [batch_size, seq_len]
        targets : torch.Tensor
            真实值，shape: [batch_size, seq_len, output_features] 或 [batch_size, seq_len]
        flood_mask : torch.Tensor
            洪水掩码，shape: [batch_size, seq_len, 1] 或 [batch_size, seq_len]，1 表示洪水期

        Returns
        -------
        torch.Tensor
            根据 reduction 返回标量或逐样本损失
        """
        # 处理维度：如果是3D，取第一个特征（通常是流量）
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

            # 1. 取出该样本洪水期间的数据
            boolean_mask = sample_flood_mask.to(torch.bool)
            if boolean_mask.sum() == 0:
                # 没有洪水，损失记为 0
                sample_losses.append(
                    torch.tensor(0.0, device=predictions.device, requires_grad=True)
                )
                continue

            pred_flood = sample_predictions[boolean_mask]
            targ_flood = sample_targets[boolean_mask]

            # 2. 处理 NaN
            valid_mask = ~torch.isnan(pred_flood) & ~torch.isnan(targ_flood)
            if valid_mask.sum() == 0:
                sample_losses.append(
                    torch.tensor(0.0, device=predictions.device, requires_grad=True)
                )
                continue

            pred_flood = pred_flood[valid_mask]
            targ_flood = targ_flood[valid_mask]

            # 3. 洪水期间整体 RMSE
            sq_error = (pred_flood - targ_flood) ** 2
            overall_rmse = torch.sqrt(torch.mean(sq_error))

            # 4. 洪峰极值误差（使用真实洪水段的最大值）
            target_peak = torch.max(targ_flood)
            pred_peak = torch.max(pred_flood)
            peak_error = torch.abs(pred_peak - target_peak)

            # 5. 高值区域 RMSE（>= target_peak * high_ratio）
            #    如果 target_peak <= 0，则高值区域退化为整个洪水段
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

            # 6. 组合该样本的损失
            sample_loss = (
                self.overall_weight * overall_rmse
                + self.peak_weight * peak_error
                + self.high_weight * high_rmse
            )

            # 确保是 0 维标量 tensor
            if sample_loss.dim() > 0:
                sample_loss = sample_loss.squeeze()
            sample_losses.append(sample_loss)

        if len(sample_losses) == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        loss = torch.stack(sample_losses)  # [batch_size]

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




































