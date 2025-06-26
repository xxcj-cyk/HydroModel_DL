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
    

def deal_gap_data(output, target, data_gap, device):
    """
    How to handle with gap data

    When there are NaN values in observation, we will perform a "reduce" operation on prediction.
    For example, pred = [0,1,2,3,4], obs=[5, nan, nan, 6, nan]; the "reduce" is sum;
    then, pred_sum = [0+1+2, 3+4], obs_sum=[5,6], loss = loss_func(pred_sum, obs_sum).
    Notice: when "sum", actually final index is not chosen,
    because the whole observation may be [5, nan, nan, 6, nan, nan, 7, nan, nan], 6 means sum of three elements.
    Just as the rho is 5, the final one is not chosen

    Parameters
    ----------
    output
        model output for k-th variable
    target
        target for k-th variable
    data_gap
        data_gap=1: reduce is sum
        data_gap=2: reduce is mean
    device
        where to save the data

    Returns
    -------
    tuple[tensor, tensor]
        output and target after dealing with gap
    """
    # all members in a batch has different NaN-gap, so we need a loop
    seg_p_lst = []
    seg_t_lst = []
    for j in range(target.shape[1]):
        non_nan_idx = torch.nonzero(
            ~torch.isnan(target[:, j]), as_tuple=False
        ).squeeze()
        if len(non_nan_idx) < 1:
            raise ArithmeticError("All NaN elements, please check your data")

        # 使用 cumsum 生成 scatter_index
        is_not_nan = ~torch.isnan(target[:, j])
        cumsum_is_not_nan = torch.cumsum(is_not_nan.to(torch.int), dim=0)
        first_non_nan = non_nan_idx[0]
        scatter_index = torch.full_like(
            target[:, j], fill_value=-1, dtype=torch.long
        )  # 将所有值初始化为 -1
        scatter_index[first_non_nan:] = cumsum_is_not_nan[first_non_nan:] - 1
        scatter_index = scatter_index.to(device=device)

        # 创建掩码，只保留有效的索引
        valid_mask = scatter_index >= 0

        if data_gap == 1:
            seg = torch.zeros(
                len(non_nan_idx), device=device, dtype=output.dtype
            ).scatter_add_(0, scatter_index[valid_mask], output[valid_mask, j])
            # for sum, better exclude final non-nan value as it didn't include all necessary periods
            seg_p_lst.append(seg[:-1])
            seg_t_lst.append(target[non_nan_idx[:-1], j])

        elif data_gap == 2:
            counts = torch.zeros(
                len(non_nan_idx), device=device, dtype=output.dtype
            ).scatter_add_(
                0,
                scatter_index[valid_mask],
                torch.ones_like(output[valid_mask, j], dtype=output.dtype),
            )
            seg = torch.zeros(
                len(non_nan_idx), device=device, dtype=output.dtype
            ).scatter_add_(0, scatter_index[valid_mask], output[valid_mask, j])
            seg = seg / counts.clamp(min=1)
            # for mean, we can include all periods
            seg_p_lst.append(seg)
            seg_t_lst.append(target[non_nan_idx, j])
        else:
            raise NotImplementedError(
                "We have not provided this reduce way now!! Please choose 1 or 2!!"
            )

    p = torch.cat(seg_p_lst)
    t = torch.cat(seg_t_lst)
    return p, t


class MultiOutLoss(torch.nn.Module):
    def __init__(
        self,
        loss_funcs: Union[torch.nn.Module, list],
        data_gap: list = None,
        device: list = None,
        limit_part: list = None,
        item_weight: list = None,
    ):
        """
        Loss function for multiple output

        Parameters
        ----------
        loss_funcs
            The loss functions for each output
        data_gap
            It belongs to the feature dim.
            If 1, then the corresponding value is uniformly-spaced with NaN values filling the gap;
            in addition, the first non-nan value means the aggregated value of the following interval,
            for example, in [5, nan, nan, nan], 5 means all four data's sum, although the next 3 values are nan
            hence the calculation is a little different;
            if 2, the first non-nan value means the average value of the following interval,
            for example, in [5, nan, nan, nan], 5 means all four data's mean value;
            default is [0, 2]
        device
            the number of device: -1 -> "cpu" or "cuda:x" (x is 0, 1 or ...)
        limit_part
            when transfer learning, we may ignore some part;
            the default is None, which means no ignorance;
            other choices are list, such as [0], [0, 1] or [1,2,..];
            0 means the first variable;
            tensor is [seq, time, var] or [time, seq, var]
        item_weight
            use different weight for each item's loss;
            for example, the default values [0.5, 0.5] means 0.5 * loss1 + 0.5 * loss2
        """
        if data_gap is None:
            data_gap = [0, 2]
        if device is None:
            device = [0]
        if item_weight is None:
            item_weight = [0.5, 0.5]
        super(MultiOutLoss, self).__init__()
        self.loss_funcs = loss_funcs
        self.data_gap = data_gap
        self.device = get_the_device(device)
        self.limit_part = limit_part
        self.item_weight = item_weight

    def forward(self, output: Tensor, target: Tensor):
        """
        Calculate the sum of losses for different variables

        When there are NaN values in observation, we will perform a "reduce" operation on prediction.
        For example, pred = [0,1,2,3,4], obs=[5, nan, nan, 6, nan]; the "reduce" is sum;
        then, pred_sum = [0+1+2, 3+4], obs_sum=[5,6], loss = loss_func(pred_sum, obs_sum).
        Notice: when "sum", actually final index is not chosen,
        because the whole observation may be [5, nan, nan, 6, nan, nan, 7, nan, nan], 6 means sum of three elements.
        Just as the rho is 5, the final one is not chosen


        Parameters
        ----------
        output
            the prediction tensor; 3-dims are time sequence, batch and feature, respectively
        target
            the observation tensor

        Returns
        -------
        Tensor
            Whole loss
        """
        n_out = target.shape[-1]
        loss = 0
        for k in range(n_out):
            if self.limit_part is not None and k in self.limit_part:
                continue
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            if self.data_gap[k] > 0:
                p, t = deal_gap_data(p0, t0, self.data_gap[k], self.device)
            if type(self.loss_funcs) is list:
                temp = self.item_weight[k] * self.loss_funcs[k](p, t)
            else:
                temp = self.item_weight[k] * self.loss_funcs(p, t)
            # sum of all k-th loss
            if torch.isnan(temp).any():
                continue
            loss = loss + temp
        return loss


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