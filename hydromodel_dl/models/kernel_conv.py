import torch.nn as nn
import torch
from torch.nn import functional as F


class KernelConv(nn.Module):
    def __init__(self, a, theta, kernel_size):
        super(KernelConv, self).__init__()
        self.a = a
        self.theta = theta
        routa = self.a.repeat(kernel_size, 1).unsqueeze(-1)
        routb = self.theta.repeat(kernel_size, 1).unsqueeze(-1)
        self.uh_gamma = uh_gamma(routa, routb, len_uh=kernel_size)

    def forward(self, x):
        # dim: permute from [len_uh, batch, feature] to [batch, feature, len_uh]
        uh = self.uh_gamma.permute(1, 2, 0)
        # the dim of conv kernel in F.conv1d is out_channels, in_channels (feature)/groups, width (seq)
        # the dim of inputs in F.conv1d are batch, in_channels (feature) and width (seq),
        # each element in a batch should has its own conv kernel,
        # hence set groups = batch_size and permute input's batch-dim to channel-dim to make "groups" work
        inputs = x.permute(2, 1, 0)
        batch_size = x.shape[1]
        # conv1d in NN is different from the general convolution: it is lack of a flip
        outputs = F.conv1d(
            inputs, torch.flip(uh, [2]), groups=batch_size, padding=uh.shape[-1] - 1
        )
        # permute from [feature, batch, seq] to [seq, batch, feature]
        return outputs[:, :, : -(uh.shape[-1] - 1)].permute(2, 1, 0)


def uh_conv(x, uh_made) -> torch.Tensor:
    uh = uh_made.permute(1, 2, 0)
    # the dim of conv kernel in F.conv1d is out_channels, in_channels (feature)/groups, width (seq)
    # the dim of inputs in F.conv1d are batch, in_channels (feature) and width (seq),
    # each element in a batch should has its own conv kernel,
    # hence set groups = batch_size and permute input's batch-dim to channel-dim to make "groups" work
    inputs = x.permute(2, 1, 0)
    batch_size = x.shape[1]
    # conv1d in NN is different from the general convolution: it is lack of a flip
    outputs = F.conv1d(
        inputs, torch.flip(uh, [2]), groups=batch_size, padding=uh.shape[-1] - 1
    )
    # cut to same shape with x and permute from [feature, batch, seq] to [seq, batch, feature]
    return outputs[:, :, : x.shape[0]].permute(2, 1, 0)


def uh_gamma(a, theta, len_uh=10):
    # dims of a: time_seq (same all time steps), batch, feature=1
    m = a.shape
    assert len_uh <= m[0]
    # aa > 0, here we set minimum 0.1 (min of a is 0, set when calling this func); First dimension of a is repeat
    aa = F.relu(a[0:len_uh, :, :]) + 0.1
    # theta > 0, here set minimum 0.5
    theta = F.relu(theta[0:len_uh, :, :]) + 0.5
    # len_f, batch, feature
    t = (
        torch.arange(0.5, len_uh * 1.0)
        .view([len_uh, 1, 1])
        .repeat([1, m[1], m[2]])
        .to(aa.device)
    )
    denominator = (aa.lgamma().exp()) * (theta**aa)
    # [len_f, m[1], m[2]]
    w = 1 / denominator * (t ** (aa - 1)) * (torch.exp(-t / theta))
    w = w / w.sum(0)  # scale to 1 for each UH
    return w
