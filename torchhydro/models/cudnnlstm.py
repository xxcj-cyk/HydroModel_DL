"""
Author: MHPI group, Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2024-10-09 16:36:34
LastEditors: Wenyu Ouyang
Description: LSTM with dropout implemented by Kuai Fang and more LSTMs using it
FilePath: \torchhydro\torchhydro\models\cudnnlstm.py
Copyright (c) 2021-2022 MHPI group, Wenyu Ouyang. All rights reserved.
"""

import math
from typing import Union

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from torchhydro.models.ann import SimpleAnn
from torchhydro.models.dropout import DropMask, create_mask


class CudnnLstm(nn.Module):
    """
    LSTM with dropout implemented by Kuai Fang: https://github.com/mhpi/hydroDL/blob/release/hydroDL/model/rnn.py

    Only run in GPU; the CPU version is LstmCellTied in this file
    """

    def __init__(self, *, input_size, hidden_size, dr=0.5):
        """

        Parameters
        ----------
        input_size
            number of neurons in input layer
        hidden_size
            number of neurons in hidden layer
        dr
            dropout rate
        """
        super(CudnnLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dr = dr
        self.w_ih = Parameter(torch.Tensor(hidden_size * 4, input_size))
        self.w_hh = Parameter(torch.Tensor(hidden_size * 4, hidden_size))
        self.b_ih = Parameter(torch.Tensor(hidden_size * 4))
        self.b_hh = Parameter(torch.Tensor(hidden_size * 4))
        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]
        # self.cuda()
        # set the mask
        self.reset_mask()
        # initialize the weights and bias of the model
        self.reset_parameters()

    def _apply(self, fn):
        """just use the default _apply function

        Parameters
        ----------
        fn : function
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # _apply is always recursively applied to all submodules and the module itself such as move all to GPU
        return super()._apply(fn)

    def __setstate__(self, d):
        """a python magic function to set the state of the object used for deserialization

        Parameters
        ----------
        d : _type_
            _description_
        """
        super().__setstate__(d)
        # set a default value for _data_ptrs
        self.__dict__.setdefault("_data_ptrs", [])
        if "all_weights" in d:
            self._all_weights = d["all_weights"]
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]

    def reset_mask(self):
        """generate mask for dropout"""
        self.mask_w_ih = create_mask(self.w_ih, self.dr)
        self.mask_w_hh = create_mask(self.w_hh, self.dr)

    def reset_parameters(self):
        """initialize the weights and bias of the model using Xavier initialization"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            # uniform distribution between -stdv and stdv for the weights and bias initialization
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None, cx=None, do_drop_mc=False, dropout_false=False):
        # dropout_false: it will ensure do_drop is false, unless do_drop_mc is true
        if dropout_false and (not do_drop_mc):
            do_drop = False
        elif self.dr > 0 and (do_drop_mc is True or self.training is True):
            # if train mode and set self.dr > 0, then do_drop is true
            # so each time the model forward function is called, the dropout is applied
            do_drop = True
        else:
            do_drop = False
        # input must be a tensor with shape (seq_len, batch, input_size)
        batch_size = input.size(1)

        if hx is None:
            hx = input.new_zeros(1, batch_size, self.hidden_size, requires_grad=False)
        if cx is None:
            cx = input.new_zeros(1, batch_size, self.hidden_size, requires_grad=False)

        # handle = torch.backends.cudnn.get_handle()
        if do_drop is True:
            # cuDNN backend - disabled flat weight
            # NOTE: each time the mask is newly generated, so for each batch the mask is different
            self.reset_mask()
            # apply the mask to the weights
            weight = [
                DropMask.apply(self.w_ih, self.mask_w_ih, True),
                DropMask.apply(self.w_hh, self.mask_w_hh, True),
                self.b_ih,
                self.b_hh,
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]
        if torch.__version__ < "1.8":
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,  # 2 means LSTM
                self.hidden_size,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )
        else:
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,  # 2 means LSTM
                self.hidden_size,
                0,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )
        return output, (hy, cy)

    @property
    def all_weights(self):
        """return all weights and bias of the model as a list"""
        # getattr() is used to get the value of an object's attribute
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]


class CudnnLstmModel(nn.Module):
    def __init__(self, n_input_features, n_output_features, n_hidden_states, dr=0.5):
        """
        An LSTM model writen by Kuai Fang from this paper: https://doi.org/10.1002/2017GL075619

        only gpu version

        Parameters
        ----------
        n_input_features
            the number of input features
        n_output_features
            the number of output features
        n_hidden_states
            the number of hidden features
        dr
            dropout rate and its default is 0.5
        """
        super(CudnnLstmModel, self).__init__()
        self.nx = n_input_features
        self.ny = n_output_features
        self.hidden_size = n_hidden_states
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(self.nx, self.hidden_size)
        self.lstm = CudnnLstm(
            input_size=self.hidden_size, hidden_size=self.hidden_size, dr=dr
        )
        self.linearOut = torch.nn.Linear(self.hidden_size, self.ny)

    def forward(self, x, do_drop_mc=False, dropout_false=False, return_h_c=False):
        x0 = F.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(
            x0, do_drop_mc=do_drop_mc, dropout_false=dropout_false
        )
        out = self.linearOut(out_lstm)
        return (out, (hn, cn)) if return_h_c else out


class CNN1dKernel(torch.nn.Module):
    def __init__(self, *, ninchannel=1, nkernel=3, kernelSize=3, stride=1, padding=0):
        super(CNN1dKernel, self).__init__()
        self.cnn1d = torch.nn.Conv1d(
            in_channels=ninchannel,
            out_channels=nkernel,
            kernel_size=kernelSize,
            padding=padding,
            stride=stride,
        )
        self.name = "CNN1dkernel"
        self.is_legacy = True

    def forward(self, x):
        return F.relu(self.cnn1d(x))

class CudnnLstmModelMultiOutput(nn.Module):
    def __init__(
        self,
        n_input_features,
        n_output_features,
        n_hidden_states,
        layer_hidden_size=(128, 64),
        dr=0.5,
        dr_hidden=0.0,
    ):
        """
        Multiple output CudnnLSTM.

        It has multiple output layers, each for one output, so that we can easily freeze any output layer.

        Parameters
        ----------
        n_input_features
            the size of input features
        n_output_features
            the size of output features; in this model, we set different nonlinear layer for each output
        n_hidden_states
            the size of LSTM's hidden features
        layer_hidden_size
            hidden_size for multi-layers
        dr
            dropout rate
        dr_hidden
            dropout rates of hidden layers
        """
        super(CudnnLstmModelMultiOutput, self).__init__()
        self.ct = 0
        multi_layers = torch.nn.ModuleList()
        for i in range(n_output_features):
            multi_layers.add_module(
                "layer%d" % (i + 1),
                SimpleAnn(n_hidden_states, 1, layer_hidden_size, dr=dr_hidden),
            )
        self.multi_layers = multi_layers
        self.linearIn = torch.nn.Linear(n_input_features, n_hidden_states)
        self.lstm = CudnnLstm(
            input_size=n_hidden_states, hidden_size=n_hidden_states, dr=dr
        )

    def forward(self, x, do_drop_mc=False, dropout_false=False, return_h_c=False):
        x0 = F.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(
            x0, do_drop_mc=do_drop_mc, dropout_false=dropout_false
        )
        outs = [mod(out_lstm) for mod in self.multi_layers]
        final = torch.cat(outs, dim=-1)
        return (final, (hn, cn)) if return_h_c else final
