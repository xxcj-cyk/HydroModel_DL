import torch
import torch.nn as nn
from torch.nn import functional as F


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dr=0.0):
        super(SimpleLSTM, self).__init__()
        self.linearIn = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            1,
            dropout=dr,
        )
        self.linearOut = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x0 = F.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(x0)
        return self.linearOut(out_lstm)

class MultiLSTM(nn.Module):
    def __init__(self, num_models, input_size, output_size, hidden_size, dr=0.0):
        super(MultiLSTM, self).__init__()
        self.models = nn.ModuleList([
            SimpleLSTM(input_size, output_size, hidden_size, dr)
            for _ in range(num_models)
        ])
        self.num_models = num_models

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return torch.stack(outputs, dim=0)