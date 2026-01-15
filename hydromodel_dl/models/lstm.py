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
        result = self.linearOut(out_lstm)
        return result


class MultiLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, dr=0.0):
        super(MultiLSTM, self).__init__()
        self.linearIn = nn.Linear(input_size, hidden_sizes[0])
        if dr is None:
            dr = [0.0] * len(hidden_sizes)
        elif isinstance(dr, (int, float)):
            dr = [dr] * len(hidden_sizes)
        assert len(dr) == len(
            hidden_sizes
        ), f"dr length ({len(dr)}) must be equal to hidden_sizes length ({len(hidden_sizes)})"
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            if i == 0:
                lstm_input_size = hidden_sizes[0]
            else:
                lstm_input_size = hidden_sizes[i - 1]
            self.lstm_layers.append(
                nn.LSTM(
                    lstm_input_size,
                    hidden_sizes[i],
                    num_layers=1,
                )
            )
            if dr[i] > 0:
                self.dropout_layers.append(nn.Dropout(dr[i]))
            else:
                self.dropout_layers.append(nn.Identity())
        self.linearOut = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x0 = F.relu(self.linearIn(x))
        lstm_out = x0
        for lstm_layer, dropout_layer in zip(self.lstm_layers, self.dropout_layers):
            lstm_out, (hn, cn) = lstm_layer(lstm_out)
            lstm_out = dropout_layer(lstm_out)

        return self.linearOut(lstm_out)


class LinearSimpleLSTM(SimpleLSTM):
    def __init__(self, linear_size, **kwargs):
        super(LinearSimpleLSTM, self).__init__(**kwargs)
        self.former_linear = nn.Linear(linear_size, kwargs["input_size"])

    def forward(self, x):
        x0 = F.relu(self.former_linear(x))
        return super(LinearSimpleLSTM, self).forward(x0)
    
    
class SimpleLSTMLinear(SimpleLSTM):
    def __init__(self, final_linear_size, **kwargs):
        super(SimpleLSTMLinear, self).__init__(**kwargs)
        self.final_linear = nn.Linear(kwargs["output_size"], final_linear_size)

    def forward(self, x):
        x = super(SimpleLSTMLinear, self).forward(x)
        x = self.final_linear(x)
        return x
