import math
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor as T
from torch.nn import Parameter as P


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
    

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, dr=0.0):
        super(SimpleLSTM, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        
        self.linearIn = nn.Linear(input_size, hidden_sizes[0])
        
        self.lstm_layers = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim = hidden_sizes[i-1] if i > 0 else hidden_sizes[0]
            self.lstm_layers.append(nn.LSTM(input_dim, hidden_sizes[i], 1, dropout=dr))
        
        self.linearOut = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        out_lstm = F.relu(self.linearIn(x))
        for lstm in self.lstm_layers:
            out_lstm, (hn, cn) = lstm(out_lstm)
        return self.linearOut(out_lstm)