import torch.nn as nn
from torch.nn import functional as F


class StandardLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, dr=0.0):
        super(StandardLSTM, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)

        self.linearIn = nn.Linear(input_size, hidden_sizes[0])

        self.lstm_layers = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim = hidden_sizes[i - 1] if i > 0 else hidden_sizes[0]
            self.lstm_layers.append(nn.LSTM(input_dim, hidden_sizes[i], 1, dropout=dr))

        self.linearOut = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        out_lstm = F.relu(self.linearIn(x))
        for lstm in self.lstm_layers:
            out_lstm, (hn, cn) = lstm(out_lstm)
        return self.linearOut(out_lstm)
