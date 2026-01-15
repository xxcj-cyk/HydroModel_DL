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
        # 插入 print 8：检查输入
        print(f"DEBUG SimpleLSTM.forward: input x shape: {x.shape}")
        print(f"  x has NaN: {torch.isnan(x).any()}, range: [{x.min():.4f}, {x.max():.4f}]")
        
        # 检查 linearIn 权重
        if torch.isnan(self.linearIn.weight).any():
            print(f"  ❌ NaN in linearIn.weight!")
        if self.linearIn.bias is not None and torch.isnan(self.linearIn.bias).any():
            print(f"  ❌ NaN in linearIn.bias!")
        
        x0 = F.relu(self.linearIn(x))
        
        # 插入 print 9：检查 linearIn 输出
        print(f"DEBUG: After linearIn, x0 shape: {x0.shape}")
        print(f"  x0 has NaN: {torch.isnan(x0).any()}, range: [{x0.min():.4f}, {x0.max():.4f}]")
        
        # 检查 LSTM 权重
        for name, param in self.lstm.named_parameters():
            if torch.isnan(param).any():
                print(f"  ❌ NaN in LSTM.{name}!")
        
        out_lstm, (hn, cn) = self.lstm(x0)
        
        # 插入 print 10：检查 LSTM 输出和状态
        print(f"DEBUG: After LSTM")
        print(f"  out_lstm shape: {out_lstm.shape}, has NaN: {torch.isnan(out_lstm).any()}")
        print(f"  hn shape: {hn.shape}, has NaN: {torch.isnan(hn).any()}")
        print(f"  cn shape: {cn.shape}, has NaN: {torch.isnan(cn).any()}")
        if torch.isnan(out_lstm).any():
            print(f"  ❌❌❌ NaN in LSTM output! ❌❌❌")
            print(f"    out_lstm range: [{out_lstm.min():.4f}, {out_lstm.max():.4f}]")
            print(f"    hn range: [{hn.min():.4f}, {hn.max():.4f}]")
            print(f"    cn range: [{cn.min():.4f}, {cn.max():.4f}]")
        
        # 检查 linearOut 权重
        if torch.isnan(self.linearOut.weight).any():
            print(f"  ❌ NaN in linearOut.weight!")
        if self.linearOut.bias is not None and torch.isnan(self.linearOut.bias).any():
            print(f"  ❌ NaN in linearOut.bias!")
        
        result = self.linearOut(out_lstm)
        
        # 插入 print 11：检查最终输出
        print(f"DEBUG: After linearOut, result shape: {result.shape}")
        print(f"  result has NaN: {torch.isnan(result).any()}, range: [{result.min():.4f}, {result.max():.4f}]")
        
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
