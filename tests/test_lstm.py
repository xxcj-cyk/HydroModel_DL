import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lstm import StandardLSTM

def test_lstm():
    # 定义模型参数
    input_size = 10         # 输入特征数量
    output_size = 1         # 输出特征数量
    hidden_sizes = [32, 64, 128]  # 隐藏层单元数量（3层）
    seq_len = 5             # 序列长度
    batch_size = 8          # 批大小
    dropout_rate = 0.8      # Dropout概率

    # 创建模型
    model = StandardLSTM(input_size, output_size, hidden_sizes, dr=dropout_rate)

    # 打印模型架构
    print(model)

    # 随机生成输入数据
    x = torch.randn(seq_len, batch_size, input_size)  # (seq_len, batch_size, input_size)

    # 前向传播
    output = model(x)

    # 打印输出形状
    print(f"Output shape: {output.shape}")

# 运行测试函数
test_lstm()