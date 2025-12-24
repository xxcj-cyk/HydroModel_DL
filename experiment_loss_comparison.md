# 损失函数对比实验方案

## 实验目标
对比不同损失函数在场次洪水预报中的效果，重点关注洪峰误差和峰现时间误差。

## 固定超参数（所有实验保持一致）
```python
# LSTM模型参数
hidden_size: 20          # 保持不变
num_layers: 1            # 保持不变
forecast_history: 30     # 序列长度，保持不变
forecast_length: 30      # 保持不变

# 训练参数
learning_rate: 0.001     # 保持不变（根据你的实际设置调整）
batch_size: 100          # 保持不变
train_epoch: 100         # 保持不变
optimizer: Adam          # 保持不变
```

## 实验组设计

### 实验组1：基准对比（基础损失函数）
| 实验编号 | loss_func | loss_param | 说明 |
|---------|-----------|------------|------|
| Exp-01 | RMSEFlood | `{}` | 基准实验：只对洪水期间计算RMSE |
| Exp-02 | HybridFlood | `{"mae_weight": 0.5}` | PES + MAE混合损失 |
| Exp-03 | HybridFlood | `{"mae_weight": 1.0}` | 增加MAE权重 |

### 实验组2：PeakFocusedFlood参数调优
| 实验编号 | loss_func | loss_param | 说明 |
|---------|-----------|------------|------|
| Exp-04 | PeakFocusedFlood | `{"peak_weight": 1.0, "time_weight": 1.0, "overall_weight": 0.5}` | 平衡权重 |
| Exp-05 | PeakFocusedFlood | `{"peak_weight": 2.0, "time_weight": 1.0, "overall_weight": 0.5}` | 默认参数（更关注洪峰） |
| Exp-06 | PeakFocusedFlood | `{"peak_weight": 3.0, "time_weight": 1.0, "overall_weight": 0.3}` | 更强调洪峰 |
| Exp-07 | PeakFocusedFlood | `{"peak_weight": 1.0, "time_weight": 2.0, "overall_weight": 0.5}` | 更关注峰现时间 |
| Exp-08 | PeakFocusedFlood | `{"peak_weight": 2.0, "time_weight": 2.0, "overall_weight": 0.3}` | 同时强调洪峰和时间 |

### 实验组3：长序列版本对比（如果使用长序列数据）
| 实验编号 | loss_func | loss_param | 说明 |
|---------|-----------|------------|------|
| Exp-09 | PeakFocused | `{"peak_weight": 2.0, "time_weight": 1.0, "overall_weight": 0.5, "peak_threshold_ratio": 0.7}` | 默认阈值 |
| Exp-10 | PeakFocused | `{"peak_weight": 2.0, "time_weight": 1.0, "overall_weight": 0.5, "peak_threshold_ratio": 0.5}` | 更宽的高值区域 |
| Exp-11 | PeakFocused | `{"peak_weight": 2.0, "time_weight": 1.0, "overall_weight": 0.5, "peak_threshold_ratio": 0.9}` | 更窄的高值区域 |

## 评估指标

### 主要指标（重点关注）
1. **洪峰误差（Peak Flow Error）**
   - 绝对误差：`|pred_peak - true_peak|`
   - 相对误差：`|pred_peak - true_peak| / true_peak`
   - RMSE of peaks

2. **峰现时间误差（Peak Time Error）**
   - 绝对误差（时间步）：`|pred_time - true_time|`
   - 相对误差：`|pred_time - true_time| / sequence_length`

### 次要指标（整体拟合）
3. **整体RMSE**（洪水期间）
4. **整体MAE**（洪水期间）
5. **NSE**（Nash-Sutcliffe Efficiency）
6. **KGE**（Kling-Gupta Efficiency）

## 实验执行脚本示例

### 方式1：命令行执行（推荐用于批量实验）

```bash
# 基准实验
python main.py \
    --loss_func RMSEFlood \
    --loss_param '{}' \
    --train_epoch 100 \
    --ctx 0

# HybridFlood实验
python main.py \
    --loss_func HybridFlood \
    --loss_param '{"mae_weight": 0.5}' \
    --train_epoch 100 \
    --ctx 0

# PeakFocusedFlood实验（默认参数）
python main.py \
    --loss_func PeakFocusedFlood \
    --loss_param '{"peak_weight": 2.0, "time_weight": 1.0, "overall_weight": 0.5}' \
    --train_epoch 100 \
    --ctx 0

# PeakFocusedFlood实验（强调洪峰）
python main.py \
    --loss_func PeakFocusedFlood \
    --loss_param '{"peak_weight": 3.0, "time_weight": 1.0, "overall_weight": 0.3}' \
    --train_epoch 100 \
    --ctx 0
```

### 方式2：批量执行脚本

创建 `run_experiments.sh`:

```bash
#!/bin/bash

# 设置固定参数
EPOCHS=100
CTX=0

# 实验组1：基准对比
echo "Running Exp-01: RMSEFlood baseline"
python main.py --loss_func RMSEFlood --loss_param '{}' --train_epoch $EPOCHS --ctx $CTX

echo "Running Exp-02: HybridFlood (mae_weight=0.5)"
python main.py --loss_func HybridFlood --loss_param '{"mae_weight": 0.5}' --train_epoch $EPOCHS --ctx $CTX

echo "Running Exp-03: HybridFlood (mae_weight=1.0)"
python main.py --loss_func HybridFlood --loss_param '{"mae_weight": 1.0}' --train_epoch $EPOCHS --ctx $CTX

# 实验组2：PeakFocusedFlood参数调优
echo "Running Exp-04: PeakFocusedFlood (balanced)"
python main.py --loss_func PeakFocusedFlood --loss_param '{"peak_weight": 1.0, "time_weight": 1.0, "overall_weight": 0.5}' --train_epoch $EPOCHS --ctx $CTX

echo "Running Exp-05: PeakFocusedFlood (default)"
python main.py --loss_func PeakFocusedFlood --loss_param '{"peak_weight": 2.0, "time_weight": 1.0, "overall_weight": 0.5}' --train_epoch $EPOCHS --ctx $CTX

echo "Running Exp-06: PeakFocusedFlood (peak-focused)"
python main.py --loss_func PeakFocusedFlood --loss_param '{"peak_weight": 3.0, "time_weight": 1.0, "overall_weight": 0.3}' --train_epoch $EPOCHS --ctx $CTX

echo "Running Exp-07: PeakFocusedFlood (time-focused)"
python main.py --loss_func PeakFocusedFlood --loss_param '{"peak_weight": 1.0, "time_weight": 2.0, "overall_weight": 0.5}' --train_epoch $EPOCHS --ctx $CTX

echo "Running Exp-08: PeakFocusedFlood (peak+time)"
python main.py --loss_func PeakFocusedFlood --loss_param '{"peak_weight": 2.0, "time_weight": 2.0, "overall_weight": 0.3}' --train_epoch $EPOCHS --ctx $CTX

echo "All experiments completed!"
```

### 方式3：Python脚本批量执行

创建 `run_experiments.py`:

```python
import subprocess
import json
import os

# 固定参数
EPOCHS = 100
CTX = 0

# 实验配置
experiments = [
    {
        "name": "Exp-01",
        "loss_func": "RMSEFlood",
        "loss_param": {}
    },
    {
        "name": "Exp-02",
        "loss_func": "HybridFlood",
        "loss_param": {"mae_weight": 0.5}
    },
    {
        "name": "Exp-03",
        "loss_func": "HybridFlood",
        "loss_param": {"mae_weight": 1.0}
    },
    {
        "name": "Exp-04",
        "loss_func": "PeakFocusedFlood",
        "loss_param": {"peak_weight": 1.0, "time_weight": 1.0, "overall_weight": 0.5}
    },
    {
        "name": "Exp-05",
        "loss_func": "PeakFocusedFlood",
        "loss_param": {"peak_weight": 2.0, "time_weight": 1.0, "overall_weight": 0.5}
    },
    {
        "name": "Exp-06",
        "loss_func": "PeakFocusedFlood",
        "loss_param": {"peak_weight": 3.0, "time_weight": 1.0, "overall_weight": 0.3}
    },
    {
        "name": "Exp-07",
        "loss_func": "PeakFocusedFlood",
        "loss_param": {"peak_weight": 1.0, "time_weight": 2.0, "overall_weight": 0.5}
    },
    {
        "name": "Exp-08",
        "loss_func": "PeakFocusedFlood",
        "loss_param": {"peak_weight": 2.0, "time_weight": 2.0, "overall_weight": 0.3}
    },
]

# 执行实验
for exp in experiments:
    print(f"\n{'='*60}")
    print(f"Running {exp['name']}: {exp['loss_func']}")
    print(f"Parameters: {exp['loss_param']}")
    print(f"{'='*60}\n")
    
    cmd = [
        "python", "main.py",
        "--loss_func", exp['loss_func'],
        "--loss_param", json.dumps(exp['loss_param']),
        "--train_epoch", str(EPOCHS),
        "--ctx", str(CTX)
    ]
    
    # 添加其他固定参数（根据你的实际脚本调整）
    # cmd.extend(["--other_param", "value"])
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ {exp['name']} completed successfully\n")
    except subprocess.CalledProcessError as e:
        print(f"✗ {exp['name']} failed with error: {e}\n")

print("\nAll experiments completed!")
```

## 结果分析建议

### 1. 结果汇总表
创建Excel或CSV文件记录所有实验结果：

| 实验编号 | loss_func | loss_param | 洪峰误差 | 峰现时间误差 | 整体RMSE | 整体MAE | NSE | KGE |
|---------|-----------|------------|---------|------------|---------|---------|-----|-----|
| Exp-01 | RMSEFlood | {} | ... | ... | ... | ... | ... | ... |
| Exp-02 | HybridFlood | {"mae_weight": 0.5} | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 2. 可视化分析
- 箱线图：对比不同损失函数的洪峰误差分布
- 散点图：洪峰误差 vs 峰现时间误差
- 折线图：训练过程中的损失变化
- 热力图：不同参数组合的效果对比

### 3. 统计分析
- 使用t检验或Mann-Whitney U检验对比不同损失函数
- 计算置信区间
- 识别最优参数组合

## 注意事项

1. **随机种子**：所有实验使用相同的随机种子，确保可复现性
2. **数据划分**：使用相同的数据划分（train/valid/test）
3. **早停策略**：如果使用早停，确保所有实验使用相同的patience
4. **模型保存**：为每个实验保存模型和结果，便于后续分析
5. **日志记录**：记录每个实验的完整配置和结果

## 推荐实验顺序

1. **第一阶段**：先运行Exp-01, Exp-02, Exp-05（快速对比基准、Hybrid和PeakFocused默认参数）
2. **第二阶段**：根据第一阶段结果，选择最有希望的损失函数进行参数调优
3. **第三阶段**：对最优参数组合进行多次运行，验证稳定性

