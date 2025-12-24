#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
损失函数对比实验批量执行脚本

使用方法：
    python run_loss_experiments.py --base_cmd "python main.py" --ctx 0 --epochs 100

或者直接修改脚本中的配置，然后运行：
    python run_loss_experiments.py
"""

import subprocess
import json
import argparse
import os
from datetime import datetime

# 默认实验配置
DEFAULT_EXPERIMENTS = [
    # 实验组1：基准对比
    {
        "name": "Exp-01-RMSEFlood-baseline",
        "loss_func": "RMSEFlood",
        "loss_param": {}
    },
    {
        "name": "Exp-02-HybridFlood-mae05",
        "loss_func": "HybridFlood",
        "loss_param": {"mae_weight": 0.5}
    },
    {
        "name": "Exp-03-HybridFlood-mae10",
        "loss_func": "HybridFlood",
        "loss_param": {"mae_weight": 1.0}
    },
    # 实验组2：PeakFocusedFlood参数调优
    {
        "name": "Exp-04-PeakFocused-balanced",
        "loss_func": "PeakFocusedFlood",
        "loss_param": {"peak_weight": 1.0, "time_weight": 1.0, "overall_weight": 0.5}
    },
    {
        "name": "Exp-05-PeakFocused-default",
        "loss_func": "PeakFocusedFlood",
        "loss_param": {"peak_weight": 2.0, "time_weight": 1.0, "overall_weight": 0.5}
    },
    {
        "name": "Exp-06-PeakFocused-peak3",
        "loss_func": "PeakFocusedFlood",
        "loss_param": {"peak_weight": 3.0, "time_weight": 1.0, "overall_weight": 0.3}
    },
    {
        "name": "Exp-07-PeakFocused-time2",
        "loss_func": "PeakFocusedFlood",
        "loss_param": {"peak_weight": 1.0, "time_weight": 2.0, "overall_weight": 0.5}
    },
    {
        "name": "Exp-08-PeakFocused-peak2time2",
        "loss_func": "PeakFocusedFlood",
        "loss_param": {"peak_weight": 2.0, "time_weight": 2.0, "overall_weight": 0.3}
    },
]


def run_experiment(base_cmd, exp_config, epochs, ctx, dry_run=False):
    """执行单个实验"""
    print(f"\n{'='*80}")
    print(f"实验: {exp_config['name']}")
    print(f"损失函数: {exp_config['loss_func']}")
    print(f"参数: {json.dumps(exp_config['loss_param'], indent=2)}")
    print(f"{'='*80}\n")
    
    if dry_run:
        print("[DRY RUN] 不会实际执行")
        return True
    
    # 构建命令
    cmd = base_cmd.split() if isinstance(base_cmd, str) else base_cmd
    cmd.extend([
        "--loss_func", exp_config['loss_func'],
        "--loss_param", json.dumps(exp_config['loss_param']),
        "--train_epoch", str(epochs),
        "--ctx", str(ctx)
    ])
    
    # 执行命令
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ {exp_config['name']} 完成\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {exp_config['name']} 失败: {e}\n")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ {exp_config['name']} 被用户中断\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="批量执行损失函数对比实验")
    parser.add_argument(
        "--base_cmd",
        type=str,
        default="python main.py",
        help="基础命令（默认: python main.py）"
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=0,
        help="GPU编号（默认: 0，-1表示CPU）"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练轮数（默认: 100）"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="实验配置文件路径（JSON格式），如果未指定则使用默认配置"
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="从第几个实验开始（默认: 0）"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅显示将要执行的命令，不实际运行"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="日志文件路径（可选）"
    )
    
    args = parser.parse_args()
    
    # 加载实验配置
    if args.experiments and os.path.exists(args.experiments):
        with open(args.experiments, 'r', encoding='utf-8') as f:
            experiments = json.load(f)
    else:
        experiments = DEFAULT_EXPERIMENTS
    
    # 过滤实验（从指定位置开始）
    experiments = experiments[args.start_from:]
    
    print(f"\n{'='*80}")
    print(f"损失函数对比实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"实验数量: {len(experiments)}")
    print(f"训练轮数: {args.epochs}")
    print(f"设备: {'GPU ' + str(args.ctx) if args.ctx >= 0 else 'CPU'}")
    print(f"{'='*80}\n")
    
    # 执行实验
    results = []
    for i, exp in enumerate(experiments, start=args.start_from):
        print(f"\n进度: {i+1}/{len(experiments) + args.start_from}")
        success = run_experiment(
            args.base_cmd,
            exp,
            args.epochs,
            args.ctx,
            dry_run=args.dry_run
        )
        results.append({
            "experiment": exp['name'],
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
    
    # 输出总结
    print(f"\n{'='*80}")
    print("实验总结")
    print(f"{'='*80}\n")
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"总实验数: {len(results)}")
    print(f"成功: {successful}")
    print(f"失败: {failed}")
    
    if failed > 0:
        print("\n失败的实验:")
        for r in results:
            if not r['success']:
                print(f"  - {r['experiment']}")
    
    # 保存结果
    if args.log_file:
        with open(args.log_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total": len(results),
                    "successful": successful,
                    "failed": failed,
                    "start_time": results[0]['timestamp'] if results else None,
                    "end_time": results[-1]['timestamp'] if results else None
                },
                "experiments": results
            }, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.log_file}")
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()





