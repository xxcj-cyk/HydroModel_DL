"""
LSTM模型SHAP分析示例

本示例展示如何使用SHAP分析器对训练好的LSTM模型进行特征重要性分析
"""

import sys
import os
import numpy as np
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydromodel_dl.trainers.shap_analyzer import LSTMSHAPAnalyzer, analyze_lstm_with_shap
from hydromodel_dl.trainers.deep_hydro import DeepHydro
from hydromodel_dl.datasets.data_sets import LongTermDataset


def shap_analysis_example(config_dict):
    """
    对训练好的LSTM模型进行SHAP分析的示例
    
    Parameters
    ----------
    config_dict : dict
        配置字典，包含模型和数据配置
    """
    # 1. 加载训练好的模型
    print("加载模型...")
    deephydro = DeepHydro(config_dict)
    model = deephydro.load_model(mode="infer")  # 使用推理模式加载已训练的模型
    device = deephydro.device
    
    # 2. 准备数据
    print("准备数据...")
    # 使用测试数据集作为背景数据
    test_dataset = deephydro.testdataset
    
    # 从数据集中提取一些样本作为背景数据
    # 注意：背景数据应该代表模型的"基准"输入
    background_samples = []
    n_background = min(200, len(test_dataset))  # 使用最多200个样本作为背景
    
    for i in range(n_background):
        x, y = test_dataset[i]
        background_samples.append(x.numpy())
    
    background_data = np.array(background_samples)  # 形状: [n_samples, seq_len, n_features]
    
    # 3. 选择要解释的实例（可以是一个或多个样本）
    # 这里我们选择前10个测试样本进行解释
    n_instances = min(10, len(test_dataset))
    instances = []
    
    for i in range(n_instances):
        x, y = test_dataset[i]
        instances.append(x.numpy())
    
    instances = np.array(instances)  # 形状: [n_instances, seq_len, n_features]
    
    # 4. 进行SHAP分析
    print("开始SHAP分析...")
    results = analyze_lstm_with_shap(
        model=model,
        background_data=background_data,
        instances=instances,
        feature_names=config_dict.get("data_cfgs", {}).get("relevant_cols", None),
        device=device,
        explainer_type="DeepExplainer",  # 或 "KernelExplainer"
        save_path=os.path.join(config_dict["data_cfgs"]["test_path"], "shap_results.pkl")
    )
    
    # 5. 打印结果
    print("\n=== SHAP分析结果 ===")
    print(f"SHAP值形状: {results['shap_values'].shape}")
    print(f"\n特征重要性 (平均值):")
    for i, (name, importance) in enumerate(zip(results['feature_names'], results['feature_importance'])):
        if results['feature_importance'].ndim == 2:
            # 多输出情况
            importance_str = ", ".join([f"{v:.4f}" for v in importance])
            print(f"  {name}: [{importance_str}]")
        else:
            # 单输出情况
            print(f"  {name}: {importance:.4f}")
    
    print(f"\n时间步重要性 (前10个时间步):")
    temporal_imp = results['temporal_importance']
    if temporal_imp.ndim == 2:
        # 多输出情况
        for t in range(min(10, temporal_imp.shape[0])):
            importance_str = ", ".join([f"{v:.4f}" for v in temporal_imp[t]])
            print(f"  时间步 {t}: [{importance_str}]")
    else:
        # 单输出情况
        for t in range(min(10, temporal_imp.shape[0])):
            print(f"  时间步 {t}: {temporal_imp[t]:.4f}")
    
    return results


def shap_analysis_advanced_example(config_dict):
    """
    高级SHAP分析示例 - 使用LSTMSHAPAnalyzer类进行更精细的控制
    """
    # 1. 加载模型和数据（同上）
    deephydro = DeepHydro(config_dict)
    model = deephydro.load_model(mode="infer")
    device = deephydro.device
    test_dataset = deephydro.testdataset
    
    # 准备背景数据
    background_samples = []
    n_background = min(200, len(test_dataset))
    for i in range(n_background):
        x, y = test_dataset[i]
        background_samples.append(x.numpy())
    background_data = np.array(background_samples)
    
    # 选择要解释的实例
    n_instances = min(10, len(test_dataset))
    instances = []
    for i in range(n_instances):
        x, y = test_dataset[i]
        instances.append(x.numpy())
    instances = np.array(instances)
    
    # 2. 创建分析器
    print("创建SHAP分析器...")
    analyzer = LSTMSHAPAnalyzer(
        model=model,
        background_data=background_data,
        device=device,
        explainer_type="DeepExplainer"
    )
    
    # 3. 计算SHAP值
    print("计算SHAP值...")
    shap_values = analyzer.explain(instances, nsamples=100)
    
    # 4. 计算不同聚合方式的特征重要性
    print("\n计算特征重要性...")
    methods = ["mean_abs", "sum_abs", "max_abs"]
    feature_importance_dict = {}
    for method in methods:
        importance = analyzer.get_feature_importance(shap_values, method=method)
        feature_importance_dict[method] = importance
        print(f"\n{method}: {importance}")
    
    # 5. 计算时间步重要性
    print("\n计算时间步重要性...")
    temporal_importance = analyzer.get_temporal_importance(shap_values, method="mean_abs")
    print(f"时间步重要性: {temporal_importance}")
    
    # 6. 可视化（如果安装了matplotlib和seaborn）
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 绘制特征重要性
        plt.figure(figsize=(10, 6))
        feature_names = config_dict.get("data_cfgs", {}).get("relevant_cols", 
                                                              [f"Feature_{i}" for i in range(shap_values.shape[2])])
        importance_mean_abs = feature_importance_dict["mean_abs"]
        
        if importance_mean_abs.ndim == 1:
            # 单输出
            indices = np.argsort(importance_mean_abs)[::-1]
            plt.barh(range(len(indices)), importance_mean_abs[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('特征重要性 (平均绝对SHAP值)')
            plt.title('LSTM模型特征重要性分析')
            plt.tight_layout()
            
            save_path = os.path.join(config_dict["data_cfgs"]["test_path"], "shap_feature_importance.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n特征重要性图已保存到: {save_path}")
            plt.close()
        
        # 绘制时间步重要性
        plt.figure(figsize=(12, 6))
        plt.plot(temporal_importance)
        plt.xlabel('时间步')
        plt.ylabel('重要性 (平均绝对SHAP值)')
        plt.title('LSTM模型时间步重要性分析')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(config_dict["data_cfgs"]["test_path"], "shap_temporal_importance.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"时间步重要性图已保存到: {save_path}")
        plt.close()
        
    except ImportError:
        print("\n注意: 未安装matplotlib和seaborn，跳过可视化")
    
    return {
        "shap_values": shap_values,
        "feature_importance": feature_importance_dict,
        "temporal_importance": temporal_importance
    }


if __name__ == "__main__":
    # 这里需要提供您的配置字典
    # 示例配置（请根据实际情况修改）
    example_config = {
        "data_cfgs": {
            "test_path": "./results",
            "dataset": "LongTermDataset",  # 或其他数据集类型
            # ... 其他数据配置
        },
        "model_cfgs": {
            "model_name": "SimpleLSTM",  # 或其他LSTM模型
            # ... 其他模型配置
        },
        # ... 其他配置
    }
    
    print("=== 基本SHAP分析示例 ===")
    # results = shap_analysis_example(example_config)
    
    print("\n=== 高级SHAP分析示例 ===")
    # results_advanced = shap_analysis_advanced_example(example_config)

