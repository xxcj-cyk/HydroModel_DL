"""
SHAP分析模块用于LSTM模型的特征重要性分析

该模块提供了对PyTorch LSTM模型进行SHAP分析的接口
"""

import numpy as np
import torch
import shap
from typing import Dict, Tuple, Optional, Callable
import warnings


class LSTMSHAPAnalyzer:
    """
    LSTM模型的SHAP分析器
    
    该类封装了SHAP分析方法，用于分析LSTM模型中输入特征的重要性
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        background_data: np.ndarray,
        device: Optional[torch.device] = None,
        explainer_type: str = "DeepExplainer"
    ):
        """
        初始化SHAP分析器
        
        Parameters
        ----------
        model : torch.nn.Module
            已训练的PyTorch LSTM模型
        background_data : np.ndarray
            背景数据集，形状为 [n_samples, seq_len, n_features]
            用于计算SHAP值的基准
        device : torch.device, optional
            计算设备，默认为None（自动检测）
        explainer_type : str
            SHAP解释器类型，可选 "DeepExplainer" 或 "KernelExplainer"
            DeepExplainer针对深度学习模型优化，速度更快
            KernelExplainer更通用但可能较慢
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # 转换背景数据为torch tensor
        if isinstance(background_data, np.ndarray):
            self.background_data = torch.from_numpy(background_data).float().to(self.device)
        else:
            self.background_data = background_data.to(self.device) if hasattr(background_data, 'to') else background_data
        
        self.explainer_type = explainer_type
        self.explainer = None
        self._init_explainer()
    
    def _init_explainer(self):
        """初始化SHAP解释器"""
        # 保存原始形状信息，用于展平和恢复
        bg_data = self.background_data
        if bg_data.shape[0] > 100:
            indices = torch.randperm(bg_data.shape[0])[:100]
            bg_data = bg_data[indices]
        
        bg_data_np = bg_data.cpu().numpy() if isinstance(bg_data, torch.Tensor) else bg_data
        
        # 保存原始形状信息
        self.original_shape = bg_data_np.shape  # [n_samples, seq_len, n_features]
        self.seq_len = bg_data_np.shape[1]
        self.n_features = bg_data_np.shape[2]
        self.n_outputs = None  # 将在第一次调用时确定
        
        # 定义一个包装函数，处理2D输入（展平的）并返回展平输出
        def model_wrapper(x_flat):
            """
            模型包装函数，处理展平的2D输入
            
            Parameters
            ----------
            x_flat : np.ndarray
                展平的输入数据，形状为 [n_samples, seq_len * n_features]
            
            Returns
            -------
            np.ndarray
                展平的模型输出，形状为 [n_samples, seq_len * n_output] 或 [n_samples, n_output]
            """
            # 将展平的输入重新整形为3D
            n_samples = x_flat.shape[0]
            x_3d = x_flat.reshape(n_samples, self.seq_len, self.n_features)
            
            # 转换为torch tensor
            if isinstance(x_3d, np.ndarray):
                x_tensor = torch.from_numpy(x_3d).float().to(self.device)
            else:
                x_tensor = x_3d.float().to(self.device) if hasattr(x_3d, 'to') else x_3d
            
            with torch.no_grad():
                output = self.model(x_tensor)
            
            # 转换为numpy并移回CPU
            if isinstance(output, torch.Tensor):
                output_np = output.cpu().numpy()
            else:
                output_np = output
            
            # 保存输出形状信息（如果还没确定）
            if self.n_outputs is None:
                if output_np.ndim == 3:
                    self.n_outputs = output_np.shape[2]
                elif output_np.ndim == 2:
                    self.n_outputs = output_np.shape[1]
                else:
                    self.n_outputs = 1
            
            # 展平输出：对于时序模型，我们通常只关心最后一个时间步的输出
            # 或者对所有时间步的输出取平均
            if output_np.ndim == 3:
                # [n_samples, seq_len, n_output] 
                # 对于SHAP分析，我们通常使用最后一个时间步的输出
                # 或者对所有时间步取平均
                if output_np.shape[1] > 1:  # 有多个时间步
                    # 使用最后一个时间步的输出（更常用）
                    output_flat = output_np[:, -1, :]  # [n_samples, n_output]
                else:
                    output_flat = output_np[:, 0, :]  # [n_samples, n_output]
            elif output_np.ndim == 2:
                # [n_samples, n_output] 或 [seq_len, n_samples] 取决于模型
                # 检查维度，如果是 [seq_len, n_samples]，需要转置
                if output_np.shape[0] == self.seq_len and output_np.shape[1] == n_samples:
                    # 这是 [seq_len, n_samples] 格式，取最后一个时间步
                    output_flat = output_np[-1, :].reshape(n_samples, -1)  # [n_samples, n_output]
                else:
                    # [n_samples, n_output] 格式
                    output_flat = output_np
            else:
                # 1D或其他格式
                output_flat = output_np.reshape(n_samples, -1)
            
            return output_flat
        
        # 将背景数据展平为2D用于KernelExplainer
        bg_data_flat = bg_data_np.reshape(bg_data_np.shape[0], -1)  # [n_samples, seq_len * n_features]
        
        # 初始化解释器
        # 对于KernelExplainer，需要使用展平的2D数据
        if self.explainer_type == "KernelExplainer":
            # 对于KernelExplainer，使用较小的背景数据集以提高速度
            if bg_data_flat.shape[0] > 50:
                indices = np.random.choice(bg_data_flat.shape[0], 50, replace=False)
                bg_data_flat_small = bg_data_flat[indices]
            else:
                bg_data_flat_small = bg_data_flat
            # 使用展平的背景数据初始化
            try:
                self.explainer = shap.KernelExplainer(model_wrapper, bg_data_flat_small)
            except (TypeError, AttributeError) as e:
                # 如果失败，尝试不同的初始化方式
                warnings.warn(f"KernelExplainer初始化时遇到问题: {e}，尝试继续...")
                self.explainer = shap.KernelExplainer(model_wrapper, bg_data_flat_small)
        elif self.explainer_type == "DeepExplainer" or self.explainer_type == "Explainer":
            # 对于这些类型，尝试使用KernelExplainer作为替代（因为时序数据的问题）
            warnings.warn(
                f"对于时序数据，{self.explainer_type}可能不稳定，改用KernelExplainer"
            )
            self.explainer_type = "KernelExplainer"
            if bg_data_flat.shape[0] > 50:
                indices = np.random.choice(bg_data_flat.shape[0], 50, replace=False)
                bg_data_flat_small = bg_data_flat[indices]
            else:
                bg_data_flat_small = bg_data_flat
            self.explainer = shap.KernelExplainer(model_wrapper, bg_data_flat_small)
        else:
            raise ValueError(f"不支持的解释器类型: {self.explainer_type}")
    
    def explain(
        self,
        instances: np.ndarray,
        nsamples: int = 100,
        output_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        计算SHAP值
        
        Parameters
        ----------
        instances : np.ndarray
            要解释的实例，形状为 [n_instances, seq_len, n_features]
        nsamples : int
            采样数量（仅用于KernelExplainer）
        output_idx : int, optional
            要解释的输出索引，如果为None则解释所有输出
            
        Returns
        -------
        np.ndarray
            SHAP值，形状为 [n_instances, seq_len, n_features, n_outputs]
        """
        if isinstance(instances, torch.Tensor):
            instances = instances.cpu().numpy()
        
        # 保存原始实例形状
        original_instances_shape = instances.shape  # [n_instances, seq_len, n_features]
        n_instances = instances.shape[0]
        
        # 将实例展平为2D用于KernelExplainer
        instances_flat = instances.reshape(n_instances, -1)  # [n_instances, seq_len * n_features]
        
        # 根据解释器类型调用相应的方法
        if self.explainer_type == "KernelExplainer":
            # KernelExplainer 需要 nsamples 参数，使用展平的输入
            shap_values_flat = self.explainer.shap_values(instances_flat, nsamples=nsamples)
        elif self.explainer_type == "Explainer":
            # 新的统一API
            shap_result = self.explainer(instances_flat)
            if hasattr(shap_result, 'values'):
                shap_values_flat = shap_result.values
            else:
                shap_values_flat = shap_result
        elif self.explainer_type == "DeepExplainer":
            shap_values_flat = self.explainer.shap_values(instances_flat)
        else:
            raise ValueError(f"不支持的解释器类型: {self.explainer_type}")
        
        # 处理SHAP值的格式
        # SHAP值形状可能是：
        # - [n_instances, n_input_features] - 单个输出
        # - [n_instances, n_input_features, n_outputs] - 多个输出
        # - list of [n_instances, n_input_features] - 多个输出（列表形式）
        
        if isinstance(shap_values_flat, list):
            # 如果有多个输出（列表形式），处理每个输出
            shap_values_list = []
            for sv_flat in shap_values_flat:
                # 将展平的SHAP值重新整形为3D
                if sv_flat.ndim == 2:
                    # [n_instances, seq_len * n_features] -> [n_instances, seq_len, n_features]
                    shap_values_3d = sv_flat.reshape(n_instances, self.seq_len, self.n_features)
                else:
                    shap_values_3d = sv_flat
                shap_values_list.append(shap_values_3d)
            # 堆叠多个输出
            shap_values = np.stack(shap_values_list, axis=-1)  # [n_instances, seq_len, n_features, n_outputs]
        else:
            # 单个输出或3D数组
            if shap_values_flat.ndim == 2:
                # [n_instances, seq_len * n_features] -> [n_instances, seq_len, n_features]
                shap_values = shap_values_flat.reshape(n_instances, self.seq_len, self.n_features)
            elif shap_values_flat.ndim == 3:
                # [n_instances, n_input_features, n_outputs] = [n_instances, seq_len*n_features, n_outputs]
                # 需要reshape为 [n_instances, seq_len, n_features, n_outputs]
                n_outputs = shap_values_flat.shape[2]
                shap_values = shap_values_flat.reshape(n_instances, self.seq_len, self.n_features, n_outputs)
            else:
                shap_values = shap_values_flat
        
        # 如果指定了输出索引，只返回该输出的SHAP值
        if output_idx is not None and shap_values.ndim == 4:
            shap_values = shap_values[:, :, :, output_idx]
        elif output_idx is None and shap_values.ndim == 4 and shap_values.shape[3] == 1:
            # 如果只有一个输出，去掉输出维度
            shap_values = shap_values[:, :, :, 0]
        
        return shap_values
    
    def get_feature_importance(
        self,
        shap_values: np.ndarray,
        method: str = "mean_abs"
    ) -> np.ndarray:
        """
        计算特征重要性（聚合SHAP值）
        
        Parameters
        ----------
        shap_values : np.ndarray
            SHAP值，形状为 [n_instances, seq_len, n_features, n_outputs]
            或 [n_instances, seq_len, n_features]
        method : str
            聚合方法：
            - "mean_abs": 平均绝对SHAP值（默认）
            - "sum_abs": 绝对SHAP值之和
            - "mean": 平均SHAP值（保留符号）
            - "max_abs": 最大绝对SHAP值
        
        Returns
        -------
        np.ndarray
            特征重要性，形状为 [n_features, n_outputs] 或 [n_features]
        """
        if method == "mean_abs":
            importance = np.mean(np.abs(shap_values), axis=(0, 1))
        elif method == "sum_abs":
            importance = np.sum(np.abs(shap_values), axis=(0, 1))
        elif method == "mean":
            importance = np.mean(shap_values, axis=(0, 1))
        elif method == "max_abs":
            importance = np.max(np.abs(shap_values), axis=(0, 1))
        else:
            raise ValueError(f"不支持的聚合方法: {method}")
        
        return importance
    
    def get_temporal_importance(
        self,
        shap_values: np.ndarray,
        method: str = "mean_abs"
    ) -> np.ndarray:
        """
        计算时间步重要性（跨特征聚合SHAP值）
        
        Parameters
        ----------
        shap_values : np.ndarray
            SHAP值，形状为 [n_instances, seq_len, n_features, n_outputs]
            或 [n_instances, seq_len, n_features]
        method : str
            聚合方法，同get_feature_importance
        
        Returns
        -------
        np.ndarray
            时间步重要性，形状为 [seq_len, n_outputs] 或 [seq_len]
        """
        if method == "mean_abs":
            importance = np.mean(np.abs(shap_values), axis=(0, 2))
        elif method == "sum_abs":
            importance = np.sum(np.abs(shap_values), axis=(0, 2))
        elif method == "mean":
            importance = np.mean(shap_values, axis=(0, 2))
        elif method == "max_abs":
            importance = np.max(np.abs(shap_values), axis=(0, 2))
        else:
            raise ValueError(f"不支持的聚合方法: {method}")
        
        return importance


def analyze_lstm_with_shap(
    model: torch.nn.Module,
    background_data: np.ndarray,
    instances: np.ndarray,
    feature_names: Optional[list] = None,
    device: Optional[torch.device] = None,
    explainer_type: str = "DeepExplainer",
    save_path: Optional[str] = None
) -> Dict:
    """
    对LSTM模型进行SHAP分析的便捷函数
    
    Parameters
    ----------
    model : torch.nn.Module
        已训练的PyTorch LSTM模型
    background_data : np.ndarray
        背景数据集，形状为 [n_samples, seq_len, n_features]
    instances : np.ndarray
        要解释的实例，形状为 [n_instances, seq_len, n_features]
    feature_names : list, optional
        特征名称列表
    device : torch.device, optional
        计算设备
    explainer_type : str
        解释器类型，可选 "DeepExplainer" 或 "KernelExplainer"
    save_path : str, optional
        保存结果的路径
    
    Returns
    -------
    dict
        包含SHAP值和重要性分析结果的字典
    """
    # 初始化分析器
    analyzer = LSTMSHAPAnalyzer(
        model=model,
        background_data=background_data,
        device=device,
        explainer_type=explainer_type
    )
    
    # 计算SHAP值
    print("正在计算SHAP值...")
    shap_values = analyzer.explain(instances)
    print(f"SHAP值计算完成，形状: {shap_values.shape}")
    
    # 计算特征重要性
    feature_importance = analyzer.get_feature_importance(shap_values)
    
    # 计算时间步重要性
    temporal_importance = analyzer.get_temporal_importance(shap_values)
    
    # 准备结果
    results = {
        "shap_values": shap_values,
        "feature_importance": feature_importance,
        "temporal_importance": temporal_importance,
        "feature_names": feature_names if feature_names else [f"Feature_{i}" for i in range(feature_importance.shape[0])],
        "explainer_type": explainer_type
    }
    
    # 保存结果
    if save_path:
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"结果已保存到: {save_path}")
    
    return results

