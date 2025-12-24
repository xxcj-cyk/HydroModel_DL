"""
SHAP分析器用于LSTM模型的特征重要性分析

支持DeepExplainer，计算每个时间步每个特征的SHAP值贡献度
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import random
import warnings

# 过滤sklearn的ConvergenceWarning（KernelExplainer内部使用LARS算法时会产生这些警告）
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.linear_model._least_angle')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("警告: shap库未安装，请使用 pip install shap 安装")


class LSTMSHAPAnalyzer:
    """
    LSTM模型的SHAP分析器
    
    用于计算LSTM模型每个时间步每个特征的SHAP值贡献度
    """
    
    def __init__(
        self,
        model: nn.Module,
        background_data: np.ndarray,
        device: torch.device,
        explainer_type: str = "KernelExplainer",
        which_first_tensor: str = "sequence",
    ):
        """
        初始化SHAP分析器
        
        Parameters
        ----------
        model : nn.Module
            训练好的LSTM模型
        background_data : np.ndarray
            背景数据，形状为 (n_samples, seq_length, n_features)
        device : torch.device
            计算设备 (CPU或GPU)
        explainer_type : str
            解释器类型，可选：
            - "GradientExplainer": 专门为PyTorch设计（推荐）
            - "Explainer": 使用SHAP新API，自动选择解释器（推荐）
            - "DeepExplainer": 旧API，可能尝试导入TensorFlow（不推荐用于PyTorch）
            - "KernelExplainer": 通用解释器，较慢但稳定
        which_first_tensor : str
            张量格式，"sequence" 表示 (seq_len, batch, features)，"batch" 表示 (batch, seq_len, features)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("shap库未安装，请使用 pip install shap 安装")
        
        self.model = model
        self.device = device
        self.explainer_type = explainer_type
        self.which_first_tensor = which_first_tensor
        self.seq_first = which_first_tensor == "sequence"
        
        # 将模型设置为评估模式
        self.model.eval()
        
        # 处理DataParallel模型
        if isinstance(model, nn.DataParallel):
            self.model_core = model.module
        else:
            self.model_core = model
        
        # 准备背景数据
        self.background_data = self._prepare_background_data(background_data)
        
        # 创建SHAP解释器
        self.explainer = self._create_explainer()
    
    def _prepare_background_data(self, background_data: np.ndarray) -> torch.Tensor:
        """
        准备背景数据，转换为模型需要的格式
        
        Parameters
        ----------
        background_data : np.ndarray
            背景数据，形状为 (n_samples, seq_length, n_features)
        
        Returns
        -------
        torch.Tensor
            转换后的背景数据
        """
        # 转换为torch tensor
        bg_tensor = torch.from_numpy(background_data).float()
        
        # 根据模型输入格式转换
        if self.seq_first:
            # 如果是sequence first，需要转换为 (seq_len, batch, features)
            # 背景数据是 (n_samples, seq_length, n_features)
            # 需要转换为 (seq_length, n_samples, n_features)
            bg_tensor = bg_tensor.permute(1, 0, 2)
        
        return bg_tensor.to(self.device)
    
    def _create_explainer(self):
        """
        创建SHAP解释器
        
        Returns
        -------
        shap.Explainer
            SHAP解释器对象
        """
        # 定义模型包装函数，用于SHAP
        # 对于时序模型，我们返回所有时间步的输出，展平以便SHAP计算
        def model_wrapper(x):
            """
            模型包装函数，将输入转换为模型需要的格式
            
            Parameters
            ----------
            x : torch.Tensor
                输入数据，形状为 (batch, seq_len, features) 或 (seq_len, batch, features)
            
            Returns
            -------
            torch.Tensor
                模型输出，形状为 (batch, seq_len * output_size)
                这样SHAP可以为每个时间步的每个特征计算贡献
            """
            self.model.eval()
            with torch.no_grad():
                # 确保输入是torch tensor
                if isinstance(x, np.ndarray):
                    x_tensor = torch.from_numpy(x).float()
                else:
                    x_tensor = x.float()
                
                # 处理输入形状
                if x_tensor.ndim == 2:
                    # 如果是展平的数据，需要reshape
                    # 假设背景数据的形状
                    n_samples = x_tensor.shape[0]
                    seq_length = self.background_data.shape[0] if self.seq_first else self.background_data.shape[1]
                    n_features = self.background_data.shape[2] if self.seq_first else self.background_data.shape[2]
                    x_tensor = x_tensor.view(n_samples, seq_length, n_features)
                
                # 转换为模型需要的格式
                if self.seq_first:
                    # sequence first: (seq_len, batch, features)
                    x_tensor = x_tensor.permute(1, 0, 2)
                else:
                    # batch first: (batch, seq_len, features)
                    pass
                
                x_tensor = x_tensor.to(self.device)
                
                # 前向传播
                output = self.model_core(x_tensor)
                
                # 处理输出格式
                if self.seq_first:
                    # 如果是sequence first，输出也是 (seq_len, batch, output_size)
                    # 转换为 (batch, seq_len, output_size)
                    output = output.permute(1, 0, 2)
                
                # 展平输出: (batch, seq_len * output_size)
                # 这样SHAP可以为每个时间步的输出计算每个时间步输入的贡献
                batch_size = output.shape[0]
                output_flat = output.reshape(batch_size, -1)
                
                # 返回numpy数组，而不是Tensor（某些explainer需要numpy数组）
                # 使用detach()确保没有gradient信息
                return output_flat.detach().cpu().numpy()
        
        # 准备背景数据用于SHAP
        bg_for_shap = self.background_data
        
        # 准备KernelExplainer的备选方案（不依赖TensorFlow）
        def get_kernel_explainer():
            """获取KernelExplainer作为备选方案"""
            bg_numpy = self.background_data.detach().cpu().numpy()
            if self.seq_first:
                bg_numpy = bg_numpy.transpose(1, 0, 2)
            bg_flat = bg_numpy.reshape(bg_numpy.shape[0], -1)
            
            def model_wrapper_numpy(x):
                self.model.eval()
                with torch.no_grad():
                    # 确保输入是numpy数组
                    if isinstance(x, torch.Tensor):
                        x_np = x.detach().cpu().numpy()
                    else:
                        x_np = np.asarray(x)
                    
                    x_tensor = torch.from_numpy(x_np).float()
                    
                    # 获取预期的形状信息
                    if self.seq_first:
                        seq_length = self.background_data.shape[0]
                        n_features = self.background_data.shape[2]
                    else:
                        seq_length = self.background_data.shape[1]
                        n_features = self.background_data.shape[2]
                    
                    expected_flat_size = seq_length * n_features
                    original_shape = x_np.shape
                    
                    # 处理输入形状（与主分支中的逻辑一致）
                    if x_tensor.ndim == 2:
                        if x_tensor.shape[1] == expected_flat_size:
                            n_samples = x_tensor.shape[0]
                        else:
                            x_flat = x_tensor.flatten()
                            total_elements = x_flat.numel()
                            n_samples = total_elements // expected_flat_size
                            remainder = total_elements % expected_flat_size
                            if remainder != 0:
                                if total_elements % n_features == 0:
                                    n_samples_single_step = total_elements // n_features
                                    raise ValueError(
                                        f"输入大小不匹配: 输入形状 {original_shape}, 总元素数 {total_elements}. "
                                        f"期望每样本 {expected_flat_size} (seq_length={seq_length} * n_features={n_features}), "
                                        f"但得到的数据可能是 {n_samples_single_step} 个样本，每个样本 {n_features} 个特征 "
                                        f"(只包含最后一个时间步?)."
                                    )
                                else:
                                    raise ValueError(
                                        f"输入大小不匹配: 输入形状 {original_shape}, 总元素数 {total_elements} 不能被 "
                                        f"{expected_flat_size} 整除."
                                    )
                            x_tensor = x_flat.view(n_samples, expected_flat_size)
                    elif x_tensor.ndim == 1:
                        if x_tensor.shape[0] != expected_flat_size:
                            raise ValueError(
                                f"输入大小不匹配: 期望 {expected_flat_size}, 但得到 {x_tensor.shape[0]}. "
                                f"输入形状: {original_shape}"
                            )
                        x_tensor = x_tensor.unsqueeze(0)
                        n_samples = 1
                    else:
                        x_flat = x_tensor.flatten()
                        total_elements = x_flat.numel()
                        n_samples = total_elements // expected_flat_size
                        remainder = total_elements % expected_flat_size
                        if remainder != 0:
                            raise ValueError(
                                f"输入大小不匹配: 输入形状 {original_shape}, 总元素数 {total_elements} 不能被 "
                                f"{expected_flat_size} 整除."
                            )
                        x_tensor = x_flat.view(n_samples, expected_flat_size)
                    
                    # Reshape为 (n_samples, seq_length, n_features)
                    x_tensor = x_tensor.view(n_samples, seq_length, n_features)
                    
                    # 转换为模型需要的格式
                    if self.seq_first:
                        x_tensor = x_tensor.permute(1, 0, 2)
                    
                    x_tensor = x_tensor.to(self.device)
                    output = self.model_core(x_tensor)
                    
                    if self.seq_first:
                        output = output.permute(1, 0, 2)
                    if output.ndim == 3:
                        output = output.reshape(output.shape[0], -1)
                    # 使用detach()确保没有gradient信息
                    return output.detach().cpu().numpy()
            
            return shap.KernelExplainer(model_wrapper_numpy, bg_flat)
        
        if self.explainer_type == "Explainer":
            # 使用SHAP新API，明确指定使用KernelExplainer算法
            # 因为PermutationExplainer可能不兼容我们的模型包装函数
            try:
                # 尝试使用KernelExplainer作为算法
                bg_numpy = self.background_data.detach().cpu().numpy()
                if self.seq_first:
                    bg_numpy = bg_numpy.transpose(1, 0, 2)
                bg_flat = bg_numpy.reshape(bg_numpy.shape[0], -1)
                
                def model_wrapper_flat(x):
                    """展平输入的模型包装函数"""
                    self.model.eval()
                    with torch.no_grad():
                        x_tensor = torch.from_numpy(x).float()
                        if self.seq_first:
                            n_samples = x_tensor.shape[0]
                            seq_length = self.background_data.shape[0]
                            n_features = self.background_data.shape[2]
                            x_tensor = x_tensor.view(n_samples, seq_length, n_features)
                            x_tensor = x_tensor.permute(1, 0, 2)
                        x_tensor = x_tensor.to(self.device)
                        output = self.model_core(x_tensor)
                        if self.seq_first:
                            output = output.permute(1, 0, 2)
                        if output.ndim == 3:
                            output = output.reshape(output.shape[0], -1)
                        # 使用detach()确保没有gradient信息
                        return output.detach().cpu().numpy()
                
                # 使用KernelExplainer算法
                explainer = shap.Explainer(model_wrapper_flat, bg_flat, algorithm="permutation")
                print("使用Explainer（Kernel算法）")
            except Exception as e:
                print(f"Explainer初始化失败，回退到KernelExplainer: {e}")
                explainer = get_kernel_explainer()
                self.explainer_type = "KernelExplainer"
                print("已切换到KernelExplainer（不依赖TensorFlow）")
        
        elif self.explainer_type == "GradientExplainer":
            # GradientExplainer可能也需要TensorFlow，尝试后回退
            try:
                explainer = shap.GradientExplainer(model_wrapper, bg_for_shap)
                print("使用GradientExplainer")
            except (ImportError, ModuleNotFoundError) as e:
                if "tensorflow" in str(e).lower() or "tf" in str(e).lower():
                    print(f"GradientExplainer需要TensorFlow，自动切换到KernelExplainer: {e}")
                    explainer = get_kernel_explainer()
                    self.explainer_type = "KernelExplainer"
                    print("已切换到KernelExplainer（不依赖TensorFlow）")
                else:
                    raise RuntimeError(f"GradientExplainer初始化失败: {e}")
        
        elif self.explainer_type == "DeepExplainer":
            # DeepExplainer会尝试导入TensorFlow
            print("警告: DeepExplainer可能尝试导入TensorFlow")
            try:
                explainer = shap.DeepExplainer(model_wrapper, bg_for_shap)
                print("DeepExplainer初始化成功")
            except (ImportError, ModuleNotFoundError) as e:
                if "tensorflow" in str(e).lower() or "tf" in str(e).lower():
                    print(f"DeepExplainer需要TensorFlow，自动切换到KernelExplainer: {e}")
                    explainer = get_kernel_explainer()
                    self.explainer_type = "KernelExplainer"
                    print("已切换到KernelExplainer（不依赖TensorFlow）")
                else:
                    raise RuntimeError(f"DeepExplainer初始化失败: {e}")
        elif self.explainer_type == "KernelExplainer":
            # 对于KernelExplainer，需要展平数据
            bg_numpy = self.background_data.detach().cpu().numpy()
            if self.seq_first:
                # 从 (seq_len, batch, features) 转换为 (batch, seq_len, features)
                bg_numpy = bg_numpy.transpose(1, 0, 2)
            # 展平为 (batch, seq_len * features)
            bg_flat = bg_numpy.reshape(bg_numpy.shape[0], -1)
            
            # KernelExplainer需要numpy包装函数
            def model_wrapper_numpy(x):
                self.model.eval()
                with torch.no_grad():
                    # 确保输入是numpy数组
                    if isinstance(x, torch.Tensor):
                        x_np = x.detach().cpu().numpy()
                    else:
                        x_np = np.asarray(x)
                    
                    x_tensor = torch.from_numpy(x_np).float()
                    
                    # 获取预期的形状信息
                    if self.seq_first:
                        # 背景数据形状: (seq_len, batch, features)
                        seq_length = self.background_data.shape[0]
                        n_features = self.background_data.shape[2]
                    else:
                        # 背景数据形状: (batch, seq_len, features)
                        seq_length = self.background_data.shape[1]
                        n_features = self.background_data.shape[2]
                    
                    expected_flat_size = seq_length * n_features
                    
                    # 调试信息（仅在出错时打印）
                    original_shape = x_np.shape
                    
                    # 处理输入形状
                    # KernelExplainer传入的x应该是 (n_samples, n_features_flat) 或 (n_features_flat,)
                    original_shape = x_np.shape
                    
                    # 首先检查是否是标准的2D格式
                    if x_tensor.ndim == 2:
                        # 标准格式: (n_samples, n_features_flat)
                        if x_tensor.shape[1] == expected_flat_size:
                            # 格式正确
                            n_samples = x_tensor.shape[0]
                        else:
                            # 特征数不匹配，可能是其他格式
                            # 尝试展平并重新计算
                            x_flat = x_tensor.flatten()
                            total_elements = x_flat.numel()
                            n_samples = total_elements // expected_flat_size
                            remainder = total_elements % expected_flat_size
                            
                            if remainder != 0:
                                # 尝试检查是否是只传入了最后一个时间步
                                if total_elements % n_features == 0:
                                    # 可能是只传入了最后一个时间步的数据
                                    n_samples_single_step = total_elements // n_features
                                    raise ValueError(
                                        f"输入大小不匹配: 输入形状 {original_shape}, 总元素数 {total_elements}. "
                                        f"期望每样本 {expected_flat_size} (seq_length={seq_length} * n_features={n_features}), "
                                        f"但得到的数据可能是 {n_samples_single_step} 个样本，每个样本 {n_features} 个特征 "
                                        f"(只包含最后一个时间步?). "
                                        f"这可能意味着 KernelExplainer 传入的数据格式与背景数据格式不一致。"
                                    )
                                else:
                                    raise ValueError(
                                        f"输入大小不匹配: 输入形状 {original_shape}, 总元素数 {total_elements} 不能被 "
                                        f"{expected_flat_size} (seq_length={seq_length} * n_features={n_features}) 整除. "
                                        f"这可能意味着数据格式不正确。"
                                    )
                            x_tensor = x_flat.view(n_samples, expected_flat_size)
                    elif x_tensor.ndim == 1:
                        # 单个样本: (n_features_flat,)
                        if x_tensor.shape[0] != expected_flat_size:
                            raise ValueError(
                                f"输入大小不匹配: 期望 {expected_flat_size} (seq_length={seq_length} * n_features={n_features}), "
                                f"但得到 {x_tensor.shape[0]}. 输入形状: {original_shape}"
                            )
                        x_tensor = x_tensor.unsqueeze(0)
                        n_samples = 1
                    else:
                        # 其他形状，尝试展平
                        x_flat = x_tensor.flatten()
                        total_elements = x_flat.numel()
                        n_samples = total_elements // expected_flat_size
                        remainder = total_elements % expected_flat_size
                        
                        if remainder != 0:
                            raise ValueError(
                                f"输入大小不匹配: 输入形状 {original_shape}, 总元素数 {total_elements} 不能被 "
                                f"{expected_flat_size} (seq_length={seq_length} * n_features={n_features}) 整除. "
                                f"这可能意味着 KernelExplainer 传入的数据格式与背景数据格式不一致。"
                            )
                        x_tensor = x_flat.view(n_samples, expected_flat_size)
                    
                    # Reshape为 (n_samples, seq_length, n_features)
                    x_tensor = x_tensor.view(n_samples, seq_length, n_features)
                    
                    # 转换为模型需要的格式
                    if self.seq_first:
                        # sequence first: (seq_len, batch, features)
                        x_tensor = x_tensor.permute(1, 0, 2)
                    # else: batch first: (batch, seq_len, features) - 已经是正确格式
                    
                    x_tensor = x_tensor.to(self.device)
                    output = self.model_core(x_tensor)
                    
                    # 处理输出格式
                    if self.seq_first:
                        # 如果是sequence first，输出也是 (seq_len, batch, output_size)
                        # 转换为 (batch, seq_len, output_size)
                        output = output.permute(1, 0, 2)
                    
                    # 展平输出以便SHAP计算: (batch, seq_len * output_size)
                    if output.ndim == 3:
                        output = output.reshape(output.shape[0], -1)
                    # 使用detach()确保没有gradient信息
                    return output.detach().cpu().numpy()
            
            explainer = shap.KernelExplainer(model_wrapper_numpy, bg_flat)
        else:
            raise ValueError(f"不支持的explainer类型: {self.explainer_type}")
        
        return explainer
    
    def explain(
        self,
        instances: np.ndarray,
        nsamples: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        计算SHAP值
        
        Parameters
        ----------
        instances : np.ndarray
            要解释的实例，形状为 (n_instances, seq_length, n_features)
        nsamples : int, optional
            采样数量（仅用于KernelExplainer）
        batch_size : int, optional
            批处理大小，用于避免内存溢出。如果为None，将根据样本数和特征数自动设置
            建议值：100-500（取决于GPU内存）
        
        Returns
        -------
        np.ndarray
            SHAP值，形状为 (n_instances, seq_length, n_features, n_outputs)
        """
        # 准备实例数据
        if isinstance(instances, np.ndarray):
            instances_tensor = torch.from_numpy(instances).float()
        else:
            instances_tensor = instances.float()
        
        # 根据explainer类型计算SHAP值
        if self.explainer_type == "KernelExplainer":
            # KernelExplainer需要numpy数组，不是Tensor
            # 展平实例数据
            # 注意：instances 输入格式是 (n_samples, seq_length, n_features)，已经是batch first
            if isinstance(instances_tensor, torch.Tensor):
                instances_numpy = instances_tensor.detach().cpu().numpy()
            else:
                instances_numpy = instances_tensor
            
            # 调试信息
            print(f"原始instances形状: {instances_numpy.shape}")
            
            # instances_numpy 形状应该是 (n_samples, seq_length, n_features)
            # 确保是3D数组
            if instances_numpy.ndim != 3:
                raise ValueError(
                    f"instances应该是3D数组 (n_samples, seq_length, n_features), "
                    f"但得到形状: {instances_numpy.shape}"
                )
            
            n_samples, seq_length, n_features = instances_numpy.shape
            print(f"解析后的维度: n_samples={n_samples}, seq_length={seq_length}, n_features={n_features}")
            
            # 直接reshape为 (n_samples, seq_length * n_features)
            instances_flat = instances_numpy.reshape(n_samples, seq_length * n_features)
            
            # 计算SHAP值
            # 注意：KernelExplainer可能很慢，特别是对于大量样本
            print(f"正在使用KernelExplainer计算SHAP值（这可能需要较长时间）...")
            print(f"传入KernelExplainer的数据形状: {instances_flat.shape}")
            print(f"期望每样本特征数: {instances_flat.shape[1]} (seq_length * n_features)")
            
            # 计算量估算
            total_features = instances_flat.shape[1]  # seq_length * n_features
            n_samples = instances_flat.shape[0]
            estimated_calls = n_samples * total_features * 2  # 粗略估算
            print(f"\n⚠️  计算量估算:")
            print(f"   - 样本数: {n_samples}")
            print(f"   - 每样本特征数: {total_features}")
            print(f"   - 预计模型调用次数: ~{estimated_calls:,} (取决于nsamples参数)")
            print(f"   - 预计计算时间: 数小时到数天（取决于硬件和nsamples设置）")
            print(f"\n💡 建议: 如果计算时间过长，请考虑:")
            print(f"   1. 使用 MAX_INSTANCES_FOR_SHAP 参数限制样本数量（如100-1000）")
            print(f"   2. 设置较小的 nsamples 参数（如100-500）以减少计算量")
            print(f"   3. 分批处理样本，每次处理一部分\n")
            
            # 限制样本数量以避免计算时间过长（可选）
            # 如果样本太多，可以先测试少量样本
            max_samples_for_test = None  # 设置为None表示处理所有样本
            if max_samples_for_test is not None and instances_flat.shape[0] > max_samples_for_test:
                print(f"警告: 样本数量 {instances_flat.shape[0]} 很大，只处理前 {max_samples_for_test} 个样本进行测试")
                instances_flat = instances_flat[:max_samples_for_test]
                n_samples = instances_flat.shape[0]
            
            # 设置默认的nsamples值
            if nsamples is None:
                default_nsamples = 100
                print(f"⚠️  未指定 nsamples 参数，使用默认值 {default_nsamples} 以减少计算时间")
                print(f"   如需更高精度，可以在调用 explain() 时设置 nsamples 参数（如 500 或 1000）")
                nsamples = default_nsamples
            else:
                print(f"使用 nsamples={nsamples} 进行计算（每个样本的计算次数）")
            
            # 自动设置批处理大小以避免内存溢出
            # 根据特征数和GPU内存估算合适的批处理大小
            if batch_size is None:
                # 对于大量特征（如 14328），使用较小的批处理大小
                if total_features > 10000:
                    batch_size = 50  # 对于大量特征，使用很小的批处理
                elif total_features > 5000:
                    batch_size = 100
                elif total_features > 1000:
                    batch_size = 200
                else:
                    batch_size = 500
                
                # 如果样本数较少，不需要批处理
                if n_samples <= batch_size:
                    batch_size = None
            
            # 分批处理以避免内存溢出
            if batch_size is not None and n_samples > batch_size:
                print(f"\n📦 使用批处理模式，批大小: {batch_size}")
                print(f"   将 {n_samples} 个样本分成 {int(np.ceil(n_samples / batch_size))} 批处理\n")
                
                all_shap_values = []
                n_batches = int(np.ceil(n_samples / batch_size))
                
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, n_samples)
                    batch_instances = instances_flat[start_idx:end_idx]
                    
                    print(f"处理第 {i+1}/{n_batches} 批 (样本 {start_idx+1}-{end_idx})...")
                    
                    # 计算当前批的SHAP值
                    batch_shap_values = self.explainer.shap_values(batch_instances, nsamples=nsamples)
                    
                    # 处理SHAP值格式
                    if isinstance(batch_shap_values, list):
                        batch_shap_values = np.array(batch_shap_values)
                        if batch_shap_values.ndim == 2:
                            # 单输出: (n_instances, n_features_flat)
                            all_shap_values.append(batch_shap_values)
                        else:
                            # 多输出: (n_outputs, n_instances, n_features_flat)
                            batch_shap_values = batch_shap_values.transpose(1, 0, 2)
                            all_shap_values.append(batch_shap_values)
                    else:
                        # 确保是numpy数组
                        if not isinstance(batch_shap_values, np.ndarray):
                            batch_shap_values = np.array(batch_shap_values)
                        all_shap_values.append(batch_shap_values)
                    
                    # 打印当前批的形状用于调试
                    if isinstance(all_shap_values[-1], np.ndarray):
                        print(f"   当前批SHAP值形状: {all_shap_values[-1].shape}")
                    
                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    print(f"   第 {i+1} 批完成\n")
                
                # 合并所有批的SHAP值
                print(f"正在合并 {len(all_shap_values)} 批的SHAP值...")
                # 检查所有批的形状是否一致
                if len(all_shap_values) > 0:
                    first_shape = all_shap_values[0].shape
                    print(f"第一批形状: {first_shape}")
                    for i, batch_shap in enumerate(all_shap_values[1:], 1):
                        if batch_shap.shape != first_shape[1:]:  # 忽略第一个维度（样本数）
                            print(f"警告: 第 {i+1} 批形状 {batch_shap.shape} 与第一批形状不一致")
                
                # 根据维度确定合并的axis
                if len(all_shap_values) > 0:
                    if all_shap_values[0].ndim == 2:
                        # 2D: (n_instances, n_features_flat)，沿axis=0合并
                        shap_values = np.concatenate(all_shap_values, axis=0)
                    elif all_shap_values[0].ndim == 3:
                        # 3D: (n_instances, n_outputs, n_features_flat)，沿axis=0合并
                        shap_values = np.concatenate(all_shap_values, axis=0)
                    else:
                        raise ValueError(f"意外的SHAP值维度: {all_shap_values[0].ndim}")
                
                print(f"合并后SHAP值形状: {shap_values.shape}")
            else:
                # 不使用批处理，一次性处理所有样本
                print(f"开始计算...（这可能需要很长时间，请耐心等待）\n")
                shap_values = self.explainer.shap_values(instances_flat, nsamples=nsamples)
            
            # 处理SHAP值格式（无论是否分批处理，都需要这一步）
            if isinstance(shap_values, list):
                # 多输出情况: list of arrays, each shape (n_instances, n_features_flat)
                shap_values = np.array(shap_values)  # (n_outputs, n_instances, n_features_flat)
                shap_values = shap_values.transpose(1, 0, 2)  # (n_instances, n_outputs, n_features_flat)
            
            # 恢复原始形状
            # 注意：实际处理的样本数可能与instances_numpy不同（如果使用了max_samples_for_test）
            actual_n_instances = shap_values.shape[0]
            seq_length = instances_numpy.shape[1]
            n_features = instances_numpy.shape[2]
            
            print(f"\n准备reshape SHAP值:")
            print(f"  实际样本数: {actual_n_instances}")
            print(f"  时间步数: {seq_length}")
            print(f"  特征数: {n_features}")
            print(f"  SHAP值当前形状: {shap_values.shape}")
            print(f"  SHAP值维度数: {shap_values.ndim}")
            
            # 获取输出数量
            if shap_values.ndim == 2:
                # 单输出: (n_instances, n_features_flat)
                n_outputs = 1
                n_features_flat = shap_values.shape[1]
                expected_flat = seq_length * n_features
                
                print(f"  单输出模式")
                print(f"  展平特征数: {n_features_flat}, 期望: {expected_flat}")
                
                if n_features_flat != expected_flat:
                    raise ValueError(
                        f"SHAP值特征数不匹配: 实际 {n_features_flat}, 期望 {expected_flat} "
                        f"(seq_length={seq_length} * n_features={n_features})"
                    )
                
                shap_values = shap_values.reshape(actual_n_instances, seq_length, n_features, n_outputs)
            elif shap_values.ndim == 3:
                # KernelExplainer对于多输出模型，返回形状可能是:
                # 1. (n_instances, n_outputs, n_features_flat) - 标准格式
                # 2. (n_instances, n_features_flat, n_outputs) - KernelExplainer的特殊格式
                
                dim1, dim2, dim3 = shap_values.shape[0], shap_values.shape[1], shap_values.shape[2]
                expected_flat = seq_length * n_features
                
                print(f"  3维SHAP值，形状: {shap_values.shape}")
                
                # 判断是哪种格式
                if dim2 == expected_flat and dim3 == seq_length:
                    # 格式2: (n_instances, n_features_flat, n_outputs)
                    # 其中 n_features_flat = seq_length * n_features
                    #      n_outputs = seq_length (每个时间步一个输出)
                    print(f"  检测到KernelExplainer格式: (n_instances, n_features_flat, n_outputs)")
                    print(f"  展平特征数: {dim2}, 输出数(时间步数): {dim3}")
                    
                    # shap_values[:, i, t] 表示第i个展平特征对第t个输出的贡献
                    # 第i个展平特征对应: 时间步 i // n_features, 特征 i % n_features
                    # 我们需要提取: 每个时间步的输入特征对每个时间步输出的贡献
                    # 对于时间序列LSTM，通常我们关心: 每个时间步的输入对当前时间步输出的贡献
                    # 即: shap_values[:, t*n_features:(t+1)*n_features, t]
                    
                    # 创建结果数组: (n_instances, seq_length, n_features, 1)
                    # 只保留每个时间步的输入对当前时间步输出的贡献
                    shap_values_corrected = np.zeros((actual_n_instances, seq_length, n_features, 1))
                    for t in range(seq_length):
                        # 提取第t个时间步的输入特征对第t个输出的贡献
                        start_idx = t * n_features
                        end_idx = (t + 1) * n_features
                        shap_values_corrected[:, t, :, 0] = shap_values[:, start_idx:end_idx, t]
                    
                    shap_values = shap_values_corrected
                    n_outputs = 1
                    print(f"  已转换为: (n_instances, seq_length, n_features, n_outputs) = {shap_values.shape}")
                elif dim2 == seq_length and dim3 == expected_flat:
                    # 可能是 (n_instances, n_outputs, n_features_flat)，但n_outputs=seq_length
                    print(f"  检测到格式: (n_instances, n_outputs, n_features_flat)")
                    print(f"  输出数: {dim2}, 特征数: {dim3}")
                    n_outputs = dim2
                    shap_values = shap_values.reshape(actual_n_instances, n_outputs, seq_length, n_features)
                    shap_values = shap_values.transpose(0, 2, 3, 1)  # (n_instances, seq_length, n_features, n_outputs)
                elif dim3 == expected_flat:
                    # 标准格式: (n_instances, n_outputs, n_features_flat)
                    n_outputs = dim2
                    print(f"  标准多输出模式，输出数: {n_outputs}")
                    print(f"  展平特征数: {dim3}, 期望: {expected_flat}")
                    
                    shap_values = shap_values.reshape(actual_n_instances, n_outputs, seq_length, n_features)
                    shap_values = shap_values.transpose(0, 2, 3, 1)  # (n_instances, seq_length, n_features, n_outputs)
                else:
                    raise ValueError(
                        f"无法解析SHAP值形状 {shap_values.shape}。"
                        f"期望特征数: {expected_flat} (seq_length={seq_length} * n_features={n_features}), "
                        f"但得到: dim1={dim1}, dim2={dim2}, dim3={dim3}"
                    )
            else:
                raise ValueError(f"意外的SHAP值形状: {shap_values.shape}, 维度数: {shap_values.ndim}")
            
            print(f"  Reshape后形状: {shap_values.shape}\n")
        
        elif self.explainer_type in ["DeepExplainer", "GradientExplainer", "Explainer"] or hasattr(self.explainer, '__call__'):
            # 转换为模型需要的格式
            if self.seq_first:
                instances_for_shap = instances_tensor.permute(1, 0, 2).to(self.device)
            else:
                instances_for_shap = instances_tensor.to(self.device)
            
            # 计算SHAP值
            # shap.Explainer使用__call__方法，旧的explainer使用shap_values方法
            if hasattr(self.explainer, 'shap_values'):
                shap_values = self.explainer.shap_values(instances_for_shap)
            else:
                # 新API使用__call__
                shap_values = self.explainer(instances_for_shap)
            
            # 处理shap.Explainer返回的Explanation对象
            if hasattr(shap_values, 'values'):
                shap_values = shap_values.values
            
            # 处理SHAP值格式
            # 由于model_wrapper返回展平的输出 (batch, seq_len * output_size)
            # SHAP值也会是展平的形状 (batch, seq_len * output_size, seq_len * n_features)
            # 我们需要将其reshape为正确的形状
            
            # 获取实例的形状信息
            n_instances = instances_tensor.shape[0]
            seq_length = instances_tensor.shape[1]
            n_features = instances_tensor.shape[2]
            
            # 获取输出大小（从模型或背景数据推断）
            if hasattr(self.model_core, 'linearOut'):
                n_outputs = self.model_core.linearOut.out_features
            else:
                # 尝试从背景数据推断
                n_outputs = 1  # 默认值
            
            # 将SHAP值转换为numpy数组
            if isinstance(shap_values, torch.Tensor):
                shap_values = shap_values.detach().cpu().numpy()
            elif isinstance(shap_values, list):
                shap_values = np.array(shap_values)
            
            # 处理SHAP值形状
            if shap_values.ndim == 3:
                # (n_instances, seq_len * output_size, seq_len * n_features)
                # 需要reshape为 (n_instances, seq_len, n_features, seq_len, output_size)
                # 然后取对角线元素 (每个时间步的输入对对应时间步输出的贡献)
                shap_reshaped = shap_values.reshape(
                    n_instances, seq_length * n_outputs, seq_length, n_features
                )
                # 取对角线：每个时间步的输入对对应时间步输出的贡献
                # 形状: (n_instances, seq_length, n_features, n_outputs)
                shap_values_final = np.zeros((n_instances, seq_length, n_features, n_outputs))
                for t in range(seq_length):
                    for o in range(n_outputs):
                        output_idx = t * n_outputs + o
                        shap_values_final[:, t, :, o] = shap_reshaped[:, output_idx, t, :]
                shap_values = shap_values_final
            elif shap_values.ndim == 2:
                # (n_instances, seq_len * n_features) - 可能是单输出情况
                # 尝试reshape
                if shap_values.shape[1] == seq_length * n_features:
                    shap_values = shap_values.reshape(n_instances, seq_length, n_features, 1)
                else:
                    # 如果形状不匹配，可能需要其他处理
                    # 假设是最后一个时间步的SHAP值
                    shap_values_full = np.zeros((n_instances, seq_length, n_features, 1))
                    if shap_values.shape[1] == n_features:
                        shap_values_full[:, -1, :, 0] = shap_values
                    shap_values = shap_values_full
            else:
                # 其他情况，尝试自动处理
                # 如果已经是正确形状，直接使用
                if shap_values.ndim == 4 and shap_values.shape[1] == seq_length:
                    pass  # 已经是正确形状
                else:
                    # 无法自动处理，抛出错误
                    raise ValueError(
                        f"无法处理SHAP值形状: {shap_values.shape}, "
                        f"期望形状: (n_instances, seq_length, n_features, n_outputs)"
                    )
        
        
        return shap_values
    
    def get_feature_importance(
        self,
        shap_values: np.ndarray,
        method: str = "mean_abs"
    ) -> np.ndarray:
        """
        计算特征重要性
        
        Parameters
        ----------
        shap_values : np.ndarray
            SHAP值，形状为 (n_instances, seq_length, n_features, n_outputs)
        method : str
            聚合方法：
            - "mean_abs": 平均绝对SHAP值
            - "sum_abs": 绝对SHAP值之和
            - "max_abs": 最大绝对SHAP值
        
        Returns
        -------
        np.ndarray
            特征重要性，形状为 (n_features, n_outputs) 或 (n_features,)
        """
        if method == "mean_abs":
            importance = np.mean(np.abs(shap_values), axis=(0, 1))  # 在样本和时间步维度上平均
        elif method == "sum_abs":
            importance = np.sum(np.abs(shap_values), axis=(0, 1))  # 在样本和时间步维度上求和
        elif method == "max_abs":
            importance = np.max(np.abs(shap_values), axis=(0, 1))  # 在样本和时间步维度上取最大值
        else:
            raise ValueError(f"不支持的聚合方法: {method}")
        
        return importance
    
    def get_temporal_importance(
        self,
        shap_values: np.ndarray,
        method: str = "mean_abs"
    ) -> np.ndarray:
        """
        计算时间步重要性
        
        Parameters
        ----------
        shap_values : np.ndarray
            SHAP值，形状为 (n_instances, seq_length, n_features, n_outputs)
        method : str
            聚合方法：
            - "mean_abs": 平均绝对SHAP值
            - "sum_abs": 绝对SHAP值之和
            - "max_abs": 最大绝对SHAP值
        
        Returns
        -------
        np.ndarray
            时间步重要性，形状为 (seq_length, n_outputs) 或 (seq_length,)
        """
        if method == "mean_abs":
            importance = np.mean(np.abs(shap_values), axis=(0, 2))  # 在样本和特征维度上平均
        elif method == "sum_abs":
            importance = np.sum(np.abs(shap_values), axis=(0, 2))  # 在样本和特征维度上求和
        elif method == "max_abs":
            importance = np.max(np.abs(shap_values), axis=(0, 2))  # 在样本和特征维度上取最大值
        else:
            raise ValueError(f"不支持的聚合方法: {method}")
        
        return importance


def analyze_lstm_with_shap(
    model: nn.Module,
    background_data: np.ndarray,
    instances: np.ndarray,
    feature_names: Optional[List[str]] = None,
    device: torch.device = None,
    explainer_type: str = "KernelExplainer",
    which_first_tensor: str = "sequence",
    save_path: Optional[str] = None,
) -> Dict:
    """
    对LSTM模型进行SHAP分析的便捷函数
    
    Parameters
    ----------
    model : nn.Module
        训练好的LSTM模型
    background_data : np.ndarray
        背景数据，形状为 (n_samples, seq_length, n_features)
    instances : np.ndarray
        要解释的实例，形状为 (n_instances, seq_length, n_features)
    feature_names : List[str], optional
        特征名称列表
    device : torch.device, optional
        计算设备
    explainer_type : str
        解释器类型
    which_first_tensor : str
        张量格式
    save_path : str, optional
        保存路径
    
    Returns
    -------
    Dict
        包含SHAP值和重要性分析结果的字典
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    analyzer = LSTMSHAPAnalyzer(
        model=model,
        background_data=background_data,
        device=device,
        explainer_type=explainer_type,
        which_first_tensor=which_first_tensor,
    )
    
    # 计算SHAP值
    shap_values = analyzer.explain(instances)
    
    # 计算特征重要性
    feature_importance = analyzer.get_feature_importance(shap_values, method="mean_abs")
    
    # 计算时间步重要性
    temporal_importance = analyzer.get_temporal_importance(shap_values, method="mean_abs")
    
    # 准备特征名称
    if feature_names is None:
        n_features = shap_values.shape[2]
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    results = {
        "shap_values": shap_values,
        "feature_importance": feature_importance,
        "temporal_importance": temporal_importance,
        "feature_names": feature_names,
    }
    
    if save_path is not None:
        import pickle
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"SHAP结果已保存到: {save_path}")
    
    return results


def compute_shap_contributions_to_excel(
    shap_values: np.ndarray,
    feature_names: List[str],
    output_path: str,
    output_names: Optional[List[str]] = None,
) -> None:
    """
    将SHAP值贡献度统计结果输出到Excel文件
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP值，形状为 (n_samples, seq_length, n_features, n_outputs)
    feature_names : List[str]
        特征名称列表
    output_path : str
        输出Excel文件路径
    output_names : List[str], optional
        输出变量名称列表，如果为None则使用默认名称
    """
    n_samples, seq_length, n_features, n_outputs = shap_values.shape
    
    # 准备输出变量名称
    if output_names is None:
        output_names = [f"Output_{i}" for i in range(n_outputs)]
    
    # 创建Excel写入器，自动检测可用的引擎
    # 优先使用 openpyxl，如果不可用则尝试 xlsxwriter
    excel_engine = None
    try:
        import openpyxl
        excel_engine = 'openpyxl'
    except ImportError:
        try:
            import xlsxwriter
            excel_engine = 'xlsxwriter'
        except ImportError:
            raise ImportError(
                "需要安装 Excel 写入库。请运行以下命令之一：\n"
                "  pip install openpyxl\n"
                "  或\n"
                "  pip install xlsxwriter"
            )
    
    # 创建Excel写入器
    with pd.ExcelWriter(output_path, engine=excel_engine) as writer:
        for output_idx, output_name in enumerate(output_names):
            # 提取当前输出的SHAP值
            shap_output = shap_values[:, :, :, output_idx]  # (n_samples, seq_length, n_features)
            
            # 计算每个时段每个特征的平均贡献度（跨样本平均）
            contribution_matrix = np.mean(np.abs(shap_output), axis=0)  # (seq_length, n_features)
            
            # 创建DataFrame
            df = pd.DataFrame(
                contribution_matrix,
                index=[f"TimeStep_{t}" for t in range(seq_length)],
                columns=feature_names
            )
            
            # 保存到Excel的sheet
            sheet_name = output_name[:31]  # Excel sheet名称限制为31个字符
            df.to_excel(writer, sheet_name=sheet_name, index=True)
            
            print(f"已保存 {output_name} 的贡献度矩阵到Excel，形状: {contribution_matrix.shape}")
        
        # 创建一个汇总sheet，包含所有输出的平均贡献度
        summary_data = []
        for output_idx, output_name in enumerate(output_names):
            shap_output = shap_values[:, :, :, output_idx]
            # 计算每个特征在所有时段的总贡献度
            feature_total_contrib = np.mean(np.abs(shap_output), axis=(0, 1))  # (n_features,)
            for feat_idx, feat_name in enumerate(feature_names):
                summary_data.append({
                    "Output": output_name,
                    "Feature": feat_name,
                    "Total_Contribution": feature_total_contrib[feat_idx],
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        print(f"已保存汇总信息到Excel")
    
    print(f"所有SHAP贡献度统计已保存到: {output_path}")

