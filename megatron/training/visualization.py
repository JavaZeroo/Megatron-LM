# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Megatron-LM 计算图可视化集成模块

此模块提供将模型导出为 ONNX 格式，以便使用 Netron 进行可视化。
它可以在训练的指定步骤捕获模型并生成 ONNX 文件。

Netron 是一个强大的神经网络可视化工具，支持查看 ONNX、PyTorch 等多种格式。
- 在线版本：https://netron.app
- 桌面版本：https://github.com/lutzroeder/netron

重要：可视化使用"延迟执行"策略 - 在 forward 时保存模型信息，
在反向传播完成后（train_step 结束时）再导出 ONNX 文件，
以避免干扰训练过程中的梯度计算。

使用示例：
    # 在 pretrain_gpt.py 或其他预训练脚本中：
    
    from megatron.training.visualization import (
        setup_model_graph_visualization,
        maybe_capture_graph_for_visualization,
        finalize_visualization,
    )
    
    # 在 forward_step 中捕获计算图
    def forward_step(data_iterator, model, ...):
        output = model(...)
        maybe_capture_graph_for_visualization(model, output)
        return output, loss_func
    
    # 在 train_step 结束后生成可视化
    finalize_visualization(iteration)
    
    # 生成的 .onnx 文件可以用 Netron 打开：
    # 1. 在线：上传到 https://netron.app
    # 2. 命令行：netron model_graph.onnx
    # 3. 桌面应用：直接双击打开
"""

import os
import gc
from typing import Optional, Union, List, Tuple, Any
from functools import wraps

import torch
import torch.nn as nn

# ONNX 导出始终可用（PyTorch 内置）
HAVE_ONNX_EXPORT = True

# 检查是否安装了 onnx 库（用于优化和验证，可选）
try:
    import onnx
    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False


class MegatronGraphVisualizer:
    """
    专门为 Megatron-LM 设计的计算图可视化器。
    
    使用 ONNX 导出模型，然后可以用 Netron 查看。
    
    使用延迟执行策略：
    1. capture_graph() - 在 forward 时保存模型参数名称映射
    2. finalize() - 在反向传播完成后，导出 ONNX 文件
    
    支持：
    - 分布式训练（数据并行、张量并行、流水线并行）
    - 虚拟流水线并行
    - 各种模型包装器（DDP, Float16Module 等）
    
    生成的 ONNX 文件可以用 Netron 打开：
    - 在线：https://netron.app
    - 命令行：netron model_graph.onnx
    - 桌面应用：直接双击打开
    """
    
    _instance = None
    
    def __init__(
        self,
        output_dir: str,
        iterations_to_visualize: List[int],
        visualize_interval: Optional[int] = None,
        output_format: str = "onnx",
        include_shapes: bool = True,
        max_depth: Optional[int] = None,
        opset_version: int = 14,
    ):
        self.output_dir = output_dir
        self.iterations_to_visualize = set(iterations_to_visualize)
        self.visualize_interval = visualize_interval
        self.output_format = output_format  # 保留参数但始终使用 onnx
        self.include_shapes = include_shapes
        self.max_depth = max_depth
        self.opset_version = opset_version
        self._visualized_iterations = set()
        self._enabled = HAVE_ONNX_EXPORT
        self._current_iteration = 0
        
        # 延迟执行：保存捕获的信息
        self._pending_visualization = None
        
        # 创建输出目录
        if self._should_create_on_this_rank():
            os.makedirs(os.path.join(output_dir, "model_graphs"), exist_ok=True)
    
    @classmethod
    def get_instance(cls) -> Optional['MegatronGraphVisualizer']:
        """获取单例实例"""
        return cls._instance
    
    @classmethod
    def initialize(cls, **kwargs) -> 'MegatronGraphVisualizer':
        """初始化并返回单例实例"""
        cls._instance = cls(**kwargs)
        return cls._instance
    
    def set_current_iteration(self, iteration: int):
        """设置当前迭代步骤"""
        self._current_iteration = iteration
    
    def get_current_iteration(self) -> int:
        """获取当前迭代步骤"""
        return self._current_iteration
    
    def _get_parallel_state(self) -> dict:
        """获取并行状态信息"""
        try:
            from megatron.core import parallel_state
            
            if not parallel_state.is_initialized():
                return {
                    'global_rank': 0,
                    'dp_rank': 0,
                    'tp_rank': 0,
                    'pp_rank': 0,
                    'vp_rank': 0,
                    'world_size': 1,
                }
            
            return {
                'global_rank': torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                'dp_rank': parallel_state.get_data_parallel_rank(),
                'tp_rank': parallel_state.get_tensor_model_parallel_rank(),
                'pp_rank': parallel_state.get_pipeline_model_parallel_rank(),
                'vp_rank': parallel_state.get_virtual_pipeline_model_parallel_rank() or 0,
                'world_size': torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
                'tp_world_size': parallel_state.get_tensor_model_parallel_world_size(),
                'pp_world_size': parallel_state.get_pipeline_model_parallel_world_size(),
                'dp_world_size': parallel_state.get_data_parallel_world_size(),
            }
        except Exception:
            return {
                'global_rank': 0,
                'dp_rank': 0,
                'tp_rank': 0,
                'pp_rank': 0,
                'vp_rank': 0,
                'world_size': 1,
            }
    
    def _should_create_on_this_rank(self) -> bool:
        """判断是否应该在当前 rank 创建输出目录"""
        state = self._get_parallel_state()
        # 只在 global rank 0 创建目录
        return state['global_rank'] == 0
    
    def _should_visualize_on_this_rank(self) -> bool:
        """判断是否应该在当前 rank 执行可视化"""
        state = self._get_parallel_state()
        # 在 DP rank 0, TP rank 0 上可视化（每个 PP stage 都会生成）
        return state['dp_rank'] == 0 and state['tp_rank'] == 0
    
    def should_visualize_at_iteration(self, iteration: int) -> bool:
        """检查是否应该在指定迭代可视化"""
        if not self._enabled:
            return False
        
        if iteration in self._visualized_iterations:
            return False
        
        if iteration in self.iterations_to_visualize:
            return True
        
        if self.visualize_interval and iteration > 0:
            if iteration % self.visualize_interval == 0:
                return True
        
        return False
    
    def _get_output_path(self, iteration: int, model_chunk_id: int = 0) -> str:
        """生成输出文件路径"""
        state = self._get_parallel_state()
        
        filename = f"model_graph_iter{iteration:06d}"
        filename += f"_pp{state['pp_rank']}"
        
        if state.get('vp_rank', 0) > 0 or model_chunk_id > 0:
            filename += f"_vp{model_chunk_id}"
        
        return os.path.join(self.output_dir, "model_graphs", filename)
    
    def _unwrap_model(self, model):
        """解包模型获取底层 nn.Module"""
        unwrapped = model
        
        # 常见的包装器类型
        wrapper_attrs = ['module', 'model']
        
        max_iterations = 10  # 防止无限循环
        for _ in range(max_iterations):
            found_wrapper = False
            for attr in wrapper_attrs:
                if hasattr(unwrapped, attr):
                    unwrapped = getattr(unwrapped, attr)
                    found_wrapper = True
                    break
            if not found_wrapper:
                break
        
        return unwrapped
    
    def _extract_output_tensor(self, output) -> Optional[torch.Tensor]:
        """从模型输出中提取张量"""
        if isinstance(output, torch.Tensor):
            return output if output.grad_fn is not None else None
        
        if isinstance(output, (list, tuple)):
            for item in output:
                tensor = self._extract_output_tensor(item)
                if tensor is not None:
                    return tensor
        
        if isinstance(output, dict):
            for value in output.values():
                tensor = self._extract_output_tensor(value)
                if tensor is not None:
                    return tensor
        
        return None
    
    def capture_graph(
        self,
        model: Union[nn.Module, List[nn.Module]],
        output: Union[torch.Tensor, Tuple, List],
    ) -> bool:
        """
        捕获计算图信息以供后续可视化。
        
        此方法应在 forward_step 中调用。它只保存必要的信息，
        不会执行实际的可视化操作，以避免干扰梯度计算。
        
        Args:
            model: 模型或模型列表（用于虚拟流水线）
            output: 模型输出
        
        Returns:
            bool: 是否成功捕获
        """
        iteration = self._current_iteration
        
        if not self.should_visualize_at_iteration(iteration):
            return False
        
        if not self._should_visualize_on_this_rank():
            return False
        
        # 只保存模型引用和参数名称，不做任何可能影响计算图的操作
        models = model if isinstance(model, list) else [model]
        
        captured_info = []
        for chunk_id, model_chunk in enumerate(models):
            try:
                unwrapped = self._unwrap_model(model_chunk)
                # 只保存参数名称，不访问参数值
                param_names = list(name for name, _ in unwrapped.named_parameters())
                captured_info.append({
                    'chunk_id': chunk_id,
                    'model': model_chunk,  # 保存模型引用
                    'param_names': param_names,
                })
            except Exception:
                pass
        
        if captured_info:
            self._pending_visualization = {
                'iteration': iteration,
                'models': captured_info,
            }
            return True
        
        return False
    
    def finalize(self) -> List[str]:
        """
        完成可视化 - 在反向传播完成后调用。
        
        此方法将模型导出为 ONNX 格式，可以使用 Netron 查看。
        
        Returns:
            生成的 ONNX 文件路径列表
        """
        if self._pending_visualization is None:
            return []
        
        pending = self._pending_visualization
        self._pending_visualization = None
        
        iteration = pending['iteration']
        
        if iteration in self._visualized_iterations:
            return []
        
        state = self._get_parallel_state()
        generated_files = []
        
        for info in pending['models']:
            chunk_id = info['chunk_id']
            model_chunk = info['model']
            
            try:
                unwrapped = self._unwrap_model(model_chunk)
                
                # 获取模型配置以创建正确大小的虚拟输入
                device = next(unwrapped.parameters()).device
                dtype = next(unwrapped.parameters()).dtype
                
                # 使用虚拟输入进行 ONNX 导出
                batch_size = 1
                seq_length = 32  # 使用较小的序列长度以节省内存
                vocab_size = 151936  # 默认词汇表大小
                
                # 尝试从模型获取实际配置
                try:
                    if hasattr(unwrapped, 'config'):
                        config = unwrapped.config
                        if hasattr(config, 'vocab_size'):
                            vocab_size = config.vocab_size
                except Exception:
                    pass
                
                # 保存原始训练状态
                was_training = unwrapped.training
                unwrapped.eval()
                
                # 创建虚拟输入
                input_ids = torch.randint(
                    0, vocab_size, (batch_size, seq_length), 
                    device=device, dtype=torch.long
                )
                position_ids = torch.arange(
                    seq_length, device=device, dtype=torch.long
                ).unsqueeze(0)
                attention_mask = torch.ones(
                    batch_size, 1, seq_length, seq_length,
                    device=device, dtype=torch.bool
                )
                
                # 确定模型的输入格式
                dummy_inputs = None
                input_names = None
                dynamic_axes = None
                
                try:
                    # 尝试使用关键字参数
                    with torch.no_grad():
                        _ = unwrapped(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                        )
                    dummy_inputs = (input_ids, position_ids, attention_mask)
                    input_names = ['input_ids', 'position_ids', 'attention_mask']
                    dynamic_axes = {
                        'input_ids': {0: 'batch_size', 1: 'seq_length'},
                        'position_ids': {0: 'batch_size', 1: 'seq_length'},
                        'attention_mask': {0: 'batch_size', 2: 'seq_length', 3: 'seq_length'},
                        'output': {0: 'batch_size', 1: 'seq_length'},
                    }
                except TypeError:
                    try:
                        with torch.no_grad():
                            _ = unwrapped(input_ids, position_ids, attention_mask)
                        dummy_inputs = (input_ids, position_ids, attention_mask)
                        input_names = ['input_ids', 'position_ids', 'attention_mask']
                        dynamic_axes = {
                            'input_ids': {0: 'batch_size', 1: 'seq_length'},
                            'position_ids': {0: 'batch_size', 1: 'seq_length'},
                            'attention_mask': {0: 'batch_size'},
                            'output': {0: 'batch_size'},
                        }
                    except TypeError:
                        try:
                            with torch.no_grad():
                                _ = unwrapped(input_ids)
                            dummy_inputs = (input_ids,)
                            input_names = ['input_ids']
                            dynamic_axes = {
                                'input_ids': {0: 'batch_size', 1: 'seq_length'},
                                'output': {0: 'batch_size', 1: 'seq_length'},
                            }
                        except Exception as e:
                            self._print_rank_0(f"Could not determine model input format: {e}")
                            if was_training:
                                unwrapped.train()
                            continue
                
                # 生成输出路径
                output_path = self._get_output_path(iteration, chunk_id) + ".onnx"
                
                # 导出 ONNX
                # 注意：使用 dynamo=False 禁用新的 dynamo-based exporter，
                # 因为 Megatron-LM 的动态特性（如 RNG state、动态形状）与 dynamo 不兼容
                try:
                    # 检查 PyTorch 版本以决定使用哪种导出方式
                    torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
                    
                    export_kwargs = {
                        'input_names': input_names,
                        'output_names': ['output'],
                        'opset_version': self.opset_version,
                        'do_constant_folding': True,
                        'export_params': True,
                        'verbose': False,
                    }
                    
                    # PyTorch 2.0+ 支持 dynamo 参数，需要显式禁用
                    if torch_version >= (2, 0):
                        export_kwargs['dynamo'] = False
                    
                    # 对于复杂模型，不使用 dynamic_axes 以避免兼容性问题
                    # dynamic_axes 在 Megatron-LM 中容易引起问题
                    
                    torch.onnx.export(
                        unwrapped,
                        dummy_inputs,
                        output_path,
                        **export_kwargs,
                    )
                    
                    self._print_rank_0(
                        f"[Rank {state['global_rank']}] ONNX model exported: {output_path}\n"
                        f"  View with Netron: https://netron.app or run 'netron {output_path}'"
                    )
                    generated_files.append(output_path)
                    
                    # 可选：验证导出的模型
                    if HAVE_ONNX:
                        try:
                            onnx_model = onnx.load(output_path)
                            onnx.checker.check_model(onnx_model)
                            self._print_rank_0(f"  ONNX model validation passed")
                        except Exception as e:
                            self._print_rank_0(f"  ONNX validation warning: {e}")
                            
                except Exception as e:
                    self._print_rank_0(f"ONNX export failed: {e}")
                    # 尝试多种后备方案
                    fallback_success = False
                    
                    # 后备方案 1: 保存为 state_dict 格式（Netron 支持查看）
                    try:
                        pt_path = self._get_output_path(iteration, chunk_id) + ".pt"
                        # 保存完整模型（包括结构和权重），Netron 可以打开
                        torch.save(unwrapped, pt_path)
                        self._print_rank_0(
                            f"[Rank {state['global_rank']}] PyTorch model saved: {pt_path}\n"
                            f"  View with Netron: https://netron.app (drag and drop the .pt file)"
                        )
                        generated_files.append(pt_path)
                        fallback_success = True
                    except Exception as e2:
                        self._print_rank_0(f"PyTorch save failed: {e2}")
                        
                        # 后备方案 2: 只保存 state_dict
                        try:
                            sd_path = self._get_output_path(iteration, chunk_id) + "_state_dict.pt"
                            torch.save({
                                'model_state_dict': unwrapped.state_dict(),
                                'model_class': unwrapped.__class__.__name__,
                                'iteration': iteration,
                                'pp_rank': state['pp_rank'],
                            }, sd_path)
                            self._print_rank_0(
                                f"[Rank {state['global_rank']}] State dict saved: {sd_path}"
                            )
                            generated_files.append(sd_path)
                            fallback_success = True
                        except Exception as e3:
                            self._print_rank_0(f"State dict save failed: {e3}")
                    
                    # 后备方案 3: 保存模型结构信息为文本文件（始终尝试）
                    if not fallback_success:
                        try:
                            info_path = self._get_output_path(iteration, chunk_id) + "_model_info.txt"
                            with open(info_path, 'w') as f:
                                f.write(f"Megatron-LM Model Structure\n")
                                f.write(f"Iteration: {iteration}\n")
                                f.write(f"PP Stage: {state['pp_rank']}\n")
                                f.write(f"=" * 80 + "\n\n")
                                
                                # 模型结构
                                f.write("Model Architecture:\n")
                                f.write("-" * 40 + "\n")
                                f.write(str(unwrapped) + "\n\n")
                                
                                # 参数统计
                                f.write("Parameters:\n")
                                f.write("-" * 40 + "\n")
                                total_params = 0
                                for name, param in unwrapped.named_parameters():
                                    param_count = param.numel()
                                    total_params += param_count
                                    f.write(f"{name}: {list(param.shape)} ({param_count:,} params)\n")
                                f.write(f"\nTotal parameters: {total_params:,}\n")
                                
                            self._print_rank_0(
                                f"[Rank {state['global_rank']}] Model info saved: {info_path}\n"
                                f"  (ONNX/TorchScript export not supported for this model)"
                            )
                            generated_files.append(info_path)
                        except Exception as e4:
                            self._print_rank_0(f"All export methods failed: {e4}")
                
                # 恢复训练状态
                if was_training:
                    unwrapped.train()
                
                # 清理
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                self._print_rank_0(f"[Rank {state['global_rank']}] Error exporting model chunk {chunk_id}: {e}")
                import traceback
                traceback.print_exc()
        
        if generated_files:
            self._visualized_iterations.add(iteration)
        
        return generated_files
    
    def visualize(
        self,
        model: Union[nn.Module, List[nn.Module]],
        output: Union[torch.Tensor, Tuple, List],
        iteration: int,
    ) -> List[str]:
        """
        [已废弃] 直接执行可视化 - 可能会干扰训练。
        
        请改用 capture_graph() + finalize() 的延迟执行模式。
        
        此方法保留用于向后兼容，但现在内部使用延迟执行。
        """
        self._current_iteration = iteration
        self.capture_graph(model, output)
        return self.finalize()
    
    def _print_rank_0(self, message: str):
        """只在 rank 0 打印消息"""
        state = self._get_parallel_state()
        if state['global_rank'] == 0:
            print(message)
    
    def visualize_with_dummy_forward(
        self,
        model: Union[nn.Module, List[nn.Module]],
        iteration: int,
        batch_size: int = 1,
        seq_length: int = 128,
        vocab_size: int = 50304,
    ) -> List[str]:
        """
        使用虚拟输入执行一次前向传播并可视化。
        
        这是一种替代方法，当无法从训练循环获取实际输出时使用。
        
        Args:
            model: 模型或模型列表
            iteration: 当前迭代
            batch_size: 批量大小
            seq_length: 序列长度
            vocab_size: 词汇表大小
        
        Returns:
            生成的文件路径列表
        """
        if not self.should_visualize_at_iteration(iteration):
            return []
        
        if not self._should_visualize_on_this_rank():
            self._visualized_iterations.add(iteration)
            return []
        
        state = self._get_parallel_state()
        generated_files = []
        
        models = model if isinstance(model, list) else [model]
        
        for chunk_id, model_chunk in enumerate(models):
            try:
                unwrapped = self._unwrap_model(model_chunk)
                device = next(unwrapped.parameters()).device
                
                # 创建虚拟输入
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
                position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
                attention_mask = torch.ones(batch_size, seq_length, device=device)
                
                # 临时设置为评估模式并执行前向传播
                was_training = unwrapped.training
                unwrapped.eval()
                
                with torch.enable_grad():
                    # 尝试不同的前向传播签名
                    try:
                        output = unwrapped(input_ids, position_ids, attention_mask)
                    except TypeError:
                        try:
                            output = unwrapped(input_ids)
                        except Exception:
                            self._print_rank_0(f"Could not perform forward pass for visualization")
                            continue
                
                if was_training:
                    unwrapped.train()
                
                # 可视化
                files = self.visualize(model_chunk, output, iteration)
                generated_files.extend(files)
                
            except Exception as e:
                self._print_rank_0(f"[Rank {state['global_rank']}] Error in dummy forward: {e}")
                import traceback
                traceback.print_exc()
        
        return generated_files


def setup_model_graph_visualization(args) -> Optional[MegatronGraphVisualizer]:
    """
    从 Megatron 参数设置可视化。
    
    生成的 ONNX 文件可以使用 Netron 查看：
    - 在线：https://netron.app
    - 桌面应用：https://github.com/lutzroeder/netron
    - 命令行：pip install netron && netron model.onnx
    
    Args:
        args: Megatron 命令行参数
    
    Returns:
        MegatronGraphVisualizer 实例，如果未启用则返回 None
    """
    if not getattr(args, 'visualize_model_graph', False):
        return None
    
    # ONNX 导出使用 PyTorch 内置功能，无需额外依赖
    # 可选安装 onnx 库用于验证：pip install onnx
    
    # 解析迭代步骤
    iterations_str = getattr(args, 'visualize_graph_iterations', "1")
    iterations = [int(x.strip()) for x in iterations_str.split(',') if x.strip()]
    
    # 确定输出目录
    output_dir = getattr(args, 'visualize_graph_output_dir', None)
    if output_dir is None:
        output_dir = getattr(args, 'save', None)
    if output_dir is None:
        output_dir = "./checkpoints"
    
    return MegatronGraphVisualizer.initialize(
        output_dir=output_dir,
        iterations_to_visualize=iterations,
        visualize_interval=getattr(args, 'visualize_graph_interval', None),
        output_format='onnx',  # 始终使用 ONNX 格式
        include_shapes=True,
        opset_version=getattr(args, 'visualize_graph_opset_version', 14),
    )


def maybe_capture_graph_for_visualization(
    model: Union[nn.Module, List[nn.Module]],
    output: Union[torch.Tensor, Tuple, List],
) -> bool:
    """
    [推荐] 在 forward_step 中捕获计算图信息。
    
    此函数只保存必要信息，不执行实际可视化，以避免干扰梯度计算。
    实际可视化会在 finalize_visualization() 中执行。
    
    Args:
        model: 模型
        output: 模型输出
    
    Returns:
        bool: 是否成功捕获
    """
    visualizer = MegatronGraphVisualizer.get_instance()
    if visualizer is None:
        return False
    
    return visualizer.capture_graph(model, output)


def finalize_visualization() -> List[str]:
    """
    [推荐] 在 train_step 结束后完成可视化。
    
    此函数应在反向传播完成后调用，它会使用虚拟前向传播
    生成计算图，避免与训练过程产生任何冲突。
    
    Returns:
        生成的文件路径列表
    """
    visualizer = MegatronGraphVisualizer.get_instance()
    if visualizer is None:
        return []
    
    return visualizer.finalize()


def maybe_visualize_model_graph(
    model: Union[nn.Module, List[nn.Module]],
    output: Union[torch.Tensor, Tuple, List],
    iteration: int,
) -> List[str]:
    """
    [已废弃] 在训练中直接执行可视化。
    
    警告：此函数可能会干扰梯度计算。
    请改用 maybe_capture_graph_for_visualization() + finalize_visualization()。
    
    此函数现在内部使用延迟执行模式。
    
    Args:
        model: 模型
        output: 模型输出
        iteration: 当前迭代
    
    Returns:
        生成的文件路径列表
    """
    visualizer = MegatronGraphVisualizer.get_instance()
    if visualizer is None:
        return []
    
    # 使用延迟执行模式
    visualizer.set_current_iteration(iteration)
    visualizer.capture_graph(model, output)
    return visualizer.finalize()


def get_visualizer() -> Optional[MegatronGraphVisualizer]:
    """获取全局可视化器实例"""
    return MegatronGraphVisualizer.get_instance()


def create_visualization_forward_step_wrapper(forward_step_func, get_iteration_func):
    """
    创建一个包装器，用于在 forward_step 中自动捕获计算图。
    
    注意：此包装器只执行捕获，不执行可视化。
    需要在 train_step 结束后调用 finalize_visualization()。
    
    Args:
        forward_step_func: 原始的 forward_step 函数
        get_iteration_func: 获取当前迭代步骤的函数
    
    Returns:
        包装后的 forward_step 函数
    """
    @wraps(forward_step_func)
    def wrapped_forward_step(data_iterator, model, *args, **kwargs):
        output_tensor, loss_func = forward_step_func(data_iterator, model, *args, **kwargs)
        
        # 只捕获，不执行可视化
        try:
            maybe_capture_graph_for_visualization(model, output_tensor)
        except Exception:
            # 捕获失败不应影响训练
            pass
        
        return output_tensor, loss_func
    
    return wrapped_forward_step
