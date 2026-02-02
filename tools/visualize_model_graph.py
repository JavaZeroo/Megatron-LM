# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Megatron-LM 模型计算图可视化工具

使用 torchviz 可视化 Megatron-LM 模型的计算图，支持分布式训练环境。
该模块可以集成到 Megatron-LM 的训练流程中，在指定的迭代步骤生成计算图。

使用方式：
1. 作为独立模块导入到训练脚本中
2. 通过命令行参数 --visualize-model-graph 启用可视化
"""

import os
import sys
from typing import Optional, Union, List
from functools import partial

import torch
import torch.nn as nn

# 尝试导入 torchviz
try:
    from torchviz import make_dot
    HAVE_TORCHVIZ = True
except ImportError:
    HAVE_TORCHVIZ = False
    print("Warning: torchviz not installed. Run 'pip install torchviz' to enable model graph visualization.")


def get_rank_info():
    """
    获取当前进程的分布式训练 rank 信息。
    
    Returns:
        dict: 包含各种并行维度 rank 信息的字典
    """
    rank_info = {
        'global_rank': 0,
        'data_parallel_rank': 0,
        'tensor_parallel_rank': 0,
        'pipeline_parallel_rank': 0,
        'world_size': 1,
    }
    
    try:
        from megatron.core import parallel_state
        
        if parallel_state.is_initialized():
            rank_info['global_rank'] = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            rank_info['data_parallel_rank'] = parallel_state.get_data_parallel_rank()
            rank_info['tensor_parallel_rank'] = parallel_state.get_tensor_model_parallel_rank()
            rank_info['pipeline_parallel_rank'] = parallel_state.get_pipeline_model_parallel_rank()
            rank_info['world_size'] = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    except Exception as e:
        # 如果无法获取并行状态，使用默认值
        pass
    
    return rank_info


def is_main_rank_for_visualization():
    """
    检查当前进程是否应该执行可视化。
    
    为了避免重复生成相同的图，只在以下情况下可视化：
    - 单机单卡训练
    - 分布式训练中，只在 data_parallel_rank=0 且 tensor_parallel_rank=0 的进程上执行
    - 对于 pipeline 并行，每个 pipeline stage 都会生成各自的图
    
    Returns:
        bool: 如果当前进程应该执行可视化，返回 True
    """
    rank_info = get_rank_info()
    
    # 只在 DP rank 0 和 TP rank 0 上可视化
    return (rank_info['data_parallel_rank'] == 0 and 
            rank_info['tensor_parallel_rank'] == 0)


def get_output_path(base_dir: str, iteration: int, suffix: str = "") -> str:
    """
    生成输出文件路径，包含分布式 rank 信息。
    
    Args:
        base_dir: 基础输出目录
        iteration: 当前训练迭代次数
        suffix: 文件名后缀
    
    Returns:
        str: 完整的输出文件路径（不含扩展名）
    """
    rank_info = get_rank_info()
    
    # 创建输出目录
    graph_dir = os.path.join(base_dir, "model_graphs")
    os.makedirs(graph_dir, exist_ok=True)
    
    # 生成文件名，包含 rank 信息
    filename = f"model_graph_iter{iteration}"
    filename += f"_pp{rank_info['pipeline_parallel_rank']}"
    
    if suffix:
        filename += f"_{suffix}"
    
    return os.path.join(graph_dir, filename)


def unwrap_model_for_visualization(model):
    """
    解包模型以获取底层的 nn.Module。
    
    Megatron-LM 中的模型可能被多层包装（DDP, Float16Module 等），
    此函数递归解包以获取实际的模型。
    
    Args:
        model: 可能被包装的模型
    
    Returns:
        nn.Module: 解包后的模型
    """
    from megatron.training.utils import unwrap_model as megatron_unwrap_model
    
    try:
        return megatron_unwrap_model(model)
    except:
        # 手动解包
        unwrapped = model
        while hasattr(unwrapped, 'module'):
            unwrapped = unwrapped.module
        return unwrapped


def visualize_model_graph(
    model: Union[nn.Module, List[nn.Module]],
    output_tensor: torch.Tensor,
    output_path: str,
    format: str = "pdf",
    show_attrs: bool = False,
    show_saved: bool = False,
    max_attr_chars: int = 50,
) -> Optional[str]:
    """
    使用 torchviz 可视化模型的计算图。
    
    Args:
        model: 要可视化的模型（或模型列表）
        output_tensor: 模型的输出张量（用于反向追溯计算图）
        output_path: 输出文件路径（不含扩展名）
        format: 输出格式 ("pdf", "png", "svg" 等)
        show_attrs: 是否显示节点属性
        show_saved: 是否显示保存的张量
        max_attr_chars: 属性字符串的最大长度
    
    Returns:
        str: 生成的文件路径，如果失败则返回 None
    """
    if not HAVE_TORCHVIZ:
        print("Error: torchviz not available. Please install it with 'pip install torchviz'")
        return None
    
    if not is_main_rank_for_visualization():
        return None
    
    try:
        # 收集所有模型的参数
        params = {}
        if isinstance(model, list):
            for i, m in enumerate(model):
                unwrapped = unwrap_model_for_visualization(m)
                for name, param in unwrapped.named_parameters():
                    params[f"model_{i}.{name}"] = param
        else:
            unwrapped = unwrap_model_for_visualization(model)
            for name, param in unwrapped.named_parameters():
                params[name] = param
        
        # 处理输出张量
        if isinstance(output_tensor, (list, tuple)):
            # 如果是列表，取第一个非 None 的张量
            for t in output_tensor:
                if t is not None and isinstance(t, torch.Tensor) and t.grad_fn is not None:
                    output_tensor = t
                    break
        
        if not isinstance(output_tensor, torch.Tensor):
            print(f"Warning: output_tensor is not a tensor, got {type(output_tensor)}")
            return None
        
        if output_tensor.grad_fn is None:
            print("Warning: output_tensor has no grad_fn. Ensure the forward pass was done with gradient tracking.")
            return None
        
        # 生成计算图
        dot = make_dot(
            output_tensor,
            params=params,
            show_attrs=show_attrs,
            show_saved=show_saved,
            max_attr_chars=max_attr_chars,
        )
        
        # 设置图形属性以提高可读性
        dot.attr(rankdir='TB')  # Top to Bottom 布局
        dot.attr('node', fontsize='10')
        dot.attr('edge', fontsize='8')
        
        # 渲染并保存
        output_file = dot.render(output_path, format=format, cleanup=True)
        
        rank_info = get_rank_info()
        print(f"[Rank {rank_info['global_rank']}] Model graph saved to: {output_file}")
        
        return output_file
        
    except Exception as e:
        rank_info = get_rank_info()
        print(f"[Rank {rank_info['global_rank']}] Error generating model graph: {e}")
        import traceback
        traceback.print_exc()
        return None


class ModelGraphVisualizer:
    """
    模型计算图可视化器类。
    
    可以集成到 Megatron-LM 训练流程中，在指定的迭代步骤自动生成计算图。
    """
    
    def __init__(
        self,
        output_dir: str,
        visualize_at_iterations: Optional[List[int]] = None,
        visualize_interval: Optional[int] = None,
        format: str = "pdf",
        enabled: bool = True,
    ):
        """
        初始化可视化器。
        
        Args:
            output_dir: 输出目录
            visualize_at_iterations: 在指定的迭代步骤生成图（例如 [1, 100, 1000]）
            visualize_interval: 每隔多少步生成一次图
            format: 输出格式
            enabled: 是否启用可视化
        """
        self.output_dir = output_dir
        self.visualize_at_iterations = visualize_at_iterations or [1]  # 默认在第一次迭代生成
        self.visualize_interval = visualize_interval
        self.format = format
        self.enabled = enabled and HAVE_TORCHVIZ
        self._generated_iterations = set()
        
        if not HAVE_TORCHVIZ and enabled:
            print("Warning: torchviz not installed. Model graph visualization is disabled.")
    
    def should_visualize(self, iteration: int) -> bool:
        """
        检查是否应该在当前迭代生成可视化图。
        
        Args:
            iteration: 当前迭代步骤
        
        Returns:
            bool: 如果应该生成，返回 True
        """
        if not self.enabled:
            return False
        
        if iteration in self._generated_iterations:
            return False
        
        # 检查是否在指定的迭代步骤
        if iteration in self.visualize_at_iterations:
            return True
        
        # 检查是否满足间隔条件
        if self.visualize_interval and iteration > 0 and iteration % self.visualize_interval == 0:
            return True
        
        return False
    
    def visualize(
        self,
        model: Union[nn.Module, List[nn.Module]],
        output_tensor: torch.Tensor,
        iteration: int,
        suffix: str = "",
    ) -> Optional[str]:
        """
        执行可视化。
        
        Args:
            model: 模型或模型列表
            output_tensor: 模型输出张量
            iteration: 当前迭代步骤
            suffix: 文件名后缀
        
        Returns:
            str: 生成的文件路径，如果未生成则返回 None
        """
        if not self.should_visualize(iteration):
            return None
        
        output_path = get_output_path(self.output_dir, iteration, suffix)
        result = visualize_model_graph(
            model=model,
            output_tensor=output_tensor,
            output_path=output_path,
            format=self.format,
        )
        
        if result:
            self._generated_iterations.add(iteration)
        
        return result


# 全局可视化器实例
_GLOBAL_VISUALIZER: Optional[ModelGraphVisualizer] = None


def get_model_graph_visualizer() -> Optional[ModelGraphVisualizer]:
    """获取全局可视化器实例。"""
    return _GLOBAL_VISUALIZER


def init_model_graph_visualizer(
    output_dir: str,
    visualize_at_iterations: Optional[List[int]] = None,
    visualize_interval: Optional[int] = None,
    format: str = "pdf",
    enabled: bool = True,
) -> ModelGraphVisualizer:
    """
    初始化全局可视化器。
    
    Args:
        output_dir: 输出目录
        visualize_at_iterations: 在指定的迭代步骤生成图
        visualize_interval: 每隔多少步生成一次图
        format: 输出格式
        enabled: 是否启用可视化
    
    Returns:
        ModelGraphVisualizer: 可视化器实例
    """
    global _GLOBAL_VISUALIZER
    _GLOBAL_VISUALIZER = ModelGraphVisualizer(
        output_dir=output_dir,
        visualize_at_iterations=visualize_at_iterations,
        visualize_interval=visualize_interval,
        format=format,
        enabled=enabled,
    )
    return _GLOBAL_VISUALIZER


def add_visualization_args(parser):
    """
    添加可视化相关的命令行参数。
    
    Args:
        parser: argparse.ArgumentParser 实例
    """
    group = parser.add_argument_group(title='Model Graph Visualization')
    
    group.add_argument(
        '--visualize-model-graph',
        action='store_true',
        default=False,
        help='Enable model computation graph visualization using torchviz.'
    )
    group.add_argument(
        '--visualize-graph-iterations',
        type=str,
        default="1",
        help='Comma-separated list of iterations at which to generate model graphs. '
             'Default is "1" (only the first iteration).'
    )
    group.add_argument(
        '--visualize-graph-interval',
        type=int,
        default=None,
        help='Generate model graph every N iterations. If set, this is in addition to '
             '--visualize-graph-iterations.'
    )
    group.add_argument(
        '--visualize-graph-format',
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help='Output format for the model graph visualization.'
    )
    group.add_argument(
        '--visualize-graph-output-dir',
        type=str,
        default=None,
        help='Output directory for model graphs. Defaults to the save directory.'
    )
    
    return parser


def setup_visualization_from_args(args):
    """
    从命令行参数设置可视化。
    
    Args:
        args: 解析后的命令行参数
    
    Returns:
        ModelGraphVisualizer: 如果启用则返回可视化器，否则返回 None
    """
    if not getattr(args, 'visualize_model_graph', False):
        return None
    
    # 解析迭代步骤列表
    iterations_str = getattr(args, 'visualize_graph_iterations', "1")
    iterations = [int(x.strip()) for x in iterations_str.split(',') if x.strip()]
    
    # 确定输出目录
    output_dir = getattr(args, 'visualize_graph_output_dir', None)
    if output_dir is None:
        output_dir = getattr(args, 'save', None)
    if output_dir is None:
        output_dir = "./model_graphs"
    
    return init_model_graph_visualizer(
        output_dir=output_dir,
        visualize_at_iterations=iterations,
        visualize_interval=getattr(args, 'visualize_graph_interval', None),
        format=getattr(args, 'visualize_graph_format', 'pdf'),
        enabled=True,
    )


def visualize_in_training_step(
    model: Union[nn.Module, List[nn.Module]],
    output_tensor: torch.Tensor,
    iteration: int,
) -> Optional[str]:
    """
    在训练步骤中调用的便捷函数。
    
    此函数会检查全局可视化器是否已初始化，如果是则尝试生成可视化图。
    
    Args:
        model: 模型或模型列表
        output_tensor: 模型输出张量
        iteration: 当前迭代步骤
    
    Returns:
        str: 生成的文件路径，如果未生成则返回 None
    """
    visualizer = get_model_graph_visualizer()
    if visualizer is None:
        return None
    
    return visualizer.visualize(model, output_tensor, iteration)


if __name__ == "__main__":
    # 简单的测试代码
    print("Testing model graph visualization module...")
    
    if not HAVE_TORCHVIZ:
        print("torchviz not installed. Please run: pip install torchviz")
        sys.exit(1)
    
    # 创建一个简单的测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(20, 5)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    model = SimpleModel()
    x = torch.randn(2, 10)
    y = model(x)
    
    # 测试可视化
    output_path = visualize_model_graph(
        model=model,
        output_tensor=y,
        output_path="./test_model_graph",
        format="pdf",
    )
    
    if output_path:
        print(f"Test successful! Graph saved to: {output_path}")
    else:
        print("Test failed or visualization not executed on this rank.")
