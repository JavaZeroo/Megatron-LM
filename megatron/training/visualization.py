# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Megatron-LM 计算图可视化集成模块

此模块提供模型结构可视化功能，生成 Graphviz DOT 格式的图形文件。
由于 Megatron-LM 的分布式特性，无法直接导出 ONNX/TorchScript，
因此使用基于模块层次结构的可视化方式。

生成的文件：
- .dot 文件：Graphviz 格式，可以用在线查看器或 graphviz 工具查看
- .svg 文件：矢量图形（需要安装 graphviz 库）
- _stats.txt 文件：模型参数统计信息

查看方式：
- 在线 Graphviz 查看器：https://dreampuf.github.io/GraphvizOnline/
- 安装 graphviz：pip install graphviz
- SVG 文件可直接用浏览器打开

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


class ModuleFlowAnalyzer:
    """
    通过静态分析模块结构来推断数据流。
    
    不使用 hooks（避免干扰 pipeline parallel），
    而是根据模块的连接关系和类型来推断数据流向。
    """
    
    def __init__(self):
        self.modules_info = {}  # name -> {class, params, children, ...}
        self.data_flow = []     # [(src, dst, info), ...]
        
    def analyze(self, model: nn.Module) -> dict:
        """
        分析模型结构，返回数据流信息。
        """
        self.modules_info.clear()
        self.data_flow.clear()
        
        # 收集所有模块信息
        self._collect_modules(model, "")
        
        # 根据模块结构推断数据流
        self._infer_data_flow(model)
        
        return {
            'modules': self.modules_info,
            'data_flow': self.data_flow,
        }
    
    def _collect_modules(self, module: nn.Module, prefix: str):
        """递归收集模块信息"""
        name = prefix if prefix else module.__class__.__name__
        
        # 收集参数信息
        params = list(module.parameters(recurse=False))
        param_count = sum(p.numel() for p in params)
        
        # 获取模块属性
        info = {
            'class': module.__class__.__name__,
            'params': param_count,
            'children': [],
            'depth': prefix.count('.') if prefix else 0,
        }
        
        # 收集维度信息
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            info['dims'] = f"{module.in_features} → {module.out_features}"
        elif hasattr(module, 'num_embeddings') and hasattr(module, 'embedding_dim'):
            info['dims'] = f"{module.num_embeddings} × {module.embedding_dim}"
        elif hasattr(module, 'normalized_shape'):
            info['dims'] = str(module.normalized_shape)
        elif hasattr(module, 'hidden_size'):
            info['dims'] = f"hidden={module.hidden_size}"
        
        self.modules_info[name] = info
        
        # 递归处理子模块
        for child_name, child_module in module.named_children():
            child_full_name = f"{prefix}.{child_name}" if prefix else child_name
            info['children'].append(child_full_name)
            self._collect_modules(child_module, child_full_name)
    
    def _infer_data_flow(self, model: nn.Module):
        """根据模块结构推断数据流"""
        # 对于 Megatron 模型，分析典型的结构
        # 1. Embedding -> TransformerLayers -> Output
        # 2. 每个 TransformerLayer 内部：Attention -> MLP
        
        # 找到主要的模块组
        embedding_modules = []
        transformer_layers = []
        output_modules = []
        
        for name, info in self.modules_info.items():
            class_name = info['class'].lower()
            if 'embedding' in class_name:
                embedding_modules.append(name)
            elif 'transformer' in class_name and 'layer' in class_name:
                transformer_layers.append(name)
            elif 'output' in class_name or 'lm_head' in class_name:
                output_modules.append(name)
        
        # 按深度和名称排序
        transformer_layers.sort(key=lambda x: (self.modules_info[x]['depth'], x))
        
        # 构建数据流
        prev_module = None
        
        # Embedding 层
        for name in embedding_modules:
            if prev_module:
                self.data_flow.append((prev_module, name, "forward"))
            prev_module = name
        
        # Transformer 层
        for name in transformer_layers:
            if prev_module:
                self.data_flow.append((prev_module, name, "forward"))
            prev_module = name
            
            # 分析层内部结构
            self._analyze_transformer_layer(name)
        
        # 输出层
        for name in output_modules:
            if prev_module:
                self.data_flow.append((prev_module, name, "forward"))
            prev_module = name
    
    def _analyze_transformer_layer(self, layer_name: str):
        """分析 Transformer 层的内部数据流"""
        layer_children = self.modules_info.get(layer_name, {}).get('children', [])
        
        # 常见的子模块顺序
        attention_modules = []
        mlp_modules = []
        norm_modules = []
        
        for child in layer_children:
            child_info = self.modules_info.get(child, {})
            class_name = child_info.get('class', '').lower()
            
            if 'attention' in class_name:
                attention_modules.append(child)
            elif 'mlp' in class_name:
                mlp_modules.append(child)
            elif 'norm' in class_name:
                norm_modules.append(child)
        
        # 构建层内数据流
        if norm_modules and attention_modules:
            # Pre-norm: Norm -> Attention -> Norm -> MLP
            for i, norm in enumerate(norm_modules):
                if i < len(attention_modules):
                    self.data_flow.append((norm, attention_modules[i], "forward"))
                elif i - len(attention_modules) < len(mlp_modules):
                    self.data_flow.append((norm, mlp_modules[i - len(attention_modules)], "forward"))


class MegatronGraphVisualizer:
    """
    专门为 Megatron-LM 设计的计算图可视化器。
    
    使用 forward/backward hooks 捕获真实的数据流，
    生成 Graphviz DOT 格式的计算图。
    
    使用策略：
    1. setup_hooks() - 在训练开始前注册 hooks
    2. capture_graph() - 在 forward 后记录数据流信息
    3. finalize() - 在 backward 完成后生成可视化（包含梯度流）
    
    支持：
    - 分布式训练（数据并行、张量并行、流水线并行）
    - 虚拟流水线并行
    - 各种模型包装器（DDP, Float16Module 等）
    
    生成的文件：
    - .dot 文件：https://dreampuf.github.io/GraphvizOnline/ 在线查看
    - .svg 文件：浏览器直接打开（需安装 graphviz）
    - _stats.txt：参数统计信息
    """
    
    _instance = None
    
    def __init__(
        self,
        output_dir: str,
        iterations_to_visualize: List[int],
        visualize_interval: Optional[int] = None,
        output_format: str = "dot",
        include_shapes: bool = True,
        max_depth: Optional[int] = None,
        opset_version: int = 14,
    ):
        self.output_dir = output_dir
        self.iterations_to_visualize = set(iterations_to_visualize)
        self.visualize_interval = visualize_interval
        self.output_format = output_format
        self.include_shapes = include_shapes
        self.max_depth = max_depth
        self.opset_version = opset_version
        self._visualized_iterations = set()
        self._enabled = True
        self._current_iteration = 0
        
        # 静态分析器（不使用 hooks，避免干扰 pipeline parallel）
        self._analyzers = {}  # model_id -> analysis_result
        
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
        
        models = model if isinstance(model, list) else [model]
        
        captured_info = []
        for chunk_id, model_chunk in enumerate(models):
            try:
                unwrapped = self._unwrap_model(model_chunk)
                
                # 使用静态分析（不使用 hooks，避免干扰 pipeline parallel）
                analyzer = ModuleFlowAnalyzer()
                analysis = analyzer.analyze(unwrapped)
                
                captured_info.append({
                    'chunk_id': chunk_id,
                    'model': model_chunk,
                    'unwrapped': unwrapped,
                    'analysis': analysis,
                })
            except Exception as e:
                self._print_rank_0(f"Error analyzing model: {e}")
        
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
        
        使用静态分析生成模型数据流图。
        
        生成的文件可以用以下方式查看：
        - DOT 文件：使用 Graphviz 或在线查看器 https://dreampuf.github.io/GraphvizOnline/
        - SVG 文件：直接用浏览器打开
        
        Returns:
            生成的文件路径列表
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
            unwrapped = info.get('unwrapped')
            analysis = info.get('analysis')
            
            if unwrapped is None:
                unwrapped = self._unwrap_model(info['model'])
            
            try:
                # 生成数据流可视化
                output_path = self._get_output_path(iteration, chunk_id)
                files = self._generate_dataflow_graph(
                    unwrapped, 
                    analysis,
                    output_path, 
                    iteration, 
                    state,
                    max_depth=self.max_depth,
                )
                generated_files.extend(files)
                
            except Exception as e:
                self._print_rank_0(f"[Rank {state['global_rank']}] Error exporting model chunk {chunk_id}: {e}")
                import traceback
                traceback.print_exc()
        
        if generated_files:
            self._visualized_iterations.add(iteration)
        
        return generated_files
    
    def _generate_dataflow_graph(
        self, 
        model: nn.Module, 
        analysis: Optional[dict],
        output_path: str, 
        iteration: int,
        state: dict,
        max_depth: Optional[int] = None,
    ) -> List[str]:
        """
        生成数据流可视化图。
        
        Args:
            model: PyTorch 模型
            analysis: 静态分析结果
            output_path: 输出文件路径（不含扩展名）
            iteration: 当前迭代
            state: 并行状态信息
            max_depth: 最大深度限制
            
        Returns:
            生成的文件路径列表
        """
        generated_files = []
        
        # 定义层类型的颜色
        layer_colors = {
            'Embedding': '#E8F5E9',  # 浅绿
            'VocabParallelEmbedding': '#C8E6C9',
            'Linear': '#E3F2FD',      # 浅蓝
            'ColumnParallelLinear': '#BBDEFB',
            'RowParallelLinear': '#90CAF9',
            'LayerNorm': '#FFF3E0',   # 浅橙
            'RMSNorm': '#FFE0B2',
            'Attention': '#FCE4EC',   # 浅粉
            'SelfAttention': '#F8BBD0',
            'CrossAttention': '#F48FB1',
            'MLP': '#F3E5F5',         # 浅紫
            'ParallelMLP': '#E1BEE7',
            'Dropout': '#ECEFF1',     # 浅灰
            'Transformer': '#E1F5FE', # 浅青
            'TransformerLayer': '#B3E5FC',
            'TransformerBlock': '#81D4FA',
            'GPTModel': '#E8EAF6',    # 浅靛蓝
            'default': '#FFFFFF',
        }
        
        def get_color(class_name: str) -> str:
            for key, color in layer_colors.items():
                if key.lower() in class_name.lower():
                    return color
            return layer_colors['default']
        
        def sanitize_name(name: str) -> str:
            """将名称转换为有效的 DOT 节点 ID"""
            s = name.replace('.', '_').replace('[', '_').replace(']', '_').replace('-', '_')
            if s[0].isdigit():
                s = 'n' + s
            return s
        
        def get_short_name(name: str) -> str:
            """获取简短的显示名称"""
            parts = name.split('.')
            if len(parts) > 4:
                return '.../' + '/'.join(parts[-2:])
            return name
        
        # 构建 DOT 格式的图
        dot_lines = [
            'digraph MegatronDataFlow {',
            '    rankdir=TB;',
            '    compound=true;',
            '    splines=ortho;',
            '    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9];',
            '    edge [fontname="Helvetica", fontsize=8];',
            f'    label="Megatron-LM Model Data Flow\\nIteration: {iteration}, PP Stage: {state["pp_rank"]}";',
            '    labelloc=t;',
            '    fontsize=14;',
            '',
        ]
        
        # 使用静态分析数据或模块层次结构
        if analysis and analysis.get('modules'):
            modules_info = analysis['modules']
            data_flow = analysis.get('data_flow', [])
            
            self._print_rank_0(f"Generating graph with {len(modules_info)} modules, {len(data_flow)} edges")
            
            # 过滤出重要的模块（有参数的或关键类型的）
            important_modules = {}
            for name, info in modules_info.items():
                class_name = info.get('class', '')
                has_params = info.get('params', 0) > 0
                is_important = any(k.lower() in class_name.lower() for k in [
                    'Embedding', 'Linear', 'Attention', 'MLP', 'Norm', 
                    'Transformer', 'Layer', 'Block', 'Model'
                ])
                if has_params or is_important:
                    # 限制深度
                    depth = info.get('depth', 0)
                    if max_depth is None or depth <= max_depth:
                        important_modules[name] = info
            
            # 添加节点
            for name, info in important_modules.items():
                node_id = sanitize_name(name)
                class_name = info.get('class', 'Unknown')
                color = get_color(class_name)
                display_name = get_short_name(name)
                
                # 构建标签
                label_parts = [display_name, f"({class_name})"]
                
                if info.get('dims'):
                    label_parts.append(info['dims'])
                
                params = info.get('params', 0)
                if params > 0:
                    if params >= 1e9:
                        label_parts.append(f"{params/1e9:.2f}B")
                    elif params >= 1e6:
                        label_parts.append(f"{params/1e6:.2f}M")
                    elif params >= 1e3:
                        label_parts.append(f"{params/1e3:.1f}K")
                    else:
                        label_parts.append(f"{params}")
                
                label = "\\n".join(label_parts)
                dot_lines.append(f'    {node_id} [label="{label}", fillcolor="{color}"];')
            
            dot_lines.append('')
            
            # 添加数据流边
            if data_flow:
                dot_lines.append('    // Data flow edges')
                for src, dst, edge_type in data_flow:
                    if src in important_modules and dst in important_modules:
                        src_id = sanitize_name(src)
                        dst_id = sanitize_name(dst)
                        dot_lines.append(f'    {src_id} -> {dst_id} [color="#1976D2"];')
            
            # 如果没有分析出的数据流，使用层次结构作为连接
            if not data_flow:
                dot_lines.append('    // Hierarchy edges (inferred)')
                for name, info in important_modules.items():
                    for child in info.get('children', []):
                        if child in important_modules:
                            src_id = sanitize_name(name)
                            dst_id = sanitize_name(child)
                            dot_lines.append(f'    {src_id} -> {dst_id} [color="#1976D2"];')
        
        else:
            # 后备方案：直接从模型结构生成
            self._print_rank_0("Using direct module hierarchy")
            
            node_counter = [0]
            node_map = {}  # module_name -> node_id
            
            def add_module_recursive(module: nn.Module, name: str, parent_node_id: Optional[str], depth: int):
                if max_depth is not None and depth > max_depth:
                    return
                
                class_name = module.__class__.__name__
                
                # 只显示有参数或重要的模块
                params = sum(p.numel() for p in module.parameters(recurse=False))
                is_important = any(k.lower() in class_name.lower() for k in [
                    'Embedding', 'Linear', 'Attention', 'MLP', 'Norm', 'Layer', 'Block'
                ])
                
                should_show = params > 0 or is_important or depth <= 2
                
                if should_show:
                    node_id = f"node{node_counter[0]}"
                    node_counter[0] += 1
                    node_map[name] = node_id
                    
                    color = get_color(class_name)
                    display_name = get_short_name(name)
                    
                    # 构建标签
                    label_parts = [display_name, f"({class_name})"]
                    if params > 0:
                        if params >= 1e6:
                            label_parts.append(f"{params/1e6:.2f}M")
                        else:
                            label_parts.append(f"{params:,}")
                    
                    label = "\\n".join(label_parts)
                    dot_lines.append(f'    {node_id} [label="{label}", fillcolor="{color}"];')
                    
                    if parent_node_id:
                        dot_lines.append(f'    {parent_node_id} -> {node_id} [color="#1976D2"];')
                    
                    current_parent = node_id
                else:
                    current_parent = parent_node_id
                
                # 递归处理子模块
                for child_name, child_module in module.named_children():
                    full_name = f"{name}.{child_name}" if name else child_name
                    add_module_recursive(child_module, full_name, current_parent, depth + 1)
            
            add_module_recursive(model, model.__class__.__name__, None, 0)
        
        dot_lines.append('}')
        dot_content = '\n'.join(dot_lines)
        
        # 保存 DOT 文件
        dot_path = output_path + ".dot"
        with open(dot_path, 'w', encoding='utf-8') as f:
            f.write(dot_content)
        generated_files.append(dot_path)
        
        # 尝试使用 graphviz 生成图片
        try:
            import graphviz
            graph = graphviz.Source(dot_content)
            # 生成 SVG（可以用浏览器打开，支持缩放）
            svg_path = graph.render(output_path, format='svg', cleanup=True)
            generated_files.append(svg_path)
            self._print_rank_0(
                f"[Rank {state['global_rank']}] Model graph saved:\n"
                f"  SVG: {svg_path} (open in browser)\n"
                f"  DOT: {dot_path} (use https://dreampuf.github.io/GraphvizOnline/)"
            )
        except ImportError:
            self._print_rank_0(
                f"[Rank {state['global_rank']}] Model graph saved: {dot_path}\n"
                f"  View online: https://dreampuf.github.io/GraphvizOnline/\n"
                f"  Or install graphviz: pip install graphviz"
            )
        except Exception as e:
            self._print_rank_0(
                f"[Rank {state['global_rank']}] DOT file saved: {dot_path}\n"
                f"  (graphviz rendering failed: {e})\n"
                f"  View online: https://dreampuf.github.io/GraphvizOnline/"
            )
        
        # 同时保存模型参数统计
        stats_path = output_path + "_stats.txt"
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write(f"Megatron-LM Model Statistics\n")
                f.write(f"Iteration: {iteration}\n")
                f.write(f"PP Stage: {state['pp_rank']}\n")
                f.write(f"=" * 80 + "\n\n")
                
                # 模型概览
                f.write("Model Overview:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Model Class: {model.__class__.__name__}\n")
                
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                f.write(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)\n")
                f.write(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)\n\n")
                
                # 按层类型统计
                f.write("Parameters by Layer Type:\n")
                f.write("-" * 40 + "\n")
                layer_stats = {}
                for name, module in model.named_modules():
                    class_name = module.__class__.__name__
                    params = sum(p.numel() for p in module.parameters(recurse=False))
                    if params > 0:
                        if class_name not in layer_stats:
                            layer_stats[class_name] = {'count': 0, 'params': 0}
                        layer_stats[class_name]['count'] += 1
                        layer_stats[class_name]['params'] += params
                
                for class_name, stats in sorted(layer_stats.items(), key=lambda x: -x[1]['params']):
                    f.write(f"  {class_name}: {stats['count']} layers, {stats['params']:,} params\n")
                
                f.write("\n")
                
                # 详细参数列表
                f.write("All Parameters:\n")
                f.write("-" * 40 + "\n")
                for name, param in model.named_parameters():
                    f.write(f"  {name}: {list(param.shape)} ({param.numel():,})\n")
                
            generated_files.append(stats_path)
        except Exception as e:
            self._print_rank_0(f"Failed to save stats: {e}")
        
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
    
    生成的文件：
    - .dot 文件：在线查看 https://dreampuf.github.io/GraphvizOnline/
    - .svg 文件：浏览器直接打开（需安装 pip install graphviz）
    - _stats.txt：参数统计信息
    
    Args:
        args: Megatron 命令行参数
    
    Returns:
        MegatronGraphVisualizer 实例，如果未启用则返回 None
    """
    if not getattr(args, 'visualize_model_graph', False):
        return None
    
    # 解析迭代步骤
    iterations_str = getattr(args, 'visualize_graph_iterations', "1")
    iterations = [int(x.strip()) for x in iterations_str.split(',') if x.strip()]
    
    # 确定输出目录
    output_dir = getattr(args, 'visualize_graph_output_dir', None)
    if output_dir is None:
        output_dir = getattr(args, 'save', None)
    if output_dir is None:
        output_dir = "./checkpoints"
    
    # 获取最大深度限制
    max_depth = getattr(args, 'visualize_graph_max_depth', None)
    
    return MegatronGraphVisualizer.initialize(
        output_dir=output_dir,
        iterations_to_visualize=iterations,
        visualize_interval=getattr(args, 'visualize_graph_interval', None),
        output_format='dot',
        include_shapes=True,
        max_depth=max_depth,
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
