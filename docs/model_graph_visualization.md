# Megatron-LM 模型计算图可视化

本文档介绍如何使用 torchviz 可视化 Megatron-LM 模型的计算图。可视化功能已集成到训练流程中，支持分布式训练。

## 安装依赖

```bash
# 安装 torchviz
pip install torchviz

# 安装 graphviz（系统依赖）
# Ubuntu/Debian:
sudo apt-get install graphviz

# macOS:
brew install graphviz

# Windows:
choco install graphviz
# 或从 https://graphviz.org/download/ 下载安装
```

## 使用方法

### 1. 在训练中启用可视化

在运行训练脚本时添加可视化参数：

```bash
python pretrain_gpt.py \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    ... \
    --visualize-model-graph \
    --visualize-graph-iterations 1,100,1000 \
    --visualize-graph-format pdf \
    --visualize-graph-output-dir ./checkpoints
```

### 2. 可视化参数说明

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--visualize-model-graph` | 启用模型计算图可视化 | False |
| `--visualize-graph-iterations` | 在哪些迭代步骤生成图（逗号分隔） | "1" |
| `--visualize-graph-interval` | 每N步生成一次图 | None |
| `--visualize-graph-format` | 输出格式 (pdf, png, svg) | "pdf" |
| `--visualize-graph-output-dir` | 输出目录 | 使用 --save 目录 |

### 3. 示例脚本

参见 `examples/gpt3/train_gpt_with_visualization.sh`:

```bash
#!/bin/bash
torchrun --nproc_per_node=1 pretrain_gpt.py \
    --num-layers 2 \
    --hidden-size 256 \
    --num-attention-heads 8 \
    --seq-length 128 \
    --micro-batch-size 2 \
    --global-batch-size 8 \
    --train-iters 20 \
    --visualize-model-graph \
    --visualize-graph-iterations 1,10 \
    --visualize-graph-format pdf \
    --save ./checkpoints/gpt2_vis
```

### 4. 独立可视化脚本

使用独立脚本可视化模型（不需要运行完整训练）：

```bash
python tools/visualize_megatron_model.py \
    --mock-model \
    --num-layers 2 \
    --hidden-size 256 \
    --num-attention-heads 8 \
    --output-dir ./model_graphs \
    --format pdf
```

## 分布式训练支持

可视化功能完全支持 Megatron-LM 的分布式训练：

### 数据并行 (DP)
- 只在 `dp_rank=0` 的进程上生成图
- 避免重复生成相同的图

### 张量并行 (TP)
- 只在 `tp_rank=0` 的进程上生成图
- 每个张量并行组只生成一份图

### 流水线并行 (PP)
- 每个流水线阶段生成独立的图
- 文件名包含 `pp{rank}` 标识

### 虚拟流水线并行 (VPP)
- 每个虚拟阶段生成独立的图
- 文件名包含 `vp{id}` 标识

## 输出文件

生成的图文件保存在 `{output_dir}/model_graphs/` 目录下：

```
checkpoints/
└── model_graphs/
    ├── model_graph_iter000001_pp0.pdf
    ├── model_graph_iter000001_pp1.pdf
    ├── model_graph_iter000100_pp0.pdf
    └── model_graph_iter000100_pp1.pdf
```

文件命名格式：`model_graph_iter{iteration:06d}_pp{pp_rank}[_vp{vp_id}].{format}`

## 计算图内容

生成的计算图包含：

1. **输入节点**: 模型输入张量
2. **参数节点**: 模型的可训练参数（显示参数名称）
3. **操作节点**: 前向传播中的各种操作
4. **梯度节点**: 反向传播的梯度流

图的颜色编码：
- 蓝色: 模型参数
- 绿色: 中间张量
- 灰色: 操作节点

## 注意事项

1. **性能影响**: 可视化会在指定迭代步骤增加一些开销，建议只在少数迭代上启用
2. **内存使用**: 对于大型模型，生成的图可能非常大
3. **文件大小**: PDF 格式通常比 PNG 更小且更清晰
4. **兼容性**: 需要安装 graphviz 系统包

## API 使用

如果您需要在自定义代码中使用可视化功能：

```python
from megatron.training.visualization import (
    setup_model_graph_visualization,
    maybe_visualize_model_graph,
    MegatronGraphVisualizer,
)

# 方式1: 使用 args 设置
visualizer = setup_model_graph_visualization(args)

# 方式2: 手动初始化
visualizer = MegatronGraphVisualizer.initialize(
    output_dir="./graphs",
    iterations_to_visualize=[1, 100, 1000],
    visualize_interval=None,
    output_format="pdf",
)

# 在 forward_step 中调用
def forward_step(data_iterator, model):
    output = model(input_ids, position_ids, attention_mask)
    
    # 可视化（如果条件满足）
    maybe_visualize_model_graph(model, output, iteration)
    
    return output, loss_func
```

## 故障排除

### 问题: "torchviz not installed"
解决: 运行 `pip install torchviz`

### 问题: "graphviz not found" 或 "dot command not found"
解决: 安装 graphviz 系统包（见上方安装说明）

### 问题: "Could not extract output tensor with grad_fn"
原因: 输出张量没有梯度函数，可能是：
- 模型处于 eval 模式
- 前向传播在 `torch.no_grad()` 上下文中
解决: 确保可视化在正常的训练前向传播中进行

### 问题: 生成的图过大无法打开
解决: 
- 使用 SVG 格式并用浏览器查看
- 减少模型层数进行测试
- 使用 `--num-layers 2` 等小型配置先测试

## 相关文件

- `megatron/training/visualization.py`: 核心可视化模块
- `megatron/training/arguments.py`: 命令行参数定义
- `tools/visualize_megatron_model.py`: 独立可视化脚本
- `tools/visualize_model_graph.py`: 通用可视化工具
- `examples/gpt3/train_gpt_with_visualization.sh`: 示例脚本
