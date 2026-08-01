# PR #5960 CP indexer loss 调试复现工具

这些工具仅用于调试未融合 DeepSeek-V4 CSA indexer loss 路径在
`context_parallel_size > 1` 时的精度问题，基于 PR #5960 的提交
`3ae4e2bb49fee19b6cefa1d562af438e9340c5a9`。

工具覆盖三个层级：

- `repro_cp2_unfused_indexer_loss.py` 是双卡的小算子数学接缝复现。它将已修正的
  CP1 window + sink teacher 与当前 CP2 helper 对比，不依赖完整 CP layout kernel。
- `repro_cp2_unfused.py` 调用仓库已有的 THD CP 端到端精度测试，并强制 DSA
  indexer 和 RoPE 走非融合路径。
- `run_cp1_cp2_training.sh` 使用真实 `pretrain_gpt.py` 分别训练 CP1 和 CP2；
  两者从相同 W0 开始，并读取相同的 mmap token 样本。

## 小算子数学接缝

在 Megatron-LM 仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc-per-node=2 \
  tools/debug/pr5960/repro_cp2_unfused_indexer_loss.py --strict-exit
```

在已审计的 PR head 上，预期输出 `ALIGNMENT=FAIL (CP>1 bug reproduced)`。
绝对误差超过 `1e-4` 时命令返回非零退出码。

## 完整 CP layer 精度测试

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python3.12 tools/debug/pr5960/repro_cp2_unfused.py --loss-mode both
```

该脚本复用仓库已有的 CP2 与 CP1 前反向测试，覆盖 sparse 和 dense 两种 ratio-4
indexer loss。已审计 PR head 上精度断言失败即表示问题复现；修复后两种 case
都应通过。

## 真实 CP1/CP2 训练

```bash
DATA_PREFIX=/data/ljb/data/deepseek-datasets/mmap_deepseekv3_datasets_text_document \
CP2_DEVICES=0,1 \
STEPS=5 \
bash tools/debug/pr5960/run_cp1_cp2_training.sh
```

设置 `BASE_CKPT=/path/to/legacy/checkpoint` 可以复用已有 W0。未设置时，脚本先用
CP1 执行一步 `lr=0` 训练并保存 legacy checkpoint，再让两个对比任务仅加载其中的
模型权重。

严格控制的对比配置如下：

| 场景 | World | TP | PP | DP | CP | 每 rank THD 行数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CP1 | 1 | 1 | 1 | 1 | 1 | 512 |
| CP2 | 2 | 1 | 1 | 1 | 2 | 256 per rank |

两种场景均使用 `dp_balanced` sequence packing、contiguous CP 切分、非融合
DSA/indexer 路径、相同 seed、global batch size 1，以及同一个 513-token mmap
验证样本（shift 后得到 512 行训练 token）。输出目录包含原始日志、manifest、W0
路径和 `loss_compare.csv`。主要复现判据为：

```text
iteration-1 abs(CP2 indexer loss - CP1 indexer loss) > 1e-4
```

解析器会先修正当前 step 2 日志的累计平均行为，再输出逐 step 的有符号误差和绝对误差。

## 环境要求

完整精度测试和训练路径要求 Python 3.12、两张 H100/H200 或受支持的 Blackwell
GPU，以及 Transformer Engine、CuTe/CUTLASS 和 `fast_hadamard_transform`。
当前 CSA CP layout kernel 不支持 A100。小算子数学接缝仍需两张 CUDA GPU，
但不调用 CP layout kernel。
