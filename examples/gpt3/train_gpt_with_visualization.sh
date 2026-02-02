#!/bin/bash

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Example script for running Megatron-LM GPT training with model graph visualization
# This script demonstrates how to enable computation graph visualization using torchviz

# Prerequisites:
# 1. Install torchviz: pip install torchviz
# 2. Install graphviz: 
#    - Ubuntu/Debian: sudo apt-get install graphviz
#    - macOS: brew install graphviz
#    - Windows: choco install graphviz (or download from https://graphviz.org/)

# Basic configuration
GPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Paths - Modify these according to your setup
CHECKPOINT_PATH="./checkpoints/gpt2_visualization"
VOCAB_FILE="./data/gpt2-vocab.json"
MERGE_FILE="./data/gpt2-merges.txt"
DATA_PATH="./data/my-gpt2_text_document"

# Model configuration (small model for testing)
NUM_LAYERS=2
HIDDEN_SIZE=256
NUM_ATTENTION_HEADS=8
SEQ_LENGTH=128
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=8

# Distributed training configuration
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Model graph visualization arguments
# --visualize-model-graph: Enable visualization
# --visualize-graph-iterations: Comma-separated list of iterations to visualize (default: "1")
# --visualize-graph-interval: Visualize every N iterations (optional)
# --visualize-graph-format: Output format - pdf, png, or svg (default: pdf)
# --visualize-graph-output-dir: Output directory (defaults to --save path)
VISUALIZATION_ARGS="
    --visualize-model-graph \
    --visualize-graph-iterations 1,10 \
    --visualize-graph-format pdf
"

# GPT model arguments
GPT_ARGS="
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters 20 \
    --lr-decay-iters 20 \
    --lr 0.00015 \
    --min-lr 0.00001 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --fp16 \
    --use-flash-attn
"

# Data arguments
DATA_ARGS="
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-path $DATA_PATH \
    --split 949,50,1
"

# Output arguments
OUTPUT_ARGS="
    --save $CHECKPOINT_PATH \
    --save-interval 10 \
    --log-interval 1 \
    --eval-interval 100 \
    --eval-iters 10
"

# Run training with visualization
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $VISUALIZATION_ARGS

echo ""
echo "=========================================="
echo "Training complete!"
echo "Model graphs saved to: $CHECKPOINT_PATH/model_graphs/"
echo "=========================================="

# List generated graph files
if [ -d "$CHECKPOINT_PATH/model_graphs" ]; then
    echo ""
    echo "Generated graph files:"
    ls -la $CHECKPOINT_PATH/model_graphs/
fi
