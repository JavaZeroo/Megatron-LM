#!/usr/bin/env python
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Standalone script for visualizing Megatron-LM model computation graphs.

This script can be used to:
1. Visualize a model from a checkpoint
2. Generate computation graphs without running full training
3. Compare graphs across different model configurations

Requirements:
    pip install torchviz graphviz

Usage:
    # Visualize a checkpoint
    python tools/visualize_megatron_model.py \
        --checkpoint-path /path/to/checkpoint \
        --output-dir ./model_graphs \
        --format pdf

    # Visualize with mock data (no checkpoint needed)
    python tools/visualize_megatron_model.py \
        --mock-model \
        --num-layers 2 \
        --hidden-size 256 \
        --num-attention-heads 8 \
        --output-dir ./model_graphs
"""

import argparse
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

# Check for torchviz
try:
    from torchviz import make_dot
    HAVE_TORCHVIZ = True
except ImportError:
    HAVE_TORCHVIZ = False
    print("Error: torchviz is required. Install with: pip install torchviz")
    print("Also ensure graphviz is installed on your system:")
    print("  Ubuntu/Debian: sudo apt-get install graphviz")
    print("  macOS: brew install graphviz")
    print("  Windows: choco install graphviz")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize Megatron-LM model computation graphs'
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--checkpoint-path',
        type=str,
        help='Path to model checkpoint directory'
    )
    input_group.add_argument(
        '--mock-model',
        action='store_true',
        help='Create a mock model for visualization (no checkpoint needed)'
    )
    
    # Model architecture (for mock model)
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Hidden size')
    parser.add_argument('--num-attention-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--seq-length', type=int, default=128,
                        help='Sequence length')
    parser.add_argument('--vocab-size', type=int, default=50304,
                        help='Vocabulary size')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for forward pass')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./model_graphs',
                        help='Output directory for graphs')
    parser.add_argument('--output-name', type=str, default='model_graph',
                        help='Base name for output files')
    parser.add_argument('--format', type=str, default='pdf',
                        choices=['pdf', 'png', 'svg'],
                        help='Output format')
    
    # Visualization options
    parser.add_argument('--show-shapes', action='store_true',
                        help='Show tensor shapes in graph')
    parser.add_argument('--show-saved', action='store_true',
                        help='Show saved tensors')
    parser.add_argument('--max-attr-chars', type=int, default=50,
                        help='Maximum characters for attributes')
    
    return parser.parse_args()


class SimplifiedGPTBlock(nn.Module):
    """Simplified GPT transformer block for visualization."""
    
    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        
        # Layer norm
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        
        # Self-attention
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        
        # MLP
        self.mlp_dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
        self.mlp_dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)
        self.mlp_act = nn.GELU()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # QKV projections
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # Reshape for attention
        batch_size, seq_len, _ = query.shape
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.dense(attn_output)
        
        hidden_states = residual + attn_output
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp_dense_h_to_4h(hidden_states)
        hidden_states = self.mlp_act(hidden_states)
        hidden_states = self.mlp_dense_4h_to_h(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class SimplifiedGPTModel(nn.Module):
    """Simplified GPT model for visualization purposes."""
    
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        vocab_size: int,
        max_position_embeddings: int,
    ):
        super().__init__()
        
        # Embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SimplifiedGPTBlock(hidden_size, num_attention_heads)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_layernorm = nn.LayerNorm(hidden_size)
        
        # Output projection (tied with input embeddings conceptually)
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        hidden_states = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Final layer norm
        hidden_states = self.final_layernorm(hidden_states)
        
        # Output logits
        logits = self.output_layer(hidden_states)
        
        return logits


def create_mock_model(args) -> nn.Module:
    """Create a simplified mock GPT model."""
    model = SimplifiedGPTModel(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.seq_length,
    )
    return model


def visualize_model(
    model: nn.Module,
    args,
    device: str = 'cpu',
) -> str:
    """
    Visualize a model's computation graph.
    
    Args:
        model: The PyTorch model
        args: Command line arguments
        device: Device to run on
    
    Returns:
        Path to the generated file
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_length), device=device)
    position_ids = torch.arange(args.seq_length, device=device).unsqueeze(0).expand(args.batch_size, -1)
    
    # Forward pass with gradient tracking
    with torch.enable_grad():
        output = model(input_ids, position_ids)
    
    # Collect parameters
    params = {name: param for name, param in model.named_parameters()}
    
    # Generate graph
    dot = make_dot(
        output,
        params=params,
        show_attrs=args.show_shapes,
        show_saved=args.show_saved,
        max_attr_chars=args.max_attr_chars,
    )
    
    # Configure graph appearance
    dot.attr(rankdir='TB')
    dot.attr('graph', label=f'GPT Model Computation Graph\nLayers: {args.num_layers}, Hidden: {args.hidden_size}, Heads: {args.num_attention_heads}')
    dot.attr('graph', fontsize='14')
    dot.attr('node', fontsize='10')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output path
    output_path = os.path.join(args.output_dir, args.output_name)
    
    # Render
    output_file = dot.render(output_path, format=args.format, cleanup=True)
    
    print(f"Model graph saved to: {output_file}")
    return output_file


def load_checkpoint_model(checkpoint_path: str):
    """
    Load a model from a Megatron-LM checkpoint.
    
    This is a placeholder - actual implementation would require
    Megatron-LM initialization and checkpoint loading.
    """
    raise NotImplementedError(
        "Loading from checkpoint requires full Megatron-LM initialization. "
        "Use --mock-model for standalone visualization, or run visualization "
        "during training with --visualize-model-graph."
    )


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("Megatron-LM Model Graph Visualization")
    print("=" * 60)
    
    if args.mock_model:
        print(f"\nCreating mock GPT model:")
        print(f"  Layers: {args.num_layers}")
        print(f"  Hidden size: {args.hidden_size}")
        print(f"  Attention heads: {args.num_attention_heads}")
        print(f"  Sequence length: {args.seq_length}")
        print(f"  Vocab size: {args.vocab_size}")
        
        model = create_mock_model(args)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
    else:
        print(f"\nLoading model from: {args.checkpoint_path}")
        model = load_checkpoint_model(args.checkpoint_path)
    
    print(f"\nGenerating computation graph...")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Output format: {args.format}")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    output_file = visualize_model(model, args, device)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Graph saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
