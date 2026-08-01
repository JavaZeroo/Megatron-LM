#!/usr/bin/env python3
"""Reproduce PR #5960's CP>1 unfused indexer-teacher loss mismatch.

This is a narrow mathematical seam, not a full attention-layer run.  The CP1
reference calls the PR's corrected ``FusedDSAIndexerLoss`` with the real
window-plus-sink LSE.  The CP2 side calls the actual
``_unfused_indexer_sparse_attn_from_topk`` helper used by the THD CP path and
reduces two local losses.  A difference above ``--atol`` isolates the missing
sliding-window mass without involving CP layout kernels or communication
beyond the final loss reduction.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(os.environ.get("MEGATRON_REPO", Path(__file__).resolve().parents[3])).resolve()
sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist

from megatron.core.transformer.experimental_attention_variant import csa as csa_module
from megatron.core.transformer.experimental_attention_variant.csa import (
    _compute_unfused_csa_non_compressed_lse,
    _unfused_indexer_sparse_attn_from_topk,
    get_window_topk_idxs,
)
from megatron.core.transformer.experimental_attention_variant.dsa import FusedDSAIndexerLoss


class _SizeOneGroup:
    def size(self):
        return 1


def _shared_randn(shape, dtype, device, rank):
    tensor = torch.empty(shape, dtype=dtype, device=device)
    if rank == 0:
        tensor.normal_()
    dist.broadcast(tensor, src=0)
    return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Exit non-zero when CP2 and corrected CP1 differ.",
    )
    args = parser.parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 2:
        raise RuntimeError(f"This reproducer requires exactly 2 ranks, got {world_size}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("This command requires CUDA; expose two GPUs to torchrun")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    torch.manual_seed(5960)
    torch.cuda.manual_seed_all(5960)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Small but structure-preserving synthetic CSA/indexer case.  Multiple
    # attention heads are essential: with one head the final L1 normalization
    # can hide the omitted denominator mass.
    seq_len, ratio = 64, 4
    compressed_len = seq_len // ratio
    attn_heads, head_dim = 4, 16
    index_heads, index_dim = 4, 16
    topk, window = 8, 16
    dtype = torch.bfloat16

    q_indexer = _shared_randn((seq_len, 1, index_heads, index_dim), dtype, device, rank)
    weights = _shared_randn((seq_len, 1, index_heads), dtype, device, rank)
    k_indexer = _shared_randn((compressed_len, 1, index_dim), dtype, device, rank)
    teacher_q = _shared_randn((seq_len, attn_heads, head_dim), dtype, device, rank)
    original_kv = _shared_randn((seq_len, head_dim), dtype, device, rank)
    compressed_kv = _shared_randn((compressed_len, head_dim), dtype, device, rank)
    attn_sink = torch.linspace(-1.5, 1.5, attn_heads, dtype=torch.float32, device=device)

    softmax_scale = head_dim**-0.5
    indexer_scale = index_dim**-0.5
    window_indices = get_window_topk_idxs(window, 1, seq_len, device).squeeze(0)

    # Correct CP1 reference: normalize compressed teacher logits together with
    # the non-compressed sliding-window and sink mass.
    non_compressed_lse = _compute_unfused_csa_non_compressed_lse(
        teacher_q, original_kv, attn_sink, window_indices, softmax_scale
    )
    positions = torch.arange(1, seq_len + 1, device=device).unsqueeze(1)
    compressed_ids = torch.arange(compressed_len, device=device).unsqueeze(0)
    causal_mask = torch.where(
        compressed_ids >= positions // ratio,
        torch.tensor(float("-inf"), device=device),
        torch.tensor(0.0, device=device),
    ).unsqueeze(0)
    teacher_key = compressed_kv.unsqueeze(1).unsqueeze(2).expand(-1, 1, attn_heads, -1)
    pg = SimpleNamespace(tp=_SizeOneGroup())

    with torch.no_grad():
        selected, cp1_loss = FusedDSAIndexerLoss.apply(
            q_indexer,
            weights.float() * indexer_scale,
            k_indexer,
            teacher_q.unsqueeze(1),
            teacher_key,
            softmax_scale,
            topk,
            1.0,
            causal_mask,
            True,
            pg,
            None,
            None,
            None,
            None,
            False,
            True,
            non_compressed_lse,
        )

    selected = selected.squeeze(0)

    def current_cp_helper_loss(start, end):
        # Sparse loss does not consume indexer_layout, but retain the real ABI.
        indexer_layout = (
            torch.tensor([0, end - start], dtype=torch.int32, device=device),
            torch.tensor([0, compressed_len], dtype=torch.int32, device=device),
            torch.tensor([start], dtype=torch.int32, device=device),
        )
        with torch.no_grad():
            _, loss = _unfused_indexer_sparse_attn_from_topk(
                teacher_q[start:end],
                compressed_kv,  # Attention output is discarded in this loss-only seam.
                attn_sink,
                selected[start:end],
                q_indexer[start:end, 0],
                k_indexer[:, 0],
                weights[start:end, 0],
                selected[start:end],
                compressed_kv,
                softmax_scale,
                indexer_scale,
                1.0,
                float(seq_len),  # Global divisor used by the real CP path.
                True,
                ratio,
                seq_len,
                indexer_layout,
                None,
            )
        return loss.float()

    rows_per_rank = seq_len // world_size
    start = rank * rows_per_rank
    end = start + rows_per_rank
    local_cp2_loss = current_cp_helper_loss(start, end)
    cp2_loss = local_cp2_loss.clone()
    dist.all_reduce(cp2_loss, op=dist.ReduceOp.SUM)

    # Control: CP splitting and reduction should reproduce the same current
    # helper formula.  This separates partition bugs from teacher-formula drift.
    current_full_loss = current_cp_helper_loss(0, seq_len)
    partition_diff = (cp2_loss - current_full_loss).abs()
    alignment_signed_diff = cp2_loss - cp1_loss.float()
    alignment_abs_diff = alignment_signed_diff.abs()

    local_losses = [torch.zeros_like(local_cp2_loss) for _ in range(world_size)]
    dist.all_gather(local_losses, local_cp2_loss)

    cp1_min = cp1_loss.detach().float().clone()
    cp1_max = cp1_min.clone()
    dist.all_reduce(cp1_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(cp1_max, op=dist.ReduceOp.MAX)

    passed = bool(alignment_abs_diff.item() <= args.atol)
    if rank == 0:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
            ).strip()
        except Exception:
            commit = "unknown"
        print(f"commit={commit}")
        print(f"runtime_csa={Path(csa_module.__file__).resolve()}")
        print(f"cp1_loss={cp1_loss.item():.10f}")
        print(f"cp1_cross_rank_spread={(cp1_max - cp1_min).item():.3e}")
        print(f"cp2_rank_losses={[float(x.item()) for x in local_losses]}")
        print(f"cp2_reduced_loss={cp2_loss.item():.10f}")
        print(f"cp2_split_vs_same_helper_full_diff={partition_diff.item():.10e}")
        print(f"signed_diff_cp2_minus_cp1={alignment_signed_diff.item():.10e}")
        print(f"max_abs_loss_diff={alignment_abs_diff.item():.10e}")
        print(f"atol={args.atol:.1e}")
        print("ALIGNMENT=" + ("PASS" if passed else "FAIL (CP>1 bug reproduced)"))

    dist.barrier()
    dist.destroy_process_group()
    if args.strict_exit and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
