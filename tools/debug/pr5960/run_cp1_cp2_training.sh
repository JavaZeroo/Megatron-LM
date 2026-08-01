#!/usr/bin/env bash
# Reproduce NVIDIA/Megatron-LM PR #5960's unfused CSA indexer-loss mismatch
# with real pretrain_gpt.py training:
#   CP1: world=1, TP1/PP1/DP1/CP1, THD, 512 global rows
#   CP2: world=2, TP1/PP1/DP1/CP2, THD, 2 x 256 contiguous rows
#
# Both runs load the same legacy W0 checkpoint, consume the same mmap-backed
# verification sample, and differ only in context-parallel topology.

set -euo pipefail

EXPECTED_PR_HEAD="3ae4e2bb49fee19b6cefa1d562af438e9340c5a9"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MEGATRON_REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MEGATRON_REPO="${MEGATRON_REPO:-${DEFAULT_MEGATRON_REPO}}"
DATA_PREFIX="${DATA_PREFIX:-/data/ljb/data/deepseek-datasets/mmap_deepseekv3_datasets_text_document}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
CP2_DEVICES="${CP2_DEVICES:-0,1}"
CP1_DEVICE="${CP1_DEVICE:-${CP2_DEVICES%%,*}}"
STEPS="${STEPS:-5}"
SEED="${SEED:-1234}"
ATOL="${ATOL:-1e-4}"
VOCAB_SIZE="${VOCAB_SIZE:-129280}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/runs/pr5960_cp_training_${RUN_ID}}"

# Optional: point BASE_CKPT at an existing legacy Megatron checkpoint.  When
# omitted, the script creates one with a one-step, lr=0 CP1 run, then reloads
# only its model weights with --finetune for both comparison runs.
BASE_CKPT="${BASE_CKPT:-}"

W0_PORT="${W0_PORT:-29560}"
CP1_PORT="${CP1_PORT:-29561}"
CP2_PORT="${CP2_PORT:-29562}"

if [[ ! -d "${MEGATRON_REPO}/megatron" || ! -f "${MEGATRON_REPO}/pretrain_gpt.py" ]]; then
  echo "ERROR: invalid Megatron-LM checkout: ${MEGATRON_REPO}" >&2
  exit 2
fi
if [[ ! -f "${DATA_PREFIX}.bin" || ! -f "${DATA_PREFIX}.idx" ]]; then
  echo "ERROR: mmap dataset prefix is incomplete: ${DATA_PREFIX}.{bin,idx}" >&2
  exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

IFS=',' read -r -a _cp2_gpu_ids <<<"${CP2_DEVICES}"
if [[ "${#_cp2_gpu_ids[@]}" -ne 2 ]]; then
  echo "ERROR: CP2_DEVICES must contain exactly two GPU ids, got: ${CP2_DEVICES}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}"

HEAD_SHA="$(git -C "${MEGATRON_REPO}" rev-parse HEAD)"
{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "repo=${MEGATRON_REPO}"
  echo "head=${HEAD_SHA}"
  echo "expected_pr_head=${EXPECTED_PR_HEAD}"
  echo "data_prefix=${DATA_PREFIX}"
  echo "steps=${STEPS}"
  echo "seed=${SEED}"
  echo "cp1_device=${CP1_DEVICE}"
  echo "cp2_devices=${CP2_DEVICES}"
  echo "python=${PYTHON_BIN}"
} | tee "${OUT_ROOT}/manifest.txt"

if [[ "${HEAD_SHA}" != "${EXPECTED_PR_HEAD}" ]]; then
  echo "NOTE: audited PR head is ${EXPECTED_PR_HEAD}; current checkout is ${HEAD_SHA}."
  echo "      This is valid for a before/after run, but record the SHA with the result."
fi

# Complete THD CP training still uses CSA's CuTe layout kernels even though the
# DSA/indexer loss itself is unfused.  Current kernels support H100/H200 and
# selected Blackwell architectures, not A100.
CUDA_VISIBLE_DEVICES="${CP2_DEVICES}" PYTHONPATH="${MEGATRON_REPO}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" - <<'PY'
import importlib
import sys

import torch

if sys.version_info < (3, 12):
    raise SystemExit(f"Python >= 3.12 is required, got {sys.version.split()[0]}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise SystemExit(f"Exactly two visible CUDA GPUs are required, got {torch.cuda.device_count()}")

supported = {(9, 0), (10, 0), (10, 3)}
caps = [torch.cuda.get_device_capability(i) for i in range(2)]
if any(cap not in supported for cap in caps):
    raise SystemExit(
        f"CSA CP layout supports sm_90a/sm_100a/sm_103a only; visible capabilities={caps}"
    )

for module in ("transformer_engine.pytorch", "cutlass.cute", "fast_hadamard_transform"):
    try:
        importlib.import_module(module)
    except Exception as exc:
        raise SystemExit(f"Missing required training dependency {module}: {exc}") from exc

print(f"preflight_ok=true cuda_capabilities={caps} torch={torch.__version__}")
PY

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring
export NCCL_DEBUG=WARN
export PYTHONHASHSEED="${SEED}"
export PYTHONPATH="${MEGATRON_REPO}:${PYTHONPATH:-}"

# Verification mode reads actual token ids from DATA_PREFIX.  length=513 is
# deliberate: MockVarlen appends EOD and applies the next-token shift, yielding
# exactly 512 real THD rows in both CP1 and CP2.
VARLEN_SPEC="{\"mode\":\"verification\",\"data_path\":\"${DATA_PREFIX}\",\"min_seq_len\":513,\"max_seq_len\":513,\"mean_seq_len\":513,\"lognormal_sigma\":0.1}"

COMMON_ARGS=(
  --enable-experimental
  --use-mcore-models
  --num-layers 1
  --hidden-size 512
  --ffn-hidden-size 1024
  --num-attention-heads 8
  --multi-latent-attention
  --q-lora-rank 192
  --kv-lora-rank 64
  --qk-head-dim 16
  --qk-pos-emb-head-dim 8
  --v-head-dim 16
  --experimental-attention-variant dsv4_hybrid
  --csa-window-size 128
  --csa-compress-ratios '([4])'
  --csa-compress-rotary-base 40000
  --dsa-indexer-n-heads 64
  --dsa-indexer-head-dim 128
  --dsa-indexer-topk 32
  --dsa-indexer-loss-coeff 1.0
  --dsa-indexer-use-sparse-loss
  --dsa-kernel-backend none
  --attention-backend unfused
  --transformer-impl transformer_engine
  --normalization RMSNorm
  --norm-epsilon 1e-6
  --qk-layernorm
  --swiglu
  --disable-bias-linear
  --untie-embeddings-and-output-weights
  --position-embedding-type rope
  --rotary-base 10000
  --seq-length 512
  --max-position-embeddings 512
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --micro-batch-size 1
  --global-batch-size 1
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --cp-partition-mode contiguous
  --sequence-packing-scheduler dp_balanced
  --use-varlen-dataset
  --mock-data
  --varlen-mock-dataset-config-json "${VARLEN_SPEC}"
  --dataloader-type single
  --num-workers 0
  --no-create-attention-mask-in-dataloader
  --tokenizer-type NullTokenizer
  --vocab-size "${VOCAB_SIZE}"
  --null-tokenizer-eod-id 0
  --null-tokenizer-pad-id 0
  --make-vocab-size-divisible-by 128
  --split 100,0,0
  --optimizer adam
  --adam-beta1 0.9
  --adam-beta2 0.95
  --weight-decay 0.0
  --clip-grad 1.0
  --lr-decay-style constant
  --lr-warmup-iters 0
  --bf16
  --seed "${SEED}"
  --deterministic-mode
  --no-gradient-accumulation-fusion
  --attention-softmax-in-fp32
  --distributed-backend nccl
  --ckpt-format torch
  --log-interval 1
  --eval-interval 1000000
  --eval-iters 1
)

run_case() {
  local name="$1"
  local cp_size="$2"
  local devices="$3"
  local port="$4"
  local max_local="$5"
  shift 5

  local case_dir="${OUT_ROOT}/${name}"
  local log_file="${case_dir}/train.log"
  mkdir -p "${case_dir}" "${case_dir}/data_cache"

  echo
  echo "===== ${name}: CP=${cp_size}, devices=${devices}, max_local=${max_local} ====="
  (
    cd "${MEGATRON_REPO}"
    CUDA_VISIBLE_DEVICES="${devices}" \
      "${PYTHON_BIN}" -m torch.distributed.run \
        --nnodes=1 \
        --node-rank=0 \
        --nproc-per-node="${cp_size}" \
        --master-addr=127.0.0.1 \
        --master-port="${port}" \
        pretrain_gpt.py \
        "${COMMON_ARGS[@]}" \
        --context-parallel-size "${cp_size}" \
        --max-seqlen-per-dp-cp-rank "${max_local}" \
        --data-cache-path "${case_dir}/data_cache" \
        "$@"
  ) 2>&1 | tee "${log_file}"
}

if [[ -n "${BASE_CKPT}" ]]; then
  W0_DIR="${BASE_CKPT}"
  if [[ ! -f "${W0_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "ERROR: BASE_CKPT is not a readable Megatron checkpoint: ${W0_DIR}" >&2
    exit 2
  fi
  echo "Using caller-provided W0: ${W0_DIR}"
else
  W0_DIR="${OUT_ROOT}/w0_checkpoint"
  if [[ ! -f "${W0_DIR}/latest_checkpointed_iteration.txt" ]]; then
    # lr=0 makes the one optimizer step a no-op on model weights.  Saving as a
    # legacy checkpoint keeps CP out of the checkpoint shard identity, so the
    # exact same mp_rank_00 model weights load into CP1 and both CP2 ranks.
    run_case w0_prepare 1 "${CP1_DEVICE}" "${W0_PORT}" 512 \
      --train-iters 1 \
      --lr-decay-iters 1 \
      --lr 0.0 \
      --min-lr 0.0 \
      --save "${W0_DIR}" \
      --save-interval 1 \
      --no-save-optim \
      --no-save-rng
  fi
  if [[ ! -f "${W0_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "ERROR: W0 checkpoint generation did not produce a tracker: ${W0_DIR}" >&2
    exit 3
  fi
fi

printf '%s\n' "${W0_DIR}" >"${OUT_ROOT}/w0_path.txt"

TRAIN_ARGS=(
  --train-iters "${STEPS}"
  --lr-decay-iters "${STEPS}"
  --lr 1.5e-4
  --min-lr 1.5e-4
  --load "${W0_DIR}"
  --finetune
  --no-load-optim
  --no-load-rng
  --exit-on-missing-checkpoint
)

# CP1 must use one GPU.  Running CP1 on both GPUs would silently make it DP2
# and would compare a different global batch against CP2.
run_case cp1 1 "${CP1_DEVICE}" "${CP1_PORT}" 512 "${TRAIN_ARGS[@]}"
run_case cp2 2 "${CP2_DEVICES}" "${CP2_PORT}" 256 "${TRAIN_ARGS[@]}"

CP1_LOG="${OUT_ROOT}/cp1/train.log"
CP2_LOG="${OUT_ROOT}/cp2/train.log"
COMPARE_CSV="${OUT_ROOT}/loss_compare.csv"

# Parse by iteration windows rather than physical lines because torchrun output
# from multiple ranks can interleave.  training.py does not clear its tracker
# after the first print, so logged step 2 is a two-step mean; restore the real
# step-2 value as 2*logged_step2 - logged_step1.
"${PYTHON_BIN}" - "${CP1_LOG}" "${CP2_LOG}" "${COMPARE_CSV}" "${ATOL}" <<'PY'
import csv
from pathlib import Path
import re
import sys

cp1_path, cp2_path, csv_path = map(Path, sys.argv[1:4])
atol = float(sys.argv[4])
start_pattern = re.compile(r"iteration\s+(\d+)\s*/\s*\d+\s*\|")


def parse(path):
    text = path.read_text(errors="replace")
    marks = list(start_pattern.finditer(text))
    rows = []
    for index, mark in enumerate(marks):
        next_start = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        end = min(len(text), mark.start() + 1600, next_start)
        window = text[mark.start():end]

        def scalar(name):
            result = re.search(rf"\b{re.escape(name)}:\s*([^|\s]+)", window)
            return float(result.group(1)) if result else None

        step = int(mark.group(1))
        lm_loss = scalar("lm loss")
        indexer_loss = scalar("indexer loss")
        if lm_loss is None or indexer_loss is None:
            raise SystemExit(f"{path}: iteration {step} is missing lm/indexer loss")
        skipped = re.search(r"number of skipped iterations:\s*(\d+)", window)
        nan = re.search(r"number of nan iterations:\s*(\d+)", window)
        if (skipped and int(skipped.group(1))) or (nan and int(nan.group(1))):
            raise SystemExit(f"{path}: iteration {step} contains skipped/nan iterations")
        rows.append([step, lm_loss, indexer_loss])

    if not rows:
        raise SystemExit(f"{path}: no training losses found")
    if len(rows) >= 2 and rows[1][0] == rows[0][0] + 1:
        rows[1][1] = 2.0 * rows[1][1] - rows[0][1]
        rows[1][2] = 2.0 * rows[1][2] - rows[0][2]
    return {step: (lm, indexer) for step, lm, indexer in rows}


cp1 = parse(cp1_path)
cp2 = parse(cp2_path)
if cp1.keys() != cp2.keys():
    raise SystemExit(f"step mismatch: CP1={list(cp1)}, CP2={list(cp2)}")

header = [
    "step",
    "cp1_lm",
    "cp2_lm",
    "lm_signed_diff",
    "lm_abs_diff",
    "cp1_indexer",
    "cp2_indexer",
    "indexer_signed_diff",
    "indexer_abs_diff",
]
output_rows = []
for step in cp1:
    lm1, idx1 = cp1[step]
    lm2, idx2 = cp2[step]
    output_rows.append(
        [step, lm1, lm2, lm2 - lm1, abs(lm2 - lm1), idx1, idx2, idx2 - idx1, abs(idx2 - idx1)]
    )

with csv_path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(header)
    writer.writerows(output_rows)

print(",".join(header))
for row in output_rows:
    print(f"{row[0]}," + ",".join(f"{value:.10e}" for value in row[1:]))

first = output_rows[0]
first_indexer_abs_diff = first[-1]
max_lm_abs_diff = max(row[4] for row in output_rows)
max_indexer_abs_diff = max(row[8] for row in output_rows)
print(f"FIRST_STEP_INDEXER_ABS_DIFF={first_indexer_abs_diff:.10e}")
print(f"MAX_ABS_LM_DIFF={max_lm_abs_diff:.10e}")
print(f"MAX_ABS_INDEXER_DIFF={max_indexer_abs_diff:.10e}")
print(f"ATOL={atol:.1e}")
print("CP_GT_1_BUG_REPRODUCED=" + ("YES" if first_indexer_abs_diff > atol else "NO"))
print(f"CSV={csv_path}")
PY

echo
echo "Done. Results: ${OUT_ROOT}"
echo "Primary gate: iteration-1 abs(CP2 indexer loss - CP1 indexer loss) > ${ATOL}"
