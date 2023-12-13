#!/bin/bash

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

SEQ_LEN=256

NAME="flamingo-8b-2node-profile"
#LOAD_NAME="gpt3-8b-multi-1.1t-gtc"
LOAD_NAME="dummy"

SCRIPTS_DIR="/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/source/"
SOURCE="/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/source/megatron-lm"

OUTPUT="/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/output/${NAME}"
mkdir -p ${OUTPUT}/logs

FINETUNE_DIR="${OUTPUT}"
LOGS_DIR="${OUTPUT}/logs"
CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/checkpoints/${LOAD_NAME}"

TENSORBOARD_DIR="${OUTPUT}/tensorboard"
mkdir -p ${TENSORBOARD_DIR}

DATA_TRAIN="1.0000 /lustre/fsw/adlr/adlr-nlp/zhuoliny/debug_folder/COCO_train_mmdata_512sl_256k_vocab"

VISUAL_ARCH="SAM_L"
VISUAL_TYPE="sam"
#VISUAL_LOAD_DIR="/lustre/fsw/adlr/adlr-nlp/zhuoliny/checkpoints/SAM_L_16"
VISUAL_LOAD_DIR="/lustre/fsw/adlr/adlr-nlp/zhuoliny/checkpoints/dummy"
VISUAL_SAVE_DIR="${FINETUNE_DIR}/${VISUAL_TYPE}"

PROMPT_PATH="${SOURCE}/GPT4-prompts.json"
DATASET_CONFIG="${SOURCE}/dataset.yaml"

options=" \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --use-distributed-optimizer \
    --use-flash-attn \
    --overlap-p2p-communication \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --rotary-percent 0.5 \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 4 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length ${SEQ_LEN} \
    --ds-seq-length 512 \
    --max-position-embeddings 4096 \
    --cyclic-train-iters 100000000 \
    --train-samples 131072 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --lr-decay-samples 25600000 \
    --lr-warmup-samples 83200 \
    --lr 2.0e-5 \
    --min-lr 2.0e-6 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 10 \
    --eval-interval 1000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_TRAIN} \
    --prompt-path ${PROMPT_PATH} \
    --dset-config ${DATASET_CONFIG} \
    --save-interval 1000 \
    --save ${FINETUNE_DIR} \
    --load ${CHECKPOINT_DIR} \
    --split 60,40,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --add-gated-xattn \
    --add-BOS \
    --visual-arch ${VISUAL_ARCH} \
    --visual-path ${VISUAL_LOAD_DIR} \
    --visual-type ${VISUAL_TYPE} \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --DDP-impl local \
    --no-load-optim \
    --eod-mask-loss \
    --finetune \
    --perceiver-type none \
    --freeze-LM \
    --freeze-ViT \
    --img-h 1024 \
    --img-w 1024 \
    --dataloader-type cyclic --no-data-sharding \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --profile \
    --profile-step-start 10 \
    --profile-step-end 13"

#torchrun --nproc_per_node 8 ${SOURCE}/pretrain_flamingo.py ${options}
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --cuda-memory-usage=false --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true -o 8b_profile torchrun --nproc_per_node 8 ${SOURCE}/pretrain_flamingo.py ${options}