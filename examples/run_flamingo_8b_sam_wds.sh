#!/bin/bash

#SBATCH -A llmservice_nlp_fm
#SBATCH -p luna
#SBATCH -t 4:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton
#SBATCH --nodes=1
#SBATCH --job-name=llmservice_nlp_fm-megatron-dev:flamingo-8b-COCO-overfit-sam-mr-wds

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

SEQ_LEN=4096

NAME="flamingo-8b-2node-profile"
LOAD_NAME="gpt3-8b-multi-1.1t-gtc"

SCRIPTS_DIR="/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/source/"
SOURCE="/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/source/megatron-lm"

OUTPUT="/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/output/${NAME}"
mkdir -p ${OUTPUT}/logs

FINETUNE_DIR="${OUTPUT}"
LOGS_DIR="${OUTPUT}/logs"
CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/checkpoints/${LOAD_NAME}"

TENSORBOARD_DIR="${OUTPUT}/tensorboard"
mkdir -p ${TENSORBOARD_DIR}

DATA_TRAIN="/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/data/coco.yaml"
DATA_VALID="/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/data/coco.yaml"

VISUAL_ARCH="SAM_L"
VISUAL_TYPE="sam"
VISUAL_LOAD_DIR="/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/checkpoint/SAM_L_16_tp4_pp1"
VISUAL_SAVE_DIR="${FINETUNE_DIR}/${VISUAL_TYPE}"

PROMPT_PATH="${SOURCE}/GPT4-prompts.json"
DATASET_CONFIG="${SOURCE}/dataset.yaml"

#    --sequence-parallel \
#    --recompute-activations \

options=" \
    --use-flash-attn \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
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
    --micro-batch-size 1 \
    --global-batch-size 256 \
    --lr-decay-samples 25600000 \
    --lr-warmup-samples 83200 \
    --lr 2.0e-5 \
    --min-lr 2.0e-6 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 10 \
    --eval-interval 1000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_TRAIN} \
    --valid-path ${DATA_VALID} \
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
    --dataset-type nvgpt4 \
    --dataloader-type cyclic --no-data-sharding \
    --tensorboard-dir ${TENSORBOARD_DIR}"

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 ${SOURCE}/pretrain_flamingo.py ${options}
run_cmd="${SCRIPTS_DIR}/bind.sh --cpu=${SCRIPTS_DIR}/dgxa100_ccx.sh --mem=${SCRIPTS_DIR}/dgxa100_ccx.sh nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true -o ${SOURCE}/8b_profile python -u ${SOURCE}/pretrain_flamingo.py ${options}"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

srun -l \
    --container-image /lustre/fsw/adlr/adlr-nlp/jbarker/checkpoints/adlr+megatron-lm+pytorch+23.04-py3-jbarker.sqsh \
    --container-mounts "/lustre" \
    --output=${LOGS_DIR}/%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x
