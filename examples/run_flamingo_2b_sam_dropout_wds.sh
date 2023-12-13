#!/bin/bash

#SBATCH -A llmservice_nlp_fm
#SBATCH -p luna
#SBATCH -t 4:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton
#SBATCH --nodes=8
#SBATCH --job-name=llmservice_nlp_fm-megatron-dev:flamingo-2b-COCO-overfit-sam-dropout-mr-wds

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

SEQ_LEN=96

NAME="flamingo-2b-1node-COCO-overfit-sam-dropout-mr-wds"
LOAD_NAME="gpt3-2b-multi-1.1t-gtc"

SCRIPTS_DIR="/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/source"
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
VISUAL_LOAD_DIR="/lustre/fsw/adlr/adlr-nlp/zhuoliny/checkpoints/SAM_L_16"
VISUAL_SAVE_DIR="${FINETUNE_DIR}/${VISUAL_TYPE}"

PROMPT_PATH="${SOURCE}/GPT4-prompts.json"
DATASET_CONFIG="${SOURCE}/dataset.yaml"

options=" \
    --use-flash-attn \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --rotary-percent 0.5 \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.1 \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --seq-length ${SEQ_LEN} \
    --ds-seq-length 512 \
    --max-position-embeddings 4096 \
    --cyclic-train-iters 100000000 \
    --train-samples 410000 \
    --micro-batch-size 1 \
    --global-batch-size 256 \
    --lr-decay-samples 25600000 \
    --lr-warmup-samples 83200 \
    --lr 1e-5 \
    --min-lr 2.5e-6 \
    --lr-decay-style cosine \
    --log-interval 10 \
    --eval-iters 10 \
    --eval-interval 1000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_TRAIN} \
    --valid-path ${DATA_VALID} \
    --prompt-path ${PROMPT_PATH} \
    --dset-config ${DATASET_CONFIG} \
    --save-interval 2000 \
    --save ${FINETUNE_DIR} \
    --load ${CHECKPOINT_DIR} \
    --split 100,0,0 \
    --clip-grad 52.0 \
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
    --dataset-type nvgpt4 \
    --SAM-randinit \
    --align-to-old \
    --fp32SAM \
    --tensorboard-dir ${TENSORBOARD_DIR}"

# torchrun --nproc-per-node 8 ${SOURCE}/pretrain_flamingo.py ${options}
# python -u ${SOURCE}/pretrain_flamingo.py ${options}
run_cmd="python -u ${SOURCE}/pretrain_flamingo.py ${options}"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

srun -l --verbose \
    --container-image /lustre/fsw/adlr/adlr-nlp/jbarker/checkpoints/adlr+megatron-lm+pytorch+23.04-py3-jbarker.sqsh \
    --container-mounts "/lustre" \
    --output=${LOGS_DIR}/%x_%j_$DATETIME.log \
    sh -c "${run_cmd}"

set +x
