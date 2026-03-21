#!/usr/bin/env bash

set -x  # Print each command for debugging

# --------- Environment Variables ---------
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2

export DS_LOG_LEVEL=WARN
export ACCELERATE_LOG_LEVEL=WARN

# --------- Directory and Paths ---------
EXP_DIR="exps/swig"
DATA_DIR="/path/to/your/datasets/swig_hoi"
DINO_DIR="/path/to/your/weights/dinov3"

CONFIG_FILE="configs/swig.yaml"
DEFAULT_CONFIG="configs/base.yaml"

# --------- Training Phase ---------
accelerate launch \
    --config_file "configs/accelerate_config.yaml" \
    --num_processes=8 \
    --main_process_port=12847 \
    train.py \
    -c ${CONFIG_FILE} \
    --default-config ${DEFAULT_CONFIG} \
    SOLVER.BATCH_SIZE=32 \
    RUNTIME.OUTPUT_DIR="${EXP_DIR}" \
    RUNTIME.NUM_WORKERS=2 \
    INPUT.PATH="${DATA_DIR}" \
    MODEL.DINO.DINOTXT_WEIGHTS="${DINO_DIR}/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth" \
    MODEL.DINO.BACKBONE_WEIGHTS="${DINO_DIR}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
    MODEL.DINO.BPE_PATH_OR_URL="${DINO_DIR}/bpe_simple_vocab_16e6.txt.gz" \
    ZERO_SHOT.CLASSIFIER.EVAL="params/swig/classifier_swig_dict.pt"


# --------- Evaluation Phase ---------

# Find the saved yaml config file in EXP_DIR
SAVED_CONFIG=$(ls ${EXP_DIR}/*.yaml 2>/dev/null | head -n 1)

if [ -f "$SAVED_CONFIG" ]; then
    EVAL_CONFIG="$SAVED_CONFIG"
    echo "Found saved config: ${EVAL_CONFIG}"
else
    EVAL_CONFIG="${CONFIG_FILE}"
    echo "Saved config not found, using default: ${EVAL_CONFIG}"
fi

# Find all checkpoint directories and sort them numerically by epoch
CHECKPOINTS=$(ls -d ${EXP_DIR}/checkpoint-epoch-* 2>/dev/null | sort -t '-' -k 3 -n)

if [ -z "$CHECKPOINTS" ]; then
    echo "Error: No checkpoints found in ${EXP_DIR}!"
    exit 1
fi

# Loop through each checkpoint and run evaluation
for CKPT in $CHECKPOINTS; do
    EPOCH_NUM=$(basename "$CKPT" | cut -d'-' -f3)
    EVAL_OUTPUT_DIR="${CKPT}/eval"

    echo "--------------------------------------"
    echo "Evaluating Epoch: ${EPOCH_NUM} (Path: ${CKPT})"
    
    accelerate launch \
        --config_file "configs/accelerate_config.yaml" \
        --num_processes=8 \
        --main_process_port=12888 \
        train.py \
        -c ${EVAL_CONFIG} \
        --default-config ${DEFAULT_CONFIG} \
        RUNTIME.EVAL=true \
        RUNTIME.PRETRAINED=${CKPT} \
        RUNTIME.OUTPUT_DIR=${EVAL_OUTPUT_DIR} \
        RUNTIME.GLOBAL_OUTPUT_DIR=${EXP_DIR} \
        RUNTIME.NUM_WORKERS=2
        
    echo "Evaluation for Epoch ${EPOCH_NUM} done."
done
