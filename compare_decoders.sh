#!/bin/bash

# Comparison script to train SegDINO with different decoders
# Run this to systematically compare decoder architectures

DATASET_PATH="/workspace/weedsgalore-dataset"
DINOV3_PATH="/workspace/segdino/dinov3"
BASE_DIR="runs/decoder_comparison"
EPOCHS=500
BATCH_SIZE=32
LR=0.0001

echo "========================================"
echo "SegDINO Decoder Comparison Training"
echo "========================================"
echo "Dataset: $DATASET_PATH"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo ""

# Array of decoders to test
DECODERS=("dpt" "upernet" "segformer" "mask2former")

for DECODER in "${DECODERS[@]}"; do
    echo "----------------------------------------"
    echo "Training with $DECODER decoder..."
    echo "----------------------------------------"
    
    python src/train.py \
        --dataset_path $DATASET_PATH \
        --model segdino_s \
        --decoder $DECODER \
        --in_channels 3 \
        --num_classes 3 \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --epochs $EPOCHS \
        --out_dir ${BASE_DIR}/segdino_s_${DECODER} \
        --log_interval 3 \
        --ckpt_interval 50 \
        --dinov3_path $DINOV3_PATH \
        --pretrained_backbone True \
        --use_class_weights True \
        --loss_type ce \
        --lr_scheduler cosine
    
    echo ""
    echo "Completed: $DECODER"
    echo ""
done

echo "========================================"
echo "All decoder comparisons complete!"
echo "Check results in: $BASE_DIR"
echo "Use TensorBoard to compare:"
echo "  tensorboard --logdir=$BASE_DIR"
echo "========================================"
