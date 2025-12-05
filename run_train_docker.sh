#!/bin/bash
# run_train_docker.sh - Run WeedsGalore training in Docker

echo "Building Docker image..."
docker build -t weedsgalore:latest .

# Default parameters
MODEL=${MODEL:-dlv3p}
IN_CHANNELS=${IN_CHANNELS:-5}
NUM_CLASSES=${NUM_CLASSES:-3}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-100}
LR=${LR:-0.001}
OUT_DIR=${OUT_DIR:-runs/experiment_$(date +%Y%m%d_%H%M%S)}
GPU_ID=${GPU_ID:-3}  # GPU 4 (0-indexed, so 3 = 4th GPU)

echo "=========================================="
echo "Training Configuration:"
echo "  Model: $MODEL"
echo "  GPU: $GPU_ID (GPU $(($GPU_ID + 1)))"
echo "  Input Channels: $IN_CHANNELS"
echo "  Number of Classes: $NUM_CLASSES"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"
echo "  Output Directory: $OUT_DIR"
echo "=========================================="

# Run training with specific GPU
docker run --rm --gpus "device=$GPU_ID" \
    --ipc=host --shm-size=16g \
    -v $(pwd):/workspace/weedsgalore \
    -v /home/ilias/weedsgalore-dataset:/workspace/weedsgalore-dataset:ro \
    -v /home/ilias/segdino:/workspace/segdino:ro \
    -v $(pwd)/runs:/workspace/weedsgalore/runs \
    -w /workspace/weedsgalore \
    -e PYTHONPATH=/workspace \
    -e CUDA_VISIBLE_DEVICES=$GPU_ID \
    weedsgalore:latest \
    python3 src/train.py \
        --dataset_path /workspace/weedsgalore-dataset \
        --dataset_size_train 104 \
        --model $MODEL \
        --in_channels $IN_CHANNELS \
        --num_classes $NUM_CLASSES \
        --batch_size $BATCH_SIZE \
        --num_workers 4 \
        --lr $LR \
        --epochs $EPOCHS \
        --out_dir $OUT_DIR \
        --log_interval 25 \
        --ckpt_interval 100 \
        --dinov3_path /workspace/segdino/dinov3 \
        --pretrained_backbone False \
        "$@"

echo "Training complete! Results saved to: $OUT_DIR"
