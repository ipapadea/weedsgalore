#!/bin/bash
# train_segdino_s.sh - Train SegDINO ViT-S/16

export MODEL=segdino_s
export IN_CHANNELS=3
export NUM_CLASSES=3
export BATCH_SIZE=8
export EPOCHS=100
export LR=0.0001
export OUT_DIR=runs/segdino_s_3classes

./run_train_docker.sh