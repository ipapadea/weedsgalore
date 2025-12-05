# ---------------------------------------------------------
# Base CUDA image (Ubuntu 22.04) -> works perfectly on Ubuntu 24.04 host
# ---------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------
# Install system dependencies
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-distutils \
    python3-pip git wget curl build-essential \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.11 the default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------
# Install PyTorch 2.4.1 + CUDA 12.4
# ---------------------------------------------------------
RUN pip install \
    torch==2.4.1+cu124 \
    torchvision==0.19.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------
# Install Python dependencies (DINOv3 + segmentation + WeedsGalore)
# ---------------------------------------------------------
RUN pip install \
    ftfy \
    iopath \
    omegaconf \
    pandas \
    regex \
    scikit-learn \
    scikit-learn-intelex \
    submitit \
    termcolor \
    torchmetrics \
    opencv-python \
    absl-py \
    pillow \
    tensorboard \
    matplotlib

# ---------------------------------------------------------
# Debug info
# ---------------------------------------------------------
RUN python - <<EOF
import torch, torchvision, cv2
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Torchvision:", torchvision.__version__)
print("OpenCV:", cv2.__version__)
EOF