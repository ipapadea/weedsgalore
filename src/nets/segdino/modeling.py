# /home/ilias/weedsgalore/src/nets/segdino/modeling.py
"""
SegDINO models for WeedsGalore dataset.
Adapted from the original SegDINO implementation.

Available decoders:
- dpt: Original DPT decoder (basic, fast)
- upernet: UperNet with PPM (better multi-scale, moderate complexity)
- segformer: SegFormer MLP decoder (lightweight, efficient)
- mask2former: Mask2Former with learnable queries (best quality, slower)
"""
import torch
import sys
import os
from .dpt import DPT
from .upernet_decoder import UperNetDPT
from .segformer_decoder import SegFormerDPT
from .mask2former_decoder import Mask2FormerDPT

def segdino_vitb16(num_classes=3, pretrained_backbone=True, dinov3_path=None, 
                   dino_ckpt=None, decoder='dpt'):
    """
    SegDINO with ViT-B/16 backbone.
    
    Args:
        num_classes: Number of output classes
        pretrained_backbone: Whether to use pretrained DINOv3 weights
        dinov3_path: Path to dinov3 repository (default: looks for it in segdino folder)
        dino_ckpt: Path to pretrained DINOv3 checkpoint
        decoder: Decoder type ('dpt', 'upernet', 'segformer', 'mask2former')
    """
    # Find dinov3 repo
    if dinov3_path is None:
        # Try to find dinov3 in the segdino repository
        segdino_repo = os.path.join(os.path.dirname(__file__), '../../../../segdino')
        dinov3_path = os.path.join(segdino_repo, 'dinov3')
    
    if not os.path.exists(dinov3_path):
        raise ValueError(f"DINOv3 repository not found at {dinov3_path}. "
                        "Please specify dinov3_path or ensure segdino repo is accessible.")
    
    # Load DINOv3 backbone
    if dino_ckpt is None:
        dino_ckpt = os.path.join(dinov3_path, 'dinov3_vitb16_pretrain.pth')
    
    if pretrained_backbone and not os.path.exists(dino_ckpt):
        raise ValueError(f"DINOv3 checkpoint not found at {dino_ckpt}")
    
    # Load backbone and verify weights
    print(f"\nLoading DINOv3 ViT-B/16 backbone...")
    if pretrained_backbone:
        print(f"  Pretrained weights: {dino_ckpt}")
        # Check if checkpoint file exists and get its size
        ckpt_size = os.path.getsize(dino_ckpt) / (1024 * 1024)  # MB
        print(f"  Checkpoint size: {ckpt_size:.1f} MB")
    else:
        print(f"  No pretrained weights (training from scratch)")
    
    backbone = torch.hub.load(dinov3_path, 'dinov3_vitb16', source='local', 
                             weights=dino_ckpt if pretrained_backbone else None)
    
    # Verify weights were loaded
    if pretrained_backbone:
        # Check a sample parameter to confirm it's not random
        sample_param = next(backbone.parameters())
        param_mean = sample_param.mean().item()
        param_std = sample_param.std().item()
        print(f"  Backbone loaded: {sum(p.numel() for p in backbone.parameters()) / 1e6:.1f}M parameters")
        print(f"  Sample weight stats - mean: {param_mean:.4f}, std: {param_std:.4f}")
        print(f"  ✓ Pretrained weights loaded successfully")
    
    # Create model with selected decoder
    if decoder == 'dpt':
        model = DPT(
            encoder_size='base',
            nclass=num_classes,
            features=128,
            out_channels=[96, 192, 384, 768],
            backbone=backbone
        )
    elif decoder == 'upernet':
        model = UperNetDPT(
            encoder_size='base',
            nclass=num_classes,
            fpn_channels=256,
            backbone=backbone
        )
    elif decoder == 'segformer':
        model = SegFormerDPT(
            encoder_size='base',
            nclass=num_classes,
            embed_dim=256,
            backbone=backbone
        )
    elif decoder == 'mask2former':
        model = Mask2FormerDPT(
            encoder_size='base',
            nclass=num_classes,
            num_queries=100,
            backbone=backbone
        )
    else:
        raise ValueError(f"Unknown decoder: {decoder}. Choose from: dpt, upernet, segformer, mask2former")
    
    return model

def segdino_vits16(num_classes=3, pretrained_backbone=True, dinov3_path=None, 
                   dino_ckpt=None, decoder='dpt'):
    """
    SegDINO with ViT-S/16 backbone.
    
    Args:
        num_classes: Number of output classes
        pretrained_backbone: Whether to use pretrained DINOv3 weights
        dinov3_path: Path to dinov3 repository
        dino_ckpt: Path to pretrained DINOv3 checkpoint
        decoder: Decoder type ('dpt', 'upernet', 'segformer', 'mask2former')
    """
    # Find dinov3 repo
    if dinov3_path is None:
        segdino_repo = os.path.join(os.path.dirname(__file__), '../../../../segdino')
        dinov3_path = os.path.join(segdino_repo, 'dinov3')
    
    if not os.path.exists(dinov3_path):
        raise ValueError(f"DINOv3 repository not found at {dinov3_path}")
    
    # Load DINOv3 backbone
    if dino_ckpt is None:
        dino_ckpt = os.path.join(dinov3_path, 'dinov3_vits16_pretrain.pth')
    
    if pretrained_backbone and not os.path.exists(dino_ckpt):
        raise ValueError(f"DINOv3 checkpoint not found at {dino_ckpt}")
    
    # Load backbone and verify weights
    print(f"\nLoading DINOv3 ViT-S/16 backbone...")
    if pretrained_backbone:
        print(f"  Pretrained weights: {dino_ckpt}")
        # Check if checkpoint file exists and get its size
        ckpt_size = os.path.getsize(dino_ckpt) / (1024 * 1024)  # MB
        print(f"  Checkpoint size: {ckpt_size:.1f} MB")
    else:
        print(f"  No pretrained weights (training from scratch)")
    
    backbone = torch.hub.load(dinov3_path, 'dinov3_vits16', source='local',
                             weights=dino_ckpt if pretrained_backbone else None)
    
    # Verify weights were loaded
    if pretrained_backbone:
        # Check a sample parameter to confirm it's not random
        sample_param = next(backbone.parameters())
        param_mean = sample_param.mean().item()
        param_std = sample_param.std().item()
        print(f"  Backbone loaded: {sum(p.numel() for p in backbone.parameters()) / 1e6:.1f}M parameters")
        print(f"  Sample weight stats - mean: {param_mean:.4f}, std: {param_std:.4f}")
        print(f"  ✓ Pretrained weights loaded successfully")
    
    # Create model with selected decoder
    if decoder == 'dpt':
        model = DPT(
            encoder_size='small',
            nclass=num_classes,
            features=128,
            out_channels=[96, 192, 384, 384],
            backbone=backbone
        )
    elif decoder == 'upernet':
        model = UperNetDPT(
            encoder_size='small',
            nclass=num_classes,
            fpn_channels=256,
            backbone=backbone
        )
    elif decoder == 'segformer':
        model = SegFormerDPT(
            encoder_size='small',
            nclass=num_classes,
            embed_dim=256,
            backbone=backbone
        )
    elif decoder == 'mask2former':
        model = Mask2FormerDPT(
            encoder_size='small',
            nclass=num_classes,
            num_queries=100,
            backbone=backbone
        )
    else:
        raise ValueError(f"Unknown decoder: {decoder}. Choose from: dpt, upernet, segformer, mask2former")
    
    return model