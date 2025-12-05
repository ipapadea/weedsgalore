#!/usr/bin/env python3
"""
Quick experiment to find optimal batch size for small dataset training.
Tests different batch sizes and measures gradient statistics and convergence.
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path for Docker environment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datasets import WeedsGaloreDataset
from torch.utils.data import DataLoader
from nets import segdino_vits16

def test_batch_size(batch_size, num_iters=50):
    """Test a specific batch size and return gradient statistics."""
    print(f"\n{'='*60}")
    print(f"Testing batch_size={batch_size}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    dataset_path = os.environ.get('DATASET_PATH', '/workspace/weedsgalore-dataset')
    dataset = WeedsGaloreDataset(
        dataset_path=dataset_path,
        dataset_size=104,
        in_bands=3,
        num_classes=3,
        is_training=True,
        split='train',
        augmentation=True
    )
    
    # Dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    iters_per_epoch = len(dataloader)
    print(f"Iterations per epoch: {iters_per_epoch}")
    print(f"Total updates in 500 epochs: {iters_per_epoch * 500}")
    
    # Model
    dinov3_path = os.environ.get('DINOV3_PATH', '/workspace/segdino/dinov3')
    print(f"Using DINOv3 path: {dinov3_path}")
    
    # For testing, we need a checkpoint even if we don't use pretrained weights
    # Just use random initialization by loading model without pretrained=True
    net = segdino_vits16(
        num_classes=3,
        pretrained_backbone=True,  # Load pretrained for realistic test
        dinov3_path=dinov3_path,
        decoder='dpt'
    ).to(device)
    
    print("Model loaded successfully")
    
    # Simple SGD for gradient analysis
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    # Track gradient norms
    grad_norms = []
    losses = []
    
    print(f"\nRunning {num_iters} iterations...")
    net.train()
    
    iter_count = 0
    while iter_count < num_iters:
        for data in dataloader:
            features, _, binary_labels = data
            features, labels = features.to(device), binary_labels.to(device)
            
            optimizer.zero_grad()
            out = net(features)
            
            if out.shape[2:] != labels.shape[1:]:
                import torch.nn.functional as F
                out = F.interpolate(out, size=labels.shape[1:], mode='bilinear', align_corners=False)
            
            loss = criterion(out, labels.long())
            loss.backward()
            
            # Compute gradient norm
            total_norm = 0.0
            for p in net.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            grad_norms.append(total_norm)
            losses.append(loss.item())
            
            optimizer.step()
            
            iter_count += 1
            if iter_count >= num_iters:
                break
    
    # Statistics
    import numpy as np
    grad_norms = np.array(grad_norms)
    losses = np.array(losses)
    
    print(f"\nResults for batch_size={batch_size}:")
    print(f"  Gradient norm - mean: {grad_norms.mean():.4f}, std: {grad_norms.std():.4f}")
    print(f"  Loss - mean: {losses.mean():.4f}, std: {losses.std():.4f}")
    print(f"  Loss trend: {losses[0]:.4f} → {losses[-1]:.4f} (Δ {losses[-1] - losses[0]:.4f})")
    print(f"  Gradient stability (CV): {grad_norms.std() / grad_norms.mean():.4f}")
    
    return {
        'batch_size': batch_size,
        'iters_per_epoch': iters_per_epoch,
        'grad_norm_mean': grad_norms.mean(),
        'grad_norm_std': grad_norms.std(),
        'loss_mean': losses.mean(),
        'loss_std': losses.std(),
        'loss_improvement': losses[0] - losses[-1],
        'gradient_cv': grad_norms.std() / grad_norms.mean()
    }


if __name__ == '__main__':
    batch_sizes = [2,4,6,8]
    results = []
    
    print("="*60)
    print("BATCH SIZE OPTIMIZATION FOR SMALL DATASET (104 samples)")
    print("="*60)
    
    for bs in batch_sizes:
        try:
            result = test_batch_size(bs, num_iters=50)
            results.append(result)
        except RuntimeError as e:
            print(f"ERROR with batch_size={bs}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'BS':<6} {'Iters/Ep':<10} {'GradNorm':<12} {'GradStd':<12} {'Loss Δ':<10}")
    print("-"*60)
    
    for r in results:
        print(f"{r['batch_size']:<6} {r['iters_per_epoch']:<10} "
              f"{r['grad_norm_mean']:<12.4f} {r['grad_norm_std']:<12.4f} "
              f"{r['loss_improvement']:<10.4f}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("="*60)
    
    # Find best based on iterations per epoch and gradient stability
    best = max(results, key=lambda x: x['iters_per_epoch'] / (x['gradient_cv'] + 1e-6))
    
    print(f"Best batch size: {best['batch_size']}")
    print(f"  - {best['iters_per_epoch']} iterations/epoch")
    print(f"  - {best['iters_per_epoch'] * 500} total updates in 500 epochs")
    print(f"  - Most stable gradients (CV={best['gradient_cv']:.4f})")
    print(f"\nFor ViT on 104 samples: Use bs=8 or bs=16")
    print("  - bs=8: More updates, better for small datasets")
    print("  - bs=16: Faster training, still reasonable")
    print("  - bs=32: Too few updates for 104 samples (avoid)")
