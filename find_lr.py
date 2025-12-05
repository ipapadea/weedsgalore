#!/usr/bin/env python3
"""
Learning Rate Finder for WeedsGalore
Implements the LR Range Test from Smith (2017) "Cyclical Learning Rates for Training Neural Networks"
"""

from absl import app, flags
import torch
import torch.nn.functional as F
from datasets import WeedsGaloreDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_path', '/home/ilias/weedsgalore-dataset', 'dataset directory')
flags.DEFINE_integer('in_channels', 3, 'options: 3 (RGB), 5 (MSI)')
flags.DEFINE_integer('num_classes', 3, 'options: 3 (uni-weed), 6 (multi-weed)')
flags.DEFINE_string('model', 'segdino_s', 'Model architecture: dlv3p, segdino_b, segdino_s')
flags.DEFINE_string('dinov3_path', '/home/ilias/segdino/dinov3', 'Path to DINOv3 repository')
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_float('start_lr', 1e-7, 'starting learning rate')
flags.DEFINE_float('end_lr', 1.0, 'ending learning rate')
flags.DEFINE_integer('num_iter', 100, 'number of iterations')
flags.DEFINE_string('output', 'lr_finder.png', 'output plot filename')

def find_lr(model, train_loader, optimizer, criterion, device, 
            start_lr=1e-7, end_lr=1.0, num_iter=100):
    """
    Perform LR range test.
    
    Returns:
        lrs: list of learning rates tested
        losses: list of corresponding losses
    """
    model.train()
    
    # Calculate the multiplication factor
    num = num_iter
    mult = (end_lr / start_lr) ** (1 / num)
    
    lr = start_lr
    optimizer.param_groups[0]['lr'] = lr
    
    avg_loss = 0.0
    best_loss = float('inf')
    batch_num = 0
    losses = []
    lrs = []
    
    iterator = iter(train_loader)
    
    for iteration in range(num_iter):
        try:
            features, unique_labels, binary_labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            features, unique_labels, binary_labels = next(iterator)
        
        labels = binary_labels if FLAGS.num_classes == 3 else unique_labels
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        
        # Resize if needed
        if outputs.shape[2:] != labels.shape[1:]:
            outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        
        loss = criterion(outputs, labels.long())
        
        # Compute smoothed loss
        avg_loss = 0.98 * avg_loss + 0.02 * loss.item()
        smoothed_loss = avg_loss / (1 - 0.98 ** (batch_num + 1))
        
        # Check if loss is diverging
        if batch_num > 0 and smoothed_loss > 4 * best_loss:
            print(f"Stopping early at iteration {batch_num}, loss is diverging")
            break
        
        # Record best loss
        if smoothed_loss < best_loss or batch_num == 0:
            best_loss = smoothed_loss
        
        # Store values
        losses.append(smoothed_loss)
        lrs.append(lr)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
        batch_num += 1
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}/{num_iter}, LR: {lr:.2e}, Loss: {smoothed_loss:.4f}")
    
    return lrs, losses

def plot_lr_finder(lrs, losses, output_file='lr_finder.png'):
    """Plot the LR finder results."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot loss vs lr
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Learning Rate Finder', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Find the steepest gradient point (suggests good LR)
    gradients = np.gradient(losses)
    min_gradient_idx = np.argmin(gradients)
    suggested_lr = lrs[min_gradient_idx]
    
    # Mark the suggested LR
    ax.axvline(x=suggested_lr, color='r', linestyle='--', 
               label=f'Steepest descent: {suggested_lr:.2e}')
    
    # Also mark 1/10th of the suggested LR (common recommendation)
    conservative_lr = suggested_lr / 10
    ax.axvline(x=conservative_lr, color='g', linestyle='--',
               label=f'Conservative (1/10): {conservative_lr:.2e}')
    
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to: {output_file}")
    print(f"\nRecommended learning rates:")
    print(f"  - Aggressive: {suggested_lr:.2e}")
    print(f"  - Conservative: {conservative_lr:.2e}")
    print(f"  - Safe range: {conservative_lr:.2e} to {suggested_lr:.2e}")

def main(_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    
    # Dataset
    train_dataset = WeedsGaloreDataset(
        dataset_path=FLAGS.dataset_path, 
        dataset_size=104,
        in_bands=FLAGS.in_channels,
        num_classes=FLAGS.num_classes, 
        is_training=True, 
        split='train', 
        augmentation=True
    )
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=FLAGS.batch_size, 
        shuffle=True,
        num_workers=4, 
        drop_last=True
    )
    
    # Model
    if FLAGS.model == 'segdino_s':
        from nets import segdino_vits16
        model = segdino_vits16(
            num_classes=FLAGS.num_classes,
            pretrained_backbone=True,
            dinov3_path=FLAGS.dinov3_path
        )
    elif FLAGS.model == 'segdino_b':
        from nets import segdino_vitb16
        model = segdino_vitb16(
            num_classes=FLAGS.num_classes,
            pretrained_backbone=True,
            dinov3_path=FLAGS.dinov3_path
        )
    else:
        from nets import deeplabv3plus_resnet50
        model = deeplabv3plus_resnet50(num_classes=FLAGS.num_classes, pretrained_backbone=True)
    
    model.to(device)
    
    # Loss and optimizer
    class_weights = torch.tensor([1.41, 8.53, 7.53], device=device) if FLAGS.num_classes == 3 else None
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.start_lr)
    
    print(f"\nRunning LR finder from {FLAGS.start_lr:.2e} to {FLAGS.end_lr:.2e}")
    print(f"Number of iterations: {FLAGS.num_iter}\n")
    
    # Find LR
    lrs, losses = find_lr(
        model, train_loader, optimizer, criterion, device,
        start_lr=FLAGS.start_lr,
        end_lr=FLAGS.end_lr,
        num_iter=FLAGS.num_iter
    )
    
    # Plot results
    plot_lr_finder(lrs, losses, FLAGS.output)

if __name__ == '__main__':
    app.run(main)
