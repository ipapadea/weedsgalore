# SPDX-FileCopyrightText: 2025 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2025 Ekin Celikkan <ekin.celikkan@gfz.de>
# SPDX-License-Identifier: Apache-2.0

from absl import app, flags
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import WeedsGaloreDataset
from torch.utils.data import DataLoader
from nets import deeplabv3plus_resnet50, deeplabv3plus_resnet50_do
from pathlib import Path
from torchmetrics.classification import MulticlassJaccardIndex
from torch.utils.tensorboard import SummaryWriter
import os
import sys


class Tee:
    """Write to both file and stdout."""
    def __init__(self, *files):
        self.files = files
    
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-1):
        """
        Args:
            alpha: Class weights (tensor or None)
            gamma: Focusing parameter (default: 2.0)
            ignore_index: Index to ignore in loss calculation
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C, H, W) - raw logits from model
            targets: (N, H, W) - ground truth labels
        """
        # Get cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Get probabilities
        p = torch.exp(-ce_loss)
        
        # Apply focal term: (1 - p)^gamma
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            # Create alpha tensor for each pixel
            alpha_t = self.alpha[targets]
            # Mask out ignored indices
            if self.ignore_index >= 0:
                alpha_t = torch.where(targets != self.ignore_index, alpha_t, torch.zeros_like(alpha_t))
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_path', 'weedsgalore-dataset', 'dataset directory')
flags.DEFINE_integer('dataset_size_train', 104, 'dataset size of train set')
flags.DEFINE_integer('in_channels', 5, 'options: 3 (RGB), 5 (MSI)')
flags.DEFINE_integer('num_classes', 6, 'options: 3 (uni-weed), 6 (multi-weed)')
flags.DEFINE_integer('ignore_index', -1, 'ignore during loss and iou calculation')
flags.DEFINE_boolean('dlv3p_do', False, 'set True to use probabilistic variant of DLv3+ with dropout')
flags.DEFINE_boolean('pretrained_backbone', True, 'set True to use pretrained ResNet50 backbone')
flags.DEFINE_string('ckpt_resnet', 'ckpts/resnet50-19c8e357.pth', 'ckpt path for pretrained backbone')
flags.DEFINE_integer('batch_size', 2, 'batch size')
flags.DEFINE_integer('num_workers', 4, 'number of subprocesses')
flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_integer('epochs', 10, 'number of epochs for training')
flags.DEFINE_string('out_dir', 'out_dir', 'directory to save logs and ckpts')
flags.DEFINE_integer('log_interval', 25, 'number of iterations to log scalars')
flags.DEFINE_integer('ckpt_interval', 500, 'number of iterations to save ckpts')

# SegDINO-specific flags
flags.DEFINE_string('model', 'dlv3p', 'Model architecture: dlv3p, dlv3p_do, segdino_b, segdino_s')
flags.DEFINE_string('dinov3_path', '/home/ilias/segdino/dinov3', 'Path to DINOv3 repository')
flags.DEFINE_string('dino_ckpt', '', 'Path to DINOv3 pretrained checkpoint (auto-detect if empty)')
flags.DEFINE_string('decoder', 'dpt', 'Decoder architecture: dpt (basic), upernet (PPM+FPN), segformer (MLP), mask2former (transformer)')
flags.DEFINE_boolean('use_class_weights', True, 'Use class weights to handle imbalance')
flags.DEFINE_string('loss_type', 'focal', 'Loss type: ce (CrossEntropy) or focal (FocalLoss)')
flags.DEFINE_float('focal_gamma', 2.0, 'Focal loss gamma parameter (default: 2.0)')
flags.DEFINE_string('lr_scheduler', 'none', 'LR scheduler: none, cosine, step, plateau')
flags.DEFINE_float('lr_decay', 0.1, 'LR decay factor for step/plateau scheduler')
flags.DEFINE_integer('lr_step_size', 50, 'Epoch interval for step scheduler')

def main(_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    if device.type == 'cuda':
        print(f"Cuda current device: {torch.cuda.current_device()}")
        print(f"Cuda device name: {torch.cuda.get_device_name(0)}")

    # Dataset
    train_dataset = WeedsGaloreDataset(dataset_path=FLAGS.dataset_path, dataset_size=FLAGS.dataset_size_train, in_bands=FLAGS.in_channels,
                                        num_classes=FLAGS.num_classes, is_training=True, split='train', augmentation=True)
    val_dataset = WeedsGaloreDataset(dataset_path=FLAGS.dataset_path, dataset_size=None, in_bands=FLAGS.in_channels,
                                        num_classes=FLAGS.num_classes, is_training=False, split='val', augmentation=False)

    # Dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=FLAGS.batch_size, shuffle=True,
                                  num_workers=FLAGS.num_workers, collate_fn=None, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=FLAGS.batch_size, shuffle=False,
                                num_workers=FLAGS.num_workers, collate_fn=None, drop_last=True)

    # Network
    if FLAGS.model == 'segdino_b':
        from nets import segdino_vitb16
        if FLAGS.in_channels != 3:
            print(f"Warning: SegDINO only supports RGB (3 channels). Forcing in_channels=3")
            print(f"Dataset will load only RGB bands from the 5-channel data.")
        dino_ckpt = FLAGS.dino_ckpt if FLAGS.dino_ckpt else None
        net = segdino_vitb16(
            num_classes=FLAGS.num_classes,
            pretrained_backbone=FLAGS.pretrained_backbone,
            dinov3_path=FLAGS.dinov3_path,
            dino_ckpt=dino_ckpt,
            decoder=FLAGS.decoder
        )
        print(f"Loaded SegDINO ViT-B/16 with {FLAGS.decoder} decoder and {FLAGS.num_classes} classes")
        
    elif FLAGS.model == 'segdino_s':
        from nets import segdino_vits16
        if FLAGS.in_channels != 3:
            print(f"Warning: SegDINO only supports RGB (3 channels). Forcing in_channels=3")
            print(f"Dataset will load only RGB bands from the 5-channel data.")
        dino_ckpt = FLAGS.dino_ckpt if FLAGS.dino_ckpt else None
        net = segdino_vits16(
            num_classes=FLAGS.num_classes,
            pretrained_backbone=FLAGS.pretrained_backbone,
            dinov3_path=FLAGS.dinov3_path,
            dino_ckpt=dino_ckpt,
            decoder=FLAGS.decoder
        )
        print(f"Loaded SegDINO ViT-S/16 with {FLAGS.decoder} decoder and {FLAGS.num_classes} classes")
        
    elif FLAGS.dlv3p_do:
        net = deeplabv3plus_resnet50_do(num_classes=FLAGS.num_classes, pretrained_backbone=FLAGS.pretrained_backbone)
        print(f"Loaded DeepLabv3+ with dropout (probabilistic)")
        # Modify first layer for multispectral
        if FLAGS.in_channels == 5:
            net.backbone.conv1 = torch.nn.Conv2d(FLAGS.in_channels, net.backbone.conv1.out_channels, 
                                                kernel_size=7, stride=2, padding=3, bias=False, device=device)
    else:
        net = deeplabv3plus_resnet50(num_classes=FLAGS.num_classes, pretrained_backbone=FLAGS.pretrained_backbone)
        print(f"Loaded DeepLabv3+ (deterministic)")
        # Modify first layer for multispectral
        if FLAGS.in_channels == 5:
            net.backbone.conv1 = torch.nn.Conv2d(FLAGS.in_channels, net.backbone.conv1.out_channels, 
                                                kernel_size=7, stride=2, padding=3, bias=False, device=device)

    # Model to device
    net.to(device=device)

    # Loss criterion with class weights to handle imbalance
    # Weights calculated using ERFNet method (Romera et al., 2017)
    # Based on WeedsGalore training set: background (93.35%), crop (2.45%), weed (4.20%)
    if FLAGS.use_class_weights:
        if FLAGS.num_classes == 3:
            # Binary: background, crop, weed
            # Calculated from training set using: 1 / log(1.10 + relative_frequency)
            class_weights = torch.tensor([1.41, 8.53, 7.53], device=device)
        else:
            # Multi-class: background, crop, weed1, weed2, weed3, weed4
            # TODO: Run calculate_class_weights.py --num_classes 6 to get exact weights
            class_weights = torch.tensor([1.41, 8.53, 7.53, 7.53, 7.53, 7.53], device=device)
    else:
        class_weights = None
    
    # Choose loss function
    if FLAGS.loss_type == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=FLAGS.focal_gamma, ignore_index=FLAGS.ignore_index).to(device)
        print(f"Using Focal Loss (gamma={FLAGS.focal_gamma})")
        if class_weights is not None:
            print(f"  with class weights: {class_weights.cpu().numpy()}")
    else:
        if class_weights is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=FLAGS.ignore_index).to(device)
            print(f"Using weighted CrossEntropyLoss with weights: {class_weights.cpu().numpy()}")
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=FLAGS.ignore_index).to(device)
            print("Using standard CrossEntropyLoss (no class weights)")

    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.lr)

    # Learning rate scheduler
    if FLAGS.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FLAGS.epochs)
        print(f"Using CosineAnnealingLR scheduler")
    elif FLAGS.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=FLAGS.lr_step_size, gamma=FLAGS.lr_decay)
        print(f"Using StepLR scheduler (step_size={FLAGS.lr_step_size}, gamma={FLAGS.lr_decay})")
    elif FLAGS.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=FLAGS.lr_decay, patience=10, verbose=True)
        print(f"Using ReduceLROnPlateau scheduler")
    else:
        scheduler = None
        print("No LR scheduler")

    # Metric
    evaluator = MulticlassJaccardIndex(num_classes=FLAGS.num_classes, average=None, ignore_index=FLAGS.ignore_index).to(device)
    val_evaluator = MulticlassJaccardIndex(num_classes=FLAGS.num_classes, average=None, ignore_index=FLAGS.ignore_index).to(device)

    # Logging
    accum_loss, accum_iter, tot_iter = 0, 0, 0
    best_val_miou = 0.0
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    writer = SummaryWriter(f'{FLAGS.out_dir}')
    
    # Setup output logging to file and terminal
    log_file = open(f'{FLAGS.out_dir}/training_output.log', 'w')
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    
    print(f'Logging to: {FLAGS.out_dir}')
    Path(FLAGS.out_dir).mkdir(parents=True, exist_ok=True)

    torch.autograd.set_detect_anomaly(True)

    # Train
    for epoch in range(FLAGS.epochs):
        net.train()
        train_iter = iter(train_dataloader)
        for i, data in enumerate(train_iter):
            features, unique_labels, binary_labels = data
            if FLAGS.num_classes == 3:
                labels = binary_labels
            else:
                labels = unique_labels
            features, labels = features.to(device), labels.to(device)  # NCHW

            optimizer.zero_grad()
            out = net(features)
            
            # Resize output to match label size if needed (for SegDINO)
            if out.shape[2:] != labels.shape[1:]:
                out = F.interpolate(out, size=labels.shape[1:], mode='bilinear', align_corners=False)
            
            loss = criterion(out, labels.long())
            loss.backward()
            optimizer.step()

            accum_loss += loss
            accum_iter += 1
            tot_iter += 1

            # compute miou
            _, pred = torch.max(out, 1)
            evaluator.update(pred, labels)

            # log scalars
            if tot_iter % FLAGS.log_interval == 0 or tot_iter == 1:
                metrics = evaluator.compute() * 100

                print(f'Epoch: {epoch} iter: {tot_iter}, Loss: {(accum_loss / accum_iter):.2f}')
                print(f'mIoU: {metrics.mean():.2f}%')

                writer.add_scalar('Training Loss', accum_loss / accum_iter, tot_iter)
                writer.add_scalar('miou (%)', metrics.mean(), tot_iter)
                writer.add_scalar('iou_crop (%)', metrics[1], tot_iter)
                for weed_idx, weed_iou in enumerate(metrics[2:], start=2):
                    writer.add_scalar(f'iou_weed_{weed_idx-1} (%)', weed_iou, tot_iter)

                evaluator.reset()
                accum_loss, accum_iter = 0, 0

            # save ckpt
            if tot_iter % FLAGS.ckpt_interval == 0 or tot_iter == 1:
                torch.save(net.state_dict(), f'{FLAGS.out_dir}/{str(epoch)}.pth')
                torch.save(optimizer.state_dict(), f'{FLAGS.out_dir}/optimizer.pth')
        
        # Validation at end of epoch
        net.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for val_data in val_dataloader:
                val_features, val_unique_labels, val_binary_labels = val_data
                if FLAGS.num_classes == 3:
                    val_labels = val_binary_labels
                else:
                    val_labels = val_unique_labels
                val_features, val_labels = val_features.to(device), val_labels.to(device)
                
                val_out = net(val_features)
                if val_out.shape[2:] != val_labels.shape[1:]:
                    val_out = F.interpolate(val_out, size=val_labels.shape[1:], mode='bilinear', align_corners=False)
                
                val_loss += criterion(val_out, val_labels.long()).item()
                val_batches += 1
                
                _, val_pred = torch.max(val_out, 1)
                val_evaluator.update(val_pred, val_labels)
        
        val_metrics = val_evaluator.compute() * 100
        val_miou = val_metrics.mean().item()
        val_loss_avg = val_loss / val_batches
        
        print(f"\nValidation - Epoch {epoch}: Loss: {val_loss_avg:.4f}, mIoU: {val_miou:.2f}%")
        
        writer.add_scalar('Validation/Loss', val_loss_avg, epoch)
        writer.add_scalar('Validation/mIoU', val_miou, epoch)
        
        # Save best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_miou,
                'val_loss': val_loss_avg
            }, f'{FLAGS.out_dir}/best_model.pth')
            print(f"✓ Saved best model: mIoU {val_miou:.2f}%")
        
        val_evaluator.reset()
        
        # Step scheduler at end of epoch
        if scheduler is not None:
            if FLAGS.lr_scheduler == 'plateau':
                scheduler.step(val_miou)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} complete. Current LR: {current_lr:.2e}\n")

if __name__ == '__main__':
    app.run(main)