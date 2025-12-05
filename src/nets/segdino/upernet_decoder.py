"""
UperNet-style decoder with Pyramid Pooling Module (PPM).
Better multi-scale context aggregation than basic DPT.

Reference: Unified Perceptual Parsing for Scene Understanding (ECCV 2018)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):
    """Pyramid Pooling Module for multi-scale context."""
    def __init__(self, in_channels, out_channels=512, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for scale in pool_scales
        ])
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_scales) * out_channels, out_channels, 
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x]
        
        for stage in self.stages:
            pyramid = stage(x)
            pyramid = F.interpolate(pyramid, size=(h, w), mode='bilinear', align_corners=True)
            pyramids.append(pyramid)
        
        out = torch.cat(pyramids, dim=1)
        out = self.bottleneck(out)
        return out


class UperNetDecoder(nn.Module):
    """
    UperNet decoder with PPM and FPN-style feature fusion.
    Significantly better than basic DPT for agricultural segmentation.
    """
    def __init__(self, in_channels, num_classes, fpn_channels=256, ppm_channels=512):
        super(UperNetDecoder, self).__init__()
        
        # PPM on the deepest feature map
        self.ppm = PPM(in_channels, out_channels=ppm_channels, pool_scales=(1, 2, 3, 6))
        
        # Lateral connections for FPN
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, fpn_channels, kernel_size=1) for _ in range(4)
        ])
        
        # FPN refinement
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(fpn_channels * 4, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        # Segmentation head
        self.seg_head = nn.Conv2d(fpn_channels, num_classes, kernel_size=1)
    
    def forward(self, features):
        """
        Args:
            features: List of 4 feature maps from backbone [c1, c2, c3, c4]
                     Each: (B, embed_dim, H/16, W/16) for ViT
        """
        # Apply PPM to deepest features
        c4_ppm = self.ppm(features[3])
        
        # Build FPN top-down
        fpn_features = []
        prev = c4_ppm
        
        for i in range(3, -1, -1):
            # Lateral connection
            lateral = self.lateral_convs[i](features[i])
            
            # Top-down fusion
            if i < 3:
                prev = F.interpolate(prev, size=lateral.shape[2:], mode='bilinear', align_corners=True)
                lateral = lateral + prev
            
            # Refine
            fpn_out = self.fpn_convs[i](lateral)
            fpn_features.insert(0, fpn_out)
            prev = fpn_out
        
        # Upsample all to same size and fuse
        target_size = fpn_features[0].shape[2:]
        upsampled = [fpn_features[0]]
        for feat in fpn_features[1:]:
            upsampled.append(F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True))
        
        fused = torch.cat(upsampled, dim=1)
        fused = self.fusion(fused)
        
        # Final segmentation
        out = self.seg_head(fused)
        
        return out


class UperNetDPT(nn.Module):
    """Complete model with DINOv3 backbone + UperNet decoder."""
    def __init__(self, encoder_size='base', nclass=3, fpn_channels=256, backbone=None):
        super(UperNetDPT, self).__init__()
        
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
            'large': [4, 11, 17, 23],
        }
        
        self.encoder_size = encoder_size
        self.backbone = backbone
        
        # UperNet decoder
        self.decoder = UperNetDecoder(
            in_channels=self.backbone.embed_dim,
            num_classes=nclass,
            fpn_channels=fpn_channels,
            ppm_channels=512
        )
    
    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        h, w = x.shape[2:]
        patch_h, patch_w = h // 16, w // 16
        
        # Extract multi-scale features
        features = self.backbone.get_intermediate_layers(
            x, n=self.intermediate_layer_idx[self.encoder_size]
        )
        
        # Reshape features to spatial maps
        feature_maps = []
        for feat in features:
            # (B, N, C) -> (B, C, H, W)
            feat = feat.permute(0, 2, 1).reshape(feat.shape[0], feat.shape[2], patch_h, patch_w)
            feature_maps.append(feat)
        
        # Decode
        out = self.decoder(feature_maps)
        
        # Upsample to input size
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        
        return out
