"""
SegFormer-style MLP decoder.
Lightweight and specifically designed for Vision Transformers.

Reference: SegFormer: Simple and Efficient Design for Semantic Segmentation 
          with Transformers (NeurIPS 2021)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLayer(nn.Module):
    """MLP layer for feature projection."""
    def __init__(self, in_channels, out_channels):
        super(MLPLayer, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.proj(x)


class SegFormerDecoder(nn.Module):
    """
    SegFormer All-MLP decoder.
    Very efficient - designed for transformers with minimal parameters.
    """
    def __init__(self, in_channels, num_classes, embed_dim=256, dropout_ratio=0.1):
        super(SegFormerDecoder, self).__init__()
        
        # Project all feature maps to same channel dimension
        self.linear_c4 = MLPLayer(in_channels, embed_dim)
        self.linear_c3 = MLPLayer(in_channels, embed_dim)
        self.linear_c2 = MLPLayer(in_channels, embed_dim)
        self.linear_c1 = MLPLayer(in_channels, embed_dim)
        
        # Fusion MLP
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Dropout and classifier
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
    
    def forward(self, features):
        """
        Args:
            features: List of 4 feature maps [c1, c2, c3, c4]
        """
        c1, c2, c3, c4 = features
        
        # Get target size from first feature map
        n, _, h, w = c1.shape
        
        # Project and upsample all features to c1 size
        _c4 = self.linear_c4(c4)
        _c4 = F.interpolate(_c4, size=(h, w), mode='bilinear', align_corners=False)
        
        _c3 = self.linear_c3(c3)
        _c3 = F.interpolate(_c3, size=(h, w), mode='bilinear', align_corners=False)
        
        _c2 = self.linear_c2(c2)
        _c2 = F.interpolate(_c2, size=(h, w), mode='bilinear', align_corners=False)
        
        _c1 = self.linear_c1(c1)
        
        # Fuse features
        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        _c = self.linear_fuse(_c)
        
        # Classify
        _c = self.dropout(_c)
        out = self.linear_pred(_c)
        
        return out


class SegFormerDPT(nn.Module):
    """Complete model with DINOv3 backbone + SegFormer decoder."""
    def __init__(self, encoder_size='base', nclass=3, embed_dim=256, backbone=None):
        super(SegFormerDPT, self).__init__()
        
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
            'large': [4, 11, 17, 23],
        }
        
        self.encoder_size = encoder_size
        self.backbone = backbone
        
        # SegFormer decoder
        self.decoder = SegFormerDecoder(
            in_channels=self.backbone.embed_dim,
            num_classes=nclass,
            embed_dim=embed_dim,
            dropout_ratio=0.1
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
        
        # Reshape to spatial maps
        feature_maps = []
        for feat in features:
            feat = feat.permute(0, 2, 1).reshape(feat.shape[0], feat.shape[2], patch_h, patch_w)
            feature_maps.append(feat)
        
        # Decode
        out = self.decoder(feature_maps)
        
        # Upsample to input size
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        
        return out
