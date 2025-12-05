"""
Simplified Mask2Former-inspired decoder with learnable queries.
Transformer-based decoder for state-of-the-art segmentation.

Reference: Masked-attention Mask Transformer for Universal Image Segmentation (CVPR 2022)
Note: This is a simplified version - full Mask2Former is very complex.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class PixelDecoder(nn.Module):
    """Multi-scale pixel decoder (simplified FPN)."""
    def __init__(self, in_channels, feat_channels=256):
        super(PixelDecoder, self).__init__()
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, feat_channels, kernel_size=1) for _ in range(4)
        ])
        
        # Output projection
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, feat_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])
        
        # Mask features
        self.mask_features = nn.Conv2d(feat_channels, feat_channels, kernel_size=1)
    
    def forward(self, features):
        """
        Args:
            features: List of multi-scale features [c1, c2, c3, c4]
        Returns:
            mask_features: (B, C, H, W) for mask prediction
            multi_scale_features: List of refined features
        """
        # Process features top-down
        laterals = [self.lateral_convs[i](features[i]) for i in range(4)]
        
        # Top-down fusion
        for i in range(3, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=laterals[i-1].shape[2:], 
                mode='bilinear', align_corners=False
            )
        
        # Refine outputs
        outputs = [self.output_convs[i](laterals[i]) for i in range(4)]
        
        # Mask features from highest resolution
        mask_features = self.mask_features(outputs[0])
        
        return mask_features, outputs


class TransformerDecoder(nn.Module):
    """Simplified transformer decoder with cross-attention to image features."""
    def __init__(self, num_queries=100, d_model=256, nhead=8, num_layers=3):
        super(TransformerDecoder, self).__init__()
        
        self.num_queries = num_queries
        
        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Level embedding for multi-scale features
        self.level_embed = nn.Embedding(4, d_model)
    
    def forward(self, multi_scale_features):
        """
        Args:
            multi_scale_features: List of 4 feature maps (B, C, H, W)
        Returns:
            query_features: (B, num_queries, C)
        """
        bs = multi_scale_features[0].shape[0]
        
        # Flatten and combine multi-scale features
        memory = []
        for i, feat in enumerate(multi_scale_features):
            # (B, C, H, W) -> (B, H*W, C)
            h, w = feat.shape[2:]
            feat_flat = feat.flatten(2).permute(0, 2, 1)
            
            # Add level embedding
            level_emb = self.level_embed.weight[i].view(1, 1, -1)
            feat_flat = feat_flat + level_emb
            
            memory.append(feat_flat)
        
        # Concatenate all levels: (B, sum(H_i*W_i), C)
        memory = torch.cat(memory, dim=1)
        
        # Query embeddings: (num_queries, C) -> (B, num_queries, C)
        queries = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        
        # Decode
        query_features = self.decoder(queries, memory)
        
        return query_features


class Mask2FormerDecoder(nn.Module):
    """
    Mask2Former-style decoder with learnable queries.
    More sophisticated than MLP or FPN-based decoders.
    """
    def __init__(self, in_channels, num_classes, num_queries=100, feat_channels=256):
        super(Mask2FormerDecoder, self).__init__()
        
        self.num_queries = num_queries
        
        # Pixel decoder (FPN-like)
        self.pixel_decoder = PixelDecoder(in_channels, feat_channels)
        
        # Transformer decoder
        self.transformer_decoder = TransformerDecoder(
            num_queries=num_queries,
            d_model=feat_channels,
            nhead=8,
            num_layers=3
        )
        
        # Prediction heads
        self.class_embed = nn.Linear(feat_channels, num_classes + 1)  # +1 for no-object
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels)
        )
    
    def forward(self, features):
        """
        Args:
            features: List of 4 feature maps
        Returns:
            pred_masks: (B, num_queries, H, W)
            pred_logits: (B, num_queries, num_classes+1)
        """
        bs = features[0].shape[0]
        
        # Pixel decoder
        mask_features, multi_scale_features = self.pixel_decoder(features)
        
        # Transformer decoder
        query_features = self.transformer_decoder(multi_scale_features)
        
        # Class predictions
        pred_logits = self.class_embed(query_features)
        
        # Mask predictions
        mask_embed_output = self.mask_embed(query_features)
        
        # Generate masks: query features × mask features
        # (B, num_queries, C) @ (B, C, H, W) -> (B, num_queries, H, W)
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embed_output, mask_features)
        
        return pred_masks, pred_logits


class Mask2FormerDPT(nn.Module):
    """Complete model with DINOv3 backbone + Mask2Former decoder."""
    def __init__(self, encoder_size='base', nclass=3, num_queries=100, backbone=None):
        super(Mask2FormerDPT, self).__init__()
        
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
            'large': [4, 11, 17, 23],
        }
        
        self.encoder_size = encoder_size
        self.backbone = backbone
        self.num_classes = nclass
        
        # Mask2Former decoder
        self.decoder = Mask2FormerDecoder(
            in_channels=self.backbone.embed_dim,
            num_classes=nclass,
            num_queries=num_queries,
            feat_channels=256
        )
    
    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        h, w = x.shape[2:]
        patch_h, patch_w = h // 16, w // 16
        
        # Extract features
        features = self.backbone.get_intermediate_layers(
            x, n=self.intermediate_layer_idx[self.encoder_size]
        )
        
        # Reshape to spatial maps
        feature_maps = []
        for feat in features:
            feat = feat.permute(0, 2, 1).reshape(feat.shape[0], feat.shape[2], patch_h, patch_w)
            feature_maps.append(feat)
        
        # Decode
        pred_masks, pred_logits = self.decoder(feature_maps)
        
        # For semantic segmentation: aggregate query predictions
        # Get class with highest confidence for each query
        pred_classes = pred_logits.argmax(dim=-1)  # (B, num_queries)
        
        # Combine masks based on class predictions
        # This is simplified - proper Mask2Former uses Hungarian matching during training
        out = self._aggregate_masks(pred_masks, pred_classes, self.num_classes)
        
        # Upsample to input size
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        
        return out
    
    def _aggregate_masks(self, pred_masks, pred_classes, num_classes):
        """Aggregate query-based masks into semantic segmentation map."""
        bs = pred_masks.shape[0]
        h, w = pred_masks.shape[2:]
        
        # Initialize output
        out = torch.zeros(bs, num_classes, h, w, device=pred_masks.device)
        
        # For each class, combine masks of queries predicting that class
        for c in range(num_classes):
            # Find queries predicting class c
            mask_indices = (pred_classes == c)
            
            if mask_indices.any():
                # Get masks for this class and apply sigmoid
                class_masks = torch.where(
                    mask_indices.unsqueeze(-1).unsqueeze(-1),
                    pred_masks,
                    torch.zeros_like(pred_masks)
                )
                class_masks = class_masks.sigmoid()
                
                # Aggregate (max pooling over queries)
                out[:, c] = class_masks.max(dim=1)[0]
        
        return out
