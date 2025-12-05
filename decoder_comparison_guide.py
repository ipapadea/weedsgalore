"""
Quick decoder comparison guide and parameter counts.
"""

DECODER_COMPARISON = """
====================================================================
SegDINO Decoder Architecture Comparison
====================================================================

1. DPT (Dense Prediction Transformer) - CURRENT BASELINE
--------------------------------------------------------------------
Architecture:
  - 4-scale feature projection (1x1 convs)
  - Simple bilinear upsampling + concatenation
  - Single final conv for segmentation

Parameters: ~2-3M (decoder only)
Memory: Low (~10GB for bs=32)
Speed: ⭐⭐⭐⭐⭐ FASTEST

Pros:
  ✓ Very fast training/inference
  ✓ Low memory footprint
  ✓ Simple, stable training

Cons:
  ✗ Limited multi-scale fusion
  ✗ No learned upsampling refinement
  ✗ Basic spatial reasoning

Expected mIoU improvement: Baseline (56.98%)


2. UperNet (Unified Perceptual Parsing)
--------------------------------------------------------------------
Architecture:
  - Pyramid Pooling Module (PPM) for context
  - FPN-style top-down refinement
  - Multi-scale feature fusion
  - Progressive upsampling with learned convs

Parameters: ~15-20M (decoder only)
Memory: Medium (~14GB for bs=32)
Speed: ⭐⭐⭐⭐ FAST

Pros:
  ✓ Superior multi-scale context (PPM at 1, 2, 3, 6 scales)
  ✓ Better boundary refinement
  ✓ Proven on scene parsing benchmarks
  ✓ Good balance of speed/accuracy

Cons:
  ✗ More parameters than DPT
  ✗ Slightly slower

Expected mIoU improvement: +3-5% → 60-62%
**RECOMMENDED for agricultural tasks with small objects**


3. SegFormer (All-MLP Decoder)
--------------------------------------------------------------------
Architecture:
  - Lightweight MLP projections
  - Simple upsampling + concatenation
  - Designed specifically for ViTs
  - Minimal parameters

Parameters: ~3-5M (decoder only)
Memory: Low (~11GB for bs=32)
Speed: ⭐⭐⭐⭐⭐ FASTEST

Pros:
  ✓ Very lightweight
  ✓ Fast training
  ✓ Optimized for transformer features
  ✓ Good generalization

Cons:
  ✗ Simpler than UperNet
  ✗ Less sophisticated multi-scale fusion

Expected mIoU improvement: +2-4% → 59-61%
**RECOMMENDED if training time/memory is critical**


4. Mask2Former (Transformer Decoder with Queries)
--------------------------------------------------------------------
Architecture:
  - Learnable query embeddings (100 queries)
  - Cross-attention to image features
  - Multi-head attention decoder (3 layers)
  - Query-based mask prediction

Parameters: ~25-35M (decoder only)
Memory: High (~18-22GB for bs=32, may need bs=16)
Speed: ⭐⭐ SLOW (2-3x slower than DPT)

Pros:
  ✓ State-of-the-art segmentation quality
  ✓ Excellent small object detection
  ✓ Learned mask proposals
  ✓ Best for complex scenes

Cons:
  ✗ Much slower training
  ✗ High memory usage
  ✗ More complex to tune
  ✗ Requires more training data ideally

Expected mIoU improvement: +4-7% → 61-64%
**RECOMMENDED only if you need maximum accuracy**


====================================================================
RECOMMENDATION FOR YOUR WEEDSGALORE TASK
====================================================================

Small dataset (104 samples), fine-grained weeds, limited time:
  → Try UperNet first

Best overall recommendation:
  1. UperNet (best balance for agricultural small objects)
  2. SegFormer (if you need speed)
  3. Mask2Former (if accuracy > speed)
  4. DPT (keep as baseline)


====================================================================
USAGE EXAMPLES
====================================================================

# Train with UperNet decoder (RECOMMENDED)
python src/train.py \\
  --dataset_path /workspace/weedsgalore-dataset \\
  --model segdino_s --decoder upernet \\
  --batch_size 32 --lr 0.0001 --epochs 500 \\
  --use_class_weights True --lr_scheduler cosine

# Train with SegFormer decoder (FAST)
python src/train.py \\
  --dataset_path /workspace/weedsgalore-dataset \\
  --model segdino_s --decoder segformer \\
  --batch_size 32 --lr 0.0001 --epochs 500 \\
  --use_class_weights True --lr_scheduler cosine

# Train with Mask2Former decoder (BEST QUALITY - reduce batch size!)
python src/train.py \\
  --dataset_path /workspace/weedsgalore-dataset \\
  --model segdino_s --decoder mask2former \\
  --batch_size 16 --lr 0.0001 --epochs 500 \\
  --use_class_weights True --lr_scheduler cosine

# Compare all decoders automatically
bash compare_decoders.sh


====================================================================
EXPECTED TRAINING TIMES (500 epochs on L40S GPU)
====================================================================

DPT:         ~6-8 hours
SegFormer:   ~7-9 hours
UperNet:     ~10-12 hours
Mask2Former: ~18-24 hours


====================================================================
MONITORING TIPS
====================================================================

1. Watch GPU memory:
   nvidia-smi -l 1
   
2. If Mask2Former OOMs, reduce batch size:
   --batch_size 16 or --batch_size 8

3. Compare results with TensorBoard:
   tensorboard --logdir=runs/decoder_comparison

4. Check mIoU curves - look for:
   - Faster convergence (better decoder)
   - Higher plateau (better capacity)
   - Less overfitting (better regularization)
"""

if __name__ == '__main__':
    print(DECODER_COMPARISON)
