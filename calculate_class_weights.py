#!/usr/bin/env python3
"""
Calculate class weights for WeedsGalore semantic segmentation dataset.
Based on the inverse frequency method with logarithmic scaling.
"""
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
import math

def convert_frequency_to_weight(relative_frequency: float, constant: float = 1.10) -> float:
    """ Convert a (relative) class frequency into a weight.
    
    We follow the definition of http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf.
    
    Args:
        relative_frequency: relative class frequency (0.0 to 1.0)
        constant: logarithmic constant (default 1.10)
    
    Returns:
        float: class weight
    """
    assert 0.0 <= relative_frequency <= 1.0
    weight = 1 / math.log(constant + relative_frequency)
    return weight

def calculate_weedsgalore_weights(dataset_path: str, num_classes: int = 3, split: str = 'train'):
    """
    Calculate class weights for WeedsGalore dataset.
    
    Args:
        dataset_path: Path to weedsgalore-dataset
        num_classes: 3 for binary (background/crop/weed) or 6 for multi-class
        split: 'train', 'val', or 'test'
    """
    # Read split file
    split_file = Path(dataset_path) / 'splits' / f'{split}.txt'
    with open(split_file, 'r') as f:
        sample_ids = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Calculating class weights for {len(sample_ids)} {split} samples...")
    print(f"Number of classes: {num_classes}")
    print("=" * 80)
    
    # Count total pixels per class
    total_class_frequencies = defaultdict(int)
    
    for sample_id in sample_ids:
        # Extract date from sample_id
        date = sample_id[:10]
        
        # Load semantic mask
        mask_path = Path(dataset_path) / date / 'semantics' / f'{sample_id}.png'
        mask = np.array(Image.open(mask_path))
        
        # Convert to binary labels if needed
        if num_classes == 3:
            # background=0, crop=1, weed=2 (merge all weed classes)
            mask = np.where(mask >= 2, 2, mask)
        
        # Count pixels for each class
        unique_values, counts = np.unique(mask, return_counts=True)
        for class_id, count in zip(unique_values, counts):
            if class_id != 255:  # Ignore void class
                total_class_frequencies[int(class_id)] += int(count)
    
    # Calculate total pixels
    total_pixels = sum(total_class_frequencies.values())
    
    # Calculate relative frequencies
    relative_frequencies = {}
    for class_id in sorted(total_class_frequencies.keys()):
        relative_frequencies[class_id] = total_class_frequencies[class_id] / total_pixels
    
    # Calculate weights
    class_weights = {}
    for class_id, rel_freq in relative_frequencies.items():
        weight = convert_frequency_to_weight(rel_freq)
        class_weights[class_id] = weight
    
    # Print results
    print("\nClass Distribution:")
    print("-" * 80)
    class_names_3 = {0: 'background', 1: 'crop', 2: 'weed'}
    class_names_6 = {0: 'background', 1: 'crop', 2: 'weed_1', 3: 'weed_2', 4: 'weed_3', 5: 'weed_4'}
    class_names = class_names_3 if num_classes == 3 else class_names_6
    
    for class_id in sorted(total_class_frequencies.keys()):
        class_name = class_names.get(class_id, f'class_{class_id}')
        pixels = total_class_frequencies[class_id]
        rel_freq = relative_frequencies[class_id]
        weight = class_weights[class_id]
        
        print(f"Class {class_id} ({class_name:12s}): "
              f"{pixels:12,} pixels ({rel_freq*100:6.2f}%) -> weight: {weight:6.2f}")
    
    print("=" * 80)
    print("\nPyTorch tensor format:")
    weights_list = [class_weights[i] for i in sorted(class_weights.keys())]
    print(f"class_weights = torch.tensor({weights_list}, device=device)")
    print("=" * 80)
    
    return class_weights

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate class weights for WeedsGalore')
    parser.add_argument('--dataset_path', type=str, default='/home/ilias/weedsgalore-dataset',
                        help='Path to weedsgalore-dataset')
    parser.add_argument('--num_classes', type=int, default=3, choices=[3, 6],
                        help='Number of classes (3 or 6)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Dataset split')
    
    args = parser.parse_args()
    
    calculate_weedsgalore_weights(args.dataset_path, args.num_classes, args.split)
