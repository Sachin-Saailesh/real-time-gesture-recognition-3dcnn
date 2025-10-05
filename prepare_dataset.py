import os
import shutil
import random
import re
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Paths - adjust these to match your downloaded dataset location
DATASET_PATH = "fingers"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

OUTPUT_PATH = "custom_dataset"

def extract_label_from_filename(filename):
    """Extract the finger count label from filename like 'xxx_5L.png' or 'xxx_3R.png'"""
    # Look for pattern like _0L, _1R, _2L, etc. before the extension

    match = re.search(r'_(\d)[LR]\.', filename)
    if match:
        return match.group(1)
    return None

def prepare_dataset():
    """Mix train and test sets, then randomly select 10% for custom training set"""
    
    print(f"Looking for dataset in: {os.path.abspath(DATASET_PATH)}")
    
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Collect all image paths
    all_images = []
    
    # Process train folder
    if os.path.exists(TRAIN_PATH):
        print(f"\nScanning train folder: {TRAIN_PATH}")
        train_files = [f for f in os.listdir(TRAIN_PATH) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  Found {len(train_files)} image files")
        
        for img_file in train_files:
            label = extract_label_from_filename(img_file)
            if label:
                all_images.append((os.path.join(TRAIN_PATH, img_file), label))
            else:
                print(f"  Warning: Could not extract label from {img_file}")
    else:
        print(f"⚠ Train folder not found: {TRAIN_PATH}")
    
    # Process test folder
    if os.path.exists(TEST_PATH):
        print(f"\nScanning test folder: {TEST_PATH}")
        test_files = [f for f in os.listdir(TEST_PATH) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  Found {len(test_files)} image files")
        
        for img_file in test_files:
            label = extract_label_from_filename(img_file)
            if label:
                all_images.append((os.path.join(TEST_PATH, img_file), label))
            else:
                print(f"  Warning: Could not extract label from {img_file}")
    else:
        print(f"⚠ Test folder not found: {TEST_PATH}")
    
    print(f"\nTotal images with valid labels: {len(all_images)}")
    
    if len(all_images) == 0:
        print("\n❌ ERROR: No images found!")
        print("Please check:")
        print("1. The 'fingers' folder exists in the current directory")
        print("2. It contains 'train' and/or 'test' subfolders")
        print("3. Those folders contain .png image files")
        return
    
    # Shuffle all images
    random.shuffle(all_images)
    
    # Take 10% for custom training set
    num_train = max(1, int(len(all_images) * 0.1))  # At least 1 image
    train_images = all_images[:num_train]
    
    print(f"Selected {num_train} images for training (10%)")
    
    # Copy images to custom dataset folder organized by label
    for img_path, label in train_images:
        label_dir = os.path.join(OUTPUT_PATH, label)
        os.makedirs(label_dir, exist_ok=True)
        
        img_name = os.path.basename(img_path)
        dest_path = os.path.join(label_dir, img_name)
        shutil.copy2(img_path, dest_path)
    
    print(f"\n✓ Dataset prepared in '{OUTPUT_PATH}' folder")
    
    # Print distribution
    print("\nLabel distribution:")
    total = 0
    for label_folder in sorted(os.listdir(OUTPUT_PATH)):
        label_path = os.path.join(OUTPUT_PATH, label_folder)
        if os.path.isdir(label_path):
            count = len([f for f in os.listdir(label_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  Label {label_folder}: {count} images")
            total += count
    print(f"  Total: {total} images")

if __name__ == "__main__":
    prepare_dataset()