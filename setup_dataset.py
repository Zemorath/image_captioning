#!/usr/bin/env python3
"""
Dataset Setup Utility for Image Captioning Project

This script helps download and setup the Flickr8k dataset for training.
"""

import os
import sys
import zipfile
import requests
from pathlib import Path

def setup_directories():
    """Create necessary directories."""
    dirs = [
        'data/Flickr8k',
        'models',
        'utils'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def check_kaggle_setup():
    """Check if Kaggle CLI is available and configured."""
    try:
        import kaggle
        print("✓ Kaggle CLI is available")
        return True
    except ImportError:
        print("✗ Kaggle CLI not found. Install with: pip install kaggle")
        return False
    except OSError as e:
        if "No such file or directory: 'kaggle.json'" in str(e):
            print("✗ Kaggle API credentials not found.")
            print("  Please setup your Kaggle API credentials:")
            print("  1. Go to https://www.kaggle.com/account")
            print("  2. Create New API Token")
            print("  3. Download kaggle.json")
            print("  4. Place it in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%/.kaggle/ (Windows)")
            print("  5. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False
        else:
            print(f"✗ Kaggle setup error: {e}")
            return False

def download_flickr8k():
    """Download Flickr8k dataset using Kaggle CLI."""
    try:
        import kaggle
        
        print("📥 Downloading Flickr8k dataset from Kaggle...")
        kaggle.api.dataset_download_files(
            'adityajn105/flickr8k',
            path='data/',
            unzip=True
        )
        print("✓ Dataset downloaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("\nManual download instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/adityajn105/flickr8k")
        print("2. Download the dataset")
        print("3. Extract to: data/Flickr8k/")
        return False

def verify_dataset():
    """Verify that the dataset is properly extracted."""
    base_path = Path('data/Flickr8k')
    
    # Check for main directories/files
    required_items = [
        'Images',
        'captions.txt'
    ]
    
    # Alternative caption files
    alt_caption_files = [
        'Flickr8k.token.txt',
        'Flickr_8k.trainImages.txt',
        'captions.txt'
    ]
    
    missing_items = []
    
    # Check images directory
    images_dir = base_path / 'Images'
    if not images_dir.exists():
        missing_items.append('Images directory')
    else:
        image_count = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.jpeg')))
        print(f"✓ Found {image_count} images in Images directory")
        
        if image_count < 1000:
            print(f"⚠️  Warning: Only {image_count} images found. Expected ~8000 for Flickr8k.")
    
    # Check for caption files
    caption_file_found = False
    for caption_file in alt_caption_files:
        if (base_path / caption_file).exists():
            caption_file_found = True
            print(f"✓ Found caption file: {caption_file}")
            break
    
    if not caption_file_found:
        missing_items.append('caption file (captions.txt or Flickr8k.token.txt)')
    
    if missing_items:
        print(f"✗ Missing items: {', '.join(missing_items)}")
        return False
    else:
        print("✓ Dataset verification passed!")
        return True

def create_sample_config():
    """Create a sample configuration file."""
    config_content = """# Image Captioning Configuration
# Modify these parameters as needed

# Model Parameters
MAX_LENGTH = 35
VOCAB_SIZE = 5000
EMBEDDING_DIM = 256
LSTM_UNITS = 256

# Training Parameters
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Data Parameters
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# Feature Extraction
FEATURE_DIM = 512
IMAGE_SIZE = (224, 224)

# Paths
MODEL_FILE = 'models/caption_model.h5'
TOKENIZER_FILE = 'tokenizer.pkl'
FEATURES_FILE = 'features.npy'
"""
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    
    print("✓ Created config.py with default parameters")

def main():
    """Main setup function."""
    print("🚀 Image Captioning Project Setup")
    print("=" * 40)
    
    # Step 1: Create directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Step 2: Check Kaggle setup
    print("\n🔑 Checking Kaggle setup...")
    if not check_kaggle_setup():
        print("\n⚠️  Kaggle setup incomplete. Please configure Kaggle CLI manually.")
        print("You can still run the project by manually downloading the dataset.")
    
    # Step 3: Download dataset
    print("\n📥 Setting up dataset...")
    dataset_exists = Path('data/Flickr8k/Images').exists()
    
    if dataset_exists:
        print("✓ Dataset directory already exists")
    else:
        if check_kaggle_setup():
            download_flickr8k()
        else:
            print("⚠️  Skipping automatic download due to Kaggle setup issues")
    
    # Step 4: Verify dataset
    print("\n🔍 Verifying dataset...")
    if verify_dataset():
        print("✅ Dataset setup complete!")
    else:
        print("❌ Dataset setup incomplete. Please check the manual download instructions.")
    
    # Step 5: Create config file
    print("\n⚙️  Creating configuration...")
    create_sample_config()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the main script: python image_captioning.py")
    print("3. Or start with a smaller test: python -c \"import tensorflow as tf; print('TensorFlow version:', tf.__version__)\"")

if __name__ == '__main__':
    main()
