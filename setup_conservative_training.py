#!/usr/bin/env python3
"""
Conservative training script for large dataset.
This script uses reduced parameters to ensure it completes successfully.
"""

import sys
import os
import shutil

def create_conservative_config():
    """Create a more conservative configuration for training."""
    print("Setting up conservative training configuration...")
    
    # Backup original config
    if os.path.exists('config.py'):
        shutil.copy('config.py', 'config_original.py')
    
    conservative_config = """# Conservative Image Captioning Configuration
# Optimized for systems with limited memory

# Model Parameters
MAX_LENGTH = 25  # Reduced from 35
VOCAB_SIZE = 3000  # Reduced from 5000
EMBEDDING_DIM = 128  # Reduced from 256
LSTM_UNITS = 128  # Reduced from 256

# Training Parameters
EPOCHS = 3  # Reduced from 5
BATCH_SIZE = 16  # Reduced from 32
LEARNING_RATE = 0.001

# Data Parameters
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# Feature Extraction
FEATURE_DIM = 512
IMAGE_SIZE = (224, 224)
FEATURE_BATCH_SIZE = 16  # Very conservative

# Paths
MODEL_FILE = 'models/caption_model.h5'
TOKENIZER_FILE = 'tokenizer.pkl'
FEATURES_FILE = 'features.npy'

# Memory Management
SEQUENCE_BATCH_SIZE = 250  # For sequence creation
"""
    
    with open('config_conservative.py', 'w') as f:
        f.write(conservative_config)
    
    print("Conservative config created: config_conservative.py")

def modify_main_script():
    """Modify the main script to use conservative settings."""
    print("Creating conservative training script...")
    
    # Read the original script
    with open('image_captioning.py', 'r') as f:
        content = f.read()
    
    # Modify key parameters for conservative training
    modified_content = content.replace(
        'MAX_LENGTH = 35', 'MAX_LENGTH = 25'
    ).replace(
        'VOCAB_SIZE = 5000', 'VOCAB_SIZE = 3000'
    ).replace(
        'EMBEDDING_DIM = 256', 'EMBEDDING_DIM = 128'
    ).replace(
        'LSTM_UNITS = 256', 'LSTM_UNITS = 128'
    ).replace(
        'EPOCHS = 5', 'EPOCHS = 3'
    ).replace(
        'BATCH_SIZE = 32', 'BATCH_SIZE = 16'
    ).replace(
        'FEATURE_BATCH_SIZE = 32', 'FEATURE_BATCH_SIZE = 16'
    ).replace(
        'batch_size=500', 'batch_size=250'
    )
    
    # Write conservative script
    with open('image_captioning_conservative.py', 'w') as f:
        f.write(modified_content)
    
    print("Conservative script created: image_captioning_conservative.py")

def main():
    """Create conservative training setup."""
    print("Setting up conservative training for large dataset...")
    print("This configuration uses reduced parameters to avoid OOM issues.")
    
    create_conservative_config()
    modify_main_script()
    
    print("\n" + "="*60)
    print("CONSERVATIVE TRAINING SETUP COMPLETE")
    print("="*60)
    print("\nTo run conservative training:")
    print("python3 image_captioning_conservative.py")
    print("\nThis version uses:")
    print("- Smaller vocabulary (3000 vs 5000 words)")
    print("- Shorter sequences (25 vs 35 max length)")
    print("- Smaller model (128 vs 256 dimensions)")
    print("- Fewer epochs (3 vs 5)")
    print("- Smaller batches (16 vs 32)")
    print("- More aggressive memory management")
    
    print("\nExpected completion time: 20-30 minutes")
    print("Memory usage: Should stay under 8GB")

if __name__ == '__main__':
    main()
