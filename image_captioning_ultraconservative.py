#!/usr/bin/env python3
"""
ULTRA-CONSERVATIVE Image Captioning System for Accessibility Tools

This script uses an extremely conservative approach to avoid OOM:
1. Small dataset (2000 caption pairs)
2. Reduced model parameters  
3. Proper start/end token handling
4. Streaming data processing

Author: Image Captioning Assistant (Ultra-Conservative)
Date: August 28, 2025
"""

# Step 1: Imports and Setup
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import nltk
import pickle
import json
import os
import random
import gc  # For garbage collection
import argparse  # For command line arguments
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# TensorFlow/Keras imports
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Dropout, Add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# NLTK imports
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
DATA_DIR = './data/Flickr8k/'
IMAGES_DIR = os.path.join(DATA_DIR, 'Images/')
ANNOTATIONS_FILE = os.path.join(DATA_DIR, 'captions.txt')
FEATURES_FILE = 'features_ultraconservative.npy'
MODEL_FILE = 'models/caption_model_ultraconservative.h5'
TOKENIZER_FILE = 'tokenizer_ultraconservative.pkl'

# ULTRA-CONSERVATIVE Configuration parameters
MAX_LENGTH = 20  # Reduced from 25
VOCAB_SIZE = 2000  # Reduced from 3000
FEATURE_DIM = 4096  # Correct VGG16 feature dimension (from fc2 layer)
EMBEDDING_DIM = 64  # Reduced from 128
LSTM_UNITS = 64  # Reduced from 128
EPOCHS = 3  # Keep conservative
BATCH_SIZE = 8  # Reduced from 16
FEATURE_BATCH_SIZE = 8  # Reduced from 16

# CRITICAL FIX: Use proper start/end tokens
START_TOKEN = 'startseq'  # Single token without special characters
END_TOKEN = 'endseq'      # Single token without special characters

print("ULTRA-CONSERVATIVE Image Captioning System for Accessibility Tools")
print("=" * 70)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {len(gpus)} device(s)")
    try:
        # Enable memory growth for GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("Running on CPU")

# Download required NLTK data
print("Downloading NLTK data...")
try:
    nltk.download('punkt', quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"NLTK download warning: {e}")


# Step 2: Data Loading and Preprocessing Functions

def check_data_exists():
    """Check if the required dataset exists."""
    if not os.path.exists(DATA_DIR):
        raise ValueError(
            f"Dataset directory not found: {DATA_DIR}\n"
            "Please download the Flickr8k dataset first."
        )
    
    if not os.path.exists(IMAGES_DIR):
        raise ValueError(f"Images directory not found: {IMAGES_DIR}")
    
    if not os.path.exists(ANNOTATIONS_FILE):
        raise ValueError(f"Annotations file not found: {ANNOTATIONS_FILE}")
    
    print("✓ Dataset files found")


def load_captions_ultraconservative(max_pairs=2000):
    """
    Load image captions - ULTRA CONSERVATIVE version with minimal data.
    
    Args:
        max_pairs (int): Maximum number of caption pairs to load
        
    Returns:
        dict: Dictionary with image names as keys and lists of captions as values
    """
    print(f"Loading captions (max {max_pairs} pairs - ULTRA CONSERVATIVE)...")
    
    captions_dict = {}
    pairs_loaded = 0
    
    try:
        with open(ANNOTATIONS_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        for line in lines[1:]:  # Skip header if present
            if pairs_loaded >= max_pairs:
                print(f"Reached maximum of {max_pairs} pairs, stopping...")
                break
                
            line = line.strip()
            if not line:
                continue
                
            # Handle different formats: image_name\tcaption or image_name,caption
            if '\t' in line:
                parts = line.split('\t', 1)
            elif ',' in line:
                parts = line.split(',', 1)
            else:
                continue
                
            if len(parts) != 2:
                continue
                
            image_name, caption = parts
            image_name = image_name.strip()
            caption = caption.strip()
            
            # Remove image extension variations and get base name
            image_base = image_name.split('#')[0] if '#' in image_name else image_name
            
            # CRITICAL FIX: Use proper start and end tokens
            caption = f"{START_TOKEN} {caption.lower()} {END_TOKEN}"
            
            if image_base not in captions_dict:
                captions_dict[image_base] = []
            captions_dict[image_base].append(caption)
            pairs_loaded += 1
    
    except Exception as e:
        print(f"Error loading captions: {e}")
        raise
    
    print(f"Loaded captions for {len(captions_dict)} images")
    
    # Show sample captions
    sample_img = list(captions_dict.keys())[0]
    print(f"Sample captions for {sample_img}:")
    for i, cap in enumerate(captions_dict[sample_img][:3]):
        print(f"  {i+1}: {cap}")
    
    return captions_dict


def create_vocabulary_ultraconservative(captions_dict):
    """
    Create vocabulary from captions - ULTRA CONSERVATIVE version.
    
    Args:
        captions_dict (dict): Image captions
        
    Returns:
        Tokenizer: Fitted tokenizer
    """
    print("Creating vocabulary (ULTRA CONSERVATIVE)...")
    
    # Collect all captions
    all_captions = []
    for captions in captions_dict.values():
        all_captions.extend(captions)
    
    print(f"Total captions for vocabulary: {len(all_captions)}")
    
    # Show sample captions to verify tokens
    print("Sample captions (first 3):")
    for i, cap in enumerate(all_captions[:3]):
        print(f"  {i+1}: {cap}")
    
    # Create tokenizer with ultra-conservative configuration
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<UNK>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(all_captions)
    
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Configured for top {VOCAB_SIZE} words")
    
    # CRITICAL CHECK: Verify our tokens are present
    print("\nToken verification:")
    print(f"  {START_TOKEN}: {'✓' if START_TOKEN in tokenizer.word_index else '✗'} " +
          f"(index {tokenizer.word_index.get(START_TOKEN, 'NOT FOUND')})")
    print(f"  {END_TOKEN}: {'✓' if END_TOKEN in tokenizer.word_index else '✗'} " +
          f"(index {tokenizer.word_index.get(END_TOKEN, 'NOT FOUND')})")
    print(f"  <UNK>: {'✓' if '<UNK>' in tokenizer.word_index else '✗'} " +
          f"(index {tokenizer.word_index.get('<UNK>', 'NOT FOUND')})")
    
    # Show top words
    word_counts = [(word, idx) for word, idx in tokenizer.word_index.items()]
    word_counts.sort(key=lambda x: x[1])
    print(f"\nTop 15 words:")
    for word, idx in word_counts[:15]:
        print(f"  {idx}: {word}")
    
    return tokenizer


def extract_features_ultraconservative(image_files, batch_size=FEATURE_BATCH_SIZE, save_features=True):
    """
    Extract image features using VGG16 - ULTRA CONSERVATIVE version.
    
    Args:
        image_files (list): List of image file paths
        batch_size (int): Batch size for processing
        save_features (bool): Whether to save features to disk
        
    Returns:
        dict: Dictionary of image features
    """
    print(f"Extracting features for {len(image_files)} images (ULTRA CONSERVATIVE)...")
    
    # Check if features already exist
    features_file_path = FEATURES_FILE
    
    if os.path.exists(features_file_path):
        print(f"Loading pre-computed features from {features_file_path}")
        features_dict = np.load(features_file_path, allow_pickle=True).item()
        print(f"Loaded features for {len(features_dict)} images")
        return features_dict
    
    features_dict = {}
    
    # Load VGG16 model
    vgg_model = VGG16()
    # Remove the final classification layer
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    print("VGG16 model loaded for feature extraction")
    
    # Process images in batches
    batch_images = []
    valid_files = []
    
    for i, img_path in enumerate(image_files):
        # Skip if already processed
        filename = os.path.basename(img_path)
        if filename in features_dict:
            continue
            
        try:
            # Load and preprocess image
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = tf.keras.applications.vgg16.preprocess_input(image)
            
            batch_images.append(image[0])  # Remove batch dimension for batching
            valid_files.append(filename)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
        
        # Process batch when full or at end
        if len(batch_images) == batch_size or i == len(image_files) - 1:
            if not batch_images:
                continue
                
            if batch_images:
                # Convert to numpy array and extract features for the batch
                batch_array = np.array(batch_images)
                batch_features = vgg_model.predict(batch_array, verbose=0)
                
                # Store features for each image
                for j, filename in enumerate(valid_files):
                    features_dict[filename] = batch_features[j]
                
                # Clear memory
                del batch_array, batch_features, batch_images
                gc.collect()  # Force garbage collection
                
            processed_count = len(features_dict)
            print(f"Processed {processed_count}/{len(image_files)} images")
            
            # Reset batch
            batch_images = []
            valid_files = []
    
    print(f"Feature extraction complete. Features for {len(features_dict)} images")
    
    # Save final features
    if save_features:
        np.save(features_file_path, features_dict)
        print(f"Features saved to {features_file_path}")
    
    return features_dict


# Step 3: Model Architecture

class CaptionModelUltraConservative:
    """Image captioning model using VGG16 + LSTM - ULTRA CONSERVATIVE version."""
    
    def __init__(self, vocab_size, max_length):
        """
        Initialize the caption model.
        
        Args:
            vocab_size (int): Size of vocabulary
            max_length (int): Maximum caption length
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.model = None
        self.build_model()
    
    def build_model(self):
        """Build the encoder-decoder model - ULTRA CONSERVATIVE version."""
        print("Building ULTRA CONSERVATIVE caption model...")
        
        # Image feature input
        image_input = Input(shape=(FEATURE_DIM,))
        image_dense = Dense(EMBEDDING_DIM, activation='relu')(image_input)
        
        # Text sequence input
        sequence_input = Input(shape=(self.max_length,))
        sequence_embedding = Embedding(self.vocab_size, EMBEDDING_DIM, mask_zero=True)(sequence_input)
        sequence_lstm = LSTM(LSTM_UNITS)(sequence_embedding)
        
        # Combine image and text features
        combined = Add()([image_dense, sequence_lstm])
        combined = Dense(LSTM_UNITS, activation='relu')(combined)
        combined = Dropout(0.2)(combined)  # Reduced dropout
        
        # Output layer
        output = Dense(self.vocab_size, activation='softmax')(combined)
        
        # Create model
        self.model = Model(inputs=[image_input, sequence_input], outputs=output)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model built successfully")
        print(self.model.summary())
    
    def generate_caption_ultraconservative(self, photo, tokenizer, max_length=MAX_LENGTH):
        """
        Generate caption for an image - ULTRA CONSERVATIVE version.
        
        Args:
            photo: Image features
            tokenizer: Fitted tokenizer
            max_length (int): Maximum caption length
            
        Returns:
            str: Generated caption
        """
        # Get token indices
        start_token_idx = tokenizer.word_index.get(START_TOKEN, 1)
        end_token_idx = tokenizer.word_index.get(END_TOKEN, 1)
        
        # Start with start token
        in_text = START_TOKEN
        
        for _ in range(max_length):
            # Encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            
            # Predict next word
            yhat = self.model.predict([photo.reshape(1, -1), sequence], verbose=0)
            yhat = np.argmax(yhat)
            
            # Map integer to word
            word = None
            for w, i in tokenizer.word_index.items():
                if i == yhat:
                    word = w
                    break
            
            # Stop if no word found or end token
            if word is None or word == END_TOKEN:
                break
                
            # Add word to caption
            in_text += ' ' + word
        
        # Remove start token and return
        caption = in_text.replace(START_TOKEN, '').strip()
        return caption


# Main function with direct training (no streaming)
def main():
    """Main training pipeline - ULTRA CONSERVATIVE version."""
    print("Starting ULTRA CONSERVATIVE image captioning training pipeline...")
    
    # Step 1: Check data
    check_data_exists()
    
    # Step 2: Load captions (MINIMAL dataset)
    captions_dict = load_captions_ultraconservative(max_pairs=2000)
    
    # Step 3: Create vocabulary
    tokenizer = create_vocabulary_ultraconservative(captions_dict)
    
    # Step 4: Extract image features
    image_files = [os.path.join(IMAGES_DIR, img) for img in captions_dict.keys() 
                   if os.path.exists(os.path.join(IMAGES_DIR, img))]
    print(f"Found {len(image_files)} valid image files")
    
    features_dict = extract_features_ultraconservative(image_files)
    
    # Step 5: Create minimal training data IN MEMORY (no saving)
    print("Creating training data in memory (MINIMAL)...")
    
    X1_list, X2_list, y_list = [], [], []
    
    count = 0
    for image_name, captions in captions_dict.items():
        if image_name not in features_dict:
            continue
            
        feature = features_dict[image_name]
        
        # Use only FIRST caption per image to minimize data
        caption = captions[0]
        seq = tokenizer.texts_to_sequences([caption])[0]
        
        # Create input-output pairs for this caption
        for i in range(1, min(len(seq), MAX_LENGTH)):  # Limit sequence length
            # Input sequence (everything before current position)
            in_seq = seq[:i]
            out_seq = seq[i]  # Target word at current position
            
            # Pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=MAX_LENGTH)[0]
            
            # One-hot encode output
            out_seq = to_categorical([out_seq], num_classes=len(tokenizer.word_index) + 1)[0]
            
            X1_list.append(feature)
            X2_list.append(in_seq)
            y_list.append(out_seq)
            count += 1
            
            # Limit total sequences to avoid OOM
            if count >= 5000:  # Very conservative limit
                break
        
        if count >= 5000:
            break
    
    # Convert to arrays
    X1 = np.array(X1_list)
    X2 = np.array(X2_list)
    y = np.array(y_list)
    
    print(f"Created {len(X1)} training sequences")
    print(f"Feature shape: {X1.shape}")
    print(f"Sequence shape: {X2.shape}")
    print(f"Target shape: {y.shape}")
    
    # Clear memory
    del X1_list, X2_list, y_list
    gc.collect()
    
    # Step 6: Split data
    print("Splitting data for training/validation...")
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
        X1, X2, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X1_train)}")
    print(f"Validation samples: {len(X1_val)}")
    
    # Step 7: Build and train model
    vocab_size = len(tokenizer.word_index) + 1
    model = CaptionModelUltraConservative(vocab_size, MAX_LENGTH)
    
    # Create model directory
    os.makedirs('models', exist_ok=True)
    
    # Create callbacks
    callbacks = [
        EarlyStopping(patience=2, restore_best_weights=True),
        ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss')
    ]
    
    # Train model
    print("Starting model training...")
    history = model.model.fit(
        [X1_train, X2_train], y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=([X1_val, X2_val], y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("Training completed")
    
    # Step 8: Save tokenizer
    print(f"Saving tokenizer to {TOKENIZER_FILE}")
    with open(TOKENIZER_FILE, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Step 9: Test model with a few samples
    print("\nTesting model with sample predictions...")
    
    sample_images = list(features_dict.keys())[:5]
    
    for img_name in sample_images:
        feature = features_dict[img_name]
        predicted_caption = model.generate_caption_ultraconservative(feature, tokenizer)
        
        # Get ground truth (first caption)
        if img_name in captions_dict and captions_dict[img_name]:
            gt_caption = captions_dict[img_name][0]
            # Remove start and end tokens for display
            gt_caption = gt_caption.replace(START_TOKEN, '').replace(END_TOKEN, '').strip()
        else:
            gt_caption = "No ground truth available"
        
        print(f"\nImage: {img_name}")
        print(f"Predicted: {predicted_caption}")
        print(f"Ground Truth: {gt_caption}")
        print("-" * 60)
    
    print("\n" + "="*70)
    print("ULTRA-CONSERVATIVE TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Tokenizer saved to: {TOKENIZER_FILE}")
    print(f"Features saved to: {FEATURES_FILE}")
    print("This model uses proper token handling and should generate better captions!")


if __name__ == "__main__":
    main()
