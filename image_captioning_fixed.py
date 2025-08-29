#!/usr/bin/env python3
"""
FIXED Image Captioning System for Accessibility Tools

This script fixes the critical tokenization issues found in the previous version:
1. Proper handling of START and END tokens
2. Consistent token usage in training and generation
3. Better caption processing

Author: Image Captioning Assistant (Fixed Version)
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

# Gradio for demo
import gradio as gr

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
DATA_DIR = './data/Flickr8k/'
IMAGES_DIR = os.path.join(DATA_DIR, 'Images/')
ANNOTATIONS_FILE = os.path.join(DATA_DIR, 'captions.txt')
FEATURES_FILE = 'features_fixed.npy'
MODEL_FILE = 'models/caption_model_fixed.h5'
TOKENIZER_FILE = 'tokenizer_fixed.pkl'

# Configuration parameters - FIXED VERSION
MAX_LENGTH = 25
VOCAB_SIZE = 3000
FEATURE_DIM = 512
EMBEDDING_DIM = 128
LSTM_UNITS = 128
EPOCHS = 5  # Increased for better learning
BATCH_SIZE = 16
FEATURE_BATCH_SIZE = 16

# CRITICAL FIX: Use proper start/end tokens
START_TOKEN = 'startseq'  # Single token without special characters
END_TOKEN = 'endseq'      # Single token without special characters

print("FIXED Image Captioning System for Accessibility Tools")
print("=" * 60)

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


def load_captions(max_pairs=10000):
    """
    Load image captions from the annotations file with PROPER token handling.
    
    Args:
        max_pairs (int): Maximum number of caption pairs to load
        
    Returns:
        dict: Dictionary with image names as keys and lists of captions as values
    """
    print(f"Loading captions (max {max_pairs} pairs)...")
    
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


def create_vocabulary(captions_dict):
    """
    Create vocabulary from captions with PROPER token handling.
    
    Args:
        captions_dict (dict): Image captions
        
    Returns:
        Tokenizer: Fitted tokenizer
    """
    print("Creating vocabulary...")
    
    # Collect all captions
    all_captions = []
    for captions in captions_dict.values():
        all_captions.extend(captions)
    
    print(f"Total captions for vocabulary: {len(all_captions)}")
    
    # Show sample captions to verify tokens
    print("Sample captions (first 3):")
    for i, cap in enumerate(all_captions[:3]):
        print(f"  {i+1}: {cap}")
    
    # Create tokenizer with proper configuration
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


def extract_features(image_files, batch_size=FEATURE_BATCH_SIZE, save_features=True):
    """
    Extract image features using VGG16 with memory-efficient batch processing.
    
    Args:
        image_files (list): List of image file paths
        batch_size (int): Batch size for processing
        save_features (bool): Whether to save features to disk
        
    Returns:
        dict: Dictionary of image features
    """
    print(f"Extracting features for {len(image_files)} images...")
    
    # Check if features already exist
    features_file_path = FEATURES_FILE
    temp_features_file = 'temp_' + features_file_path
    
    if os.path.exists(features_file_path):
        print(f"Loading pre-computed features from {features_file_path}")
        features_dict = np.load(features_file_path, allow_pickle=True).item()
        print(f"Loaded features for {len(features_dict)} images")
        return features_dict
    
    # Check for temporary file (resume interrupted processing)
    if os.path.exists(temp_features_file):
        print(f"Resuming from temporary file: {temp_features_file}")
        features_dict = np.load(temp_features_file, allow_pickle=True).item()
        print(f"Resumed with {len(features_dict)} existing features")
    else:
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
            
            # Save progress every 10 batches (or ~160 images with default batch size)
            if (i // batch_size + 1) % 10 == 0:
                np.save(temp_features_file, features_dict)
                print(f"Progress saved to {temp_features_file}")
    
    print(f"Feature extraction complete. Features for {len(features_dict)} images")
    
    # Save final features
    if save_features:
        np.save(features_file_path, features_dict)
        print(f"Features saved to {features_file_path}")
        
        # Remove temporary file
        if os.path.exists(temp_features_file):
            os.remove(temp_features_file)
            print("Temporary features file cleaned up")
    
    return features_dict


def create_sequences_fixed(captions_dict, tokenizer, features_dict, max_length=MAX_LENGTH, batch_size=100):
    """
    Create input sequences for training with FIXED token handling.
    
    Args:
        captions_dict (dict): Image captions
        tokenizer: Fitted tokenizer
        features_dict (dict): Image features
        max_length (int): Maximum sequence length
        batch_size (int): Number of sequences to process at once
        
    Returns:
        tuple: Training data (X1, X2, y)
    """
    print("Creating training sequences with FIXED token processing...")
    
    # Check if sequences already exist
    sequences_file = 'training_sequences_fixed.npz'
    if os.path.exists(sequences_file):
        print(f"Loading pre-computed sequences from {sequences_file}")
        data = np.load(sequences_file)
        return data['X1'], data['X2'], data['y']
    
    # Get token indices
    start_token_idx = tokenizer.word_index.get(START_TOKEN, 1)
    end_token_idx = tokenizer.word_index.get(END_TOKEN, 1)
    
    print(f"Using token indices: {START_TOKEN}={start_token_idx}, {END_TOKEN}={end_token_idx}")
    
    # Count total sequences first
    total_sequences = 0
    valid_items = []
    
    for image_name, captions in captions_dict.items():
        if image_name in features_dict:
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                total_sequences += len(seq) - 1  # Each caption creates len(seq)-1 training pairs
                valid_items.append((image_name, caption, seq))
    
    print(f"Will create approximately {total_sequences} training sequences")
    
    # Process in batches to avoid memory issues
    X1_list, X2_list, y_list = [], [], []
    X1, X2, y = None, None, None  # Initialize arrays
    current_batch_size = 0
    
    for item_idx, (image_name, caption, seq) in enumerate(valid_items):
        feature = features_dict[image_name]
        
        # Create input-output pairs for this caption
        for i in range(1, len(seq)):
            # Input sequence (everything before current position)
            in_seq = seq[:i]
            out_seq = seq[i]  # Target word at current position
            
            # Pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            
            # One-hot encode output
            out_seq = to_categorical([out_seq], num_classes=len(tokenizer.word_index) + 1)[0]
            
            X1_list.append(feature)
            X2_list.append(in_seq)
            y_list.append(out_seq)
            current_batch_size += 1
            
            # Process batch when it reaches the specified size
            if current_batch_size >= batch_size:
                # Convert current batch to arrays
                batch_X1 = np.array(X1_list)
                batch_X2 = np.array(X2_list)
                batch_y = np.array(y_list)
                
                if X1 is None:  # First batch - initialize arrays
                    X1 = batch_X1
                    X2 = batch_X2
                    y = batch_y
                else:  # Subsequent batches - concatenate
                    X1 = np.concatenate([X1, batch_X1], axis=0)
                    X2 = np.concatenate([X2, batch_X2], axis=0)
                    y = np.concatenate([y, batch_y], axis=0)
                
                # Clear lists and collect garbage
                X1_list, X2_list, y_list = [], [], []
                current_batch_size = 0
                gc.collect()
                
                print(f"Processed {len(X1)} sequences so far...")
        
        # Progress update
        if (item_idx + 1) % 100 == 0:
            print(f"Processed {item_idx + 1}/{len(valid_items)} image-caption pairs")
    
    # Process remaining sequences
    if X1_list:
        batch_X1 = np.array(X1_list)
        batch_X2 = np.array(X2_list)
        batch_y = np.array(y_list)
        
        if X1 is None:  # If no batches were processed yet
            X1 = batch_X1
            X2 = batch_X2
            y = batch_y
        else:
            X1 = np.concatenate([X1, batch_X1], axis=0)
            X2 = np.concatenate([X2, batch_X2], axis=0)
            y = np.concatenate([y, batch_y], axis=0)
    
    print(f"Created {len(X1)} training sequences")
    print(f"Feature shape: {X1.shape}")
    print(f"Sequence shape: {X2.shape}")
    print(f"Target shape: {y.shape}")
    
    # Save sequences for future use
    print(f"Saving sequences to {sequences_file}")
    np.savez_compressed(sequences_file, X1=X1, X2=X2, y=y)
    
    return X1, X2, y


# Step 3: Model Architecture

class CaptionModel:
    """Image captioning model using VGG16 + LSTM with FIXED architecture."""
    
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
        """Build the encoder-decoder model."""
        print("Building caption model...")
        
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
        combined = Dropout(0.3)(combined)
        
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
    
    def generate_caption_fixed(self, photo, tokenizer, max_length=MAX_LENGTH):
        """
        Generate caption for an image using FIXED greedy search.
        
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


# Step 4: Training Functions

def train_model_fixed(model, X1_train, X2_train, y_train, X1_val, X2_val, y_val):
    """
    Train the caption model with FIXED configuration.
    
    Args:
        model: Caption model to train
        X1_train, X2_train, y_train: Training data
        X1_val, X2_val, y_val: Validation data
        
    Returns:
        History object
    """
    print("Starting model training...")
    
    # Create model directory
    os.makedirs('models', exist_ok=True)
    
    # Create callbacks
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss')
    ]
    
    # Train model
    history = model.model.fit(
        [X1_train, X2_train], y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=([X1_val, X2_val], y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("Training completed")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (Fixed Version)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy (Fixed Version)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_fixed.png', dpi=150, bbox_inches='tight')
    print("Training history saved to training_history_fixed.png")
    
    return history


# Step 5: Evaluation and Testing

def evaluate_model_fixed(model, tokenizer, features_dict, captions_dict, num_samples=10):
    """
    Evaluate model and generate sample predictions with FIXED generation.
    
    Args:
        model: Trained caption model
        tokenizer: Fitted tokenizer  
        features_dict (dict): Image features
        captions_dict (dict): Ground truth captions
        num_samples (int): Number of sample predictions to generate
    """
    print(f"Generating {num_samples} sample predictions...")
    
    # Select random images
    image_names = list(features_dict.keys())
    sample_images = random.sample(image_names, min(num_samples, len(image_names)))
    
    # Generate predictions
    predictions = []
    ground_truths = []
    
    for img_name in sample_images:
        # Get image feature
        feature = features_dict[img_name]
        
        # Generate caption
        predicted_caption = model.generate_caption_fixed(feature, tokenizer)
        
        # Get ground truth (first caption)
        if img_name in captions_dict and captions_dict[img_name]:
            gt_caption = captions_dict[img_name][0]
            # Remove start and end tokens for display
            gt_caption = gt_caption.replace(START_TOKEN, '').replace(END_TOKEN, '').strip()
        else:
            gt_caption = "No ground truth available"
        
        predictions.append((img_name, predicted_caption, gt_caption))
        ground_truths.append([gt_caption.split()])
    
    # Display results
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS (FIXED VERSION)")
    print("="*80)
    
    for i, (img_name, pred, gt) in enumerate(predictions, 1):
        print(f"\n{i}. Image: {img_name}")
        print(f"   Predicted: {pred}")
        print(f"   Ground Truth: {gt}")
        print("-" * 60)
    
    # Create visual comparison
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Image Captioning Results (Fixed Version)', fontsize=16)
    
    for i, (img_name, pred, gt) in enumerate(predictions):
        if i >= 10:  # Limit to 10 images
            break
            
        row = i // 5
        col = i % 5
        
        try:
            # Load and display image
            img_path = os.path.join(IMAGES_DIR, img_name)
            if os.path.exists(img_path):
                image = Image.open(img_path)
                axes[row, col].imshow(image)
            else:
                axes[row, col].text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
            
            axes[row, col].set_title(f"Image {i+1}", fontsize=10)
            axes[row, col].axis('off')
            
            # Add captions below image
            caption_text = f"Pred: {pred[:50]}{'...' if len(pred) > 50 else ''}\nGT: {gt[:50]}{'...' if len(gt) > 50 else ''}"
            axes[row, col].text(0.5, -0.1, caption_text, ha='center', va='top', 
                              transform=axes[row, col].transAxes, fontsize=8, wrap=True)
            
        except Exception as e:
            axes[row, col].text(0.5, 0.5, f'Error loading\n{img_name}', ha='center', va='center')
            axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(predictions), 10):
        row = i // 5
        col = i % 5
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions_fixed.png', dpi=150, bbox_inches='tight')
    print(f"\nVisual results saved to sample_predictions_fixed.png")
    
    return predictions


# Step 6: Main Training Pipeline

def main():
    """Main training pipeline with FIXED implementation."""
    print("Starting FIXED image captioning training pipeline...")
    
    # Step 1: Check data
    check_data_exists()
    
    # Step 2: Load captions
    captions_dict = load_captions(max_pairs=10000)  # Start with subset
    
    # Step 3: Create vocabulary
    tokenizer = create_vocabulary(captions_dict)
    
    # Step 4: Extract image features
    image_files = [os.path.join(IMAGES_DIR, img) for img in captions_dict.keys() 
                   if os.path.exists(os.path.join(IMAGES_DIR, img))]
    print(f"Found {len(image_files)} valid image files")
    
    features_dict = extract_features(image_files)
    
    # Step 5: Create training sequences
    X1, X2, y = create_sequences_fixed(captions_dict, tokenizer, features_dict)
    
    # Step 6: Split data
    print("Splitting data for training/validation...")
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
        X1, X2, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X1_train)}")
    print(f"Validation samples: {len(X1_val)}")
    
    # Step 7: Build and train model
    vocab_size = len(tokenizer.word_index) + 1
    model = CaptionModel(vocab_size, MAX_LENGTH)
    
    history = train_model_fixed(model, X1_train, X2_train, y_train, X1_val, X2_val, y_val)
    
    # Step 8: Save tokenizer
    print(f"Saving tokenizer to {TOKENIZER_FILE}")
    with open(TOKENIZER_FILE, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Step 9: Evaluate model
    evaluate_model_fixed(model, tokenizer, features_dict, captions_dict)
    
    print("\n" + "="*60)
    print("FIXED TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Tokenizer saved to: {TOKENIZER_FILE}")
    print(f"Features saved to: {FEATURES_FILE}")
    print(f"Visual results: sample_predictions_fixed.png")
    print(f"Training plots: training_history_fixed.png")


if __name__ == "__main__":
    main()
