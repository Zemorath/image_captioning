#!/usr/bin/env python3
"""
Entry-Level Multimodal Image Captioning for Accessibility Tools

This script implements a complete image captioning system using:
- VGG16 for feature extraction
- LSTM decoder for text generation
- Flickr8k dataset for training
- Gradio interface for demonstration

Author: Image Captioning Assistant
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
FEATURES_FILE = 'features.npy'
MODEL_FILE = 'models/caption_model.h5'
TOKENIZER_FILE = 'tokenizer.pkl'

# Configuration parameters
MAX_LENGTH = 35
VOCAB_SIZE = 5000
FEATURE_DIM = 512
EMBEDDING_DIM = 256
LSTM_UNITS = 256
EPOCHS = 5  # Reduced for faster testing (increase to 20+ for better results)
BATCH_SIZE = 32
FEATURE_BATCH_SIZE = 32  # Batch size for feature extraction (reduce if OOM issues)

print("Image Captioning System for Accessibility Tools")
print("=" * 50)

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
            "Flickr8k dataset not found. Please download it from:\n"
            "kaggle datasets download -d adityajn105/flickr8k\n"
            f"And extract to {DATA_DIR}"
        )
    
    if not os.path.exists(IMAGES_DIR):
        print(f"Warning: Images directory not found at {IMAGES_DIR}")
        print("Expected structure: data/Flickr8k/Images/")
    
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"Warning: Captions file not found at {ANNOTATIONS_FILE}")
        print("Looking for alternative caption files...")
        
        # Check for alternative caption file names
        alt_files = [
            os.path.join(DATA_DIR, 'Flickr8k.token.txt'),
            os.path.join(DATA_DIR, 'Flickr_8k.trainImages.txt'),
            os.path.join(DATA_DIR, 'captions.txt')
        ]
        
        for alt_file in alt_files:
            if os.path.exists(alt_file):
                print(f"Found alternative caption file: {alt_file}")
                return alt_file
        
        raise ValueError(f"No caption file found. Expected at {ANNOTATIONS_FILE}")
    
    return ANNOTATIONS_FILE


def load_captions(filename):
    """
    Load captions from text file.
    
    Args:
        filename (str): Path to captions file
        
    Returns:
        dict: Dictionary mapping image names to list of captions
    """
    print(f"Loading captions from {filename}...")
    captions_dict = {}
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines[1:]:  # Skip header if present
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
            
            # Add start and end tokens
            caption = f"<start> {caption.lower()} <end>"
            
            if image_base not in captions_dict:
                captions_dict[image_base] = []
            captions_dict[image_base].append(caption)
    
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
    Create vocabulary from captions.
    
    Args:
        captions_dict (dict): Dictionary of image captions
        
    Returns:
        tuple: (tokenizer, vocab_size)
    """
    print("Creating vocabulary...")
    
    # Collect all captions
    all_captions = []
    for captions in captions_dict.values():
        all_captions.extend(captions)
    
    print(f"Total captions: {len(all_captions)}")
    
    # Create tokenizer
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<unk>')
    tokenizer.fit_on_texts(all_captions)
    
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")
    
    # Show most common words
    print("Most common words:")
    word_freq = [(word, count) for word, count in tokenizer.word_counts.items()]
    word_freq.sort(key=lambda x: x[1], reverse=True)
    for word, count in word_freq[:10]:
        print(f"  {word}: {count}")
    
    return tokenizer, vocab_size


def extract_features(images_dir, save_features=True, batch_size=32):
    """
    Extract features from images using VGG16 with batch processing.
    
    Args:
        images_dir (str): Directory containing images
        save_features (bool): Whether to save features to disk
        batch_size (int): Number of images to process in each batch
        
    Returns:
        dict: Dictionary mapping image names to feature vectors
    """
    features_file_path = FEATURES_FILE
    temp_features_file = 'features_temp.npy'
    
    # Load existing features if available
    if os.path.exists(features_file_path):
        print(f"Loading pre-extracted features from {features_file_path}")
        features_dict = np.load(features_file_path, allow_pickle=True).item()
        print(f"Loaded features for {len(features_dict)} images")
        return features_dict
    
    # Load temporary features if available (resume from interruption)
    if os.path.exists(temp_features_file):
        print(f"Loading temporary features from {temp_features_file}")
        features_dict = np.load(temp_features_file, allow_pickle=True).item()
        print(f"Resuming from {len(features_dict)} processed images")
    else:
        features_dict = {}
    
    print("Extracting image features using VGG16 with batch processing...")
    
    # Load VGG16 model
    vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    vgg_model.trainable = False
    
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return features_dict
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    # Filter out already processed images
    remaining_files = [f for f in image_files if f not in features_dict]
    
    print(f"Processing {len(remaining_files)} remaining images out of {len(image_files)} total in batches of {batch_size}...")
    
    # Process images in batches
    for i in range(0, len(remaining_files), batch_size):
        batch_files = remaining_files[i:i + batch_size]
        batch_images = []
        valid_files = []
        
        # Load batch of images
        for filename in batch_files:
            try:
                img_path = os.path.join(images_dir, filename)
                img = load_img(img_path, target_size=(224, 224))
                img_array = img_to_array(img)
                img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
                batch_images.append(img_array)
                valid_files.append(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
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
        
        # Save progress every 10 batches (or ~320 images with default batch size)
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


def create_sequences(captions_dict, tokenizer, features_dict, max_length=MAX_LENGTH, batch_size=500):
    """
    Create input sequences for training with memory-efficient batch processing.
    
    Args:
        captions_dict (dict): Image captions
        tokenizer: Fitted tokenizer
        features_dict (dict): Image features
        max_length (int): Maximum sequence length
        batch_size (int): Number of sequences to process at once
        
    Returns:
        tuple: Training data (X1, X2, y)
    """
    print("Creating training sequences with memory-efficient processing...")
    
    # Check if sequences already exist
    sequences_file = 'training_sequences.npz'
    if os.path.exists(sequences_file):
        print(f"Loading pre-computed sequences from {sequences_file}")
        data = np.load(sequences_file)
        return data['X1'], data['X2'], data['y']
    
    # Count total sequences first
    total_sequences = 0
    valid_items = []
    
    for image_name, captions in captions_dict.items():
        if image_name in features_dict:
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                total_sequences += len(seq) - 1  # Each caption creates len(seq)-1 training pairs
                valid_items.append((image_name, caption))
    
    print(f"Will create approximately {total_sequences} training sequences")
    
    # Process in batches to avoid memory issues
    X1_list, X2_list, y_list = [], [], []
    X1, X2, y = None, None, None  # Initialize arrays
    current_batch_size = 0
    
    for item_idx, (image_name, caption) in enumerate(valid_items):
        feature = features_dict[image_name]
        seq = tokenizer.texts_to_sequences([caption])[0]
        
        # Create input-output pairs for this caption
        for i in range(1, len(seq)):
            # Input sequence
            in_seq = seq[:i]
            out_seq = seq[i]
            
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


# Step 3: Model Building

class CaptionModel:
    """Simple encoder-decoder model for image captioning."""
    
    def __init__(self, vocab_size, max_length, feature_dim=FEATURE_DIM):
        """
        Initialize the caption model.
        
        Args:
            vocab_size (int): Size of vocabulary
            max_length (int): Maximum sequence length
            feature_dim (int): Dimension of image features
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.feature_dim = feature_dim
        self.model = None
        self.build_model()
    
    def build_model(self):
        """Build the encoder-decoder model."""
        print("Building caption model...")
        
        # Encoder - Image features
        image_input = Input(shape=(self.feature_dim,))
        image_features = Dense(EMBEDDING_DIM, activation='relu')(image_input)
        image_features = Dropout(0.5)(image_features)
        
        # Decoder - Text sequences
        sequence_input = Input(shape=(self.max_length,))
        seq_features = Embedding(self.vocab_size, EMBEDDING_DIM, mask_zero=True)(sequence_input)
        seq_features = Dropout(0.5)(seq_features)
        seq_features = LSTM(LSTM_UNITS)(seq_features)
        
        # Merge encoder and decoder
        decoder = Add()([image_features, seq_features])
        decoder = Dense(LSTM_UNITS, activation='relu')(decoder)
        decoder = Dropout(0.5)(decoder)
        output = Dense(self.vocab_size, activation='softmax')(decoder)
        
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
    
    def generate_caption(self, photo, tokenizer, max_length=MAX_LENGTH):
        """
        Generate caption for an image using greedy search.
        
        Args:
            photo: Image features
            tokenizer: Fitted tokenizer
            max_length (int): Maximum caption length
            
        Returns:
            str: Generated caption
        """
        # Start with start token
        in_text = '<start>'
        
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
            if word is None or word == '<end>':
                break
                
            # Add word to caption
            in_text += ' ' + word
        
        # Remove start token and return
        caption = in_text.replace('<start>', '').strip()
        return caption


# Step 4: Training Functions

def train_model(model, X1_train, X2_train, y_train, X1_val, X2_val, y_val):
    """
    Train the caption model.
    
    Args:
        model: Caption model to train
        X1_train, X2_train, y_train: Training data
        X1_val, X2_val, y_val: Validation data
        
    Returns:
        History object
    """
    print("Starting model training...")
    
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
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return history


# Step 5: Evaluation Functions

def evaluate_model(model, X1_val, captions_dict, tokenizer, features_dict):
    """
    Evaluate model using BLEU scores.
    
    Args:
        model: Trained caption model
        X1_val: Validation image features
        captions_dict: Original captions
        tokenizer: Fitted tokenizer
        features_dict: Image features dictionary
    """
    print("Evaluating model with BLEU scores...")
    
    # Sample some validation images
    val_images = list(features_dict.keys())[:100]  # Use first 100 for evaluation
    
    references = []
    predictions = []
    
    for img_name in val_images:
        if img_name not in captions_dict:
            continue
            
        # Get image features
        features = features_dict[img_name]
        
        # Generate caption
        predicted_caption = model.generate_caption(features, tokenizer)
        
        # Get reference captions (remove start/end tokens)
        ref_captions = []
        for cap in captions_dict[img_name]:
            cap_clean = cap.replace('<start>', '').replace('<end>', '').strip()
            ref_captions.append(cap_clean.split())
        
        # Add to evaluation lists
        predictions.append(predicted_caption.split())
        references.append(ref_captions)
    
    # Calculate BLEU scores
    smoothing = SmoothingFunction().method1
    
    bleu1 = corpus_bleu(references, predictions, weights=(1.0, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(references, predictions, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(references, predictions, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(references, predictions, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    
    # Visualize some predictions
    visualize_predictions(val_images[:5], predictions[:5], references[:5], features_dict)


def visualize_predictions(image_names, predictions, references, features_dict):
    """Visualize sample predictions."""
    print("Visualizing sample predictions...")
    
    plt.figure(figsize=(15, 10))
    
    for i, (img_name, pred, ref) in enumerate(zip(image_names[:5], predictions, references)):
        try:
            # Load image
            img_path = os.path.join(IMAGES_DIR, img_name)
            if os.path.exists(img_path):
                img = load_img(img_path)
                
                plt.subplot(2, 3, i + 1)
                plt.imshow(img)
                plt.title(f'Predicted: {" ".join(pred)}\nGround Truth: {" ".join(ref[0])}', fontsize=8)
                plt.axis('off')
        except Exception as e:
            print(f"Error visualizing {img_name}: {e}")
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()


# Step 6: Gradio Demo

def create_demo():
    """Create Gradio demo interface."""
    print("Setting up Gradio demo...")
    
    # Load trained model and tokenizer
    try:
        model = load_model(MODEL_FILE)
        with open(TOKENIZER_FILE, 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load VGG16 for feature extraction
        vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        vgg_model.trainable = False
        
        print("Model and tokenizer loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    def caption_image(image):
        """Generate caption for uploaded image."""
        try:
            # Preprocess image
            img = image.resize((224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
            
            # Extract features
            features = vgg_model.predict(img_array, verbose=0)[0]
            
            # Generate caption using the same method as CaptionModel
            in_text = '<start>'
            for _ in range(MAX_LENGTH):
                sequence = tokenizer.texts_to_sequences([in_text])[0]
                sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)
                
                yhat = model.predict([features.reshape(1, -1), sequence], verbose=0)
                yhat = np.argmax(yhat)
                
                word = None
                for w, i in tokenizer.word_index.items():
                    if i == yhat:
                        word = w
                        break
                
                if word is None or word == '<end>':
                    break
                    
                in_text += ' ' + word
            
            caption = in_text.replace('<start>', '').strip()
            
            return caption, image
            
        except Exception as e:
            return f"Error generating caption: {e}", image
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=caption_image,
        inputs=gr.Image(type='pil'),
        outputs=[
            gr.Textbox(label='Generated Caption'),
            gr.Image(label='Input Image')
        ],
        title='Image Captioning for Accessibility Tools',
        description='Upload an image to generate a descriptive caption that can aid visually impaired users.',
        examples=None
    )
    
    return interface


# Step 7: Main Execution

def main():
    """Main execution function."""
    try:
        print("Starting Image Captioning System...")
        
        # Check if dataset exists
        caption_file = check_data_exists()
        
        # Load and preprocess data
        captions_dict = load_captions(caption_file)
        tokenizer, vocab_size = create_vocabulary(captions_dict)
        
        # Save tokenizer
        with open(TOKENIZER_FILE, 'wb') as f:
            pickle.dump(tokenizer, f)
        print(f"Tokenizer saved to {TOKENIZER_FILE}")
        
        # Extract image features
        features_dict = extract_features(IMAGES_DIR, batch_size=FEATURE_BATCH_SIZE)
        
        if len(features_dict) == 0:
            print("No image features extracted. Please check your dataset.")
            return
        
        # Create training sequences
        X1, X2, y = create_sequences(captions_dict, tokenizer, features_dict)
        
        # Split data
        indices = np.arange(len(X1))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        X1_train, X1_val = X1[train_idx], X1[val_idx]
        X2_train, X2_val = X2[train_idx], X2[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"Training samples: {len(X1_train)}")
        print(f"Validation samples: {len(X1_val)}")
        
        # Build and train model
        caption_model = CaptionModel(vocab_size, MAX_LENGTH)
        
        # Train model
        history = train_model(
            caption_model, 
            X1_train, X2_train, y_train,
            X1_val, X2_val, y_val
        )
        
        # Evaluate model
        evaluate_model(caption_model, X1_val, captions_dict, tokenizer, features_dict)
        
        # Create demo
        demo = create_demo()
        if demo:
            print("Launching Gradio demo...")
            demo.launch(share=True)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise


if __name__ == '__main__':
    main()
