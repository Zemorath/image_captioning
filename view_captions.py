#!/usr/bin/env python3
"""
Simple script to generate and display captions for images.
Shows text output of generated captions.
"""

import tensorflow as tf
import numpy as np
import pickle
import os
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Add, Dropout
from tensorflow.keras.models import Model

# Configuration
MAX_LENGTH = 25
FEATURE_DIM = 512
EMBEDDING_DIM = 128
LSTM_UNITS = 128
TOKENIZER_FILE = 'tokenizer.pkl'

def load_tokenizer():
    """Load the saved tokenizer."""
    with open(TOKENIZER_FILE, 'rb') as f:
        return pickle.load(f)

def build_inference_model(vocab_size):
    """Build model for inference."""
    # Encoder - Image features
    image_input = Input(shape=(FEATURE_DIM,))
    image_features = Dense(EMBEDDING_DIM, activation='relu')(image_input)
    image_features = Dropout(0.5)(image_features)
    
    # Decoder - Text sequences  
    sequence_input = Input(shape=(MAX_LENGTH,))
    seq_features = Embedding(vocab_size, EMBEDDING_DIM)(sequence_input)
    seq_features = Dropout(0.5)(seq_features)
    seq_features = LSTM(LSTM_UNITS)(seq_features)
    
    # Combine features
    combined = Add()([image_features, seq_features])
    combined = Dense(EMBEDDING_DIM, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(vocab_size, activation='softmax')(combined)
    
    model = Model(inputs=[image_input, sequence_input], outputs=output)
    return model

def load_model_and_tokenizer():
    """Load model and tokenizer."""
    print("Loading tokenizer and model...")
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer.word_index) + 1
    
    model = build_inference_model(vocab_size)
    model.load_weights('models/caption_model.h5', by_name=True, skip_mismatch=True)
    
    return model, tokenizer

def extract_image_features(image_path):
    """Extract features from an image."""
    # Load VGG16 for feature extraction
    vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    
    # Load and preprocess image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract features
    features = vgg_model.predict(img_array, verbose=0)[0]
    return features

def generate_caption_beam_search(model, tokenizer, image_features, beam_width=3):
    """Generate caption using beam search for better results."""
    vocab_size = len(tokenizer.word_index) + 1
    
    # Initialize with start token
    start_seq = tokenizer.texts_to_sequences(['<start>'])[0]
    
    # Beam search
    sequences = [[start_seq, 0.0]]
    
    for _ in range(MAX_LENGTH):
        all_candidates = []
        
        for seq, score in sequences:
            if len(seq) > 0 and seq[-1] == tokenizer.word_index.get('<end>', 0):
                all_candidates.append([seq, score])
                continue
                
            # Pad sequence
            padded_seq = pad_sequences([seq], maxlen=MAX_LENGTH)[0]
            
            # Predict next words
            preds = model.predict([np.array([image_features]), np.array([padded_seq])], verbose=0)[0]
            
            # Get top candidates
            top_indices = np.argsort(preds)[-beam_width:]
            
            for idx in top_indices:
                candidate_seq = seq + [idx]
                candidate_score = score - np.log(preds[idx] + 1e-8)
                all_candidates.append([candidate_seq, candidate_score])
        
        # Select best sequences
        ordered = sorted(all_candidates, key=lambda x: x[1])
        sequences = ordered[:beam_width]
    
    # Get best sequence
    best_seq = sequences[0][0]
    
    # Convert to words
    words = []
    for idx in best_seq:
        word = None
        for w, i in tokenizer.word_index.items():
            if i == idx:
                word = w
                break
        if word and word not in ['<start>', '<end>']:
            words.append(word)
    
    return ' '.join(words)

def simple_generate_caption(model, tokenizer, image_features):
    """Generate caption using greedy search."""
    in_text = '<start>'
    
    for _ in range(MAX_LENGTH):
        # Encode the sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)[0]
        
        # Predict next word
        prediction = model.predict([np.array([image_features]), np.array([sequence])], verbose=0)
        pred_id = np.argmax(prediction[0])
        
        # Convert prediction to word
        word = None
        for w, idx in tokenizer.word_index.items():
            if idx == pred_id:
                word = w
                break
        
        if word is None or word == '<end>':
            break
            
        in_text += ' ' + word
    
    # Clean up the caption
    caption = in_text.replace('<start>', '').strip()
    return caption

def main():
    """Generate captions for sample images."""
    print("🖼️  IMAGE CAPTION GENERATOR")
    print("=" * 50)
    
    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get sample images
    images_dir = "./data/Flickr8k/Images/"
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("❌ No images found!")
        return
    
    # Sample random images
    sample_images = random.sample(image_files, min(5, len(image_files)))
    
    print(f"\n📸 Generating captions for {len(sample_images)} images:")
    print("-" * 50)
    
    for i, image_file in enumerate(sample_images, 1):
        print(f"\n{i}. Image: {image_file}")
        
        try:
            # Extract features
            image_path = os.path.join(images_dir, image_file)
            features = extract_image_features(image_path)
            
            # Generate caption using simple method
            caption = simple_generate_caption(model, tokenizer, features)
            print(f"   Caption: '{caption}'")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Caption generation complete!")
    print("\nTo view visual results, check these files:")
    print("- sample_predictions.png (images with captions)")
    print("- training_history.png (training progress)")

if __name__ == "__main__":
    main()
