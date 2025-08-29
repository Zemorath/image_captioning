#!/usr/bin/env python3
"""
Fixed demo script that handles the NotEqual layer loading issue.
"""

import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import gradio as gr
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Configuration
MAX_LENGTH = 25
FEATURE_DIM = 512
MODEL_FILE = 'models/caption_model.h5'
TOKENIZER_FILE = 'tokenizer.pkl'

def load_tokenizer():
    """Load the saved tokenizer."""
    try:
        with open(TOKENIZER_FILE, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"Tokenizer loaded successfully from {TOKENIZER_FILE}")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

def create_model_architecture(vocab_size):
    """Recreate the model architecture without mask_zero to avoid NotEqual layer."""
    from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Add, Dropout
    from tensorflow.keras.models import Model
    
    # Encoder - Image features
    image_input = Input(shape=(FEATURE_DIM,))
    image_features = Dense(128, activation='relu')(image_input)
    image_features = Dropout(0.5)(image_features)
    
    # Decoder - Text sequences  
    sequence_input = Input(shape=(MAX_LENGTH,))
    seq_features = Embedding(vocab_size, 128, mask_zero=False)(sequence_input)  # No mask_zero
    seq_features = Dropout(0.5)(seq_features)
    seq_features = LSTM(128)(seq_features)
    
    # Combine features
    combined = Add()([image_features, seq_features])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(vocab_size, activation='softmax')(combined)
    
    model = Model(inputs=[image_input, sequence_input], outputs=output)
    return model

def load_model_weights(tokenizer):
    """Load model by recreating architecture and loading weights."""
    try:
        vocab_size = len(tokenizer.word_index) + 1
        print(f"Creating model architecture for vocab size: {vocab_size}")
        
        # Create model with same architecture but without mask_zero
        model = create_model_architecture(vocab_size)
        
        # Load the saved model to get weights
        print(f"Loading weights from {MODEL_FILE}")
        saved_model = tf.keras.models.load_model(MODEL_FILE, compile=False)
        
        # Transfer weights layer by layer (skipping problematic layers if needed)
        for i, layer in enumerate(model.layers):
            if i < len(saved_model.layers):
                try:
                    layer.set_weights(saved_model.get_layer(index=i).get_weights())
                except Exception as e:
                    print(f"Warning: Could not transfer weights for layer {i}: {e}")
        
        print("Model loaded successfully with weight transfer")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try direct loading with custom scope
        try:
            print("Attempting direct load with custom object scope...")
            with tf.keras.utils.custom_object_scope({'NotEqual': tf.keras.layers.Lambda}):
                model = tf.keras.models.load_model(MODEL_FILE, compile=False)
                print("Model loaded with custom object scope")
                return model
        except Exception as e2:
            print(f"Both loading methods failed: {e2}")
            return None

def extract_image_features(image_path):
    """Extract features from image using VGG16."""
    try:
        # Load VGG16 model
        vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        
        # Load and preprocess image
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extract features
        features = vgg_model.predict(img_array, verbose=0)[0]
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def generate_caption(model, tokenizer, image_features, max_length=MAX_LENGTH):
    """Generate caption for image features."""
    try:
        # Start with start token
        in_text = '<start>'
        
        for _ in range(max_length):
            # Encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)[0]
            
            # Predict next word
            pred = model.predict([np.array([image_features]), np.array([sequence])], verbose=0)[0]
            pred_id = np.argmax(pred)
            
            # Map prediction to word
            word = None
            for w, idx in tokenizer.word_index.items():
                if idx == pred_id:
                    word = w
                    break
            
            if word is None or word == '<end>':
                break
                
            in_text += ' ' + word
        
        # Clean up caption
        caption = in_text.replace('<start>', '').strip()
        return caption
        
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "Error generating caption"

def caption_image(image):
    """Main function to caption an uploaded image."""
    try:
        # Save uploaded image temporarily
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        # Extract features
        features = extract_image_features(temp_path)
        if features is None:
            return "Error: Could not extract image features"
        
        # Generate caption
        caption = generate_caption(model, tokenizer, features)
        return f"Caption: {caption}"
        
    except Exception as e:
        return f"Error: {str(e)}"

# Load model and tokenizer
print("Loading tokenizer and model...")
tokenizer = load_tokenizer()
if tokenizer is None:
    print("Failed to load tokenizer")
    exit(1)

model = load_model_weights(tokenizer)
if model is None:
    print("Failed to load model")
    exit(1)

print("Setup complete! Creating Gradio interface...")

# Create Gradio interface
iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🖼️ Image Caption Generator",
    description="Upload an image and get an AI-generated caption! This model was trained on the Flickr8k dataset.",
    examples=None
)

if __name__ == "__main__":
    print("Starting Gradio demo...")
    print("Open your browser to: http://localhost:7860")
    iface.launch(server_name="0.0.0.0", server_port=7860)
