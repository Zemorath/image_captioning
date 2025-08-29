#!/usr/bin/env python3
"""
Simple Demo Script for Image Captioning

This script demonstrates the image captioning functionality using a pre-trained model
or a minimal example for testing without the full Flickr8k dataset.
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import gradio as gr
import os
import pickle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
MAX_LENGTH = 35
FEATURE_DIM = 512

def load_demo_model():
    """Load the trained model for demo, or create a dummy model for testing."""
    model_path = 'models/caption_model.h5'
    tokenizer_path = 'tokenizer.pkl'
    
    try:
        # Try to load the real trained model
        model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        print("✓ Loaded trained model and tokenizer")
        return model, tokenizer
        
    except Exception as e:
        print(f"⚠️  Could not load trained model: {e}")
        print("Creating demo model for testing...")
        
        # Create a simple demo model and tokenizer for testing
        model, tokenizer = create_demo_model()
        return model, tokenizer

def create_demo_model():
    """Create a simple demo model for testing purposes."""
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Add
    
    # Create a simple tokenizer with basic vocabulary
    demo_captions = [
        "a dog is running on the beach",
        "a cat is sitting on a chair", 
        "a person is walking in the park",
        "a bird is flying in the sky",
        "a car is driving on the road"
    ]
    
    tokenizer = Tokenizer(num_words=1000, oov_token='<unk>')
    tokenizer.fit_on_texts(demo_captions)
    vocab_size = len(tokenizer.word_index) + 1
    
    # Create a simple model
    image_input = Input(shape=(FEATURE_DIM,))
    image_features = Dense(256, activation='relu')(image_input)
    
    sequence_input = Input(shape=(MAX_LENGTH,))
    seq_features = Embedding(vocab_size, 256, mask_zero=True)(sequence_input)
    seq_features = LSTM(256)(seq_features)
    
    merged = Add()([image_features, seq_features])
    output = Dense(vocab_size, activation='softmax')(merged)
    
    model = Model(inputs=[image_input, sequence_input], outputs=output)
    
    # Initialize with random weights (this is just for demo)
    print("⚠️  Using demo model with random weights")
    print("⚠️  This will generate random captions - train the real model for actual results")
    
    return model, tokenizer

def extract_image_features(image, vgg_model):
    """Extract features from an image using VGG16."""
    # Resize and preprocess image
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    
    # Extract features
    features = vgg_model.predict(img_array, verbose=0)[0]
    return features

def generate_caption(model, tokenizer, features):
    """Generate a caption for the given image features."""
    # Start with start token
    in_text = '<start>'
    
    for _ in range(MAX_LENGTH):
        # Encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)
        
        # Predict next word
        try:
            yhat = model.predict([features.reshape(1, -1), sequence], verbose=0)
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
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            break
    
    # Remove start token and return
    caption = in_text.replace('<start>', '').strip()
    return caption if caption else "Unable to generate caption"

def create_demo_interface():
    """Create Gradio interface for the demo."""
    
    print("Loading models...")
    
    # Load VGG16 for feature extraction
    vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    vgg_model.trainable = False
    
    # Load caption model
    caption_model, tokenizer = load_demo_model()
    
    def caption_image(image):
        """Generate caption for uploaded image."""
        try:
            if image is None:
                return "Please upload an image", None
            
            # Extract features
            features = extract_image_features(image, vgg_model)
            
            # Generate caption
            caption = generate_caption(caption_model, tokenizer, features)
            
            return caption, image
            
        except Exception as e:
            return f"Error: {str(e)}", image
    
    # Create interface
    interface = gr.Interface(
        fn=caption_image,
        inputs=gr.Image(type='pil', label="Upload an Image"),
        outputs=[
            gr.Textbox(label="Generated Caption", lines=3),
            gr.Image(label="Input Image")
        ],
        title="🖼️ Image Captioning Demo",
        description="""
        Upload an image to generate a descriptive caption. 
        
        **Note**: If you haven't trained the model yet, this will use a demo model 
        that generates random captions. To get real results, run the full training 
        pipeline first with `python image_captioning.py`.
        """,
        examples=[
            # Add example images if available
        ],
        css="""
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .gr-button {
            background-color: #4CAF50;
            color: white;
        }
        """
    )
    
    return interface

def test_basic_functionality():
    """Test basic functionality without GUI."""
    print("🧪 Testing basic functionality...")
    
    try:
        # Test TensorFlow
        print(f"TensorFlow version: {tf.__version__}")
        
        # Test VGG16 loading
        vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        print("✓ VGG16 model loaded successfully")
        
        # Test model creation
        model, tokenizer = create_demo_model()
        print("✓ Demo model created successfully")
        
        # Create a test image (random noise)
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Extract features
        features = extract_image_features(test_image, vgg_model)
        print(f"✓ Feature extraction successful, shape: {features.shape}")
        
        # Generate caption
        caption = generate_caption(model, tokenizer, features)
        print(f"✓ Caption generation successful: '{caption}'")
        
        print("\n✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main demo function."""
    print("🎮 Image Captioning Demo")
    print("=" * 30)
    
    # Check if we should run tests or demo
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_basic_functionality()
        return
    
    try:
        # Create and launch demo
        demo = create_demo_interface()
        print("\n🚀 Launching demo interface...")
        print("📱 The interface will open in your browser")
        print("🛑 Press Ctrl+C to stop the demo")
        
        demo.launch(
            share=True,  # Creates public link
            server_name="0.0.0.0",  # Allows external connections
            server_port=7860,
            show_error=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        print("\nTrying basic test instead...")
        test_basic_functionality()

if __name__ == '__main__':
    main()
