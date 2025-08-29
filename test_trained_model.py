#!/usr/bin/env python3
"""
Test the trained model with sample images
"""

import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# Configuration
MAX_LENGTH = 35

def load_components():
    """Load the trained components."""
    print("🔧 Loading trained components...")
    
    # Load tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print("✅ Tokenizer loaded")
    
    # Load features
    features_dict = np.load('features.npy', allow_pickle=True).item()
    print("✅ Features loaded")
    
    # Load VGG16 for new images
    vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    vgg_model.trainable = False
    print("✅ VGG16 loaded")
    
    # Load the trained model (with custom object scope for masking)
    try:
        from tensorflow.keras.layers import NotEqual
        custom_objects = {'NotEqual': NotEqual}
        model = tf.keras.models.load_model('models/caption_model.h5', custom_objects=custom_objects)
        print("✅ Caption model loaded")
    except Exception as e:
        print(f"⚠️  Model loading issue: {e}")
        print("🔧 Creating model architecture manually...")
        model = create_model_architecture(len(tokenizer.word_index) + 1)
        try:
            model.load_weights('models/caption_model.h5')
            print("✅ Model weights loaded successfully")
        except:
            print("❌ Could not load model weights")
            return None, None, None, None
    
    return model, tokenizer, features_dict, vgg_model

def create_model_architecture(vocab_size):
    """Recreate the model architecture without problematic layers."""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Add, Dropout
    
    # Encoder - Image features
    image_input = Input(shape=(512,), name='image_input')
    image_features = Dense(256, activation='relu')(image_input)
    image_features = Dropout(0.5)(image_features)
    
    # Decoder - Text sequences (without mask_zero to avoid NotEqual layer)
    sequence_input = Input(shape=(MAX_LENGTH,), name='sequence_input')
    seq_features = Embedding(vocab_size, 256, mask_zero=False)(sequence_input)
    seq_features = Dropout(0.5)(seq_features)
    seq_features = LSTM(256)(seq_features)
    
    # Merge encoder and decoder
    merged = Add()([image_features, seq_features])
    merged = Dense(256, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    output = Dense(vocab_size, activation='softmax')(merged)
    
    # Create model
    model = Model(inputs=[image_input, sequence_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def generate_caption(model, tokenizer, features):
    """Generate caption for image features."""
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

def test_with_sample_images():
    """Test the model with our sample images."""
    model, tokenizer, features_dict, vgg_model = load_components()
    
    if model is None:
        print("❌ Could not load model")
        return
    
    print("\n🧪 Testing caption generation...")
    
    # Test with sample images
    sample_images = ['sample1.jpg', 'sample2.jpg', 'sample3.jpg']
    
    for img_name in sample_images:
        if img_name in features_dict:
            print(f"\n🖼️  Testing {img_name}:")
            
            # Generate caption
            features = features_dict[img_name]
            caption = generate_caption(model, tokenizer, features)
            
            print(f"Generated caption: '{caption}'")
            
            # Show expected vs generated
            expected_captions = {
                'sample1.jpg': 'A dog is running on the beach',
                'sample2.jpg': 'A cat is sitting on a chair', 
                'sample3.jpg': 'A person is walking in the park'
            }
            
            expected = expected_captions.get(img_name, 'Unknown')
            print(f"Expected: '{expected}'")
            
            # Simple similarity check
            expected_words = set(expected.lower().split())
            generated_words = set(caption.lower().split())
            common_words = expected_words.intersection(generated_words)
            
            if common_words:
                print(f"✅ Common words: {common_words}")
            else:
                print("⚠️  No common words found")

def test_with_new_image():
    """Test with a new image (if available)."""
    model, tokenizer, features_dict, vgg_model = load_components()
    
    if model is None:
        return
    
    # Test with one of our sample images by loading it fresh
    img_path = 'data/Flickr8k/Images/sample1.jpg'
    
    try:
        print(f"\n🆕 Testing with fresh image: {img_path}")
        
        # Load and preprocess image
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        
        # Extract features
        features = vgg_model.predict(img_array, verbose=0)[0]
        
        # Generate caption
        caption = generate_caption(model, tokenizer, features)
        
        print(f"Generated caption: '{caption}'")
        
        # Display image and caption
        plt.figure(figsize=(8, 6))
        img_display = Image.open(img_path)
        plt.imshow(img_display)
        plt.title(f"Generated: {caption}")
        plt.axis('off')
        plt.savefig('test_caption_result.png', dpi=100, bbox_inches='tight')
        print("📊 Result saved as 'test_caption_result.png'")
        
    except Exception as e:
        print(f"❌ Error testing new image: {e}")

def main():
    """Main test function."""
    print("🧪 Testing Trained Image Captioning Model")
    print("=" * 45)
    
    # Test with sample images
    test_with_sample_images()
    
    # Test with fresh image processing
    test_with_new_image()
    
    print("\n🎉 Testing complete!")
    print("\n💡 To get better results:")
    print("1. Download the full Flickr8k dataset")
    print("2. Train on more images and epochs")
    print("3. The current model was trained on only 3 sample images")

if __name__ == '__main__':
    main()
