#!/usr/bin/env python3
"""
Test Script for Image Captioning Project

This script verifies that all dependencies are installed correctly
and tests basic functionality before running the full pipeline.
"""

import sys
import importlib
import traceback
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("🧪 Testing package imports...")
    
    required_packages = [
        ('tensorflow', 'tf'),
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('PIL', 'PIL'),
        ('sklearn', 'sklearn'),
        ('nltk', 'nltk'),
        ('matplotlib.pyplot', 'plt'),
        ('gradio', 'gr')
    ]
    
    optional_packages = [
        ('pyttsx3', 'pyttsx3')  # Text-to-speech (optional)
    ]
    
    results = {}
    
    # Test required packages
    for package_name, import_name in required_packages:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            results[package_name] = {'status': 'success', 'version': version}
            print(f"  ✓ {package_name}: {version}")
        except ImportError as e:
            results[package_name] = {'status': 'error', 'error': str(e)}
            print(f"  ✗ {package_name}: {e}")
    
    # Test optional packages
    print("\n📦 Testing optional packages...")
    for package_name, import_name in optional_packages:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            results[package_name] = {'status': 'success', 'version': version}
            print(f"  ✓ {package_name}: {version} (optional)")
        except ImportError as e:
            results[package_name] = {'status': 'optional', 'error': str(e)}
            print(f"  ⚠️  {package_name}: Not installed (optional)")
    
    return results

def test_tensorflow():
    """Test TensorFlow functionality."""
    print("\n🤖 Testing TensorFlow...")
    
    try:
        import tensorflow as tf
        
        # Check TensorFlow version
        print(f"  ✓ TensorFlow version: {tf.__version__}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✓ GPU detected: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"    - GPU {i}: {gpu.name}")
        else:
            print("  ⚠️  No GPU detected, will use CPU")
        
        # Test basic tensor operations
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        print(f"  ✓ Basic tensor operations: {c.numpy()}")
        
        # Test Keras model creation
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])
        print("  ✓ Keras model creation successful")
        
        return True
        
    except Exception as e:
        print(f"  ✗ TensorFlow test failed: {e}")
        return False

def test_image_processing():
    """Test image processing functionality."""
    print("\n🖼️  Testing image processing...")
    
    try:
        import numpy as np
        from PIL import Image
        import tensorflow as tf
        from tensorflow.keras.applications import VGG16
        from tensorflow.keras.preprocessing.image import img_to_array
        
        # Create a test image
        test_image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_image_array)
        print("  ✓ Test image created")
        
        # Test image conversion
        img_array = img_to_array(test_image)
        print(f"  ✓ Image to array conversion: {img_array.shape}")
        
        # Test VGG16 loading (this might take a moment)
        print("  📥 Loading VGG16 model (this may take a moment)...")
        vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        print("  ✓ VGG16 model loaded successfully")
        
        # Test feature extraction
        img_batch = np.expand_dims(img_array, axis=0)
        img_batch = tf.keras.applications.vgg16.preprocess_input(img_batch)
        features = vgg_model.predict(img_batch, verbose=0)
        print(f"  ✓ Feature extraction: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Image processing test failed: {e}")
        traceback.print_exc()
        return False

def test_nlp_functionality():
    """Test natural language processing functionality."""
    print("\n📝 Testing NLP functionality...")
    
    try:
        import nltk
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        # Download required NLTK data
        print("  📥 Downloading NLTK data...")
        try:
            nltk.download('punkt', quiet=True)
            print("  ✓ NLTK data downloaded")
        except Exception as e:
            print(f"  ⚠️  NLTK download warning: {e}")
        
        # Test tokenization
        sample_texts = [
            "a dog is running on the beach",
            "a cat is sitting on a chair",
            "a person is walking in the park"
        ]
        
        tokenizer = Tokenizer(num_words=1000, oov_token='<unk>')
        tokenizer.fit_on_texts(sample_texts)
        print(f"  ✓ Tokenizer created, vocabulary size: {len(tokenizer.word_index)}")
        
        # Test sequence conversion
        sequences = tokenizer.texts_to_sequences(sample_texts)
        padded = pad_sequences(sequences, maxlen=10)
        print(f"  ✓ Sequence padding: {padded.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ NLP test failed: {e}")
        return False

def test_demo_functionality():
    """Test demo interface functionality."""
    print("\n🎮 Testing demo functionality...")
    
    try:
        import gradio as gr
        print(f"  ✓ Gradio version: {gr.__version__}")
        
        # Test simple interface creation (don't launch)
        def dummy_function(x):
            return f"Processed: {x}"
        
        interface = gr.Interface(
            fn=dummy_function,
            inputs=gr.Textbox(),
            outputs=gr.Textbox()
        )
        print("  ✓ Gradio interface creation successful")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Demo test failed: {e}")
        return False

def test_file_structure():
    """Test project file structure."""
    print("\n📁 Testing file structure...")
    
    project_root = Path.cwd()
    
    # Check for main files
    main_files = [
        'image_captioning.py',
        'demo.py',
        'config.py',
        'requirements.txt',
        'README.md'
    ]
    
    # Check for directories
    directories = [
        'data',
        'models',
        'utils'
    ]
    
    all_good = True
    
    for file_name in main_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name} (missing)")
            all_good = False
    
    for dir_name in directories:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/ (directory)")
        else:
            print(f"  ⚠️  {dir_name}/ (directory missing, will be created)")
    
    return all_good

def test_dataset_availability():
    """Test if dataset is available."""
    print("\n📊 Testing dataset availability...")
    
    from config import DATA_DIR, IMAGES_DIR, ALT_CAPTION_FILES
    
    if not DATA_DIR.exists():
        print(f"  ⚠️  Dataset directory not found: {DATA_DIR}")
        print("  💡 Run 'python setup_dataset.py' to download the dataset")
        return False
    
    if not IMAGES_DIR.exists():
        print(f"  ⚠️  Images directory not found: {IMAGES_DIR}")
        return False
    
    # Count images
    image_files = list(IMAGES_DIR.glob('*.jpg')) + list(IMAGES_DIR.glob('*.jpeg')) + list(IMAGES_DIR.glob('*.png'))
    print(f"  📸 Found {len(image_files)} images")
    
    # Check for caption files
    caption_file_found = False
    for caption_file in ALT_CAPTION_FILES:
        if caption_file.exists():
            print(f"  ✓ Caption file found: {caption_file.name}")
            caption_file_found = True
            break
    
    if not caption_file_found:
        print("  ⚠️  No caption file found")
        return False
    
    if len(image_files) > 0 and caption_file_found:
        print("  ✅ Dataset appears to be available")
        return True
    else:
        return False

def run_comprehensive_test():
    """Run all tests and provide a summary."""
    print("🧪 Running Comprehensive Test Suite")
    print("=" * 50)
    
    test_results = {}
    
    # Run all tests
    test_results['imports'] = test_imports()
    test_results['tensorflow'] = test_tensorflow()
    test_results['image_processing'] = test_image_processing()
    test_results['nlp'] = test_nlp_functionality()
    test_results['demo'] = test_demo_functionality()
    test_results['file_structure'] = test_file_structure()
    test_results['dataset'] = test_dataset_availability()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    
    passed_tests = 0
    total_tests = 0
    
    for test_name, result in test_results.items():
        total_tests += 1
        if isinstance(result, bool):
            if result:
                print(f"✅ {test_name.replace('_', ' ').title()}: PASSED")
                passed_tests += 1
            else:
                print(f"❌ {test_name.replace('_', ' ').title()}: FAILED")
        elif isinstance(result, dict):
            # For imports test
            success_count = sum(1 for r in result.values() if r['status'] == 'success')
            total_packages = len([r for r in result.values() if r['status'] in ['success', 'error']])
            if success_count == total_packages:
                print(f"✅ {test_name.replace('_', ' ').title()}: PASSED ({success_count}/{total_packages})")
                passed_tests += 1
            else:
                print(f"❌ {test_name.replace('_', ' ').title()}: FAILED ({success_count}/{total_packages})")
    
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! You're ready to run the image captioning system.")
        print("\nNext steps:")
        print("1. If dataset is not available: python setup_dataset.py")
        print("2. Run full pipeline: python image_captioning.py")
        print("3. Or start with demo: python demo.py")
    else:
        print("⚠️  Some tests failed. Please address the issues above.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Download dataset: python setup_dataset.py")
        print("- Check TensorFlow GPU setup if needed")

def main():
    """Main test function."""
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == 'imports':
            test_imports()
        elif test_type == 'tensorflow':
            test_tensorflow()
        elif test_type == 'image':
            test_image_processing()
        elif test_type == 'nlp':
            test_nlp_functionality()
        elif test_type == 'demo':
            test_demo_functionality()
        elif test_type == 'files':
            test_file_structure()
        elif test_type == 'dataset':
            test_dataset_availability()
        else:
            print(f"Unknown test type: {test_type}")
            print("Available tests: imports, tensorflow, image, nlp, demo, files, dataset")
    else:
        run_comprehensive_test()

if __name__ == '__main__':
    main()
