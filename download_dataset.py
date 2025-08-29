#!/usr/bin/env python3
"""
Alternative dataset download script for Flickr8k

This script provides multiple options to get the dataset:
1. Download from Kaggle (requires API key)
2. Manual download instructions
3. Use a sample dataset for testing
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import sys

def download_with_kaggle():
    """Try to download using Kaggle API."""
    try:
        import kaggle
        
        print("📥 Downloading Flickr8k from Kaggle...")
        kaggle.api.dataset_download_files(
            'adityajn105/flickr8k',
            path='data/',
            unzip=True
        )
        print("✅ Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Kaggle download failed: {e}")
        return False

def setup_kaggle_credentials():
    """Help user set up Kaggle credentials."""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_file = kaggle_dir / 'kaggle.json'
    
    print("\n🔑 Kaggle API Setup Required")
    print("=" * 40)
    print("To download from Kaggle, you need API credentials:")
    print()
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. Download the kaggle.json file")
    print("5. Run the following commands:")
    print()
    print(f"   mkdir -p {kaggle_dir}")
    print(f"   cp ~/Downloads/kaggle.json {kaggle_file}")
    print(f"   chmod 600 {kaggle_file}")
    print()
    print("Then run this script again!")
    
    return str(kaggle_file)

def create_sample_dataset():
    """Create a minimal sample dataset for testing."""
    print("📝 Creating sample dataset for testing...")
    
    data_dir = Path('data/Flickr8k')
    images_dir = data_dir / 'Images'
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample captions file
    captions_content = """image,caption
sample1.jpg,A dog is running on the beach
sample1.jpg,A brown dog plays near the ocean
sample1.jpg,A pet runs along the sandy shore
sample1.jpg,An animal enjoys time at the beach
sample1.jpg,A canine is active by the water
sample2.jpg,A cat is sitting on a chair
sample2.jpg,A feline rests on furniture
sample2.jpg,A pet sits comfortably indoors
sample2.jpg,An animal relaxes on a seat
sample2.jpg,A cat enjoys a quiet moment
sample3.jpg,A person is walking in the park
sample3.jpg,Someone strolls through green space
sample3.jpg,A human enjoys outdoor exercise
sample3.jpg,An individual walks among trees
sample3.jpg,A person moves through the park
"""
    
    captions_file = data_dir / 'captions.txt'
    with open(captions_file, 'w') as f:
        f.write(captions_content)
    
    # Create dummy image files (small colored squares)
    try:
        from PIL import Image
        import numpy as np
        
        colors = [
            (255, 200, 100),  # Orange (dog/beach)
            (150, 150, 150),  # Gray (cat/chair)
            (100, 200, 100),  # Green (person/park)
        ]
        
        for i, color in enumerate(colors, 1):
            # Create a 224x224 colored image
            img_array = np.full((224, 224, 3), color, dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = images_dir / f'sample{i}.jpg'
            img.save(img_path)
            print(f"  ✅ Created {img_path}")
        
        print(f"✅ Sample dataset created with {len(colors)} images")
        print(f"📁 Images: {images_dir}")
        print(f"📝 Captions: {captions_file}")
        
        return True
        
    except ImportError:
        print("❌ PIL not available for creating sample images")
        return False

def download_alternative_dataset():
    """Try to download from alternative sources."""
    print("🔍 Looking for alternative download sources...")
    
    # You could add other sources here if available
    print("⚠️  No alternative sources configured yet.")
    print("   Consider using the sample dataset for testing.")
    
    return False

def check_existing_dataset():
    """Check if dataset already exists."""
    data_dir = Path('data/Flickr8k')
    images_dir = data_dir / 'Images'
    captions_file = data_dir / 'captions.txt'
    
    # Alternative caption files
    alt_caption_files = [
        data_dir / 'captions.txt',
        data_dir / 'Flickr8k.token.txt',
        data_dir / 'Flickr_8k.trainImages.txt'
    ]
    
    if images_dir.exists():
        image_count = len(list(images_dir.glob('*.jpg')))
        print(f"📸 Found {image_count} images in {images_dir}")
        
        # Check for caption files
        caption_found = None
        for caption_file in alt_caption_files:
            if caption_file.exists():
                caption_found = caption_file
                break
        
        if caption_found:
            print(f"📝 Found captions: {caption_found}")
            print("✅ Dataset appears to be already available!")
            return True
    
    return False

def main():
    """Main function to handle dataset download."""
    print("📊 Flickr8k Dataset Setup")
    print("=" * 30)
    
    # Check if dataset already exists
    if check_existing_dataset():
        print("\n🎉 Dataset is ready! You can now run:")
        print("   python3 image_captioning.py")
        return
    
    print("\n📥 Dataset not found. Choose an option:")
    print("1. Download from Kaggle (requires API key)")
    print("2. Set up Kaggle credentials")
    print("3. Create sample dataset for testing")
    print("4. Manual download instructions")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                if download_with_kaggle():
                    print("\n🎉 Success! Dataset downloaded.")
                    break
                else:
                    print("\n❌ Kaggle download failed. Try option 2 to setup credentials.")
                    
            elif choice == '2':
                kaggle_file = setup_kaggle_credentials()
                print(f"\n💡 After setting up credentials, try option 1 again.")
                
            elif choice == '3':
                if create_sample_dataset():
                    print("\n🎉 Sample dataset created! You can test with:")
                    print("   python3 image_captioning.py")
                    break
                    
            elif choice == '4':
                print("\n📋 Manual Download Instructions:")
                print("1. Go to: https://www.kaggle.com/datasets/adityajn105/flickr8k")
                print("2. Click 'Download' (you may need to create a Kaggle account)")
                print("3. Extract the zip file")
                print("4. Copy contents to: data/Flickr8k/")
                print("   Expected structure:")
                print("   data/Flickr8k/")
                print("   ├── Images/")
                print("   │   ├── *.jpg files")
                print("   └── captions.txt (or Flickr8k.token.txt)")
                
            elif choice == '5':
                print("👋 Exiting...")
                sys.exit(0)
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()
