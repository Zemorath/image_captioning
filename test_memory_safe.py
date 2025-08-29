#!/usr/bin/env python3
"""
Memory-safe test script for image captioning with small dataset subset.
This script tests the pipeline with a small number of images to avoid OOM issues.
"""

import sys
import os
import shutil

def create_test_subset(source_dir, target_dir, num_images=50):
    """Create a small subset of images for testing."""
    print(f"Creating test subset with {num_images} images...")
    
    if not os.path.exists(source_dir):
        print(f"Source directory not found: {source_dir}")
        return False
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Get list of images
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) < num_images:
        num_images = len(image_files)
        print(f"Only {num_images} images available")
    
    # Copy subset of images
    selected_files = image_files[:num_images]
    
    for i, filename in enumerate(selected_files):
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        shutil.copy2(source_path, target_path)
        if (i + 1) % 10 == 0:
            print(f"Copied {i + 1}/{num_images} images")
    
    print(f"Test subset created in {target_dir}")
    return True

def create_test_captions(original_captions, test_images_dir, output_file):
    """Create captions file for test subset."""
    print("Creating test captions file...")
    
    # Get list of test images
    test_images = set(os.listdir(test_images_dir))
    
    # Read original captions
    with open(original_captions, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Write filtered captions
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header if present
        if lines and ('image' in lines[0].lower() or 'caption' in lines[0].lower()):
            f.write(lines[0])
            start_idx = 1
        else:
            start_idx = 0
        
        count = 0
        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
            
            # Extract image name
            if '\t' in line:
                image_name = line.split('\t')[0]
            elif ',' in line:
                image_name = line.split(',')[0]
            else:
                continue
            
            # Remove any suffix (like #0, #1, etc.)
            image_base = image_name.split('#')[0] if '#' in image_name else image_name
            
            if image_base in test_images:
                f.write(line + '\n')
                count += 1
        
        print(f"Created captions for {count} entries")
    
    return output_file

def main():
    """Main function to create test subset and run pipeline."""
    # Paths
    original_images = './data/Flickr8k/Images/'
    original_captions = './data/Flickr8k/captions.txt'
    test_images = './data/test_subset/Images/'
    test_captions = './data/test_subset/captions.txt'
    
    # Create test directories
    os.makedirs('./data/test_subset/', exist_ok=True)
    
    # Create subset
    if create_test_subset(original_images, test_images, num_images=100):
        create_test_captions(original_captions, test_images, test_captions)
        
        print("\nTest subset created successfully!")
        print(f"Test images: {test_images}")
        print(f"Test captions: {test_captions}")
        print("\nTo test the pipeline, temporarily modify image_captioning.py:")
        print("1. Change IMAGES_DIR to './data/test_subset/Images/'")
        print("2. Change ANNOTATIONS_FILE to './data/test_subset/captions.txt'")
        print("3. Run: python3 image_captioning.py")
        print("\nOr run with modified paths directly...")
        
        # Option to run the test directly
        response = input("\nRun test automatically? (y/n): ").lower().strip()
        if response == 'y':
            run_test_pipeline(test_images, test_captions)
    else:
        print("Failed to create test subset")

def run_test_pipeline(test_images_dir, test_captions_file):
    """Run the image captioning pipeline with test data."""
    print("Running test pipeline...")
    
    # Temporarily modify paths in the main script
    import subprocess
    import tempfile
    
    # Read the main script
    with open('image_captioning.py', 'r') as f:
        content = f.read()
    
    # Modify paths
    modified_content = content.replace(
        "IMAGES_DIR = os.path.join(DATA_DIR, 'Images/')",
        f"IMAGES_DIR = '{test_images_dir}'"
    ).replace(
        "ANNOTATIONS_FILE = os.path.join(DATA_DIR, 'captions.txt')",
        f"ANNOTATIONS_FILE = '{test_captions_file}'"
    )
    
    # Write temporary script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(modified_content)
        temp_script = f.name
    
    try:
        # Run the modified script
        result = subprocess.run([sys.executable, temp_script], 
                               capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("Process timed out after 30 minutes")
    except Exception as e:
        print(f"Error running test: {e}")
    finally:
        # Clean up temporary script
        if os.path.exists(temp_script):
            os.unlink(temp_script)

if __name__ == '__main__':
    main()
