# Makefile for Image Captioning Project
# Simple commands to manage the project

.PHONY: help install test setup train demo clean lint format

# Default target
help:
	@echo "🖼️  Image Captioning for Accessibility Tools"
	@echo "============================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install all dependencies"
	@echo "  make setup       - Setup dataset and directories"
	@echo "  make test        - Run comprehensive tests"
	@echo "  make train       - Train the model (full pipeline)"
	@echo "  make demo        - Launch demo interface"
	@echo "  make clean       - Clean generated files"
	@echo "  make lint        - Run code linting"
	@echo "  make format      - Format code"
	@echo "  make quicktest   - Quick functionality test"
	@echo "  make gpu-test    - Test GPU availability"
	@echo ""
	@echo "Quick start:"
	@echo "  1. make install"
	@echo "  2. make setup"
	@echo "  3. make test"
	@echo "  4. make train"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

# Setup dataset and project structure
setup:
	@echo "🏗️  Setting up project..."
	python setup_dataset.py
	@echo "✅ Project setup complete!"

# Run comprehensive tests
test:
	@echo "🧪 Running comprehensive tests..."
	python test_system.py
	@echo "✅ Tests complete!"

# Quick test for basic functionality
quicktest:
	@echo "⚡ Running quick tests..."
	python demo.py test
	@echo "✅ Quick test complete!"

# Test GPU availability
gpu-test:
	@echo "🎮 Testing GPU availability..."
	python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

# Train the model (full pipeline)
train:
	@echo "🚀 Starting training pipeline..."
	python image_captioning.py
	@echo "✅ Training complete!"

# Launch demo interface
demo:
	@echo "🎮 Launching demo interface..."
	python demo.py
	@echo "👋 Demo stopped"

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -f features.npy
	rm -f tokenizer.pkl
	rm -f *.png
	rm -f *.jpg
	rm -rf models/caption_model.h5
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "✅ Cleanup complete!"

# Code linting (optional)
lint:
	@echo "🔍 Running code linting..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 *.py --max-line-length=100 --ignore=E501,W503; \
	else \
		echo "⚠️  flake8 not installed. Install with: pip install flake8"; \
	fi

# Code formatting (optional)
format:
	@echo "🎨 Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		black *.py --line-length=100; \
	else \
		echo "⚠️  black not installed. Install with: pip install black"; \
	fi

# Show project status
status:
	@echo "📊 Project Status"
	@echo "=================="
	@echo -n "📦 Dependencies: "
	@if pip list | grep -q tensorflow; then echo "✅"; else echo "❌"; fi
	@echo -n "📁 Dataset: "
	@if [ -d "data/Flickr8k/Images" ]; then echo "✅"; else echo "❌"; fi
	@echo -n "🤖 Model: "
	@if [ -f "models/caption_model.h5" ]; then echo "✅ Trained"; else echo "❌ Not trained"; fi
	@echo -n "🔧 Features: "
	@if [ -f "features.npy" ]; then echo "✅ Extracted"; else echo "❌ Not extracted"; fi

# Development setup
dev-setup: install
	@echo "🛠️  Setting up development environment..."
	pip install black flake8 pytest jupyter
	@echo "✅ Development setup complete!"

# Run Jupyter notebook (optional)
notebook:
	@echo "📓 Starting Jupyter notebook..."
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Download dataset using Kaggle (requires setup)
download-dataset:
	@echo "📥 Downloading dataset from Kaggle..."
	@if command -v kaggle >/dev/null 2>&1; then \
		kaggle datasets download -d adityajn105/flickr8k -p data/; \
		cd data && unzip -q flickr8k.zip; \
		echo "✅ Dataset downloaded!"; \
	else \
		echo "❌ Kaggle CLI not found. Install with: pip install kaggle"; \
		echo "Then setup your API credentials."; \
	fi

# Create virtual environment
venv:
	@echo "🐍 Creating virtual environment..."
	python -m venv venv
	@echo "✅ Virtual environment created!"
	@echo "Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"

# All-in-one setup for new users
full-setup: venv install setup test
	@echo "🎉 Full setup complete!"
	@echo "Ready to train your model with: make train"

# Check system requirements
check-requirements:
	@echo "🔍 Checking system requirements..."
	@echo -n "Python version: "; python --version
	@echo -n "Pip version: "; pip --version
	@echo -n "Available memory: "; free -h | grep "Mem:" | awk '{print $$2}' 2>/dev/null || echo "Unknown"
	@echo -n "Available disk space: "; df -h . | tail -1 | awk '{print $$4}' 2>/dev/null || echo "Unknown"
	@python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)" 2>/dev/null || echo "TensorFlow: Not installed"

# Show project information
info:
	@echo "ℹ️  Project Information"
	@echo "======================"
	@echo "Name: Image Captioning for Accessibility Tools"
	@echo "Description: Generate descriptive captions for images"
	@echo "Architecture: VGG16 + LSTM encoder-decoder"
	@echo "Dataset: Flickr8k (8,000 images with captions)"
	@echo "Framework: TensorFlow/Keras"
	@echo "Interface: Gradio web app"
	@echo ""
	@echo "Key features:"
	@echo "- Entry-level multimodal AI project"
	@echo "- Accessibility-focused design"
	@echo "- Complete end-to-end pipeline"
	@echo "- Web-based demo interface"
	@echo "- CPU and GPU support"
