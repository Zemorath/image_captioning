# Image Captioning for Accessibility Tools

A complete multimodal image captioning system designed to generate descriptive captions for images, helping visually impaired users understand visual content. This project uses deep learning techniques with TensorFlow/Keras to build an encoder-decoder architecture.

## 🎯 Project Goal

Build an entry-level image captioning model that:
- Generates descriptive captions like "A dog is running on the beach"
- Uses VGG16 for image feature extraction
- Employs LSTM decoder for text generation
- Provides a user-friendly web interface via Gradio
- Achieves reasonable BLEU scores for evaluation

## 🏗️ Architecture

- **Encoder**: VGG16 (pre-trained on ImageNet) for image feature extraction
- **Decoder**: LSTM-based text generator with attention-like mechanism
- **Dataset**: Flickr8k (8,000 images with 5 captions each)
- **Vocabulary**: Top 5,000 most frequent words
- **Max Caption Length**: 35 words

## 📋 Requirements

- Python 3.10+
- TensorFlow 2.15+
- 4GB+ RAM (8GB recommended)
- GPU support optional but recommended for training

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd image_captioning
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download the Flickr8k dataset from Kaggle:

```bash
# Install Kaggle CLI if not already installed
pip install kaggle

# Download dataset (requires Kaggle account and API key)
kaggle datasets download -d adityajn105/flickr8k

# Extract to the correct location
unzip flickr8k.zip -d data/Flickr8k/
```

Expected directory structure:
```
data/Flickr8k/
├── Images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── captions.txt (or Flickr8k.token.txt)
└── other metadata files
```

### 4. Run the Complete Pipeline

```bash
python image_captioning.py
```

This will:
1. Load and preprocess the dataset
2. Extract image features using VGG16
3. Build and train the caption model
4. Evaluate performance with BLEU scores
5. Launch a Gradio web interface for testing

## 📁 Project Structure

```
image_captioning/
├── image_captioning.py      # Main script with complete pipeline
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/
│   └── Flickr8k/          # Dataset directory
├── models/
│   └── caption_model.h5   # Trained model (generated)
├── features.npy           # Pre-extracted image features (generated)
├── tokenizer.pkl          # Saved tokenizer (generated)
├── training_history.png   # Training plots (generated)
└── sample_predictions.png # Sample results (generated)
```

## 🔧 Configuration

Key parameters can be modified at the top of `image_captioning.py`:

```python
MAX_LENGTH = 35        # Maximum caption length
VOCAB_SIZE = 5000      # Vocabulary size
FEATURE_DIM = 512      # VGG16 feature dimension
EMBEDDING_DIM = 256    # Word embedding dimension
LSTM_UNITS = 256       # LSTM hidden units
EPOCHS = 20            # Training epochs
BATCH_SIZE = 32        # Batch size
```

## 📊 Model Performance

The model is evaluated using BLEU scores:
- **BLEU-1**: Measures unigram precision
- **BLEU-2**: Measures bigram precision  
- **BLEU-3**: Measures trigram precision
- **BLEU-4**: Measures 4-gram precision

Target performance: BLEU-1 > 0.5 for basic functionality

## 🎮 Using the Demo

After training, the Gradio interface will launch automatically:

1. **Web Interface**: Opens in your browser
2. **Upload Image**: Drag & drop or click to upload
3. **Generate Caption**: Automatic caption generation
4. **Results**: View generated caption alongside the image

Example usage:
- Upload a photo of a beach scene
- Get caption: "a person walking on the beach near the ocean"

## 🔧 Advanced Features

### GPU Support
The system automatically detects and uses GPU if available:
```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {len(gpus)} device(s)")
```

### Feature Caching
Image features are cached to avoid recomputation:
- First run: Extracts and saves features to `features.npy`
- Subsequent runs: Loads pre-computed features

### Error Handling
Comprehensive error handling for:
- Missing dataset files
- Corrupted images
- Memory issues
- File I/O problems

## 🎯 Accessibility Features

This project is specifically designed for accessibility:

1. **Descriptive Captions**: Generate meaningful descriptions for screen readers
2. **Web Interface**: Easy-to-use interface for image upload
3. **Batch Processing**: Can process multiple images efficiently
4. **Confidence Scoring**: Model confidence can be extracted from prediction probabilities

### Optional Text-to-Speech
To add voice output, uncomment `pyttsx3` in requirements.txt and add:

```python
import pyttsx3

def speak_caption(caption):
    engine = pyttsx3.init()
    engine.say(caption)
    engine.runAndWait()
```

## 🐛 Troubleshooting

### Common Issues

1. **Dataset Not Found**
   ```
   Error: Download Flickr8k to ./data/Flickr8k/
   ```
   Solution: Download and extract dataset to correct location

2. **Memory Error**
   ```
   ResourceExhaustedError: OOM when allocating tensor
   ```
   Solution: Reduce batch size or use CPU-only mode

3. **Feature Extraction Slow**
   Solution: Features are cached after first run

4. **Poor Performance**
   - Increase training epochs
   - Ensure sufficient training data
   - Check caption quality in dataset

### Performance Optimization

- **Use GPU**: Significantly faster training
- **Reduce Vocabulary**: Lower memory usage
- **Batch Processing**: More efficient than single predictions
- **Feature Caching**: Avoid recomputing VGG16 features

## 📈 Extending the Project

### Possible Improvements

1. **Better Architecture**: 
   - Attention mechanisms
   - Transformer-based models
   - Beam search decoding

2. **Larger Datasets**:
   - MS-COCO (330K images)
   - Conceptual Captions (3M images)

3. **Advanced Features**:
   - Object detection integration
   - Sentiment analysis
   - Multi-language support

4. **Production Deployment**:
   - REST API with Flask/FastAPI
   - Docker containerization
   - Cloud deployment (AWS/GCP/Azure)

## 📝 Model Details

### Architecture Specifics

```python
# Encoder: Image Feature Processing
image_input = Input(shape=(512,))  # VGG16 features
image_features = Dense(256, activation='relu')(image_input)

# Decoder: Text Generation
sequence_input = Input(shape=(35,))  # Word sequences
seq_features = Embedding(5000, 256, mask_zero=True)(sequence_input)
seq_features = LSTM(256)(seq_features)

# Fusion and Output
merged = Add()([image_features, seq_features])
output = Dense(5000, activation='softmax')(merged)
```

### Training Process

1. **Data Preparation**: Load images and captions
2. **Feature Extraction**: VGG16 processing (one-time)
3. **Sequence Creation**: Convert captions to input-output pairs
4. **Model Training**: 20 epochs with early stopping
5. **Evaluation**: BLEU score computation
6. **Demo Launch**: Gradio interface

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Flickr8k Dataset**: University of Illinois
- **VGG16 Model**: Visual Geometry Group, Oxford
- **TensorFlow/Keras**: Google
- **Gradio**: Hugging Face

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review error messages carefully
3. Ensure dataset is properly downloaded
4. Verify all dependencies are installed

For additional help, please open an issue in the repository.

---

**Built for accessibility, powered by AI** 🤖♿