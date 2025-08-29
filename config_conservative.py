# Conservative Image Captioning Configuration
# Optimized for systems with limited memory

# Model Parameters
MAX_LENGTH = 25  # Reduced from 35
VOCAB_SIZE = 3000  # Reduced from 5000
EMBEDDING_DIM = 128  # Reduced from 256
LSTM_UNITS = 128  # Reduced from 256

# Training Parameters
EPOCHS = 3  # Reduced from 5
BATCH_SIZE = 16  # Reduced from 32
LEARNING_RATE = 0.001

# Data Parameters
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# Feature Extraction
FEATURE_DIM = 512
IMAGE_SIZE = (224, 224)
FEATURE_BATCH_SIZE = 16  # Very conservative

# Paths
MODEL_FILE = 'models/caption_model.h5'
TOKENIZER_FILE = 'tokenizer.pkl'
FEATURES_FILE = 'features.npy'

# Memory Management
SEQUENCE_BATCH_SIZE = 250  # For sequence creation
