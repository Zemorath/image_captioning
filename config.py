# Image Captioning Configuration
# Modify these parameters as needed

# Model Parameters
MAX_LENGTH = 35
VOCAB_SIZE = 5000
EMBEDDING_DIM = 256
LSTM_UNITS = 256

# Training Parameters
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Data Parameters
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# Feature Extraction
FEATURE_DIM = 512
IMAGE_SIZE = (224, 224)
FEATURE_BATCH_SIZE = 64  # Batch size for feature extraction (reduce if OOM issues)

# Paths
MODEL_FILE = 'models/caption_model.h5'
TOKENIZER_FILE = 'tokenizer.pkl'
FEATURES_FILE = 'features.npy'
