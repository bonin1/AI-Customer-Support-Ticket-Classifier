"""
Configuration settings for the ticket classification system.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Model paths
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"

# Default model parameters
DEFAULT_MODEL_PARAMS = {
    'lstm': {
        'vocab_size': 10000,
        'embedding_dim': 128,
        'lstm_units': 64,
        'max_length': 100,
        'dropout': 0.5,
        'epochs': 50,
        'batch_size': 32
    },
    'cnn': {
        'vocab_size': 10000,
        'embedding_dim': 128,
        'max_length': 100,
        'dropout': 0.5,
        'epochs': 50,
        'batch_size': 32
    },
    'bert': {
        'max_length': 128,
        'epochs': 3,
        'batch_size': 16
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_leaf': 1,
        'max_features': 'sqrt'
    },
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale'
    },
    'logistic_regression': {
        'max_iter': 1000,
        'C': 1.0
    }
}

# Text preprocessing parameters
PREPROCESSING_PARAMS = {
    'max_features': 10000,
    'max_length': 100,
    'min_word_length': 2,
    'remove_stopwords': True,
    'lemmatize': True
}

# Category definitions
CATEGORIES = [
    'Billing',
    'Technical Issue',
    'Feature Request',
    'Account Management',
    'Product Information',
    'Refund & Return',
    'General Inquiry'
]

# Priority levels
PRIORITY_LEVELS = ['Low', 'Medium', 'High', 'Critical']

# Communication channels
CHANNELS = ['Email', 'Chat', 'Phone', 'Social Media', 'Web Form']

# Streamlit configuration
STREAMLIT_CONFIG = {
    'page_title': "AI Ticket Classifier",
    'page_icon': "🎫",
    'layout': "wide",
    'sidebar_state': "expanded"
}

# Model evaluation thresholds
CONFIDENCE_THRESHOLDS = {
    'high': 0.8,
    'medium': 0.5,
    'low': 0.3
}

# File size limits (in bytes)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

# Supported file formats
SUPPORTED_FORMATS = ['csv', 'xlsx', 'json']

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# API configuration (if needed)
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False
}

# Environment-specific settings
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

if ENVIRONMENT == 'production':
    DEBUG = False
    LOG_LEVEL = 'WARNING'
elif ENVIRONMENT == 'testing':
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
else:  # development
    DEBUG = True
    LOG_LEVEL = 'INFO'
