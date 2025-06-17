"""
Data preprocessing module for customer support ticket classification.
Handles text cleaning, tokenization, and vectorization.
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Tuple, Dict, Any

import nltk
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TextPreprocessor:
    """Text preprocessing pipeline for customer support tickets."""
    
    def __init__(self, max_features: int = 10000, max_len: int = 100):
        """
        Initialize the text preprocessor.
        
        Args:
            max_features: Maximum number of features for tokenization
            max_len: Maximum sequence length for padding
        """
        self.max_features = max_features
        self.max_len = max_len
        self.tokenizer = None
        self.label_encoder = None
        self.tfidf_vectorizer = None
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        # Initialize spaCy
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("spaCy English model not found. Please install it using:")
            print("python -m spacy download en_core_web_sm")
            self.nlp = None
            
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text: Text to process
            
        Returns:
            Text without stopwords
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize text using spaCy.
        
        Args:
            text: Text to lemmatize
            
        Returns:
            Lemmatized text
        """
        if self.nlp is None:
            return text
            
        doc = self.nlp(text)
        lemmatized = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        return ' '.join(lemmatized)
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True, 
                       lemmatize: bool = True) -> str:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text: Text to preprocess
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize
            
        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Remove stopwords
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        # Lemmatize
        if lemmatize:
            text = self.lemmatize_text(text)
            
        return text
    
    def prepare_sequences(self, texts: List[str], fit_tokenizer: bool = True) -> np.ndarray:
        """
        Convert texts to padded sequences for neural networks.
        
        Args:
            texts: List of texts to convert
            fit_tokenizer: Whether to fit the tokenizer
            
        Returns:
            Padded sequences
        """
        if fit_tokenizer or self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.max_features, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        return padded
    
    def prepare_tfidf(self, texts: List[str], fit_vectorizer: bool = True) -> np.ndarray:
        """
        Convert texts to TF-IDF vectors.
        
        Args:
            texts: List of texts to convert
            fit_vectorizer: Whether to fit the vectorizer
            
        Returns:
            TF-IDF matrix
        """
        if fit_vectorizer or self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            
        return tfidf_matrix.toarray()
    
    def encode_labels(self, labels: List[str], fit_encoder: bool = True) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Encode categorical labels to integers.
        
        Args:
            labels: List of category labels
            fit_encoder: Whether to fit the encoder
            
        Returns:
            Encoded labels and label mapping
        """
        if fit_encoder or self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            encoded = self.label_encoder.fit_transform(labels)
        else:
            encoded = self.label_encoder.transform(labels)
        
        label_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        return encoded, label_mapping
    
    def decode_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """
        Decode integer labels back to categories.
        
        Args:
            encoded_labels: Encoded labels
            
        Returns:
            Decoded category labels
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted. Call encode_labels first.")
        
        return self.label_encoder.inverse_transform(encoded_labels)

class DataLoader:
    """Data loading and preparation for ticket classification."""
    
    def __init__(self, preprocessor: TextPreprocessor = None):
        """
        Initialize data loader.
        
        Args:
            preprocessor: Text preprocessor instance
        """
        self.preprocessor = preprocessor or TextPreprocessor()
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load ticket data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with ticket data
        """
        try:
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = ['customer_message', 'category']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Remove rows with missing essential data
            df = df.dropna(subset=required_columns)
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path}")
        except pd.errors.EmptyDataError:
            raise ValueError("Data file is empty")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def prepare_data(self, df: pd.DataFrame, text_column: str = 'customer_message',
                    label_column: str = 'category', test_size: float = 0.2,
                    val_size: float = 0.1, vectorization: str = 'sequences') -> Dict[str, Any]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with ticket data
            text_column: Name of text column
            label_column: Name of label column
            test_size: Proportion of test data
            val_size: Proportion of validation data
            vectorization: Type of vectorization ('sequences' or 'tfidf')
            
        Returns:
            Dictionary containing prepared data splits
        """
        # Preprocess text
        print("Preprocessing text data...")
        df['processed_text'] = df[text_column].apply(self.preprocessor.preprocess_text)
        
        # Remove empty texts after preprocessing
        df = df[df['processed_text'].str.len() > 0]
        
        # Prepare features and labels
        texts = df['processed_text'].tolist()
        labels = df[label_column].tolist()
        
        # Encode labels
        encoded_labels, label_mapping = self.preprocessor.encode_labels(labels)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, encoded_labels, test_size=test_size, random_state=42, stratify=encoded_labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        # Vectorize text
        print(f"Vectorizing text using {vectorization}...")
        if vectorization == 'sequences':
            X_train_vec = self.preprocessor.prepare_sequences(X_train, fit_tokenizer=True)
            X_val_vec = self.preprocessor.prepare_sequences(X_val, fit_tokenizer=False)
            X_test_vec = self.preprocessor.prepare_sequences(X_test, fit_tokenizer=False)
        elif vectorization == 'tfidf':
            X_train_vec = self.preprocessor.prepare_tfidf(X_train, fit_vectorizer=True)
            X_val_vec = self.preprocessor.prepare_tfidf(X_val, fit_vectorizer=False)
            X_test_vec = self.preprocessor.prepare_tfidf(X_test, fit_vectorizer=False)
        else:
            raise ValueError("vectorization must be 'sequences' or 'tfidf'")
        
        return {
            'X_train': X_train_vec,
            'X_val': X_val_vec,
            'X_test': X_test_vec,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'label_mapping': label_mapping,
            'num_classes': len(label_mapping),
            'vocab_size': self.preprocessor.max_features if vectorization == 'sequences' else X_train_vec.shape[1],
            'max_length': self.preprocessor.max_len if vectorization == 'sequences' else None
        }
    
    def save_preprocessor(self, file_path: str):
        """
        Save preprocessor configuration.
        
        Args:
            file_path: Path to save configuration
        """
        import pickle
        
        config = {
            'tokenizer': self.preprocessor.tokenizer,
            'label_encoder': self.preprocessor.label_encoder,
            'tfidf_vectorizer': self.preprocessor.tfidf_vectorizer,
            'max_features': self.preprocessor.max_features,
            'max_len': self.preprocessor.max_len
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(config, f)
    
    def load_preprocessor(self, file_path: str):
        """
        Load preprocessor configuration.
        
        Args:
            file_path: Path to load configuration from
        """
        import pickle
        
        with open(file_path, 'rb') as f:
            config = pickle.load(f)
        
        self.preprocessor.tokenizer = config['tokenizer']
        self.preprocessor.label_encoder = config['label_encoder']
        self.preprocessor.tfidf_vectorizer = config['tfidf_vectorizer']
        self.preprocessor.max_features = config['max_features']
        self.preprocessor.max_len = config['max_len']

if __name__ == "__main__":
    # Example usage
    data_loader = DataLoader()
    
    # Load sample data
    df = data_loader.load_data("../data/sample/sample_tickets.csv")
    print(f"Loaded {len(df)} tickets")
    
    # Prepare data
    data_splits = data_loader.prepare_data(df, vectorization='sequences')
    
    print(f"Training samples: {len(data_splits['X_train'])}")
    print(f"Validation samples: {len(data_splits['X_val'])}")
    print(f"Test samples: {len(data_splits['X_test'])}")
    print(f"Number of classes: {data_splits['num_classes']}")
    print(f"Label mapping: {data_splits['label_mapping']}")
