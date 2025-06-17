"""
Model building module for customer support ticket classification.
Contains different model architectures including LSTM, BERT, and traditional ML models.
"""

import numpy as np
from typing import Dict, Any, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

class LSTMModelBuilder:
    """LSTM-based model for text classification."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 lstm_units: int = 64, num_classes: int = 7, 
                 max_length: int = 100, dropout: float = 0.5):
        """
        Initialize LSTM model builder.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embedding layer
            lstm_units: Number of LSTM units
            num_classes: Number of output classes
            max_length: Maximum sequence length
            dropout: Dropout rate
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_classes = num_classes
        self.max_length = max_length
        self.dropout = dropout
        self.model = None
    
    def build_model(self) -> keras.Model:
        """
        Build LSTM model architecture.
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # Embedding layer
            layers.Embedding(input_dim=self.vocab_size, 
                           output_dim=self.embedding_dim,
                           input_length=self.max_length,
                           mask_zero=True),
            
            # LSTM layers
            layers.LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout),
            layers.LSTM(self.lstm_units // 2, dropout=self.dropout),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.dropout),
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout / 2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def build_bidirectional_model(self) -> keras.Model:
        """
        Build bidirectional LSTM model.
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # Embedding layer
            layers.Embedding(input_dim=self.vocab_size,
                           output_dim=self.embedding_dim,
                           input_length=self.max_length,
                           mask_zero=True),
            
            # Bidirectional LSTM layers
            layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout)),
            layers.Bidirectional(layers.LSTM(self.lstm_units // 2, dropout=self.dropout)),
            
            # Dense layers with batch normalization
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout / 2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model

class CNNModelBuilder:
    """CNN-based model for text classification."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 num_classes: int = 7, max_length: int = 100, 
                 dropout: float = 0.5):
        """
        Initialize CNN model builder.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embedding layer
            num_classes: Number of output classes
            max_length: Maximum sequence length
            dropout: Dropout rate
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.max_length = max_length
        self.dropout = dropout
        self.model = None
    
    def build_model(self) -> keras.Model:
        """
        Build CNN model architecture.
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # Embedding layer
            layers.Embedding(input_dim=self.vocab_size,
                           output_dim=self.embedding_dim,
                           input_length=self.max_length),
            
            # Conv1D layers with different filter sizes
            layers.Conv1D(128, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(32, 3, activation='relu'),
            layers.GlobalMaxPooling1D(),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.dropout),
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout / 2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model

class BERTModelBuilder:
    """BERT-based model for text classification."""
    
    def __init__(self, num_classes: int = 7, max_length: int = 128):
        """
        Initialize BERT model builder.
        
        Args:
            num_classes: Number of output classes
            max_length: Maximum sequence length
        """
        self.num_classes = num_classes
        self.max_length = max_length
        self.model = None
    
    def build_model(self) -> keras.Model:
        """
        Build BERT-based model.
        Note: This is a simplified version. For production, use transformers library.
        
        Returns:
            Compiled Keras model
        """
        try:
            from transformers import TFBertModel, BertTokenizer
            
            # Load pre-trained BERT
            bert_model = TFBertModel.from_pretrained('bert-base-uncased')
            
            # Input layers
            input_ids = layers.Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
            attention_mask = layers.Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
            
            # BERT layer
            bert_output = bert_model([input_ids, attention_mask])
            
            # Classification head
            pooled_output = bert_output.pooler_output
            x = layers.Dropout(0.3)(pooled_output)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            
            model = keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)
            
            model.compile(
                optimizer=optimizers.Adam(learning_rate=2e-5),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            return model
            
        except ImportError:
            print("Transformers library not available. Using alternative architecture.")
            return self._build_transformer_like_model()
    
    def _build_transformer_like_model(self) -> keras.Model:
        """
        Build a transformer-like model without external dependencies.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(self.max_length,))
        
        # Embedding
        x = layers.Embedding(input_dim=10000, output_dim=256, mask_zero=True)(inputs)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=8, key_dim=32, dropout=0.1
        )(x, x)
        
        # Add & Norm
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)
        
        # Feed forward
        ffn_output = layers.Dense(512, activation='relu')(x)
        ffn_output = layers.Dense(256)(ffn_output)
        
        # Add & Norm
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization()(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model

class TraditionalMLBuilder:
    """Traditional machine learning models for text classification."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize traditional ML model builder.
        
        Args:
            model_type: Type of model ('random_forest', 'svm', 'logistic_regression')
        """
        self.model_type = model_type
        self.model = None
    def build_model(self) -> Any:
        """
        Build traditional ML model.
        
        Returns:
            Scikit-learn model
        """
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        if self.model is None:
            self.build_model()
        
        self.model.fit(x_train, y_train)
        return self.model
    
    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        return self.model.predict(x_data)
    
    def predict_proba(self, x_data: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Prediction probabilities
        """
        return self.model.predict_proba(x_data)

class ModelTrainer:
    """Unified model trainer for different architectures."""
    
    def __init__(self, model_type: str = 'lstm'):
        """
        Initialize model trainer.
        
        Args:
            model_type: Type of model to train
        """
        self.model_type = model_type
        self.model = None
        self.history = None
    def train_deep_learning_model(self, model: keras.Model, x_train: np.ndarray, 
                                 y_train: np.ndarray, x_val: np.ndarray, 
                                 y_val: np.ndarray, epochs: int = 50,
                                 batch_size: int = 32) -> Dict[str, Any]:
        """
        Train deep learning model.
        
        Args:
            model: Keras model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
          # Train model
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.model = model
        self.history = history
        
        return history.history
    def evaluate_model(self, x_test: np.ndarray, y_test: np.ndarray,
                      label_mapping: Dict[int, str]) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            label_mapping: Mapping from indices to labels
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
          # Make predictions
        if hasattr(self.model, 'predict_proba'):
            # Traditional ML model
            y_pred_proba = self.model.predict_proba(x_test)
            y_pred = self.model.predict(x_test)
        else:
            # Deep learning model
            y_pred_proba = self.model.predict(x_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        target_names = [label_mapping[i] for i in range(len(label_mapping))]
        class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def save_model(self, filepath: str):
        """
        Save trained model.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if hasattr(self.model, 'save'):
            # Keras model
            self.model.save(filepath)
        else:
            # Scikit-learn model
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
    
    def load_model(self, filepath: str):
        try:
            # Try loading as Keras model
            self.model = keras.models.load_model(filepath)
        except (ValueError, OSError):
            # Try loading as pickle file
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)

def get_model_builder(model_type: str, **kwargs) -> Any:
    """
    Factory function to get model builder.
    
    Args:
        model_type: Type of model to build
        **kwargs: Additional arguments for model builder
        
    Returns:
        Model builder instance
    """
    if model_type == 'lstm':
        return LSTMModelBuilder(**kwargs)
    elif model_type == 'cnn':
        return CNNModelBuilder(**kwargs)
    elif model_type == 'bert':
        return BERTModelBuilder(**kwargs)
    elif model_type in ['random_forest', 'svm', 'logistic_regression']:
        return TraditionalMLBuilder(model_type=model_type)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Example usage
    print("Testing model builders...")
    
    # Test LSTM builder
    lstm_builder = LSTMModelBuilder(vocab_size=10000, num_classes=7)
    lstm_model = lstm_builder.build_model()
    print(f"LSTM model built with {lstm_model.count_params()} parameters")
    
    # Test CNN builder
    cnn_builder = CNNModelBuilder(vocab_size=10000, num_classes=7)
    cnn_model = cnn_builder.build_model()
    print(f"CNN model built with {cnn_model.count_params()} parameters")
    
    # Test traditional ML builder
    rf_builder = TraditionalMLBuilder(model_type='random_forest')
    rf_model = rf_builder.build_model()
    print(f"Random Forest model built: {type(rf_model)}")
    
    print("All model builders working correctly!")
