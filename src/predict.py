"""
Prediction utilities for customer support ticket classification.
"""

import os
import sys
import pickle
import json
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_preprocessing import TextPreprocessor

class TicketClassifier:
    """Main class for ticket classification predictions."""
    
    def __init__(self, model_path: str, preprocessor_path: str, 
                 model_info_path: Optional[str] = None):
        """
        Initialize ticket classifier.
        
        Args:
            model_path: Path to saved model
            preprocessor_path: Path to saved preprocessor
            model_info_path: Path to model info JSON file
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model_info_path = model_info_path
        
        self.model = None
        self.preprocessor = None
        self.label_mapping = None
        self.model_info = None
        
        self._load_components()
    
    def _load_components(self):
        """Load model, preprocessor, and metadata."""
        try:
            # Load model
            if self.model_path.endswith('.h5') or self.model_path.endswith('.keras'):
                self.model = keras.models.load_model(self.model_path)
                self.model_type = 'deep_learning'
            else:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.model_type = 'traditional_ml'
            
            print(f"Model loaded from: {self.model_path}")
            
            # Load preprocessor
            self.preprocessor = TextPreprocessor()
            with open(self.preprocessor_path, 'rb') as f:
                config = pickle.load(f)
            
            self.preprocessor.tokenizer = config['tokenizer']
            self.preprocessor.label_encoder = config['label_encoder']
            self.preprocessor.tfidf_vectorizer = config.get('tfidf_vectorizer')
            self.preprocessor.max_features = config['max_features']
            self.preprocessor.max_len = config['max_len']
            
            print(f"Preprocessor loaded from: {self.preprocessor_path}")
            
            # Create label mapping
            if self.preprocessor.label_encoder:
                self.label_mapping = {
                    i: label for i, label in enumerate(self.preprocessor.label_encoder.classes_)
                }
            
            # Load model info if available
            if self.model_info_path and os.path.exists(self.model_info_path):
                with open(self.model_info_path, 'r') as f:
                    self.model_info = json.load(f)
                print(f"Model info loaded from: {self.model_info_path}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading model components: {str(e)}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        return self.preprocessor.preprocess_text(text)
    
    def predict_single(self, text: str, return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Predict category for a single ticket.
        
        Args:
            text: Ticket text
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Prediction results
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text.strip():
            return {
                'predicted_category': 'General Inquiry',
                'confidence': 0.0,
                'probabilities': {},
                'error': 'Empty text after preprocessing'
            }
        
        try:            # Vectorize text
            if self.model_type == 'deep_learning':
                # Use tokenizer for deep learning models
                sequences = self.preprocessor.tokenizer.texts_to_sequences([processed_text])
                x_input = pad_sequences(sequences, maxlen=self.preprocessor.max_len, 
                                      padding='post', truncating='post')
            else:
                # Use TF-IDF for traditional ML models
                x_input = self.preprocessor.tfidf_vectorizer.transform([processed_text]).toarray()
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                # Traditional ML model
                probabilities = self.model.predict_proba(x_input)[0]
                predicted_class = np.argmax(probabilities)
            else:
                # Deep learning model
                probabilities = self.model.predict(x_input, verbose=0)[0]
                predicted_class = np.argmax(probabilities)
            
            # Get predicted category
            predicted_category = self.label_mapping[predicted_class]
            confidence = float(probabilities[predicted_class])
            
            # Prepare results
            results = {
                'predicted_category': predicted_category,
                'confidence': confidence,
                'processed_text': processed_text
            }
            
            if return_probabilities:
                prob_dict = {
                    self.label_mapping[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }
                results['probabilities'] = prob_dict
            
            return results
            
        except Exception as e:
            return {
                'predicted_category': 'General Inquiry',
                'confidence': 0.0,
                'probabilities': {},
                'error': f'Prediction error: {str(e)}'
            }
    
    def predict_batch(self, texts: List[str], return_probabilities: bool = False) -> List[Dict[str, Any]]:
        """
        Predict categories for multiple tickets.
        
        Args:
            texts: List of ticket texts
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction results
        """
        results = []
        
        for text in texts:
            result = self.predict_single(text, return_probabilities)
            results.append(result)
        
        return results
    
    def predict_dataframe(self, df: pd.DataFrame, text_column: str = 'customer_message') -> pd.DataFrame:
        """
        Predict categories for a DataFrame of tickets.
        
        Args:
            df: DataFrame with ticket data
            text_column: Name of the text column
            
        Returns:
            DataFrame with predictions added
        """
        # Make predictions
        predictions = self.predict_batch(df[text_column].tolist(), return_probabilities=True)
        
        # Add predictions to dataframe
        df_result = df.copy()
        df_result['predicted_category'] = [pred['predicted_category'] for pred in predictions]
        df_result['confidence'] = [pred['confidence'] for pred in predictions]
        df_result['processed_text'] = [pred.get('processed_text', '') for pred in predictions]
        
        # Add probability columns
        if predictions and 'probabilities' in predictions[0]:
            for category in self.label_mapping.values():
                col_name = f'prob_{category.replace(" ", "_").replace("&", "and").lower()}'
                df_result[col_name] = [
                    pred['probabilities'].get(category, 0.0) for pred in predictions
                ]
        
        return df_result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Model information dictionary
        """
        info = {
            'model_path': self.model_path,
            'preprocessor_path': self.preprocessor_path,
            'model_type': self.model_type,
            'label_mapping': self.label_mapping,
            'num_classes': len(self.label_mapping) if self.label_mapping else 0
        }
        
        if self.model_info:
            info.update(self.model_info)
        
        return info
    
    def get_feature_importance(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get feature importance for a prediction (for traditional ML models).
        
        Args:
            text: Input text
            top_n: Number of top features to return
            
        Returns:
            List of (feature, importance) tuples
        """
        if self.model_type != 'traditional_ml' or not hasattr(self.model, 'feature_importances_'):
            return []
        
        # Get feature names from TF-IDF vectorizer
        if not self.preprocessor.tfidf_vectorizer:
            return []
        
        feature_names = self.preprocessor.tfidf_vectorizer.get_feature_names_out()
        importances = self.model.feature_importances_
        
        # Get top features
        top_indices = np.argsort(importances)[-top_n:][::-1]
        top_features = [(feature_names[i], importances[i]) for i in top_indices]
        
        return top_features

class ModelEvaluator:
    """Class for evaluating model performance."""
    
    def __init__(self, classifier: TicketClassifier):
        """
        Initialize evaluator.
        
        Args:
            classifier: Trained ticket classifier
        """
        self.classifier = classifier
    
    def evaluate_on_test_data(self, test_data_path: str, text_column: str = 'customer_message',
                             label_column: str = 'category') -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            test_data_path: Path to test data CSV
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Evaluation metrics
        """
        # Load test data
        df = pd.read_csv(test_data_path)
        
        # Make predictions
        predictions_df = self.classifier.predict_dataframe(df, text_column)
        
        # Calculate metrics
        y_true = df[label_column].tolist()
        y_pred = predictions_df['predicted_category'].tolist()
        
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for category in self.classifier.label_mapping.values():
            if category in class_report:
                per_class_metrics[category] = class_report[category]
        
        return {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'per_class_metrics': per_class_metrics,
            'predictions_df': predictions_df
        }
    
    def analyze_predictions(self, df: pd.DataFrame, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze prediction results.
        
        Args:
            df: DataFrame with predictions
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Analysis results
        """
        analysis = {}
        
        # Overall statistics
        analysis['total_predictions'] = len(df)
        analysis['avg_confidence'] = df['confidence'].mean()
        analysis['min_confidence'] = df['confidence'].min()
        analysis['max_confidence'] = df['confidence'].max()
        
        # High/low confidence predictions
        high_conf = df[df['confidence'] >= confidence_threshold]
        low_conf = df[df['confidence'] < confidence_threshold]
        
        analysis['high_confidence_count'] = len(high_conf)
        analysis['low_confidence_count'] = len(low_conf)
        analysis['high_confidence_percentage'] = len(high_conf) / len(df) * 100
        
        # Category distribution
        analysis['category_distribution'] = df['predicted_category'].value_counts().to_dict()
        
        # Confidence by category
        conf_by_category = df.groupby('predicted_category')['confidence'].agg(['mean', 'std']).to_dict()
        analysis['confidence_by_category'] = conf_by_category
        
        return analysis

def load_classifier_from_directory(model_dir: str, model_name: str) -> TicketClassifier:
    """
    Load classifier from a model directory.
    
    Args:
        model_dir: Directory containing model files
        model_name: Name of the model (without extension)
        
    Returns:
        Loaded classifier
    """
    # Find model file
    model_path = None
    for ext in ['.h5', '.keras', '.pkl']:
        candidate_path = os.path.join(model_dir, f"{model_name}{ext}")
        if os.path.exists(candidate_path):
            model_path = candidate_path
            break
    
    if not model_path:
        raise FileNotFoundError(f"Model file not found for {model_name}")
    
    # Find preprocessor file
    preprocessor_path = os.path.join(model_dir, f"{model_name}_preprocessor.pkl")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
    
    # Find model info file
    model_info_path = os.path.join(model_dir, f"{model_name}_info.json")
    
    return TicketClassifier(model_path, preprocessor_path, model_info_path)

if __name__ == "__main__":
    # Example usage
    print("Testing prediction utilities...")
    
    # This would work with a trained model
    # classifier = load_classifier_from_directory("../models/saved_models", "lstm_20241217_120000")
    # 
    # # Test single prediction
    # test_text = "My credit card was charged twice for the same order"
    # result = classifier.predict_single(test_text)
    # print(f"Prediction: {result}")
    
    print("Prediction utilities ready!")
