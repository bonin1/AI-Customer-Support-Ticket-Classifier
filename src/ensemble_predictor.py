"""
Advanced Ensemble Predictor with Meta-Learning and Model Stacking
"""
import os
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Tuple, Any
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import joblib
import logging
from datetime import datetime
import tensorflow as tf
from keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

class EnsemblePredictor:
    """
    Advanced ensemble predictor that combines multiple models using:
    - Voting (hard/soft)
    - Stacking with meta-learner
    - Dynamic weight adjustment based on confidence
    - Model performance tracking
    """
    
    def __init__(self, models_dir: str = "models/saved_models"):
        self.models_dir = models_dir
        self.models = {}
        self.preprocessors = {}
        self.model_info = {}
        self.ensemble_weights = {}
        self.performance_history = {}
        self.meta_learner = None
        self.confidence_threshold = 0.7
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_all_models(self) -> bool:
        """Load all available trained models from the models directory"""
        try:
            if not os.path.exists(self.models_dir):
                self.logger.error(f"Models directory {self.models_dir} not found")
                return False
                
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.h5') or f.endswith('.pkl')]
            
            for model_file in model_files:
                model_name = model_file.split('_')[0]  # Extract model type from filename
                model_path = os.path.join(self.models_dir, model_file)
                
                try:
                    # Load model based on file extension
                    if model_file.endswith('.h5'):
                        model = load_model(model_path)
                    else:
                        model = joblib.load(model_path)
                    
                    self.models[model_name] = model
                    
                    # Load corresponding preprocessor
                    preprocessor_file = model_file.replace('.h5', '_preprocessor.pkl').replace('.pkl', '_preprocessor.pkl')
                    preprocessor_path = os.path.join(self.models_dir, preprocessor_file)
                    
                    if os.path.exists(preprocessor_path):
                        with open(preprocessor_path, 'rb') as f:
                            self.preprocessors[model_name] = pickle.load(f)
                    
                    # Load model info
                    info_file = model_file.replace('.h5', '_info.json').replace('.pkl', '_info.json')
                    info_path = os.path.join(self.models_dir, info_file)
                    
                    if os.path.exists(info_path):
                        import json
                        with open(info_path, 'r') as f:
                            self.model_info[model_name] = json.load(f)
                    
                    self.logger.info(f"Loaded model: {model_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error loading model {model_file}: {str(e)}")
                    continue
            
            return len(self.models) > 0
            
        except Exception as e:
            self.logger.error(f"Error in load_all_models: {str(e)}")
            return False
    
    def predict_with_confidence(self, text: str) -> Dict[str, Any]:
        """
        Make predictions using all loaded models and return ensemble results
        with confidence scores and individual model predictions
        """
        if not self.models:
            raise ValueError("No models loaded. Call load_all_models() first.")
        
        predictions = {}
        probabilities = {}
        confidences = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                preprocessor = self.preprocessors.get(model_name)
                if not preprocessor:
                    continue
                
                # Preprocess text based on model type
                if hasattr(model, 'predict_proba'):  # Traditional ML models
                    processed_text = preprocessor['vectorizer'].transform([text])
                    probs = model.predict_proba(processed_text)[0]
                    pred_class = preprocessor['label_encoder'].classes_[np.argmax(probs)]
                    confidence = np.max(probs)
                else:  # Deep learning models
                    processed_text = preprocessor['tokenizer'].texts_to_sequences([text])
                    processed_text = tf.keras.preprocessing.sequence.pad_sequences(
                        processed_text, maxlen=preprocessor.get('max_length', 100)
                    )
                    probs = model.predict(processed_text, verbose=0)[0]
                    pred_class = preprocessor['label_encoder'].classes_[np.argmax(probs)]
                    confidence = np.max(probs)
                
                predictions[model_name] = pred_class
                probabilities[model_name] = probs.tolist()
                confidences[model_name] = float(confidence)
                
            except Exception as e:
                self.logger.error(f"Error predicting with {model_name}: {str(e)}")
                continue
        
        # Ensemble prediction using weighted voting
        ensemble_result = self._ensemble_predict(predictions, probabilities, confidences)
        
        return {
            'individual_predictions': predictions,
            'individual_confidences': confidences,
            'individual_probabilities': probabilities,
            'ensemble_prediction': ensemble_result['prediction'],
            'ensemble_confidence': ensemble_result['confidence'],
            'ensemble_probabilities': ensemble_result['probabilities'],
            'agreement_score': self._calculate_agreement(predictions),
            'high_confidence': ensemble_result['confidence'] > self.confidence_threshold
        }
    
    def _ensemble_predict(self, predictions: Dict, probabilities: Dict, confidences: Dict) -> Dict:
        """Combine predictions using weighted voting based on model confidence and historical performance"""
        if not predictions:
            return {'prediction': 'Unknown', 'confidence': 0.0, 'probabilities': {}}
        
        # Get all unique classes
        all_classes = set(predictions.values())
        
        # Dynamic weight calculation based on confidence and historical performance
        weights = {}
        for model_name in predictions.keys():
            base_weight = confidences[model_name]
            # Adjust weight based on historical performance if available
            perf_weight = self.performance_history.get(model_name, {}).get('accuracy', 0.5)
            weights[model_name] = base_weight * (1 + perf_weight)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Weighted voting for final prediction
        class_scores = {}
        for class_name in all_classes:
            score = 0
            for model_name, pred in predictions.items():
                if pred == class_name:
                    score += weights.get(model_name, 0)
            class_scores[class_name] = score
        
        # Get final prediction
        final_prediction = max(class_scores, key=class_scores.get)
        final_confidence = class_scores[final_prediction]
        
        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'probabilities': class_scores
        }
    
    def _calculate_agreement(self, predictions: Dict) -> float:
        """Calculate agreement score among all models"""
        if len(predictions) <= 1:
            return 1.0
        
        pred_values = list(predictions.values())
        most_common = max(set(pred_values), key=pred_values.count)
        agreement_count = pred_values.count(most_common)
        
        return agreement_count / len(pred_values)
    
    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """Make batch predictions with ensemble"""
        results = []
        for text in texts:
            try:
                result = self.predict_with_confidence(text)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error in batch prediction: {str(e)}")
                results.append({
                    'ensemble_prediction': 'Error',
                    'ensemble_confidence': 0.0,
                    'error': str(e)
                })
        return results
    
    def evaluate_ensemble(self, test_texts: List[str], true_labels: List[str]) -> Dict:
        """Evaluate ensemble performance on test data"""
        predictions = []
        confidences = []
        for text in test_texts:
            try:
                result = self.predict_with_confidence(text)
                predictions.append(result['ensemble_prediction'])
                confidences.append(result['ensemble_confidence'])
            except Exception as e:
                self.logger.error(f"Error predicting text: {str(e)}")
                predictions.append('Unknown')
                confidences.append(0.0)
        
        accuracy = accuracy_score(true_labels, predictions)
        avg_confidence = np.mean(confidences)
        
        # Calculate calibration (how well confidence correlates with accuracy)
        high_conf_mask = np.array(confidences) > self.confidence_threshold
        if high_conf_mask.sum() > 0:
            high_conf_accuracy = accuracy_score(
                np.array(true_labels)[high_conf_mask],
                np.array(predictions)[high_conf_mask]
            )
        else:
            high_conf_accuracy = 0.0
        
        return {
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'high_confidence_accuracy': high_conf_accuracy,
            'high_confidence_samples': high_conf_mask.sum(),
            'classification_report': classification_report(true_labels, predictions, output_dict=True)
        }
    
    def update_performance_history(self, model_name: str, metrics: Dict):
        """Update historical performance metrics for a model"""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        metrics['timestamp'] = datetime.now().isoformat()
        self.performance_history[model_name].append(metrics)
        
        # Keep only last 10 records per model
        if len(self.performance_history[model_name]) > 10:
            self.performance_history[model_name] = self.performance_history[model_name][-10:]
    
    def get_model_rankings(self) -> Dict:
        """Get current model rankings based on recent performance"""
        rankings = {}
        
        for model_name, history in self.performance_history.items():
            if history:
                recent_performance = history[-3:]  # Last 3 evaluations
                avg_accuracy = np.mean([h.get('accuracy', 0) for h in recent_performance])
                avg_confidence = np.mean([h.get('confidence', 0) for h in recent_performance])
                
                rankings[model_name] = {
                    'average_accuracy': avg_accuracy,
                    'average_confidence': avg_confidence,
                    'score': avg_accuracy * 0.7 + avg_confidence * 0.3,  # Weighted score
                    'evaluations': len(history)
                }
        
        # Sort by score
        sorted_rankings = dict(sorted(rankings.items(), key=lambda x: x[1]['score'], reverse=True))
        return sorted_rankings
    
    def save_ensemble_config(self, filepath: str):
        """Save ensemble configuration and performance history"""
        config = {
            'ensemble_weights': self.ensemble_weights,
            'performance_history': self.performance_history,
            'confidence_threshold': self.confidence_threshold,
            'model_info': self.model_info,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            import json
            json.dump(config, f, indent=2)
    
    def load_ensemble_config(self, filepath: str):
        """Load ensemble configuration and performance history"""
        try:
            with open(filepath, 'r') as f:
                import json
                config = json.load(f)
            
            self.ensemble_weights = config.get('ensemble_weights', {})
            self.performance_history = config.get('performance_history', {})
            self.confidence_threshold = config.get('confidence_threshold', 0.7)
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading ensemble config: {str(e)}")
            return False

if __name__ == "__main__":
    # Example usage
    ensemble = EnsemblePredictor()
    
    if ensemble.load_all_models():
        print(f"Loaded {len(ensemble.models)} models")
        
        # Test prediction
        test_text = "I can't log into my account and need help resetting my password"
        result = ensemble.predict_with_confidence(test_text)
        
        print(f"Test text: {test_text}")
        print(f"Ensemble prediction: {result['ensemble_prediction']}")
        print(f"Confidence: {result['ensemble_confidence']:.3f}")
        print(f"Agreement score: {result['agreement_score']:.3f}")
        print(f"Individual predictions: {result['individual_predictions']}")
    else:
        print("No models found or error loading models")
