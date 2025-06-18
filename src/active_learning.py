"""
Active Learning System with Uncertainty Quantification and Human-in-the-Loop
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging
from datetime import datetime
import json
import os
from scipy.stats import entropy
from collections import defaultdict
import pickle

class ActiveLearningOracle:
    """
    Advanced active learning system that:
    - Identifies uncertain predictions for human review
    - Suggests most informative samples for labeling
    - Tracks model improvement over time
    - Implements various uncertainty sampling strategies
    """
    
    def __init__(self, uncertainty_threshold: float = 0.8, min_samples_for_retrain: int = 50):
        self.uncertainty_threshold = uncertainty_threshold
        self.min_samples_for_retrain = min_samples_for_retrain
        self.feedback_data = []
        self.uncertainty_history = []
        self.model_performance_tracker = []
        self.human_corrections = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def calculate_uncertainty_metrics(self, prediction_result: Dict) -> Dict[str, float]:
        """
        Calculate various uncertainty metrics for a prediction
        """
        probabilities = prediction_result.get('ensemble_probabilities', {})
        individual_preds = prediction_result.get('individual_predictions', {})
        
        if not probabilities:
            return {'entropy': 0.0, 'max_probability': 0.0, 'prediction_variance': 0.0}
        
        # Convert probabilities to numpy array
        prob_values = list(probabilities.values())
        prob_array = np.array(prob_values)
        
        # 1. Entropy-based uncertainty
        entropy_uncertainty = entropy(prob_array + 1e-10)  # Add small epsilon to avoid log(0)
        
        # 2. Maximum probability (confidence)
        max_prob = np.max(prob_array)
        
        # 3. Variance in predictions across models
        if len(individual_preds) > 1:
            # Create one-hot encoding for each prediction
            unique_classes = list(set(individual_preds.values()))
            class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
            
            pred_vectors = []
            for pred in individual_preds.values():
                vector = np.zeros(len(unique_classes))
                vector[class_to_idx[pred]] = 1
                pred_vectors.append(vector)
            
            pred_variance = np.var(pred_vectors, axis=0).mean()
        else:
            pred_variance = 0.0
        
        # 4. Agreement score (inverse of disagreement)
        agreement = prediction_result.get('agreement_score', 1.0)
        disagreement = 1.0 - agreement
        
        return {
            'entropy': float(entropy_uncertainty),
            'max_probability': float(max_prob),
            'prediction_variance': float(pred_variance),
            'disagreement': float(disagreement),
            'combined_uncertainty': float(entropy_uncertainty * 0.4 + (1-max_prob) * 0.3 + pred_variance * 0.3)
        }
    
    def identify_uncertain_samples(self, predictions: List[Dict], texts: List[str], 
                                 strategy: str = 'combined') -> List[Dict]:
        """
        Identify samples that need human review based on uncertainty
        
        Strategies:
        - 'entropy': Highest entropy predictions
        - 'confidence': Lowest confidence predictions  
        - 'disagreement': Highest model disagreement
        - 'combined': Combined uncertainty score
        - 'diverse': Diverse uncertain samples using clustering
        """
        uncertain_samples = []
        
        for i, (pred_result, text) in enumerate(zip(predictions, texts)):
            uncertainty_metrics = self.calculate_uncertainty_metrics(pred_result)
            
            # Determine if sample is uncertain based on strategy
            is_uncertain = False
            uncertainty_score = 0.0
            
            if strategy == 'entropy':
                uncertainty_score = uncertainty_metrics['entropy']
                is_uncertain = uncertainty_score > np.log(7) * 0.7  # 70% of max entropy for 7 classes
            elif strategy == 'confidence':
                uncertainty_score = 1 - uncertainty_metrics['max_probability']
                is_uncertain = uncertainty_metrics['max_probability'] < self.uncertainty_threshold
            elif strategy == 'disagreement':
                uncertainty_score = uncertainty_metrics['disagreement']
                is_uncertain = uncertainty_score > 0.3  # More than 30% disagreement
            elif strategy == 'combined':
                uncertainty_score = uncertainty_metrics['combined_uncertainty']
                is_uncertain = uncertainty_score > 0.5
            
            if is_uncertain:
                sample_info = {
                    'index': i,
                    'text': text,
                    'prediction': pred_result.get('ensemble_prediction', 'Unknown'),
                    'confidence': pred_result.get('ensemble_confidence', 0.0),
                    'uncertainty_score': uncertainty_score,
                    'uncertainty_metrics': uncertainty_metrics,
                    'individual_predictions': pred_result.get('individual_predictions', {}),
                    'needs_review': True,
                    'timestamp': datetime.now().isoformat()
                }
                uncertain_samples.append(sample_info)
        
        # Sort by uncertainty score (descending)
        uncertain_samples.sort(key=lambda x: x['uncertainty_score'], reverse=True)
        
        # Apply diversity sampling if requested
        if strategy == 'diverse' and len(uncertain_samples) > 10:
            uncertain_samples = self._apply_diversity_sampling(uncertain_samples, max_samples=20)
        
        return uncertain_samples
    
    def _apply_diversity_sampling(self, uncertain_samples: List[Dict], max_samples: int = 20) -> List[Dict]:
        """Apply diversity sampling to select diverse uncertain samples"""
        if len(uncertain_samples) <= max_samples:
            return uncertain_samples
        
        try:
            # Use text length and prediction features for clustering
            features = []
            for sample in uncertain_samples:
                feature_vector = [
                    len(sample['text']),
                    sample['uncertainty_score'],
                    sample['confidence'],
                    len(sample['text'].split()),  # word count
                    sample['text'].count('?'),    # question marks
                    sample['text'].count('!'),    # exclamation marks
                ]
                features.append(feature_vector)
            
            features = np.array(features)
            
            # Apply K-means clustering
            n_clusters = min(max_samples, len(uncertain_samples))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features)
            
            # Select one sample from each cluster (the most uncertain one)
            diverse_samples = []
            for cluster_id in range(n_clusters):
                cluster_samples = [sample for i, sample in enumerate(uncertain_samples) if clusters[i] == cluster_id]
                if cluster_samples:
                    # Pick the most uncertain from this cluster
                    most_uncertain = max(cluster_samples, key=lambda x: x['uncertainty_score'])
                    diverse_samples.append(most_uncertain)
            
            return diverse_samples[:max_samples]
            
        except Exception as e:
            self.logger.error(f"Error in diversity sampling: {str(e)}")
            return uncertain_samples[:max_samples]
    
    def record_human_feedback(self, sample_index: int, original_text: str, 
                            predicted_label: str, correct_label: str, 
                            confidence: float, metadata: Dict = None) -> bool:
        """Record human feedback for model improvement"""
        try:
            feedback_entry = {
                'sample_index': sample_index,
                'text': original_text,
                'predicted_label': predicted_label,
                'correct_label': correct_label,
                'was_correct': predicted_label == correct_label,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            self.feedback_data.append(feedback_entry)
            
            # Track correction patterns
            if predicted_label != correct_label:
                error_pattern = f"{predicted_label} -> {correct_label}"
                if error_pattern not in self.human_corrections:
                    self.human_corrections[error_pattern] = []
                self.human_corrections[error_pattern].append(feedback_entry)
            
            self.logger.info(f"Recorded feedback: {predicted_label} -> {correct_label}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording feedback: {str(e)}")
            return False
    
    def analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in human corrections to identify systematic errors"""
        if not self.human_corrections:
            return {'error_patterns': {}, 'insights': []}
        
        error_analysis = {}
        
        # Count error patterns
        for pattern, corrections in self.human_corrections.items():
            predicted, correct = pattern.split(' -> ')
            
            error_analysis[pattern] = {
                'count': len(corrections),
                'predicted_class': predicted,
                'correct_class': correct,
                'sample_texts': [c['text'][:100] + '...' if len(c['text']) > 100 else c['text'] 
                               for c in corrections[:5]],  # First 5 examples
                'avg_confidence': np.mean([c['confidence'] for c in corrections]),
                'recent_occurrences': len([c for c in corrections 
                                         if (datetime.now() - datetime.fromisoformat(c['timestamp'])).days <= 7])
            }
        
        # Generate insights
        insights = []
        
        # Most common errors
        most_common_errors = sorted(error_analysis.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
        if most_common_errors:
            insights.append(f"Most common error: {most_common_errors[0][0]} ({most_common_errors[0][1]['count']} occurrences)")
        
        # High confidence errors (overconfident model)
        high_conf_errors = [pattern for pattern, data in error_analysis.items() 
                           if data['avg_confidence'] > 0.8]
        if high_conf_errors:
            insights.append(f"Model is overconfident in these error patterns: {', '.join(high_conf_errors[:3])}")
        
        # Recent error trends
        recent_errors = [pattern for pattern, data in error_analysis.items() 
                        if data['recent_occurrences'] > 0]
        if recent_errors:
            insights.append(f"Recent errors (last 7 days): {len(recent_errors)} different patterns")
        
        return {
            'error_patterns': error_analysis,
            'insights': insights,
            'total_corrections': len(self.feedback_data),
            'accuracy_from_feedback': self._calculate_feedback_accuracy()
        }
    
    def _calculate_feedback_accuracy(self) -> float:
        """Calculate accuracy based on human feedback"""
        if not self.feedback_data:
            return 0.0
        
        correct_predictions = sum(1 for feedback in self.feedback_data if feedback['was_correct'])
        return correct_predictions / len(self.feedback_data)
    
    def suggest_retraining_strategy(self) -> Dict[str, Any]:
        """Suggest optimal retraining strategy based on accumulated feedback"""
        if len(self.feedback_data) < self.min_samples_for_retrain:
            return {
                'should_retrain': False,
                'reason': f'Need at least {self.min_samples_for_retrain} feedback samples (have {len(self.feedback_data)})',
                'current_samples': len(self.feedback_data)
            }
        
        error_patterns = self.analyze_error_patterns()
        
        # Determine retraining strategy
        strategy_recommendations = []
        
        # Check if we have systematic errors
        if error_patterns['error_patterns']:
            most_common = max(error_patterns['error_patterns'].items(), key=lambda x: x[1]['count'])
            if most_common[1]['count'] >= 5:  # At least 5 instances of same error
                strategy_recommendations.append({
                    'type': 'focused_retraining',
                    'target': f"Focus on distinguishing {most_common[0].replace(' -> ', ' from ')}",
                    'priority': 'high'
                })
        
        # Check for data imbalance in corrections
        correction_classes = defaultdict(int)
        for feedback in self.feedback_data:
            correction_classes[feedback['correct_label']] += 1
        
        if correction_classes:
            min_class_count = min(correction_classes.values())
            max_class_count = max(correction_classes.values())
            if max_class_count / min_class_count > 3:  # Significant imbalance
                strategy_recommendations.append({
                    'type': 'balance_training_data',
                    'target': 'Address class imbalance in training data',
                    'priority': 'medium'
                })
        
        # Overall strategy
        overall_accuracy = self._calculate_feedback_accuracy()
        if overall_accuracy < 0.8:
            strategy_recommendations.append({
                'type': 'general_retraining',
                'target': 'Improve overall model performance',
                'priority': 'high'
            })
        
        return {
            'should_retrain': len(strategy_recommendations) > 0,
            'strategies': strategy_recommendations,
            'feedback_accuracy': overall_accuracy,
            'total_feedback_samples': len(self.feedback_data),
            'estimated_improvement': self._estimate_improvement_potential()
        }
    
    def _estimate_improvement_potential(self) -> float:
        """Estimate potential improvement from retraining"""
        if not self.feedback_data:
            return 0.0
        
        current_accuracy = self._calculate_feedback_accuracy()
        
        # Simple heuristic: improvement potential based on error patterns and confidence
        high_conf_errors = sum(1 for f in self.feedback_data 
                              if not f['was_correct'] and f['confidence'] > 0.8)
        total_errors = sum(1 for f in self.feedback_data if not f['was_correct'])
        
        if total_errors == 0:
            return 0.0
        
        # Potential improvement is higher when we have many high-confidence errors
        improvement_factor = (high_conf_errors / total_errors) * 0.15  # Max 15% improvement
        
        return min(improvement_factor, 1.0 - current_accuracy)  # Can't exceed perfect accuracy
    
    def export_training_data(self, filepath: str, format: str = 'csv') -> bool:
        """Export corrected samples for retraining"""
        try:
            if not self.feedback_data:
                self.logger.warning("No feedback data to export")
                return False
            
            # Prepare data
            export_data = []
            for feedback in self.feedback_data:
                export_data.append({
                    'text': feedback['text'],
                    'category': feedback['correct_label'],
                    'original_prediction': feedback['predicted_label'],
                    'confidence': feedback['confidence'],
                    'timestamp': feedback['timestamp']
                })
            
            df = pd.DataFrame(export_data)
            
            if format.lower() == 'csv':
                df.to_csv(filepath, index=False)
            elif format.lower() == 'json':
                df.to_json(filepath, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Exported {len(export_data)} samples to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting training data: {str(e)}")
            return False
    
    def save_active_learning_state(self, filepath: str):
        """Save the active learning state"""
        state = {
            'feedback_data': self.feedback_data,
            'uncertainty_history': self.uncertainty_history,
            'human_corrections': self.human_corrections,
            'model_performance_tracker': self.model_performance_tracker,
            'config': {
                'uncertainty_threshold': self.uncertainty_threshold,
                'min_samples_for_retrain': self.min_samples_for_retrain
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_active_learning_state(self, filepath: str) -> bool:
        """Load the active learning state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.feedback_data = state.get('feedback_data', [])
            self.uncertainty_history = state.get('uncertainty_history', [])
            self.human_corrections = state.get('human_corrections', {})
            self.model_performance_tracker = state.get('model_performance_tracker', [])
            
            config = state.get('config', {})
            self.uncertainty_threshold = config.get('uncertainty_threshold', 0.8)
            self.min_samples_for_retrain = config.get('min_samples_for_retrain', 50)
            
            self.logger.info(f"Loaded active learning state with {len(self.feedback_data)} feedback samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading active learning state: {str(e)}")
            return False

if __name__ == "__main__":
    # Example usage
    oracle = ActiveLearningOracle()
    
    # Simulate some predictions
    sample_predictions = [
        {
            'ensemble_prediction': 'Technical Issue',
            'ensemble_confidence': 0.6,
            'ensemble_probabilities': {'Technical Issue': 0.6, 'Billing': 0.4},
            'individual_predictions': {'lstm': 'Technical Issue', 'rf': 'Billing'},
            'agreement_score': 0.5
        }
    ]
    
    sample_texts = ["My app keeps crashing when I try to upload files"]
    
    uncertain_samples = oracle.identify_uncertain_samples(sample_predictions, sample_texts)
    print(f"Found {len(uncertain_samples)} uncertain samples")
    
    if uncertain_samples:
        print(f"Most uncertain: {uncertain_samples[0]['text'][:50]}...")
        print(f"Uncertainty score: {uncertain_samples[0]['uncertainty_score']:.3f}")
