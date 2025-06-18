"""
Advanced Data Drift Detection System
Monitors and detects changes in data distributions that might affect model performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import os
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pickle
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

class DataDriftDetector:
    """
    Advanced data drift detection system that monitors:
    - Statistical drift in text features
    - Vocabulary drift and semantic changes
    - Distribution drift in predictions
    - Concept drift in true labels
    - Covariate shift detection
    """
    
    def __init__(self, 
                 reference_data_path: str = None,
                 drift_threshold: float = 0.05,
                 window_size: int = 1000,
                 min_samples: int = 100):
        """
        Initialize drift detector.
        
        Args:
            reference_data_path: Path to reference/baseline data
            drift_threshold: P-value threshold for drift detection
            window_size: Size of sliding window for monitoring
            min_samples: Minimum samples needed for drift detection
        """
        self.reference_data_path = reference_data_path
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Reference data and statistics
        self.reference_data = None
        self.reference_features = None
        self.reference_stats = {}
        
        # Monitoring buffers
        self.current_window = deque(maxlen=window_size)
        self.drift_history = []
        self.alerts = []
        
        # Feature extractors
        self.tfidf_vectorizer = None
        self.pca_reducer = None
        self.scaler = StandardScaler()
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Drift detection methods
        self.drift_methods = {
            'ks_test': self._kolmogorov_smirnov_test,
            'chi2_test': self._chi_square_test,
            'psi': self._population_stability_index,
            'js_divergence': self._jensen_shannon_divergence,
            'wasserstein': self._wasserstein_distance,
            'covariate_shift': self._covariate_shift_detection
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def fit_reference_data(self, data: Union[pd.DataFrame, List[str]], 
                          labels: Optional[List[str]] = None):
        """
        Fit the drift detector on reference/baseline data.
        
        Args:
            data: Reference data (text or DataFrame)
            labels: Optional labels for supervised drift detection
        """
        try:
            if isinstance(data, list):
                # Text data
                self.reference_data = pd.DataFrame({'text': data})
                if labels:
                    self.reference_data['label'] = labels
            else:
                self.reference_data = data.copy()
            
            # Extract features
            self._extract_reference_features()
            
            # Compute reference statistics
            self._compute_reference_statistics()
            
            self.logger.info(f"Reference data fitted with {len(self.reference_data)} samples")
            
        except Exception as e:
            self.logger.error(f"Error fitting reference data: {e}")
            raise
    
    def _extract_reference_features(self):
        """Extract features from reference data."""
        if 'text' in self.reference_data.columns:
            # Text features
            texts = self.reference_data['text'].astype(str)
            
            # TF-IDF features
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
            
            # Dimensionality reduction
            self.pca_reducer = PCA(n_components=min(100, tfidf_features.shape[1]))
            reduced_features = self.pca_reducer.fit_transform(tfidf_features.toarray())
            
            # Scale features
            self.reference_features = self.scaler.fit_transform(reduced_features)
            
            # Fit outlier detector
            self.outlier_detector.fit(self.reference_features)
            
        else:
            # Numerical features
            numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.reference_features = self.scaler.fit_transform(
                    self.reference_data[numeric_cols]
                )
    
    def _compute_reference_statistics(self):
        """Compute reference statistics for drift detection."""
        if self.reference_features is not None:
            self.reference_stats = {
                'mean': np.mean(self.reference_features, axis=0),
                'std': np.std(self.reference_features, axis=0),
                'min': np.min(self.reference_features, axis=0),
                'max': np.max(self.reference_features, axis=0),
                'percentiles': {
                    p: np.percentile(self.reference_features, p, axis=0)
                    for p in [25, 50, 75, 90, 95, 99]
                }
            }
        
        # Text-specific statistics
        if 'text' in self.reference_data.columns:
            texts = self.reference_data['text'].astype(str)
            self.reference_stats.update({
                'avg_length': np.mean([len(text) for text in texts]),
                'vocab_size': len(self.tfidf_vectorizer.vocabulary_),
                'common_words': self._get_common_words(texts),
                'avg_words': np.mean([len(text.split()) for text in texts])
            })
    
    def _get_common_words(self, texts: List[str], top_k: int = 100) -> List[str]:
        """Get most common words from texts."""
        from collections import Counter
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())
        return [word for word, _ in Counter(all_words).most_common(top_k)]
    
    def detect_drift(self, new_data: Union[pd.DataFrame, List[str]], 
                    labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect drift in new data compared to reference data.
        
        Args:
            new_data: New data to check for drift
            labels: Optional labels for new data
            
        Returns:
            Dictionary with drift detection results
        """
        try:
            if self.reference_data is None:
                raise ValueError("Reference data not fitted. Call fit_reference_data first.")
            
            # Prepare new data
            if isinstance(new_data, list):
                current_data = pd.DataFrame({'text': new_data})
                if labels:
                    current_data['label'] = labels
            else:
                current_data = new_data.copy()
            
            # Extract features from new data
            current_features = self._extract_current_features(current_data)
            
            # Perform drift detection
            drift_results = {}
            for method_name, method_func in self.drift_methods.items():
                try:
                    drift_results[method_name] = method_func(current_features, current_data)
                except Exception as e:
                    self.logger.warning(f"Error in {method_name}: {e}")
                    drift_results[method_name] = {'error': str(e)}
            
            # Overall drift assessment
            overall_drift = self._assess_overall_drift(drift_results)
            
            # Create drift report
            drift_report = {
                'timestamp': datetime.now().isoformat(),
                'sample_size': len(current_data),
                'drift_detected': overall_drift['drift_detected'],
                'drift_score': overall_drift['drift_score'],
                'drift_methods': drift_results,
                'recommendations': self._generate_recommendations(drift_results)
            }
            
            # Store in history
            self.drift_history.append(drift_report)
            
            # Generate alerts if needed
            if drift_report['drift_detected']:
                self._generate_alert(drift_report)
                
            return drift_report
            
        except Exception as e:
            self.logger.error(f"Error detecting drift: {e}")
            return {'error': str(e)}
    
    def _extract_current_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features from current data using fitted transformers."""
        if 'text' in data.columns and self.tfidf_vectorizer is not None:
            texts = data['text'].astype(str)
            tfidf_features = self.tfidf_vectorizer.transform(texts)
            reduced_features = self.pca_reducer.transform(tfidf_features.toarray())
            return self.scaler.transform(reduced_features)
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return self.scaler.transform(data[numeric_cols])
        return np.array([])
    
    def _kolmogorov_smirnov_test(self, current_features: np.ndarray, 
                                current_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test for drift detection."""
        if len(current_features) == 0:
            return {'error': 'No features to test'}
        
        ks_results = []
        for i in range(current_features.shape[1]):
            ref_feature = self.reference_features[:, i]
            curr_feature = current_features[:, i]
            
            ks_stat, p_value = stats.ks_2samp(ref_feature, curr_feature)
            ks_results.append({
                'feature_idx': i,
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'drift_detected': p_value < self.drift_threshold
            })
        
        drift_detected = any(r['drift_detected'] for r in ks_results)
        avg_p_value = np.mean([r['p_value'] for r in ks_results])
        
        return {
            'drift_detected': drift_detected,
            'avg_p_value': avg_p_value,
            'feature_results': ks_results
        }
    
    def _chi_square_test(self, current_features: np.ndarray, 
                        current_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform Chi-square test for categorical drift."""
        if 'label' not in current_data.columns or 'label' not in self.reference_data.columns:
            return {'error': 'Labels required for chi-square test'}
        
        ref_labels = self.reference_data['label'].value_counts()
        curr_labels = current_data['label'].value_counts()
        
        # Align categories
        all_categories = set(ref_labels.index) | set(curr_labels.index)
        ref_counts = [ref_labels.get(cat, 0) for cat in all_categories]
        curr_counts = [curr_labels.get(cat, 0) for cat in all_categories]
        
        chi2_stat, p_value = stats.chisquare(curr_counts, ref_counts)
        
        return {
            'drift_detected': p_value < self.drift_threshold,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'reference_distribution': dict(zip(all_categories, ref_counts)),
            'current_distribution': dict(zip(all_categories, curr_counts))
        }
    
    def _population_stability_index(self, current_features: np.ndarray, 
                                   current_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Population Stability Index (PSI)."""
        if len(current_features) == 0:
            return {'error': 'No features to calculate PSI'}
        
        psi_values = []
        for i in range(current_features.shape[1]):
            ref_feature = self.reference_features[:, i]
            curr_feature = current_features[:, i]
            
            # Create bins
            bins = np.linspace(
                min(ref_feature.min(), curr_feature.min()),
                max(ref_feature.max(), curr_feature.max()),
                10
            )
            
            # Calculate distributions
            ref_dist, _ = np.histogram(ref_feature, bins=bins)
            curr_dist, _ = np.histogram(curr_feature, bins=bins)
            
            # Normalize
            ref_dist = ref_dist / ref_dist.sum() + 1e-10
            curr_dist = curr_dist / curr_dist.sum() + 1e-10
            
            # Calculate PSI
            psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
            psi_values.append(psi)
        
        avg_psi = np.mean(psi_values)
        drift_detected = avg_psi > 0.1  # Standard PSI threshold
        
        return {
            'drift_detected': drift_detected,
            'avg_psi': avg_psi,
            'psi_values': psi_values,
            'interpretation': self._interpret_psi(avg_psi)
        }
    
    def _interpret_psi(self, psi_value: float) -> str:
        """Interpret PSI value."""
        if psi_value < 0.1:
            return "No significant change"
        elif psi_value < 0.2:
            return "Minor change"
        else:
            return "Major change - model may need retraining"
    
    def _jensen_shannon_divergence(self, current_features: np.ndarray, 
                                  current_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Jensen-Shannon divergence."""
        if len(current_features) == 0:
            return {'error': 'No features to calculate JS divergence'}
        
        js_values = []
        for i in range(current_features.shape[1]):
            ref_feature = self.reference_features[:, i]
            curr_feature = current_features[:, i]
            
            # Create probability distributions
            bins = np.linspace(
                min(ref_feature.min(), curr_feature.min()),
                max(ref_feature.max(), curr_feature.max()),
                20
            )
            
            ref_hist, _ = np.histogram(ref_feature, bins=bins)
            curr_hist, _ = np.histogram(curr_feature, bins=bins)
            
            # Normalize to probabilities
            ref_prob = ref_hist / ref_hist.sum() + 1e-10
            curr_prob = curr_hist / curr_hist.sum() + 1e-10
            
            # Calculate JS divergence
            m = 0.5 * (ref_prob + curr_prob)
            js_div = 0.5 * stats.entropy(ref_prob, m) + 0.5 * stats.entropy(curr_prob, m)
            js_values.append(js_div)
        
        avg_js = np.mean(js_values)
        drift_detected = avg_js > 0.1  # Adjustable threshold
        
        return {
            'drift_detected': drift_detected,
            'avg_js_divergence': avg_js,
            'js_values': js_values
        }
    
    def _wasserstein_distance(self, current_features: np.ndarray, 
                             current_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Wasserstein distance."""
        if len(current_features) == 0:
            return {'error': 'No features to calculate Wasserstein distance'}
        
        wasserstein_values = []
        for i in range(current_features.shape[1]):
            ref_feature = self.reference_features[:, i]
            curr_feature = current_features[:, i]
            
            # Calculate Wasserstein distance
            w_dist = stats.wasserstein_distance(ref_feature, curr_feature)
            wasserstein_values.append(w_dist)
        
        avg_wasserstein = np.mean(wasserstein_values)
        # Normalize by feature std for threshold
        normalized_w = avg_wasserstein / np.mean(self.reference_stats['std'])
        drift_detected = normalized_w > 0.1  # Adjustable threshold
        
        return {
            'drift_detected': drift_detected,
            'avg_wasserstein_distance': avg_wasserstein,
            'normalized_distance': normalized_w,
            'wasserstein_values': wasserstein_values
        }
    
    def _covariate_shift_detection(self, current_features: np.ndarray, 
                                  current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect covariate shift using classifier approach."""
        if len(current_features) == 0:
            return {'error': 'No features for covariate shift detection'}
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import roc_auc_score
            
            # Create binary classification problem
            ref_size = len(self.reference_features)
            curr_size = len(current_features)
            
            # Balance datasets
            min_size = min(ref_size, curr_size)
            ref_sample = self.reference_features[:min_size]
            curr_sample = current_features[:min_size]
            
            # Combine data
            X = np.vstack([ref_sample, curr_sample])
            y = np.hstack([np.zeros(min_size), np.ones(min_size)])
            
            # Train classifier
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            
            # Evaluate
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # AUC close to 0.5 indicates no covariate shift
            drift_detected = auc_score > 0.7  # Adjustable threshold
            
            return {
                'drift_detected': drift_detected,
                'auc_score': auc_score,
                'feature_importance': clf.feature_importances_.tolist(),
                'interpretation': f"AUC = {auc_score:.3f} - {'Covariate shift detected' if drift_detected else 'No significant covariate shift'}"
            }
            
        except Exception as e:
            return {'error': f'Covariate shift detection failed: {e}'}
    
    def _assess_overall_drift(self, drift_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall drift based on multiple methods."""
        valid_results = {k: v for k, v in drift_results.items() if 'error' not in v}
        
        if not valid_results:
            return {'drift_detected': False, 'drift_score': 0.0}
        
        drift_votes = sum(1 for result in valid_results.values() 
                         if result.get('drift_detected', False))
        total_methods = len(valid_results)
        
        drift_score = drift_votes / total_methods
        drift_detected = drift_score > 0.5  # Majority vote
        
        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'methods_detecting_drift': drift_votes,
            'total_methods': total_methods
        }
    
    def _generate_recommendations(self, drift_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on drift detection results."""
        recommendations = []
        
        # Check PSI results
        if 'psi' in drift_results and 'avg_psi' in drift_results['psi']:
            psi_value = drift_results['psi']['avg_psi']
            if psi_value > 0.2:
                recommendations.append("Major distribution shift detected - consider retraining the model")
            elif psi_value > 0.1:
                recommendations.append("Minor distribution shift detected - monitor performance closely")
        
        # Check covariate shift
        if 'covariate_shift' in drift_results and drift_results['covariate_shift'].get('drift_detected', False):
            recommendations.append("Covariate shift detected - input data distribution has changed")
        
        # Check multiple methods
        drift_count = sum(1 for result in drift_results.values() 
                         if result.get('drift_detected', False))
        
        if drift_count >= 3:
            recommendations.append("Multiple drift detection methods triggered - immediate attention required")
        elif drift_count >= 2:
            recommendations.append("Moderate drift detected - consider model update")
        
        if not recommendations:
            recommendations.append("No significant drift detected - model appears stable")
        
        return recommendations
    
    def _generate_alert(self, drift_report: Dict[str, Any]):
        """Generate alert for detected drift."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': 'drift_detected',
            'severity': 'high' if drift_report['drift_score'] > 0.7 else 'medium',
            'message': f"Data drift detected with score {drift_report['drift_score']:.2f}",
            'sample_size': drift_report['sample_size'],
            'recommendations': drift_report['recommendations']
        }
        
        self.alerts.append(alert)
        self.logger.warning(f"DRIFT ALERT: {alert['message']}")
    
    def get_drift_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get drift detection history for the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        return [
            report for report in self.drift_history
            if datetime.fromisoformat(report['timestamp']) > cutoff_date
        ]
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection status."""
        if not self.drift_history:
            return {'status': 'No drift history available'}
        
        recent_reports = self.get_drift_history(7)  # Last 7 days
        
        if not recent_reports:
            return {'status': 'No recent drift reports'}
        
        avg_drift_score = np.mean([r['drift_score'] for r in recent_reports])
        drift_detected_count = sum(1 for r in recent_reports if r['drift_detected'])
        
        return {
            'status': 'active',
            'recent_reports': len(recent_reports),
            'avg_drift_score': avg_drift_score,
            'drift_detected_count': drift_detected_count,
            'latest_report': recent_reports[-1] if recent_reports else None,
            'alerts': len(self.alerts)
        }
    
    def save_state(self, filepath: str):
        """Save drift detector state."""
        state = {
            'reference_stats': self.reference_stats,
            'drift_history': self.drift_history,
            'alerts': self.alerts,
            'config': {
                'drift_threshold': self.drift_threshold,
                'window_size': self.window_size,
                'min_samples': self.min_samples
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self, filepath: str):
        """Load drift detector state."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.reference_stats = state.get('reference_stats', {})
        self.drift_history = state.get('drift_history', [])
        self.alerts = state.get('alerts', [])
        
        config = state.get('config', {})
        self.drift_threshold = config.get('drift_threshold', self.drift_threshold)
        self.window_size = config.get('window_size', self.window_size)
        self.min_samples = config.get('min_samples', self.min_samples)
