"""
Real-time Learning System with Feedback Loops
Implements online learning and continuous model improvement based on user feedback.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import os
import sqlite3
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import time
import pickle
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

@dataclass
class FeedbackEntry:
    """Data class for user feedback."""
    timestamp: datetime
    text: str
    predicted_category: str
    true_category: str
    confidence: float
    user_id: Optional[str] = None
    feedback_type: str = "correction"  # correction, confirmation, rating
    additional_info: Dict[str, Any] = None

@dataclass
class LearningStats:
    """Statistics for online learning."""
    total_predictions: int
    correct_predictions: int
    feedback_received: int
    model_updates: int
    last_update_time: datetime
    current_accuracy: float
    improvement_rate: float

class OnlineLearner(BaseEstimator, ClassifierMixin):
    """
    Online learning system that:
    - Collects user feedback in real-time
    - Updates model weights incrementally
    - Maintains confidence-based learning
    - Implements active learning strategies
    - Provides performance monitoring
    """
    
    def __init__(self, 
                 base_model=None,
                 learning_rate: float = 0.01,
                 confidence_threshold: float = 0.7,
                 batch_size: int = 10,
                 update_frequency: int = 50,
                 max_memory: int = 10000,
                 db_path: str = "models/feedback.db"):
        """
        Initialize online learner.
        
        Args:
            base_model: Base classifier to adapt
            learning_rate: Learning rate for updates
            confidence_threshold: Threshold for confident predictions
            batch_size: Size of mini-batches for updates
            update_frequency: Number of feedback items before update
            max_memory: Maximum items to keep in memory
            db_path: Path to feedback database
        """
        self.base_model = base_model
        self.learning_rate = learning_rate
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.max_memory = max_memory
        self.db_path = db_path
        
        # Feedback storage
        self.feedback_buffer = deque(maxlen=max_memory)
        self.pending_updates = deque()
        
        # Learning statistics
        self.stats = LearningStats(
            total_predictions=0,
            correct_predictions=0,
            feedback_received=0,
            model_updates=0,
            last_update_time=datetime.now(),
            current_accuracy=0.0,
            improvement_rate=0.0
        )
        
        # Online learning state
        self.feature_weights = {}
        self.class_counts = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        
        # Setup database and logging
        self._setup_database()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Background update thread
        self.update_thread = None
        self.stop_learning = False
        
    def _setup_database(self):
        """Setup SQLite database for persistent feedback storage."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                true_category TEXT NOT NULL,
                confidence REAL NOT NULL,
                user_id TEXT,
                feedback_type TEXT DEFAULT 'correction',
                additional_info TEXT,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Create stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_predictions INTEGER,
                correct_predictions INTEGER,
                feedback_received INTEGER,
                model_updates INTEGER,
                current_accuracy REAL,
                improvement_rate REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def predict(self, texts: Union[str, List[str]], 
                return_confidence: bool = True) -> Union[str, List[str], Tuple]:
        """
        Make predictions and track for learning.
        
        Args:
            texts: Input text(s) to classify
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predictions and optionally confidence scores
        """
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        # Get predictions from base model
        if hasattr(self.base_model, 'predict_proba'):
            probabilities = self.base_model.predict_proba(texts)
            predictions = self.base_model.classes_[np.argmax(probabilities, axis=1)]
            confidences = np.max(probabilities, axis=1)
        else:
            predictions = self.base_model.predict(texts)
            confidences = np.ones(len(predictions)) * 0.5  # Default confidence
        
        # Update statistics
        self.stats.total_predictions += len(texts)
        
        # Store predictions for potential feedback
        for i, (text, pred, conf) in enumerate(zip(texts, predictions, confidences)):
            self._store_prediction(text, pred, conf)
        
        if return_confidence:
            if single_input:
                return predictions[0], confidences[0]
            return predictions, confidences
        else:
            if single_input:
                return predictions[0]
            return predictions
    
    def _store_prediction(self, text: str, prediction: str, confidence: float):
        """Store prediction for potential feedback collection."""
        # This could be extended to store in a temporary cache
        # for matching with future feedback
        pass
    
    def add_feedback(self, 
                    text: str,
                    predicted_category: str,
                    true_category: str,
                    confidence: float,
                    user_id: Optional[str] = None,
                    feedback_type: str = "correction") -> bool:
        """
        Add user feedback for online learning.
        
        Args:
            text: Original text
            predicted_category: What the model predicted
            true_category: Correct category from user
            confidence: Model's confidence in prediction
            user_id: Optional user identifier
            feedback_type: Type of feedback
            
        Returns:
            Whether feedback was successfully added
        """
        try:
            feedback = FeedbackEntry(
                timestamp=datetime.now(),
                text=text,
                predicted_category=predicted_category,
                true_category=true_category,
                confidence=confidence,
                user_id=user_id,
                feedback_type=feedback_type
            )
            
            # Add to memory buffer
            self.feedback_buffer.append(feedback)
            
            # Store in database
            self._store_feedback_db(feedback)
            
            # Update statistics
            self.stats.feedback_received += 1
            if predicted_category == true_category:
                self.stats.correct_predictions += 1
            
            # Add to pending updates
            self.pending_updates.append(feedback)
            
            # Trigger update if needed
            if len(self.pending_updates) >= self.update_frequency:
                self._trigger_model_update()
            
            self.logger.info(f"Feedback added: {predicted_category} -> {true_category}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding feedback: {e}")
            return False
    
    def _store_feedback_db(self, feedback: FeedbackEntry):
        """Store feedback in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback 
            (timestamp, text, predicted_category, true_category, confidence, 
             user_id, feedback_type, additional_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback.timestamp.isoformat(),
            feedback.text,
            feedback.predicted_category,
            feedback.true_category,
            feedback.confidence,
            feedback.user_id,
            feedback.feedback_type,
            json.dumps(feedback.additional_info) if feedback.additional_info else None
        ))
        
        conn.commit()
        conn.close()
    
    def _trigger_model_update(self):
        """Trigger model update in background thread."""
        if self.update_thread is None or not self.update_thread.is_alive():
            self.update_thread = threading.Thread(target=self._update_model)
            self.update_thread.daemon = True
            self.update_thread.start()
    
    def _update_model(self):
        """Update model with accumulated feedback."""
        try:
            if not self.pending_updates:
                return
            
            # Get batch of feedback for update
            batch = []
            for _ in range(min(self.batch_size, len(self.pending_updates))):
                if self.pending_updates:
                    batch.append(self.pending_updates.popleft())
            
            if not batch:
                return
            
            # Prepare training data
            texts = [fb.text for fb in batch]
            labels = [fb.true_category for fb in batch]
            
            # Update model incrementally
            self._incremental_update(texts, labels, batch)
            
            # Update statistics
            self.stats.model_updates += 1
            self.stats.last_update_time = datetime.now()
            
            # Calculate current accuracy
            self._update_accuracy_stats()
            
            # Store stats in database
            self._store_stats_db()
            
            self.logger.info(f"Model updated with {len(batch)} feedback items")
            
        except Exception as e:
            self.logger.error(f"Error updating model: {e}")
    
    def _incremental_update(self, texts: List[str], labels: List[str], 
                          feedback_batch: List[FeedbackEntry]):
        """Perform incremental model update."""
        # This is a simplified implementation
        # In practice, you would implement specific incremental learning
        # algorithms based on your base model type
        
        if hasattr(self.base_model, 'partial_fit'):
            # For models that support partial_fit
            try:
                # Try to extract features if the model has a preprocessing pipeline
                if hasattr(self.base_model, 'named_steps'):
                    # Pipeline case
                    self.base_model.partial_fit(texts, labels)
                else:
                    # Direct model case - may need feature extraction
                    self.base_model.partial_fit(texts, labels)
                    
            except Exception as e:
                self.logger.warning(f"Partial fit failed: {e}")
                # Fallback to retraining with recent data
                self._retrain_with_recent_data()
        else:
            # For models that don't support incremental learning
            self._retrain_with_recent_data()
    
    def _retrain_with_recent_data(self):
        """Retrain model with recent feedback data."""
        try:
            # Get recent feedback
            recent_feedback = list(self.feedback_buffer)[-self.batch_size * 5:]  # Last 5 batches
            
            if len(recent_feedback) < 10:  # Need minimum samples
                return
            
            texts = [fb.text for fb in recent_feedback]
            labels = [fb.true_category for fb in recent_feedback]
            
            # Retrain model
            self.base_model.fit(texts, labels)
            
            self.logger.info(f"Model retrained with {len(recent_feedback)} recent samples")
            
        except Exception as e:
            self.logger.error(f"Error retraining model: {e}")
    
    def _update_accuracy_stats(self):
        """Update accuracy statistics."""
        if self.stats.total_predictions > 0:
            self.stats.current_accuracy = self.stats.correct_predictions / self.stats.total_predictions
        
        # Calculate improvement rate (simplified)
        if self.stats.model_updates > 1:
            # This would ideally compare performance before/after updates
            self.stats.improvement_rate = 0.01  # Placeholder
    
    def _store_stats_db(self):
        """Store learning statistics in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO learning_stats 
            (timestamp, total_predictions, correct_predictions, feedback_received,
             model_updates, current_accuracy, improvement_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            self.stats.total_predictions,
            self.stats.correct_predictions,
            self.stats.feedback_received,
            self.stats.model_updates,
            self.stats.current_accuracy,
            self.stats.improvement_rate
        ))
        
        conn.commit()
        conn.close()
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics."""
        return asdict(self.stats)
    
    def get_feedback_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of feedback received in the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get feedback counts by category
        cursor.execute('''
            SELECT true_category, COUNT(*) as count
            FROM feedback 
            WHERE datetime(timestamp) > datetime(?)
            GROUP BY true_category
            ORDER BY count DESC
        ''', (cutoff_date.isoformat(),))
        
        category_counts = dict(cursor.fetchall())
        
        # Get accuracy by category
        cursor.execute('''
            SELECT true_category, 
                   COUNT(*) as total,
                   SUM(CASE WHEN predicted_category = true_category THEN 1 ELSE 0 END) as correct
            FROM feedback 
            WHERE datetime(timestamp) > datetime(?)
            GROUP BY true_category
        ''', (cutoff_date.isoformat(),))
        
        accuracy_by_category = {}
        for category, total, correct in cursor.fetchall():
            accuracy_by_category[category] = correct / total if total > 0 else 0
        
        # Get feedback by user
        cursor.execute('''
            SELECT user_id, COUNT(*) as count
            FROM feedback 
            WHERE datetime(timestamp) > datetime(?) AND user_id IS NOT NULL
            GROUP BY user_id
            ORDER BY count DESC
            LIMIT 10
        ''', (cutoff_date.isoformat(),))
        
        user_feedback = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_feedback': sum(category_counts.values()),
            'category_counts': category_counts,
            'accuracy_by_category': accuracy_by_category,
            'top_feedback_users': user_feedback,
            'period_days': days
        }
    
    def get_learning_trend(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get learning performance trend over time."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DATE(timestamp) as date,
                   current_accuracy,
                   model_updates,
                   feedback_received
            FROM learning_stats
            WHERE datetime(timestamp) > datetime(?)
            ORDER BY timestamp
        ''', (cutoff_date.isoformat(),))
        
        trend_data = []
        for row in cursor.fetchall():
            trend_data.append({
                'date': row[0],
                'accuracy': row[1],
                'model_updates': row[2],
                'feedback_received': row[3]
            })
        
        conn.close()
        return trend_data
    
    def get_uncertain_predictions(self, threshold: float = None) -> List[Dict[str, Any]]:
        """Get predictions with low confidence for active learning."""
        if threshold is None:
            threshold = self.confidence_threshold
        
        # This would typically store recent predictions with confidence scores
        # For now, return a placeholder
        return []
    
    def export_feedback_data(self, filepath: str, days: int = None):
        """Export feedback data to CSV."""
        conn = sqlite3.connect(self.db_path)
        
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            query = '''
                SELECT * FROM feedback 
                WHERE datetime(timestamp) > datetime(?)
                ORDER BY timestamp DESC
            '''
            df = pd.read_sql_query(query, conn, params=(cutoff_date.isoformat(),))
        else:
            query = 'SELECT * FROM feedback ORDER BY timestamp DESC'
            df = pd.read_sql_query(query, conn)
        
        df.to_csv(filepath, index=False)
        conn.close()
        
        self.logger.info(f"Exported {len(df)} feedback records to {filepath}")
    
    def reset_learning_state(self):
        """Reset the learning state (use with caution)."""
        self.feedback_buffer.clear()
        self.pending_updates.clear()
        
        self.stats = LearningStats(
            total_predictions=0,
            correct_predictions=0,
            feedback_received=0,
            model_updates=0,
            last_update_time=datetime.now(),
            current_accuracy=0.0,
            improvement_rate=0.0
        )
        
        self.logger.info("Learning state reset")
    
    def start_background_learning(self):
        """Start background learning process."""
        def learning_loop():
            while not self.stop_learning:
                try:
                    if len(self.pending_updates) >= self.update_frequency:
                        self._update_model()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    self.logger.error(f"Error in background learning: {e}")
        
        if self.update_thread is None or not self.update_thread.is_alive():
            self.stop_learning = False
            self.update_thread = threading.Thread(target=learning_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
            self.logger.info("Background learning started")
    
    def stop_background_learning(self):
        """Stop background learning process."""
        self.stop_learning = True
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
        self.logger.info("Background learning stopped")
    
    def save_learner_state(self, filepath: str):
        """Save the learner state."""
        state = {
            'stats': asdict(self.stats),
            'learning_rate': self.learning_rate,
            'confidence_threshold': self.confidence_threshold,
            'batch_size': self.batch_size,
            'update_frequency': self.update_frequency
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_learner_state(self, filepath: str):
        """Load the learner state."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        stats_data = state['stats']
        stats_data['last_update_time'] = datetime.fromisoformat(stats_data['last_update_time'])
        self.stats = LearningStats(**stats_data)
        
        self.learning_rate = state['learning_rate']
        self.confidence_threshold = state['confidence_threshold']
        self.batch_size = state['batch_size']
        self.update_frequency = state['update_frequency']
