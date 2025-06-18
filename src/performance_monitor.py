"""
Model Performance Monitoring and Alerting System
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import json
import os
from datetime import datetime, timedelta
import sqlite3
from dataclasses import dataclass
from collections import defaultdict, deque
import threading
import time
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PerformanceAlert:
    """Data class for performance alerts"""
    alert_type: str
    message: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold: float
    model_name: str = "default"
    
class ModelPerformanceMonitor:
    """
    Advanced model performance monitoring system that:
    - Tracks prediction accuracy, confidence, and latency in real-time
    - Detects performance degradation and concept drift
    - Sends automated alerts when thresholds are breached
    - Maintains historical performance metrics
    - Provides recommendations for model maintenance
    """
    
    def __init__(self, db_path: str = "models/monitoring.db", 
                 alert_config: Dict = None):
        self.db_path = db_path
        self.alert_config = alert_config or self._default_alert_config()
        self.performance_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.alerts = []
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
        
        # Performance thresholds
        self.thresholds = {
            'accuracy_drop': 0.05,  # 5% drop in accuracy
            'confidence_drop': 0.1,  # 10% drop in average confidence
            'latency_increase': 2.0,  # 2x increase in latency
            'error_rate_increase': 0.02,  # 2% increase in error rate
            'drift_score': 0.3  # Drift detection threshold
        }
        
    def _default_alert_config(self) -> Dict:
        """Default alert configuration"""
        return {
            'email_enabled': False,
            'email_smtp_server': 'smtp.gmail.com',
            'email_smtp_port': 587,
            'email_username': '',
            'email_password': '',
            'email_recipients': [],
            'alert_cooldown_minutes': 60,  # Minimum time between similar alerts
            'log_alerts': True
        }
    
    def _init_database(self):
        """Initialize SQLite database for storing metrics"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT,
                    input_text TEXT,
                    predicted_class TEXT,
                    confidence REAL,
                    actual_class TEXT,
                    is_correct BOOLEAN,
                    latency_ms REAL,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    window_size INTEGER,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    model_name TEXT,
                    metric_name TEXT,
                    current_value REAL,
                    threshold REAL,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
    
    def log_prediction(self, model_name: str, input_text: str, 
                      predicted_class: str, confidence: float,
                      latency_ms: float, actual_class: str = None,
                      metadata: Dict = None) -> int:
        """Log a single prediction for monitoring"""
        is_correct = None
        if actual_class is not None:
            is_correct = predicted_class == actual_class
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions 
                (model_name, input_text, predicted_class, confidence, 
                 actual_class, is_correct, latency_ms, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_name, input_text, predicted_class, confidence,
                actual_class, is_correct, latency_ms, 
                json.dumps(metadata) if metadata else None
            ))
            
            prediction_id = cursor.lastrowid
            conn.commit()
        
        # Update real-time metrics
        self._update_realtime_metrics(model_name, confidence, latency_ms, is_correct)
        
        return prediction_id
    
    def _update_realtime_metrics(self, model_name: str, confidence: float, 
                                latency_ms: float, is_correct: Optional[bool]):
        """Update real-time performance metrics"""
        current_time = datetime.now()
        
        # Add to buffers
        self.performance_buffer[f"{model_name}_confidence"].append((current_time, confidence))
        self.performance_buffer[f"{model_name}_latency"].append((current_time, latency_ms))
        
        if is_correct is not None:
            self.performance_buffer[f"{model_name}_accuracy"].append((current_time, int(is_correct)))
    
    def calculate_performance_metrics(self, model_name: str, 
                                    time_window_hours: int = 24) -> Dict[str, float]:
        """Calculate performance metrics for a given time window"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        with sqlite3.connect(self.db_path) as conn:
            # Query predictions in time window
            df = pd.read_sql_query('''
                SELECT * FROM predictions 
                WHERE model_name = ? AND timestamp >= ? AND timestamp <= ?
            ''', conn, params=(model_name, start_time, end_time))
        
        if df.empty:
            return {}
        
        metrics = {}
        
        # Accuracy (only where actual_class is known)
        accurate_predictions = df[df['is_correct'].notna()]
        if not accurate_predictions.empty:
            metrics['accuracy'] = accurate_predictions['is_correct'].mean()
            metrics['total_labeled_predictions'] = len(accurate_predictions)
        
        # Confidence statistics
        metrics['avg_confidence'] = df['confidence'].mean()
        metrics['min_confidence'] = df['confidence'].min()
        metrics['max_confidence'] = df['confidence'].max()
        metrics['confidence_std'] = df['confidence'].std()
        
        # Latency statistics
        metrics['avg_latency_ms'] = df['latency_ms'].mean()
        metrics['p95_latency_ms'] = df['latency_ms'].quantile(0.95)
        metrics['p99_latency_ms'] = df['latency_ms'].quantile(0.99)
        
        # Prediction volume
        metrics['total_predictions'] = len(df)
        metrics['predictions_per_hour'] = len(df) / time_window_hours
        
        # Error rate
        error_predictions = df[df['predicted_class'] == 'Error']
        metrics['error_rate'] = len(error_predictions) / len(df)
        
        # Class distribution
        class_dist = df['predicted_class'].value_counts(normalize=True)
        metrics['class_distribution'] = class_dist.to_dict()
        
        # Store metrics in database
        self._store_metrics(model_name, metrics, time_window_hours)
        
        return metrics
    
    def _store_metrics(self, model_name: str, metrics: Dict, window_size: int):
        """Store calculated metrics in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    cursor.execute('''
                        INSERT INTO performance_metrics 
                        (model_name, metric_name, metric_value, window_size)
                        VALUES (?, ?, ?, ?)
                    ''', (model_name, metric_name, metric_value, window_size))
            
            conn.commit()
    
    def detect_performance_degradation(self, model_name: str) -> List[PerformanceAlert]:
        """Detect performance degradation and generate alerts"""
        alerts = []
        
        # Get current metrics (last 1 hour)
        current_metrics = self.calculate_performance_metrics(model_name, 1)
        
        # Get baseline metrics (last 24 hours, excluding last 1 hour)
        baseline_metrics = self._get_baseline_metrics(model_name)
        
        if not current_metrics or not baseline_metrics:
            return alerts
        
        # Check accuracy degradation
        if 'accuracy' in current_metrics and 'accuracy' in baseline_metrics:
            accuracy_drop = baseline_metrics['accuracy'] - current_metrics['accuracy']
            if accuracy_drop > self.thresholds['accuracy_drop']:
                alerts.append(PerformanceAlert(
                    alert_type='accuracy_degradation',
                    message=f"Accuracy dropped by {accuracy_drop:.2%} from baseline",
                    severity='high' if accuracy_drop > 0.1 else 'medium',
                    timestamp=datetime.now(),
                    metric_name='accuracy',
                    current_value=current_metrics['accuracy'],
                    threshold=baseline_metrics['accuracy'] - self.thresholds['accuracy_drop'],
                    model_name=model_name
                ))
        
        # Check confidence degradation
        if 'avg_confidence' in current_metrics and 'avg_confidence' in baseline_metrics:
            confidence_drop = baseline_metrics['avg_confidence'] - current_metrics['avg_confidence']
            if confidence_drop > self.thresholds['confidence_drop']:
                alerts.append(PerformanceAlert(
                    alert_type='confidence_degradation',
                    message=f"Average confidence dropped by {confidence_drop:.2%} from baseline",
                    severity='medium',
                    timestamp=datetime.now(),
                    metric_name='avg_confidence',
                    current_value=current_metrics['avg_confidence'],
                    threshold=baseline_metrics['avg_confidence'] - self.thresholds['confidence_drop'],
                    model_name=model_name
                ))
        
        # Check latency increase
        if 'avg_latency_ms' in current_metrics and 'avg_latency_ms' in baseline_metrics:
            latency_ratio = current_metrics['avg_latency_ms'] / baseline_metrics['avg_latency_ms']
            if latency_ratio > self.thresholds['latency_increase']:
                alerts.append(PerformanceAlert(
                    alert_type='latency_increase',
                    message=f"Average latency increased by {latency_ratio:.1f}x from baseline",
                    severity='low' if latency_ratio < 3 else 'medium',
                    timestamp=datetime.now(),
                    metric_name='avg_latency_ms',
                    current_value=current_metrics['avg_latency_ms'],
                    threshold=baseline_metrics['avg_latency_ms'] * self.thresholds['latency_increase'],
                    model_name=model_name
                ))
        
        # Check error rate increase
        if 'error_rate' in current_metrics and 'error_rate' in baseline_metrics:
            error_rate_increase = current_metrics['error_rate'] - baseline_metrics['error_rate']
            if error_rate_increase > self.thresholds['error_rate_increase']:
                alerts.append(PerformanceAlert(
                    alert_type='error_rate_increase',
                    message=f"Error rate increased by {error_rate_increase:.2%} from baseline",
                    severity='high',
                    timestamp=datetime.now(),
                    metric_name='error_rate',
                    current_value=current_metrics['error_rate'],
                    threshold=baseline_metrics['error_rate'] + self.thresholds['error_rate_increase'],
                    model_name=model_name
                ))
        
        # Process alerts
        for alert in alerts:
            self._process_alert(alert)
        
        return alerts
    
    def _get_baseline_metrics(self, model_name: str) -> Dict[str, float]:
        """Get baseline metrics for comparison"""
        # Use metrics from 2-26 hours ago as baseline (excluding last 1 hour)
        end_time = datetime.now() - timedelta(hours=1)
        start_time = end_time - timedelta(hours=24)
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT * FROM predictions 
                WHERE model_name = ? AND timestamp >= ? AND timestamp <= ?
            ''', conn, params=(model_name, start_time, end_time))
        
        if df.empty:
            return {}
        
        baseline = {}
        
        # Calculate baseline metrics
        accurate_predictions = df[df['is_correct'].notna()]
        if not accurate_predictions.empty:
            baseline['accuracy'] = accurate_predictions['is_correct'].mean()
        
        baseline['avg_confidence'] = df['confidence'].mean()
        baseline['avg_latency_ms'] = df['latency_ms'].mean()
        
        error_predictions = df[df['predicted_class'] == 'Error']
        baseline['error_rate'] = len(error_predictions) / len(df)
        
        return baseline
    
    def _process_alert(self, alert: PerformanceAlert):
        """Process and potentially send an alert"""
        # Check cooldown period
        if self._is_in_cooldown(alert):
            return
        
        # Store alert in database
        self._store_alert(alert)
        
        # Add to alerts list
        self.alerts.append(alert)
        
        # Log alert
        if self.alert_config['log_alerts']:
            self.logger.warning(f"ALERT [{alert.severity.upper()}] {alert.alert_type}: {alert.message}")
        
        # Send email alert if configured
        if self.alert_config['email_enabled'] and alert.severity in ['high', 'critical']:
            self._send_email_alert(alert)
    
    def _is_in_cooldown(self, alert: PerformanceAlert) -> bool:
        """Check if similar alert is in cooldown period"""
        cooldown_minutes = self.alert_config['alert_cooldown_minutes']
        cutoff_time = datetime.now() - timedelta(minutes=cooldown_minutes)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM alerts 
                WHERE alert_type = ? AND model_name = ? AND timestamp > ?
            ''', (alert.alert_type, alert.model_name, cutoff_time))
            
            count = cursor.fetchone()[0]
            return count > 0
    
    def _store_alert(self, alert: PerformanceAlert):
        """Store alert in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alerts 
                (alert_type, severity, message, model_name, metric_name, 
                 current_value, threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_type, alert.severity, alert.message,
                alert.model_name, alert.metric_name, 
                alert.current_value, alert.threshold
            ))
            conn.commit()
    
    def _send_email_alert(self, alert: PerformanceAlert):
        """Send email alert"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.alert_config['email_username']
            msg['To'] = ', '.join(self.alert_config['email_recipients'])
            msg['Subject'] = f"Model Performance Alert: {alert.alert_type}"
            
            body = f"""
            Alert: {alert.alert_type}
            Severity: {alert.severity.upper()}
            Model: {alert.model_name}
            
            Message: {alert.message}
            
            Metric: {alert.metric_name}
            Current Value: {alert.current_value:.4f}
            Threshold: {alert.threshold:.4f}
            
            Timestamp: {alert.timestamp}
            
            Please investigate and take appropriate action.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.alert_config['email_smtp_server'], 
                                self.alert_config['email_smtp_port'])
            server.starttls()
            server.login(self.alert_config['email_username'], 
                        self.alert_config['email_password'])
            
            text = msg.as_string()
            server.sendmail(self.alert_config['email_username'], 
                          self.alert_config['email_recipients'], text)
            server.quit()
            
            self.logger.info(f"Email alert sent for {alert.alert_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")
    
    def start_monitoring(self, check_interval_minutes: int = 10):
        """Start continuous monitoring"""
        if self.is_monitoring:
            self.logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval_minutes,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info(f"Started monitoring with {check_interval_minutes}-minute intervals")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        self.logger.info("Stopped monitoring")
    
    def _monitoring_loop(self, check_interval_minutes: int):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Get all unique model names from recent predictions
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT DISTINCT model_name FROM predictions 
                        WHERE timestamp > datetime('now', '-1 day')
                    ''')
                    model_names = [row[0] for row in cursor.fetchall()]
                
                # Check each model for performance issues
                for model_name in model_names:
                    self.detect_performance_degradation(model_name)
                
                # Sleep until next check
                time.sleep(check_interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def generate_performance_report(self, model_name: str, 
                                  time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        metrics = self.calculate_performance_metrics(model_name, time_window_hours)
        
        # Get recent alerts
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        with sqlite3.connect(self.db_path) as conn:
            alerts_df = pd.read_sql_query('''
                SELECT * FROM alerts 
                WHERE model_name = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            ''', conn, params=(model_name, cutoff_time))
        
        # Performance summary
        performance_status = "Good"
        if not alerts_df.empty:
            high_severity_alerts = alerts_df[alerts_df['severity'].isin(['high', 'critical'])]
            if not high_severity_alerts.empty:
                performance_status = "Critical"
            elif len(alerts_df) > 0:
                performance_status = "Warning"
        
        report = {
            'model_name': model_name,
            'time_window_hours': time_window_hours,
            'performance_status': performance_status,
            'metrics': metrics,
            'alerts': alerts_df.to_dict('records') if not alerts_df.empty else [],
            'recommendations': self._generate_recommendations(metrics, alerts_df),
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def _generate_recommendations(self, metrics: Dict, alerts_df: pd.DataFrame) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Accuracy recommendations
        accuracy = metrics.get('accuracy')
        if accuracy is not None:
            if accuracy < 0.8:
                recommendations.append("Consider retraining the model with additional data")
            if accuracy < 0.9 and metrics.get('total_labeled_predictions', 0) < 100:
                recommendations.append("Collect more labeled data for better accuracy assessment")
        
        # Confidence recommendations
        avg_confidence = metrics.get('avg_confidence', 0)
        if avg_confidence < 0.7:
            recommendations.append("Low average confidence detected - consider model calibration")
        
        confidence_std = metrics.get('confidence_std', 0)
        if confidence_std > 0.3:
            recommendations.append("High confidence variance - investigate prediction consistency")
        
        # Latency recommendations
        avg_latency = metrics.get('avg_latency_ms', 0)
        if avg_latency > 1000:  # 1 second
            recommendations.append("High latency detected - optimize model inference")
        
        p99_latency = metrics.get('p99_latency_ms', 0)
        if p99_latency > 5000:  # 5 seconds
            recommendations.append("Very high P99 latency - investigate performance bottlenecks")
        
        # Error rate recommendations
        error_rate = metrics.get('error_rate', 0)
        if error_rate > 0.05:  # 5%
            recommendations.append("High error rate - investigate input validation and error handling")
        
        # Alert-based recommendations
        if not alerts_df.empty:
            accuracy_alerts = alerts_df[alerts_df['alert_type'] == 'accuracy_degradation']
            if not accuracy_alerts.empty:
                recommendations.append("Accuracy degradation detected - investigate concept drift")
            
            latency_alerts = alerts_df[alerts_df['alert_type'] == 'latency_increase']
            if not latency_alerts.empty:
                recommendations.append("Latency increase detected - check system resources and model complexity")
        
        # Volume recommendations
        predictions_per_hour = metrics.get('predictions_per_hour', 0)
        if predictions_per_hour > 1000:
            recommendations.append("High prediction volume - consider caching and load balancing")
        
        if not recommendations:
            recommendations.append("Model performance is within acceptable ranges")
        
        return recommendations

    def export_metrics(self, filepath: str, model_name: str = None, 
                      time_window_hours: int = 24):
        """Export performance metrics to file"""
        with sqlite3.connect(self.db_path) as conn:
            if model_name:
                query = '''
                    SELECT * FROM performance_metrics 
                    WHERE model_name = ? AND timestamp >= datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                '''.format(time_window_hours)
                df = pd.read_sql_query(query, conn, params=(model_name,))
            else:
                query = '''
                    SELECT * FROM performance_metrics 
                    WHERE timestamp >= datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                '''.format(time_window_hours)
                df = pd.read_sql_query(query, conn)
        
        if filepath.endswith('.csv'):
            df.to_csv(filepath, index=False)
        elif filepath.endswith('.json'):
            df.to_json(filepath, orient='records', indent=2)
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")
        
        self.logger.info(f"Metrics exported to {filepath}")

if __name__ == "__main__":
    # Example usage
    monitor = ModelPerformanceMonitor()
    
    # Simulate some predictions
    import random
    model_name = "lstm_classifier"
    
    for i in range(100):
        confidence = random.uniform(0.5, 1.0)
        latency = random.uniform(50, 200)
        actual_class = random.choice(['Billing', 'Technical Issue', 'Feature Request'])
        predicted_class = actual_class if random.random() > 0.1 else random.choice(['Billing', 'Technical Issue'])
        
        monitor.log_prediction(
            model_name=model_name,
            input_text=f"Sample ticket {i}",
            predicted_class=predicted_class,
            confidence=confidence,
            latency_ms=latency,
            actual_class=actual_class
        )
    
    # Calculate metrics
    metrics = monitor.calculate_performance_metrics(model_name)
    print(f"Performance metrics: {metrics}")
    
    # Check for degradation
    alerts = monitor.detect_performance_degradation(model_name)
    print(f"Generated {len(alerts)} alerts")
    
    # Generate report
    report = monitor.generate_performance_report(model_name)
    print(f"Performance status: {report['performance_status']}")
    print(f"Recommendations: {report['recommendations']}")
