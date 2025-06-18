"""
Model Versioning and A/B Testing System
Manages multiple model versions and enables A/B testing for model performance comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import os
import logging
from datetime import datetime, timedelta
import hashlib
import pickle
import shutil
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading
import time
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelVersion:
    """Data class for model version information."""
    version_id: str
    model_name: str
    version_number: str
    creation_date: datetime
    model_path: str
    preprocessor_path: str
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    status: str = "active"  # active, deprecated, experimental
    
@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    test_name: str
    model_a: str
    model_b: str
    traffic_split: float  # Percentage to model_b (0.0 to 1.0)
    start_date: datetime
    end_date: Optional[datetime]
    success_metrics: List[str]
    min_sample_size: int
    significance_level: float = 0.05

class ModelVersionManager:
    """
    Advanced model versioning system that:
    - Tracks multiple model versions with metadata
    - Enables A/B testing between models
    - Manages model deployment and rollbacks
    - Monitors model performance over time
    - Provides automated model selection
    """
    
    def __init__(self, 
                 models_dir: str = "models/saved_models",
                 versions_dir: str = "models/versions",
                 metadata_file: str = "models/model_registry.json"):
        """
        Initialize model version manager.
        
        Args:
            models_dir: Directory containing saved models
            versions_dir: Directory for versioned models
            metadata_file: File to store model metadata
        """
        self.models_dir = models_dir
        self.versions_dir = versions_dir
        self.metadata_file = metadata_file
        
        # Create directories
        os.makedirs(self.versions_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        
        # Model registry
        self.model_registry = self._load_registry()
        self.ab_tests = {}
        self.performance_logs = defaultdict(list)
        
        # Current deployments
        self.deployed_models = {}
        self.traffic_router = TrafficRouter()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_registry(self) -> Dict[str, ModelVersion]:
        """Load model registry from file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                registry = {}
                for version_id, version_data in data.items():
                    # Convert datetime strings back to datetime objects
                    version_data['creation_date'] = datetime.fromisoformat(
                        version_data['creation_date']
                    )
                    registry[version_id] = ModelVersion(**version_data)
                
                return registry
            except Exception as e:
                self.logger.error(f"Error loading registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save model registry to file."""
        try:
            data = {}
            for version_id, version in self.model_registry.items():
                version_dict = asdict(version)
                version_dict['creation_date'] = version.creation_date.isoformat()
                data[version_id] = version_dict
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving registry: {e}")
    
    def register_model(self, 
                      model_name: str,
                      model_path: str,
                      preprocessor_path: str,
                      performance_metrics: Dict[str, float],
                      metadata: Dict[str, Any] = None,
                      version_number: str = None) -> str:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model file
            preprocessor_path: Path to the preprocessor file
            performance_metrics: Performance metrics dict
            metadata: Additional metadata
            version_number: Specific version number (auto-generated if None)
            
        Returns:
            Version ID of the registered model
        """
        try:
            # Generate version info
            if version_number is None:
                existing_versions = [
                    v.version_number for v in self.model_registry.values()
                    if v.model_name == model_name
                ]
                version_number = self._generate_version_number(existing_versions)
            
            version_id = self._generate_version_id(model_name, version_number)
            
            # Copy model files to versioned directory
            version_dir = os.path.join(self.versions_dir, version_id)
            os.makedirs(version_dir, exist_ok=True)
            
            versioned_model_path = os.path.join(version_dir, os.path.basename(model_path))
            versioned_preprocessor_path = os.path.join(
                version_dir, os.path.basename(preprocessor_path)
            )
            
            shutil.copy2(model_path, versioned_model_path)
            shutil.copy2(preprocessor_path, versioned_preprocessor_path)
            
            # Create model version
            model_version = ModelVersion(
                version_id=version_id,
                model_name=model_name,
                version_number=version_number,
                creation_date=datetime.now(),
                model_path=versioned_model_path,
                preprocessor_path=versioned_preprocessor_path,
                performance_metrics=performance_metrics,
                metadata=metadata or {},
                status="active"
            )
            
            # Register in registry
            self.model_registry[version_id] = model_version
            self._save_registry()
            
            self.logger.info(f"Registered model version: {version_id}")
            return version_id
            
        except Exception as e:
            self.logger.error(f"Error registering model: {e}")
            raise
    
    def _generate_version_number(self, existing_versions: List[str]) -> str:
        """Generate next version number."""
        if not existing_versions:
            return "1.0.0"
        
        # Parse versions and find the highest
        max_version = [0, 0, 0]
        for version in existing_versions:
            try:
                parts = [int(x) for x in version.split('.')]
                if len(parts) == 3:
                    if parts > max_version:
                        max_version = parts
            except ValueError:
                continue
        
        # Increment patch version
        max_version[2] += 1
        return '.'.join(map(str, max_version))
    
    def _generate_version_id(self, model_name: str, version_number: str) -> str:
        """Generate unique version ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        content = f"{model_name}_{version_number}_{timestamp}"
        hash_object = hashlib.md5(content.encode())
        return f"{model_name}_v{version_number}_{hash_object.hexdigest()[:8]}"
    
    def get_model_versions(self, model_name: str = None) -> List[ModelVersion]:
        """Get all versions of a model or all models."""
        if model_name:
            return [
                version for version in self.model_registry.values()
                if version.model_name == model_name
            ]
        return list(self.model_registry.values())
    
    def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the latest version of a model."""
        versions = self.get_model_versions(model_name)
        if not versions:
            return None
        
        # Sort by creation date
        return max(versions, key=lambda v: v.creation_date)
    
    def get_best_performing_version(self, 
                                   model_name: str,
                                   metric: str = "accuracy") -> Optional[ModelVersion]:
        """Get the best performing version based on a metric."""
        versions = self.get_model_versions(model_name)
        if not versions:
            return None
        
        valid_versions = [
            v for v in versions 
            if metric in v.performance_metrics
        ]
        
        if not valid_versions:
            return None
        
        return max(valid_versions, key=lambda v: v.performance_metrics[metric])
    
    def deploy_model(self, version_id: str, deployment_name: str = "production"):
        """Deploy a model version."""
        if version_id not in self.model_registry:
            raise ValueError(f"Version {version_id} not found in registry")
        
        self.deployed_models[deployment_name] = version_id
        self.logger.info(f"Deployed {version_id} to {deployment_name}")
    
    def rollback_model(self, deployment_name: str = "production"):
        """Rollback to previous model version."""
        if deployment_name not in self.deployed_models:
            raise ValueError(f"No deployment found: {deployment_name}")
        
        current_version_id = self.deployed_models[deployment_name]
        current_version = self.model_registry[current_version_id]
        
        # Find previous version
        model_versions = self.get_model_versions(current_version.model_name)
        model_versions.sort(key=lambda v: v.creation_date, reverse=True)
        
        if len(model_versions) < 2:
            raise ValueError("No previous version available for rollback")
        
        previous_version = model_versions[1]  # Second most recent
        self.deploy_model(previous_version.version_id, deployment_name)
        
        self.logger.info(f"Rolled back from {current_version_id} to {previous_version.version_id}")
    
    def create_ab_test(self, config: ABTestConfig) -> str:
        """Create an A/B test configuration."""
        test_id = f"ab_test_{config.test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate models exist
        if config.model_a not in self.model_registry:
            raise ValueError(f"Model A {config.model_a} not found")
        if config.model_b not in self.model_registry:
            raise ValueError(f"Model B {config.model_b} not found")
        
        self.ab_tests[test_id] = config
        
        # Configure traffic routing
        self.traffic_router.add_ab_test(test_id, config)
        
        self.logger.info(f"Created A/B test: {test_id}")
        return test_id
    
    def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get A/B test results and statistical analysis."""
        if test_id not in self.ab_tests:
            raise ValueError(f"A/B test {test_id} not found")
        
        config = self.ab_tests[test_id]
        
        # Get performance data for both models
        model_a_data = self.performance_logs.get(config.model_a, [])
        model_b_data = self.performance_logs.get(config.model_b, [])
        
        # Filter data by test period
        test_start = config.start_date
        test_end = config.end_date or datetime.now()
        
        model_a_test_data = [
            entry for entry in model_a_data
            if test_start <= entry['timestamp'] <= test_end
        ]
        model_b_test_data = [
            entry for entry in model_b_data
            if test_start <= entry['timestamp'] <= test_end
        ]
        
        results = {
            'test_id': test_id,
            'config': asdict(config),
            'model_a_samples': len(model_a_test_data),
            'model_b_samples': len(model_b_test_data),
            'metrics_comparison': {},
            'statistical_significance': {},
            'recommendation': None
        }
        
        # Compare metrics
        for metric in config.success_metrics:
            model_a_values = [
                entry['metrics'].get(metric, 0) 
                for entry in model_a_test_data 
                if 'metrics' in entry
            ]
            model_b_values = [
                entry['metrics'].get(metric, 0) 
                for entry in model_b_test_data 
                if 'metrics' in entry
            ]
            
            if model_a_values and model_b_values:
                results['metrics_comparison'][metric] = {
                    'model_a_mean': np.mean(model_a_values),
                    'model_a_std': np.std(model_a_values),
                    'model_b_mean': np.mean(model_b_values),
                    'model_b_std': np.std(model_b_values),
                    'improvement': (np.mean(model_b_values) - np.mean(model_a_values)) / np.mean(model_a_values) * 100
                }
                
                # Statistical significance test
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(model_a_values, model_b_values)
                results['statistical_significance'][metric] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < config.significance_level
                }
        
        # Generate recommendation
        results['recommendation'] = self._generate_ab_recommendation(results, config)
        
        return results
    
    def _generate_ab_recommendation(self, results: Dict[str, Any], 
                                   config: ABTestConfig) -> str:
        """Generate recommendation based on A/B test results."""
        significant_improvements = 0
        significant_degradations = 0
        
        for metric in config.success_metrics:
            if metric in results['statistical_significance']:
                sig_result = results['statistical_significance'][metric]
                metric_comparison = results['metrics_comparison'][metric]
                
                if sig_result['significant']:
                    if metric_comparison['improvement'] > 0:
                        significant_improvements += 1
                    else:
                        significant_degradations += 1
        
        sample_size_adequate = (
            results['model_a_samples'] >= config.min_sample_size and
            results['model_b_samples'] >= config.min_sample_size
        )
        
        if not sample_size_adequate:
            return "Insufficient sample size - continue test"
        
        if significant_improvements > significant_degradations:
            return "Deploy Model B - shows significant improvement"
        elif significant_degradations > significant_improvements:
            return "Keep Model A - Model B shows significant degradation"
        else:
            return "No clear winner - consider business factors or extend test"
    
    def log_prediction_performance(self, 
                                  model_version_id: str,
                                  metrics: Dict[str, float],
                                  metadata: Dict[str, Any] = None):
        """Log prediction performance for a model version."""
        log_entry = {
            'timestamp': datetime.now(),
            'model_version_id': model_version_id,
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        self.performance_logs[model_version_id].append(log_entry)
    
    def get_performance_trend(self, 
                             model_version_id: str,
                             metric: str,
                             days: int = 30) -> List[Dict[str, Any]]:
        """Get performance trend for a model version."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        logs = self.performance_logs.get(model_version_id, [])
        recent_logs = [
            log for log in logs 
            if log['timestamp'] > cutoff_date and metric in log['metrics']
        ]
        
        return [
            {
                'timestamp': log['timestamp'].isoformat(),
                'value': log['metrics'][metric]
            }
            for log in recent_logs
        ]
    
    def get_model_comparison(self, 
                           model_versions: List[str],
                           metrics: List[str]) -> Dict[str, Any]:
        """Compare multiple model versions across metrics."""
        comparison = {
            'models': model_versions,
            'metrics': metrics,
            'comparison_data': {},
            'summary': {}
        }
        
        for version_id in model_versions:
            if version_id in self.model_registry:
                version = self.model_registry[version_id]
                comparison['comparison_data'][version_id] = {
                    'version_info': asdict(version),
                    'recent_performance': {}
                }
                
                # Get recent performance
                for metric in metrics:
                    trend_data = self.get_performance_trend(version_id, metric, days=7)
                    if trend_data:
                        values = [entry['value'] for entry in trend_data]
                        comparison['comparison_data'][version_id]['recent_performance'][metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'trend': 'improving' if len(values) > 1 and values[-1] > values[0] else 'stable'
                        }
        
        # Generate summary
        for metric in metrics:
            metric_values = {}
            for version_id in model_versions:
                perf_data = comparison['comparison_data'].get(version_id, {}).get('recent_performance', {})
                if metric in perf_data:
                    metric_values[version_id] = perf_data[metric]['mean']
            
            if metric_values:
                best_model = max(metric_values.items(), key=lambda x: x[1])
                comparison['summary'][metric] = {
                    'best_model': best_model[0],
                    'best_value': best_model[1],
                    'all_values': metric_values
                }
        
        return comparison

class TrafficRouter:
    """Routes traffic between models for A/B testing."""
    
    def __init__(self):
        self.ab_tests = {}
        self.routing_decisions = defaultdict(list)
    
    def add_ab_test(self, test_id: str, config: ABTestConfig):
        """Add A/B test configuration."""
        self.ab_tests[test_id] = config
    
    def route_request(self, request_id: str = None) -> str:
        """Route a request to appropriate model based on A/B test config."""
        if not self.ab_tests:
            return "default"
        
        # Use hash of request_id for consistent routing
        if request_id:
            hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
            random_value = (hash_value % 100) / 100.0
        else:
            random_value = np.random.random()
        
        # Apply A/B test routing (using the first active test for simplicity)
        for test_id, config in self.ab_tests.items():
            if config.start_date <= datetime.now():
                if config.end_date is None or datetime.now() <= config.end_date:
                    model_choice = config.model_b if random_value < config.traffic_split else config.model_a
                    
                    # Log routing decision
                    self.routing_decisions[test_id].append({
                        'timestamp': datetime.now(),
                        'request_id': request_id,
                        'model_chosen': model_choice,
                        'random_value': random_value
                    })
                    
                    return model_choice
        
        return "default"
    
    def get_routing_stats(self, test_id: str) -> Dict[str, Any]:
        """Get routing statistics for an A/B test."""
        if test_id not in self.routing_decisions:
            return {'error': 'No routing data found'}
        
        decisions = self.routing_decisions[test_id]
        config = self.ab_tests.get(test_id)
        
        if not config:
            return {'error': 'Test configuration not found'}
        
        model_a_count = sum(1 for d in decisions if d['model_chosen'] == config.model_a)
        model_b_count = sum(1 for d in decisions if d['model_chosen'] == config.model_b)
        total = len(decisions)
        
        return {
            'total_requests': total,
            'model_a_count': model_a_count,
            'model_b_count': model_b_count,
            'model_a_percentage': (model_a_count / total * 100) if total > 0 else 0,
            'model_b_percentage': (model_b_count / total * 100) if total > 0 else 0,
            'expected_split': config.traffic_split * 100,
            'actual_split': (model_b_count / total * 100) if total > 0 else 0
        }
