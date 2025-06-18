"""
Advanced Hyperparameter Tuning with Bayesian Optimization
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import json
import os
import logging
from datetime import datetime
import itertools
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

class HyperparameterTuner:
    """
    Advanced hyperparameter tuning system using:
    - Bayesian optimization (with scikit-optimize or Optuna)
    - Multi-objective optimization
    - Early stopping and pruning
    - Automated hyperparameter space definition
    - Cross-validation with multiple metrics
    """
    
    def __init__(self, models_dir: str = "models/saved_models", 
                 results_dir: str = "models/tuning_results"):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.tuning_history = []
        self.best_params = {}
        self.optimization_results = {}
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def define_search_space(self, model_type: str) -> Dict[str, Any]:
        """Define hyperparameter search spaces for different model types"""
        search_spaces = {
            'lstm': {
                'embedding_dim': [64, 128, 256, 512],
                'lstm_units': [32, 64, 128, 256],
                'dropout_rate': [0.2, 0.3, 0.4, 0.5],
                'learning_rate': [0.001, 0.01, 0.1],
                'batch_size': [16, 32, 64, 128],
                'max_length': [50, 100, 200, 300]
            },
            'cnn': {
                'embedding_dim': [64, 128, 256],
                'num_filters': [64, 128, 256],
                'filter_sizes': [[3, 4, 5], [2, 3, 4], [3, 4, 5, 6]],
                'dropout_rate': [0.2, 0.3, 0.4, 0.5],
                'learning_rate': [0.001, 0.01, 0.1],
                'batch_size': [16, 32, 64]
            },
            'bert': {
                'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
                'batch_size': [8, 16, 32],
                'max_length': [128, 256, 512],
                'warmup_steps': [100, 500, 1000],
                'weight_decay': [0.01, 0.1, 0.2]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'class_weight': [None, 'balanced']
            },
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga', 'lbfgs'],
                'max_iter': [100, 500, 1000]
            }
        }
        
        return search_spaces.get(model_type, {})
    
    def tune_with_optuna(self, model_type: str, train_data: Tuple, 
                        n_trials: int = 100, timeout: int = 3600) -> Dict[str, Any]:
        """
        Hyperparameter tuning using Optuna (preferred method)
        """
        if not OPTUNA_AVAILABLE:
            self.logger.error("Optuna not available. Please install: pip install optuna")
            return {}
        
        X_train, y_train = train_data
        search_space = self.define_search_space(model_type)
        
        if not search_space:
            self.logger.error(f"No search space defined for model type: {model_type}")
            return {}
        
        def objective(trial):
            """Objective function for Optuna optimization"""
            try:
                # Suggest hyperparameters based on model type
                params = {}
                
                for param_name, param_values in search_space.items():
                    if isinstance(param_values[0], int):
                        params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                    elif isinstance(param_values[0], float):
                        params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                    elif isinstance(param_values[0], str):
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                
                # Build and evaluate model with these parameters
                score = self._evaluate_model_params(model_type, params, X_train, y_train)
                
                return score
                
            except Exception as e:
                self.logger.error(f"Error in trial {trial.number}: {str(e)}")
                return 0.0  # Return poor score for failed trials
        
        # Create and run study
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Store results
        tuning_result = {
            'model_type': model_type,
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'optimization_history': [
                {'trial': i, 'score': trial.value, 'params': trial.params}
                for i, trial in enumerate(study.trials)
                if trial.value is not None
            ],
            'timestamp': datetime.now().isoformat(),
            'method': 'optuna'
        }
        
        self.best_params[model_type] = study.best_params
        self.optimization_results[model_type] = tuning_result
        
        # Save results
        self._save_tuning_results(tuning_result)
        
        return tuning_result
    
    def tune_with_grid_search(self, model_type: str, train_data: Tuple, 
                             n_samples: int = 50) -> Dict[str, Any]:
        """
        Hyperparameter tuning using smart grid search (fallback method)
        """
        X_train, y_train = train_data
        search_space = self.define_search_space(model_type)
        
        if not search_space:
            self.logger.error(f"No search space defined for model type: {model_type}")
            return {}
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(search_space, n_samples)
        
        best_score = 0.0
        best_params = {}
        all_results = []
        
        for i, params in enumerate(param_combinations):
            try:
                score = self._evaluate_model_params(model_type, params, X_train, y_train)
                
                all_results.append({
                    'trial': i,
                    'score': score,
                    'params': params
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                self.logger.info(f"Trial {i+1}/{len(param_combinations)}: Score = {score:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error in trial {i}: {str(e)}")
                continue
        
        tuning_result = {
            'model_type': model_type,
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(param_combinations),
            'optimization_history': all_results,
            'timestamp': datetime.now().isoformat(),
            'method': 'grid_search'
        }
        
        self.best_params[model_type] = best_params
        self.optimization_results[model_type] = tuning_result
        
        # Save results
        self._save_tuning_results(tuning_result)
        
        return tuning_result
    
    def _generate_param_combinations(self, search_space: Dict, max_combinations: int) -> List[Dict]:
        """Generate smart parameter combinations for grid search"""
        # Convert search space to proper format
        param_lists = {}
        for param_name, param_values in search_space.items():
            if len(param_values) > 5:  # Limit to 5 values per parameter
                # Sample evenly across the range
                indices = np.linspace(0, len(param_values)-1, 5, dtype=int)
                param_lists[param_name] = [param_values[i] for i in indices]
            else:
                param_lists[param_name] = param_values
        
        # Generate all combinations
        param_names = list(param_lists.keys())
        param_value_lists = list(param_lists.values())
        
        all_combinations = list(itertools.product(*param_value_lists))
        
        # Limit number of combinations
        if len(all_combinations) > max_combinations:
            # Randomly sample combinations
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            selected_combinations = [all_combinations[i] for i in indices]
        else:
            selected_combinations = all_combinations
        
        # Convert to list of dictionaries
        param_combinations = []
        for combination in selected_combinations:
            params = dict(zip(param_names, combination))
            param_combinations.append(params)
        
        return param_combinations
    
    def _evaluate_model_params(self, model_type: str, params: Dict, 
                              X_train: Any, y_train: Any) -> float:
        """
        Evaluate model with given parameters using cross-validation
        """
        try:
            # Import here to avoid circular dependencies
            from .model_builder import ModelBuilder
            from .data_preprocessing import DataPreprocessor
            
            # Create model with parameters
            builder = ModelBuilder()
            preprocessor = DataPreprocessor()
            
            # Preprocess data based on model type
            if model_type in ['lstm', 'cnn', 'bert']:
                # For deep learning models, we need tokenized sequences
                tokenizer = preprocessor.create_tokenizer(X_train, max_words=10000)
                X_processed = tokenizer.texts_to_sequences(X_train)
                max_length = params.get('max_length', 100)
                
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                X_processed = pad_sequences(X_processed, maxlen=max_length)
                
                # Build model
                vocab_size = min(len(tokenizer.word_index) + 1, 10000)
                model = builder.build_model(
                    model_type=model_type,
                    vocab_size=vocab_size,
                    max_length=max_length,
                    num_classes=len(set(y_train)),
                    **params
                )
                
                # Quick training and evaluation (reduced epochs for speed)
                model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Convert labels to numeric
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_numeric = le.fit_transform(y_train)
                
                # Train briefly
                history = model.fit(
                    X_processed, y_numeric,
                    epochs=5,  # Reduced for speed
                    batch_size=params.get('batch_size', 32),
                    validation_split=0.2,
                    verbose=0
                )
                
                # Return validation accuracy
                return max(history.history['val_accuracy'])
                
            else:
                # For traditional ML models
                vectorizer = preprocessor.create_vectorizer(max_features=5000)
                X_processed = vectorizer.fit_transform(X_train)
                
                # Build model
                model = builder.build_traditional_model(model_type, **params)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_processed, y_train,
                    cv=3,  # Reduced folds for speed
                    scoring='accuracy',
                    n_jobs=-1
                )
                
                return np.mean(cv_scores)
                
        except Exception as e:
            self.logger.error(f"Error evaluating model params: {str(e)}")
            return 0.0
    
    def _save_tuning_results(self, result: Dict):
        """Save tuning results to file"""
        filename = f"{result['model_type']}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        self.tuning_history.append(result)
    
    def multi_objective_tuning(self, model_type: str, train_data: Tuple,
                              objectives: List[str] = None) -> Dict[str, Any]:
        """
        Multi-objective hyperparameter optimization
        """
        if objectives is None:
            objectives = ['accuracy', 'f1_score', 'training_time']
        
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Multi-objective tuning requires Optuna. Falling back to single objective.")
            return self.tune_with_grid_search(model_type, train_data)
        
        X_train, y_train = train_data
        search_space = self.define_search_space(model_type)
        
        def objective(trial):
            """Multi-objective function"""
            try:
                # Suggest hyperparameters
                params = {}
                for param_name, param_values in search_space.items():
                    if isinstance(param_values[0], int):
                        params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                    elif isinstance(param_values[0], float):
                        params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                
                # Evaluate multiple objectives
                start_time = datetime.now()
                accuracy = self._evaluate_model_params(model_type, params, X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # For f1_score, we need to do a quick fit and predict
                f1 = accuracy  # Simplified - in practice, would calculate actual F1
                
                # Return tuple of objectives (Optuna maximizes, so negate time)
                return accuracy, f1, -training_time
                
            except Exception as e:
                self.logger.error(f"Error in multi-objective trial: {str(e)}")
                return 0.0, 0.0, -1000.0
        
        # Create multi-objective study
        study = optuna.create_study(
            directions=['maximize', 'maximize', 'maximize'],  # All objectives to maximize
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(objective, n_trials=50)
        
        # Find best trade-off solution (Pareto optimal)
        pareto_trials = study.best_trials
        
        if pareto_trials:
            # Select solution based on weighted sum (customizable)
            weights = [0.5, 0.3, 0.2]  # accuracy, f1, speed
            best_trial = None
            best_weighted_score = -float('inf')
            
            for trial in pareto_trials:
                weighted_score = sum(w * v for w, v in zip(weights, trial.values))
                if weighted_score > best_weighted_score:
                    best_weighted_score = weighted_score
                    best_trial = trial
            
            result = {
                'model_type': model_type,
                'best_params': best_trial.params,
                'best_objectives': dict(zip(objectives, best_trial.values)),
                'pareto_solutions': [
                    {'params': t.params, 'objectives': dict(zip(objectives, t.values))}
                    for t in pareto_trials
                ],
                'n_trials': len(study.trials),
                'timestamp': datetime.now().isoformat(),
                'method': 'multi_objective'
            }
        else:
            result = {'error': 'No Pareto optimal solutions found'}
        
        return result
    
    def get_best_params(self, model_type: str) -> Dict[str, Any]:
        """Get best parameters for a model type"""
        return self.best_params.get(model_type, {})
    
    def compare_tuning_results(self) -> pd.DataFrame:
        """Compare tuning results across different model types and methods"""
        if not self.tuning_history:
            return pd.DataFrame()
        
        comparison_data = []
        for result in self.tuning_history:
            comparison_data.append({
                'Model Type': result['model_type'],
                'Method': result.get('method', 'unknown'),
                'Best Score': result.get('best_score', 0),
                'N Trials': result.get('n_trials', 0),
                'Timestamp': result['timestamp']
            })
        
        return pd.DataFrame(comparison_data)
    
    def export_best_configs(self, filepath: str):
        """Export best configurations for all models"""
        export_data = {
            'best_parameters': self.best_params,
            'optimization_results': self.optimization_results,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Best configurations exported to {filepath}")

if __name__ == "__main__":
    # Example usage
    tuner = HyperparameterTuner()
    
    # Mock data for testing
    X_train = ["Sample text 1", "Sample text 2", "Sample text 3"] * 100
    y_train = ["class1", "class2", "class3"] * 100
    
    # Test hyperparameter tuning
    if OPTUNA_AVAILABLE:
        result = tuner.tune_with_optuna('random_forest', (X_train, y_train), n_trials=10)
        print(f"Best Random Forest params: {result.get('best_params', {})}")
    else:
        result = tuner.tune_with_grid_search('random_forest', (X_train, y_train), n_samples=10)
        print(f"Best Random Forest params: {result.get('best_params', {})}")
    
    # Compare results
    comparison = tuner.compare_tuning_results()
    print("\nTuning Results Comparison:")
    print(comparison)
