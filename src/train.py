"""
Training script for customer support ticket classification models.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_preprocessing import DataLoader, TextPreprocessor
from model_builder import get_model_builder, ModelTrainer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ticket classification model')
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data CSV file')
    parser.add_argument('--model_type', type=str, default='lstm',
                       choices=['lstm', 'cnn', 'bert', 'random_forest', 'svm', 'logistic_regression'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (for deep learning models)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--max_features', type=int, default=10000,
                       help='Maximum number of features')
    parser.add_argument('--max_len', type=int, default=100,
                       help='Maximum sequence length')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of test data')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Proportion of validation data')    
    parser.add_argument('--output_dir', type=str, default='models/saved_models',
                       help='Directory to save trained models')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save training plots')
    
    return parser.parse_args()

def save_training_plots(history: Dict[str, Any], output_dir: str, model_name: str):
    """
    Save training history plots.
    
    Args:
        history: Training history
        output_dir: Output directory
        model_name: Name of the model
    """
    if not history:
        return
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot training history
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    axes[0].plot(history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        axes[1].plot(history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_training_history.png'))
    plt.close()

def save_evaluation_plots(y_true: np.ndarray, y_pred: np.ndarray, 
                         label_mapping: Dict[int, str], output_dir: str, 
                         model_name: str):
    """
    Save evaluation plots.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_mapping: Label mapping
        output_dir: Output directory
        model_name: Name of the model
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(label_mapping.values()),
                yticklabels=list(label_mapping.values()))
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

def save_model_info(model_info: Dict[str, Any], output_dir: str, model_name: str):
    """
    Save model information and metrics.
    
    Args:
        model_info: Model information dictionary
        output_dir: Output directory
        model_name: Name of the model
    """
    info_file = os.path.join(output_dir, f'{model_name}_info.json')
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_info = {}
    for key, value in model_info.items():
        if isinstance(value, np.ndarray):
            serializable_info[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_info[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                    for k, v in value.items()}
        else:
            serializable_info[key] = value
    
    with open(info_file, 'w') as f:
        json.dump(serializable_info, f, indent=2, default=str)

def train_model(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    print("Starting model training...")
    print(f"Model type: {args.model_type}")
    print(f"Data path: {args.data_path}")
    
    # Initialize components
    preprocessor = TextPreprocessor(max_features=args.max_features, max_len=args.max_len)
    data_loader = DataLoader(preprocessor)
    
    # Load and prepare data
    print("Loading and preprocessing data...")
    try:
        df = data_loader.load_data(args.data_path)
        print(f"Loaded {len(df)} tickets")
        
        # Display class distribution
        print("\nClass distribution:")
        print(df['category'].value_counts())
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Determine vectorization method
    vectorization = 'sequences' if args.model_type in ['lstm', 'cnn', 'bert'] else 'tfidf'
    
    # Prepare data splits
    try:
        data_splits = data_loader.prepare_data(
            df, 
            test_size=args.test_size,
            val_size=args.val_size,
            vectorization=vectorization
        )
        
        print(f"Training samples: {len(data_splits['X_train'])}")
        print(f"Validation samples: {len(data_splits['X_val'])}")
        print(f"Test samples: {len(data_splits['X_test'])}")
        print(f"Number of classes: {data_splits['num_classes']}")
        
    except Exception as e:
        print(f"Error preparing data: {e}")
        return
      # Build model
    print(f"Building {args.model_type} model...")
    try:
        if args.model_type in ['lstm', 'cnn', 'bert']:
            model_builder = get_model_builder(
                model_type=args.model_type,
                vocab_size=data_splits['vocab_size'],
                num_classes=data_splits['num_classes'],
                max_length=data_splits['max_length']
            )
            model = model_builder.build_model()
            # Build the model by passing dummy input
            if hasattr(model, 'build'):
                model.build(input_shape=(None, data_splits['max_length']))
            print(f"Model built with {model.count_params()} parameters")
            
        else:
            model_builder = get_model_builder(model_type=args.model_type)
            model = model_builder.build_model()
            print(f"Model built: {type(model)}")
            
    except Exception as e:
        print(f"Error building model: {e}")
        return
    
    # Train model
    print("Training model...")
    trainer = ModelTrainer(model_type=args.model_type)
    
    try:
        if args.model_type in ['lstm', 'cnn', 'bert']:
            # Deep learning training
            history = trainer.train_deep_learning_model(
                model=model,
                x_train=data_splits['X_train'],
                y_train=data_splits['y_train'],
                x_val=data_splits['X_val'],
                y_val=data_splits['y_val'],
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            
        else:
            # Traditional ML training
            model_builder.train(data_splits['X_train'], data_splits['y_train'])
            history = None
            trainer.model = model
            
        print("Training completed!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # Evaluate model
    print("Evaluating model...")
    try:
        eval_results = trainer.evaluate_model(
            x_test=data_splits['X_test'],
            y_test=data_splits['y_test'],
            label_mapping=data_splits['label_mapping']
        )
        
        print(f"Test Accuracy: {eval_results['accuracy']:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            data_splits['y_test'], 
            eval_results['predictions'],
            target_names=list(data_splits['label_mapping'].values())
        ))
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return
    
    # Save model and results
    print("Saving model and results...")
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate model name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{args.model_type}_{timestamp}"
        
        # Save model
        model_path = os.path.join(args.output_dir, f"{model_name}.h5")
        if args.model_type in ['lstm', 'cnn', 'bert']:
            model_path = os.path.join(args.output_dir, f"{model_name}.h5")
        else:
            model_path = os.path.join(args.output_dir, f"{model_name}.pkl")
        
        trainer.save_model(model_path)
        print(f"Model saved to: {model_path}")
        
        # Save preprocessor
        preprocessor_path = os.path.join(args.output_dir, f"{model_name}_preprocessor.pkl")
        data_loader.save_preprocessor(preprocessor_path)
        print(f"Preprocessor saved to: {preprocessor_path}")
        
        # Save model information
        model_info = {
            'model_type': args.model_type,
            'model_name': model_name,
            'timestamp': timestamp,
            'test_accuracy': eval_results['accuracy'],
            'classification_report': eval_results['classification_report'],
            'label_mapping': data_splits['label_mapping'],
            'num_classes': data_splits['num_classes'],
            'training_samples': len(data_splits['X_train']),
            'test_samples': len(data_splits['X_test']),
            'parameters': {
                'max_features': args.max_features,
                'max_len': args.max_len,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'test_size': args.test_size,
                'val_size': args.val_size
            }
        }
        
        save_model_info(model_info, args.output_dir, model_name)
        print(f"Model info saved to: {args.output_dir}/{model_name}_info.json")
        
        # Save plots if requested
        if args.save_plots:
            if history:
                save_training_plots(history, args.output_dir, model_name)
            
            save_evaluation_plots(
                data_splits['y_test'], 
                eval_results['predictions'],
                data_splits['label_mapping'],
                args.output_dir,
                model_name
            )
            print("Plots saved!")
        
        print("Training completed successfully!")
        print(f"Final test accuracy: {eval_results['accuracy']:.4f}")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        return

if __name__ == "__main__":
    args = parse_arguments()
    train_model(args)
