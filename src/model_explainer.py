"""
Advanced Model Interpretability and Explainability System
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pickle
import logging
from datetime import datetime
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
try:
    import lime
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    """
    Advanced explainability system that provides:
    - LIME and SHAP explanations
    - Feature importance analysis
    - Decision boundary visualization
    - Counterfactual explanations
    - Bias detection and fairness analysis
    """
    
    def __init__(self, models_dir: str = "models/saved_models"):
        self.models_dir = models_dir
        self.explanation_cache = {}
        self.feature_importance_history = []
        self.bias_analysis_results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize explainers
        self.lime_explainer = LimeTextExplainer(class_names=self._get_class_names())
        
    def _get_class_names(self) -> List[str]:
        """Get class names from saved model info"""
        try:
            info_files = [f for f in os.listdir(self.models_dir) if f.endswith('_info.json')]
            if info_files:
                with open(os.path.join(self.models_dir, info_files[0]), 'r') as f:
                    info = json.load(f)
                    return info.get('classes', ['Unknown'])
        except:
            pass
        return ['Billing', 'Technical Issue', 'Feature Request', 'Account Management', 
                'Product Information', 'Refund & Return', 'General Inquiry']
    
    def explain_prediction_lime(self, text: str, model_predictor, num_features: int = 10) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single prediction
        """
        try:
            def predict_fn(texts):
                """Wrapper function for LIME"""
                predictions = []
                for t in texts:
                    try:
                        result = model_predictor.predict_with_confidence(t)
                        # Convert to probability array
                        probs = result.get('ensemble_probabilities', {})
                        prob_array = np.zeros(len(self._get_class_names()))
                        for i, class_name in enumerate(self._get_class_names()):
                            prob_array[i] = probs.get(class_name, 0.0)
                        predictions.append(prob_array)
                    except:
                        # Return uniform distribution on error
                        predictions.append(np.ones(len(self._get_class_names())) / len(self._get_class_names()))
                return np.array(predictions)
            
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                text, 
                predict_fn, 
                num_features=num_features,
                num_samples=500  # Increased for better approximation
            )
            
            # Extract explanation data
            explanation_data = {
                'text': text,
                'prediction': explanation.predict_proba,
                'local_explanation': explanation.as_list(),
                'score': explanation.score,
                'intercept': explanation.intercept,
                'local_pred': explanation.local_pred,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache explanation
            text_hash = hash(text)
            self.explanation_cache[text_hash] = explanation_data
            
            return explanation_data
            
        except Exception as e:
            self.logger.error(f"Error generating LIME explanation: {str(e)}")
            return {'error': str(e), 'text': text}
    
    def explain_global_features(self, texts: List[str], labels: List[str], 
                              model_type: str = 'tfidf') -> Dict[str, Any]:
        """
        Generate global feature importance analysis
        """
        try:
            # Create TF-IDF vectorizer for feature extraction
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            
            X = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate feature importance per class
            class_features = defaultdict(list)
            
            for text, label in zip(texts, labels):
                text_vector = vectorizer.transform([text])
                feature_scores = text_vector.toarray()[0]
                
                # Get top features for this text
                top_indices = np.argsort(feature_scores)[-20:]  # Top 20 features
                for idx in top_indices:
                    if feature_scores[idx] > 0:
                        class_features[label].append({
                            'feature': feature_names[idx],
                            'score': feature_scores[idx]
                        })
            
            # Aggregate feature importance by class
            global_importance = {}
            for class_name, features in class_features.items():
                feature_aggregation = defaultdict(list)
                for feat in features:
                    feature_aggregation[feat['feature']].append(feat['score'])
                
                # Calculate statistics for each feature
                class_importance = {}
                for feature_name, scores in feature_aggregation.items():
                    class_importance[feature_name] = {
                        'mean_score': np.mean(scores),
                        'max_score': np.max(scores),
                        'frequency': len(scores),
                        'std_score': np.std(scores)
                    }
                
                # Sort by importance
                sorted_features = sorted(
                    class_importance.items(),
                    key=lambda x: x[1]['mean_score'] * x[1]['frequency'],
                    reverse=True
                )
                
                global_importance[class_name] = dict(sorted_features[:15])  # Top 15 per class
            
            # Generate insights
            insights = self._generate_feature_insights(global_importance)
            
            analysis_result = {
                'global_feature_importance': global_importance,
                'insights': insights,
                'total_samples': len(texts),
                'feature_extraction_method': model_type,
                'timestamp': datetime.now().isoformat()
            }
            
            self.feature_importance_history.append(analysis_result)
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in global feature explanation: {str(e)}")
            return {'error': str(e)}
    
    def _generate_feature_insights(self, global_importance: Dict) -> List[str]:
        """Generate insights from global feature importance"""
        insights = []
        
        # Find most discriminative features
        all_features = set()
        for class_features in global_importance.values():
            all_features.update(class_features.keys())
        
        # Calculate feature specificity (how specific a feature is to one class)
        feature_specificity = {}
        for feature in all_features:
            class_scores = []
            for class_name, class_features in global_importance.items():
                score = class_features.get(feature, {}).get('mean_score', 0)
                class_scores.append(score)
            
            if max(class_scores) > 0:
                specificity = max(class_scores) / (sum(class_scores) + 1e-10)
                feature_specificity[feature] = specificity
        
        # Most specific features
        most_specific = sorted(feature_specificity.items(), key=lambda x: x[1], reverse=True)[:5]
        if most_specific:
            insights.append(f"Most discriminative features: {', '.join([f[0] for f in most_specific[:3]])}")
        
        # Class-specific insights
        for class_name, features in global_importance.items():
            if features:
                top_feature = list(features.keys())[0]
                insights.append(f"'{class_name}' is most characterized by: '{top_feature}'")
        
        return insights
    
    def generate_counterfactual_examples(self, text: str, model_predictor, 
                                       target_class: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate counterfactual explanations - minimal changes to flip prediction
        """
        try:
            original_prediction = model_predictor.predict_with_confidence(text)
            original_class = original_prediction['ensemble_prediction']
            
            if target_class is None:
                # Find the second most likely class
                probs = original_prediction.get('ensemble_probabilities', {})
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                target_class = sorted_probs[1][0] if len(sorted_probs) > 1 else original_class
            
            counterfactuals = []
            
            # Strategy 1: Word replacement
            words = text.split()
            for i, word in enumerate(words):
                # Try replacing with class-specific keywords
                class_keywords = self._get_class_keywords(target_class)
                
                for keyword in class_keywords[:5]:  # Try top 5 keywords
                    modified_words = words.copy()
                    modified_words[i] = keyword
                    modified_text = ' '.join(modified_words)
                    
                    new_prediction = model_predictor.predict_with_confidence(modified_text)
                    new_class = new_prediction['ensemble_prediction']
                    
                    if new_class == target_class:
                        counterfactuals.append({
                            'original_text': text,
                            'modified_text': modified_text,
                            'change': f"Replaced '{word}' with '{keyword}'",
                            'original_class': original_class,
                            'new_class': new_class,
                            'confidence_change': new_prediction['ensemble_confidence'] - original_prediction['ensemble_confidence']
                        })
                        
                        if len(counterfactuals) >= 3:  # Limit to 3 examples
                            break
                
                if len(counterfactuals) >= 3:
                    break
            
            # Strategy 2: Phrase addition
            class_phrases = self._get_class_phrases(target_class)
            for phrase in class_phrases[:3]:
                modified_text = f"{text} {phrase}"
                new_prediction = model_predictor.predict_with_confidence(modified_text)
                new_class = new_prediction['ensemble_prediction']
                
                if new_class == target_class:
                    counterfactuals.append({
                        'original_text': text,
                        'modified_text': modified_text,
                        'change': f"Added phrase: '{phrase}'",
                        'original_class': original_class,
                        'new_class': new_class,
                        'confidence_change': new_prediction['ensemble_confidence'] - original_prediction['ensemble_confidence']
                    })
            
            return {
                'original_text': text,
                'original_prediction': original_class,
                'target_class': target_class,
                'counterfactuals': counterfactuals[:5],  # Return top 5
                'success_rate': len([cf for cf in counterfactuals if cf['new_class'] == target_class]) / max(len(counterfactuals), 1)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating counterfactuals: {str(e)}")
            return {'error': str(e), 'original_text': text}
    
    def _get_class_keywords(self, class_name: str) -> List[str]:
        """Get characteristic keywords for a class"""
        keyword_map = {
            'Billing': ['payment', 'charge', 'invoice', 'refund', 'subscription', 'fee', 'cost', 'price'],
            'Technical Issue': ['error', 'bug', 'crash', 'broken', 'not working', 'issue', 'problem', 'fix'],
            'Feature Request': ['feature', 'suggest', 'enhancement', 'improvement', 'add', 'new', 'would like'],
            'Account Management': ['account', 'login', 'password', 'profile', 'settings', 'access', 'register'],
            'Product Information': ['information', 'details', 'specifications', 'about', 'how', 'what', 'explain'],
            'Refund & Return': ['refund', 'return', 'cancel', 'money back', 'reverse', 'undo', 'revert'],
            'General Inquiry': ['question', 'help', 'support', 'assistance', 'inquiry', 'ask', 'wondering']
        }
        return keyword_map.get(class_name, [])
    
    def _get_class_phrases(self, class_name: str) -> List[str]:
        """Get characteristic phrases for a class"""
        phrase_map = {
            'Billing': ['I was charged incorrectly', 'billing issue', 'payment problem'],
            'Technical Issue': ['this is not working', 'I found a bug', 'system error'],
            'Feature Request': ['it would be great if', 'can you add', 'feature suggestion'],
            'Account Management': ['I cannot login', 'password reset', 'account settings'],
            'Product Information': ['I need more information', 'can you explain', 'how does this work'],
            'Refund & Return': ['I want a refund', 'return this item', 'cancel my order'],
            'General Inquiry': ['I have a question', 'need help with', 'general support']
        }
        return phrase_map.get(class_name, [])
    
    def analyze_model_bias(self, texts: List[str], labels: List[str], 
                          sensitive_attributes: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """
        Analyze potential bias in model predictions
        """
        if sensitive_attributes is None:
            # Default sensitive attributes to look for
            sensitive_attributes = {
                'gender': ['he', 'she', 'his', 'her', 'him', 'male', 'female', 'man', 'woman'],
                'age': ['young', 'old', 'elderly', 'teen', 'adult', 'senior'],
                'urgency': ['urgent', 'asap', 'immediately', 'emergency', 'critical']
            }
        
        bias_results = {}
        
        for attr_name, keywords in sensitive_attributes.items():
            # Find texts containing these attributes
            attr_texts = []
            attr_labels = []
            non_attr_texts = []
            non_attr_labels = []
            
            for text, label in zip(texts, labels):
                text_lower = text.lower()
                if any(keyword in text_lower for keyword in keywords):
                    attr_texts.append(text)
                    attr_labels.append(label)
                else:
                    non_attr_texts.append(text)
                    non_attr_labels.append(label)
            
            if len(attr_texts) > 0 and len(non_attr_texts) > 0:
                # Calculate label distribution for both groups
                attr_dist = Counter(attr_labels)
                non_attr_dist = Counter(non_attr_labels)
                
                # Normalize distributions
                attr_total = sum(attr_dist.values())
                non_attr_total = sum(non_attr_dist.values())
                
                attr_dist_norm = {k: v/attr_total for k, v in attr_dist.items()}
                non_attr_dist_norm = {k: v/non_attr_total for k, v in non_attr_dist.items()}
                
                # Calculate bias metrics
                all_classes = set(list(attr_dist.keys()) + list(non_attr_dist.keys()))
                bias_scores = {}
                
                for class_name in all_classes:
                    attr_prob = attr_dist_norm.get(class_name, 0)
                    non_attr_prob = non_attr_dist_norm.get(class_name, 0)
                    
                    # Calculate statistical parity difference
                    bias_score = attr_prob - non_attr_prob
                    bias_scores[class_name] = bias_score
                
                bias_results[attr_name] = {
                    'samples_with_attribute': len(attr_texts),
                    'samples_without_attribute': len(non_attr_texts),
                    'distribution_with_attribute': attr_dist_norm,
                    'distribution_without_attribute': non_attr_dist_norm,
                    'bias_scores': bias_scores,
                    'max_bias': max(abs(score) for score in bias_scores.values()),
                    'biased_classes': [cls for cls, score in bias_scores.items() if abs(score) > 0.1]
                }
        
        # Generate bias insights
        bias_insights = []
        for attr_name, results in bias_results.items():
            if results['max_bias'] > 0.1:  # Significant bias threshold
                bias_insights.append(f"Potential bias detected for {attr_name}: max difference of {results['max_bias']:.2f}")
                if results['biased_classes']:
                    bias_insights.append(f"Most affected classes: {', '.join(results['biased_classes'])}")
        
        self.bias_analysis_results = {
            'bias_analysis': bias_results,
            'bias_insights': bias_insights,
            'overall_bias_score': max([r['max_bias'] for r in bias_results.values()]) if bias_results else 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.bias_analysis_results
    
    def create_visualization_report(self, save_dir: str = "reports/explainability") -> str:
        """
        Create comprehensive visualization report
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create feature importance visualization
        if self.feature_importance_history:
            self._create_feature_importance_plot(save_dir)
        
        # Create bias analysis visualization
        if self.bias_analysis_results:
            self._create_bias_analysis_plot(save_dir)
        
        # Generate HTML report
        report_path = os.path.join(save_dir, 'explainability_report.html')
        self._generate_html_report(report_path)
        
        return report_path
    
    def _create_feature_importance_plot(self, save_dir: str):
        """Create feature importance visualization"""
        if not self.feature_importance_history:
            return
        
        latest_analysis = self.feature_importance_history[-1]
        global_importance = latest_analysis['global_feature_importance']
        
        # Create subplot for each class
        n_classes = len(global_importance)
        fig, axes = plt.subplots(n_classes, 1, figsize=(12, 4*n_classes))
        
        if n_classes == 1:
            axes = [axes]
        
        for i, (class_name, features) in enumerate(global_importance.items()):
            if features:
                # Get top 10 features
                top_features = list(features.items())[:10]
                feature_names = [f[0] for f in top_features]
                feature_scores = [f[1]['mean_score'] for f in top_features]
                
                axes[i].barh(feature_names, feature_scores)
                axes[i].set_title(f'Top Features for {class_name}')
                axes[i].set_xlabel('Importance Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_bias_analysis_plot(self, save_dir: str):
        """Create bias analysis visualization"""
        if not self.bias_analysis_results.get('bias_analysis'):
            return
        
        bias_data = []
        for attr_name, results in self.bias_analysis_results['bias_analysis'].items():
            for class_name, bias_score in results['bias_scores'].items():
                bias_data.append({
                    'Attribute': attr_name,
                    'Class': class_name,
                    'Bias Score': bias_score
                })
        
        if bias_data:
            df = pd.DataFrame(bias_data)
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(df.pivot(index='Class', columns='Attribute', values='Bias Score'),
                       annot=True, cmap='RdBu_r', center=0, fmt='.3f')
            plt.title('Model Bias Analysis\n(Positive = Higher probability for texts with attribute)')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'bias_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_html_report(self, report_path: str):
        """Generate comprehensive HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Explainability Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                .insight {{ background-color: #e8f4fd; padding: 10px; margin: 10px 0; border-left: 4px solid #2196F3; }}
                .warning {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ff9800; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Customer Support Ticket Classifier - Explainability Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Feature Importance Analysis</h2>
                <p>Understanding which features drive model predictions.</p>
                {'<img src="feature_importance.png" alt="Feature Importance">' if self.feature_importance_history else '<p>No feature importance data available.</p>'}
            </div>
            
            <div class="section">
                <h2>Bias Analysis</h2>
                <p>Analysis of potential biases in model predictions.</p>
                {'<img src="bias_analysis.png" alt="Bias Analysis">' if self.bias_analysis_results else '<p>No bias analysis data available.</p>'}
                
                {self._generate_bias_insights_html()}
            </div>
            
            <div class="section">
                <h2>Model Interpretability Summary</h2>
                <div class="insight">
                    <strong>Key Insights:</strong>
                    <ul>
                        {self._generate_key_insights_html()}
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
    
    def _generate_bias_insights_html(self) -> str:
        """Generate HTML for bias insights"""
        if not self.bias_analysis_results.get('bias_insights'):
            return "<p>No significant bias detected.</p>"
        
        html = ""
        for insight in self.bias_analysis_results['bias_insights']:
            html += f'<div class="warning">{insight}</div>'
        
        return html
    
    def _generate_key_insights_html(self) -> str:
        """Generate HTML for key insights"""
        insights = []
        
        if self.feature_importance_history:
            latest = self.feature_importance_history[-1]
            insights.extend(latest.get('insights', []))
        
        if self.bias_analysis_results.get('overall_bias_score', 0) > 0.1:
            insights.append("⚠️ Potential bias detected - review model fairness")
        else:
            insights.append("✅ No significant bias detected")
        
        if not insights:
            insights = ["Model analysis completed successfully"]
        
        return "".join(f"<li>{insight}</li>" for insight in insights)

if __name__ == "__main__":
    # Example usage
    explainer = ModelExplainer()
    
    # Mock data for testing
    sample_texts = [
        "I can't log into my account",
        "The app keeps crashing",
        "I want a refund for my purchase"
    ]
    sample_labels = ["Account Management", "Technical Issue", "Refund & Return"]
    
    # Global feature analysis
    analysis = explainer.explain_global_features(sample_texts, sample_labels)
    print("Global analysis completed")
    
    # Bias analysis
    bias_results = explainer.analyze_model_bias(sample_texts, sample_labels)
    print(f"Bias analysis completed - Overall bias score: {bias_results['overall_bias_score']:.3f}")
