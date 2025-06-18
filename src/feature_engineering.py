"""
Advanced Feature Engineering Pipeline
Automated feature engineering for text classification with advanced NLP techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import re
import string
import logging
from datetime import datetime
import pickle
import json
import os
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering pipeline that extracts:
    - Traditional text features (TF-IDF, n-grams)
    - Linguistic features (POS tags, named entities, syntactic patterns)
    - Semantic features (topic modeling, word embeddings)
    - Statistical features (text statistics, readability scores)
    - Domain-specific features (support ticket patterns)
    """
    
    def __init__(self, 
                 enable_spacy: bool = True,
                 enable_nltk: bool = True,
                 max_features: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 3),
                 min_df: int = 2,
                 max_df: float = 0.95,
                 num_topics: int = 20,
                 embedding_dim: int = 100):
        """
        Initialize feature engineer.
        
        Args:
            enable_spacy: Whether to use spaCy features
            enable_nltk: Whether to use NLTK features
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for text features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            num_topics: Number of topics for LDA
            embedding_dim: Dimension for embeddings
        """
        self.enable_spacy = enable_spacy and SPACY_AVAILABLE
        self.enable_nltk = enable_nltk and NLTK_AVAILABLE
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.num_topics = num_topics
        self.embedding_dim = embedding_dim
        
        # Feature extractors
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.lda_model = None
        self.topic_vectorizer = None
        self.scaler = StandardScaler()
        
        # NLP models
        self.nlp_model = None
        self.sentiment_analyzer = None
        
        # Feature vocabulary and statistics
        self.feature_names = []
        self.feature_stats = {}
        self.domain_patterns = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)        
        self.logger = logging.getLogger(__name__)
        
        self._initialize_nlp_models()
    
    def _initialize_nlp_models(self):
        """Initialize NLP models."""
        try:
            if self.enable_spacy:
                # Try to load spaCy model
                try:
                    self.nlp_model = spacy.load("en_core_web_sm")
                except OSError:
                    self.logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                    self.enable_spacy = False
            
            if self.enable_nltk:
                # Download required NLTK data
                try:
                    nltk.data.find('tokenizers/punkt')
                    nltk.data.find('corpora/stopwords')
                    nltk.data.find('taggers/averaged_perceptron_tagger')
                    nltk.data.find('chunkers/maxent_ne_chunker')
                    nltk.data.find('corpora/words')
                    nltk.data.find('vader_lexicon')
                    
                    self.sentiment_analyzer = SentimentIntensityAnalyzer()
                
                except LookupError:
                    self.logger.info("Downloading required NLTK data...")
                try:
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                    nltk.download('averaged_perceptron_tagger', quiet=True)
                    nltk.download('maxent_ne_chunker', quiet=True)
                    nltk.download('words', quiet=True)
                    nltk.download('vader_lexicon', quiet=True)
                    
                    self.sentiment_analyzer = SentimentIntensityAnalyzer()
                    self.logger.info("NLTK data downloaded successfully")
                    
                except Exception as download_error:
                    self.logger.warning(f"Failed to download NLTK data: {download_error}")
                    self.sentiment_analyzer = None
                    
        except Exception as e:
            self.logger.error(f"Error initializing NLP models: {e}")
    
    def fit(self, X: Union[List[str], pd.Series], y: Optional[List[str]] = None):
        """
        Fit the feature engineering pipeline.
        
        Args:
            X: Text data
            y: Optional labels for supervised feature engineering
        """
        try:
            texts = self._prepare_texts(X)
            
            # Fit text vectorizers
            self._fit_text_features(texts)
            
            # Fit topic modeling
            self._fit_topic_modeling(texts)
            
            # Learn domain patterns
            self._learn_domain_patterns(texts, y)
            
            # Extract all features for scaling
            all_features = self._extract_all_features(texts)
            self.scaler.fit(all_features)
            
            # Store feature names
            self._build_feature_names()
            
            self.logger.info(f"Feature engineering fitted with {len(self.feature_names)} features")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting feature engineer: {e}")
            raise
    
    def transform(self, X: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Transform texts to feature vectors.
        
        Args:
            X: Text data to transform
            
        Returns:
            Feature matrix
        """
        try:
            texts = self._prepare_texts(X)
            features = self._extract_all_features(texts)
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            return scaled_features
            
        except Exception as e:
            self.logger.error(f"Error transforming features: {e}")
            raise
    
    def _prepare_texts(self, X: Union[List[str], pd.Series]) -> List[str]:
        """Prepare and clean texts."""
        if isinstance(X, pd.Series):
            texts = X.astype(str).tolist()
        else:
            texts = [str(text) for text in X]
        
        # Basic cleaning
        cleaned_texts = []
        for text in texts:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            cleaned_texts.append(text)
        return cleaned_texts
    
    def _fit_text_features(self, texts: List[str]):
        """Fit text-based feature extractors."""
        # Adjust parameters for small datasets to prevent over-pruning
        min_df = self.min_df
        max_df = self.max_df
        
        # For small datasets, adjust min_df to prevent all terms from being pruned
        if len(texts) <= 5 and min_df > 1:
            min_df = 1
            self.logger.warning(f"Small dataset ({len(texts)} texts), adjusting min_df to 1")
        
        try:
            # TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=min_df,
                max_df=max_df,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )
            self.tfidf_vectorizer.fit(texts)
            
        except ValueError as e:
            if "no terms remain" in str(e).lower():
                # Fallback: use very relaxed parameters
                self.logger.warning("TF-IDF failed with current parameters, using fallback parameters")
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=min(1000, self.max_features),
                    ngram_range=(1, 2),  # Simpler n-grams
                    min_df=1,  # No minimum frequency
                    max_df=1.0,  # No maximum frequency
                    stop_words=None,  # Keep all words
                    lowercase=True,
                    strip_accents='unicode'
                )
                self.tfidf_vectorizer.fit(texts)
            else:
                raise
        
        try:
            # Count vectorizer for topic modeling
            self.count_vectorizer = CountVectorizer(
                max_features=min(5000, self.max_features),
                min_df=min_df,
                max_df=max_df,
                stop_words='english',
                lowercase=True
            )
            self.count_vectorizer.fit(texts)
            
        except ValueError as e:
            if "no terms remain" in str(e).lower():
                # Fallback for count vectorizer
                self.logger.warning("CountVectorizer failed with current parameters, using fallback parameters")
                self.count_vectorizer = CountVectorizer(
                    max_features=min(1000, self.max_features),
                    min_df=1,
                    max_df=1.0,
                    stop_words=None,
                    lowercase=True
                )
                self.count_vectorizer.fit(texts)
            else:
                raise
    
    def _fit_topic_modeling(self, texts: List[str]):
        """Fit topic modeling."""
        try:
            # Prepare documents for LDA
            doc_term_matrix = self.count_vectorizer.transform(texts)
            
            # Fit LDA model
            self.lda_model = LatentDirichletAllocation(
                n_components=self.num_topics,
                max_iter=20,
                learning_method='online',
                learning_offset=50.0,
                random_state=42
            )
            self.lda_model.fit(doc_term_matrix)
            
        except Exception as e:
            self.logger.warning(f"Error fitting topic modeling: {e}")
            self.lda_model = None
    
    def _learn_domain_patterns(self, texts: List[str], labels: Optional[List[str]] = None):
        """Learn domain-specific patterns."""
        # Support ticket specific patterns
        self.domain_patterns = {
            'urgency_words': ['urgent', 'asap', 'immediately', 'critical', 'emergency'],
            'emotion_words': ['frustrated', 'angry', 'disappointed', 'pleased', 'satisfied'],
            'action_words': ['refund', 'cancel', 'fix', 'resolve', 'help', 'support'],
            'technical_words': ['error', 'bug', 'issue', 'problem', 'failure', 'crash'],
            'greeting_patterns': [r'\b(hi|hello|dear|greetings)\b', r'\bthank you\b'],
            'question_patterns': [r'\?', r'\bhow\b', r'\bwhat\b', r'\bwhen\b', r'\bwhere\b', r'\bwhy\b'],
            'time_patterns': [r'\b\d{1,2}:\d{2}\b', r'\b(today|tomorrow|yesterday)\b', r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b']
        }
        
        # Learn category-specific patterns if labels provided
        if labels:
            self._learn_category_patterns(texts, labels)
    
    def _learn_category_patterns(self, texts: List[str], labels: List[str]):
        """Learn patterns specific to each category."""
        category_texts = defaultdict(list)
        for text, label in zip(texts, labels):
            category_texts[label].append(text.lower())
        
        self.domain_patterns['category_keywords'] = {}
        
        for category, cat_texts in category_texts.items():
            # Find common words in this category
            all_words = []
            for text in cat_texts:
                words = re.findall(r'\b\w+\b', text)
                all_words.extend(words)
            
            word_counts = Counter(all_words)
            # Get top keywords for this category
            top_words = [word for word, count in word_counts.most_common(20) 
                        if len(word) > 2 and word not in STOP_WORDS]
            
            self.domain_patterns['category_keywords'][category] = top_words
    
    def _extract_all_features(self, texts: List[str]) -> np.ndarray:
        """Extract all features from texts."""
        feature_matrices = []
        
        # Text-based features
        tfidf_features = self._extract_tfidf_features(texts)
        feature_matrices.append(tfidf_features)
        
        # Statistical features
        stat_features = self._extract_statistical_features(texts)
        feature_matrices.append(stat_features)
        
        # Linguistic features
        if self.enable_spacy or self.enable_nltk:
            ling_features = self._extract_linguistic_features(texts)
            feature_matrices.append(ling_features)
        
        # Topic features
        if self.lda_model:
            topic_features = self._extract_topic_features(texts)
            feature_matrices.append(topic_features)
        
        # Domain-specific features
        domain_features = self._extract_domain_features(texts)
        feature_matrices.append(domain_features)
        
        # Sentiment features
        sentiment_features = self._extract_sentiment_features(texts)
        feature_matrices.append(sentiment_features)
        
        # Combine all features
        combined_features = np.hstack(feature_matrices)
        
        return combined_features
    
    def _extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Extract TF-IDF features."""
        return self.tfidf_vectorizer.transform(texts).toarray()
    
    def _extract_statistical_features(self, texts: List[str]) -> np.ndarray:
        """Extract statistical text features."""
        features = []
        
        for text in texts:
            feature_vector = [
                len(text),  # Character count
                len(text.split()),  # Word count
                len(re.findall(r'[.!?]+', text)),  # Sentence count
                len(re.findall(r'[A-Z]', text)),  # Uppercase count
                len(re.findall(r'[0-9]', text)),  # Digit count
                len(re.findall(r'[^\w\s]', text)),  # Punctuation count
                text.count('?'),  # Question marks
                text.count('!'),  # Exclamation marks
                len(re.findall(r'\b[A-Z]{2,}\b', text)),  # Acronyms
                len(re.findall(r'http[s]?://\S+', text)),  # URLs
                len(re.findall(r'\S+@\S+', text)),  # Email addresses
            ]
            
            # Average word length
            words = text.split()
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            feature_vector.append(avg_word_length)
            
            # Readability score (simplified)
            sentences = len(re.findall(r'[.!?]+', text))
            if sentences > 0 and words:
                avg_sentence_length = len(words) / sentences
                feature_vector.append(avg_sentence_length)
            else:
                feature_vector.append(0)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """Extract linguistic features using NLP libraries."""
        features = []
        
        for text in texts:
            feature_vector = []
            
            if self.enable_spacy and self.nlp_model:
                doc = self.nlp_model(text)
                
                # POS tag counts
                pos_counts = Counter([token.pos_ for token in doc])
                pos_features = [
                    pos_counts.get('NOUN', 0),
                    pos_counts.get('VERB', 0),
                    pos_counts.get('ADJ', 0),
                    pos_counts.get('ADV', 0),
                    pos_counts.get('PRON', 0),
                ]
                feature_vector.extend(pos_features)
                
                # Named entity counts
                ent_counts = Counter([ent.label_ for ent in doc.ents])
                ent_features = [
                    ent_counts.get('PERSON', 0),
                    ent_counts.get('ORG', 0),
                    ent_counts.get('GPE', 0),
                    ent_counts.get('MONEY', 0),
                    ent_counts.get('DATE', 0),
                ]
                feature_vector.extend(ent_features)
                
                # Dependency parsing features
                dep_counts = Counter([token.dep_ for token in doc])
                dep_features = [
                    dep_counts.get('nsubj', 0),
                    dep_counts.get('dobj', 0),
                    dep_counts.get('prep', 0),
                ]
                feature_vector.extend(dep_features)
                
            elif self.enable_nltk:
                # NLTK-based features
                try:
                    tokens = word_tokenize(text.lower())
                    pos_tags = pos_tag(tokens)
                    
                    # POS tag counts
                    pos_counts = Counter([tag for word, tag in pos_tags])
                    pos_features = [
                        pos_counts.get('NN', 0) + pos_counts.get('NNS', 0),  # Nouns
                        pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0),  # Verbs
                        pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0),  # Adjectives
                        pos_counts.get('RB', 0) + pos_counts.get('RBR', 0),  # Adverbs
                        pos_counts.get('PRP', 0) + pos_counts.get('PRP$', 0),  # Pronouns
                    ]
                    feature_vector.extend(pos_features)
                    
                    # Add placeholder values for missing spaCy features
                    feature_vector.extend([0] * 8)  # NER + DEP features
                    
                except Exception:
                    # Fallback to zeros if NLTK processing fails
                    feature_vector.extend([0] * 13)
            else:
                # No NLP library available
                feature_vector.extend([0] * 13)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _extract_topic_features(self, texts: List[str]) -> np.ndarray:
        """Extract topic modeling features."""
        if not self.lda_model:
            return np.zeros((len(texts), self.num_topics))
        
        try:
            doc_term_matrix = self.count_vectorizer.transform(texts)
            topic_distributions = self.lda_model.transform(doc_term_matrix)
            return topic_distributions
        except Exception as e:
            self.logger.warning(f"Error extracting topic features: {e}")
            return np.zeros((len(texts), self.num_topics))
    
    def _extract_domain_features(self, texts: List[str]) -> np.ndarray:
        """Extract domain-specific features."""
        features = []
        
        for text in texts:
            text_lower = text.lower()
            feature_vector = []
            
            # Pattern matching features
            for pattern_name, patterns in self.domain_patterns.items():
                if pattern_name == 'category_keywords':
                    continue  # Skip category keywords for now
                
                if isinstance(patterns[0], str):
                    # Word-based patterns
                    count = sum(1 for word in patterns if word in text_lower)
                else:
                    # Regex patterns
                    count = sum(1 for pattern in patterns if re.search(pattern, text_lower))
                
                feature_vector.append(count)
            
            # Support ticket specific features
            feature_vector.extend([
                1 if any(word in text_lower for word in ['please', 'help', 'support']) else 0,  # Politeness
                1 if re.search(r'\b(can\'t|cannot|won\'t|doesn\'t|isn\'t)\b', text_lower) else 0,  # Negation
                len(re.findall(r'[!]{2,}', text)),  # Multiple exclamations
                1 if re.search(r'\ball caps\b|[A-Z]{10,}', text) else 0,  # All caps (shouting)
                len(re.findall(r'\$\d+\.?\d*', text)),  # Money mentions
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _extract_sentiment_features(self, texts: List[str]) -> np.ndarray:
        """Extract sentiment features."""
        features = []
        
        for text in texts:
            feature_vector = []
            
            if self.enable_nltk and self.sentiment_analyzer:
                try:
                    scores = self.sentiment_analyzer.polarity_scores(text)
                    feature_vector = [
                        scores['compound'],
                        scores['pos'],
                        scores['neu'],
                        scores['neg']
                    ]
                except Exception:
                    feature_vector = [0.0, 0.0, 0.0, 0.0]
            else:
                # Simple rule-based sentiment
                positive_words = ['good', 'great', 'excellent', 'satisfied', 'happy', 'pleased']
                negative_words = ['bad', 'terrible', 'awful', 'frustrated', 'angry', 'disappointed']
                
                text_lower = text.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                
                total_sentiment_words = pos_count + neg_count
                if total_sentiment_words > 0:
                    sentiment_score = (pos_count - neg_count) / total_sentiment_words
                else:
                    sentiment_score = 0.0
                
                feature_vector = [sentiment_score, pos_count, 0, neg_count]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _build_feature_names(self):
        """Build feature names for interpretability."""
        self.feature_names = []
        
        # TF-IDF feature names
        if self.tfidf_vectorizer:
            tfidf_names = [f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]
            self.feature_names.extend(tfidf_names)
        
        # Statistical feature names
        stat_names = [
            'char_count', 'word_count', 'sentence_count', 'uppercase_count',
            'digit_count', 'punct_count', 'question_marks', 'exclamation_marks',
            'acronym_count', 'url_count', 'email_count', 'avg_word_length', 'avg_sentence_length'
        ]
        self.feature_names.extend(stat_names)
        
        # Linguistic feature names
        ling_names = [
            'noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count',
            'person_ent', 'org_ent', 'gpe_ent', 'money_ent', 'date_ent',
            'nsubj_dep', 'dobj_dep', 'prep_dep'
        ]
        self.feature_names.extend(ling_names)
        
        # Topic feature names
        if self.lda_model:
            topic_names = [f"topic_{i}" for i in range(self.num_topics)]
            self.feature_names.extend(topic_names)
        
        # Domain feature names
        domain_names = [
            'urgency_words', 'emotion_words', 'action_words', 'technical_words',
            'greeting_patterns', 'question_patterns', 'time_patterns',
            'politeness', 'negation', 'multiple_exclamation', 'all_caps', 'money_mentions'
        ]
        self.feature_names.extend(domain_names)
        
        # Sentiment feature names
        sentiment_names = ['sentiment_compound', 'sentiment_pos', 'sentiment_neu', 'sentiment_neg']
        self.feature_names.extend(sentiment_names)
    
    def get_feature_importance(self, model=None) -> Dict[str, float]:
        """Get feature importance if model is provided."""
        if model is None or not hasattr(model, 'feature_importances_'):
            return {}
        
        if len(self.feature_names) != len(model.feature_importances_):
            return {}
        
        importance_dict = dict(zip(self.feature_names, model.feature_importances_))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def get_top_features_by_category(self, n_features: int = 20) -> Dict[str, List[str]]:
        """Get top features by category type."""
        if not self.feature_names:
            return {}
        
        categories = {
            'tfidf': [name for name in self.feature_names if name.startswith('tfidf_')],
            'statistical': [name for name in self.feature_names if name in [
                'char_count', 'word_count', 'sentence_count', 'uppercase_count',
                'digit_count', 'punct_count', 'question_marks', 'exclamation_marks',
                'acronym_count', 'url_count', 'email_count', 'avg_word_length', 'avg_sentence_length'
            ]],
            'linguistic': [name for name in self.feature_names if any(name.endswith(suffix) for suffix in ['_count', '_ent', '_dep'])],
            'topic': [name for name in self.feature_names if name.startswith('topic_')],
            'domain': [name for name in self.feature_names if name in [
                'urgency_words', 'emotion_words', 'action_words', 'technical_words',
                'greeting_patterns', 'question_patterns', 'time_patterns',
                'politeness', 'negation', 'multiple_exclamation', 'all_caps', 'money_mentions'
            ]],
            'sentiment': [name for name in self.feature_names if name.startswith('sentiment_')]
        }
        
        # Limit to n_features per category
        for category in categories:
            categories[category] = categories[category][:n_features]
        
        return categories
    
    def save_pipeline(self, filepath: str):
        """Save the feature engineering pipeline."""
        pipeline_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'count_vectorizer': self.count_vectorizer,
            'lda_model': self.lda_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'domain_patterns': self.domain_patterns,
            'config': {
                'max_features': self.max_features,
                'ngram_range': self.ngram_range,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'num_topics': self.num_topics,
                'embedding_dim': self.embedding_dim
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
    
    def load_pipeline(self, filepath: str):
        """Load the feature engineering pipeline."""
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.tfidf_vectorizer = pipeline_data['tfidf_vectorizer']
        self.count_vectorizer = pipeline_data['count_vectorizer']
        self.lda_model = pipeline_data['lda_model']
        self.scaler = pipeline_data['scaler']
        self.feature_names = pipeline_data['feature_names']
        self.domain_patterns = pipeline_data['domain_patterns']
        
        config = pipeline_data['config']
        self.max_features = config['max_features']
        self.ngram_range = config['ngram_range']
        self.min_df = config['min_df']
        self.max_df = config['max_df']
        self.num_topics = config['num_topics']
        self.embedding_dim = config['embedding_dim']
