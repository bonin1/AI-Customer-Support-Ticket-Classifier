"""
Multi-modal Classification System
Handles classification of tickets with multiple data types: text, images, audio, attachments.
"""

import os
import json
import base64
import mimetypes
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

# Image processing
try:
    from PIL import Image, ImageEnhance
    import cv2
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False

# OCR capabilities
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except (ImportError, AttributeError) as e:
    EASYOCR_AVAILABLE = False

OCR_AVAILABLE = TESSERACT_AVAILABLE or EASYOCR_AVAILABLE

# Audio processing
try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Advanced vision models
try:
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration,
        CLIPProcessor, CLIPModel,
        AutoProcessor, AutoModelForVision2Seq
    )
    import torch
    VISION_MODELS_AVAILABLE = True
except ImportError:
    VISION_MODELS_AVAILABLE = False

# Document processing
try:
    import fitz  # PyMuPDF
    from docx import Document
    DOCUMENT_AVAILABLE = True
except ImportError:
    DOCUMENT_AVAILABLE = False

# Core components
from predict import TicketClassifier
from data_preprocessing import TextPreprocessor

@dataclass
class MultiModalInput:
    """Multi-modal input data structure."""
    text: Optional[str] = None
    images: Optional[List[str]] = None  # Base64 encoded or file paths
    audio: Optional[str] = None  # File path or base64
    attachments: Optional[List[str]] = None  # File paths
    metadata: Dict[str, Any] = None

@dataclass
class ProcessedModalData:
    """Processed modal data."""
    text_features: Optional[Dict[str, Any]] = None
    image_features: Optional[Dict[str, Any]] = None
    audio_features: Optional[Dict[str, Any]] = None
    document_features: Optional[Dict[str, Any]] = None
    combined_text: str = ""
    confidence_scores: Dict[str, float] = None

@dataclass
class MultiModalResult:
    """Multi-modal classification result."""
    predicted_category: str
    confidence: float
    modal_contributions: Dict[str, float]  # Contribution of each modality
    extracted_features: ProcessedModalData
    processing_details: Dict[str, Any]
    timestamp: datetime

class MultiModalClassifier:
    """
    Multi-modal classification system supporting:
    - Text analysis (primary)
    - Image classification and OCR
    - Audio transcription and analysis
    - Document parsing (PDF, DOCX)
    - Attachment analysis
    """
    
    def __init__(self,
                 text_classifier: TicketClassifier,
                 enable_ocr: bool = True,
                 enable_image_classification: bool = True,
                 enable_audio_processing: bool = True,
                 enable_document_parsing: bool = True,
                 vision_model: str = "Salesforce/blip-image-captioning-base"):
        """
        Initialize multi-modal classifier.
        
        Args:
            text_classifier: Primary text classifier
            enable_ocr: Enable OCR for images
            enable_image_classification: Enable image classification
            enable_audio_processing: Enable audio transcription
            enable_document_parsing: Enable document parsing
            vision_model: Vision model for image understanding
        """
        self.text_classifier = text_classifier
        self.text_preprocessor = TextPreprocessor()
        
        # Feature flags
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.enable_image_classification = enable_image_classification and VISION_MODELS_AVAILABLE
        self.enable_audio_processing = enable_audio_processing and AUDIO_AVAILABLE
        self.enable_document_parsing = enable_document_parsing and DOCUMENT_AVAILABLE
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize vision models
        self.vision_processor = None
        self.vision_model = None
        self.clip_processor = None
        self.clip_model = None
        
        if self.enable_image_classification:
            try:
                self._initialize_vision_models(vision_model)
            except Exception as e:
                self.logger.warning(f"Failed to initialize vision models: {str(e)}")
                self.enable_image_classification = False
          # Initialize OCR
        self.ocr_reader = None
        self.use_easyocr = False
        if self.enable_ocr:
            try:
                if EASYOCR_AVAILABLE:
                    self.ocr_reader = easyocr.Reader(['en'])
                    self.use_easyocr = True
                    self.logger.info("EasyOCR initialized successfully")
                elif TESSERACT_AVAILABLE:
                    self.ocr_reader = None  # Will use pytesseract directly
                    self.use_easyocr = False
                    self.logger.info("Tesseract OCR available")
                else:
                    self.enable_ocr = False
                    self.logger.warning("No OCR engines available")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OCR: {str(e)}")
                self.enable_ocr = False
        
        # Processing statistics
        self.processing_stats = {
            'total_processed': 0,
            'modality_usage': {
                'text_only': 0,
                'with_images': 0,
                'with_audio': 0,
                'with_documents': 0,
                'multi_modal': 0
            }
        }
    
    def _initialize_vision_models(self, vision_model: str):
        """Initialize vision models."""
        try:
            # BLIP for image captioning
            self.vision_processor = BlipProcessor.from_pretrained(vision_model)
            self.vision_model = BlipForConditionalGeneration.from_pretrained(vision_model)
            
            # CLIP for image classification
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            self.logger.info(f"Vision models initialized: {vision_model}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vision models: {str(e)}")
    
    def classify_multimodal(self, input_data: MultiModalInput) -> MultiModalResult:
        """
        Classify multi-modal input.
        
        Args:
            input_data: Multi-modal input data
            
        Returns:
            MultiModalResult with classification and feature extraction
        """
        start_time = datetime.now()
        
        try:
            # Process each modality
            processed_data = self._process_all_modalities(input_data)
            
            # Combine features and classify
            final_prediction = self._combine_and_classify(processed_data, input_data)
            
            # Update statistics
            self._update_processing_stats(input_data)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return MultiModalResult(
                predicted_category=final_prediction['predicted_category'],
                confidence=final_prediction['confidence'],
                modal_contributions=final_prediction['modal_contributions'],
                extracted_features=processed_data,
                processing_details={
                    'processing_time_seconds': processing_time,
                    'modalities_used': final_prediction['modalities_used'],
                    'primary_modality': final_prediction['primary_modality']
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Multi-modal classification error: {str(e)}")
            raise
    
    def _process_all_modalities(self, input_data: MultiModalInput) -> ProcessedModalData:
        """Process all available modalities."""
        processed = ProcessedModalData(confidence_scores={})
        combined_texts = []
        
        # Process text
        if input_data.text:
            processed.text_features = self._process_text(input_data.text)
            combined_texts.append(input_data.text)
            processed.confidence_scores['text'] = 1.0
        
        # Process images
        if input_data.images and (self.enable_ocr or self.enable_image_classification):
            try:
                image_result = self._process_images(input_data.images)
                processed.image_features = image_result
                
                # Add extracted text from images
                if image_result.get('extracted_text'):
                    combined_texts.append(image_result['extracted_text'])
                
                # Add image descriptions
                if image_result.get('descriptions'):
                    combined_texts.extend(image_result['descriptions'])
                
                processed.confidence_scores['images'] = image_result.get('confidence', 0.0)
                
            except Exception as e:
                self.logger.warning(f"Image processing failed: {str(e)}")
                processed.confidence_scores['images'] = 0.0
        
        # Process audio
        if input_data.audio and self.enable_audio_processing:
            try:
                audio_result = self._process_audio(input_data.audio)
                processed.audio_features = audio_result
                
                # Add transcribed text
                if audio_result.get('transcription'):
                    combined_texts.append(audio_result['transcription'])
                
                processed.confidence_scores['audio'] = audio_result.get('confidence', 0.0)
                
            except Exception as e:
                self.logger.warning(f"Audio processing failed: {str(e)}")
                processed.confidence_scores['audio'] = 0.0
        
        # Process documents
        if input_data.attachments and self.enable_document_parsing:
            try:
                doc_result = self._process_documents(input_data.attachments)
                processed.document_features = doc_result
                
                # Add extracted text from documents
                if doc_result.get('extracted_text'):
                    combined_texts.append(doc_result['extracted_text'])
                
                processed.confidence_scores['documents'] = doc_result.get('confidence', 0.0)
                
            except Exception as e:
                self.logger.warning(f"Document processing failed: {str(e)}")
                processed.confidence_scores['documents'] = 0.0
        
        # Combine all text
        processed.combined_text = ' '.join(combined_texts).strip()
        
        return processed
    
    def _process_text(self, text: str) -> Dict[str, Any]:
        """Process text input."""
        try:
            # Use existing text preprocessing
            processed_text = self.text_preprocessor.preprocess_text(text)
            
            return {
                'original_text': text,
                'processed_text': processed_text,
                'length': len(text),
                'word_count': len(text.split()),
                'confidence': 1.0
            }
        except Exception as e:
            self.logger.error(f"Text processing error: {str(e)}")
            return {'original_text': text, 'confidence': 0.0}
    
    def _process_images(self, images: List[str]) -> Dict[str, Any]:
        """Process image inputs."""
        extracted_texts = []
        descriptions = []
        classifications = []
        total_confidence = 0.0
        
        for img_path in images[:5]:  # Limit to 5 images for performance
            try:
                # Load image
                if img_path.startswith('data:'):
                    # Base64 encoded image
                    image = self._decode_base64_image(img_path)
                else:
                    # File path
                    image = Image.open(img_path)
                  # OCR extraction
                if self.enable_ocr:
                    try:
                        if self.use_easyocr and self.ocr_reader:
                            ocr_result = self.ocr_reader.readtext(np.array(image))
                            extracted_text = ' '.join([text for _, text, confidence in ocr_result if confidence > 0.5])
                        elif TESSERACT_AVAILABLE:
                            extracted_text = pytesseract.image_to_string(image).strip()
                        else:
                            extracted_text = ""
                        
                        if extracted_text.strip():
                            extracted_texts.append(extracted_text)
                    except Exception as e:
                        self.logger.warning(f"OCR failed for image: {str(e)}")
                
                # Image captioning
                if self.enable_image_classification and self.vision_model:
                    try:
                        inputs = self.vision_processor(image, return_tensors="pt")
                        out = self.vision_model.generate(**inputs, max_length=50)
                        description = self.vision_processor.decode(out[0], skip_special_tokens=True)
                        descriptions.append(description)
                    except Exception as e:
                        self.logger.warning(f"Image captioning failed: {str(e)}")
                
                # Image classification for support categories
                if self.clip_model:
                    try:
                        category_labels = [
                            "technical problem screenshot",
                            "billing receipt document", 
                            "product defect photo",
                            "error message screen",
                            "account settings page",
                            "general inquiry image"
                        ]
                        
                        inputs = self.clip_processor(text=category_labels, images=image, return_tensors="pt", padding=True)
                        outputs = self.clip_model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        probs = logits_per_image.softmax(dim=1)
                        
                        max_prob = float(probs.max())
                        predicted_label = category_labels[probs.argmax().item()]
                        
                        classifications.append({
                            'label': predicted_label,
                            'confidence': max_prob
                        })
                        total_confidence += max_prob
                        
                    except Exception as e:
                        self.logger.warning(f"Image classification failed: {str(e)}")
                
            except Exception as e:
                self.logger.error(f"Failed to process image {img_path}: {str(e)}")
        
        avg_confidence = total_confidence / len(images) if images else 0.0
        
        return {
            'extracted_text': ' '.join(extracted_texts),
            'descriptions': descriptions,
            'classifications': classifications,
            'image_count': len(images),
            'confidence': avg_confidence
        }
    
    def _process_audio(self, audio_path: str) -> Dict[str, Any]:
        """Process audio input."""
        try:
            if not AUDIO_AVAILABLE:
                return {'error': 'Audio processing not available', 'confidence': 0.0}
            
            # Load audio file
            if audio_path.startswith('data:'):
                # Handle base64 encoded audio (simplified)
                return {'error': 'Base64 audio not implemented', 'confidence': 0.0}
            
            # Load with librosa
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Extract basic audio features
            duration = len(y) / sr
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            
            # Detect if it's speech vs music/noise
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Simple speech detection (this would need a proper ASR system)
            # For now, we'll simulate transcription
            transcription = f"[Audio detected: {duration:.1f}s, tempo: {tempo:.1f}]"
            
            return {
                'transcription': transcription,
                'duration_seconds': duration,
                'sample_rate': sr,
                'tempo': float(tempo),
                'features': {
                    'mfcc_mean': float(np.mean(mfccs)),
                    'spectral_centroid_mean': float(np.mean(spectral_centroids))
                },
                'confidence': 0.3  # Low confidence since we're not doing real ASR
            }
            
        except Exception as e:
            self.logger.error(f"Audio processing error: {str(e)}")
            return {'error': str(e), 'confidence': 0.0}
    
    def _process_documents(self, attachment_paths: List[str]) -> Dict[str, Any]:
        """Process document attachments."""
        extracted_texts = []
        document_types = []
        total_confidence = 0.0
        
        for doc_path in attachment_paths[:10]:  # Limit to 10 documents
            try:
                # Determine document type
                mime_type, _ = mimetypes.guess_type(doc_path)
                doc_type = mime_type or 'unknown'
                document_types.append(doc_type)
                
                extracted_text = ""
                confidence = 0.0
                
                # PDF processing
                if doc_path.lower().endswith('.pdf') and DOCUMENT_AVAILABLE:
                    try:
                        doc = fitz.open(doc_path)
                        text_parts = []
                        for page in doc:
                            text_parts.append(page.get_text())
                        extracted_text = '\n'.join(text_parts)
                        confidence = 0.9
                        doc.close()
                    except Exception as e:
                        self.logger.warning(f"PDF processing failed: {str(e)}")
                
                # DOCX processing
                elif doc_path.lower().endswith('.docx') and DOCUMENT_AVAILABLE:
                    try:
                        doc = Document(doc_path)
                        text_parts = [paragraph.text for paragraph in doc.paragraphs]
                        extracted_text = '\n'.join(text_parts)
                        confidence = 0.9
                    except Exception as e:
                        self.logger.warning(f"DOCX processing failed: {str(e)}")
                
                # Plain text files
                elif doc_path.lower().endswith(('.txt', '.log', '.csv')):
                    try:
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            extracted_text = f.read()
                        confidence = 1.0
                    except UnicodeDecodeError:
                        try:
                            with open(doc_path, 'r', encoding='latin-1') as f:
                                extracted_text = f.read()
                            confidence = 0.8
                        except Exception as e:
                            self.logger.warning(f"Text file processing failed: {str(e)}")
                
                if extracted_text.strip():
                    extracted_texts.append(extracted_text[:5000])  # Limit text length
                    total_confidence += confidence
                
            except Exception as e:
                self.logger.error(f"Failed to process document {doc_path}: {str(e)}")
        
        avg_confidence = total_confidence / len(attachment_paths) if attachment_paths else 0.0
        
        return {
            'extracted_text': '\n'.join(extracted_texts),
            'document_types': document_types,
            'document_count': len(attachment_paths),
            'confidence': avg_confidence
        }
    
    def _decode_base64_image(self, base64_string: str) -> Image.Image:
        """Decode base64 encoded image."""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:'):
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Create PIL Image
            from io import BytesIO
            image = Image.open(BytesIO(image_data))
            
            return image
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")
    
    def _combine_and_classify(self, processed_data: ProcessedModalData, input_data: MultiModalInput) -> Dict[str, Any]:
        """Combine multi-modal features and classify."""
        try:
            # Use combined text for classification
            if processed_data.combined_text.strip():
                text_prediction = self.text_classifier.predict_single(processed_data.combined_text)
            else:
                # Fallback to original text if available
                if input_data.text:
                    text_prediction = self.text_classifier.predict_single(input_data.text)
                else:
                    # No text available - use default category
                    text_prediction = {
                        'predicted_category': 'General Inquiry',
                        'confidence': 0.1,
                        'probabilities': {}
                    }
            
            # Calculate modal contributions
            modal_contributions = {}
            total_weight = 0.0
            
            # Text contribution (primary modality)
            text_weight = processed_data.confidence_scores.get('text', 0.0) * 0.6
            modal_contributions['text'] = text_weight
            total_weight += text_weight
            
            # Image contribution
            image_weight = processed_data.confidence_scores.get('images', 0.0) * 0.25
            modal_contributions['images'] = image_weight
            total_weight += image_weight
            
            # Audio contribution
            audio_weight = processed_data.confidence_scores.get('audio', 0.0) * 0.1
            modal_contributions['audio'] = audio_weight
            total_weight += audio_weight
            
            # Document contribution
            doc_weight = processed_data.confidence_scores.get('documents', 0.0) * 0.05
            modal_contributions['documents'] = doc_weight
            total_weight += doc_weight
            
            # Normalize contributions
            if total_weight > 0:
                modal_contributions = {k: v/total_weight for k, v in modal_contributions.items()}
            
            # Adjust confidence based on multi-modal evidence
            base_confidence = text_prediction['confidence']
            
            # Boost confidence if multiple modalities agree
            modality_count = sum(1 for v in processed_data.confidence_scores.values() if v > 0.3)
            if modality_count > 1:
                confidence_boost = min(0.2, (modality_count - 1) * 0.1)
                adjusted_confidence = min(1.0, base_confidence + confidence_boost)
            else:
                adjusted_confidence = base_confidence
            
            # Determine primary modality
            primary_modality = max(modal_contributions.items(), key=lambda x: x[1])[0] if modal_contributions else 'text'
            
            return {
                'predicted_category': text_prediction['predicted_category'],
                'confidence': adjusted_confidence,
                'modal_contributions': modal_contributions,
                'modalities_used': [k for k, v in processed_data.confidence_scores.items() if v > 0],
                'primary_modality': primary_modality,
                'base_text_prediction': text_prediction
            }
            
        except Exception as e:
            self.logger.error(f"Classification combination error: {str(e)}")
            raise
    
    def _update_processing_stats(self, input_data: MultiModalInput):
        """Update processing statistics."""
        self.processing_stats['total_processed'] += 1
        
        # Count modality usage
        modalities_used = 0
        if input_data.text:
            modalities_used += 1
        if input_data.images:
            self.processing_stats['modality_usage']['with_images'] += 1
            modalities_used += 1
        if input_data.audio:
            self.processing_stats['modality_usage']['with_audio'] += 1
            modalities_used += 1
        if input_data.attachments:
            self.processing_stats['modality_usage']['with_documents'] += 1
            modalities_used += 1
        
        if modalities_used == 1:
            self.processing_stats['modality_usage']['text_only'] += 1
        elif modalities_used > 1:
            self.processing_stats['modality_usage']['multi_modal'] += 1
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get current system capabilities."""
        return {
            'text_processing': True,
            'image_processing': IMAGE_AVAILABLE,
            'ocr': self.enable_ocr,
            'image_classification': self.enable_image_classification,
            'audio_processing': self.enable_audio_processing,
            'document_parsing': self.enable_document_parsing,
            'vision_models': VISION_MODELS_AVAILABLE,
            'streaming_support': True
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()

# Example usage
if __name__ == "__main__":
    # This would be used for testing the multi-modal classifier
    pass
