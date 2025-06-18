"""
AI Response Generator
Intelligent response generation for customer support tickets using open-source LLMs.
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Hugging Face transformers
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
        pipeline, GenerationConfig
    )
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Template engines
try:
    from jinja2 import Template, Environment, BaseLoader
    JINJA_AVAILABLE = True
except ImportError:
    JINJA_AVAILABLE = False

@dataclass
class ResponseTemplate:
    """Response template structure."""
    category: str
    urgency: str
    template: str
    variables: List[str]
    tone: str = "professional"
    language: str = "en"

@dataclass
class GeneratedResponse:
    """Generated response structure."""
    response_text: str
    confidence_score: float
    template_used: str
    generation_method: str
    metadata: Dict
    suggested_actions: List[str]
    escalation_recommended: bool

class AIResponseGenerator:
    """
    AI-powered response generator for customer support tickets.
    
    Features:
    - Multiple open-source LLM support (GPT-2, FLAN-T5, DialoGPT, etc.)
    - Template-based generation with personalization
    - Multi-language support
    - Response quality scoring
    - Tone adjustment (professional, friendly, empathetic)
    - Context-aware generation
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-small",
                 use_gpu: bool = True,
                 template_file: Optional[str] = None,
                 max_length: int = 150,
                 temperature: float = 0.7):
        """
        Initialize response generator.
        
        Args:
            model_name: Hugging Face model name
            use_gpu: Whether to use GPU if available
            template_file: Path to response templates JSON
            max_length: Maximum response length
            temperature: Generation temperature (creativity)
        """
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.max_length = max_length
        self.temperature = temperature
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models and templates
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.templates = {}
        self.response_history = []
        
        # Supported models configuration
        self.supported_models = {
            # Conversational models
            "microsoft/DialoGPT-small": {"type": "causal", "max_tokens": 512},
            "microsoft/DialoGPT-medium": {"type": "causal", "max_tokens": 512},
            "microsoft/DialoGPT-large": {"type": "causal", "max_tokens": 512},
            
            # Text generation models
            "gpt2": {"type": "causal", "max_tokens": 1024},
            "gpt2-medium": {"type": "causal", "max_tokens": 1024},
            "distilgpt2": {"type": "causal", "max_tokens": 1024},
            
            # Sequence-to-sequence models
            "google/flan-t5-small": {"type": "seq2seq", "max_tokens": 512},
            "google/flan-t5-base": {"type": "seq2seq", "max_tokens": 512},
            "google/flan-t5-large": {"type": "seq2seq", "max_tokens": 512},
            
            # Instruction-following models
            "microsoft/GODEL-v1_1-base-seq2seq": {"type": "seq2seq", "max_tokens": 512},
            "facebook/blenderbot-400M-distill": {"type": "seq2seq", "max_tokens": 512},
            
            # Multilingual models
            "facebook/mbart-large-50-many-to-many-mmt": {"type": "seq2seq", "max_tokens": 512},
        }
        
        # Load default templates
        self._load_default_templates()
        
        # Load custom templates if provided
        if template_file and os.path.exists(template_file):
            self._load_templates_from_file(template_file)
        
        # Initialize model
        if HF_AVAILABLE:
            self._initialize_model()
        else:
            self.logger.warning("Hugging Face transformers not available. Install with: pip install transformers torch")
    
    def _initialize_model(self):
        """Initialize the language model."""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            
            # Set device
            device = "cuda" if self.use_gpu else "cpu"
            
            # Get model configuration
            model_config = self.supported_models.get(
                self.model_name, 
                {"type": "causal", "max_tokens": 512}
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on type
            if model_config["type"] == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32
                )
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.use_gpu else -1
                )
            else:  # causal
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32
                )
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.use_gpu else -1
                )
            
            # Move model to device
            if self.use_gpu:
                self.model = self.model.to("cuda")
            
            self.logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
    
    def _load_default_templates(self):
        """Load default response templates."""
        default_templates = {
            "billing": {
                "high": ResponseTemplate(
                    category="billing",
                    urgency="high",
                    template="I understand your billing concern is urgent. Let me immediately review your account and resolve this issue. I'll personally ensure this gets the highest priority.",
                    variables=["customer_name", "account_id", "issue_details"],
                    tone="empathetic"
                ),
                "medium": ResponseTemplate(
                    category="billing",
                    urgency="medium",
                    template="Thank you for contacting us about your billing inquiry. I'll review your account details and provide you with a comprehensive explanation within 24 hours.",
                    variables=["customer_name", "account_id"],
                    tone="professional"
                ),
                "low": ResponseTemplate(
                    category="billing",
                    urgency="low",
                    template="Thank you for your billing question. I'll look into this and respond with detailed information within 2-3 business days.",
                    variables=["customer_name"],
                    tone="friendly"
                )
            },
            "technical": {
                "high": ResponseTemplate(
                    category="technical",
                    urgency="high",
                    template="I see you're experiencing a critical technical issue. Our technical team is being immediately notified, and we'll have someone working on this within the hour.",
                    variables=["customer_name", "issue_type", "error_details"],
                    tone="urgent"
                ),
                "medium": ResponseTemplate(
                    category="technical",
                    urgency="medium",
                    template="Thank you for reporting this technical issue. I've documented the details and will escalate this to our technical team for resolution within 24 hours.",
                    variables=["customer_name", "issue_type"],
                    tone="professional"
                ),
                "low": ResponseTemplate(
                    category="technical",
                    urgency="low",
                    template="Thanks for the technical question. I'll research this and provide you with detailed troubleshooting steps or solutions within 2-3 business days.",
                    variables=["customer_name"],
                    tone="helpful"
                )
            },
            "general": {
                "high": ResponseTemplate(
                    category="general",
                    urgency="high",
                    template="Thank you for contacting us. I understand this is important to you, and I'll personally ensure you get a response within 4 hours.",
                    variables=["customer_name", "inquiry_type"],
                    tone="attentive"
                ),
                "medium": ResponseTemplate(
                    category="general",
                    urgency="medium",
                    template="Thank you for your inquiry. I'll review this and provide you with a comprehensive response within 24 hours.",
                    variables=["customer_name"],
                    tone="professional"
                ),
                "low": ResponseTemplate(
                    category="general",
                    urgency="low",
                    template="Thank you for reaching out. I'll look into this and respond within 2-3 business days with the information you need.",
                    variables=["customer_name"],
                    tone="friendly"
                )
            },
            "complaint": {
                "high": ResponseTemplate(
                    category="complaint",
                    urgency="high",
                    template="I sincerely apologize for the experience you've had. This is absolutely not the level of service we strive for. I'm escalating this immediately to management and will personally follow up within 2 hours.",
                    variables=["customer_name", "complaint_details", "incident_date"],
                    tone="apologetic"
                ),
                "medium": ResponseTemplate(
                    category="complaint",
                    urgency="medium",
                    template="I'm sorry to hear about your experience. Your feedback is valuable to us, and I want to make this right. I'll investigate this thoroughly and respond within 24 hours.",
                    variables=["customer_name", "complaint_details"],
                    tone="empathetic"
                ),
                "low": ResponseTemplate(
                    category="complaint",
                    urgency="low",
                    template="Thank you for bringing this to our attention. I'll review your concerns and respond with our plan to address this within 2-3 business days.",
                    variables=["customer_name"],
                    tone="understanding"
                )
            },
            "compliment": {
                "high": ResponseTemplate(
                    category="compliment",
                    urgency="high",
                    template="Wow, thank you so much for this wonderful feedback! I'll make sure to share this with the team mentioned. Your kind words truly make our day!",
                    variables=["customer_name", "team_mentioned", "specific_praise"],
                    tone="enthusiastic"
                ),
                "medium": ResponseTemplate(
                    category="compliment",
                    urgency="medium",
                    template="Thank you so much for taking the time to share this positive feedback! I'll make sure the team knows about your appreciation.",
                    variables=["customer_name", "team_mentioned"],
                    tone="grateful"
                ),
                "low": ResponseTemplate(
                    category="compliment",
                    urgency="low",
                    template="Thank you for your kind words! We really appreciate customers like you who take the time to share positive feedback.",
                    variables=["customer_name"],
                    tone="appreciative"
                )
            }
        }
        
        # Flatten templates for easier access
        for category, urgency_templates in default_templates.items():
            for urgency, template in urgency_templates.items():
                key = f"{category}_{urgency}"
                self.templates[key] = template
    
    def _load_templates_from_file(self, file_path: str):
        """Load templates from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                custom_templates = json.load(f)
            
            for key, template_data in custom_templates.items():
                self.templates[key] = ResponseTemplate(**template_data)
            
            self.logger.info(f"Loaded {len(custom_templates)} custom templates")
            
        except Exception as e:
            self.logger.error(f"Error loading templates from {file_path}: {e}")
    
    def generate_response(self, 
                         ticket_text: str,
                         predicted_category: str,
                         confidence: float,
                         urgency: str = "medium",
                         customer_name: str = "Customer",
                         context: Optional[Dict] = None,
                         generation_method: str = "hybrid") -> GeneratedResponse:
        """
        Generate a response for a support ticket.
        
        Args:
            ticket_text: Original ticket content
            predicted_category: Predicted ticket category
            confidence: Model confidence score
            urgency: Ticket urgency (low, medium, high)
            customer_name: Customer's name for personalization
            context: Additional context information
            generation_method: "template", "ai", or "hybrid"
            
        Returns:
            GeneratedResponse object
        """
        try:
            context = context or {}
            
            # Generate response based on method
            if generation_method == "template":
                response = self._generate_template_response(
                    predicted_category, urgency, customer_name, context
                )
            elif generation_method == "ai" and self.model is not None:
                response = self._generate_ai_response(
                    ticket_text, predicted_category, urgency, customer_name, context
                )
            else:  # hybrid
                if confidence > 0.8 and self.model is not None:
                    # High confidence: enhance template with AI
                    response = self._generate_hybrid_response(
                        ticket_text, predicted_category, urgency, customer_name, context
                    )
                else:
                    # Low confidence: fall back to template
                    response = self._generate_template_response(
                        predicted_category, urgency, customer_name, context
                    )
            
            # Add metadata
            response.metadata.update({
                "original_ticket": ticket_text[:100] + "..." if len(ticket_text) > 100 else ticket_text,
                "generation_timestamp": datetime.now().isoformat(),
                "model_used": self.model_name,
                "confidence_threshold": confidence
            })
            
            # Store in history
            self.response_history.append({
                "timestamp": datetime.now().isoformat(),
                "category": predicted_category,
                "urgency": urgency,
                "method": generation_method,
                "response_length": len(response.response_text),
                "confidence": response.confidence_score
            })
            
            # Suggest actions
            response.suggested_actions = self._suggest_actions(predicted_category, urgency, context)
            
            # Determine if escalation is needed
            response.escalation_recommended = self._should_escalate(
                predicted_category, urgency, confidence, context
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response(predicted_category, customer_name)
    
    def _generate_template_response(self, 
                                   category: str, 
                                   urgency: str, 
                                   customer_name: str, 
                                   context: Dict) -> GeneratedResponse:
        """Generate response using templates."""
        template_key = f"{category.lower()}_{urgency.lower()}"
        
        # Get template or fallback to general
        template = self.templates.get(template_key) or self.templates.get(f"general_{urgency.lower()}")
        if not template:
            template = self.templates["general_medium"]
        
        # Replace variables
        response_text = template.template
        
        # Basic variable replacement
        variables = {
            "customer_name": customer_name,
            "account_id": context.get("account_id", "your account"),
            "issue_details": context.get("issue_details", "your issue"),
            "issue_type": context.get("issue_type", "technical issue"),
            "complaint_details": context.get("complaint_details", "your concerns"),
            "incident_date": context.get("incident_date", "recently"),
            "team_mentioned": context.get("team_mentioned", "our team"),
            "specific_praise": context.get("specific_praise", "your positive experience"),
            "inquiry_type": context.get("inquiry_type", "your inquiry")
        }
        
        for var, value in variables.items():
            response_text = response_text.replace(f"{{{var}}}", str(value))
        
        return GeneratedResponse(
            response_text=response_text,
            confidence_score=0.8,  # Template responses have high confidence
            template_used=template_key,
            generation_method="template",
            metadata={"template_tone": template.tone},
            suggested_actions=[],
            escalation_recommended=False
        )
    
    def _generate_ai_response(self, 
                             ticket_text: str, 
                             category: str, 
                             urgency: str, 
                             customer_name: str, 
                             context: Dict) -> GeneratedResponse:
        """Generate response using AI model."""
        if not self.pipeline:
            return self._generate_template_response(category, urgency, customer_name, context)
        
        try:
            # Create prompt based on model type
            model_config = self.supported_models.get(self.model_name, {"type": "causal"})
            
            if model_config["type"] == "seq2seq":
                prompt = self._create_seq2seq_prompt(ticket_text, category, urgency, customer_name)
            else:
                prompt = self._create_causal_prompt(ticket_text, category, urgency, customer_name)
            
            # Generate response
            generation_config = {
                "max_length": self.max_length,
                "temperature": self.temperature,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.pad_token_id
            }
            
            if model_config["type"] == "seq2seq":
                generated = self.pipeline(prompt, **generation_config)
                response_text = generated[0]["generated_text"].strip()
            else:
                generated = self.pipeline(prompt, **generation_config)
                full_text = generated[0]["generated_text"]
                # Extract only the new part (remove prompt)
                response_text = full_text[len(prompt):].strip()
            
            # Clean up response
            response_text = self._clean_generated_response(response_text)
            
            # Calculate confidence based on response quality
            confidence = self._calculate_response_confidence(response_text, category)
            
            return GeneratedResponse(
                response_text=response_text,
                confidence_score=confidence,
                template_used="ai_generated",
                generation_method="ai",
                metadata={
                    "prompt_used": prompt[:100] + "...",
                    "model_type": model_config["type"]
                },
                suggested_actions=[],
                escalation_recommended=False
            )
            
        except Exception as e:
            self.logger.error(f"Error in AI generation: {e}")
            return self._generate_template_response(category, urgency, customer_name, context)
    
    def _generate_hybrid_response(self, 
                                 ticket_text: str, 
                                 category: str, 
                                 urgency: str, 
                                 customer_name: str, 
                                 context: Dict) -> GeneratedResponse:
        """Generate response using hybrid template + AI approach."""
        # Start with template
        template_response = self._generate_template_response(category, urgency, customer_name, context)
        
        # Enhance with AI if available
        if self.pipeline:
            try:
                # Create enhancement prompt
                enhancement_prompt = f"""
Enhance this customer service response to be more personalized and specific to the customer's issue:

Original ticket: {ticket_text[:200]}...
Current response: {template_response.response_text}

Enhanced response:"""
                
                # Generate enhancement
                enhanced = self.pipeline(
                    enhancement_prompt,
                    max_length=self.max_length + len(enhancement_prompt),
                    temperature=0.6,
                    do_sample=True,
                    top_p=0.9
                )
                
                if enhanced and len(enhanced) > 0:
                    full_text = enhanced[0]["generated_text"]
                    enhanced_text = full_text[len(enhancement_prompt):].strip()
                    enhanced_text = self._clean_generated_response(enhanced_text)
                    
                    if len(enhanced_text) > 20:  # Valid enhancement
                        template_response.response_text = enhanced_text
                        template_response.generation_method = "hybrid"
                        template_response.confidence_score = 0.85
                        template_response.metadata["enhancement_applied"] = True
            
            except Exception as e:
                self.logger.warning(f"Enhancement failed, using template: {e}")
        
        return template_response
    
    def _create_seq2seq_prompt(self, ticket_text: str, category: str, urgency: str, customer_name: str) -> str:
        """Create prompt for sequence-to-sequence models."""
        return f"""Generate a professional customer service response for this {urgency} urgency {category} ticket:

Customer message: {ticket_text}

Response:"""
    
    def _create_causal_prompt(self, ticket_text: str, category: str, urgency: str, customer_name: str) -> str:
        """Create prompt for causal language models."""
        return f"""Customer Support Conversation:

Customer ({customer_name}): {ticket_text}

Support Agent: Thank you for contacting us, {customer_name}."""
    
    def _clean_generated_response(self, text: str) -> str:
        """Clean and format generated response."""
        # Remove common artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove brackets
        text = re.sub(r'\(.*?\)', '', text)  # Remove parentheses with artifacts
        text = re.sub(r'<.*?>', '', text)    # Remove HTML-like tags
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        # Ensure proper ending
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text
    
    def _calculate_response_confidence(self, response: str, category: str) -> float:
        """Calculate confidence score for generated response."""
        confidence = 0.5  # Base confidence
        
        # Length check
        if 50 <= len(response) <= 300:
            confidence += 0.2
        
        # Professional tone indicators
        professional_words = ["thank", "appreciate", "understand", "resolve", "assist", "help"]
        for word in professional_words:
            if word.lower() in response.lower():
                confidence += 0.05
        
        # Category-specific checks
        if category.lower() == "complaint" and any(word in response.lower() for word in ["sorry", "apologize"]):
            confidence += 0.1
        
        if category.lower() == "compliment" and any(word in response.lower() for word in ["thank", "appreciate"]):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _suggest_actions(self, category: str, urgency: str, context: Dict) -> List[str]:
        """Suggest follow-up actions based on ticket."""
        actions = []
        
        if urgency == "high":
            actions.append("Schedule immediate follow-up within 4 hours")
            actions.append("Escalate to senior support if not resolved")
        
        if category.lower() == "billing":
            actions.extend([
                "Review customer account details",
                "Check recent transactions",
                "Verify billing information"
            ])
        elif category.lower() == "technical":
            actions.extend([
                "Gather system logs",
                "Test reproduction steps",
                "Check known issues database"
            ])
        elif category.lower() == "complaint":
            actions.extend([
                "Document complaint details",
                "Schedule management review",
                "Prepare compensation if applicable"
            ])
        
        return actions
    
    def _should_escalate(self, category: str, urgency: str, confidence: float, context: Dict) -> bool:
        """Determine if ticket should be escalated."""
        # High urgency always escalates
        if urgency == "high":
            return True
        
        # Low confidence predictions should be reviewed
        if confidence < 0.6:
            return True
        
        # Complaints should often be escalated
        if category.lower() == "complaint" and urgency in ["medium", "high"]:
            return True
        
        # Context-based escalation
        if context.get("previous_complaints", 0) > 2:
            return True
        
        return False
    
    def _generate_fallback_response(self, category: str, customer_name: str) -> GeneratedResponse:
        """Generate fallback response when all else fails."""
        fallback_text = f"Thank you for contacting us, {customer_name}. We have received your {category} inquiry and will respond within 24 hours. We appreciate your patience."
        
        return GeneratedResponse(
            response_text=fallback_text,
            confidence_score=0.7,
            template_used="fallback",
            generation_method="fallback",
            metadata={"fallback_reason": "error_in_generation"},
            suggested_actions=["Review ticket manually", "Assign to appropriate agent"],
            escalation_recommended=True
        )
    
    def batch_generate_responses(self, 
                                tickets: List[Dict],
                                generation_method: str = "hybrid") -> List[GeneratedResponse]:
        """
        Generate responses for multiple tickets.
        
        Args:
            tickets: List of ticket dictionaries with required fields
            generation_method: Generation method to use
            
        Returns:
            List of GeneratedResponse objects
        """
        responses = []
        
        for ticket in tickets:
            try:
                response = self.generate_response(
                    ticket_text=ticket["text"],
                    predicted_category=ticket["category"],
                    confidence=ticket["confidence"],
                    urgency=ticket.get("urgency", "medium"),
                    customer_name=ticket.get("customer_name", "Customer"),
                    context=ticket.get("context", {}),
                    generation_method=generation_method
                )
                responses.append(response)
                
            except Exception as e:
                self.logger.error(f"Error processing ticket: {e}")
                responses.append(self._generate_fallback_response(
                    ticket.get("category", "general"),
                    ticket.get("customer_name", "Customer")
                ))
        
        return responses
    
    def get_response_statistics(self) -> Dict:
        """Get statistics about generated responses."""
        if not self.response_history:
            return {}
        
        df = pd.DataFrame(self.response_history)
        
        stats = {
            "total_responses": len(df),
            "average_confidence": df["confidence"].mean(),
            "responses_by_category": df["category"].value_counts().to_dict(),
            "responses_by_urgency": df["urgency"].value_counts().to_dict(),
            "responses_by_method": df["method"].value_counts().to_dict(),
            "average_response_length": df["response_length"].mean(),
            "recent_activity": df.tail(10).to_dict("records")
        }
        
        return stats
    
    def export_templates(self, file_path: str):
        """Export current templates to JSON file."""
        try:
            exportable_templates = {}
            
            for key, template in self.templates.items():
                exportable_templates[key] = {
                    "category": template.category,
                    "urgency": template.urgency,
                    "template": template.template,
                    "variables": template.variables,
                    "tone": template.tone,
                    "language": template.language
                }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(exportable_templates, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Templates exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting templates: {e}")
    
    def add_custom_template(self, 
                           category: str, 
                           urgency: str, 
                           template_text: str, 
                           variables: List[str] = None,
                           tone: str = "professional"):
        """Add a custom response template."""
        key = f"{category.lower()}_{urgency.lower()}"
        
        self.templates[key] = ResponseTemplate(
            category=category.lower(),
            urgency=urgency.lower(),
            template=template_text,
            variables=variables or [],
            tone=tone
        )
        
        self.logger.info(f"Added custom template: {key}")
    
    def switch_model(self, model_name: str):
        """Switch to a different model."""
        if model_name in self.supported_models:
            self.model_name = model_name
            self._initialize_model()
            self.logger.info(f"Switched to model: {model_name}")
        else:
            self.logger.error(f"Unsupported model: {model_name}")
            self.logger.info(f"Supported models: {list(self.supported_models.keys())}")

# Example usage and testing functions
def test_response_generator():
    """Test the response generator with sample tickets."""
    generator = AIResponseGenerator()
    
    # Sample tickets
    test_tickets = [
        {
            "text": "I was charged twice for my subscription this month. This is unacceptable!",
            "category": "billing",
            "confidence": 0.95,
            "urgency": "high",
            "customer_name": "John Smith",
            "context": {"account_id": "12345", "subscription_type": "premium"}
        },
        {
            "text": "How do I reset my password? I can't log into my account.",
            "category": "technical",
            "confidence": 0.88,
            "urgency": "medium",
            "customer_name": "Jane Doe"
        },
        {
            "text": "Your customer service team was amazing! Sarah helped me so quickly.",
            "category": "compliment",
            "confidence": 0.92,
            "urgency": "low",
            "customer_name": "Mike Johnson",
            "context": {"team_mentioned": "Sarah from customer service"}
        }
    ]
    
    # Generate responses
    print("=== AI Response Generator Test ===")
    for i, ticket in enumerate(test_tickets, 1):
        print(f"\n--- Test {i} ---")
        print(f"Original: {ticket['text']}")
        
        response = generator.generate_response(
            ticket_text=ticket["text"],
            predicted_category=ticket["category"],
            confidence=ticket["confidence"],
            urgency=ticket["urgency"],
            customer_name=ticket["customer_name"],
            context=ticket.get("context", {}),
            generation_method="hybrid"
        )
        
        print(f"Response: {response.response_text}")
        print(f"Confidence: {response.confidence_score:.2f}")
        print(f"Method: {response.generation_method}")
        print(f"Escalation needed: {response.escalation_recommended}")
        print(f"Suggested actions: {', '.join(response.suggested_actions)}")

if __name__ == "__main__":
    test_response_generator()
