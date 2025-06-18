"""
Real-time Streaming Processor
Handles real-time ticket classification with streaming data sources.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import threading
import queue
import time

# Streaming libraries
try:
    import websockets
    import aiohttp
    from kafka import KafkaConsumer, KafkaProducer
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

# Core components
from predict import TicketClassifier
from response_generator import AIResponseGenerator
from performance_monitor import ModelPerformanceMonitor
from drift_detector import DataDriftDetector

@dataclass
class StreamingTicket:
    """Streaming ticket data structure."""
    ticket_id: str
    customer_message: str
    timestamp: datetime
    channel: str = "stream"
    priority: str = "medium"
    customer_id: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class StreamingResult:
    """Result of streaming ticket processing."""
    ticket_id: str
    predicted_category: str
    confidence: float
    processing_time_ms: float
    generated_response: Optional[str] = None
    escalation_required: bool = False
    drift_detected: bool = False
    timestamp: datetime = None

class StreamingProcessor:
    """
    Real-time streaming processor for customer support tickets.
    
    Features:
    - Multiple streaming sources (WebSocket, Kafka, HTTP webhooks)
    - Real-time classification and response generation
    - Performance monitoring and drift detection
    - Batch processing with configurable windows
    - Auto-scaling based on load
    """
    
    def __init__(self,
                 classifier: TicketClassifier,
                 response_generator: Optional[AIResponseGenerator] = None,
                 enable_monitoring: bool = True,
                 enable_drift_detection: bool = True,
                 batch_size: int = 10,
                 batch_timeout: float = 5.0):
        """
        Initialize streaming processor.
        
        Args:
            classifier: Trained ticket classifier
            response_generator: AI response generator (optional)
            enable_monitoring: Enable performance monitoring
            enable_drift_detection: Enable drift detection
            batch_size: Size of processing batches
            batch_timeout: Timeout for batch processing (seconds)
        """
        self.classifier = classifier
        self.response_generator = response_generator
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize monitoring components
        self.monitor = ModelPerformanceMonitor() if enable_monitoring else None
        self.drift_detector = DataDriftDetector() if enable_drift_detection else None
        
        # Streaming state
        self.is_running = False
        self.ticket_queue = queue.Queue()
        self.result_callbacks = []
        self.processing_stats = {
            'total_processed': 0,
            'total_errors': 0,
            'avg_processing_time': 0,
            'start_time': None
        }
        
        # Batch processing
        self.current_batch = []
        self.last_batch_time = time.time()
        
        # Threading
        self.processing_thread = None
        self.batch_thread = None
    
    def add_result_callback(self, callback: Callable[[StreamingResult], None]):
        """Add callback for processing results."""
        self.result_callbacks.append(callback)
    
    def start_streaming(self):
        """Start the streaming processor."""
        if self.is_running:
            self.logger.warning("Streaming processor is already running")
            return
        
        self.is_running = True
        self.processing_stats['start_time'] = datetime.now()
        
        # Start processing threads
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.batch_thread = threading.Thread(target=self._batch_processing_loop, daemon=True)
        
        self.processing_thread.start()
        self.batch_thread.start()
        
        self.logger.info("Streaming processor started")
    
    def stop_streaming(self):
        """Stop the streaming processor."""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        if self.batch_thread:
            self.batch_thread.join(timeout=5)
        
        self.logger.info("Streaming processor stopped")
    
    def add_ticket(self, ticket: StreamingTicket):
        """Add a ticket to the processing queue."""
        if not self.is_running:
            raise RuntimeError("Streaming processor is not running")
        
        self.ticket_queue.put(ticket)
    
    def _processing_loop(self):
        """Main processing loop."""
        while self.is_running:
            try:
                # Get ticket from queue (with timeout)
                try:
                    ticket = self.ticket_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process ticket
                start_time = time.time()
                result = self._process_single_ticket(ticket)
                processing_time = (time.time() - start_time) * 1000
                
                result.processing_time_ms = processing_time
                result.timestamp = datetime.now()
                
                # Update stats
                self._update_stats(processing_time)
                
                # Call callbacks
                for callback in self.result_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        self.logger.error(f"Callback error: {str(e)}")
                
                # Add to current batch
                self.current_batch.append(result)
                
            except Exception as e:
                self.logger.error(f"Processing error: {str(e)}")
                self.processing_stats['total_errors'] += 1
    
    def _batch_processing_loop(self):
        """Batch processing loop for monitoring and drift detection."""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if batch should be processed
                should_process = (
                    len(self.current_batch) >= self.batch_size or
                    (len(self.current_batch) > 0 and 
                     current_time - self.last_batch_time >= self.batch_timeout)
                )
                
                if should_process and self.current_batch:
                    self._process_batch(self.current_batch.copy())
                    self.current_batch.clear()
                    self.last_batch_time = current_time
                
                time.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                self.logger.error(f"Batch processing error: {str(e)}")
    
    def _process_single_ticket(self, ticket: StreamingTicket) -> StreamingResult:
        """Process a single ticket."""
        try:
            # Classify ticket
            prediction = self.classifier.predict_single(ticket.customer_message)
            
            # Generate response if generator is available
            generated_response = None
            escalation_required = False
            
            if self.response_generator:
                try:
                    response_obj = self.response_generator.generate_response(
                        ticket_text=ticket.customer_message,
                        predicted_category=prediction['predicted_category'],
                        confidence=prediction['confidence'],
                        urgency=ticket.priority,
                        customer_name=ticket.customer_id or "Customer"
                    )
                    generated_response = response_obj.response_text
                    escalation_required = response_obj.escalation_recommended
                except Exception as e:
                    self.logger.warning(f"Response generation failed: {str(e)}")
            
            # Check for escalation based on confidence
            if prediction['confidence'] < 0.5:
                escalation_required = True
            
            return StreamingResult(
                ticket_id=ticket.ticket_id,
                predicted_category=prediction['predicted_category'],
                confidence=prediction['confidence'],
                processing_time_ms=0,  # Will be set by caller
                generated_response=generated_response,
                escalation_required=escalation_required,
                drift_detected=False  # Will be updated in batch processing
            )
            
        except Exception as e:
            self.logger.error(f"Error processing ticket {ticket.ticket_id}: {str(e)}")
            raise
    
    def _process_batch(self, batch: List[StreamingResult]):
        """Process a batch of results for monitoring and drift detection."""
        try:
            if not batch:
                return
            
            # Extract data for monitoring
            texts = []
            predictions = []
            confidences = []
            
            for result in batch:
                # Note: We need the original text, but it's not in the result
                # This is a limitation - in a real implementation, we'd store more data
                predictions.append(result.predicted_category)
                confidences.append(result.confidence)
            
            # Update performance monitoring
            if self.monitor:
                try:
                    batch_df = pd.DataFrame({
                        'predicted_category': predictions,
                        'confidence': confidences,
                        'processing_time_ms': [r.processing_time_ms for r in batch]
                    })
                    
                    # Log performance metrics
                    avg_confidence = np.mean(confidences)
                    avg_processing_time = np.mean([r.processing_time_ms for r in batch])
                    
                    self.logger.debug(f"Batch processed: {len(batch)} tickets, "
                                    f"avg_confidence={avg_confidence:.3f}, "
                                    f"avg_time={avg_processing_time:.1f}ms")
                    
                except Exception as e:
                    self.logger.warning(f"Monitoring update failed: {str(e)}")
            
            # Drift detection (simplified - would need more sophisticated implementation)
            if self.drift_detector:
                try:
                    # Check if confidence distribution has shifted
                    if len(confidences) >= 5:  # Minimum sample size
                        low_confidence_rate = sum(1 for c in confidences if c < 0.7) / len(confidences)
                        if low_confidence_rate > 0.3:  # More than 30% low confidence
                            self.logger.warning(f"Potential drift detected: {low_confidence_rate:.1%} low confidence predictions")
                            # Mark recent results as having drift detected
                            for result in batch[-5:]:  # Mark last 5 results
                                result.drift_detected = True
                                
                except Exception as e:
                    self.logger.warning(f"Drift detection failed: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Batch processing error: {str(e)}")
    
    def _update_stats(self, processing_time_ms: float):
        """Update processing statistics."""
        self.processing_stats['total_processed'] += 1
        
        # Update rolling average processing time
        current_avg = self.processing_stats['avg_processing_time']
        total = self.processing_stats['total_processed']
        
        if total == 1:
            self.processing_stats['avg_processing_time'] = processing_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.processing_stats['avg_processing_time'] = (
                alpha * processing_time_ms + (1 - alpha) * current_avg
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        stats = self.processing_stats.copy()
        
        if stats['start_time']:
            uptime = datetime.now() - stats['start_time']
            stats['uptime_seconds'] = uptime.total_seconds()
            
            if stats['total_processed'] > 0:
                stats['throughput_per_second'] = stats['total_processed'] / uptime.total_seconds()
        
        stats['queue_size'] = self.ticket_queue.qsize()
        stats['current_batch_size'] = len(self.current_batch)
        stats['is_running'] = self.is_running
        
        return stats

class WebSocketStreaming:
    """WebSocket streaming handler."""
    
    def __init__(self, processor: StreamingProcessor, port: int = 8765):
        self.processor = processor
        self.port = port
        self.server = None
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection."""
        try:
            async for message in websocket:
                try:
                    # Parse incoming message
                    data = json.loads(message)
                    
                    # Create streaming ticket
                    ticket = StreamingTicket(
                        ticket_id=data.get('ticket_id', f"ws_{int(time.time())}"),
                        customer_message=data['customer_message'],
                        timestamp=datetime.now(),
                        channel='websocket',
                        priority=data.get('priority', 'medium'),
                        customer_id=data.get('customer_id'),
                        metadata=data.get('metadata', {})
                    )
                    
                    # Add to processor
                    self.processor.add_ticket(ticket)
                    
                    # Send acknowledgment
                    await websocket.send(json.dumps({
                        'status': 'received',
                        'ticket_id': ticket.ticket_id,
                        'timestamp': ticket.timestamp.isoformat()
                    }))
                    
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'error': 'Invalid JSON format'
                    }))
                except KeyError as e:
                    await websocket.send(json.dumps({
                        'error': f'Missing required field: {str(e)}'
                    }))
                except Exception as e:
                    await websocket.send(json.dumps({
                        'error': f'Processing error: {str(e)}'
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            pass
    
    async def start_server(self):
        """Start WebSocket server."""
        if not STREAMING_AVAILABLE:
            raise RuntimeError("Streaming libraries not available")
        
        self.server = await websockets.serve(self.handle_client, "localhost", self.port)
        print(f"WebSocket server started on ws://localhost:{self.port}")
    
    def stop_server(self):
        """Stop WebSocket server."""
        if self.server:
            self.server.close()

class KafkaStreaming:
    """Kafka streaming handler."""
    
    def __init__(self, 
                 processor: StreamingProcessor,
                 bootstrap_servers: str = 'localhost:9092',
                 input_topic: str = 'support_tickets',
                 output_topic: str = 'ticket_results'):
        self.processor = processor
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.consumer = None
        self.producer = None
        self.is_running = False
    
    def start_kafka_streaming(self):
        """Start Kafka streaming."""
        if not STREAMING_AVAILABLE:
            raise RuntimeError("Kafka libraries not available")
        
        # Initialize consumer and producer
        self.consumer = KafkaConsumer(
            self.input_topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        # Add result callback to send results to output topic
        self.processor.add_result_callback(self._send_result_to_kafka)
        
        self.is_running = True
        
        # Start consuming
        threading.Thread(target=self._consume_loop, daemon=True).start()
        print(f"Kafka streaming started: {self.input_topic} -> {self.output_topic}")
    
    def _consume_loop(self):
        """Kafka consumer loop."""
        while self.is_running:
            try:
                for message in self.consumer:
                    if not self.is_running:
                        break
                    
                    # Parse message
                    data = message.value
                    
                    # Create streaming ticket
                    ticket = StreamingTicket(
                        ticket_id=data.get('ticket_id', f"kafka_{message.offset}"),
                        customer_message=data['customer_message'],
                        timestamp=datetime.now(),
                        channel='kafka',
                        priority=data.get('priority', 'medium'),
                        customer_id=data.get('customer_id'),
                        metadata=data.get('metadata', {})
                    )
                    
                    # Add to processor
                    self.processor.add_ticket(ticket)
                    
            except Exception as e:
                print(f"Kafka consumer error: {str(e)}")
                time.sleep(1)
    
    def _send_result_to_kafka(self, result: StreamingResult):
        """Send processing result to Kafka output topic."""
        try:
            if self.producer:
                self.producer.send(self.output_topic, asdict(result))
        except Exception as e:
            print(f"Error sending result to Kafka: {str(e)}")
    
    def stop_kafka_streaming(self):
        """Stop Kafka streaming."""
        self.is_running = False
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()

# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the streaming processor
    pass
