"""
Real-time Streaming Data Processor with Kafka Integration
Handles high-velocity data streams with complex event processing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Union, AsyncGenerator
from enum import Enum
import asyncio
import json
import logging

import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class StreamPriority(Enum):
    """Stream processing priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class StreamEvent(BaseModel):
    """Base stream event structure."""
    event_id: str
    timestamp: datetime
    data: Dict[str, Any]
    priority: StreamPriority = StreamPriority.NORMAL
    source: str
    metadata: Optional[Dict[str, Any]] = None


class StreamProcessor(ABC):
    """Abstract base class for stream processors."""
    
    @abstractmethod
    async def process_event(self, event: StreamEvent) -> Any:
        """Process a single stream event."""
        pass


class StatisticalProcessor(StreamProcessor):
    """Processor for statistical analysis of stream events."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._data_windows: Dict[str, List[float]] = {}
    
    async def process_event(self, event: StreamEvent) -> Dict[str, Any]:
        """Process event for statistical analysis."""
        if event.data.get('metric_value') is not None:
            metric_name = event.data.get('metric_name', 'unknown')
            value = float(event.data['metric_value'])
            
            if metric_name not in self._data_windows:
                self._data_windows[metric_name] = []
            
            self._data_windows[metric_name].append(value)
            
            # Keep only recent values
            if len(self._data_windows[metric_name]) > self.window_size:
                self._data_windows[metric_name] = self._data_windows[metric_name][-self.window_size:]
            
            # Calculate statistics
            values = self._data_windows[metric_name]
            return {
                'metric': metric_name,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'percentile_95': np.percentile(values, 95),
                'percentile_5': np.percentile(values, 5),
                'count': len(values),
                'latest_value': value
            }
        
        return {}


class AnomalyDetectionProcessor(StreamProcessor):
    """Real-time anomaly detection processor."""
    
    def __init__(self, threshold: float = 2.5):
        self.threshold = threshold
        self._baselines: Dict[str, Dict[str, float]] = {}
    
    async def process_event(self, event: StreamEvent) -> Dict[str, Any]:
        """Detect anomalies in stream events."""
        if event.data.get('metric_value') is not None:
            metric_name = event.data.get('metric_name', 'unknown')
            value = float(event.data['metric_value'])
            
            if metric_name not in self._baselines:
                # Initialize baseline
                self._baselines[metric_name] = {'mean': value, 'std': 1.0, 'count': 1}
            else:
                baseline = self._baselines[metric_name]
                
                # Update baseline
                baseline['count'] += 1
                baseline['mean'] = ((baseline['mean'] * (baseline['count'] - 1)) + value) / baseline['count']
                baseline['std'] = np.sqrt(
                    ((baseline['std'] ** 2 * (baseline['count'] - 2)) + 
                     (value - baseline['mean']) ** 2) / (baseline['count'] - 1)
                ) if baseline['count'] > 2 else 1.0
            
            # Calculate anomaly score
            if self._baselines[metric_name]['std'] > 0:
                z_score = abs(value - self._baselines[metric_name]['mean']) / self._baselines[metric_name]['std']
                is_anomaly = z_score > self.threshold
            else:
                z_score = 0
                is_anomaly = False
            
            return {
                'metric': metric_name,
                'value': value,
                'baseline_mean': self._baselines[metric_name]['mean'],
                'baseline_std': self._baselines[metric_name]['std'],
                'z_score': z_score,
                'is_anomaly': is_anomaly,
                'anomaly_score': z_score / self.threshold
            }
        
        return {}


class StreamingDataProcessor:
    """
    Advanced streaming data processor with Kafka integration.
    Handles real-time data processing with complex event processing capabilities.
    """
    
    def __init__(self, kafka_servers: List[str] = None):
        self.kafka_servers = kafka_servers or ['localhost:9092']
        self._processors: Dict[str, StreamProcessor] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._kafka_consumer = None
        self._kafka_producer = None
        
    def add_processor(self, name: str, processor: StreamProcessor) -> None:
        """Add a stream processor."""
        self._processors[name] = processor
        logger.info(f"Added processor: {name}")
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add an event handler for specific event types."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    async def initialize_kafka(self) -> None:
        """Initialize Kafka connections."""
        try:
            self._kafka_producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                acks='all'  # Wait for all replicas
            )
            
            self._kafka_consumer = KafkaConsumer(
                bootstrap_servers=self.kafka_servers,
                auto_offset_reset='latest',
                value_deserializer=lambda m: json.loads(m.decode('utf-8')) if m else None,
                consumer_timeout_ms=1000
            )
            
            logger.info("Kafka connections initialized")
            
        except KafkaError as e:
            logger.error(f"Failed to initialize Kafka: {e}")
            raise
    
    async def publish_event(self, topic: str, event: StreamEvent) -> None:
        """Publish event to Kafka topic."""
        if self._kafka_producer:
            try:
                future = self._kafka_producer.send(
                    topic,
                    value=event.dict()
                )
                # Wait for confirmation
                future.get(timeout=10)
                logger.debug(f"Published event to {topic}: {event.event_id}")
                
            except KafkaError as e:
                logger.error(f"Failed to publish event to {topic}: {e}")
                raise
    
    async def consume_events(self, topics: List[str]) -> AsyncGenerator[StreamEvent, None]:
        """Consume events from Kafka topics."""
        if not self._kafka_consumer:
            await self.initialize_kafka()
        
        self._kafka_consumer.subscribe(topics)
        
        try:
            while True:
                for message in self._kafka_consumer:
                    if message.value:
                        try:
                            # Convert to StreamEvent
                            data = message.value
                            event = StreamEvent(
                                event_id=data['event_id'],
                                timestamp=datetime.fromisoformat(data['timestamp']),
                                data=data['data'],
                                priority=StreamPriority(data.get('priority', 2)),
                                source=data.get('source', 'unknown'),
                                metadata=data.get('metadata')
                            )
                            yield event
                        except Exception as e:
                            logger.error(f"Failed to parse event: {e}")
                            continue
                            
        except KeyboardInterrupt:
            logger.info("Consumer interrupted")
        finally:
            self._kafka_consumer.close()
    
    async def process_events(self, topics: List[str]) -> None:
        """Main event processing loop."""
        logger.info(f"Starting event processing for topics: {topics}")
        
        try:
            async for event in self.consume_events(topics):
                # Route event through processors
                for processor_name, processor in self._processors.items():
                    try:
                        result = await processor.process_event(event)
                        if result:
                            # Publish results to analysis topic
                            result_event = StreamEvent(
                                event_id=f"{event.event_id}_processed_{processor_name}",
                                timestamp=datetime.now(),
                                data={
                                    'processor': processor_name,
                                    'original_event': event.dict(),
                                    'result': result
                                },
                                priority=event.priority,
                                source=f"processor_{processor_name}"
                            )
                            await self.publish_event(f"processed_{processor_name}", result_event)
                    except Exception as e:
                        logger.error(f"Processor {processor_name} failed: {e}")
                        continue
                
                # Trigger event handlers
                event_type = event.data.get('event_type', 'default')
                if event_type in self._event_handlers:
                    for handler in self._event_handlers[event_type]:
                        try:
                            await handler(event)
                        except Exception as e:
                            logger.error(f"Event handler failed: {e}")
                            
        except Exception as e:
            logger.error(f"Event processing error: {e}")
            raise
    
    async def create_time_window_aggregation(self, 
                                           events: AsyncGenerator[StreamEvent, None],
                                           window_seconds: int = 60) -> AsyncGenerator[Dict[str, Any], None]:
        """Create time-windowed aggregations from event stream."""
        current_window = []
        window_start = None
        
        async for event in events:
            if window_start is None:
                window_start = event.timestamp
            
            # Check if we need to start a new window
            if (event.timestamp - window_start).total_seconds() >= window_seconds:
                if current_window:
                    # Process current window
                    aggregated = await self._aggregate_window(current_window)
                    yield aggregated
                
                # Start new window
                current_window = [event]
                window_start = event.timestamp
            else:
                current_window.append(event)
    
    async def _aggregate_window(self, events: List[StreamEvent]) -> Dict[str, Any]:
        """Aggregate events within a time window."""
        if not events:
            return {}
        
        # Group events by metric name
        metrics = {}
        for event in events:
            if 'metric_name' in event.data and 'metric_value' in event.data:
                metric_name = event.data['metric_name']
                value = float(event.data['metric_value'])
                
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(value)
        
        # Calculate aggregations for each metric
        aggregations = {}
        for metric_name, values in metrics.items():
            aggregations[metric_name] = {
                'count': len(values),
                'sum': sum(values),
                'mean': np.mean(values),
                'std': np.std(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'median': np.median(values),
                'percentile_95': np.percentile(values, 95) if len(values) > 1 else values[0]
            }
        
        return {
            'window_start': events[0].timestamp,
            'window_end': events[-1].timestamp,
            'event_count': len(events),
            'metrics': aggregations
        }
    
    async def close(self) -> None:
        """Cleanup resources."""
        if self._kafka_consumer:
            self._kafka_consumer.close()
        if self._kafka_producer:
            self._kafka_producer.close()
        logger.info("Streaming processor closed")


# Decorator for complex event processing
def complex_event_processor(pattern: str):
    """Decorator for complex event pattern processing."""
    def decorator(func):
        async def wrapper(self, events: List[StreamEvent]):
            # Implement complex event pattern matching
            # This would include sequence detection, temporal patterns, etc.
            matched_events = []
            
            # Simple pattern matching (can be extended)
            for event in events:
                if event.data.get('pattern_match') == pattern:
                    matched_events.append(event)
            
            if matched_events:
                return await func(self, matched_events)
            return None
        
        return wrapper
    return decorator


class ComplexEventProcessor:
    """Handles complex event processing patterns."""
    
    def __init__(self):
        self._patterns = {}
    
    def register_pattern(self, name: str, pattern_func: Callable) -> None:
        """Register a complex event pattern."""
        self._patterns[name] = pattern_func
    
    async def detect_pattern(self, pattern_name: str, events: List[StreamEvent]) -> bool:
        """Detect if events match a complex pattern."""
        if pattern_name in self._patterns:
            try:
                return await self._patterns[pattern_name](events)
            except Exception as e:
                logger.error(f"Pattern detection failed for {pattern_name}: {e}")
                return False
        return False
    
    async def sequence_detection(self, 
                               events: List[StreamEvent], 
                               sequence_pattern: List[str]) -> bool:
        """Detect if events follow a specific sequence pattern."""
        event_types = [event.data.get('event_type', '') for event in events]
        
        # Look for the sequence in order
        pattern_idx = 0
        for event_type in event_types:
            if pattern_idx < len(sequence_pattern) and event_type == sequence_pattern[pattern_idx]:
                pattern_idx += 1
        
        return pattern_idx == len(sequence_pattern)