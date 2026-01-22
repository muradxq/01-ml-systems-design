# Data Collection

## Overview

Data collection is the first step in building ML systems. Effective data collection strategies ensure you have the right data, in the right format, at the right time.

---

## 📊 Data Sources

### 1. User-Generated Data

**Types:**
- User interactions (clicks, views, purchases)
- User feedback (ratings, reviews)
- User profiles (demographics, preferences)

**Collection Methods:**
- Event tracking (analytics platforms)
- Log files (server logs, application logs)
- Databases (user tables, transaction tables)

**Considerations:**
- Privacy and consent
- Data volume
- Real-time requirements

---

### 2. Internal Systems

**Types:**
- Business metrics (sales, revenue)
- Operational data (inventory, logistics)
- System metrics (performance, errors)

**Collection Methods:**
- Database queries
- API integrations
- File exports

**Considerations:**
- Data freshness
- Access permissions
- Schema stability

---

### 3. External Data

**Types:**
- Third-party APIs (weather, financial data)
- Public datasets
- Partner data

**Collection Methods:**
- API calls
- Web scraping
- File downloads

**Considerations:**
- API rate limits
- Data licensing
- Reliability

---

### 4. Sensor/IoT Data

**Types:**
- Device sensors
- IoT devices
- Telemetry data

**Collection Methods:**
- Message queues (Kafka, MQTT)
- Stream processing
- Batch ingestion

**Considerations:**
- High volume
- Real-time processing
- Data quality

---

## 🔄 Collection Patterns

### 1. Batch Collection

**When to Use:**
- Historical data analysis
- Large datasets
- Non-real-time requirements
- Cost optimization

**Approaches:**
- Scheduled jobs (daily, weekly)
- ETL pipelines
- Database exports
- File transfers

**Tools:**
- Apache Airflow, Prefect (orchestration)
- Apache Spark, Pandas (processing)
- AWS Glue, Azure Data Factory (managed)

**Example:**
```python
# Daily batch collection
@daily_schedule
def collect_daily_data():
    # Extract from source
    data = extract_from_database()
    
    # Transform
    cleaned_data = transform(data)
    
    # Load to storage
    load_to_data_lake(cleaned_data)
```

---

### 2. Real-Time Collection

**When to Use:**
- Real-time predictions
- Streaming analytics
- Live monitoring
- User interactions

**Approaches:**
- Event streaming (Kafka, Kinesis)
- Change data capture (CDC)
- API webhooks
- Message queues

**Tools:**
- Apache Kafka, AWS Kinesis (streaming)
- Debezium (CDC)
- Apache Flink, Kafka Streams (processing)

**Example:**
```python
# Real-time event collection
def collect_events():
    consumer = KafkaConsumer('user-events')
    for message in consumer:
        event = parse_message(message)
        validate_event(event)
        store_event(event)
```

---

### 3. Hybrid Collection

**When to Use:**
- Multiple data sources
- Different latency requirements
- Cost optimization

**Approaches:**
- Real-time for critical data
- Batch for historical data
- Lambda architecture

**Example:**
- Real-time: User clicks, purchases
- Batch: Historical sales, reports

---

## 🏗️ Collection Architecture

### Simple Architecture
```
Data Source → Collector → Storage
```

### Scalable Architecture
```
┌──────────────┐
│ Data Sources │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  Message Queue   │  (Kafka, Kinesis)
│  (Buffering)     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   Collectors     │  (Multiple workers)
│   (Processing)   │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   Validation     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   Storage        │  (Data Lake, Warehouse)
└──────────────────┘
```

---

## ✅ Best Practices

### 1. Schema Validation
- Validate data structure early
- Enforce schemas at ingestion
- Handle schema evolution

**Example:**
```python
from pydantic import BaseModel

class UserEvent(BaseModel):
    user_id: int
    event_type: str
    timestamp: datetime
    properties: dict

# Validate at collection
event = UserEvent(**raw_data)
```

---

### 2. Error Handling
- Handle failures gracefully
- Retry mechanisms
- Dead letter queues
- Error logging

**Example:**
```python
def collect_with_retry(data_source, max_retries=3):
    for attempt in range(max_retries):
        try:
            return collect_data(data_source)
        except Exception as e:
            if attempt == max_retries - 1:
                send_to_dlq(data_source, e)
                raise
            time.sleep(2 ** attempt)
```

---

### 3. Monitoring
- Track collection rates
- Monitor errors
- Alert on failures
- Track data quality

**Metrics:**
- Collection volume
- Error rates
- Latency
- Data quality scores

---

### 4. Data Privacy
- Anonymize sensitive data
- Encrypt in transit
- Comply with regulations (GDPR, CCPA)
- Access controls

---

### 5. Cost Optimization
- Batch when possible
- Compress data
- Use appropriate storage tiers
- Monitor costs

---

## 🎯 Collection Strategies by Use Case

### Recommendation Systems
- **Real-time**: User clicks, views, purchases
- **Batch**: Historical interactions, user profiles
- **Frequency**: Continuous for real-time, daily for batch

### Fraud Detection
- **Real-time**: Transactions, login attempts
- **Batch**: Historical patterns, blacklists
- **Frequency**: Continuous

### Time Series Forecasting
- **Real-time**: Latest measurements
- **Batch**: Historical data
- **Frequency**: Continuous for real-time, periodic for batch

---

## 🛠️ Advanced Collection Patterns

### Change Data Capture (CDC)

CDC captures changes in source databases for real-time data synchronization.

```python
from debezium import DebeziumConnector
import json

class CDCDataCollector:
    """Collect data changes from database using CDC."""
    
    def __init__(self, database_config: Dict):
        self.connector = DebeziumConnector(database_config)
        self.buffer = []
        
    def process_change(self, change_event: Dict):
        """Process a single CDC event."""
        operation = change_event['op']  # c=create, u=update, d=delete
        
        if operation == 'c':
            return self._handle_insert(change_event['after'])
        elif operation == 'u':
            return self._handle_update(
                change_event['before'], 
                change_event['after']
            )
        elif operation == 'd':
            return self._handle_delete(change_event['before'])
    
    def _handle_insert(self, record: Dict):
        """Handle new record insertion."""
        validated_record = self.validate_record(record)
        self.buffer.append({
            'operation': 'insert',
            'record': validated_record,
            'timestamp': datetime.now().isoformat()
        })
        
    def _handle_update(self, before: Dict, after: Dict):
        """Handle record update with change tracking."""
        changes = {}
        for key in after:
            if before.get(key) != after.get(key):
                changes[key] = {
                    'old': before.get(key),
                    'new': after.get(key)
                }
        
        self.buffer.append({
            'operation': 'update',
            'record_id': after['id'],
            'changes': changes,
            'timestamp': datetime.now().isoformat()
        })
    
    def flush_buffer(self, destination):
        """Write buffered changes to destination."""
        if self.buffer:
            write_to_storage(destination, self.buffer)
            self.buffer = []
```

### Event Sourcing Pattern

Store all changes as immutable events for complete audit trails.

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import uuid

@dataclass
class Event:
    event_id: str
    event_type: str
    entity_id: str
    timestamp: datetime
    payload: Dict
    metadata: Dict

class EventStore:
    """Event sourcing for ML data collection."""
    
    def __init__(self, storage_backend):
        self.storage = storage_backend
        self.event_handlers = {}
    
    def record_event(self, event_type: str, entity_id: str, 
                     payload: Dict, metadata: Optional[Dict] = None):
        """Record a new event."""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            entity_id=entity_id,
            timestamp=datetime.utcnow(),
            payload=payload,
            metadata=metadata or {}
        )
        
        # Persist event
        self.storage.append(event)
        
        # Notify handlers
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                handler(event)
        
        return event.event_id
    
    def get_entity_history(self, entity_id: str) -> List[Event]:
        """Get complete history for an entity."""
        return self.storage.query(entity_id=entity_id)
    
    def replay_events(self, start_time: datetime, end_time: datetime):
        """Replay events for reprocessing or backfilling."""
        events = self.storage.query_by_time(start_time, end_time)
        for event in events:
            self._process_event(event)

# Usage for ML data collection
event_store = EventStore(storage_backend=S3Storage())

# Record user interaction events
event_store.record_event(
    event_type='user_click',
    entity_id='user_123',
    payload={
        'item_id': 'product_456',
        'page': 'home',
        'position': 3
    },
    metadata={
        'session_id': 'sess_789',
        'device': 'mobile'
    }
)
```

---

## 📊 Data Collection for Specific ML Use Cases

### Recommendation Systems Data Collection

```python
class RecommendationDataCollector:
    """Collect data for recommendation system training."""
    
    def __init__(self, kafka_config: Dict):
        self.consumer = KafkaConsumer(**kafka_config)
        self.interaction_types = {
            'view': 1.0,
            'click': 2.0,
            'add_to_cart': 3.0,
            'purchase': 5.0,
            'rating': lambda x: x  # Use actual rating
        }
    
    def collect_interactions(self):
        """Collect user-item interactions."""
        for message in self.consumer:
            interaction = json.loads(message.value)
            
            processed = {
                'user_id': interaction['user_id'],
                'item_id': interaction['item_id'],
                'interaction_type': interaction['type'],
                'weight': self._get_interaction_weight(interaction),
                'timestamp': interaction['timestamp'],
                'context': {
                    'device': interaction.get('device'),
                    'location': interaction.get('location'),
                    'session_length': interaction.get('session_length')
                }
            }
            
            yield processed
    
    def _get_interaction_weight(self, interaction: Dict) -> float:
        """Calculate implicit feedback weight."""
        int_type = interaction['type']
        if int_type == 'rating':
            return self.interaction_types['rating'](interaction['value'])
        return self.interaction_types.get(int_type, 1.0)
```

### Time Series Data Collection

```python
class TimeSeriesCollector:
    """Collect time series data with proper handling of gaps and anomalies."""
    
    def __init__(self, expected_frequency: str = '1H'):
        self.expected_frequency = expected_frequency
        self.last_timestamp = None
        
    def collect_and_validate(self, data_stream):
        """Collect time series data with gap detection."""
        for record in data_stream:
            timestamp = pd.to_datetime(record['timestamp'])
            
            # Detect gaps
            if self.last_timestamp:
                expected_next = self.last_timestamp + pd.Timedelta(self.expected_frequency)
                if timestamp > expected_next:
                    # Gap detected - handle appropriately
                    gap_records = self._generate_gap_records(
                        self.last_timestamp, timestamp
                    )
                    for gap_record in gap_records:
                        yield gap_record
            
            # Validate and yield current record
            validated_record = self._validate_record(record)
            yield validated_record
            
            self.last_timestamp = timestamp
    
    def _generate_gap_records(self, start: datetime, end: datetime):
        """Generate placeholder records for gaps."""
        timestamps = pd.date_range(start, end, freq=self.expected_frequency)[1:-1]
        for ts in timestamps:
            yield {
                'timestamp': ts,
                'value': None,  # Will be imputed later
                'is_gap': True
            }
    
    def _validate_record(self, record: Dict) -> Dict:
        """Validate individual record."""
        record['is_gap'] = False
        
        # Check for anomalies
        if record.get('value') is not None:
            record['is_anomaly'] = self._is_anomaly(record['value'])
        
        return record
```

---

## 🎯 Interview Questions

**Q1: How would you design a data collection system for a fraud detection ML model?**

**Answer Framework:**
```
1. Data Sources:
   - Real-time transactions (Kafka)
   - User profile data (database CDC)
   - Device fingerprints (API)
   - Historical fraud labels (batch)

2. Collection Strategy:
   - Real-time: Transaction events with <1s latency
   - Near-real-time: User behavior aggregations (5-min windows)
   - Batch: Historical pattern analysis (daily)

3. Key Requirements:
   - Sub-second latency for blocking decisions
   - Complete audit trail for compliance
   - Sampling for training data (handle class imbalance)
   
4. Architecture:
   Transactions → Kafka → Flink → Feature Store → Model
                     ↓
                  Data Lake (for training)
```

**Q2: How do you handle late-arriving data in streaming collection?**

**Answer:**
- Use **watermarks** to define lateness tolerance
- Implement **windowing** with allowed lateness
- Store late data in **side outputs** for later processing
- Use **event time** not processing time
- Consider **eventual consistency** for aggregations

**Q3: What's the difference between CDC and event sourcing?**

| Aspect | CDC | Event Sourcing |
|--------|-----|----------------|
| Source | Database binlog | Application events |
| Granularity | Row changes | Business events |
| Control | Less control | Full control |
| Use case | Sync systems | Audit, replay |

---

## 🔑 Key Takeaways

1. **Choose the right pattern** - batch vs real-time based on requirements
2. **Validate early** - catch issues at collection time
3. **Handle errors gracefully** - retry and dead letter queues
4. **Monitor everything** - track collection metrics
5. **Consider privacy** - anonymize and encrypt sensitive data
6. **Use CDC for sync** - capture database changes efficiently
7. **Event sourcing for audit** - maintain complete history

---

## 📚 Further Reading

- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [AWS Kinesis Best Practices](https://docs.aws.amazon.com/streams/latest/dev/best-practices.html)
- [Streaming Systems](https://www.oreilly.com/library/view/streaming-systems/9781491983867/)
- [Designing Data-Intensive Applications](https://dataintensive.net/)

---

## 🔗 Related Topics

- [Data Storage](./data-storage.md)
- [Data Versioning](./data-versioning.md)
- [Data Quality](./data-quality.md)
