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

## 🔑 Key Takeaways

1. **Choose the right pattern** - batch vs real-time based on requirements
2. **Validate early** - catch issues at collection time
3. **Handle errors gracefully** - retry and dead letter queues
4. **Monitor everything** - track collection metrics
5. **Consider privacy** - anonymize and encrypt sensitive data

---

## 📚 Further Reading

- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [AWS Kinesis Best Practices](https://docs.aws.amazon.com/streams/latest/dev/best-practices.html)

---

## 🔗 Related Topics

- [Data Storage](./data-storage.md)
- [Data Versioning](./data-versioning.md)
- [Data Quality](./data-quality.md)
