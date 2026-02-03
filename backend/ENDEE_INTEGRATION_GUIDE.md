# Endee Vector Database Integration Guide

## Overview

This document provides comprehensive documentation for all Endee vector database integration points in the RAG system. Endee serves as the core vector storage and similarity search engine that enables semantic document retrieval for the Retrieval Augmented Generation pipeline.

## Endee Architecture Integration

### System Architecture
```
┌─────────────────┐    HTTP/REST    ┌─────────────────┐
│   FastAPI       │ ◄──────────────► │   Endee (ndd)   │
│   Backend       │                 │   Vector DB     │
│   (Port 8000)   │                 │   (Port 8081)   │
└─────────────────┘                 └─────────────────┘
         ▲
         │ HTTP/REST
         ▼
┌─────────────────┐
│   Client        │
│   Applications  │
└─────────────────┘
```

### Integration Layers
1. **EndeeHTTPClient** (`endee_client.py`) - Low-level HTTP communication
2. **Service Layer** (`ingest.py`, `search.py`, `rag.py`) - Business logic
3. **API Layer** (`app.py`) - HTTP endpoints and error handling
4. **Docker Layer** (`docker-compose.yml`) - Service orchestration

## Endee API Endpoints

### 1. Vector Add Endpoint
- **URL**: `POST {endee_url}/vectors/add`
- **Purpose**: Store document embeddings with metadata
- **Payload Format**:
  ```json
  {
    "id": "unique-document-id",
    "vector": [384 float values],
    "metadata": {
      "content": "original document text",
      "created_at": "2024-01-01T12:00:00Z",
      "title": "optional document title",
      ...
    }
  }
  ```
- **Response Format**:
  ```json
  {
    "status": "success",
    "id": "unique-document-id"
  }
  ```

### 2. Vector Search Endpoint
- **URL**: `POST {endee_url}/vectors/search`
- **Purpose**: Find similar vectors using cosine similarity
- **Payload Format**:
  ```json
  {
    "vector": [384 float values],
    "top_k": 5,
    "similarity_threshold": 0.0
  }
  ```
- **Response Format**:
  ```json
  {
    "results": [
      {
        "id": "document-id",
        "score": 0.85,
        "metadata": {
          "content": "document text",
          "created_at": "timestamp",
          ...
        }
      }
    ]
  }
  ```

## Endee Integration Points

### 1. Document Ingestion (`ingest.py`)

**Integration Flow**:
```
Text Input → Preprocessing → Embedding (384d) → Endee Storage
```

**Key Methods**:
- `DocumentIngestionService._store_in_endee()`: Stores embeddings in Endee
- Uses `EndeeHTTPClient.add_vector()` for HTTP communication

**Endee Requirements**:
- Vector dimensions: Exactly 384 (all-MiniLM-L6-v2 model)
- Document ID: Non-empty string, unique identifier
- Metadata: Includes original content for retrieval

**Error Handling**:
- `EndeeConnectionError`: Network/server issues
- `EndeeRequestError`: Invalid payload format
- `EndeeTimeoutError`: Request timeouts

### 2. Vector Search (`search.py`)

**Integration Flow**:
```
Query → Embedding (384d) → Endee Search → Ranked Results
```

**Key Methods**:
- `VectorSearchService._perform_endee_search()`: Searches Endee for similar vectors
- Uses direct HTTP requests to Endee search endpoint

**Endee Search Process**:
1. Generate query embedding (384 dimensions)
2. Send search request to Endee
3. Receive ranked results by similarity score
4. Apply client-side threshold filtering
5. Format results for RAG pipeline

**Performance Considerations**:
- Higher top_k values increase search time
- Similarity threshold filtering reduces processing overhead
- Connection pooling minimizes request overhead

### 3. RAG Pipeline (`rag.py`)

**Integration Flow**:
```
Query → Vector Search → Context Combination → LLM Response
```

**Endee Role in RAG**:
- **Document Retrieval**: Finds relevant documents via similarity search
- **Content Extraction**: Retrieves original text from Endee metadata
- **Source Attribution**: Provides document IDs and similarity scores

**Context Preparation**:
1. Search Endee for relevant documents
2. Extract content from Endee metadata
3. Combine documents into coherent context
4. Optimize context for LLM consumption

### 4. HTTP Client (`endee_client.py`)

**Core Functionality**:
- **Connection Management**: Persistent HTTP sessions with pooling
- **Retry Logic**: Exponential backoff for transient failures
- **Error Handling**: Comprehensive error categorization
- **Performance Monitoring**: Operation statistics and logging

**Connection Configuration**:
- Base URL: `ENDEE_URL` environment variable (default: http://localhost:8081)
- Connection pooling: 10 connections per pool, max 20
- Request timeout: 30 seconds
- Retry attempts: 3 with exponential backoff (1s, 2s, 4s, max 10s)

**Error Categories**:
- **Client Errors (4xx)**: Not retried (bad request, payload too large)
- **Server Errors (5xx)**: Retried with backoff
- **Network Errors**: Connection failures, timeouts (retried)

## Configuration

### Environment Variables
- `ENDEE_URL`: Endee service endpoint (default: http://localhost:8081)

### Docker Configuration
```yaml
# docker-compose.yml
services:
  endee:
    image: endee/ndd:latest
    ports:
      - "8081:8081"
    
  rag-backend:
    build: ./backend
    ports:
      - "8000:8000"
    depends_on:
      - endee
    environment:
      - ENDEE_URL=http://endee:8081
```

### Connection Settings
- **Pool Connections**: 10 (configurable in EndeeHTTPClient)
- **Pool Max Size**: 20 (configurable in EndeeHTTPClient)
- **Request Timeout**: 30 seconds
- **Max Retries**: 3 attempts
- **Retry Delays**: 1s, 2s, 4s (exponential backoff)

## Data Flow

### 1. Document Ingestion Flow
```
POST /ingest
    ↓
DocumentIngestionService.ingest_document()
    ↓
Text preprocessing & validation
    ↓
SentenceTransformer.encode() → 384d embedding
    ↓
EndeeHTTPClient.add_vector()
    ↓
POST {endee_url}/vectors/add
    ↓
Endee storage confirmation
    ↓
Return document ID & status
```

### 2. Semantic Search Flow
```
POST /search
    ↓
VectorSearchService.search()
    ↓
Query preprocessing & validation
    ↓
SentenceTransformer.encode() → 384d embedding
    ↓
VectorSearchService._perform_endee_search()
    ↓
POST {endee_url}/vectors/search
    ↓
Endee similarity search results
    ↓
Threshold filtering & formatting
    ↓
Return ranked search results
```

### 3. RAG Pipeline Flow
```
POST /rag
    ↓
RAGService.generate_response()
    ↓
VectorSearchService.search() → Endee search
    ↓
Context combination from Endee results
    ↓
LLM response generation
    ↓
Return answer with sources
```

## Error Handling

### Endee-Specific Errors
- **EndeeConnectionError**: Network connectivity, server errors
- **EndeeRequestError**: Client errors (4xx), invalid payloads
- **EndeeTimeoutError**: Request timeouts, slow responses

### HTTP Status Code Mapping
- `EndeeConnectionError` → 503 Service Unavailable
- `EndeeRequestError` → 400 Bad Request
- `EndeeTimeoutError` → 503 Service Unavailable

### Retry Strategy
1. **Immediate Retry**: For transient network errors
2. **Exponential Backoff**: 1s, 2s, 4s delays
3. **Max Attempts**: 3 total attempts before failure
4. **No Retry**: Client errors (4xx) are not retried

## Performance Monitoring

### Metrics Tracked
- **Operation Counts**: Total calls per operation type
- **Success Rates**: Percentage of successful operations
- **Average Duration**: Mean response times
- **Error Rates**: Failure counts by error type

### Logging
- **Structured Logging**: Context-aware log entries
- **Operation Context**: Document IDs, attempt numbers, durations
- **Performance Metrics**: Request/response times, payload sizes
- **Error Details**: Exception types, retry attempts, failure reasons

### Performance Stats
Access via `EndeeHTTPClient.get_performance_stats()`:
```python
{
  "add_vector": {
    "total_calls": 150,
    "error_count": 2,
    "success_rate": 98.67,
    "avg_duration_ms": 45.2,
    "total_duration_ms": 6780.0
  },
  "search_vectors": {
    "total_calls": 89,
    "error_count": 1,
    "success_rate": 98.88,
    "avg_duration_ms": 78.5,
    "total_duration_ms": 6986.5
  }
}
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check if Endee service is running
   - Verify ENDEE_URL configuration
   - Check Docker container status

2. **Vector Dimension Mismatch**
   - Ensure embeddings are exactly 384 dimensions
   - Verify sentence-transformers model (all-MiniLM-L6-v2)
   - Check embedding generation process

3. **Timeout Errors**
   - Increase request timeout if needed
   - Check Endee service performance
   - Monitor network connectivity

4. **Payload Too Large**
   - Reduce document size or metadata
   - Check 1MB payload limit
   - Implement document chunking if needed

### Debugging Steps

1. **Check Endee Service Health**:
   ```bash
   curl http://localhost:8081/health
   ```

2. **Verify Endee Configuration**:
   ```python
   import os
   print(f"ENDEE_URL: {os.getenv('ENDEE_URL', 'not set')}")
   ```

3. **Monitor Performance Stats**:
   ```python
   client = EndeeHTTPClient()
   stats = client.get_performance_stats()
   print(stats)
   ```

4. **Check Logs**:
   - Look for Endee-related log entries
   - Check error patterns and retry attempts
   - Monitor response times and success rates

## Best Practices

### 1. Connection Management
- Use connection pooling for better performance
- Implement proper session cleanup
- Monitor connection pool utilization

### 2. Error Handling
- Distinguish between retryable and non-retryable errors
- Implement exponential backoff for retries
- Log errors with sufficient context for debugging

### 3. Performance Optimization
- Cache embedding models to avoid reloading
- Use appropriate top_k values for search
- Implement similarity threshold filtering
- Monitor and optimize payload sizes

### 4. Monitoring
- Track operation metrics and success rates
- Monitor response times and error patterns
- Set up alerts for service availability
- Log performance statistics regularly

## Security Considerations

### 1. Network Security
- Use HTTPS in production environments
- Implement proper firewall rules
- Secure inter-service communication

### 2. Data Validation
- Validate all inputs before sending to Endee
- Sanitize metadata to prevent injection
- Implement payload size limits

### 3. Error Information
- Avoid exposing internal Endee details in API responses
- Log sensitive information securely
- Implement proper error message sanitization