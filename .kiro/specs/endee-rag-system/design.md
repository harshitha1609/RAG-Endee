# Endee RAG System - Design Document

## System Architecture

### High-Level Architecture
```
┌─────────────────┐    HTTP/REST    ┌─────────────────┐
│   FastAPI       │ ◄──────────────► │   Endee (ndd)   │
│   Backend       │                 │   Vector DB     │
│                 │                 │   (Docker)      │
└─────────────────┘                 └─────────────────┘
         ▲
         │ HTTP/REST
         ▼
┌─────────────────┐
│   Client        │
│   Applications  │
└─────────────────┘
```

### Component Design

#### 1. FastAPI Backend (app.py)
**Responsibilities:**
- API endpoint management
- Request/response handling
- Service orchestration
- Error handling and logging

**Key Endpoints:**
- `POST /ingest` - Document ingestion
- `POST /search` - Semantic search
- `POST /rag` - RAG query processing
- `GET /health` - Health check

#### 2. Document Ingestion Service (ingest.py)
**Responsibilities:**
- Text preprocessing
- Embedding generation using sentence-transformers
- Endee vector storage via HTTP API
- Metadata management

**Flow:**
```
Raw Text → Preprocessing → Embedding (384d) → Endee Storage
```

#### 3. Vector Search Service (search.py)
**Responsibilities:**
- Query embedding generation
- Endee vector similarity search
- Result ranking and filtering
- Response formatting

**Flow:**
```
Query → Embedding (384d) → Endee Search → Top-K Results
```

#### 4. RAG Service (rag.py)
**Responsibilities:**
- Context retrieval coordination
- Text chunk combination
- LLM integration interface
- Response generation

**Flow:**
```
Query → Vector Search → Context Assembly → LLM → Response
```

## Data Models

### Document Model
```python
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]  # 384 dimensions
    created_at: datetime
```

### Search Result Model
```python
class SearchResult:
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
```

### RAG Response Model
```python
class RAGResponse:
    query: str
    answer: str
    sources: List[SearchResult]
    context_used: str
```

## API Design

### Ingestion Endpoint
```
POST /ingest
Content-Type: application/json

{
    "content": "Document text content",
    "metadata": {
        "title": "Document title",
        "source": "file.txt"
    }
}

Response:
{
    "document_id": "uuid",
    "status": "success",
    "embedding_dimension": 384
}
```

### Search Endpoint
```
POST /search
Content-Type: application/json

{
    "query": "search query text",
    "top_k": 5
}

Response:
{
    "results": [
        {
            "document_id": "uuid",
            "content": "relevant text",
            "score": 0.85,
            "metadata": {...}
        }
    ]
}
```

### RAG Endpoint
```
POST /rag
Content-Type: application/json

{
    "query": "user question",
    "top_k": 3
}

Response:
{
    "query": "user question",
    "answer": "generated response",
    "sources": [...],
    "context_used": "combined context"
}
```

## Endee Integration

### Vector Storage
- **Endpoint:** `POST /vectors/add`
- **Payload:** Vector data with metadata
- **Response:** Storage confirmation

### Vector Search
- **Endpoint:** `POST /vectors/search`
- **Payload:** Query vector and parameters
- **Response:** Similar vectors with scores

### Configuration
- **Host:** Configurable via environment variables
- **Port:** Default 8080
- **Timeout:** 30 seconds for requests
- **Retry:** 3 attempts with exponential backoff

## Embedding Strategy

### Model Configuration
- **Model:** sentence-transformers/all-MiniLM-L6-v2
- **Dimension:** 384
- **Normalization:** L2 normalization
- **Batch Size:** 32 for efficient processing

### Text Processing
1. Text cleaning and normalization
2. Chunking for large documents (512 tokens max)
3. Embedding generation
4. Vector normalization

## Error Handling

### Endee Connection Errors
- Retry with exponential backoff
- Fallback to cached responses if available
- Clear error messages to client

### Embedding Errors
- Model loading validation
- Input text validation
- Dimension verification

### API Errors
- Input validation with clear messages
- HTTP status codes following REST conventions
- Structured error responses

## Performance Considerations

### Caching Strategy
- Embedding model caching in memory
- Query result caching (optional)
- Connection pooling for Endee

### Scalability
- Stateless service design
- Horizontal scaling capability
- Async processing for I/O operations

### Monitoring
- Request/response logging
- Performance metrics
- Health check endpoints

## Security Considerations

### Input Validation
- Text content sanitization
- Query parameter validation
- File size limits

### API Security
- Rate limiting
- Input size restrictions
- Error message sanitization

## Deployment Architecture

### Docker Compose Setup
```yaml
services:
  endee:
    image: endee/ndd:latest
    ports:
      - "8080:8080"
    
  rag-backend:
    build: ./backend
    ports:
      - "8000:8000"
    depends_on:
      - endee
    environment:
      - ENDEE_URL=http://endee:8080
```

### Environment Configuration
- `ENDEE_URL`: Endee service endpoint
- `MODEL_NAME`: Embedding model identifier
- `LOG_LEVEL`: Logging configuration
- `MAX_CONTENT_LENGTH`: Input size limits