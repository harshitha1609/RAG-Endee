# Endee RAG System

A Retrieval Augmented Generation (RAG) system built with FastAPI and Endee vector database, enabling semantic search and context-aware response generation from ingested documents.

## Overview

This system allows you to:
- Ingest text documents and generate semantic embeddings
- Perform semantic search across your document collection
- Generate contextual responses using retrieved information
- Leverage Endee as a high-performance vector database

## Architecture

### System Overview

The Endee RAG System follows a microservices architecture with clear separation of concerns. The system consists of two main services: a FastAPI backend that handles all business logic and API endpoints, and an Endee vector database that provides high-performance vector storage and similarity search capabilities.

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

### Component Architecture

#### 1. FastAPI Backend (Port 8000)
The backend service orchestrates all RAG operations and provides RESTful API endpoints.

**Core Components:**
- **app.py**: Main FastAPI application with endpoint routing and middleware
- **ingest.py**: Document ingestion pipeline with embedding generation
- **search.py**: Semantic search service using vector similarity
- **rag.py**: RAG pipeline for contextual response generation
- **endee_client.py**: HTTP client for Endee vector database communication

**Key Responsibilities:**
- API endpoint management and request/response handling
- Document preprocessing and embedding generation (384-dimensional vectors)
- Vector similarity search coordination
- Context assembly and RAG response generation
- Error handling, logging, and input validation

#### 2. Endee Vector Database (Port 8081)
High-performance vector database running in Docker container.

**Capabilities:**
- Vector storage with metadata support
- Cosine similarity search
- Batch operations for efficient processing
- RESTful API for vector operations

**Key Endpoints:**
- `POST /vectors/add`: Store document embeddings with metadata
- `POST /vectors/search`: Perform similarity search queries
- `GET /health`: Service health monitoring

### Data Flow Architecture

#### Document Ingestion Flow
```
Raw Text Document
       ↓
Text Preprocessing & Validation
       ↓
Embedding Generation (all-MiniLM-L6-v2)
       ↓
384-Dimensional Vector
       ↓
Endee Vector Storage (with metadata)
       ↓
Document ID Response
```

#### Semantic Search Flow
```
User Query
       ↓
Query Preprocessing
       ↓
Query Embedding Generation
       ↓
Vector Similarity Search (Endee)
       ↓
Result Ranking & Filtering
       ↓
Formatted Search Results
```

#### RAG Pipeline Flow
```
User Question
       ↓
Semantic Search (Top-K Documents)
       ↓
Context Assembly & Optimization
       ↓
LLM Integration Point (Placeholder)
       ↓
Contextual Response with Sources
```

### Technical Specifications

#### Embedding Configuration
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimensions**: 384 (fixed)
- **Normalization**: L2 normalization for optimal similarity search
- **Processing**: Batch processing with configurable batch size (default: 32)
- **Memory**: ~500MB model footprint when loaded

#### Vector Database Integration
- **Protocol**: HTTP/REST communication
- **Timeout**: 30 seconds for requests
- **Retry Logic**: 3 attempts with exponential backoff
- **Connection Pooling**: Persistent connections for efficiency
- **Data Format**: JSON payloads with vector arrays and metadata objects

#### Performance Characteristics
- **Model Loading**: One-time initialization with in-memory caching
- **Embedding Generation**: ~50-200ms per document (depending on length)
- **Vector Search**: Sub-second response times for typical datasets
- **Scalability**: Stateless design enables horizontal scaling
- **Memory Usage**: ~1GB total (model + application overhead)

### Security Architecture

#### Input Validation
- Content length limits (3-50,000 characters)
- Query length validation (2-1,000 characters)
- Metadata size restrictions (max 10KB JSON)
- Parameter range validation (top_k, similarity thresholds)

#### API Security
- CORS middleware configuration
- Request size limitations
- Error message sanitization (no internal details exposed)
- Input sanitization for XSS prevention

#### Network Security
- Service isolation via Docker networking
- Configurable port bindings
- Internal service communication over Docker network
- External API access via defined ports only

### Deployment Architecture

#### Container Strategy
```yaml
# Docker Compose Configuration
services:
  endee:
    image: endee/ndd:latest
    ports:
      - "8081:8080"
    networks:
      - rag-network
    
  rag-backend:
    build: ./backend
    ports:
      - "8000:8000"
    depends_on:
      - endee
    environment:
      - ENDEE_URL=http://endee:8080
    networks:
      - rag-network
```

#### Environment Configuration
- **ENDEE_URL**: Vector database endpoint (default: http://localhost:8081)
- **MODEL_NAME**: Embedding model identifier (default: all-MiniLM-L6-v2)
- **LOG_LEVEL**: Application logging level (INFO, DEBUG, WARNING, ERROR)
- **MAX_CONTENT_LENGTH**: Maximum document size in bytes
- **BATCH_SIZE**: Embedding generation batch size for optimization

#### Scaling Considerations
- **Horizontal Scaling**: Multiple backend instances behind load balancer
- **Database Scaling**: Endee supports clustering for large-scale deployments
- **Resource Requirements**: 4GB RAM minimum, 8GB recommended per instance
- **Storage**: Vector data grows with document collection size

### Integration Points

#### External LLM Integration
The system provides a clean integration point for Large Language Models:
- **Interface**: Standardized context input format
- **Context Optimization**: Automatic text chunking and relevance ranking
- **Source Attribution**: Maintains document provenance for responses
- **Extensibility**: Plugin architecture for different LLM providers

#### API Integration
- **RESTful Design**: Standard HTTP methods and status codes
- **JSON Format**: Consistent request/response structure
- **Interactive Documentation**: Swagger UI and ReDoc available
- **Client Libraries**: Easy integration with any HTTP client

#### Monitoring Integration
- **Health Endpoints**: Service status monitoring
- **Logging**: Structured logging with configurable levels
- **Metrics**: Request/response times, error rates, resource usage
- **Observability**: Ready for integration with monitoring systems

## Features

- **Document Ingestion**: Convert text documents to 384-dimensional embeddings using sentence-transformers
- **Semantic Search**: Find relevant documents using vector similarity search
- **RAG Pipeline**: Generate contextual responses by combining retrieved document chunks
- **REST API**: Clean HTTP endpoints for all operations
- **Docker Integration**: Easy deployment with Docker Compose

## Prerequisites

### System Requirements

**Operating System:**
- Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+, CentOS 7+, or equivalent)
- Minimum 4GB RAM (8GB recommended for optimal performance)
- 2GB free disk space for dependencies and models

**Required Software:**

1. **Python 3.8 or higher** (3.9+ recommended)
   - Download from [python.org](https://www.python.org/downloads/)
   - Verify installation: `python --version` or `python3 --version`
   - Ensure pip is installed: `pip --version`

2. **Docker and Docker Compose**
   - **Docker Desktop** (recommended for Windows/macOS): [docker.com/get-started](https://www.docker.com/get-started)
   - **Docker Engine + Docker Compose** (Linux): Follow [Docker installation guide](https://docs.docker.com/engine/install/)
   - Verify installation: `docker --version` and `docker-compose --version`
   - Ensure Docker daemon is running

3. **Git** (for cloning the repository)
   - Download from [git-scm.com](https://git-scm.com/downloads)
   - Verify installation: `git --version`

4. **curl** (for testing API endpoints)
   - Usually pre-installed on macOS/Linux
   - Windows: Available in PowerShell or install via [curl.se](https://curl.se/windows/)

### Network Requirements

- **Internet connection** required for:
  - Initial setup and dependency downloads
  - Downloading the sentence-transformers model (~90MB)
  - Docker image pulls

- **Port availability:**
  - Port 8000: FastAPI backend
  - Port 8081: Endee vector database (mapped from container port 8080)

### Hardware Recommendations

**Minimum:**
- 2 CPU cores
- 4GB RAM
- 2GB free disk space

**Recommended:**
- 4+ CPU cores
- 8GB+ RAM
- 5GB+ free disk space

**Note:** The sentence-transformers model requires additional memory during loading (~500MB)

## Quick Start

### Step 1: Clone and Navigate to Project

```bash
# Clone the repository
git clone <repository-url>
cd endee-rag-system

# Verify project structure
ls -la
```

**Expected output:**
```
backend/    # Python FastAPI application
docker/     # Docker configuration for Endee
README.md   # This documentation
```

### Step 2: Start Endee Vector Database

The system uses Endee as the vector database, which runs in a Docker container.

```bash
# Navigate to docker directory
cd docker

# Start Endee service in background
docker-compose up -d

# Verify Endee is running
docker-compose ps
```

**Expected output:**
```
NAME              COMMAND                  SERVICE   STATUS    PORTS
endee-vector-db   "python mock_endee.py"   endee     running   0.0.0.0:8081->8080/tcp
```

**Verify Endee is accessible:**
```bash
# Test health endpoint (may take 30-60 seconds to be ready)
curl http://localhost:8081/health

# Expected response:
# {"status": "healthy", "service": "Endee Mock", "version": "1.0.0"}
```

**Troubleshooting Endee startup:**
- If port 8081 is in use, modify `docker-compose.yml` to use a different port
- Check Docker logs: `docker-compose logs endee`
- Ensure Docker daemon is running: `docker info`

### Step 3: Set Up Python Environment

**Option A: Using Virtual Environment (Recommended)**
```bash
# Navigate to backend directory
cd ../backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Option B: System-wide Installation**
```bash
cd backend
pip install -r requirements.txt
```

**Verify installation:**
```bash
# Check key dependencies
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
python -c "import sentence_transformers; print('sentence-transformers installed')"
python -c "import torch; print('PyTorch:', torch.__version__)"
```

### Step 4: Download Embedding Model

The system will automatically download the embedding model on first use, but you can pre-download it:

```bash
# Pre-download the model (optional but recommended)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Note:** This downloads ~90MB and may take 1-2 minutes depending on your internet connection.

### Step 5: Start the FastAPI Backend

```bash
# Ensure you're in the backend directory
cd backend  # if not already there

# Start the API server
python app.py
```

**Expected output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Verify the API is running:**
```bash
# In a new terminal window
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "timestamp": "...", "version": "1.0.0"}
```

### Step 6: Test the System

**Test document ingestion:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The quick brown fox jumps over the lazy dog. This is a test document for the RAG system.",
    "metadata": {
      "title": "Test Document",
      "source": "quick_start_test"
    }
  }'
```

**Test semantic search:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "brown fox jumping",
    "top_k": 5
  }'
```

**Test RAG query:**
```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What animal jumps over another animal?",
    "top_k": 3
  }'
```

### Verification Checklist

✅ **System is ready when all of these work:**

1. **Docker service:** `curl http://localhost:8081/health` returns healthy status
2. **API service:** `curl http://localhost:8000/health` returns healthy status  
3. **Document ingestion:** POST to `/ingest` returns success with document_id
4. **Search functionality:** POST to `/search` returns relevant results
5. **RAG pipeline:** POST to `/rag` returns contextual response

### Next Steps

Once setup is complete:
1. **Explore the API:** Visit `http://localhost:8000/docs` for interactive documentation
2. **Ingest your documents:** Use the `/ingest` endpoint to add your content
3. **Test semantic search:** Query your documents using the `/search` endpoint
4. **Try RAG queries:** Ask questions using the `/rag` endpoint

## System Verification

After completing the setup, run these verification steps to ensure everything is working correctly:

### 1. Prerequisites Check

```bash
# Verify Python version (should be 3.8+)
python --version

# Verify pip is working
pip --version

# Verify Docker is running
docker info

# Verify Docker Compose is available
docker-compose --version

# Check available disk space (need ~2GB)
df -h .  # Linux/macOS
dir     # Windows
```

### 2. Service Health Check

```bash
# Check Endee vector database
curl http://localhost:8081/health
# Expected: {"status": "healthy", "service": "Endee Mock", "version": "1.0.0"}

# Check FastAPI backend
curl http://localhost:8000/health  
# Expected: {"status": "healthy", "timestamp": "...", "version": "1.0.0"}

# Check Docker container status
cd docker && docker-compose ps
# Expected: endee service should be "running"
```

### 3. Dependency Verification

```bash
cd backend

# Test Python imports
python -c "
import fastapi
import sentence_transformers  
import torch
import requests
import numpy
print('✅ All core dependencies imported successfully')
"

# Test embedding model loading
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f'✅ Model loaded: {model.get_sentence_embedding_dimension()} dimensions')
"
```

### 4. End-to-End System Test

Run this complete workflow test:

```bash
# 1. Ingest a test document
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Artificial intelligence and machine learning are transforming how we process and understand data.",
    "metadata": {"title": "AI Overview", "source": "verification_test"}
  }'

# Expected response should include: "status": "success", "document_id": "..."

# 2. Search for relevant content  
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning data processing",
    "top_k": 3
  }'

# Expected: Should return the ingested document with high similarity score

# 3. Test RAG functionality
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How is AI transforming data processing?",
    "top_k": 2
  }'

# Expected: Should return contextual answer based on ingested document
```

### 5. Performance Verification

```bash
# Check response times (should be under 2 seconds for small documents)
time curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "top_k": 5}'

# Check memory usage
docker stats endee-vector-db --no-stream

# Check API server memory (if running in foreground, check terminal output)
```

### 6. Interactive API Testing

Visit these URLs in your browser:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces should load without errors and allow you to test all endpoints interactively.

### Verification Checklist

Mark each item as complete:

- [ ] **Prerequisites**: Python 3.8+, Docker, and Docker Compose installed
- [ ] **Services Running**: Both Endee (8081) and FastAPI (8000) respond to health checks
- [ ] **Dependencies**: All Python packages import without errors
- [ ] **Model Loading**: sentence-transformers model loads and returns 384 dimensions
- [ ] **Document Ingestion**: Can successfully ingest documents via API
- [ ] **Semantic Search**: Search returns relevant results with similarity scores
- [ ] **RAG Pipeline**: RAG endpoint returns contextual responses
- [ ] **Interactive Docs**: Swagger UI and ReDoc load correctly
- [ ] **Performance**: API responses complete within reasonable time (< 5 seconds)

### Troubleshooting Failed Verification

If any verification step fails:

1. **Review the error message** and check the troubleshooting section
2. **Check service logs**:
   ```bash
   # FastAPI logs (in terminal where app.py runs)
   # Docker logs
   docker-compose logs endee
   ```
3. **Restart services**:
   ```bash
   # Restart Endee
   cd docker && docker-compose restart endee
   
   # Restart FastAPI (Ctrl+C and run python app.py again)
   ```
4. **Verify network connectivity** between services
5. **Check system resources** (memory, disk space)

Once all verification steps pass, your Endee RAG System is ready for use!

## API Testing Examples

This section provides comprehensive examples for testing all API endpoints with various scenarios and use cases.

### Prerequisites for Testing

Ensure both services are running:
```bash
# Check Endee vector database
curl http://localhost:8081/health

# Check FastAPI backend  
curl http://localhost:8000/health
```

### Complete Workflow Example

Here's a complete example demonstrating the full RAG pipeline:

#### Step 1: Ingest Sample Documents

```bash
# Ingest first document about AI
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. Machine learning, a subset of AI, enables systems to automatically learn and improve from experience without being explicitly programmed.",
    "metadata": {
      "title": "Introduction to Artificial Intelligence",
      "source": "ai_basics.txt",
      "category": "technology",
      "author": "Dr. Smith"
    }
  }'

# Ingest second document about machine learning
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Machine learning algorithms can be categorized into three main types: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Unsupervised learning finds hidden patterns in data without labeled examples. Reinforcement learning learns through interaction with an environment using rewards and penalties.",
    "metadata": {
      "title": "Types of Machine Learning",
      "source": "ml_types.txt",
      "category": "education",
      "author": "Prof. Johnson"
    }
  }'

# Ingest third document about data science
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Data science is an interdisciplinary field that combines statistics, mathematics, programming, and domain expertise to extract insights from data. Data scientists use various tools and techniques including data mining, machine learning, and statistical analysis to solve complex business problems. The data science process typically involves data collection, cleaning, exploration, modeling, and interpretation of results.",
    "metadata": {
      "title": "Data Science Overview",
      "source": "data_science.txt",
      "category": "analytics",
      "author": "Dr. Williams"
    }
  }'
```

#### Step 2: Test Semantic Search

```bash
# Search for AI-related content
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence machine learning",
    "top_k": 3,
    "similarity_threshold": 0.2
  }'

# Search for learning algorithms
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "supervised unsupervised learning algorithms",
    "top_k": 2,
    "similarity_threshold": 0.3
  }'

# Search for data analysis
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "data analysis statistical methods",
    "top_k": 5,
    "similarity_threshold": 0.1
  }'
```

#### Step 3: Test RAG Queries

```bash
# Ask about AI fundamentals
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence and how does it relate to machine learning?",
    "top_k": 2,
    "similarity_threshold": 0.2
  }'

# Ask about learning types
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the different types of machine learning algorithms?",
    "top_k": 3,
    "similarity_threshold": 0.1
  }'

# Ask about data science process
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does the data science process work?",
    "top_k": 2,
    "max_context_length": 2000
  }'
```

### Edge Case Testing

#### Testing Input Validation

```bash
# Test minimum content length (should fail)
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Hi"
  }'
# Expected: 400 Bad Request - content too short

# Test maximum content length (should fail)
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "'$(python -c "print('x' * 50001)")'",
    "metadata": {"title": "Too Long"}
  }'
# Expected: 400 Bad Request - content too long

# Test invalid top_k (should fail)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query",
    "top_k": 101
  }'
# Expected: 422 Validation Error - top_k too high

# Test invalid similarity threshold (should fail)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query",
    "similarity_threshold": 1.5
  }'
# Expected: 422 Validation Error - threshold out of range
```

#### Testing Empty Results

```bash
# Search with very high threshold (likely no results)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "completely unrelated topic like underwater basket weaving",
    "top_k": 5,
    "similarity_threshold": 0.9
  }'

# RAG query with no relevant context
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of Mars?",
    "top_k": 3,
    "similarity_threshold": 0.8
  }'
```

### Performance Testing

#### Batch Document Ingestion

```bash
# Ingest multiple documents quickly
for i in {1..5}; do
  curl -X POST http://localhost:8000/ingest \
    -H "Content-Type: application/json" \
    -d "{
      \"content\": \"This is test document number $i with some sample content about technology and innovation. Document $i contains information about various topics including software development, data analysis, and system architecture.\",
      \"metadata\": {
        \"title\": \"Test Document $i\",
        \"source\": \"batch_test_$i.txt\",
        \"batch_id\": \"performance_test\"
      }
    }" &
done
wait
```

#### Concurrent Search Testing

```bash
# Run multiple searches concurrently
for i in {1..3}; do
  curl -X POST http://localhost:8000/search \
    -H "Content-Type: application/json" \
    -d '{
      "query": "technology innovation software",
      "top_k": 5
    }' &
done
wait
```

### Testing with Different Content Types

#### Technical Documentation

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "REST APIs (Representational State Transfer Application Programming Interfaces) are architectural styles for designing networked applications. They use standard HTTP methods like GET, POST, PUT, and DELETE to perform operations on resources. RESTful services are stateless, meaning each request contains all necessary information. Common response formats include JSON and XML, with JSON being preferred for modern applications.",
    "metadata": {
      "title": "REST API Fundamentals",
      "source": "api_guide.md",
      "category": "documentation",
      "type": "technical"
    }
  }'
```

#### Scientific Content

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Photosynthesis is the biological process by which plants, algae, and certain bacteria convert light energy into chemical energy stored in glucose molecules. This process occurs in two main stages: the light-dependent reactions (occurring in the thylakoids) and the light-independent reactions or Calvin cycle (occurring in the stroma). The overall equation for photosynthesis is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2.",
    "metadata": {
      "title": "Photosynthesis Process",
      "source": "biology_textbook.pdf",
      "category": "science",
      "subject": "biology",
      "chapter": 8
    }
  }'
```

#### Business Content

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Project management methodologies provide structured approaches to planning, executing, and completing projects. Agile methodology emphasizes iterative development, customer collaboration, and responding to change. Waterfall methodology follows a sequential approach with distinct phases. Scrum is an Agile framework that uses sprints, daily standups, and retrospectives to manage development cycles.",
    "metadata": {
      "title": "Project Management Methodologies",
      "source": "pm_handbook.docx",
      "category": "business",
      "department": "operations"
    }
  }'
```

### Advanced Query Examples

#### Cross-Domain Queries

```bash
# Query spanning multiple domains
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How can machine learning be applied to improve project management processes?",
    "top_k": 4,
    "similarity_threshold": 0.1
  }'

# Technical implementation query
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the best practices for designing REST APIs for machine learning services?",
    "top_k": 3,
    "max_context_length": 3000
  }'
```

#### Specific Detail Queries

```bash
# Query for specific processes
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the Calvin cycle in photosynthesis",
    "top_k": 2,
    "similarity_threshold": 0.3
  }'

# Query for comparisons
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare Agile and Waterfall project management approaches",
    "top_k": 3,
    "similarity_threshold": 0.2
  }'
```

### Testing Response Formats

#### Verify JSON Structure

```bash
# Test and format JSON response
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence",
    "top_k": 2
  }' | python -m json.tool

# Test RAG response structure
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 2
  }' | python -m json.tool
```

#### Extract Specific Fields

```bash
# Extract only document IDs from search results
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "data science",
    "top_k": 3
  }' | python -c "
import json, sys
data = json.load(sys.stdin)
for result in data['results']:
    print(f\"ID: {result['document_id']}, Score: {result['score']:.4f}\")
"

# Extract answer and source count from RAG response
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does supervised learning work?",
    "top_k": 2
  }' | python -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Answer: {data['answer'][:100]}...\")
print(f\"Sources used: {data['sources_count']}\")
print(f\"Context length: {data['context_length']} characters\")
"
```

### Automated Testing Script

Create a test script to verify all functionality:

```bash
#!/bin/bash
# save as test_api.sh

echo "=== Endee RAG System API Test Suite ==="

# Test health endpoints
echo "1. Testing health endpoints..."
curl -s http://localhost:8000/health | grep -q "healthy" && echo "✅ API health check passed" || echo "❌ API health check failed"
curl -s http://localhost:8081/health | grep -q "healthy" && echo "✅ Endee health check passed" || echo "❌ Endee health check failed"

# Test document ingestion
echo "2. Testing document ingestion..."
INGEST_RESPONSE=$(curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Test document for automated testing with sample content about technology.",
    "metadata": {"title": "Test Doc", "source": "test.txt"}
  }')

if echo "$INGEST_RESPONSE" | grep -q "success"; then
    echo "✅ Document ingestion passed"
    DOC_ID=$(echo "$INGEST_RESPONSE" | python -c "import json,sys; print(json.load(sys.stdin)['document_id'])")
else
    echo "❌ Document ingestion failed"
    exit 1
fi

# Test search
echo "3. Testing search..."
SEARCH_RESPONSE=$(curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "technology testing", "top_k": 5}')

if echo "$SEARCH_RESPONSE" | grep -q "results"; then
    echo "✅ Search functionality passed"
else
    echo "❌ Search functionality failed"
fi

# Test RAG
echo "4. Testing RAG..."
RAG_RESPONSE=$(curl -s -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "top_k": 3}')

if echo "$RAG_RESPONSE" | grep -q "answer"; then
    echo "✅ RAG functionality passed"
else
    echo "❌ RAG functionality failed"
fi

echo "=== Test Suite Complete ==="
```

Make it executable and run:
```bash
chmod +x test_api.sh
./test_api.sh
```

## Project Structure

```
backend/
├── app.py              # FastAPI entry point
├── ingest.py           # Document ingestion logic
├── search.py           # Vector search logic
├── rag.py              # RAG logic
└── requirements.txt    # Python dependencies

docker/
├── docker-compose.yml  # Endee service configuration
├── manage.bat          # Windows management script
├── manage.sh           # Unix management script
└── README.md           # Docker setup instructions

README.md               # This file
```

## Configuration

### Environment Variables

- `ENDEE_URL`: Endee service endpoint (default: `http://localhost:8081`)
- `MODEL_NAME`: Embedding model name (default: `all-MiniLM-L6-v2`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `MAX_CONTENT_LENGTH`: Maximum input size in bytes

### Embedding Model

The system uses `sentence-transformers/all-MiniLM-L6-v2` which generates 384-dimensional embeddings. This model provides a good balance of performance and accuracy for semantic search tasks.

## API Reference

The Endee RAG System provides a RESTful API with the following endpoints. All endpoints return JSON responses and use standard HTTP status codes.

**Base URL:** `http://localhost:8000`

**Interactive Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### GET /

Root endpoint providing basic API information.

**Example Request:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "message": "Endee RAG System API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

**Status Codes:**
- `200`: Success

---

### GET /health

Health check endpoint to verify API status and system health.

**Example Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "version": "1.0.0"
}
```

**Status Codes:**
- `200`: API is healthy and operational

---

### POST /ingest

Ingest a text document into the system for semantic search and RAG operations.

This endpoint processes raw text content, generates 384-dimensional embeddings using the sentence-transformers model, and stores the document in the Endee vector database with associated metadata.

**Request Body:**
```json
{
  "content": "Document text content",
  "metadata": {
    "title": "Document title",
    "source": "file.txt",
    "author": "John Doe",
    "category": "technical"
  }
}
```

**Parameters:**
- `content` (required, string): Document text content (3-50,000 characters)
- `metadata` (optional, object): Document metadata (max 10KB JSON object)

**Example Requests:**

**Basic document ingestion:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Artificial intelligence is transforming the way we process and understand data. Machine learning algorithms can identify patterns in large datasets that would be impossible for humans to detect manually.",
    "metadata": {
      "title": "AI and Data Processing",
      "source": "tech_article.txt",
      "category": "technology"
    }
  }'
```

**Minimal ingestion (content only):**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The quick brown fox jumps over the lazy dog."
  }'
```

**Large document with rich metadata:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Climate change represents one of the most significant challenges of our time. Rising global temperatures, changing precipitation patterns, and increasing frequency of extreme weather events are already impacting ecosystems, agriculture, and human societies worldwide. The scientific consensus is clear: human activities, particularly the emission of greenhouse gases from burning fossil fuels, are the primary drivers of current climate change.",
    "metadata": {
      "title": "Climate Change Overview",
      "source": "climate_report_2024.pdf",
      "author": "Dr. Jane Smith",
      "category": "environment",
      "publication_date": "2024-01-15",
      "tags": ["climate", "environment", "science"],
      "page_number": 1
    }
  }'
```

**Success Response:**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "success",
  "embedding_dimension": 384,
  "content_length": 156,
  "metadata": {
    "title": "AI and Data Processing",
    "source": "tech_article.txt",
    "category": "technology"
  }
}
```

**Status Codes:**
- `200`: Document successfully ingested
- `400`: Invalid request (content too short/long, invalid metadata)
- `422`: Request validation failed
- `500`: Embedding generation failed
- `503`: Endee vector database unavailable

---

### POST /search

Perform semantic search across all ingested documents using vector similarity.

This endpoint converts the query text to embeddings and searches for the most similar documents in the Endee vector database. Results are ranked by similarity score and can be filtered by a minimum threshold.

**Request Body:**
```json
{
  "query": "search query text",
  "top_k": 5,
  "similarity_threshold": 0.3
}
```

**Parameters:**
- `query` (required, string): Search query text (2-1,000 characters)
- `top_k` (optional, integer): Number of top results to return (1-100, default: 5)
- `similarity_threshold` (optional, float): Minimum similarity score (0.0-1.0, default: 0.0)

**Example Requests:**

**Basic semantic search:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning artificial intelligence",
    "top_k": 3
  }'
```

**Search with similarity threshold:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "climate change environmental impact",
    "top_k": 10,
    "similarity_threshold": 0.4
  }'
```

**Precise search (high threshold):**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "data processing algorithms",
    "top_k": 5,
    "similarity_threshold": 0.7
  }'
```

**Success Response:**
```json
{
  "query": "machine learning artificial intelligence",
  "results": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "content": "Artificial intelligence is transforming the way we process and understand data. Machine learning algorithms can identify patterns in large datasets that would be impossible for humans to detect manually.",
      "score": 0.8547,
      "metadata": {
        "title": "AI and Data Processing",
        "source": "tech_article.txt",
        "category": "technology"
      }
    },
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440001",
      "content": "Deep learning neural networks have revolutionized computer vision and natural language processing tasks.",
      "score": 0.7234,
      "metadata": {
        "title": "Deep Learning Applications",
        "source": "ml_paper.pdf",
        "category": "research"
      }
    }
  ],
  "total_results": 2,
  "similarity_threshold": 0.0
}
```

**Empty Results Response:**
```json
{
  "query": "quantum computing",
  "results": [],
  "total_results": 0,
  "similarity_threshold": 0.8
}
```

**Status Codes:**
- `200`: Search completed successfully (may return empty results)
- `400`: Invalid request (query too short/long, invalid parameters)
- `422`: Request validation failed
- `500`: Query embedding generation failed
- `503`: Endee vector database unavailable

---

### POST /rag

Generate contextual answers using Retrieval Augmented Generation (RAG).

This endpoint performs the complete RAG pipeline: retrieves relevant documents using semantic search, combines them into coherent context, and generates responses using the retrieved information. The response includes source attribution and context details.

**Request Body:**
```json
{
  "query": "user question",
  "top_k": 3,
  "similarity_threshold": 0.2,
  "max_context_length": 4000
}
```

**Parameters:**
- `query` (required, string): User question (2-1,000 characters)
- `top_k` (optional, integer): Number of documents to retrieve (1-20, default: 3)
- `similarity_threshold` (optional, float): Minimum similarity score (0.0-1.0, default: 0.0)
- `max_context_length` (optional, integer): Maximum context length in characters (100-50,000, default: 4000)

**Example Requests:**

**Basic RAG query:**
```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does machine learning help with data processing?",
    "top_k": 3
  }'
```

**RAG with custom parameters:**
```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main impacts of climate change?",
    "top_k": 5,
    "similarity_threshold": 0.3,
    "max_context_length": 2000
  }'
```

**Focused RAG query (high threshold):**
```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain deep learning neural networks",
    "top_k": 2,
    "similarity_threshold": 0.6,
    "max_context_length": 1500
  }'
```

**Success Response:**
```json
{
  "query": "How does machine learning help with data processing?",
  "answer": "Based on the retrieved documents, machine learning significantly enhances data processing by enabling automated pattern recognition in large datasets. Machine learning algorithms can identify complex patterns and relationships that would be impossible for humans to detect manually, making data processing more efficient and accurate. These algorithms can process vast amounts of information quickly and extract meaningful insights from raw data.",
  "sources": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "score": 0.8547,
      "metadata": {
        "title": "AI and Data Processing",
        "source": "tech_article.txt",
        "category": "technology"
      },
      "content_preview": "Artificial intelligence is transforming the way we process and understand data. Machine learning algorithms can identify patterns..."
    },
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440001",
      "score": 0.7234,
      "metadata": {
        "title": "Deep Learning Applications",
        "source": "ml_paper.pdf",
        "category": "research"
      },
      "content_preview": "Deep learning neural networks have revolutionized computer vision and natural language processing tasks..."
    }
  ],
  "context_used": "Artificial intelligence is transforming the way we process and understand data. Machine learning algorithms can identify patterns in large datasets that would be impossible for humans to detect manually. Deep learning neural networks have revolutionized computer vision and natural language processing tasks.",
  "context_length": 287,
  "sources_count": 2,
  "truncated": false,
  "max_context_length": 4000
}
```

**No Relevant Sources Response:**
```json
{
  "query": "What is quantum computing?",
  "answer": "I don't have enough relevant information in the knowledge base to answer this question about quantum computing. Please try ingesting relevant documents first or rephrase your question.",
  "sources": [],
  "context_used": "",
  "context_length": 0,
  "sources_count": 0,
  "truncated": false,
  "max_context_length": 4000
}
```

**Truncated Context Response:**
```json
{
  "query": "Tell me about artificial intelligence",
  "answer": "Based on the available context, artificial intelligence is transforming data processing and analysis. AI systems can identify patterns in large datasets and automate complex tasks that were previously manual...",
  "sources": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "score": 0.8547,
      "metadata": {
        "title": "AI Overview",
        "source": "ai_comprehensive.txt"
      },
      "content_preview": "Artificial intelligence encompasses machine learning, deep learning, natural language processing..."
    }
  ],
  "context_used": "Artificial intelligence encompasses machine learning, deep learning, natural language processing, computer vision, and robotics. These technologies are transforming industries by automating complex tasks...",
  "context_length": 1000,
  "sources_count": 1,
  "truncated": true,
  "max_context_length": 1000
}
```

**Status Codes:**
- `200`: RAG processing completed successfully
- `400`: Invalid request (query too short/long, invalid parameters)
- `422`: Request validation failed
- `500`: Context combination failed or RAG pipeline error
- `503`: Endee vector database unavailable

## Error Handling

The API returns structured error responses with appropriate HTTP status codes:

### Error Response Format

```json
{
  "error": "Error Type",
  "detail": "Detailed error message",
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

### HTTP Status Codes

- **200**: Success
- **400**: Bad Request (validation errors, invalid input)
- **422**: Unprocessable Entity (request validation failed)
- **500**: Internal Server Error (embedding generation, context combination failures)
- **503**: Service Unavailable (Endee vector database unavailable)

### Common Error Scenarios

- **Content too short/long**: Content must be 3-50,000 characters
- **Query too short/long**: Query must be 2-1,000 characters
- **Invalid top_k**: Must be 1-100 for search, 1-20 for RAG
- **Invalid similarity_threshold**: Must be between 0.0 and 1.0
- **Endee connection failed**: Vector database is not accessible
- **Model loading failed**: Embedding model could not be loaded

## Troubleshooting

### Prerequisites Issues

**Python Version Problems:**
```bash
# Check Python version
python --version
# or
python3 --version

# If Python < 3.8, install newer version from python.org
```

**Docker Issues:**
```bash
# Check if Docker is running
docker info

# If Docker daemon not running:
# Windows/macOS: Start Docker Desktop
# Linux: sudo systemctl start docker

# Check Docker Compose version
docker-compose --version
# Should be 1.25.0 or higher
```

**Permission Issues (Linux/macOS):**
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Log out and back in

# Or run with sudo (not recommended for production)
sudo docker-compose up -d
```

### Endee Connection Issues

1. **Service not running**: Ensure Docker Compose is up
   ```bash
   cd docker
   docker-compose ps
   ```

2. **Port conflicts**: Check if port 8081 is available
   ```bash
   # Windows
   netstat -an | findstr 8081
   # macOS/Linux  
   netstat -an | grep 8081
   ```

3. **Health check fails**: Verify Endee is responding
   ```bash
   curl http://localhost:8081/health
   ```

4. **Container startup issues**: Check Docker logs
   ```bash
   cd docker
   docker-compose logs endee
   ```

5. **Firewall blocking connections**: Ensure ports 8000 and 8081 are not blocked

### Backend Issues

**Dependency Installation Problems:**
```bash
# Upgrade pip first
pip install --upgrade pip

# Clear pip cache and reinstall
pip cache purge
pip install -r backend/requirements.txt --no-cache-dir

# For PyTorch issues on older systems
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Model Download Issues:**
```bash
# Test internet connectivity
curl -I https://huggingface.co

# Manual model download with verbose output
python -c "
from sentence_transformers import SentenceTransformer
import logging
logging.basicConfig(level=logging.INFO)
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Model loaded successfully')
"

# Check model cache location
python -c "
from sentence_transformers import SentenceTransformer
print('Cache dir:', SentenceTransformer._get_cache_folder())
"
```

**Port 8000 Already in Use:**
```bash
# Find process using port 8000
# Windows
netstat -ano | findstr :8000
# macOS/Linux
lsof -i :8000

# Kill the process (replace PID with actual process ID)
# Windows
taskkill /PID <PID> /F
# macOS/Linux
kill -9 <PID>

# Or change port in app.py
# Edit: uvicorn.run(app, host="0.0.0.0", port=8001)
```

**Virtual Environment Issues:**
```bash
# Recreate virtual environment
rm -rf venv  # or rmdir /s venv on Windows
python -m venv venv

# Activate and install
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### API Testing Issues

**Connection Refused Errors:**
```bash
# Verify services are running
curl http://localhost:8000/health  # FastAPI
curl http://localhost:8081/health  # Endee

# Check if services are bound to correct interfaces
netstat -tlnp | grep 8000  # Linux
netstat -an | grep 8000    # macOS
netstat -an | findstr 8000 # Windows
```

**SSL/TLS Certificate Errors:**
```bash
# Use -k flag to ignore SSL issues during testing
curl -k http://localhost:8000/health

# Or use http instead of https
curl http://localhost:8000/health
```

**JSON Parsing Errors:**
```bash
# Ensure proper JSON formatting
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"content": "test", "metadata": {}}'

# Use single quotes on Windows Command Prompt
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d "{\"content\": \"test\", \"metadata\": {}}"
```

### Performance Issues

**Slow Model Loading:**
- First-time model download can take 2-5 minutes
- Subsequent loads should be under 10 seconds
- Consider pre-downloading: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`

**High Memory Usage:**
- Model requires ~500MB RAM when loaded
- Reduce batch size if running on limited memory
- Consider using CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

**Slow API Responses:**
- Check if Endee container has sufficient resources
- Monitor Docker stats: `docker stats endee-vector-db`
- Increase Docker memory limits if needed

### Common Error Messages

**"Connection refused"**: 
- Endee service is not running or not accessible
- Check `docker-compose ps` and `curl http://localhost:8081/health`

**"Model not found"**: 
- sentence-transformers model download failed
- Check internet connection and retry model download

**"Invalid embedding dimension"**: 
- Model mismatch, ensure using all-MiniLM-L6-v2
- Clear model cache and re-download

**"Timeout error"**: 
- Increase request timeout or check network connectivity
- Verify both services are responding to health checks

**"Permission denied"**: 
- Docker permission issues (Linux/macOS)
- File permission issues in project directory

### Data and Storage Issues

**Embedding Dimension Mismatch:**
```bash
# Verify model produces 384-dimensional embeddings
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f'Embedding dimension: {model.get_sentence_embedding_dimension()}')
"
# Should output: Embedding dimension: 384
```

**Document Storage Issues:**
```bash
# Test Endee vector storage directly
curl -X POST http://localhost:8081/vectors/add \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test-vector",
    "vector": [0.1, 0.2, 0.3],
    "metadata": {"test": true}
  }'
```

**Search Result Quality Issues:**
- Low similarity scores may indicate poor document-query matching
- Try different query phrasings or synonyms
- Consider ingesting more relevant documents
- Adjust similarity threshold (lower values = more results)

### Environment-Specific Issues

**Windows-Specific:**
```cmd
# Use Windows-style paths and commands
cd backend
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt

# PowerShell execution policy issues
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS-Specific:**
```bash
# Install Python via Homebrew if needed
brew install python@3.9

# Use python3 explicitly
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

**Linux-Specific:**
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install python3-pip python3-venv docker.io docker-compose

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

### Advanced Troubleshooting

**Memory and Resource Issues:**
```bash
# Monitor system resources
# Linux/macOS
top -p $(pgrep -f "python app.py")
htop

# Windows
tasklist /fi "imagename eq python.exe"
```

**Network Connectivity Issues:**
```bash
# Test internal Docker networking
docker exec -it endee-vector-db curl http://localhost:8080/health

# Test from host to container
curl http://localhost:8081/health

# Check Docker network
docker network ls
docker network inspect docker_default
```

**Clean Restart Procedure:**
```bash
# Complete system reset
cd docker
docker-compose down -v  # Remove volumes
docker-compose pull     # Update images
docker-compose up -d    # Restart services

# Clear Python cache
cd ../backend
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Restart FastAPI
python app.py
```

### Debugging Mode

**Enable Verbose Logging:**
```python
# Add to app.py for debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export LOG_LEVEL=DEBUG  # Linux/macOS
set LOG_LEVEL=DEBUG     # Windows
```

**Test Individual Components:**
```bash
# Test embedding generation only
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode('test text')
print(f'Generated embedding shape: {embedding.shape}')
"

# Test Endee connection only
python -c "
import requests
response = requests.get('http://localhost:8081/health')
print(f'Endee status: {response.status_code} - {response.text}')
"
```

### Getting Help

If issues persist:

1. **Check logs:**
   ```bash
   # FastAPI logs (in terminal where app.py is running)
   # Docker logs
   docker-compose logs endee
   
   # System logs (Linux)
   journalctl -u docker
   
   # Docker daemon logs (macOS)
   tail -f ~/Library/Containers/com.docker.docker/Data/log/vm/dockerd.log
   ```

2. **Verify system requirements:**
   - Python 3.8+
   - Docker and Docker Compose working
   - Sufficient disk space (2GB+)
   - Available memory (4GB+ recommended)
   - Network connectivity for model downloads

3. **Test minimal setup:**
   ```bash
   # Test Python imports
   python -c "import fastapi, sentence_transformers, torch; print('All imports successful')"
   
   # Test Docker
   docker run hello-world
   
   # Test network connectivity
   curl -I https://huggingface.co
   ```

4. **Collect diagnostic information:**
   ```bash
   # System information
   python --version
   docker --version
   docker-compose --version
   
   # Resource usage
   df -h .  # Disk space
   free -h  # Memory (Linux)
   
   # Network status
   netstat -tlnp | grep -E '8000|8081'
   ```

5. **Create issue with:**
   - Operating system and version
   - Python version (`python --version`)
   - Docker version (`docker --version`)
   - Error messages and logs
   - Steps to reproduce the issue
   - Output from diagnostic commands above

## Development

### Running Tests

```bash
cd backend
python -m pytest tests/
```

### Code Quality

The codebase follows Python best practices:
- Type hints for better code clarity
- Comprehensive docstrings
- Clear separation of concerns
- Proper error handling

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Performance Notes

- **Model Caching**: The embedding model is loaded once and cached in memory
- **Connection Pooling**: HTTP connections to Endee are pooled for efficiency
- **Async Operations**: I/O-bound operations use async/await for better performance
- **Batch Processing**: Multiple documents can be processed in batches

## License

[Add your license information here]

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the API documentation
3. Check Docker and Endee logs for detailed error messages
4. Open an issue in the repository

## Next Steps

After setup, you can:
1. Ingest your document collection
2. Test semantic search capabilities
3. Experiment with RAG queries
4. Integrate with your preferred LLM for response generation
5. Customize the embedding model for your domain