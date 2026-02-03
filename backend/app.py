# FastAPI entry point
# Main application file for the Endee RAG System
#
# ENDEE INTEGRATION COORDINATION
# ==============================
#
# This FastAPI application serves as the coordination layer for all Endee vector database
# interactions in the RAG system. It orchestrates document ingestion, semantic search,
# and RAG response generation through service layers that communicate with Endee.
#
# ENDEE INTEGRATION ARCHITECTURE:
# -------------------------------
# 1. Document Ingestion Flow:
#    POST /ingest → DocumentIngestionService → EndeeHTTPClient.add_vector() → Endee storage
#
# 2. Semantic Search Flow:
#    POST /search → VectorSearchService → EndeeHTTPClient.search_vectors() → Endee search
#
# 3. RAG Pipeline Flow:
#    POST /rag → RAGService → VectorSearchService → EndeeHTTPClient → Endee → LLM response
#
# ENDEE SERVICE DEPENDENCIES:
# ---------------------------
# - DocumentIngestionService: Handles text → embedding → Endee storage
# - VectorSearchService: Handles query → embedding → Endee search → results
# - RAGService: Orchestrates search → context → LLM response generation
# - EndeeHTTPClient: Low-level HTTP communication with Endee API
#
# ENDEE ERROR HANDLING STRATEGY:
# ------------------------------
# The application maps Endee-specific errors to appropriate HTTP status codes:
# - EndeeConnectionError → 503 Service Unavailable (Endee service issues)
# - EndeeRequestError → 400 Bad Request (invalid payload/parameters)
# - EndeeTimeoutError → 503 Service Unavailable (Endee response timeout)
# - ValidationError → 400 Bad Request (input validation failures)
#
# ENDEE CONFIGURATION:
# -------------------
# - ENDEE_URL environment variable configures Endee service endpoint
# - Default: http://localhost:8081 (matches docker-compose setup)
# - Services initialize Endee clients automatically on startup
# - Connection pooling and retry logic handled by EndeeHTTPClient

import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from endee_client import EndeeConnectionError, ServiceUnavailableError, EndeeTimeoutError
from ingest import (
    DocumentIngestionService,
    EmbeddingError,
    IngestionError,
    ValidationError,
)
from rag import ContextCombinationError, LLMError, RAGError, RAGService
from search import SearchError, VectorSearchService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger(__name__)

class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str
    timestamp: str  # Use string instead of datetime for JSON serialization
    version: str


class ErrorResponse(BaseModel):
    """Response model for error responses."""
    
    error: str
    detail: str
    timestamp: str  # Use string instead of datetime for JSON serialization


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    
    content: str = Field(
        ...,
        min_length=3,
        max_length=50000,
        description="Document text content (3-50000 characters)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional document metadata (max 10KB)"
    )


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    
    document_id: str
    status: str
    embedding_dimension: int
    content_length: int
    metadata: Dict[str, Any]


class SearchRequest(BaseModel):
    """Request model for semantic search."""
    
    query: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Search query text (2-1000 characters)"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of top results to return (1-100)"
    )
    similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0.0-1.0)"
    )


class SearchResultItem(BaseModel):
    """Individual search result item."""
    
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response model for semantic search."""
    
    query: str
    results: List[SearchResultItem]
    total_results: int
    similarity_threshold: float


class RAGRequest(BaseModel):
    """Request model for RAG queries."""
    
    query: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="User question (2-1000 characters)"
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of documents to retrieve (1-20)"
    )
    similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0.0-1.0)"
    )
    max_context_length: Optional[int] = Field(
        default=None,
        ge=100,
        le=50000,
        description="Maximum context length in characters (100-50000)"
    )


class RAGSourceItem(BaseModel):
    """Individual RAG source item."""
    
    document_id: str
    score: float
    metadata: Dict[str, Any]
    content_preview: str


class RAGResponse(BaseModel):
    """Response model for RAG queries."""
    
    query: str
    answer: str
    sources: List[RAGSourceItem]
    context_used: str
    context_length: int
    sources_count: int
    truncated: bool
    max_context_length: int

# Initialize FastAPI app
app = FastAPI(
    title="Endee RAG System",
    description="Retrieval Augmented Generation system using Endee vector database",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize services with memory optimization
ingestion_service = DocumentIngestionService()
search_service = VectorSearchService()
rag_service = RAGService(search_service)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def sanitize_error_message(error_message: str) -> str:
    """Sanitize error messages to prevent exposure of sensitive information.
    
    This function implements comprehensive error message sanitization to ensure
    that detailed error information is provided to users while protecting
    sensitive internal system details from exposure.
    
    Sanitization Strategy:
    1. Remove or mask sensitive authentication details (passwords, tokens, keys)
    2. Obscure internal server names and network topology
    3. Hide database connection strings and internal URLs
    4. Mask file paths that could reveal system structure
    5. Remove stack traces and internal module references
    6. Preserve error context and actionable information for users
    
    Args:
        error_message: Raw error message that may contain sensitive data
        
    Returns:
        Sanitized error message safe for client consumption with helpful details
    """
    if not error_message:
        return "An error occurred"
    
    import re
    sanitized = str(error_message)
    
    # Enhanced list of sensitive patterns to remove or replace
    # Order matters - more specific patterns should come first
    sensitive_patterns = [
        # API keys and tokens (most specific patterns first)
        (r'\bapi_key=\S+', 'api_key=***'),
        (r'\bapikey=\S+', 'apikey=***'),
        (r'\btoken=\S+', 'token=***'),
        (r'\bsecret=\S+', 'secret=***'),
        (r'\bkey=\S+', 'key=***'),
        (r'bearer\s+\S+', 'bearer ***'),
        
        # Authentication and credentials
        (r'\bpassword=\S+', 'password=***'),
        (r'\bpasswd=\S+', 'passwd=***'),
        (r'\bpwd=\S+', 'pwd=***'),
        (r'\buser=\S+', 'user=***'),
        (r'\busername=\S+', 'username=***'),
        (r'\blogin=\S+', 'login=***'),
        (r'\bauth=\S+', 'auth=***'),
        
        # Server and network details
        (r'\bhost=[\w\.-]+', 'host=***'),
        (r'\bhostname=[\w\.-]+', 'hostname=***'),
        (r'\bserver=[\w\.-]+', 'server=***'),
        (r'internal-[\w\.-]+', 'internal-***'),
        (r'localhost:\d+', 'localhost:***'),
        (r'127\.0\.0\.1:\d+', '127.0.0.1:***'),
        (r'192\.168\.\d+\.\d+:\d+', '192.168.*.*:***'),
        (r'10\.\d+\.\d+\.\d+:\d+', '10.*.*.*:***'),
        
        # Database connection strings
        (r'postgresql://[^/\s]+', 'postgresql://***'),
        (r'mysql://[^/\s]+', 'mysql://***'),
        (r'mongodb://[^/\s]+', 'mongodb://***'),
        (r'redis://[^/\s]+', 'redis://***'),
        
        # File paths that could reveal system structure
        (r'/home/[\w\.-]+', '/home/***'),
        (r'/usr/local/[\w\.-/]+', '/usr/local/***'),
        (r'/opt/[\w\.-/]+', '/opt/***'),
        (r'/var/[\w\.-/]+', '/var/***'),
        (r'C:\\\\Users\\\\[\w\.-]+', r'C:\\Users\\***'),
        (r'C:\\\\Program Files\\\\[\w\.-\\\\]+', r'C:\\Program Files\\***'),
        
        # Stack traces and internal module references
        (r'File ".*?", line \d+', 'File "***", line ***'),
        (r'Traceback \(most recent call last\):', 'Internal error occurred'),
        (r'at [\w\.]+\([\w\.]+:\d+\)', 'at ***(***:***)'),
        
        # Environment variables (only for clearly non-API patterns)
        (r'DATABASE_PASSWORD=\S+', 'DATABASE_PASSWORD=***'),
        (r'DB_PASSWORD=\S+', 'DB_PASSWORD=***'),
        (r'MYSQL_PASSWORD=\S+', 'MYSQL_PASSWORD=***'),
        (r'POSTGRES_PASSWORD=\S+', 'POSTGRES_PASSWORD=***'),
    ]
    
    # Apply sanitization patterns
    for pattern, replacement in sensitive_patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    
    # Additional sanitization for common error patterns
    # Remove excessive technical details while preserving useful information
    sanitized = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', 
                      '***-***-***-***-***', sanitized)  # UUIDs
    sanitized = re.sub(r'\b[0-9a-f]{32,}\b', '***', sanitized)  # Long hex strings
    
    # Ensure the message is still informative
    if len(sanitized.strip()) < 5 or sanitized.strip() == "***":
        return "An error occurred while processing your request"
    
    return sanitized


def create_detailed_error_message(error_type: str, base_message: str, context: dict = None) -> str:
    """Create detailed error messages that are helpful to users without exposing internals.
    
    This function generates user-friendly error messages that provide sufficient detail
    for troubleshooting while maintaining security by not exposing internal system details.
    
    Args:
        error_type: Type of error (validation, connection, timeout, etc.)
        base_message: Base error message to enhance
        context: Optional context dictionary with additional error details
        
    Returns:
        Detailed, user-friendly error message
    """
    context = context or {}
    
    # Define detailed error message templates
    error_templates = {
        "validation": {
            "prefix": "Input validation failed",
            "suggestions": [
                "Please check that all required fields are provided",
                "Ensure field values meet the specified format requirements",
                "Verify that numeric values are within acceptable ranges"
            ]
        },
        "connection": {
            "prefix": "Service connection failed",
            "suggestions": [
                "The vector database service may be temporarily unavailable",
                "Please try again in a few moments",
                "If the problem persists, contact support"
            ]
        },
        "timeout": {
            "prefix": "Request timed out",
            "suggestions": [
                "The operation took longer than expected to complete",
                "This may be due to high system load or network issues",
                "Please try again with a smaller request or wait a few minutes"
            ]
        },
        "embedding": {
            "prefix": "Text processing failed",
            "suggestions": [
                "The text content may contain unsupported characters or format",
                "Try reducing the text length or simplifying the content",
                "Ensure the text is in a supported language"
            ]
        },
        "search": {
            "prefix": "Search operation failed",
            "suggestions": [
                "The search query may be too complex or contain invalid characters",
                "Try simplifying your search terms",
                "Ensure the search parameters are within acceptable ranges"
            ]
        },
        "storage": {
            "prefix": "Data storage failed",
            "suggestions": [
                "The document may be too large or contain invalid data",
                "Check that the document format is supported",
                "Verify that all required metadata is provided"
            ]
        },
        "service_unavailable": {
            "prefix": "Service temporarily unavailable",
            "suggestions": [
                "The system is experiencing high load or maintenance",
                "Please try again in a few minutes",
                "If urgent, contact support for assistance"
            ]
        }
    }
    
    template = error_templates.get(error_type, {
        "prefix": "An error occurred",
        "suggestions": ["Please try again or contact support if the problem persists"]
    })
    
    # Build detailed message
    detailed_parts = [template["prefix"]]
    
    # Add sanitized base message if it provides additional context
    if base_message and base_message.strip():
        sanitized_base = sanitize_error_message(base_message)
        if sanitized_base and sanitized_base != template["prefix"]:
            detailed_parts.append(f"Details: {sanitized_base}")
    
    # Add context information if available
    if context:
        if "operation" in context:
            detailed_parts.append(f"Operation: {context['operation']}")
        if "field" in context:
            detailed_parts.append(f"Field: {context['field']}")
        if "expected_format" in context:
            detailed_parts.append(f"Expected format: {context['expected_format']}")
    
    # Add helpful suggestions
    detailed_parts.append("Suggestions:")
    for suggestion in template["suggestions"]:
        detailed_parts.append(f"• {suggestion}")
    
    return " | ".join(detailed_parts)


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Handle HTTP exceptions and return structured error responses.
    
    Args:
        request: The HTTP request that caused the exception
        exc: The HTTP exception that was raised
        
    Returns:
        JSONResponse with error details
    """
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    # Map status codes to appropriate error types
    error_type_mapping = {
        400: "Bad Request",
        401: "Unauthorized", 
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        409: "Conflict",
        422: "Unprocessable Entity",
        429: "Too Many Requests",
        500: "Internal Server Error",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout"
    }
    
    error_type = error_type_mapping.get(exc.status_code, "HTTP Error")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=error_type,
            detail=exc.detail,
            timestamp=datetime.now().isoformat()
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions and return generic error responses.
    
    Args:
        request: The HTTP request that caused the exception
        exc: The exception that was raised
        
    Returns:
        JSONResponse with generic error message
    """
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            timestamp=datetime.now().isoformat()
        ).model_dump()
    )

@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the FastAPI application on startup with memory optimizations."""
    logger.info("Starting Endee RAG System API")
    
    # Initialize memory monitoring
    try:
        from memory_monitor import memory_monitor
        memory_monitor.reset_baseline()
        memory_monitor.start_monitoring(interval_seconds=60.0)  # Monitor every minute
        logger.info("Memory monitoring initialized and started")
    except Exception as e:
        logger.warning(f"Failed to initialize memory monitoring: {str(e)}")
    
    # Pre-warm embedding model cache to avoid cold start delays
    try:
        from embedding_cache import embedding_cache
        logger.info("Pre-warming embedding model cache...")
        # Generate a small test embedding to load the model
        embedding_cache.generate_embedding("warmup", normalize=True)
        logger.info("Embedding model cache pre-warmed successfully")
    except Exception as e:
        logger.warning(f"Failed to pre-warm embedding model cache: {str(e)}")
    
    # Pre-warm connection pools for better performance
    try:
        logger.info("Pre-warming Endee connection pools...")
        await ingestion_service.endee_client.warmup_connections(num_connections=3)
        await search_service.endee_client.warmup_connections(num_connections=3)
        logger.info("Connection pools pre-warmed successfully")
    except Exception as e:
        logger.warning(f"Failed to pre-warm connection pools: {str(e)}")
    
    logger.info("FastAPI app initialized successfully with memory optimizations")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up resources when the FastAPI application shuts down."""
    logger.info("Shutting down Endee RAG System API")
    
    # Stop memory monitoring
    try:
        from memory_monitor import memory_monitor
        memory_monitor.stop_monitoring()
        logger.info("Memory monitoring stopped")
    except Exception as e:
        logger.warning(f"Error stopping memory monitoring: {str(e)}")
    
    # Clean up connection pools to free memory
    try:
        await ingestion_service.endee_client.close()
        await search_service.endee_client.close()
        logger.info("Connection pools closed successfully")
    except Exception as e:
        logger.warning(f"Error closing connection pools: {str(e)}")
    
    # Optional: Clear embedding cache if needed for memory cleanup
    # Note: Only uncomment if memory pressure is critical
    # try:
    #     from embedding_cache import embedding_cache
    #     embedding_cache.clear_cache()
    #     logger.info("Embedding cache cleared")
    # except Exception as e:
    #     logger.warning(f"Error clearing embedding cache: {str(e)}")
    
    logger.info("Shutdown complete")


@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    """Root endpoint providing basic API information."""
    logger.info("Root endpoint accessed")
    return {
        "message": "Endee RAG System API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health/endee", response_model=Dict[str, Any])
async def endee_health_check() -> Dict[str, Any]:
    """Health check endpoint specifically for Endee service availability."""
    try:
        logger.info("Endee health check endpoint accessed")
        
        # Check Endee service availability using ingestion service client
        health_result = await ingestion_service.endee_client.check_service_availability()
        
        response = {
            "status": "healthy" if health_result["available"] else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "endee_service": health_result,
            "circuit_breaker": ingestion_service.endee_client.get_service_health()
        }
        
        logger.info(f"Endee health check completed: {response['status']}")
        return response
        
    except ServiceUnavailableError as e:
        logger.warning(f"Endee service unavailable during health check: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": "Service unavailable",
            "detail": str(e),
            "circuit_breaker": ingestion_service.endee_client.get_service_health()
        }
        
    except Exception as e:
        logger.error(f"Error during Endee health check: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": "Health check failed",
            "detail": str(e)
        }


@app.post("/admin/endee/circuit-breaker/reset", response_model=Dict[str, Any])
async def reset_endee_circuit_breaker() -> Dict[str, Any]:
    """Administrative endpoint to manually reset the Endee circuit breaker."""
    try:
        logger.info("Manual circuit breaker reset requested")
        
        # Reset circuit breaker for both ingestion and search services
        ingestion_result = ingestion_service.endee_client.reset_circuit_breaker()
        search_result = search_service.endee_client.reset_circuit_breaker()
        
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "ingestion_service": ingestion_result,
            "search_service": search_result,
            "message": "Circuit breakers reset successfully"
        }
        
        logger.info("Circuit breakers reset successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error resetting circuit breakers: {str(e)}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "message": "Failed to reset circuit breakers"
        }


@app.get("/debug/error-test/{error_type}")
async def debug_error_test(error_type: str) -> Dict[str, str]:
    """Debug endpoint to test various error scenarios and their HTTP status codes.
    
    This endpoint is for testing purposes only and should be removed in production.
    
    Args:
        error_type: Type of error to simulate
        
    Returns:
        Error response with appropriate HTTP status code
        
    Raises:
        HTTPException: With appropriate status code for the error type
    """
    logger.info(f"Debug error test requested: {error_type}")
    
    if error_type == "validation":
        raise HTTPException(status_code=400, detail="Validation error: Invalid input parameters")
    elif error_type == "unauthorized":
        raise HTTPException(status_code=401, detail="Unauthorized: Authentication required")
    elif error_type == "forbidden":
        raise HTTPException(status_code=403, detail="Forbidden: Access denied")
    elif error_type == "not-found":
        raise HTTPException(status_code=404, detail="Not Found: Resource does not exist")
    elif error_type == "method-not-allowed":
        raise HTTPException(status_code=405, detail="Method Not Allowed: HTTP method not supported")
    elif error_type == "conflict":
        raise HTTPException(status_code=409, detail="Conflict: Resource already exists")
    elif error_type == "unprocessable":
        raise HTTPException(status_code=422, detail="Unprocessable Entity: Request format is invalid")
    elif error_type == "rate-limit":
        raise HTTPException(status_code=429, detail="Too Many Requests: Rate limit exceeded")
    elif error_type == "internal":
        raise HTTPException(status_code=500, detail="Internal Server Error: Unexpected server error")
    elif error_type == "bad-gateway":
        raise HTTPException(status_code=502, detail="Bad Gateway: Upstream service error")
    elif error_type == "service-unavailable":
        raise HTTPException(status_code=503, detail="Service Unavailable: Service temporarily unavailable")
    elif error_type == "timeout":
        raise HTTPException(status_code=504, detail="Gateway Timeout: Request timed out")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown error type: {error_type}")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint to verify API status."""
    logger.info("Health check endpoint accessed")
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.get("/memory", response_model=Dict[str, Any])
async def memory_status() -> Dict[str, Any]:
    """Memory status endpoint for monitoring memory usage and optimization."""
    try:
        from memory_monitor import memory_monitor
        
        # Get comprehensive memory statistics
        memory_stats = memory_monitor.get_memory_stats()
        
        # Get memory recommendations
        recommendations = memory_monitor.get_memory_recommendations()
        
        # Add embedding cache memory info if available
        try:
            from embedding_cache import embedding_cache
            cache_memory = embedding_cache.get_memory_usage()
            memory_stats["embedding_cache"] = cache_memory
        except Exception as e:
            error_msg = str(e)
            memory_stats["embedding_cache"] = {"error": error_msg}
            logger.error(f"Error getting embedding cache memory usage: {error_msg}", exc_info=True)
        
        # Add connection pool stats
        try:
            ingestion_pool_stats = ingestion_service.endee_client.get_connection_pool_stats()
            search_pool_stats = search_service.endee_client.get_connection_pool_stats()
            memory_stats["connection_pools"] = {
                "ingestion_service": ingestion_pool_stats,
                "search_service": search_pool_stats
            }
        except Exception as e:
            error_msg = str(e)
            memory_stats["connection_pools"] = {"error": error_msg}
            logger.error(f"Error getting connection pool stats: {error_msg}", exc_info=True)
        
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "memory_stats": memory_stats,
            "recommendations": recommendations
        }
        
        logger.info(f"Memory status requested: {memory_stats['current']['rss_mb']:.1f}MB RSS")
        return response
        
    except Exception as e:
        logger.error(f"Error getting memory status: {str(e)}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@app.post("/memory/optimize", response_model=Dict[str, Any])
async def optimize_memory() -> Dict[str, Any]:
    """Memory optimization endpoint to trigger memory cleanup and optimization."""
    try:
        from memory_monitor import memory_monitor
        
        logger.info("Manual memory optimization requested")
        
        # Perform memory optimization
        optimization_results = memory_monitor.optimize_memory()
        
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "optimization_results": optimization_results
        }
        
        logger.info(f"Memory optimization completed: freed {optimization_results.get('memory_freed_mb', 0):.1f}MB")
        return response
        
    except Exception as e:
        logger.error(f"Error during memory optimization: {str(e)}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest) -> IngestResponse:
    """Ingest a text document into the system.
    
    ENDEE INTEGRATION: DOCUMENT STORAGE ENDPOINT
    ===========================================
    This endpoint implements the complete document ingestion pipeline that stores
    documents in the Endee vector database for later semantic search and retrieval.
    
    ENDEE INTEGRATION FLOW:
    ----------------------
    1. Receive document content and metadata from client
    2. DocumentIngestionService.ingest_document():
       - Validate and preprocess text content
       - Generate 384d embedding using all-MiniLM-L6-v2
       - Store embedding + metadata in Endee via EndeeHTTPClient.add_vector()
    3. Return success response with document ID and embedding info
    
    ENDEE STORAGE DETAILS:
    ---------------------
    - Vector Dimensions: 384 (sentence-transformers all-MiniLM-L6-v2)
    - Metadata Stored: Original content, timestamps, user metadata
    - Document ID: UUID generated for unique identification in Endee
    - Storage Endpoint: POST {endee_url}/vectors/add
    
    ENDEE ERROR MAPPING:
    -------------------
    - EndeeConnectionError → 503 Service Unavailable (Endee service down)
    - ValidationError → 400 Bad Request (invalid content/metadata)
    - EmbeddingError → 500 Internal Server Error (model issues)
    - IngestionError → 500 Internal Server Error (pipeline failures)
    
    This endpoint accepts raw text content and optional metadata,
    generates embeddings using sentence-transformers, and stores
    the document in the Endee vector database.
    """
    try:
        logger.info(
            f"Ingesting document with content length: {len(request.content)}"
        )
        
        result = await ingestion_service.ingest_document(
            content=request.content,
            metadata=request.metadata
        )
        
        logger.info(f"Document ingestion successful: {result['document_id']}")
        return IngestResponse(**result)
        
    except ValidationError as e:
        logger.error(f"Validation error in document ingestion: {str(e)}")
        detailed_message = create_detailed_error_message(
            "validation", 
            str(e), 
            {"operation": "document_ingestion"}
        )
        raise HTTPException(
            status_code=400, detail=detailed_message
        )
        
    except EmbeddingError as e:
        logger.error(f"Embedding error in document ingestion: {str(e)}")
        detailed_message = create_detailed_error_message(
            "embedding", 
            str(e), 
            {"operation": "document_ingestion"}
        )
        raise HTTPException(
            status_code=422, detail=detailed_message
        )
        
    except EndeeConnectionError as e:
        logger.error(f"Endee connection error in document ingestion: {str(e)}")
        detailed_message = create_detailed_error_message(
            "connection", 
            str(e), 
            {"operation": "document_ingestion"}
        )
        raise HTTPException(
            status_code=503, detail=detailed_message
        )
        
    except ServiceUnavailableError as e:
        logger.error(f"Endee service unavailable in document ingestion: {str(e)}")
        detailed_message = create_detailed_error_message(
            "service_unavailable", 
            str(e), 
            {"operation": "document_ingestion"}
        )
        raise HTTPException(
            status_code=503, detail=detailed_message
        )
        
    except EndeeTimeoutError as e:
        logger.error(f"Endee timeout error in document ingestion: {str(e)}")
        detailed_message = create_detailed_error_message(
            "timeout", 
            str(e), 
            {"operation": "document_ingestion"}
        )
        raise HTTPException(
            status_code=504, detail=detailed_message
        )
        
    except IngestionError as e:
        logger.error(f"Ingestion error in document ingestion: {str(e)}")
        detailed_message = create_detailed_error_message(
            "storage", 
            str(e), 
            {"operation": "document_ingestion"}
        )
        raise HTTPException(
            status_code=500, detail=detailed_message
        )
        
    except Exception as e:
        logger.error(
            f"Unexpected error in document ingestion: {str(e)}", exc_info=True
        )
        detailed_message = create_detailed_error_message(
            "storage", 
            str(e), 
            {"operation": "document_ingestion"}
        )
        raise HTTPException(
            status_code=500,
            detail=detailed_message
        )

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest) -> SearchResponse:
    """Perform semantic search on ingested documents.
    
    ENDEE INTEGRATION: SEMANTIC SEARCH ENDPOINT
    ==========================================
    This endpoint implements semantic document search using the Endee vector database
    to find documents most similar to a natural language query.
    
    ENDEE INTEGRATION FLOW:
    ----------------------
    1. Receive search query and parameters from client
    2. VectorSearchService.search():
       - Validate and preprocess query text
       - Generate 384d query embedding using all-MiniLM-L6-v2
       - Search Endee for similar vectors via EndeeHTTPClient.search_vectors()
       - Filter results by similarity threshold
       - Format results with content and metadata
    3. Return ranked search results with similarity scores
    
    ENDEE SEARCH DETAILS:
    --------------------
    - Query Vector: 384d embedding from same model as stored documents
    - Similarity Metric: Cosine similarity (0.0 to 1.0 range)
    - Search Endpoint: POST {endee_url}/vectors/search
    - Result Ranking: Descending order by similarity score
    - Threshold Filtering: Client-side filtering of low-relevance results
    
    ENDEE ERROR MAPPING:
    -------------------
    - EndeeConnectionError → 503 Service Unavailable (Endee service issues)
    - ValidationError → 400 Bad Request (invalid query/parameters)
    - EmbeddingError → 500 Internal Server Error (query embedding failure)
    - SearchError → 500 Internal Server Error (search pipeline failure)
    
    This endpoint accepts a natural language query and returns
    the most relevant documents based on vector similarity search
    in the Endee database. Results can be filtered by similarity threshold.
    """
    try:
        query_preview = (
            f"'{request.query[:50]}{'...' if len(request.query) > 50 else ''}'"
        )
        logger.info(
            f"Performing search for query: {query_preview} "
            f"with top_k={request.top_k}, "
            f"threshold={request.similarity_threshold}"
        )
        
        results = await search_service.search(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        # Convert results to response format
        search_results = [
            SearchResultItem(
                document_id=result["document_id"],
                content=result["content"],
                score=result["score"],
                metadata=result["metadata"]
            )
            for result in results
        ]
        
        logger.info(
            f"Search completed successfully: {len(search_results)} results returned"
        )
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            similarity_threshold=request.similarity_threshold
        )
        
    except ValidationError as e:
        logger.error(f"Validation error in search: {str(e)}")
        detailed_message = create_detailed_error_message(
            "validation", 
            str(e), 
            {"operation": "semantic_search"}
        )
        raise HTTPException(
            status_code=400, detail=detailed_message
        )
        
    except EmbeddingError as e:
        logger.error(f"Embedding error in search: {str(e)}")
        detailed_message = create_detailed_error_message(
            "embedding", 
            str(e), 
            {"operation": "semantic_search"}
        )
        raise HTTPException(
            status_code=422, detail=detailed_message
        )
        
    except EndeeConnectionError as e:
        logger.error(f"Endee connection error in search: {str(e)}")
        detailed_message = create_detailed_error_message(
            "connection", 
            str(e), 
            {"operation": "semantic_search"}
        )
        raise HTTPException(
            status_code=503, detail=detailed_message
        )
        
    except ServiceUnavailableError as e:
        logger.error(f"Endee service unavailable in search: {str(e)}")
        detailed_message = create_detailed_error_message(
            "service_unavailable", 
            str(e), 
            {"operation": "semantic_search"}
        )
        raise HTTPException(
            status_code=503, detail=detailed_message
        )
        
    except EndeeTimeoutError as e:
        logger.error(f"Endee timeout error in search: {str(e)}")
        detailed_message = create_detailed_error_message(
            "timeout", 
            str(e), 
            {"operation": "semantic_search"}
        )
        raise HTTPException(
            status_code=504, detail=detailed_message
        )
        
    except SearchError as e:
        logger.error(f"Search error: {str(e)}")
        detailed_message = create_detailed_error_message(
            "search", 
            str(e), 
            {"operation": "semantic_search"}
        )
        raise HTTPException(status_code=500, detail=detailed_message)
        
    except Exception as e:
        logger.error(f"Unexpected error in search: {str(e)}", exc_info=True)
        detailed_message = create_detailed_error_message(
            "search", 
            str(e), 
            {"operation": "semantic_search"}
        )
        raise HTTPException(
            status_code=500, detail=detailed_message
        )

@app.post("/rag", response_model=RAGResponse)
async def rag_query(request: RAGRequest) -> RAGResponse:
    """Generate contextual answers using Retrieval Augmented Generation.
    
    ENDEE INTEGRATION: RAG PIPELINE ENDPOINT
    =======================================
    This endpoint implements the complete RAG pipeline that uses Endee vector database
    for document retrieval and combines retrieved context for LLM response generation.
    
    ENDEE INTEGRATION FLOW:
    ----------------------
    1. Receive user query and RAG parameters from client
    2. RAGService.generate_response():
       a. Document Retrieval Phase:
          - Generate query embedding (384d)
          - Search Endee via VectorSearchService → EndeeHTTPClient.search_vectors()
          - Retrieve top-k most similar documents with metadata
       b. Context Preparation Phase:
          - Extract document content from Endee metadata
          - Combine documents into coherent context
          - Optimize context formatting for LLM consumption
       c. Response Generation Phase:
          - Generate response using LLM with retrieved context
          - Include source attribution and metadata
    3. Return structured response with answer, sources, and context info
    
    ENDEE RAG INTEGRATION DETAILS:
    -----------------------------
    - Document Retrieval: Endee similarity search finds relevant documents
    - Content Extraction: Original text retrieved from Endee metadata
    - Source Attribution: Document IDs and scores from Endee results
    - Context Limits: Configurable max context length for LLM constraints
    - Relevance Filtering: Similarity threshold filtering via Endee search
    
    ENDEE ERROR MAPPING:
    -------------------
    - EndeeConnectionError → 503 Service Unavailable (Endee service issues)
    - ValidationError → 400 Bad Request (invalid query/parameters)
    - SearchError → 500 Internal Server Error (document retrieval failure)
    - ContextCombinationError → 500 Internal Server Error (context processing)
    - RAGError → 500 Internal Server Error (RAG pipeline failure)
    
    This endpoint performs the complete RAG pipeline:
    1. Retrieves relevant documents using semantic search
    2. Combines retrieved documents into coherent context
    3. Generates responses using the retrieved context
    4. Returns the answer with source attribution
    """
    try:
        query_preview = (
            f"'{request.query[:50]}{'...' if len(request.query) > 50 else ''}'"
        )
        logger.info(
            f"Processing RAG query: {query_preview} "
            f"with top_k={request.top_k}, "
            f"threshold={request.similarity_threshold}"
        )
        
        result = await rag_service.generate_response(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            max_context_length=request.max_context_length
        )
        
        # Convert sources to response format
        rag_sources = [
            RAGSourceItem(
                document_id=source["document_id"],
                score=source["score"],
                metadata=source["metadata"],
                content_preview=source["content_preview"]
            )
            for source in result["sources"]
        ]
        
        logger.info(
            f"RAG query completed successfully: {result['sources_count']} sources used"
        )
        return RAGResponse(
            query=result["query"],
            answer=result["answer"],
            sources=rag_sources,
            context_used=result["context_used"],
            context_length=result["context_length"],
            sources_count=result["sources_count"],
            truncated=result["truncated"],
            max_context_length=result["max_context_length"]
        )
        
    except ValidationError as e:
        logger.error(f"Validation error in RAG: {str(e)}")
        detailed_message = create_detailed_error_message(
            "validation", 
            str(e), 
            {"operation": "rag_query"}
        )
        raise HTTPException(
            status_code=400, detail=detailed_message
        )
        
    except ContextCombinationError as e:
        logger.error(f"Context combination error in RAG: {str(e)}")
        detailed_message = create_detailed_error_message(
            "search", 
            str(e), 
            {"operation": "rag_context_combination"}
        )
        raise HTTPException(
            status_code=500, detail=detailed_message
        )
        
    except SearchError as e:
        logger.error(f"Search error in RAG: {str(e)}")
        detailed_message = create_detailed_error_message(
            "search", 
            str(e), 
            {"operation": "rag_document_retrieval"}
        )
        raise HTTPException(
            status_code=500, detail=detailed_message
        )
        
    except EmbeddingError as e:
        logger.error(f"Embedding error in RAG: {str(e)}")
        detailed_message = create_detailed_error_message(
            "embedding", 
            str(e), 
            {"operation": "rag_query_processing"}
        )
        raise HTTPException(
            status_code=422, detail=detailed_message
        )
        
    except EndeeConnectionError as e:
        logger.error(f"Endee connection error in RAG: {str(e)}")
        detailed_message = create_detailed_error_message(
            "connection", 
            str(e), 
            {"operation": "rag_document_retrieval"}
        )
        raise HTTPException(
            status_code=503, detail=detailed_message
        )
        
    except ServiceUnavailableError as e:
        logger.error(f"Endee service unavailable in RAG: {str(e)}")
        detailed_message = create_detailed_error_message(
            "service_unavailable", 
            str(e), 
            {"operation": "rag_document_retrieval"}
        )
        raise HTTPException(
            status_code=503, detail=detailed_message
        )
        
    except EndeeTimeoutError as e:
        logger.error(f"Endee timeout error in RAG: {str(e)}")
        detailed_message = create_detailed_error_message(
            "timeout", 
            str(e), 
            {"operation": "rag_document_retrieval"}
        )
        raise HTTPException(
            status_code=504, detail=detailed_message
        )
        
    except RAGError as e:
        logger.error(f"RAG error: {str(e)}")
        detailed_message = create_detailed_error_message(
            "search", 
            str(e), 
            {"operation": "rag_pipeline"}
        )
        raise HTTPException(
            status_code=500, detail=detailed_message
        )
        
    except LLMError as e:
        logger.error(f"LLM error in RAG: {str(e)}")
        detailed_message = create_detailed_error_message(
            "service_unavailable", 
            str(e), 
            {"operation": "rag_response_generation"}
        )
        raise HTTPException(
            status_code=502, detail=detailed_message
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in RAG: {str(e)}", exc_info=True)
        detailed_message = create_detailed_error_message(
            "search", 
            str(e), 
            {"operation": "rag_pipeline"}
        )
        raise HTTPException(
            status_code=500, detail=detailed_message
        )