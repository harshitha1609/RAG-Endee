# Endee HTTP Client
# Robust HTTP client for Endee API communication with error handling and retries
#
# ENDEE INTEGRATION DOCUMENTATION
# ===============================
#
# This module provides the primary integration layer with the Endee vector database.
# Endee is a high-performance vector database that stores and searches document embeddings
# for the RAG (Retrieval Augmented Generation) system.
#
# ENDEE API ENDPOINTS USED:
# ------------------------
# 1. POST /vectors/add - Store document embeddings with metadata
#    - Payload: {"id": str, "vector": List[float], "metadata": Dict}
#    - Response: {"status": "success", "id": str}
#
# 2. POST /vectors/search - Search for similar vectors
#    - Payload: {"vector": List[float], "top_k": int, "similarity_threshold": float}
#    - Response: {"results": [{"id": str, "score": float, "metadata": Dict}]}
#
# ENDEE CONFIGURATION:
# -------------------
# - ENDEE_URL: Base URL for Endee service (default: http://localhost:8081)
# - Connection pooling: 10 connections, max 20 per pool
# - Request timeout: 30 seconds
# - Retry strategy: 3 attempts with exponential backoff
#
# ENDEE DATA FLOW:
# ---------------
# 1. Document Ingestion: text → embedding (384d) → Endee storage via add_vector()
# 2. Semantic Search: query → embedding (384d) → Endee search via search_vectors()
# 3. RAG Pipeline: search results → context combination → LLM response
#
# ENDEE ERROR HANDLING:
# --------------------
# - EndeeConnectionError: Network/server issues (retryable)
# - EndeeRequestError: Client errors like bad payload (not retryable)
# - EndeeTimeoutError: Request timeouts (retryable)
# - Exponential backoff: 1s, 2s, 4s delays with 10s max
#
# ENDEE PERFORMANCE MONITORING:
# ----------------------------
# - Operation statistics tracking (calls, duration, errors)
# - Structured logging with operation context
# - Performance metrics via get_performance_stats()
#
# ENDEE INTEGRATION POINTS IN SYSTEM:
# -----------------------------------
# - ingest.py: Uses add_vector() to store document embeddings
# - search.py: Uses search_vectors() for semantic search
# - app.py: Coordinates Endee operations through service layers
# - Docker: Endee runs as separate service via docker-compose.yml

import json
import logging
import os
import socket
import time
from typing import Any, Dict, List, Optional

import aiohttp
import asyncio
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure structured logging for better error tracking
logger = logging.getLogger(__name__)

# Add custom log formatter for structured logging
class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with context."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured context information.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log message string
        """
        # Add structured context to log records
        if hasattr(record, 'operation'):
            record.msg = f"[{record.operation}] {record.msg}"
        if hasattr(record, 'document_id'):
            record.msg = f"{record.msg} (doc_id: {record.document_id})"
        if hasattr(record, 'attempt'):
            record.msg = f"{record.msg} (attempt: {record.attempt})"
        if hasattr(record, 'duration_ms'):
            record.msg = f"{record.msg} (duration: {record.duration_ms}ms)"
        return super().format(record)

# Configuration constants
DEFAULT_ENDEE_URL = "http://localhost:8081"
DEFAULT_TIMEOUT = 30  # Request timeout in seconds
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_RETRY_DELAY = 1.0  # Initial delay in seconds
DEFAULT_MAX_RETRY_DELAY = 10.0  # Maximum delay in seconds

# Optimized connection pooling configuration
DEFAULT_POOL_CONNECTIONS = 20  # Increased connection pool size for better concurrency
DEFAULT_POOL_MAXSIZE = 50  # Increased maximum pool size for high-load scenarios
DEFAULT_KEEPALIVE_TIMEOUT = 30  # Keep connections alive for 30 seconds
DEFAULT_DNS_CACHE_TTL = 300  # DNS cache TTL in seconds
DEFAULT_SOCKET_KEEPALIVE = True  # Enable TCP keepalive

class EndeeClientError(Exception):
    """Base exception for Endee client errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the Endee client error.
        
        Args:
            message: Error message
            error_code: Optional error code for categorization
            context: Optional context dictionary with additional error information
        """
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = time.time()


class EndeeConnectionError(EndeeClientError):
    """Exception for Endee connection failures."""
    
    def __init__(
        self,
        message: str,
        retry_count: Optional[int] = None,
        last_status_code: Optional[int] = None
    ) -> None:
        """Initialize the Endee connection error.
        
        Args:
            message: Error message
            retry_count: Number of retry attempts made
            last_status_code: Last HTTP status code received
        """
        super().__init__(message, error_code="CONNECTION_ERROR")
        self.retry_count = retry_count
        self.last_status_code = last_status_code


class EndeeRequestError(EndeeClientError):
    """Exception for Endee request failures."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None
    ) -> None:
        """Initialize the Endee request error.
        
        Args:
            message: Error message
            status_code: HTTP status code from the failed request
            response_body: Response body content from the failed request
        """
        super().__init__(message, error_code="REQUEST_ERROR")
        self.status_code = status_code
        self.response_body = response_body


class EndeeTimeoutError(EndeeClientError):
    """Exception for Endee timeout errors."""
    
    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        operation: Optional[str] = None
    ) -> None:
        """Initialize the Endee timeout error.
        
        Args:
            message: Error message
            timeout_duration: Duration of the timeout in seconds
            operation: Name of the operation that timed out
        """
        super().__init__(message, error_code="TIMEOUT_ERROR")
        self.timeout_duration = timeout_duration
        self.operation = operation

class ServiceUnavailableError(EndeeClientError):
    """Exception for when Endee service is completely unavailable."""
    
    def __init__(
        self,
        message: str,
        consecutive_failures: Optional[int] = None,
        last_success_time: Optional[float] = None
    ) -> None:
        """Initialize the service unavailable error.
        
        Args:
            message: Error message
            consecutive_failures: Number of consecutive failures
            last_success_time: Timestamp of last successful operation
        """
        super().__init__(message, error_code="SERVICE_UNAVAILABLE")
        self.consecutive_failures = consecutive_failures
        self.last_success_time = last_success_time


class CircuitBreakerState:
    """Circuit breaker state management for Endee service availability."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ) -> None:
        """Initialize circuit breaker state.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery (seconds)
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        # State tracking
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.last_failure_time = None
        self.last_success_time = time.time()
        self.half_open_calls = 0
        
        logger.info(
            f"Circuit breaker initialized: threshold={failure_threshold}, "
            f"recovery_timeout={recovery_timeout}s"
        )
    
    def can_execute(self) -> bool:
        """Check if operation can be executed based on circuit breaker state."""
        current_time = time.time()
        
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                current_time - self.last_failure_time >= self.recovery_timeout):
                self.state = "HALF_OPEN"
                self.half_open_calls = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN state")
                return True
            return False
        elif self.state == "HALF_OPEN":
            return self.half_open_calls < self.half_open_max_calls
        
        return False
    
    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == "HALF_OPEN":
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker closed - service recovered")
        elif self.state == "CLOSED":
            self.failure_count = 0
        
        self.last_success_time = time.time()
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == "CLOSED" and self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"Circuit breaker opened - {self.failure_count} consecutive failures"
            )
        elif self.state == "HALF_OPEN":
            self.state = "OPEN"
            logger.warning("Circuit breaker reopened - recovery attempt failed")
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current circuit breaker state information."""
        current_time = time.time()
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "time_since_last_failure": (
                current_time - self.last_failure_time 
                if self.last_failure_time else None
            ),
            "time_since_last_success": (
                current_time - self.last_success_time 
                if self.last_success_time else None
            ),
            "recovery_timeout": self.recovery_timeout,
            "half_open_calls": self.half_open_calls if self.state == "HALF_OPEN" else None
        }


class EndeeHTTPClient:
    """
    Robust HTTP client for Endee vector database API communication.
    
    ENDEE INTEGRATION OVERVIEW:
    ==========================
    This class serves as the primary interface to the Endee vector database,
    providing robust HTTP communication with comprehensive error handling,
    retry logic, circuit breaker pattern, and performance monitoring.
    
    ENDEE API INTEGRATION:
    ---------------------
    The client integrates with two main Endee API endpoints:
    
    1. Vector Add Endpoint (POST /vectors/add):
       - Purpose: Store document embeddings with metadata in Endee
       - Payload Format: {"id": str, "vector": List[float], "metadata": Dict}
       - Used by: Document ingestion pipeline (ingest.py)
       - Vector Requirements: Exactly 384 dimensions (all-MiniLM-L6-v2 model)
       - Metadata: Includes document content, timestamps, and user metadata
    
    2. Vector Search Endpoint (POST /vectors/search):
       - Purpose: Find similar vectors using cosine similarity
       - Payload Format: {"vector": List[float], "top_k": int}
       - Used by: Semantic search service (search.py)
       - Returns: Ranked results with similarity scores and metadata
    
    ENDEE CONNECTION MANAGEMENT:
    ---------------------------
    - Base URL: Configurable via ENDEE_URL environment variable
    - Default: http://localhost:8081 (matches docker-compose setup)
    - Connection Pooling: 10 connections per pool, max 20 connections
    - Session Management: Persistent HTTP session with connection reuse
    - Headers: JSON content-type, custom User-Agent for identification
    
    ENDEE ERROR HANDLING STRATEGY:
    -----------------------------
    The client implements sophisticated error categorization with circuit breaker:
    
    - Client Errors (4xx): Not retried (bad request, payload too large)
    - Server Errors (5xx): Retried with exponential backoff
    - Network Errors: Connection failures, timeouts (retried)
    - Circuit Breaker: Prevents overwhelming failing service
    - Retry Logic: 3 attempts with 1s, 2s, 4s delays (max 10s)
    - Service Unavailable: Graceful degradation when Endee is down
    
    ENDEE PERFORMANCE MONITORING:
    ----------------------------
    - Operation Tracking: Counts, durations, error rates per operation
    - Structured Logging: Context-aware logs with operation metadata
    - Performance Stats: Available via get_performance_stats()
    - Metrics Tracked: Total calls, success rate, average duration
    - Circuit Breaker Stats: State, failure counts, recovery timing
    
    ENDEE INTEGRATION IN RAG PIPELINE:
    ---------------------------------
    1. Document Ingestion: text → embedding → add_vector() → Endee storage
    2. Query Processing: query → embedding → search_vectors() → similar docs
    3. Context Retrieval: search results → document content → RAG context
    
    Features:
    - Connection pooling for efficient resource usage
    - Retry logic with exponential backoff
    - Circuit breaker pattern for service protection
    - Graceful handling of service unavailability
    - Proper error handling and logging
    - Support for vector add and search operations
    """
    
    def __init__(
        self,
        endee_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        pool_connections: int = DEFAULT_POOL_CONNECTIONS,
        pool_maxsize: int = DEFAULT_POOL_MAXSIZE,
        keepalive_timeout: int = DEFAULT_KEEPALIVE_TIMEOUT,
        enable_socket_keepalive: bool = DEFAULT_SOCKET_KEEPALIVE,
        circuit_breaker_enabled: bool = True,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_recovery_timeout: float = 60.0
    ) -> None:
        """Initialize the Endee HTTP client with optimized connection pooling and circuit breaker.
        
        Args:
            endee_url: Base URL for Endee service (defaults to environment variable or localhost)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            pool_connections: Number of connection pools to cache (increased for better concurrency)
            pool_maxsize: Maximum number of connections in each pool (increased for high-load scenarios)
            keepalive_timeout: Keep-alive timeout for connections in seconds
            enable_socket_keepalive: Enable TCP socket keep-alive for persistent connections
            circuit_breaker_enabled: Enable circuit breaker pattern for service protection
            circuit_breaker_failure_threshold: Number of failures before opening circuit
            circuit_breaker_recovery_timeout: Time to wait before attempting recovery (seconds)
        """
        self.endee_url = endee_url or os.getenv("ENDEE_URL", DEFAULT_ENDEE_URL)
        self.timeout = timeout
        self.max_retries = max_retries
        self.keepalive_timeout = keepalive_timeout
        self.enable_socket_keepalive = enable_socket_keepalive
        
        # Remove trailing slash from URL
        self.endee_url = self.endee_url.rstrip('/')
        
        # Initialize circuit breaker
        self.circuit_breaker_enabled = circuit_breaker_enabled
        if circuit_breaker_enabled:
            self.circuit_breaker = CircuitBreakerState(
                failure_threshold=circuit_breaker_failure_threshold,
                recovery_timeout=circuit_breaker_recovery_timeout
            )
        else:
            self.circuit_breaker = None
        
        # Create HTTP session with optimized connection pooling (for sync fallback)
        self.session = requests.Session()
        
        # Configure optimized retry strategy for connection pooling
        retry_strategy = Retry(
            total=0,  # We handle retries manually for better control
            connect=0,
            read=0,
            redirect=3,
            status_forcelist=[500, 502, 503, 504],
            backoff_factor=0.3,
            respect_retry_after_header=True  # Respect server retry-after headers
        )
        
        # Configure HTTP adapter with optimized connection pooling
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=retry_strategy
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set optimized default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "EndeeRAGSystem/1.0.0",
            "Connection": "keep-alive",  # Explicitly request keep-alive
            "Keep-Alive": f"timeout={keepalive_timeout}, max=100"  # Keep-alive parameters
        })
        
        # Async session will be created when needed
        self._async_session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        
        # Initialize performance tracking
        self._operation_stats = {
            "add_vector": {
                "total_calls": 0,
                "total_duration_ms": 0,
                "error_count": 0
            },
            "search_vectors": {
                "total_calls": 0,
                "total_duration_ms": 0,
                "error_count": 0
            }
        }
        
        # Enhanced logging with structured context
        logger.info(
            "Endee HTTP client initialized with optimized connection pooling and circuit breaker",
            extra={
                "operation": "client_init",
                "endee_url": self.endee_url,
                "timeout": timeout,
                "max_retries": max_retries,
                "circuit_breaker_enabled": circuit_breaker_enabled,
                "pool_config": {
                    "connections": pool_connections,
                    "maxsize": pool_maxsize,
                    "keepalive_timeout": keepalive_timeout,
                    "socket_keepalive": enable_socket_keepalive
                }
            }
        )
    
    async def _get_async_session(self) -> aiohttp.ClientSession:
        """Get or create async HTTP session with optimized connection pooling."""
        if self._async_session is None or self._async_session.closed:
            async with self._session_lock:
                if self._async_session is None or self._async_session.closed:
                    # Create optimized connector with enhanced connection pooling
                    connector = aiohttp.TCPConnector(
                        limit=DEFAULT_POOL_MAXSIZE,  # Total connection pool limit
                        limit_per_host=DEFAULT_POOL_CONNECTIONS,  # Per-host connection limit
                        ttl_dns_cache=DEFAULT_DNS_CACHE_TTL,  # DNS cache TTL
                        use_dns_cache=True,  # Enable DNS caching
                        keepalive_timeout=self.keepalive_timeout,  # Keep-alive timeout
                        enable_cleanup_closed=True,  # Clean up closed connections
                        force_close=False,  # Don't force close connections
                        ssl=False,  # Disable SSL for local Endee service
                    )
                    
                    # Create optimized timeout configuration
                    timeout = aiohttp.ClientTimeout(
                        total=self.timeout,
                        connect=10,  # Connection timeout
                        sock_read=self.timeout,  # Socket read timeout
                        sock_connect=10  # Socket connection timeout
                    )
                    
                    # Create session with optimized headers
                    headers = {
                        "Content-Type": "application/json",
                        "User-Agent": "EndeeRAGSystem/1.0.0",
                        "Connection": "keep-alive",
                        "Keep-Alive": f"timeout={self.keepalive_timeout}, max=100"
                    }
                    
                    self._async_session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers=headers,
                        connector_owner=True,  # Session owns the connector
                        auto_decompress=True,  # Enable automatic decompression
                        trust_env=True  # Trust environment proxy settings
                    )
                    
                    logger.debug(
                        "Created optimized async HTTP session",
                        extra={
                            "operation": "async_session_create",
                            "pool_limit": DEFAULT_POOL_MAXSIZE,
                            "per_host_limit": DEFAULT_POOL_CONNECTIONS,
                            "keepalive_timeout": self.keepalive_timeout,
                            "dns_cache_ttl": DEFAULT_DNS_CACHE_TTL
                        }
                    )
        
        return self._async_session

    async def _handle_async_response(
        self,
        response: aiohttp.ClientResponse,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle async HTTP response and extract JSON data with enhanced error logging.
        
        Args:
            response: HTTP response object from aiohttp
            operation: Operation name for structured logging
            context: Additional context for error logging and debugging
            
        Returns:
            Parsed JSON response data for successful requests
            
        Raises:
            EndeeRequestError: For client errors (4xx) that shouldn't be retried
            EndeeConnectionError: For server errors (5xx) that can be retried
        """
        # Prepare structured logging context
        log_context = {
            "operation": operation,
            "status_code": response.status,
            "response_size": response.content_length or 0
        }
        
        if context:
            log_context.update(context)
        
        # SUCCESS CASE: Parse and return JSON data
        if response.status == 200:
            try:
                result = await response.json()
                logger.debug("Async response processed successfully", extra=log_context)
                return result
            except Exception as e:
                # JSON parsing failed - this is a server-side issue
                response_text = await response.text()
                error_msg = f"Invalid JSON response from Endee {operation}: {str(e)}"
                logger.error(
                    error_msg,
                    extra={**log_context, "error_type": "json_parse_error"}
                )
                raise EndeeRequestError(
                    error_msg, 
                    status_code=response.status,
                    response_body=response_text[:500]
                )
        
        # CLIENT ERROR CASES: Don't retry these
        elif response.status == 400:
            response_text = await response.text()
            error_msg = f"Bad request to Endee {operation}: {response_text}"
            logger.error(
                error_msg,
                extra={
                    **log_context,
                    "error_type": "bad_request",
                    "response_body": response_text[:200]
                }
            )
            raise EndeeRequestError(
                error_msg,
                status_code=response.status,
                response_body=response_text
            )
        
        elif response.status == 413:
            error_msg = f"Payload too large for Endee {operation}"
            logger.error(
                error_msg,
                extra={**log_context, "error_type": "payload_too_large"}
            )
            raise EndeeRequestError(
                error_msg, status_code=response.status
            )
        
        elif response.status == 404:
            error_msg = f"Endee {operation} endpoint not found"
            logger.error(
                error_msg,
                extra={**log_context, "error_type": "endpoint_not_found"}
            )
            raise EndeeRequestError(
                error_msg, status_code=response.status
            )
        
        else:
            # SERVER ERROR CASES: These can potentially be retried
            response_text = await response.text()
            error_msg = (
                f"Endee {operation} failed with status {response.status}: "
                f"{response_text}"
            )
            logger.warning(
                error_msg,
                extra={
                    **log_context,
                    "error_type": "server_error",
                    "retryable": True
                }
            )
            raise EndeeConnectionError(
                error_msg, last_status_code=response.status
            )

    def _log_operation_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log operation performance metrics with structured context.
        
        Args:
            operation: Operation name (add_vector, search_vectors)
            duration_ms: Operation duration in milliseconds
            success: Whether the operation was successful
            context: Additional context for logging
        """
        # Update performance statistics
        if operation in self._operation_stats:
            stats = self._operation_stats[operation]
            stats["total_calls"] += 1
            stats["total_duration_ms"] += duration_ms
            if not success:
                stats["error_count"] += 1
        
        # Log with structured context
        log_context = {
            "operation": operation,
            "duration_ms": round(duration_ms, 2),
            "success": success,
            "avg_duration_ms": round(
                self._operation_stats.get(operation, {}).get("total_duration_ms", 0) / 
                max(1, self._operation_stats.get(operation, {}).get("total_calls", 1)), 2
            )
        }
        
        if context:
            log_context.update(context)
        
        if success:
            logger.info(f"Operation completed successfully", extra=log_context)
        else:
            logger.warning(f"Operation failed", extra=log_context)
    
    def _check_circuit_breaker(self, operation: str) -> None:
        """Check circuit breaker state before executing operation.
        
        Args:
            operation: Name of the operation being attempted
            
        Raises:
            ServiceUnavailableError: If circuit breaker is open
        """
        if not self.circuit_breaker_enabled or not self.circuit_breaker:
            return
        
        if not self.circuit_breaker.can_execute():
            state_info = self.circuit_breaker.get_state_info()
            time_since_failure = state_info['time_since_last_failure']
            failure_time_str = f"{time_since_failure:.1f}s ago" if time_since_failure is not None else "unknown"
            
            error_msg = (
                f"Endee service unavailable - circuit breaker is {state_info['state']}. "
                f"Consecutive failures: {state_info['failure_count']}, "
                f"Last failure: {failure_time_str}"
            )
            
            logger.warning(
                f"Circuit breaker blocking {operation} operation",
                extra={
                    "operation": operation,
                    "circuit_breaker_state": state_info
                }
            )
            
            raise ServiceUnavailableError(
                error_msg,
                consecutive_failures=state_info['failure_count'],
                last_success_time=state_info['last_success_time']
            )
    
    def _record_circuit_breaker_success(self, operation: str) -> None:
        """Record successful operation for circuit breaker.
        
        Args:
            operation: Name of the operation that succeeded
        """
        if not self.circuit_breaker_enabled or not self.circuit_breaker:
            return
        
        old_state = self.circuit_breaker.state
        self.circuit_breaker.record_success()
        new_state = self.circuit_breaker.state
        
        if old_state != new_state:
            logger.info(
                f"Circuit breaker state changed: {old_state} → {new_state}",
                extra={
                    "operation": operation,
                    "circuit_breaker_transition": f"{old_state}_to_{new_state}"
                }
            )
    
    def _record_circuit_breaker_failure(self, operation: str, error: Exception) -> None:
        """Record failed operation for circuit breaker.
        
        Args:
            operation: Name of the operation that failed
            error: The exception that occurred
        """
        if not self.circuit_breaker_enabled or not self.circuit_breaker:
            return
        
        # Only record failures for connection/server errors, not client errors
        if isinstance(error, (EndeeConnectionError, EndeeTimeoutError, 
                             aiohttp.ClientConnectorError, aiohttp.ClientError,
                             requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            old_state = self.circuit_breaker.state
            self.circuit_breaker.record_failure()
            new_state = self.circuit_breaker.state
            
            logger.warning(
                f"Circuit breaker recorded failure for {operation}",
                extra={
                    "operation": operation,
                    "error_type": type(error).__name__,
                    "circuit_breaker_state": new_state,
                    "failure_count": self.circuit_breaker.failure_count
                }
            )
            
            if old_state != new_state:
                logger.warning(
                    f"Circuit breaker state changed: {old_state} → {new_state}",
                    extra={
                        "operation": operation,
                        "circuit_breaker_transition": f"{old_state}_to_{new_state}",
                        "consecutive_failures": self.circuit_breaker.failure_count
                    }
                )
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get current service health status including circuit breaker state.
        
        Returns:
            Dictionary containing service health information
        """
        health_info = {
            "endee_url": self.endee_url,
            "circuit_breaker_enabled": self.circuit_breaker_enabled,
            "service_available": True,
            "last_check_time": time.time()
        }
        
        if self.circuit_breaker_enabled and self.circuit_breaker:
            circuit_state = self.circuit_breaker.get_state_info()
            health_info.update({
                "circuit_breaker": circuit_state,
                "service_available": circuit_state["state"] != "OPEN"
            })
        
        return health_info
    
    async def check_service_availability(self) -> Dict[str, Any]:
        """Check if Endee service is available by making a lightweight test request.
        
        This method performs a minimal health check to determine if the Endee service
        is responding. It uses a small test vector search that should complete quickly
        and not impact service performance.
        
        Returns:
            Dictionary containing availability status and response time
            
        Raises:
            ServiceUnavailableError: If service is unavailable
        """
        check_start = time.time()
        
        try:
            # Check circuit breaker first
            self._check_circuit_breaker("health_check")
            
            # Create minimal test payload for health check
            test_payload = {
                "vector": [0.0] * 384,  # Minimal test vector
                "top_k": 1
            }
            
            session = await self._get_async_session()
            url = f"{self.endee_url}/vectors/search"
            
            async with session.post(
                url,
                json=test_payload,
                timeout=aiohttp.ClientTimeout(total=5)  # Short timeout for health check
            ) as response:
                await response.read()  # Consume response
                
                response_time = (time.time() - check_start) * 1000
                
                # Record success for circuit breaker
                self._record_circuit_breaker_success("health_check")
                
                result = {
                    "available": True,
                    "response_time_ms": round(response_time, 2),
                    "status_code": response.status,
                    "endee_url": self.endee_url,
                    "check_time": time.time()
                }
                
                logger.debug(
                    f"Endee service health check successful: {response_time:.1f}ms",
                    extra={
                        "operation": "health_check",
                        "response_time_ms": round(response_time, 2),
                        "status_code": response.status
                    }
                )
                
                return result
                
        except Exception as e:
            response_time = (time.time() - check_start) * 1000
            
            # Record failure for circuit breaker
            self._record_circuit_breaker_failure("health_check", e)
            
            error_info = {
                "available": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "response_time_ms": round(response_time, 2),
                "endee_url": self.endee_url,
                "check_time": time.time()
            }
            
            logger.warning(
                f"Endee service health check failed: {str(e)}",
                extra={
                    "operation": "health_check",
                    "error_type": type(e).__name__,
                    "response_time_ms": round(response_time, 2)
                }
            )
            
            # Raise ServiceUnavailableError for consistent error handling
            raise ServiceUnavailableError(
                f"Endee service health check failed: {str(e)}",
                consecutive_failures=self.circuit_breaker.failure_count if self.circuit_breaker else None
            )
    
    def reset_circuit_breaker(self) -> Dict[str, Any]:
        """Manually reset the circuit breaker to closed state.
        
        This method allows manual recovery from circuit breaker open state,
        useful for administrative operations or after confirming service recovery.
        
        Returns:
            Dictionary containing the previous and new circuit breaker state
        """
        if not self.circuit_breaker_enabled or not self.circuit_breaker:
            return {
                "circuit_breaker_enabled": False,
                "message": "Circuit breaker is not enabled"
            }
        
        old_state = self.circuit_breaker.get_state_info()
        
        # Reset circuit breaker state
        self.circuit_breaker.state = "CLOSED"
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.last_failure_time = None
        self.circuit_breaker.half_open_calls = 0
        
        new_state = self.circuit_breaker.get_state_info()
        
        logger.info(
            "Circuit breaker manually reset to CLOSED state",
            extra={
                "operation": "circuit_breaker_reset",
                "old_state": old_state["state"],
                "new_state": new_state["state"]
            }
        )
        
        return {
            "circuit_breaker_enabled": True,
            "old_state": old_state,
            "new_state": new_state,
            "message": "Circuit breaker reset to CLOSED state"
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for monitoring and debugging.
        
        Returns:
            Dictionary containing performance metrics, connection pool stats, and circuit breaker state
        """
        stats = {}
        for operation, data in self._operation_stats.items():
            if data["total_calls"] > 0:
                stats[operation] = {
                    "total_calls": data["total_calls"],
                    "error_count": data["error_count"],
                    "success_rate": round((data["total_calls"] - data["error_count"]) / data["total_calls"] * 100, 2),
                    "avg_duration_ms": round(data["total_duration_ms"] / data["total_calls"], 2),
                    "total_duration_ms": round(data["total_duration_ms"], 2)
                }
        
        # Add connection pool statistics
        stats["connection_pool"] = self.get_connection_pool_stats()
        
        # Add circuit breaker statistics
        if self.circuit_breaker_enabled and self.circuit_breaker:
            stats["circuit_breaker"] = self.circuit_breaker.get_state_info()
        else:
            stats["circuit_breaker"] = {"enabled": False}
        
        # Add service health information
        stats["service_health"] = self.get_service_health()
        
        return stats
        """
        Get performance statistics for monitoring and debugging.
        
        Returns:
            Dictionary containing performance metrics and connection pool stats
        """
        stats = {}
        for operation, data in self._operation_stats.items():
            if data["total_calls"] > 0:
                stats[operation] = {
                    "total_calls": data["total_calls"],
                    "error_count": data["error_count"],
                    "success_rate": round((data["total_calls"] - data["error_count"]) / data["total_calls"] * 100, 2),
                    "avg_duration_ms": round(data["total_duration_ms"] / data["total_calls"], 2),
                    "total_duration_ms": round(data["total_duration_ms"], 2)
                }
        
        # Add connection pool statistics
        stats["connection_pool"] = self.get_connection_pool_stats()
        
        return stats
    
    def get_connection_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics for monitoring connection usage.
        
        Returns:
            Dictionary containing connection pool metrics
        """
        pool_stats = {
            "sync_session": {
                "configured": True,
                "pool_connections": DEFAULT_POOL_CONNECTIONS,
                "pool_maxsize": DEFAULT_POOL_MAXSIZE,
                "keepalive_timeout": self.keepalive_timeout,
                "socket_keepalive": self.enable_socket_keepalive
            },
            "async_session": {
                "created": self._async_session is not None and not self._async_session.closed,
                "pool_limit": DEFAULT_POOL_MAXSIZE,
                "per_host_limit": DEFAULT_POOL_CONNECTIONS,
                "dns_cache_ttl": DEFAULT_DNS_CACHE_TTL,
                "keepalive_timeout": self.keepalive_timeout
            }
        }
        
        # Add async connector stats if session exists
        if self._async_session and not self._async_session.closed:
            connector = self._async_session.connector
            if hasattr(connector, '_conns'):
                # Get connection statistics from aiohttp connector
                total_connections = sum(len(conns) for conns in connector._conns.values())
                pool_stats["async_session"]["active_connections"] = total_connections
                pool_stats["async_session"]["connection_keys"] = len(connector._conns)
        
        return pool_stats
    
    async def warmup_connections(self, num_connections: int = 5) -> Dict[str, Any]:
        """
        Warm up connection pool by pre-establishing connections to Endee.
        
        This method creates multiple concurrent connections to the Endee service
        to populate the connection pool, reducing latency for subsequent requests.
        
        Args:
            num_connections: Number of connections to establish (default: 5)
            
        Returns:
            Dictionary with warmup results and statistics
        """
        warmup_start = time.time()
        successful_connections = 0
        failed_connections = 0
        
        logger.info(
            f"Starting connection pool warmup with {num_connections} connections",
            extra={
                "operation": "connection_warmup",
                "target_connections": num_connections,
                "endee_url": self.endee_url
            }
        )
        
        # Get async session to initialize connection pool
        session = await self._get_async_session()
        
        async def test_connection(connection_id: int) -> bool:
            """Test a single connection to Endee."""
            try:
                # Use a lightweight endpoint or health check if available
                # For now, we'll use a minimal vector search that should fail gracefully
                test_payload = {
                    "vector": [0.0] * 384,  # Minimal test vector
                    "top_k": 1
                }
                
                async with session.post(
                    f"{self.endee_url}/vectors/search",
                    json=test_payload,
                    timeout=aiohttp.ClientTimeout(total=5)  # Short timeout for warmup
                ) as response:
                    # We don't care about the response content, just that we can connect
                    await response.read()
                    logger.debug(
                        f"Warmup connection {connection_id} established successfully",
                        extra={"operation": "warmup_connection", "connection_id": connection_id}
                    )
                    return True
                    
            except Exception as e:
                logger.debug(
                    f"Warmup connection {connection_id} failed: {str(e)}",
                    extra={
                        "operation": "warmup_connection",
                        "connection_id": connection_id,
                        "error": str(e)
                    }
                )
                return False
        
        # Create concurrent connections
        tasks = [test_connection(i) for i in range(num_connections)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful connections
        for result in results:
            if isinstance(result, bool) and result:
                successful_connections += 1
            else:
                failed_connections += 1
        
        warmup_duration = (time.time() - warmup_start) * 1000
        
        warmup_stats = {
            "total_attempts": num_connections,
            "successful_connections": successful_connections,
            "failed_connections": failed_connections,
            "success_rate": round(successful_connections / num_connections * 100, 2),
            "warmup_duration_ms": round(warmup_duration, 2)
        }
        
        logger.info(
            f"Connection pool warmup completed: {successful_connections}/{num_connections} successful",
            extra={
                "operation": "connection_warmup_complete",
                **warmup_stats
            }
        )
        
        return warmup_stats
    
    def _exponential_backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay for retry attempts.
        
        This implements an exponential backoff strategy to avoid overwhelming
        the Endee service during temporary failures. The delay doubles with each
        retry attempt but is capped at a maximum to prevent excessively long waits.
        
        Algorithm:
        - Attempt 0: 1.0 seconds
        - Attempt 1: 2.0 seconds  
        - Attempt 2: 4.0 seconds
        - Attempt 3+: 10.0 seconds (capped)
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds before next retry attempt
        """
        # Calculate exponential delay: base_delay * 2^attempt
        # This creates progressively longer delays to reduce load on failing service
        delay = DEFAULT_INITIAL_RETRY_DELAY * (2 ** attempt)
        
        # Cap the delay to prevent excessive wait times that could timeout clients
        # Maximum delay ensures we don't wait more than 10 seconds between retries
        return min(delay, DEFAULT_MAX_RETRY_DELAY)
    
    def _handle_response(
        self,
        response: requests.Response,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle HTTP response and extract JSON data with enhanced error logging.
        
        This method implements sophisticated HTTP response handling that categorizes
        errors and determines retry strategies. The error handling logic includes:
        
        1. Success case (200): Parse JSON and return data
        2. Client errors (4xx): Don't retry, raise immediately
        3. Server errors (5xx): Can be retried with backoff
        4. Special cases: Payload too large, endpoint not found
        
        The categorization is critical for the retry logic - client errors
        indicate problems with the request that won't be fixed by retrying,
        while server errors may be transient and worth retrying.
        
        Args:
            response: HTTP response object from requests
            operation: Operation name for structured logging
            context: Additional context for error logging and debugging
            
        Returns:
            Parsed JSON response data for successful requests
            
        Raises:
            EndeeRequestError: For client errors (4xx) that shouldn't be retried
            EndeeConnectionError: For server errors (5xx) that can be retried
        """
        # Prepare structured logging context
        log_context = {
            "operation": operation,
            "status_code": response.status_code,
            "response_size": len(response.content) if response.content else 0
        }
        
        if context:
            log_context.update(context)
        
        # SUCCESS CASE: Parse and return JSON data
        if response.status_code == 200:
            try:
                result = response.json()
                logger.debug("Response processed successfully", extra=log_context)
                return result
            except ValueError as e:
                # JSON parsing failed - this is a server-side issue
                error_msg = f"Invalid JSON response from Endee {operation}: {str(e)}"
                logger.error(
                    error_msg,
                    extra={**log_context, "error_type": "json_parse_error"}
                )
                raise EndeeRequestError(
                    error_msg, 
                    status_code=response.status_code,
                    response_body=response.text[:500]  # Limit response body size in error
                )
        
        # CLIENT ERROR CASES: Don't retry these
        elif response.status_code == 400:
            # Bad request - indicates problem with our request format/data
            error_msg = f"Bad request to Endee {operation}: {response.text}"
            logger.error(
                error_msg,
                extra={
                    **log_context,
                    "error_type": "bad_request",
                    "response_body": response.text[:200]
                }
            )
            raise EndeeRequestError(
                error_msg,
                status_code=response.status_code,
                response_body=response.text
            )
        
        elif response.status_code == 413:
            # Payload too large - our data exceeds Endee's limits
            error_msg = f"Payload too large for Endee {operation}"
            logger.error(
                error_msg,
                extra={**log_context, "error_type": "payload_too_large"}
            )
            raise EndeeRequestError(
                error_msg, status_code=response.status_code
            )
        
        elif response.status_code == 404:
            # Not found - endpoint doesn't exist or is misconfigured
            error_msg = f"Endee {operation} endpoint not found"
            logger.error(
                error_msg,
                extra={**log_context, "error_type": "endpoint_not_found"}
            )
            raise EndeeRequestError(
                error_msg, status_code=response.status_code
            )
        
        else:
            # SERVER ERROR CASES: These can potentially be retried
            # Includes 5xx errors, rate limiting, temporary unavailability
            error_msg = (
                f"Endee {operation} failed with status {response.status_code}: "
                f"{response.text}"
            )
            logger.warning(
                error_msg,
                extra={
                    **log_context,
                    "error_type": "server_error",
                    "retryable": True
                }
            )
            raise EndeeConnectionError(
                error_msg, last_status_code=response.status_code
            )
    
    async def add_vector_async(
        self,
        document_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a vector to the Endee database with retry logic and enhanced error handling.
        
        ENDEE INTEGRATION: VECTOR STORAGE
        ================================
        This method implements the core document storage functionality for the RAG system.
        It communicates with Endee's POST /vectors/add endpoint to store document embeddings
        along with their metadata for later retrieval during semantic search.
        
        Args:
            document_id: Unique identifier for the document in Endee
            vector: Vector embedding (should be 384-dimensional from all-MiniLM-L6-v2)
            metadata: Optional metadata dictionary (includes content, timestamps, etc.)
            
        Returns:
            Response from Endee add operation with confirmation details
            
        Raises:
            EndeeRequestError: For client errors (4xx) that shouldn't be retried
            EndeeConnectionError: For connection/server errors after all retries
            EndeeTimeoutError: For timeout errors after all retries
        """
        operation_start = time.time()
        operation_context = {
            "operation": "add_vector",
            "document_id": document_id,
            "vector_dimension": len(vector) if vector else 0,
            "has_metadata": metadata is not None
        }
        
        try:
            # Check circuit breaker before attempting operation
            self._check_circuit_breaker("add_vector")
            
            # Input validation with enhanced error context
            if not document_id:
                raise EndeeRequestError("Document ID is required for vector add operation")
            
            if not vector:
                raise EndeeRequestError("Vector is required for add operation")
            
            if not isinstance(vector, list):
                raise EndeeRequestError("Vector must be a list of floats")
            
            if len(vector) != 384:
                raise EndeeRequestError(f"Vector must be 384-dimensional, got {len(vector)}")
            
            # Validate vector values
            if not all(isinstance(x, (int, float)) for x in vector):
                raise EndeeRequestError("Vector must contain only numeric values")
            
            # Prepare payload
            payload = {
                "id": document_id,
                "vector": vector
            }
            
            if metadata:
                payload["metadata"] = metadata
            
            # Validate payload size (1MB limit)
            payload_size = len(str(payload))
            if payload_size > 1000000:
                raise EndeeRequestError(f"Payload too large: {payload_size} bytes (max 1MB)")
            
            url = f"{self.endee_url}/vectors/add"
            last_exception = None
            
            # Get async session
            session = await self._get_async_session()
            
            for attempt in range(self.max_retries):
                attempt_start = time.time()
                attempt_context = {**operation_context, "attempt": attempt + 1, "max_retries": self.max_retries}
                
                try:
                    logger.debug(
                        f"Attempting async vector add for document {document_id}",
                        extra=attempt_context
                    )
                    
                    async with session.post(url, json=payload) as response:
                        # Handle response with enhanced context
                        result = await self._handle_async_response(response, "vector add", attempt_context)
                    
                    # Log successful operation with performance metrics
                    duration_ms = (time.time() - operation_start) * 1000
                    self._log_operation_performance("add_vector", duration_ms, True, {
                        "document_id": document_id,
                        "attempts_used": attempt + 1,
                        "payload_size_bytes": payload_size
                    })
                    
                    # Record success for circuit breaker
                    self._record_circuit_breaker_success("add_vector")
                    
                    logger.info(
                        f"Successfully added vector for document {document_id}",
                        extra={**attempt_context, "duration_ms": round(duration_ms, 2)}
                    )
                    return result
                    
                except EndeeRequestError:
                    # Client errors shouldn't be retried
                    duration_ms = (time.time() - operation_start) * 1000
                    self._log_operation_performance("add_vector", duration_ms, False, {
                        "document_id": document_id,
                        "error_type": "client_error",
                        "attempts_used": attempt + 1
                    })
                    raise
                    
                except asyncio.TimeoutError as e:
                    attempt_duration = (time.time() - attempt_start) * 1000
                    error_msg = f"Timeout adding vector to Endee (attempt {attempt + 1}): {str(e)}"
                    logger.warning(
                        error_msg,
                        extra={**attempt_context, "error_type": "timeout", "attempt_duration_ms": round(attempt_duration, 2)}
                    )
                    last_exception = EndeeTimeoutError(error_msg, timeout_duration=self.timeout, operation="add_vector")
                    
                except aiohttp.ClientConnectorError as e:
                    attempt_duration = (time.time() - attempt_start) * 1000
                    # Handle potential string conversion issues with mocked exceptions
                    try:
                        error_str = str(e)
                    except Exception:
                        error_str = f"{type(e).__name__}: Connection error"
                    
                    error_msg = f"Connection error adding vector to Endee (attempt {attempt + 1}): {error_str}"
                    logger.warning(
                        error_msg,
                        extra={**attempt_context, "error_type": "connection_error", "attempt_duration_ms": round(attempt_duration, 2)}
                    )
                    last_exception = EndeeConnectionError(error_msg, retry_count=attempt + 1)
                    
                except aiohttp.ClientError as e:
                    attempt_duration = (time.time() - attempt_start) * 1000
                    # Handle potential string conversion issues with mocked exceptions
                    try:
                        error_str = str(e)
                    except Exception:
                        error_str = f"{type(e).__name__}: Client error"
                    
                    error_msg = f"Client error adding vector to Endee (attempt {attempt + 1}): {error_str}"
                    logger.warning(
                        error_msg,
                        extra={**attempt_context, "error_type": "client_error", "attempt_duration_ms": round(attempt_duration, 2)}
                    )
                    last_exception = EndeeConnectionError(error_msg, retry_count=attempt + 1)
                    
                except EndeeConnectionError as e:
                    # Server errors can be retried
                    attempt_duration = (time.time() - attempt_start) * 1000
                    logger.warning(
                        f"Server error adding vector to Endee (attempt {attempt + 1}): {str(e)}",
                        extra={**attempt_context, "error_type": "server_error", "attempt_duration_ms": round(attempt_duration, 2)}
                    )
                    last_exception = e
                    
                except Exception as e:
                    attempt_duration = (time.time() - attempt_start) * 1000
                    error_msg = f"Unexpected error adding vector to Endee (attempt {attempt + 1}): {str(e)}"
                    logger.warning(
                        error_msg,
                        extra={**attempt_context, "error_type": "unexpected_error", "attempt_duration_ms": round(attempt_duration, 2)}
                    )
                    last_exception = EndeeConnectionError(error_msg, retry_count=attempt + 1)
                
                # Wait before retry (except on last attempt)
                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff_delay(attempt)
                    logger.info(
                        f"Retrying vector add in {delay:.1f} seconds...",
                        extra={**attempt_context, "retry_delay_seconds": delay}
                    )
                    await asyncio.sleep(delay)
            
            # All retries failed - log final failure with comprehensive context
            duration_ms = (time.time() - operation_start) * 1000
            self._log_operation_performance("add_vector", duration_ms, False, {
                "document_id": document_id,
                "error_type": "max_retries_exceeded",
                "attempts_used": self.max_retries,
                "final_exception_type": type(last_exception).__name__ if last_exception else "unknown"
            })
            
            # Record failure for circuit breaker only once per operation (not per retry)
            if last_exception:
                self._record_circuit_breaker_failure("add_vector", last_exception)
            
            error_msg = f"Failed to add vector to Endee after {self.max_retries} attempts"
            logger.error(
                error_msg,
                extra={
                    **operation_context,
                    "total_duration_ms": round(duration_ms, 2),
                    "final_exception": str(last_exception) if last_exception else "unknown"
                }
            )
            
            if isinstance(last_exception, EndeeTimeoutError):
                raise EndeeTimeoutError(f"{error_msg}: {str(last_exception)}", timeout_duration=self.timeout, operation="add_vector")
            elif last_exception:
                raise EndeeConnectionError(f"{error_msg}: {str(last_exception)}", retry_count=self.max_retries)
            else:
                raise EndeeConnectionError(error_msg, retry_count=self.max_retries)
                
        except (EndeeRequestError, EndeeConnectionError, EndeeTimeoutError, ServiceUnavailableError):
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            # Handle any unexpected exceptions
            duration_ms = (time.time() - operation_start) * 1000
            self._log_operation_performance("add_vector", duration_ms, False, {
                "document_id": document_id,
                "error_type": "unexpected_exception",
                "exception_type": type(e).__name__
            })
            
            error_msg = f"Unexpected error in add_vector: {str(e)}"
            logger.error(
                error_msg,
                extra={**operation_context, "total_duration_ms": round(duration_ms, 2)},
                exc_info=True
            )
            raise EndeeClientError(error_msg)
            raise EndeeClientError(error_msg)
    
    async def search_vectors_async(
        self,
        query_vector: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Search for similar vectors in the Endee database with retry logic and enhanced error handling.
        
        ENDEE INTEGRATION: VECTOR SIMILARITY SEARCH
        ==========================================
        This method implements the core semantic search functionality for the RAG system.
        It communicates with Endee's POST /vectors/search endpoint to find documents
        most similar to a query vector using cosine similarity.
        
        ENDEE API INTERACTION:
        ---------------------
        - Endpoint: POST {endee_url}/vectors/search
        - Payload Structure:
          {
            "vector": [384 float values],
            "top_k": 5,
            "similarity_threshold": 0.0  // Optional client-side filtering
          }
        - Response Structure:
          {
            "results": [
              {
                "id": "document-id",
                "score": 0.85,  // Cosine similarity score (0-1)
                "metadata": {
                  "content": "document text",
                  "created_at": "timestamp",
                  ...
                }
              }
            ]
          }
        
        ENDEE SEARCH MECHANICS:
        ----------------------
        - Similarity Metric: Cosine similarity between query and stored vectors
        - Score Range: 0.0 (no similarity) to 1.0 (identical vectors)
        - Ranking: Results returned in descending order of similarity score
        - Top-K Selection: Endee returns the K most similar documents
        - Threshold Filtering: Client-side filtering by minimum similarity score
        
        ENDEE SEARCH REQUIREMENTS:
        -------------------------
        - Query Vector: Exactly 384 dimensions (matches stored embeddings)
        - Vector Values: All numeric, normalized for consistent similarity calculation
        - Top-K Range: 1-1000 results (reasonable limits for performance)
        - Similarity Threshold: 0.0-1.0 range for score filtering
        
        INTEGRATION WITH RAG PIPELINE:
        -----------------------------
        This method is called by the vector search service (search.py) during
        the RAG retrieval phase:
        1. User query → embedding generation (384d vector)
        2. search_vectors() → Find similar documents in Endee
        3. Result processing → Extract content and metadata
        4. Context combination → Prepare for LLM generation
        
        ENDEE PERFORMANCE CONSIDERATIONS:
        --------------------------------
        - Vector search is computationally intensive in Endee
        - Larger top_k values increase search time and memory usage
        - Similarity threshold filtering reduces result processing overhead
        - Connection pooling minimizes per-request connection overhead
        
        Args:
            query_vector: Query vector embedding (should be 384-dimensional from all-MiniLM-L6-v2)
            top_k: Number of top results to return (default: 5, max: 1000)
            similarity_threshold: Minimum similarity score for results (0.0 to 1.0, default: 0.0)
            
        Returns:
            Response from Endee search operation containing ranked results list
            
        Raises:
            EndeeRequestError: For client errors (4xx) that shouldn't be retried
            EndeeConnectionError: For connection/server errors after all retries
            EndeeTimeoutError: For timeout errors after all retries
        """
        operation_start = time.time()
        operation_context = {
            "operation": "search_vectors",
            "vector_dimension": len(query_vector) if query_vector else 0,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold
        }
        
        try:
            # Check circuit breaker before attempting operation
            self._check_circuit_breaker("search_vectors")
            
            # Input validation with enhanced error context
            if not query_vector:
                raise EndeeRequestError("Query vector is required for search operation")
            
            if not isinstance(query_vector, list):
                raise EndeeRequestError("Query vector must be a list of floats")
            
            if len(query_vector) != 384:
                raise EndeeRequestError(f"Query vector must be 384-dimensional, got {len(query_vector)}")
            
            # Validate vector values
            if not all(isinstance(x, (int, float)) for x in query_vector):
                raise EndeeRequestError("Query vector must contain only numeric values")
            
            # Validate top_k parameter
            if not isinstance(top_k, int) or top_k <= 0:
                raise EndeeRequestError("top_k must be a positive integer")
            
            if top_k > 1000:
                raise EndeeRequestError("top_k cannot exceed 1000")
            
            # Validate similarity_threshold parameter
            if not isinstance(similarity_threshold, (int, float)):
                raise EndeeRequestError("similarity_threshold must be a number")
            
            if not (0.0 <= similarity_threshold <= 1.0):
                raise EndeeRequestError("similarity_threshold must be between 0.0 and 1.0")
            
            # Prepare payload
            payload = {
                "vector": query_vector,
                "top_k": top_k
            }
            
            # Add similarity threshold if specified
            if similarity_threshold > 0.0:
                payload["similarity_threshold"] = similarity_threshold
            
            # Validate payload size (1MB limit)
            payload_size = len(str(payload))
            if payload_size > 1000000:
                raise EndeeRequestError(f"Payload too large: {payload_size} bytes (max 1MB)")
            
            url = f"{self.endee_url}/vectors/search"
            last_exception = None
            
            # Get async session
            session = await self._get_async_session()
            
            for attempt in range(self.max_retries):
                attempt_start = time.time()
                attempt_context = {**operation_context, "attempt": attempt + 1, "max_retries": self.max_retries}
                
                try:
                    logger.debug(
                        f"Attempting async vector search",
                        extra=attempt_context
                    )
                    
                    async with session.post(url, json=payload) as response:
                        # Handle response with enhanced context
                        result = await self._handle_async_response(response, "vector search", attempt_context)
                    
                    # Log successful operation with performance metrics
                    duration_ms = (time.time() - operation_start) * 1000
                    results_count = len(result.get('results', []))
                    
                    self._log_operation_performance("search_vectors", duration_ms, True, {
                        "top_k": top_k,
                        "results_count": results_count,
                        "attempts_used": attempt + 1,
                        "payload_size_bytes": payload_size
                    })
                    
                    # Record success for circuit breaker
                    self._record_circuit_breaker_success("search_vectors")
                    
                    logger.info(
                        f"Successfully performed vector search, found {results_count} results",
                        extra={**attempt_context, "duration_ms": round(duration_ms, 2), "results_count": results_count}
                    )
                    return result
                    
                except EndeeRequestError:
                    # Client errors shouldn't be retried
                    duration_ms = (time.time() - operation_start) * 1000
                    self._log_operation_performance("search_vectors", duration_ms, False, {
                        "top_k": top_k,
                        "error_type": "client_error",
                        "attempts_used": attempt + 1
                    })
                    raise
                    
                except asyncio.TimeoutError as e:
                    attempt_duration = (time.time() - attempt_start) * 1000
                    error_msg = f"Timeout searching vectors in Endee (attempt {attempt + 1}): {str(e)}"
                    logger.warning(
                        error_msg,
                        extra={**attempt_context, "error_type": "timeout", "attempt_duration_ms": round(attempt_duration, 2)}
                    )
                    last_exception = EndeeTimeoutError(error_msg, timeout_duration=self.timeout, operation="search_vectors")
                    
                except aiohttp.ClientConnectorError as e:
                    attempt_duration = (time.time() - attempt_start) * 1000
                    # Handle potential string conversion issues with mocked exceptions
                    try:
                        error_str = str(e)
                    except Exception:
                        error_str = f"{type(e).__name__}: Connection error"
                    
                    error_msg = f"Connection error searching vectors in Endee (attempt {attempt + 1}): {error_str}"
                    logger.warning(
                        error_msg,
                        extra={**attempt_context, "error_type": "connection_error", "attempt_duration_ms": round(attempt_duration, 2)}
                    )
                    last_exception = EndeeConnectionError(error_msg, retry_count=attempt + 1)
                    
                except aiohttp.ClientError as e:
                    attempt_duration = (time.time() - attempt_start) * 1000
                    # Handle potential string conversion issues with mocked exceptions
                    try:
                        error_str = str(e)
                    except Exception:
                        error_str = f"{type(e).__name__}: Client error"
                    
                    error_msg = f"Client error searching vectors in Endee (attempt {attempt + 1}): {error_str}"
                    logger.warning(
                        error_msg,
                        extra={**attempt_context, "error_type": "client_error", "attempt_duration_ms": round(attempt_duration, 2)}
                    )
                    last_exception = EndeeConnectionError(error_msg, retry_count=attempt + 1)
                    
                except EndeeConnectionError as e:
                    # Server errors can be retried
                    attempt_duration = (time.time() - attempt_start) * 1000
                    logger.warning(
                        f"Server error searching vectors in Endee (attempt {attempt + 1}): {str(e)}",
                        extra={**attempt_context, "error_type": "server_error", "attempt_duration_ms": round(attempt_duration, 2)}
                    )
                    last_exception = e
                    
                except Exception as e:
                    attempt_duration = (time.time() - attempt_start) * 1000
                    error_msg = f"Unexpected error searching vectors in Endee (attempt {attempt + 1}): {str(e)}"
                    logger.warning(
                        error_msg,
                        extra={**attempt_context, "error_type": "unexpected_error", "attempt_duration_ms": round(attempt_duration, 2)}
                    )
                    last_exception = EndeeConnectionError(error_msg, retry_count=attempt + 1)
                
                # Wait before retry (except on last attempt)
                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff_delay(attempt)
                    logger.info(
                        f"Retrying vector search in {delay:.1f} seconds...",
                        extra={**attempt_context, "retry_delay_seconds": delay}
                    )
                    await asyncio.sleep(delay)
            
            # All retries failed - log final failure with comprehensive context
            duration_ms = (time.time() - operation_start) * 1000
            self._log_operation_performance("search_vectors", duration_ms, False, {
                "top_k": top_k,
                "error_type": "max_retries_exceeded",
                "attempts_used": self.max_retries,
                "final_exception_type": type(last_exception).__name__ if last_exception else "unknown"
            })
            
            # Record failure for circuit breaker only once per operation (not per retry)
            if last_exception:
                self._record_circuit_breaker_failure("search_vectors", last_exception)
            
            error_msg = f"Failed to search vectors in Endee after {self.max_retries} attempts"
            logger.error(
                error_msg,
                extra={
                    **operation_context,
                    "total_duration_ms": round(duration_ms, 2),
                    "final_exception": str(last_exception) if last_exception else "unknown"
                }
            )
            
            if isinstance(last_exception, EndeeTimeoutError):
                raise EndeeTimeoutError(f"{error_msg}: {str(last_exception)}", timeout_duration=self.timeout, operation="search_vectors")
            elif last_exception:
                raise EndeeConnectionError(f"{error_msg}: {str(last_exception)}", retry_count=self.max_retries)
            else:
                raise EndeeConnectionError(error_msg, retry_count=self.max_retries)
                
        except (EndeeRequestError, EndeeConnectionError, EndeeTimeoutError, ServiceUnavailableError):
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            # Handle any unexpected exceptions
            duration_ms = (time.time() - operation_start) * 1000
            self._log_operation_performance("search_vectors", duration_ms, False, {
                "top_k": top_k,
                "error_type": "unexpected_exception",
                "exception_type": type(e).__name__
            })
            
            error_msg = f"Unexpected error in search_vectors: {str(e)}"
            logger.error(
                error_msg,
                extra={**operation_context, "total_duration_ms": round(duration_ms, 2)},
                exc_info=True
            )
            raise EndeeClientError(error_msg)
    
    async def close(self) -> None:
        """Close the HTTP sessions and clean up resources with performance summary."""
        # Close async session if it exists
        if self._async_session and not self._async_session.closed:
            await self._async_session.close()
            logger.debug("Closed async HTTP session")
        
        # Close sync session
        if self.session:
            # Log performance summary before closing
            stats = self.get_performance_stats()
            if stats:
                logger.info(
                    "Endee HTTP client sessions closing - Performance Summary",
                    extra={
                        "operation": "client_close",
                        "performance_stats": stats
                    }
                )
            else:
                logger.info(
                    "Endee HTTP client sessions closing - No operations performed"
                )
            
            self.session.close()
            logger.info("Endee HTTP client sessions closed")
    
    def close_sync(self) -> None:
        """Synchronous close method for compatibility."""
        if self.session:
            stats = self.get_performance_stats()
            if stats:
                logger.info(
                    "Endee HTTP client session closing - Performance Summary",
                    extra={
                        "operation": "client_close",
                        "performance_stats": stats
                    }
                )
            else:
                logger.info(
                    "Endee HTTP client session closing - No operations performed"
                )
            
            self.session.close()
            logger.info("Endee HTTP client session closed")
    
    def __enter__(self) -> "EndeeHTTPClient":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close_sync()
    
    async def __aenter__(self) -> "EndeeHTTPClient":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
    
    def add_vector(
        self,
        document_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add a vector to the Endee database (synchronous wrapper).
        
        This method provides backward compatibility for existing code that expects
        synchronous operation. It runs the async implementation in an event loop.
        
        Args:
            document_id: Unique identifier for the document in Endee
            vector: Vector embedding (should be 384-dimensional from all-MiniLM-L6-v2)
            metadata: Optional metadata dictionary (includes content, timestamps, etc.)
            
        Returns:
            Response from Endee add operation with confirmation details
            
        Raises:
            EndeeRequestError: For client errors (4xx) that shouldn't be retried
            EndeeConnectionError: For connection/server errors after all retries
            EndeeTimeoutError: For timeout errors after all retries
            ServiceUnavailableError: If circuit breaker is open
        """
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we can't use run_until_complete
                # This typically happens in async contexts like FastAPI
                raise RuntimeError(
                    "Cannot call sync method from async context. Use add_vector_async() instead."
                )
            else:
                return loop.run_until_complete(
                    self.add_vector_async(document_id, vector, metadata)
                )
        except RuntimeError:
            # No event loop exists, create a new one
            return asyncio.run(self.add_vector_async(document_id, vector, metadata))
    
    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Search for similar vectors in the Endee database (synchronous wrapper).
        
        This method provides backward compatibility for existing code that expects
        synchronous operation. It runs the async implementation in an event loop.
        
        Args:
            query_vector: Query vector embedding (should be 384-dimensional from all-MiniLM-L6-v2)
            top_k: Number of top results to return (default: 5, max: 1000)
            similarity_threshold: Minimum similarity score for results (0.0 to 1.0, default: 0.0)
            
        Returns:
            Response from Endee search operation containing ranked results list
            
        Raises:
            EndeeRequestError: For client errors (4xx) that shouldn't be retried
            EndeeConnectionError: For connection/server errors after all retries
            EndeeTimeoutError: For timeout errors after all retries
            ServiceUnavailableError: If circuit breaker is open
        """
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we can't use run_until_complete
                # This typically happens in async contexts like FastAPI
                raise RuntimeError(
                    "Cannot call sync method from async context. Use search_vectors_async() instead."
                )
            else:
                return loop.run_until_complete(
                    self.search_vectors_async(query_vector, top_k, similarity_threshold)
                )
        except RuntimeError:
            # No event loop exists, create a new one
            return asyncio.run(self.search_vectors_async(query_vector, top_k, similarity_threshold))