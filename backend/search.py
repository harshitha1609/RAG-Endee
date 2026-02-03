# Vector search logic
# Handles query embedding generation and Endee vector similarity search

import logging
import os
import time
from typing import Any, Dict, List, Optional

from embedding_cache import embedding_cache
from endee_client import EndeeHTTPClient, EndeeConnectionError, EndeeRequestError, EndeeTimeoutError, ServiceUnavailableError

logger = logging.getLogger(__name__)

# Configuration constants
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # Initial delay in seconds
MAX_RETRY_DELAY = 10.0  # Maximum delay in seconds
REQUEST_TIMEOUT = 30  # Request timeout in seconds

class SearchError(Exception):
    """Custom exception for search-related errors."""
    pass

class EmbeddingError(SearchError):
    """Exception for embedding generation failures."""
    pass

class EndeeConnectionError(SearchError):
    """Exception for Endee connection failures."""
    pass

class ValidationError(SearchError):
    """Exception for input validation failures."""
    pass

class VectorSearchService:
    """Service for performing semantic search using Endee vector database."""
    
    def __init__(self) -> None:
        """Initialize the vector search service."""
        self.endee_client = EndeeHTTPClient()
        logger.info("VectorSearchService initialized with cached embedding model and async Endee client")
    
    def _validate_query(self, query: str) -> str:
        """
        Validate and preprocess the search query.
        
        Args:
            query: Raw search query
            
        Returns:
            Validated and preprocessed query
            
        Raises:
            ValidationError: If query is invalid
        """
        # Validate query is not None or empty
        if not query:
            raise ValidationError("Query cannot be None or empty")
        
        if not isinstance(query, str):
            raise ValidationError(f"Query must be a string, got {type(query).__name__}")
        
        # Check if query is just whitespace
        if not query.strip():
            raise ValidationError("Query cannot be empty or contain only whitespace")
        
        # Basic query preprocessing
        processed = query.strip()
        # Remove excessive whitespace
        processed = ' '.join(processed.split())
        
        # Ensure minimum query length
        if len(processed) < 2:
            raise ValidationError("Query too short: minimum 2 characters required after preprocessing")
        
        # Check maximum query length
        if len(processed) > 1000:
            raise ValidationError(f"Query too long: {len(processed)} characters. Maximum allowed: 1000 characters")
        
        return processed
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate 384-dimensional embedding for the search query.
        
        Args:
            query: Preprocessed search query
            
        Returns:
            384-dimensional embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not query or not query.strip():
            raise EmbeddingError("Cannot generate embedding for empty query")
        
        try:
            logger.debug(f"Generating query embedding for: '{query[:50]}{'...' if len(query) > 50 else ''}'")
            
            # Generate embedding using cached model
            embedding_list = embedding_cache.generate_embedding(query, normalize=True)
            
            logger.debug("Query embedding generated successfully using cached model")
            return embedding_list
            
        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise
            logger.error(f"Failed to generate query embedding: {str(e)}")
            raise EmbeddingError("Query processing failed. Please check that your search query is valid text and try again.")
    
    async def _perform_endee_search(
        self,
        query_embedding: List[float],
        top_k: int,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search in Endee with robust error handling.
        
        ENDEE INTEGRATION: SEMANTIC SEARCH
        =================================
        This method implements the core vector search logic that enables semantic
        document retrieval for the RAG system. It communicates directly with the
        Endee vector database to find documents most similar to the query embedding.
        
        ENDEE SEARCH PROCESS:
        --------------------
        1. Validate query embedding dimensions (must be 384d)
        2. Construct search payload with vector and parameters
        3. Make HTTP POST request to Endee /vectors/search endpoint
        4. Parse response and extract search results
        5. Apply client-side similarity threshold filtering
        6. Return filtered results for RAG context combination
        
        ENDEE API COMMUNICATION:
        -----------------------
        - Endpoint: POST {endee_url}/vectors/search
        - Request Headers: Content-Type: application/json
        - Payload: {"vector": List[float], "top_k": int}
        - Response: {"results": [{"id": str, "score": float, "metadata": Dict}]}
        - Timeout: 30 seconds per request (configurable)
        
        ENDEE SIMILARITY SCORING:
        ------------------------
        - Metric: Cosine similarity between query and stored document embeddings
        - Score Range: 0.0 (no similarity) to 1.0 (identical vectors)
        - Ranking: Results sorted by similarity score (highest first)
        - Threshold Filtering: Client-side filtering to remove low-relevance results
        
        ENDEE ERROR HANDLING STRATEGY:
        -----------------------------
        The method implements comprehensive retry logic for Endee failures:
        
        1. Success Case (200): Parse JSON response and filter by threshold
        2. Server Errors (5xx): Retry with exponential backoff (1s, 2s, 4s)
        3. Network Errors: Connection failures, timeouts (retried)
        4. Client Errors (4xx): Not retried (indicates request problems)
        5. Max Retries: 3 attempts total before giving up
        
        ENDEE PERFORMANCE CONSIDERATIONS:
        --------------------------------
        - Vector search is computationally expensive in Endee
        - Higher top_k values increase search time and memory usage
        - Similarity threshold filtering reduces downstream processing
        - Connection reuse minimizes per-request overhead
        - Retry delays prevent overwhelming a struggling Endee service
        
        INTEGRATION WITH RAG PIPELINE:
        -----------------------------
        Search results from this method flow into:
        1. Result formatting and metadata extraction
        2. Context combination for LLM consumption
        3. Source attribution for response traceability
        4. Relevance scoring for result ranking
        
        Args:
            query_embedding: 384-dimensional query vector for similarity search
            top_k: Number of top results to return from Endee
            similarity_threshold: Minimum similarity score for results (0.0 to 1.0)
            
        Returns:
            List of search results from Endee filtered by similarity threshold
            
        Raises:
            EndeeConnectionError: If Endee search fails after all retries
        """
    async def _perform_endee_search(
        self,
        query_embedding: List[float],
        top_k: int,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search in Endee with robust error handling.
        
        ENDEE INTEGRATION: SEMANTIC SEARCH
        =================================
        This method implements the core vector search logic that enables semantic
        document retrieval for the RAG system. It communicates directly with the
        Endee vector database to find documents most similar to the query embedding.
        
        ENDEE SEARCH PROCESS:
        --------------------
        1. Validate query embedding dimensions (must be 384d)
        2. Use async EndeeHTTPClient.search_vectors() for the search
        3. Apply client-side similarity threshold filtering
        4. Return filtered results for RAG context combination
        
        ENDEE API COMMUNICATION:
        -----------------------
        - Uses EndeeHTTPClient.search_vectors() method
        - Automatic retry logic with exponential backoff
        - Connection pooling and timeout handling
        - Comprehensive error handling and logging
        
        ENDEE SIMILARITY SCORING:
        ------------------------
        - Metric: Cosine similarity between query and stored document embeddings
        - Score Range: 0.0 (no similarity) to 1.0 (identical vectors)
        - Ranking: Results sorted by similarity score (highest first)
        - Threshold Filtering: Client-side filtering to remove low-relevance results
        
        ENDEE ERROR HANDLING STRATEGY:
        -----------------------------
        The method leverages EndeeHTTPClient's comprehensive error handling:
        
        1. Success Case (200): Parse JSON response and filter by threshold
        2. Server Errors (5xx): Automatic retry with exponential backoff
        3. Network Errors: Connection failures, timeouts (retried automatically)
        4. Client Errors (4xx): Not retried (indicates request problems)
        5. Max Retries: 3 attempts total before giving up
        
        ENDEE PERFORMANCE CONSIDERATIONS:
        --------------------------------
        - Vector search is computationally expensive in Endee
        - Higher top_k values increase search time and memory usage
        - Similarity threshold filtering reduces downstream processing
        - Async operations improve concurrency and responsiveness
        - Connection reuse minimizes per-request overhead
        
        INTEGRATION WITH RAG PIPELINE:
        -----------------------------
        Search results from this method flow into:
        1. Result formatting and metadata extraction
        2. Context combination for LLM consumption
        3. Source attribution for response traceability
        4. Relevance scoring for result ranking
        
        Args:
            query_embedding: 384-dimensional query vector for similarity search
            top_k: Number of top results to return from Endee
            similarity_threshold: Minimum similarity score for results (0.0 to 1.0)
            
        Returns:
            List of search results from Endee filtered by similarity threshold
            
        Raises:
            EndeeConnectionError: If Endee search fails after all retries
        """
        # CRITICAL: Validate inputs before making expensive network calls
        if not query_embedding:
            raise EndeeConnectionError(
                "Query embedding is required for Endee search"
            )
        
        if len(query_embedding) != 384:
            raise EndeeConnectionError(
                f"Expected 384 dimensions, got {len(query_embedding)}"
            )
        
        if top_k <= 0:
            raise EndeeConnectionError("top_k must be greater than 0")
        
        if not (0.0 <= similarity_threshold <= 1.0):
            raise EndeeConnectionError(
                "similarity_threshold must be between 0.0 and 1.0"
            )
        
        try:
            logger.debug(
                f"Performing async Endee search with top_k={top_k}, "
                f"threshold={similarity_threshold}"
            )
            
            # Use async Endee client for vector search
            result = await self.endee_client.search_vectors_async(
                query_vector=query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            raw_results = result.get('results', [])
            
            # STEP: Filter results by similarity threshold
            # This client-side filtering ensures only relevant results
            # are returned, reducing noise in the RAG pipeline
            filtered_results = []
            for res in raw_results:
                score = float(res.get('score', 0.0))
                if score >= similarity_threshold:
                    filtered_results.append(res)
            
            logger.debug(
                f"Async Endee search successful, found {len(raw_results)} results, "
                f"{len(filtered_results)} above threshold {similarity_threshold}"
            )
            return filtered_results
            
        except (EndeeConnectionError, EndeeRequestError, EndeeTimeoutError, ServiceUnavailableError) as e:
            # Re-raise Endee client errors as-is
            logger.error(f"Endee client error during search: {str(e)}")
            raise EndeeConnectionError("Search service is temporarily unavailable. Please try again in a few moments.")
            
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error during Endee search: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise EndeeConnectionError("Search operation failed due to an unexpected error. Please try again or contact support.")

    async def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Perform semantic search for the given query with memory optimization.
        
        Args:
            query: Natural language search query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score for results (0.0 to 1.0)
            
        Returns:
            List of search results with scores and metadata filtered by similarity threshold
            
        Raises:
            ValidationError: If input validation fails
            EmbeddingError: If query embedding generation fails
            EndeeConnectionError: If Endee search fails
            SearchError: For other search failures
        """
        try:
            query_preview = (
                f"'{query[:50]}{'...' if len(query) > 50 else ''}'"
            )
            logger.info(f"Performing semantic search for query: {query_preview}")
            
            # Validate input parameters
            if top_k <= 0:
                raise ValidationError("top_k must be greater than 0")
            if top_k > 100:
                raise ValidationError("top_k cannot exceed 100")
            if not (0.0 <= similarity_threshold <= 1.0):
                raise ValidationError(
                    "similarity_threshold must be between 0.0 and 1.0"
                )
            
            # Validate and preprocess query
            processed_query = self._validate_query(query)
            logger.debug(f"Processed query: '{processed_query}'")
            
            # Generate query embedding
            query_embedding = self._generate_query_embedding(processed_query)
            logger.debug("Query embedding generated successfully")
            
            # Perform vector similarity search in Endee
            endee_results = await self._perform_endee_search(
                query_embedding, top_k, similarity_threshold
            )
            logger.debug(
                f"Endee search completed, processing {len(endee_results)} results "
                f"above threshold {similarity_threshold}"
            )
            
            # Memory optimization: Clear query embedding after use
            del query_embedding
            
            # Process and format results with memory-efficient approach
            formatted_results = []
            for i, result in enumerate(endee_results):
                try:
                    # Memory-efficient result processing
                    formatted_result = {
                        "document_id": result.get("id", "unknown"),
                        "content": result.get("metadata", {}).get("content", ""),
                        "score": float(result.get("score", 0.0)),
                        "metadata": result.get("metadata", {})
                    }
                    formatted_results.append(formatted_result)
                    
                    # Memory optimization: Clear processed result to free memory
                    del result
                    
                except Exception as e:
                    logger.warning(f"Error processing search result {i}: {str(e)}")
                    continue
            
            # Memory optimization: Clear intermediate results
            del endee_results
            
            # Force garbage collection if we processed many results
            if len(formatted_results) > 20:
                import gc
                gc.collect()
            
            logger.info(
                f"Search completed successfully, returning {len(formatted_results)} results"
            )
            return formatted_results
            
        except ValidationError as e:
            logger.error(f"Validation error during search: {str(e)}")
            raise ValidationError(str(e))  # Keep original validation messages as they're user-facing
            
        except EmbeddingError as e:
            logger.error(f"Embedding error during search: {str(e)}")
            raise EmbeddingError(str(e))  # Keep original embedding error messages
            
        except EndeeConnectionError as e:
            logger.error(f"Endee connection error during search: {str(e)}")
            raise EndeeConnectionError(str(e))  # Keep original connection error messages
            
        except Exception as e:
            error_msg = f"Unexpected error during search: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise SearchError("Search operation failed due to an unexpected error. Please try again or contact support if the problem persists.")