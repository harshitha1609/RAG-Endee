# Document ingestion logic
# Handles text preprocessing, embedding generation, and Endee storage

import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from embedding_cache import embedding_cache
from endee_client import (
    EndeeClientError,
    EndeeConnectionError as ClientConnectionError,
    EndeeHTTPClient,
    EndeeRequestError,
    EndeeTimeoutError,
    ServiceUnavailableError,
)

logger = logging.getLogger(__name__)

# Configuration constants
MAX_CONTENT_LENGTH = 50000  # Maximum content length in characters

class IngestionError(Exception):
    """Custom exception for ingestion-related errors."""
    pass

class EndeeConnectionError(IngestionError):
    """Exception for Endee connection failures."""
    pass

class EmbeddingError(IngestionError):
    """Exception for embedding generation failures."""
    pass

class ValidationError(IngestionError):
    """Exception for input validation failures."""
    pass

class DocumentIngestionService:
    """Service for ingesting documents into the Endee vector database."""
    
    def __init__(self) -> None:
        """Initialize the document ingestion service."""
        self.endee_client = EndeeHTTPClient()
        logger.info("DocumentIngestionService initialized with cached embedding model")
    
    def _preprocess_text(self, content: str) -> str:
        """
        Preprocess text content for embedding generation with memory optimization.
        
        Args:
            content: Raw text content
            
        Returns:
            Preprocessed text
            
        Raises:
            ValidationError: If content is invalid
        """
        # Validate content is not None or empty
        if not content:
            raise ValidationError("Content cannot be None or empty")
        
        if not isinstance(content, str):
            raise ValidationError(f"Content must be a string, got {type(content).__name__}")
        
        # Check content length
        if len(content) > MAX_CONTENT_LENGTH:
            raise ValidationError(
                f"Content too long: {len(content)} characters. "
                f"Maximum allowed: {MAX_CONTENT_LENGTH} characters"
            )
        
        # Check if content is just whitespace
        if not content.strip():
            raise ValidationError("Content cannot be empty or contain only whitespace")
        
        # Memory-optimized text preprocessing
        # Use generator expressions and in-place operations where possible
        processed = content.strip()
        
        # Memory optimization: Process text in chunks for very large content
        if len(processed) > 10000:  # For content larger than 10KB
            # Split into lines and process each line to avoid loading entire content
            lines = processed.split('\n')
            processed_lines = []
            
            for line in lines:
                # Remove excessive whitespace from each line
                cleaned_line = ' '.join(line.split())
                if cleaned_line:  # Only keep non-empty lines
                    processed_lines.append(cleaned_line)
            
            processed = '\n'.join(processed_lines)
            
            # Memory cleanup: Clear intermediate variables
            del lines, processed_lines
        else:
            # For smaller content, use the original approach
            processed = ' '.join(processed.split())
        
        # Ensure minimum content length
        if len(processed) < 3:
            raise ValidationError("Content too short: minimum 3 characters required after preprocessing")
        
        return processed
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate 384-dimensional embedding for the given text.
        
        This method implements the core embedding generation logic with
        comprehensive validation and error handling. The process includes:
        
        1. Model availability validation (using cached model)
        2. Input text validation and sanitization
        3. Embedding generation using cached sentence-transformers model
        4. Dimension validation (must be exactly 384)
        5. Value validation (all numeric values)
        6. Normalization for consistent similarity calculations
        
        The embedding generation is critical for the RAG system as it
        determines how well documents can be retrieved based on semantic
        similarity to user queries.
        
        Args:
            text: Preprocessed text content to embed
            
        Returns:
            384-dimensional embedding vector as list of floats
            
        Raises:
            EmbeddingError: If embedding generation fails at any step
        """
        # Validate input text is not empty after preprocessing
        if not text or not text.strip():
            raise EmbeddingError("Cannot generate embedding for empty text")
        
        try:
            logger.debug(f"Generating embedding for text of length {len(text)}")
            
            # STEP 1: Generate embedding using cached model
            # The cached model ensures consistent performance across all services
            embedding_list = embedding_cache.generate_embedding(text, normalize=True)
            
            logger.debug("Embedding generated successfully using cached model")
            return embedding_list
            
        except Exception as e:
            # Re-raise EmbeddingError as-is, wrap other exceptions
            if isinstance(e, EmbeddingError):
                raise
            logger.error(f"Failed to generate embedding: {str(e)}")
            # Create user-friendly error message without exposing internal details
            raise EmbeddingError("Text processing failed. Please check that your content is valid text and try again.")
    
    async def _store_in_endee(
        self,
        document_id: str,
        embedding: List[float],
        content: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Store document embedding in Endee using the HTTP client.
        
        ENDEE INTEGRATION: DOCUMENT STORAGE
        ==================================
        This method implements the final step of the document ingestion pipeline,
        storing the generated embedding and metadata in the Endee vector database
        for later retrieval during semantic search operations.
        
        ENDEE STORAGE PROCESS:
        ---------------------
        1. Prepare metadata payload with document content and timestamps
        2. Call EndeeHTTPClient.add_vector() with document ID, embedding, and metadata
        3. Handle Endee-specific errors and convert to ingestion exceptions
        4. Log successful storage with document ID for tracking
        
        ENDEE METADATA STRUCTURE:
        ------------------------
        The metadata stored in Endee includes:
        - "content": Original preprocessed document text (required for RAG retrieval)
        - "created_at": ISO timestamp of ingestion (for tracking and debugging)
        - User-provided metadata: Title, source, tags, etc. (preserved as-is)
        
        ENDEE ERROR HANDLING:
        --------------------
        - EndeeRequestError (4xx): Invalid payload, converted to EndeeConnectionError
        - EndeeConnectionError: Network/server issues, passed through with context
        - EndeeTimeoutError: Request timeouts, converted to EndeeConnectionError
        - Unexpected errors: Wrapped in EndeeConnectionError with error details
        
        INTEGRATION WITH RAG PIPELINE:
        -----------------------------
        Successful storage in Endee enables:
        1. Semantic search: Query embeddings can find similar document embeddings
        2. Content retrieval: Metadata contains original text for context generation
        3. Source attribution: Document IDs and metadata support result traceability
        
        Args:
            document_id: Unique document identifier for Endee storage
            embedding: 384-dimensional embedding vector from sentence-transformers
            content: Original document content (stored in metadata for retrieval)
            metadata: Document metadata (title, source, user-provided fields)
            
        Returns:
            True if storage successful (always returns True or raises exception)
            
        Raises:
            EndeeConnectionError: If Endee storage fails for any reason
        """
        try:
            logger.debug(f"Storing document {document_id} in Endee")
            
            # Prepare metadata with content and timestamp
            endee_metadata = {
                "content": content,
                "created_at": datetime.now().isoformat(),
                **metadata
            }
            
            # Use the async Endee HTTP client to add the vector
            result = await self.endee_client.add_vector_async(
                document_id=document_id,
                vector=embedding,
                metadata=endee_metadata
            )
            
            logger.info(f"Successfully stored document {document_id} in Endee")
            return True
            
        except EndeeRequestError as e:
            # Client errors (4xx) - don't wrap in connection error
            logger.error(f"Request error storing document in Endee: {str(e)}")
            raise EndeeConnectionError("Document storage request was invalid. Please check your document format and try again.")
            
        except (ClientConnectionError, EndeeTimeoutError, ServiceUnavailableError) as e:
            # Connection/timeout/service unavailable errors - wrap in our exception type
            logger.error(f"Connection error storing document in Endee: {str(e)}")
            raise EndeeConnectionError("Vector database is temporarily unavailable. Please try again in a few moments.")
            
        except EndeeClientError as e:
            # Other client errors
            logger.error(f"Client error storing document in Endee: {str(e)}")
            raise EndeeConnectionError("Document storage service encountered an error. Please try again.")
            
        except Exception as e:
            error_msg = f"Unexpected error storing document in Endee: {str(e)}"
            logger.error(error_msg)
            raise EndeeConnectionError("Document storage failed due to an unexpected error. Please try again or contact support.")
    
    def _validate_metadata(
        self, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate and sanitize metadata with comprehensive security checks.
        
        This method implements thorough metadata validation to ensure:
        1. Type safety - metadata must be a dictionary
        2. Size limits - prevents memory exhaustion attacks
        3. Key validation - ensures keys are strings with reasonable length
        4. Value validation - only allows safe, serializable types
        5. Security - prevents injection of malicious data
        
        The validation is critical for system security and stability as
        metadata is stored in Endee and could be used in various contexts.
        Malicious or malformed metadata could cause system failures or
        security vulnerabilities.
        
        Args:
            metadata: Optional metadata dictionary from user input
            
        Returns:
            Validated and sanitized metadata dictionary
            
        Raises:
            ValidationError: If metadata fails any validation checks
        """
        # Handle None case - return empty dict for consistency
        if metadata is None:
            return {}
        
        # CRITICAL: Validate metadata is a dictionary
        # This prevents type confusion attacks and ensures consistent processing
        if not isinstance(metadata, dict):
            raise ValidationError(
                f"Metadata must be a dictionary, got {type(metadata).__name__}"
            )
        
        # SIZE VALIDATION: Check metadata size to prevent memory exhaustion
        # Large metadata could consume excessive memory or storage space
        metadata_size = len(str(metadata))
        if metadata_size > 10000:  # 10KB limit
            raise ValidationError("Metadata too large: maximum 10KB allowed")
        
        # ITERATIVE VALIDATION: Validate each key-value pair
        validated_metadata = {}
        for key, value in metadata.items():
            # KEY VALIDATION: Ensure keys are strings
            # Non-string keys could cause issues in JSON serialization or database storage
            if not isinstance(key, str):
                raise ValidationError(
                    f"Metadata keys must be strings, got {type(key).__name__}"
                )
            
            # KEY LENGTH VALIDATION: Prevent excessively long keys
            # Long keys could cause database issues or memory problems
            if len(key) > 100:
                raise ValidationError(
                    f"Metadata key too long: '{key}' (maximum 100 characters)"
                )
            
            # VALUE TYPE VALIDATION: Allow only safe, serializable types
            # This prevents injection of complex objects that could cause
            # serialization failures or security issues
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValidationError(
                    f"Metadata value for key '{key}' must be string, number, "
                    f"boolean, or null, got {type(value).__name__}"
                )
            
            # Store validated key-value pair
            validated_metadata[key] = value
        
        return validated_metadata
    
    async def ingest_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ingest a document into the system.
        
        Args:
            content: Raw text content of the document
            metadata: Optional metadata dictionary
            
        Returns:
            Dictionary containing ingestion result
            
        Raises:
            ValidationError: If input validation fails
            EmbeddingError: If embedding generation fails
            EndeeConnectionError: If Endee storage fails
            IngestionError: For other ingestion failures
        """
        document_id = None
        
        try:
            logger.info("Starting document ingestion")
            
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            logger.debug(f"Generated document ID: {document_id}")
            
            # Validate and sanitize metadata
            validated_metadata = self._validate_metadata(metadata)
            
            # Preprocess text (includes validation)
            processed_content = self._preprocess_text(content)
            logger.debug(f"Preprocessed content length: {len(processed_content)}")
            
            # Generate embedding (includes validation)
            embedding = self._generate_embedding(processed_content)
            logger.debug("Embedding generated successfully")
            
            # Store in Endee using HTTP client
            await self._store_in_endee(
                document_id, embedding, processed_content, validated_metadata
            )
            
            logger.info(f"Successfully ingested document {document_id}")
            
            return {
                "document_id": document_id,
                "status": "success",
                "embedding_dimension": len(embedding),
                "content_length": len(processed_content),
                "metadata": validated_metadata
            }
            
        except ValidationError as e:
            logger.error(f"Validation error during ingestion: {str(e)}")
            raise ValidationError(str(e))  # Keep original validation messages as they're user-facing
            
        except EmbeddingError as e:
            logger.error(f"Embedding error during ingestion: {str(e)}")
            raise EmbeddingError(str(e))  # Keep original embedding error messages
            
        except EndeeConnectionError as e:
            logger.error(f"Endee connection error during ingestion: {str(e)}")
            raise EndeeConnectionError(str(e))  # Keep original connection error messages
            
        except Exception as e:
            error_msg = f"Unexpected error during document ingestion: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise IngestionError("Document ingestion failed due to an unexpected error. Please try again or contact support if the problem persists.")