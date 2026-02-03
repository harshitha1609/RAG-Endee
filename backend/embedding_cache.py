# Embedding model cache
# Singleton pattern for caching the sentence-transformers model across services

import logging
import threading
from typing import List, Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingModelCache:
    """
    Singleton cache for the sentence-transformers embedding model.
    
    This class ensures that the embedding model is loaded only once and shared
    across all services (ingestion, search, RAG) to improve performance and
    reduce memory usage.
    
    Features:
    - Thread-safe singleton implementation
    - Lazy loading of the model
    - Model validation and error handling
    - Consistent embedding generation across services
    """
    
    _instance: Optional['EmbeddingModelCache'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'EmbeddingModelCache':
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the cache (only once)."""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._model: Optional[SentenceTransformer] = None
        self._model_name = 'all-MiniLM-L6-v2'
        self._expected_dimensions = 384
        self._initialized = True
        logger.info("EmbeddingModelCache initialized")
    
    def get_model(self) -> SentenceTransformer:
        """
        Get the cached embedding model, loading it if necessary.
        
        Returns:
            The cached SentenceTransformer model
            
        Raises:
            RuntimeError: If model loading fails
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load_model()
        
        return self._model
    
    def _load_model(self) -> None:
        """
        Load the sentence-transformers model.
        
        This method is called only once when the model is first requested.
        It includes comprehensive error handling and validation.
        
        Raises:
            RuntimeError: If model loading or validation fails
        """
        try:
            logger.info(f"Loading sentence-transformers model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            
            # Validate model by generating a test embedding
            test_embedding = self._model.encode("test", normalize_embeddings=True)
            
            if len(test_embedding) != self._expected_dimensions:
                raise RuntimeError(
                    f"Model validation failed: expected {self._expected_dimensions} "
                    f"dimensions, got {len(test_embedding)}"
                )
            
            logger.info(
                f"Model loaded and validated successfully: "
                f"{self._model_name} ({self._expected_dimensions}d)"
            )
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            self._model = None
            raise RuntimeError("Failed to initialize text processing model. Please contact support if this problem persists.")
    
    def generate_embedding(self, text: str, normalize: bool = True) -> List[float]:
        """
        Generate embedding for the given text using the cached model with memory optimization.
        
        Args:
            text: Text to embed
            normalize: Whether to normalize the embedding (default: True)
            
        Returns:
            384-dimensional embedding vector as list of floats
            
        Raises:
            RuntimeError: If model is not available
            ValueError: If text is invalid
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        
        model = self.get_model()
        
        try:
            # Memory optimization: Use convert_to_tensor=False to avoid GPU memory allocation
            # and return numpy arrays directly instead of tensors
            embedding = model.encode(
                text, 
                normalize_embeddings=normalize,
                convert_to_tensor=False,  # Return numpy array instead of tensor
                show_progress_bar=False,   # Disable progress bar to reduce memory overhead
                batch_size=1,             # Process one text at a time for memory efficiency
                device='cpu'              # Force CPU usage to avoid GPU memory issues
            )
            
            # Validate embedding dimensions
            if len(embedding) != self._expected_dimensions:
                raise RuntimeError(
                    f"Embedding dimension mismatch: expected {self._expected_dimensions}, "
                    f"got {len(embedding)}"
                )
            
            # Convert to list and validate values (memory-efficient conversion)
            embedding_list = embedding.tolist()
            
            # Memory optimization: Clear the numpy array to free memory immediately
            del embedding
            
            # Validate all values are numeric
            if not all(isinstance(x, (int, float)) for x in embedding_list):
                raise RuntimeError("Embedding contains invalid values")
            
            # Check for NaN or infinite values
            import math
            if any(math.isnan(x) or math.isinf(x) for x in embedding_list):
                raise RuntimeError("Embedding contains NaN or infinite values")
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise RuntimeError("Text processing failed. Please check that your input is valid text and try again.")
    
    def is_model_loaded(self) -> bool:
        """
        Check if the model is currently loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._model is not None
    
    def get_model_info(self) -> dict:
        """
        Get information about the cached model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self._model_name,
            "expected_dimensions": self._expected_dimensions,
            "is_loaded": self.is_model_loaded()
        }
    
    def clear_cache(self) -> None:
        """
        Clear the cached model (for testing or memory management).
        
        Note: This will force the model to be reloaded on next access.
        """
        with self._lock:
            if self._model is not None:
                logger.info("Clearing cached embedding model")
                # Memory optimization: Explicitly delete model to free memory
                del self._model
                self._model = None
                
                # Force garbage collection to free memory immediately
                import gc
                gc.collect()
                logger.info("Embedding model cache cleared and memory freed")
    
    def get_memory_usage(self) -> dict:
        """
        Get memory usage information for the cached model.
        
        Returns:
            Dictionary with memory usage statistics
        """
        import sys
        import gc
        
        memory_info = {
            "model_loaded": self.is_model_loaded(),
            "model_size_bytes": 0,
            "cache_references": 0
        }
        
        if self._model is not None:
            try:
                # Estimate model memory usage
                model_size = sys.getsizeof(self._model)
                
                # Add size of model parameters if accessible
                if hasattr(self._model, '_modules'):
                    for module in self._model._modules.values():
                        if hasattr(module, 'parameters'):
                            for param in module.parameters():
                                if hasattr(param, 'data'):
                                    model_size += param.data.nelement() * param.data.element_size()
                
                memory_info["model_size_bytes"] = model_size
                memory_info["model_size_mb"] = round(model_size / (1024 * 1024), 2)
                
                # Count references to the model
                memory_info["cache_references"] = sys.getrefcount(self._model)
                
            except Exception as e:
                logger.debug(f"Error calculating model memory usage: {str(e)}")
                memory_info["error"] = str(e)
        
        return memory_info
    
    def optimize_memory(self) -> dict:
        """
        Perform memory optimization operations.
        
        Returns:
            Dictionary with optimization results
        """
        import gc
        
        optimization_results = {
            "garbage_collected": False,
            "memory_freed_bytes": 0,
            "model_reloaded": False
        }
        
        try:
            # Get initial memory info
            initial_memory = self.get_memory_usage()
            
            # Force garbage collection
            collected = gc.collect()
            optimization_results["garbage_collected"] = True
            optimization_results["objects_collected"] = collected
            
            # If model is loaded but has many references, consider reloading
            if (self._model is not None and 
                initial_memory.get("cache_references", 0) > 10):
                logger.info("High reference count detected, reloading model for memory optimization")
                self.clear_cache()
                # Reload model
                self.get_model()
                optimization_results["model_reloaded"] = True
            
            # Get final memory info
            final_memory = self.get_memory_usage()
            
            # Calculate memory freed (approximate)
            initial_size = initial_memory.get("model_size_bytes", 0)
            final_size = final_memory.get("model_size_bytes", 0)
            optimization_results["memory_freed_bytes"] = max(0, initial_size - final_size)
            
            logger.info(f"Memory optimization completed: {optimization_results}")
            
        except Exception as e:
            logger.error(f"Error during memory optimization: {str(e)}")
            optimization_results["error"] = str(e)
        
        return optimization_results

# Global instance for easy access
embedding_cache = EmbeddingModelCache()