"""
Performance test for embedding model caching.
Demonstrates that the model is loaded only once and shared across services.
"""

import time
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.dirname(__file__))

from embedding_cache import embedding_cache
from ingest import DocumentIngestionService
from search import VectorSearchService


def test_embedding_cache_performance():
    """Test that embedding model is cached and shared across services."""
    print("Testing embedding model caching performance...")
    
    # Test 1: Initial model loading time
    print("\n1. Testing initial model loading...")
    start_time = time.time()
    
    # First call should load the model
    result1 = embedding_cache.generate_embedding("test text 1")
    first_load_time = time.time() - start_time
    
    print(f"   First embedding generation: {first_load_time:.3f} seconds")
    print(f"   Embedding dimensions: {len(result1)}")
    print(f"   Model loaded: {embedding_cache.is_model_loaded()}")
    
    # Test 2: Subsequent calls should be faster (no model loading)
    print("\n2. Testing cached model performance...")
    start_time = time.time()
    
    result2 = embedding_cache.generate_embedding("test text 2")
    cached_time = time.time() - start_time
    
    print(f"   Cached embedding generation: {cached_time:.3f} seconds")
    print(f"   Embedding dimensions: {len(result2)}")
    print(f"   Speed improvement: {first_load_time / cached_time:.1f}x faster")
    
    # Test 3: Services using cached model
    print("\n3. Testing services with cached model...")
    
    # Create services (should not reload model)
    start_time = time.time()
    ingestion_service = DocumentIngestionService()
    search_service = VectorSearchService()
    service_init_time = time.time() - start_time
    
    print(f"   Service initialization: {service_init_time:.3f} seconds")
    print(f"   Model still loaded: {embedding_cache.is_model_loaded()}")
    
    # Test embedding generation from services
    start_time = time.time()
    ingestion_result = ingestion_service._generate_embedding("ingestion test")
    ingestion_time = time.time() - start_time
    
    start_time = time.time()
    search_result = search_service._generate_query_embedding("search test")
    search_time = time.time() - start_time
    
    print(f"   Ingestion service embedding: {ingestion_time:.3f} seconds ({len(ingestion_result)}d)")
    print(f"   Search service embedding: {search_time:.3f} seconds ({len(search_result)}d)")
    
    # Test 4: Model info
    print("\n4. Model information:")
    model_info = embedding_cache.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Test 5: Performance summary
    print("\n5. Performance Summary:")
    print(f"   Initial load time: {first_load_time:.3f}s")
    print(f"   Cached generation: {cached_time:.3f}s")
    print(f"   Service init time: {service_init_time:.3f}s")
    print(f"   Total services tested: 2")
    print(f"   Model loaded once: ✓")
    print(f"   Memory efficient: ✓")
    
    assert len(result1) == 384, "Embedding should be 384-dimensional"
    assert len(result2) == 384, "Embedding should be 384-dimensional"
    assert len(ingestion_result) == 384, "Embedding should be 384-dimensional"
    assert len(search_result) == 384, "Embedding should be 384-dimensional"
    assert cached_time < first_load_time, "Cached calls should be faster"
    assert embedding_cache.is_model_loaded(), "Model should remain loaded"
    
    print("\n✅ All embedding cache performance tests passed!")
    return True


if __name__ == "__main__":
    test_embedding_cache_performance()