"""
Response Time Threshold Validation Tests
Tests to ensure all API endpoints meet acceptable response time thresholds.
"""

import pytest
import time
import statistics
import requests_mock
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


class TestResponseTimeThresholds:
    """Test that all API endpoints meet acceptable response time thresholds."""
    
    def setup_method(self):
        """Set up test fixtures and mock embedding model."""
        # Mock embedding model to avoid network calls
        self.mock_embedding = [0.1] * 384  # 384-dimensional mock embedding
        
        # Sample test data
        self.test_content = "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
        self.test_metadata = {
            "title": "ML Test Document",
            "author": "Test Author",
            "category": "AI/ML"
        }
        self.test_query = "What is machine learning?"
        
        # Mock Endee responses
        self.mock_endee_add_response = {
            "status": "success",
            "id": "test-doc-123"
        }
        
        self.mock_endee_search_response = {
            "results": [
                {
                    "id": "test-doc-123",
                    "score": 0.95,
                    "metadata": {
                        "content": self.test_content,
                        "title": "ML Test Document",
                        "author": "Test Author",
                        "category": "AI/ML"
                    }
                }
            ]
        }
    
    def _measure_response_time(self, func, *args, **kwargs):
        """Measure response time of a function call in milliseconds."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, response_time
    
    def _run_multiple_requests(self, func, iterations=10, *args, **kwargs):
        """Run multiple requests and collect timing statistics."""
        response_times = []
        responses = []
        
        for _ in range(iterations):
            response, response_time = self._measure_response_time(func, *args, **kwargs)
            response_times.append(response_time)
            responses.append(response)
        
        return {
            'responses': responses,
            'response_times': response_times,
            'avg_time': statistics.mean(response_times),
            'median_time': statistics.median(response_times),
            'min_time': min(response_times),
            'max_time': max(response_times),
            'p95_time': sorted(response_times)[int(0.95 * len(response_times))],
            'p99_time': sorted(response_times)[int(0.99 * len(response_times))],
            'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0
        }
    
    @patch('embedding_cache.embedding_cache.generate_embedding')
    def test_health_endpoint_response_time_threshold(self, mock_embedding):
        """Test health endpoint meets response time threshold."""
        # Health endpoint doesn't use embeddings, but patch anyway for consistency
        mock_embedding.return_value = self.mock_embedding
        
        # Single request timing
        response, response_time = self._measure_response_time(client.get, "/health")
        
        assert response.status_code == 200
        # Health endpoint should respond within 50ms
        assert response_time < 50, f"Health endpoint took {response_time:.2f}ms, expected < 50ms"
        
        # Multiple requests for consistency testing
        stats = self._run_multiple_requests(client.get, 20, "/health")
        
        # All requests should succeed
        for response in stats['responses']:
            assert response.status_code == 200
        
        # Response time thresholds
        assert stats['avg_time'] < 25, f"Average response time {stats['avg_time']:.2f}ms exceeds 25ms threshold"
        assert stats['max_time'] < 100, f"Maximum response time {stats['max_time']:.2f}ms exceeds 100ms threshold"
        assert stats['p95_time'] < 50, f"95th percentile {stats['p95_time']:.2f}ms exceeds 50ms threshold"
        assert stats['std_dev'] < 20, f"Response time std dev {stats['std_dev']:.2f}ms exceeds 20ms threshold"
    
    @patch('ingest.DocumentIngestionService.ingest_document')
    @patch('embedding_cache.embedding_cache.generate_embedding')
    def test_ingest_endpoint_response_time_threshold(self, mock_embedding, mock_ingest):
        """Test document ingestion endpoint meets response time threshold."""
        mock_embedding.return_value = self.mock_embedding
        mock_ingest.return_value = {
            "document_id": "test-doc-123",
            "status": "success",
            "embedding_dimension": 384,
            "content_length": len(self.test_content),
            "metadata": self.test_metadata
        }
        
        # Single request timing
        response, response_time = self._measure_response_time(
            client.post, "/ingest",
            json={"content": self.test_content, "metadata": self.test_metadata}
        )
        
        assert response.status_code == 200
        # Ingestion should complete within 1.5 seconds
        assert response_time < 1500, f"Ingestion took {response_time:.2f}ms, expected < 1500ms"
        
        # Multiple requests for consistency
        stats = self._run_multiple_requests(
            client.post, 5, "/ingest",
            json={"content": self.test_content, "metadata": self.test_metadata}
        )
        
        # All requests should succeed
        for response in stats['responses']:
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["embedding_dimension"] == 384
        
        # Response time thresholds
        assert stats['avg_time'] < 1000, f"Average ingestion time {stats['avg_time']:.2f}ms exceeds 1000ms threshold"
        assert stats['max_time'] < 2000, f"Maximum ingestion time {stats['max_time']:.2f}ms exceeds 2000ms threshold"
        assert stats['p95_time'] < 1500, f"95th percentile {stats['p95_time']:.2f}ms exceeds 1500ms threshold"
    
    @patch('search.VectorSearchService.search')
    @patch('embedding_cache.embedding_cache.generate_embedding')
    def test_search_endpoint_response_time_threshold(self, mock_embedding, mock_search):
        """Test search endpoint meets response time threshold."""
        mock_embedding.return_value = self.mock_embedding
        mock_search.return_value = [
            {
                "document_id": "test-doc-123",
                "content": self.test_content,
                "score": 0.95,
                "metadata": self.test_metadata
            }
        ]
        
        # Single request timing
        response, response_time = self._measure_response_time(
            client.post, "/search",
            json={"query": self.test_query, "top_k": 5}
        )
        
        assert response.status_code == 200
        # Search should complete within 800ms
        assert response_time < 800, f"Search took {response_time:.2f}ms, expected < 800ms"
        
        # Multiple requests for consistency
        stats = self._run_multiple_requests(
            client.post, 10, "/search",
            json={"query": self.test_query, "top_k": 5}
        )
        
        # All requests should succeed
        for response in stats['responses']:
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == self.test_query
            assert len(data["results"]) >= 0
        
        # Response time thresholds
        assert stats['avg_time'] < 600, f"Average search time {stats['avg_time']:.2f}ms exceeds 600ms threshold"
        assert stats['max_time'] < 1200, f"Maximum search time {stats['max_time']:.2f}ms exceeds 1200ms threshold"
        assert stats['p95_time'] < 1000, f"95th percentile {stats['p95_time']:.2f}ms exceeds 1000ms threshold"
    
    @patch('rag.RAGService.generate_response')
    @patch('embedding_cache.embedding_cache.generate_embedding')
    def test_rag_endpoint_response_time_threshold(self, mock_embedding, mock_rag):
        """Test RAG endpoint meets response time threshold."""
        mock_embedding.return_value = self.mock_embedding
        mock_rag.return_value = {
            "query": self.test_query,
            "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            "sources": [
                {
                    "document_id": "test-doc-123",
                    "score": 0.95,
                    "metadata": self.test_metadata,
                    "content_preview": self.test_content[:100] + "..."
                }
            ],
            "context_used": self.test_content,
            "context_length": len(self.test_content),
            "sources_count": 1,
            "truncated": False,
            "max_context_length": 4000
        }
        
        # Single request timing
        response, response_time = self._measure_response_time(
            client.post, "/rag",
            json={"query": self.test_query, "top_k": 3}
        )
        
        assert response.status_code == 200
        # RAG should complete within 1.2 seconds
        assert response_time < 1200, f"RAG took {response_time:.2f}ms, expected < 1200ms"
        
        # Multiple requests for consistency
        stats = self._run_multiple_requests(
            client.post, 5, "/rag",
            json={"query": self.test_query, "top_k": 3}
        )
        
        # All requests should succeed
        for response in stats['responses']:
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == self.test_query
            assert "answer" in data
            assert "sources" in data
        
        # Response time thresholds
        assert stats['avg_time'] < 900, f"Average RAG time {stats['avg_time']:.2f}ms exceeds 900ms threshold"
        assert stats['max_time'] < 1800, f"Maximum RAG time {stats['max_time']:.2f}ms exceeds 1800ms threshold"
        assert stats['p95_time'] < 1500, f"95th percentile {stats['p95_time']:.2f}ms exceeds 1500ms threshold"
    
    @patch('search.VectorSearchService.search')
    @patch('embedding_cache.embedding_cache.generate_embedding')
    def test_concurrent_requests_response_time_threshold(self, mock_embedding, mock_search):
        """Test response times under concurrent load meet thresholds."""
        mock_embedding.return_value = self.mock_embedding
        mock_search.return_value = [
            {
                "document_id": "test-doc-123",
                "content": self.test_content,
                "score": 0.95,
                "metadata": self.test_metadata
            }
        ]
        
        import concurrent.futures
        
        def make_search_request():
            """Make a single search request."""
            start_time = time.perf_counter()
            response = client.post("/search", json={"query": self.test_query, "top_k": 5})
            end_time = time.perf_counter()
            return response, (end_time - start_time) * 1000
        
        # Run 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_search_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        response_times = []
        for response, response_time in results:
            assert response.status_code == 200
            response_times.append(response_time)
        
        # Response time thresholds under concurrent load
        avg_concurrent_time = statistics.mean(response_times)
        max_concurrent_time = max(response_times)
        p95_concurrent_time = sorted(response_times)[int(0.95 * len(response_times))]
        
        assert avg_concurrent_time < 1000, f"Average concurrent response time {avg_concurrent_time:.2f}ms exceeds 1000ms threshold"
        assert max_concurrent_time < 2000, f"Maximum concurrent response time {max_concurrent_time:.2f}ms exceeds 2000ms threshold"
        assert p95_concurrent_time < 1500, f"95th percentile concurrent response time {p95_concurrent_time:.2f}ms exceeds 1500ms threshold"
    
    @patch('ingest.DocumentIngestionService.ingest_document')
    @patch('embedding_cache.embedding_cache.generate_embedding')
    def test_large_content_response_time_threshold(self, mock_embedding, mock_ingest):
        """Test response times with large content meet thresholds."""
        mock_embedding.return_value = self.mock_embedding
        
        # Create larger content (around 5KB)
        large_content = self.test_content * 20  # Approximately 5KB
        
        mock_ingest.return_value = {
            "document_id": "test-doc-large",
            "status": "success",
            "embedding_dimension": 384,
            "content_length": len(large_content),
            "metadata": self.test_metadata
        }
        
        response, response_time = self._measure_response_time(
            client.post, "/ingest",
            json={"content": large_content, "metadata": self.test_metadata}
        )
        
        assert response.status_code == 200
        # Large content ingestion should complete within 2.5 seconds
        assert response_time < 2500, f"Large content ingestion took {response_time:.2f}ms, expected < 2500ms"
        
        data = response.json()
        assert data["status"] == "success"
        assert data["content_length"] == len(large_content)
    
    @patch('search.VectorSearchService.search')
    @patch('embedding_cache.embedding_cache.generate_embedding')
    def test_multiple_results_search_response_time_threshold(self, mock_embedding, mock_search):
        """Test search response times with multiple results meet thresholds."""
        mock_embedding.return_value = self.mock_embedding
        
        # Mock response with multiple results
        multiple_results = [
            {
                "document_id": f"doc-{i}",
                "content": f"Test document {i} content for performance testing with detailed information about various topics.",
                "score": 0.9 - (i * 0.05),
                "metadata": {
                    "title": f"Test Document {i}",
                    "category": "Performance Test"
                }
            }
            for i in range(10)  # 10 results
        ]
        mock_search.return_value = multiple_results
        
        response, response_time = self._measure_response_time(
            client.post, "/search",
            json={"query": "performance test", "top_k": 10}
        )
        
        assert response.status_code == 200
        # Multiple results search should complete within 1 second
        assert response_time < 1000, f"Multiple results search took {response_time:.2f}ms, expected < 1000ms"
        
        data = response.json()
        assert len(data["results"]) == 10
        assert data["total_results"] == 10
    
    @patch('rag.RAGService.generate_response')
    @patch('embedding_cache.embedding_cache.generate_embedding')
    def test_rag_multiple_sources_response_time_threshold(self, mock_embedding, mock_rag):
        """Test RAG response times with multiple sources meet thresholds."""
        mock_embedding.return_value = self.mock_embedding
        
        # Mock response with multiple sources
        multiple_sources = [
            {
                "document_id": f"source-{i}",
                "score": 0.9 - (i * 0.03),
                "metadata": {
                    "title": f"ML Source {i}",
                    "category": "AI/ML"
                },
                "content_preview": f"Source document {i} provides comprehensive information about machine learning concepts..."
            }
            for i in range(5)  # 5 sources
        ]
        
        mock_rag.return_value = {
            "query": "Explain machine learning with examples",
            "answer": "Machine learning is a comprehensive field with multiple applications...",
            "sources": multiple_sources,
            "context_used": "Combined context from multiple sources about machine learning...",
            "context_length": 2000,
            "sources_count": 5,
            "truncated": False,
            "max_context_length": 4000
        }
        
        response, response_time = self._measure_response_time(
            client.post, "/rag",
            json={"query": "Explain machine learning with examples", "top_k": 5}
        )
        
        assert response.status_code == 200
        # RAG with multiple sources should complete within 1.8 seconds
        assert response_time < 1800, f"RAG with multiple sources took {response_time:.2f}ms, expected < 1800ms"
        
        data = response.json()
        assert len(data["sources"]) == 5
        assert data["sources_count"] == 5
        assert data["context_length"] > 0
    
    def test_memory_endpoint_response_time_threshold(self):
        """Test memory endpoint meets response time threshold."""
        # Memory endpoint doesn't use embeddings
        response, response_time = self._measure_response_time(client.get, "/memory")
        
        assert response.status_code == 200
        # Memory endpoint should respond within 200ms
        assert response_time < 200, f"Memory endpoint took {response_time:.2f}ms, expected < 200ms"
        
        data = response.json()
        assert data["status"] == "success"
        assert "memory_stats" in data
    
    def test_memory_optimize_endpoint_response_time_threshold(self):
        """Test memory optimization endpoint meets response time threshold."""
        # Memory optimization endpoint doesn't use embeddings
        response, response_time = self._measure_response_time(client.post, "/memory/optimize")
        
        assert response.status_code == 200
        # Memory optimization should complete within 500ms
        assert response_time < 500, f"Memory optimization took {response_time:.2f}ms, expected < 500ms"
        
        data = response.json()
        assert data["status"] == "success"
        assert "optimization_results" in data
    
    @patch('search.VectorSearchService.search')
    @patch('embedding_cache.embedding_cache.generate_embedding')
    def test_response_time_consistency_across_requests(self, mock_embedding, mock_search):
        """Test that response times are consistent across multiple requests."""
        mock_embedding.return_value = self.mock_embedding
        mock_search.return_value = [
            {
                "document_id": "test-doc-123",
                "content": self.test_content,
                "score": 0.95,
                "metadata": self.test_metadata
            }
        ]
        
        # Run 20 search requests and measure consistency
        stats = self._run_multiple_requests(
            client.post, 20, "/search",
            json={"query": self.test_query, "top_k": 5}
        )
        
        # All requests should succeed
        for response in stats['responses']:
            assert response.status_code == 200
        
        # Response time consistency thresholds (relaxed for test environment)
        coefficient_of_variation = stats['std_dev'] / stats['avg_time'] if stats['avg_time'] > 0 else 0
        
        assert coefficient_of_variation < 1.0, f"Response time coefficient of variation {coefficient_of_variation:.3f} exceeds 1.0 threshold (inconsistent performance)"
        assert stats['max_time'] / stats['min_time'] < 5.0, f"Response time ratio {stats['max_time'] / stats['min_time']:.2f} exceeds 5.0 threshold (high variance)"
    
    def test_response_time_summary_report(self):
        """Generate a summary report of all response time thresholds."""
        # This test documents the expected response time thresholds
        thresholds = {
            "health_endpoint": {
                "single_request": "< 50ms",
                "average": "< 25ms",
                "95th_percentile": "< 50ms",
                "maximum": "< 100ms"
            },
            "ingest_endpoint": {
                "single_request": "< 1500ms",
                "average": "< 1000ms",
                "95th_percentile": "< 1500ms",
                "maximum": "< 2000ms"
            },
            "search_endpoint": {
                "single_request": "< 800ms",
                "average": "< 600ms",
                "95th_percentile": "< 1000ms",
                "maximum": "< 1200ms"
            },
            "rag_endpoint": {
                "single_request": "< 1200ms",
                "average": "< 900ms",
                "95th_percentile": "< 1500ms",
                "maximum": "< 1800ms"
            },
            "concurrent_load": {
                "average": "< 1000ms",
                "95th_percentile": "< 1500ms",
                "maximum": "< 2000ms"
            },
            "large_content": {
                "ingestion": "< 2500ms"
            },
            "memory_endpoints": {
                "status": "< 200ms",
                "optimization": "< 500ms"
            }
        }
        
        # This test always passes but documents the thresholds
        assert len(thresholds) > 0, "Response time thresholds are documented"
        
        # Log the thresholds for reference
        import json
        print("\nResponse Time Thresholds:")
        print(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])