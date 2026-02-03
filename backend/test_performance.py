"""
Performance tests for the Endee RAG System API.
Tests response times and ensures basic performance requirements are met.
"""

import pytest
import time
import statistics
import requests_mock
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


class TestAPIPerformance:
    """Test API performance and response times."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Sample test data for performance tests
        self.test_content = "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions."
        
        self.test_metadata = {
            "title": "ML Performance Test",
            "author": "Test Author",
            "category": "AI/ML"
        }
        
        self.test_query = "What is machine learning?"
        
        # Mock Endee responses for consistent testing
        self.mock_endee_add_response = {
            "status": "success",
            "id": "perf-test-doc"
        }
        
        self.mock_endee_search_response = {
            "results": [
                {
                    "id": "perf-test-doc",
                    "score": 0.95,
                    "metadata": {
                        "content": self.test_content,
                        "title": "ML Performance Test",
                        "author": "Test Author",
                        "category": "AI/ML"
                    }
                }
            ]
        }
    
    def _measure_response_time(self, func, *args, **kwargs):
        """Measure response time of a function call."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
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
            'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0
        }
    
    def test_health_endpoint_performance(self):
        """Test health endpoint response time."""
        # Single request timing
        response, response_time = self._measure_response_time(
            client.get, "/health"
        )
        
        assert response.status_code == 200
        assert response_time < 100  # Should respond within 100ms
        
        # Multiple requests for consistency
        stats = self._run_multiple_requests(client.get, 20, "/health")
        
        # All requests should succeed
        for response in stats['responses']:
            assert response.status_code == 200
        
        # Performance requirements
        assert stats['avg_time'] < 50  # Average under 50ms
        assert stats['max_time'] < 200  # No request over 200ms
        assert stats['std_dev'] < 30  # Consistent response times
    
    def test_ingest_endpoint_performance(self):
        """Test document ingestion endpoint performance."""
        with requests_mock.Mocker() as m:
            m.post("http://localhost:8081/vectors/add", json=self.mock_endee_add_response)
            
            # Single request timing
            response, response_time = self._measure_response_time(
                client.post, "/ingest",
                json={"content": self.test_content, "metadata": self.test_metadata}
            )
            
            assert response.status_code == 200
            assert response_time < 2000  # Should complete within 2 seconds
            
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
            
            # Performance requirements
            assert stats['avg_time'] < 1500  # Average under 1.5 seconds
            assert stats['max_time'] < 3000  # No request over 3 seconds
    
    def test_search_endpoint_performance(self):
        """Test search endpoint performance."""
        with requests_mock.Mocker() as m:
            m.post("http://localhost:8081/vectors/search", json=self.mock_endee_search_response)
            
            # Single request timing
            response, response_time = self._measure_response_time(
                client.post, "/search",
                json={"query": self.test_query, "top_k": 5}
            )
            
            assert response.status_code == 200
            assert response_time < 1000  # Should complete within 1 second
            
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
            
            # Performance requirements
            assert stats['avg_time'] < 800  # Average under 800ms
            assert stats['max_time'] < 1500  # No request over 1.5 seconds
    
    def test_rag_endpoint_performance(self):
        """Test RAG endpoint performance."""
        with requests_mock.Mocker() as m:
            m.post("http://localhost:8081/vectors/search", json=self.mock_endee_search_response)
            
            # Single request timing
            response, response_time = self._measure_response_time(
                client.post, "/rag",
                json={"query": self.test_query, "top_k": 3}
            )
            
            assert response.status_code == 200
            assert response_time < 1500  # Should complete within 1.5 seconds
            
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
            
            # Performance requirements
            assert stats['avg_time'] < 1200  # Average under 1.2 seconds
            assert stats['max_time'] < 2000  # No request over 2 seconds
    
    def test_concurrent_requests_performance(self):
        """Test performance under concurrent load."""
        import concurrent.futures
        import threading
        
        with requests_mock.Mocker() as m:
            m.post("http://localhost:8081/vectors/search", json=self.mock_endee_search_response)
            
            def make_search_request():
                """Make a single search request."""
                start_time = time.time()
                response = client.post("/search", json={"query": self.test_query, "top_k": 5})
                end_time = time.time()
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
            
            # Performance under concurrent load
            avg_concurrent_time = statistics.mean(response_times)
            max_concurrent_time = max(response_times)
            
            assert avg_concurrent_time < 1000  # Average under 1 second under load
            assert max_concurrent_time < 2000  # No request over 2 seconds under load
    
    def test_large_content_performance(self):
        """Test performance with larger content."""
        with requests_mock.Mocker() as m:
            m.post("http://localhost:8081/vectors/add", json=self.mock_endee_add_response)
            
            # Create larger content (around 5KB)
            large_content = self.test_content * 20  # Approximately 5KB
            
            response, response_time = self._measure_response_time(
                client.post, "/ingest",
                json={"content": large_content, "metadata": self.test_metadata}
            )
            
            assert response.status_code == 200
            assert response_time < 3000  # Should handle large content within 3 seconds
            
            data = response.json()
            assert data["status"] == "success"
            assert data["content_length"] == len(large_content.strip())
    
    def test_multiple_results_search_performance(self):
        """Test search performance with multiple results."""
        with requests_mock.Mocker() as m:
            # Mock response with multiple results
            multiple_results_response = {
                "results": [
                    {
                        "id": f"doc-{i}",
                        "score": 0.9 - (i * 0.1),
                        "metadata": {
                            "content": f"Test document {i} content for performance testing.",
                            "title": f"Test Document {i}",
                            "category": "Performance Test"
                        }
                    }
                    for i in range(10)  # 10 results
                ]
            }
            m.post("http://localhost:8081/vectors/search", json=multiple_results_response)
            
            response, response_time = self._measure_response_time(
                client.post, "/search",
                json={"query": "performance test", "top_k": 10}
            )
            
            assert response.status_code == 200
            assert response_time < 1200  # Should handle multiple results within 1.2 seconds
            
            data = response.json()
            assert len(data["results"]) == 10
            assert data["total_results"] == 10
    
    def test_rag_with_multiple_sources_performance(self):
        """Test RAG performance with multiple sources."""
        with requests_mock.Mocker() as m:
            # Mock response with multiple sources
            multiple_sources_response = {
                "results": [
                    {
                        "id": f"source-{i}",
                        "score": 0.9 - (i * 0.05),
                        "metadata": {
                            "content": f"Source document {i} provides detailed information about machine learning concepts and applications in various domains.",
                            "title": f"ML Source {i}",
                            "category": "AI/ML"
                        }
                    }
                    for i in range(5)  # 5 sources
                ]
            }
            m.post("http://localhost:8081/vectors/search", json=multiple_sources_response)
            
            response, response_time = self._measure_response_time(
                client.post, "/rag",
                json={"query": "Explain machine learning with examples", "top_k": 5}
            )
            
            assert response.status_code == 200
            assert response_time < 2000  # Should handle multiple sources within 2 seconds
            
            data = response.json()
            assert len(data["sources"]) == 5
            assert data["sources_count"] == 5
            assert data["context_length"] > 0
    
    def test_api_response_size_efficiency(self):
        """Test that API responses are reasonably sized."""
        with requests_mock.Mocker() as m:
            m.post("http://localhost:8081/vectors/add", json=self.mock_endee_add_response)
            m.post("http://localhost:8081/vectors/search", json=self.mock_endee_search_response)
            
            # Test ingest response size
            ingest_response = client.post("/ingest", json={
                "content": self.test_content,
                "metadata": self.test_metadata
            })
            ingest_size = len(ingest_response.content)
            assert ingest_size < 1024  # Response should be under 1KB
            
            # Test search response size
            search_response = client.post("/search", json={
                "query": self.test_query,
                "top_k": 5
            })
            search_size = len(search_response.content)
            assert search_size < 5120  # Response should be under 5KB for reasonable results
            
            # Test RAG response size
            rag_response = client.post("/rag", json={
                "query": self.test_query,
                "top_k": 3
            })
            rag_size = len(rag_response.content)
            assert rag_size < 10240  # Response should be under 10KB
    
    def test_memory_usage_stability(self):
        """Test that repeated requests don't cause memory leaks."""
        import gc
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with requests_mock.Mocker() as m:
            m.post("http://localhost:8081/vectors/search", json=self.mock_endee_search_response)
            
            # Make many requests
            for _ in range(50):
                response = client.post("/search", json={
                    "query": f"test query {_}",
                    "top_k": 5
                })
                assert response.status_code == 200
            
            # Force garbage collection
            gc.collect()
            
            # Check memory usage hasn't grown excessively
            final_memory = process.memory_info().rss
            memory_growth = final_memory - initial_memory
            
            # Memory growth should be reasonable (less than 50MB)
            assert memory_growth < 50 * 1024 * 1024  # 50MB limit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])