"""
Real integration test with running mock Endee service.
This test verifies that the Endee client can communicate with the actual mock service.
"""

import pytest
import time
import requests
from endee_client import EndeeHTTPClient, EndeeConnectionError


class TestEndeeRealIntegration:
    """Test Endee client with actual running mock service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use the actual mock Endee service URL
        self.client = EndeeHTTPClient(endee_url="http://localhost:8081")
        self.test_vector = [0.1] * 384  # 384-dimensional test vector
        self.test_metadata = {"title": "Test Document", "content": "Test content"}
        
        # Wait for service to be ready
        self._wait_for_service()
    
    def _wait_for_service(self, max_attempts=10, delay=2):
        """Wait for the Endee service to be ready."""
        for attempt in range(max_attempts):
            try:
                response = requests.get("http://localhost:8081/health", timeout=5)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            
            if attempt < max_attempts - 1:
                time.sleep(delay)
        
        pytest.skip("Mock Endee service is not available")
    
    def test_health_check(self):
        """Test that the mock Endee service is responding."""
        try:
            response = requests.get("http://localhost:8081/health", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
        except requests.exceptions.RequestException:
            pytest.skip("Mock Endee service is not available")
    
    def test_add_vector_real_service(self):
        """Test adding a vector to the real mock service."""
        try:
            result = self.client.add_vector("test-doc-1", self.test_vector, self.test_metadata)
            
            # Verify successful response
            assert result["status"] == "success"
            assert result["id"] == "test-doc-1"
            assert result["dimension"] == 384
            
        except EndeeConnectionError:
            pytest.skip("Mock Endee service is not available")
    
    def test_search_vectors_real_service(self):
        """Test searching vectors in the real mock service."""
        try:
            # Use a unique vector for this test to avoid conflicts
            unique_vector = [0.5] * 384
            unique_metadata = {"title": "Unique Search Document", "content": "Unique search content"}
            
            # First add a vector
            self.client.add_vector("search-doc-unique", unique_vector, unique_metadata)
            
            # Then search for it
            result = self.client.search_vectors(unique_vector, top_k=5)
            
            # Verify search results
            assert "results" in result
            assert len(result["results"]) >= 1
            
            # Find our specific document in the results
            found_doc = None
            for res in result["results"]:
                if res["id"] == "search-doc-unique":
                    found_doc = res
                    break
            
            assert found_doc is not None, "Could not find the added document in search results"
            assert "score" in found_doc
            assert "metadata" in found_doc
            assert found_doc["metadata"]["title"] == "Unique Search Document"
            
        except EndeeConnectionError:
            pytest.skip("Mock Endee service is not available")
    
    def test_add_and_search_multiple_vectors(self):
        """Test adding multiple vectors and searching."""
        try:
            # Add multiple vectors with different content and unique vectors
            vectors_data = [
                ("multi-doc-1", [0.7] * 384, {"title": "Multi Document 1", "content": "Machine learning content"}),
                ("multi-doc-2", [0.8] * 384, {"title": "Multi Document 2", "content": "Deep learning content"}),
                ("multi-doc-3", [0.9] * 384, {"title": "Multi Document 3", "content": "AI research content"}),
            ]
            
            # Add all vectors
            for doc_id, vector, metadata in vectors_data:
                result = self.client.add_vector(doc_id, vector, metadata)
                assert result["status"] == "success"
                assert result["id"] == doc_id
            
            # Search with the first vector
            search_result = self.client.search_vectors([0.7] * 384, top_k=10)
            
            # Verify search results
            assert "results" in search_result
            assert len(search_result["results"]) >= 1
            
            # Find our specific document in the results
            found_doc = None
            for res in search_result["results"]:
                if res["id"] == "multi-doc-1":
                    found_doc = res
                    break
            
            assert found_doc is not None, "Could not find the added document in search results"
            assert found_doc["metadata"]["title"] == "Multi Document 1"
            
        except EndeeConnectionError:
            pytest.skip("Mock Endee service is not available")
    
    def test_search_with_similarity_threshold(self):
        """Test search with similarity threshold."""
        try:
            # Add a test vector
            self.client.add_vector("threshold-doc", self.test_vector, self.test_metadata)
            
            # Search with high similarity threshold (should find exact match)
            result = self.client.search_vectors(self.test_vector, top_k=5, similarity_threshold=0.9)
            
            # Should find the exact match
            assert "results" in result
            results = result["results"]
            
            # All results should have high similarity scores
            for res in results:
                assert res["score"] >= 0.9
                
        except EndeeConnectionError:
            pytest.skip("Mock Endee service is not available")
    
    def test_vector_count_endpoint(self):
        """Test the vector count endpoint of mock service."""
        try:
            # Add a few vectors
            for i in range(3):
                self.client.add_vector(f"count-doc-{i}", self.test_vector, {"index": i})
            
            # Check count via direct HTTP call
            response = requests.get("http://localhost:8081/vectors/count", timeout=5)
            assert response.status_code == 200
            
            data = response.json()
            assert "count" in data
            assert data["count"] >= 3  # At least the 3 we just added
            
        except (EndeeConnectionError, requests.exceptions.RequestException):
            pytest.skip("Mock Endee service is not available")
    
    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(self, 'client'):
            self.client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])