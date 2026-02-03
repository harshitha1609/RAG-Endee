"""
Integration tests for all API endpoints with invalid inputs.
Tests that error handling works for invalid inputs across all endpoints.
"""

import pytest
import requests_mock
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


class TestAPIEndpointsInvalidInputs:
    """Test all API endpoints with invalid inputs to ensure proper error handling."""
    
    def test_ingest_endpoint_empty_content(self):
        """Test ingest endpoint with empty content."""
        response = client.post("/ingest", json={
            "content": ""
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        # Check that the error mentions content length requirement
        error_msg = str(data["detail"])
        assert "min_length" in error_msg or "at least 3 characters" in error_msg
    
    def test_ingest_endpoint_whitespace_only_content(self):
        """Test ingest endpoint with whitespace-only content."""
        response = client.post("/ingest", json={
            "content": "   \n\t   "
        })
        
        assert response.status_code == 400  # Custom validation error
        data = response.json()
        assert "detail" in data
        assert "whitespace" in data["detail"]
    
    def test_ingest_endpoint_too_short_content(self):
        """Test ingest endpoint with content too short."""
        response = client.post("/ingest", json={
            "content": "Hi"  # Only 2 characters, minimum is 3
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "min_length" in error_msg or "at least 3 characters" in error_msg
    
    def test_ingest_endpoint_too_long_content(self):
        """Test ingest endpoint with content too long."""
        # Create content longer than 50000 characters
        long_content = "A" * 50001
        
        response = client.post("/ingest", json={
            "content": long_content
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "max_length" in error_msg or "50000" in error_msg
    
    def test_ingest_endpoint_missing_content(self):
        """Test ingest endpoint with missing content field."""
        response = client.post("/ingest", json={
            "metadata": {"title": "Test"}
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "content" in error_msg and "required" in error_msg
    
    def test_ingest_endpoint_invalid_content_type(self):
        """Test ingest endpoint with non-string content."""
        response = client.post("/ingest", json={
            "content": 12345  # Number instead of string
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
    
    def test_ingest_endpoint_invalid_metadata_type(self):
        """Test ingest endpoint with invalid metadata type."""
        response = client.post("/ingest", json={
            "content": "Valid content here",
            "metadata": "invalid metadata type"  # String instead of dict
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
    
    def test_ingest_endpoint_malformed_json(self):
        """Test ingest endpoint with malformed JSON."""
        response = client.post("/ingest", 
                             data='{"content": "test", "metadata":}',  # Malformed JSON
                             headers={"Content-Type": "application/json"})
        
        assert response.status_code == 422  # JSON decode error
        data = response.json()
        assert "detail" in data
    
    def test_ingest_endpoint_endee_connection_error(self):
        """Test ingest endpoint when Endee is unavailable."""
        with requests_mock.Mocker() as m:
            # Mock connection error
            import requests
            m.post("http://localhost:8081/vectors/add", 
                   exc=requests.exceptions.ConnectionError)
            
            response = client.post("/ingest", json={
                "content": "Valid content for testing connection error"
            })
            
            assert response.status_code == 503  # Service unavailable
            data = response.json()
            assert "Vector database unavailable" in data["detail"]
    
    def test_ingest_endpoint_endee_server_error(self):
        """Test ingest endpoint when Endee returns server error."""
        with requests_mock.Mocker() as m:
            # Mock server error
            m.post("http://localhost:8081/vectors/add", 
                   status_code=500, text="Internal Server Error")
            
            response = client.post("/ingest", json={
                "content": "Valid content for testing server error"
            })
            
            assert response.status_code == 503  # Service unavailable
            data = response.json()
            assert "Vector database unavailable" in data["detail"]
    
    def test_search_endpoint_empty_query(self):
        """Test search endpoint with empty query."""
        response = client.post("/search", json={
            "query": ""
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "min_length" in error_msg or "at least 2 characters" in error_msg
    
    def test_search_endpoint_whitespace_only_query(self):
        """Test search endpoint with whitespace-only query."""
        response = client.post("/search", json={
            "query": "   \n\t   "
        })
        
        assert response.status_code == 500  # Search error
        data = response.json()
        assert "detail" in data
        assert "whitespace" in data["detail"]
    
    def test_search_endpoint_too_short_query(self):
        """Test search endpoint with query too short."""
        response = client.post("/search", json={
            "query": "A"  # Only 1 character, minimum is 2
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "min_length" in error_msg or "at least 2 characters" in error_msg
    
    def test_search_endpoint_too_long_query(self):
        """Test search endpoint with query too long."""
        long_query = "A" * 1001  # Longer than 1000 characters
        
        response = client.post("/search", json={
            "query": long_query
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "max_length" in error_msg or "1000" in error_msg
    
    def test_search_endpoint_missing_query(self):
        """Test search endpoint with missing query field."""
        response = client.post("/search", json={
            "top_k": 5
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "query" in error_msg and "required" in error_msg
    
    def test_search_endpoint_invalid_top_k_zero(self):
        """Test search endpoint with top_k = 0."""
        response = client.post("/search", json={
            "query": "valid query",
            "top_k": 0
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "greater than or equal to 1" in error_msg or "ge=1" in error_msg
    
    def test_search_endpoint_invalid_top_k_negative(self):
        """Test search endpoint with negative top_k."""
        response = client.post("/search", json={
            "query": "valid query",
            "top_k": -5
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
    
    def test_search_endpoint_invalid_top_k_too_large(self):
        """Test search endpoint with top_k too large."""
        response = client.post("/search", json={
            "query": "valid query",
            "top_k": 101  # Maximum is 100
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "less than or equal to 100" in error_msg or "le=100" in error_msg
    
    def test_search_endpoint_invalid_similarity_threshold_negative(self):
        """Test search endpoint with negative similarity threshold."""
        response = client.post("/search", json={
            "query": "valid query",
            "similarity_threshold": -0.1
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "greater than or equal to 0" in error_msg or "ge=0" in error_msg
    
    def test_search_endpoint_invalid_similarity_threshold_too_large(self):
        """Test search endpoint with similarity threshold > 1.0."""
        response = client.post("/search", json={
            "query": "valid query",
            "similarity_threshold": 1.1
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "less than or equal to 1" in error_msg or "le=1" in error_msg
    
    def test_search_endpoint_invalid_query_type(self):
        """Test search endpoint with non-string query."""
        response = client.post("/search", json={
            "query": 12345  # Number instead of string
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
    
    def test_search_endpoint_endee_connection_error(self):
        """Test search endpoint when Endee is unavailable."""
        with requests_mock.Mocker() as m:
            # Mock connection error
            import requests
            m.post("http://localhost:8081/vectors/search", 
                   exc=requests.exceptions.ConnectionError)
            
            response = client.post("/search", json={
                "query": "valid query for testing connection error"
            })
            
            assert response.status_code == 500  # Search error (after retries)
            data = response.json()
            assert "Search failed" in data["detail"]
    
    def test_search_endpoint_endee_server_error(self):
        """Test search endpoint when Endee returns server error."""
        with requests_mock.Mocker() as m:
            # Mock server error
            m.post("http://localhost:8081/vectors/search", 
                   status_code=500, text="Internal Server Error")
            
            response = client.post("/search", json={
                "query": "valid query for testing server error"
            })
            
            assert response.status_code == 500  # Internal server error (after retries)
            data = response.json()
            assert "Search failed" in data["detail"]
    
    def test_rag_endpoint_empty_query(self):
        """Test RAG endpoint with empty query."""
        response = client.post("/rag", json={
            "query": ""
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "min_length" in error_msg or "at least 2 characters" in error_msg
    
    def test_rag_endpoint_whitespace_only_query(self):
        """Test RAG endpoint with whitespace-only query."""
        response = client.post("/rag", json={
            "query": "   \n\t   "
        })
        
        assert response.status_code == 500  # Internal server error
        data = response.json()
        assert "detail" in data
        assert "whitespace" in data["detail"]
    
    def test_rag_endpoint_too_short_query(self):
        """Test RAG endpoint with query too short."""
        response = client.post("/rag", json={
            "query": "A"  # Only 1 character, minimum is 2
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "min_length" in error_msg or "at least 2 characters" in error_msg
    
    def test_rag_endpoint_too_long_query(self):
        """Test RAG endpoint with query too long."""
        long_query = "A" * 1001  # Longer than 1000 characters
        
        response = client.post("/rag", json={
            "query": long_query
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "max_length" in error_msg or "1000" in error_msg
    
    def test_rag_endpoint_missing_query(self):
        """Test RAG endpoint with missing query field."""
        response = client.post("/rag", json={
            "top_k": 3
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "query" in error_msg and "required" in error_msg
    
    def test_rag_endpoint_invalid_top_k_zero(self):
        """Test RAG endpoint with top_k = 0."""
        response = client.post("/rag", json={
            "query": "valid query",
            "top_k": 0
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "greater than or equal to 1" in error_msg or "ge=1" in error_msg
    
    def test_rag_endpoint_invalid_top_k_negative(self):
        """Test RAG endpoint with negative top_k."""
        response = client.post("/rag", json={
            "query": "valid query",
            "top_k": -3
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
    
    def test_rag_endpoint_invalid_top_k_too_large(self):
        """Test RAG endpoint with top_k too large."""
        response = client.post("/rag", json={
            "query": "valid query",
            "top_k": 21  # Maximum is 20
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "less than or equal to 20" in error_msg or "le=20" in error_msg
    
    def test_rag_endpoint_invalid_similarity_threshold_negative(self):
        """Test RAG endpoint with negative similarity threshold."""
        response = client.post("/rag", json={
            "query": "valid query",
            "similarity_threshold": -0.1
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "greater than or equal to 0" in error_msg or "ge=0" in error_msg
    
    def test_rag_endpoint_invalid_similarity_threshold_too_large(self):
        """Test RAG endpoint with similarity threshold > 1.0."""
        response = client.post("/rag", json={
            "query": "valid query",
            "similarity_threshold": 1.1
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "less than or equal to 1" in error_msg or "le=1" in error_msg
    
    def test_rag_endpoint_invalid_max_context_length_too_small(self):
        """Test RAG endpoint with max_context_length too small."""
        response = client.post("/rag", json={
            "query": "valid query",
            "max_context_length": 50  # Minimum is 100
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "greater than or equal to 100" in error_msg or "ge=100" in error_msg
    
    def test_rag_endpoint_invalid_max_context_length_too_large(self):
        """Test RAG endpoint with max_context_length too large."""
        response = client.post("/rag", json={
            "query": "valid query",
            "max_context_length": 50001  # Maximum is 50000
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        error_msg = str(data["detail"])
        assert "less than or equal to 50000" in error_msg or "le=50000" in error_msg
    
    def test_rag_endpoint_invalid_query_type(self):
        """Test RAG endpoint with non-string query."""
        response = client.post("/rag", json={
            "query": 12345  # Number instead of string
        })
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
    
    def test_rag_endpoint_endee_connection_error(self):
        """Test RAG endpoint when Endee is unavailable."""
        with requests_mock.Mocker() as m:
            # Mock connection error
            import requests
            m.post("http://localhost:8081/vectors/search", 
                   exc=requests.exceptions.ConnectionError)
            
            response = client.post("/rag", json={
                "query": "valid query for testing connection error"
            })
            
            assert response.status_code == 500  # Internal server error (from RAG service)
            data = response.json()
            assert "Document retrieval failed" in data["detail"]
    
    def test_rag_endpoint_endee_server_error(self):
        """Test RAG endpoint when Endee returns server error."""
        with requests_mock.Mocker() as m:
            # Mock server error
            m.post("http://localhost:8081/vectors/search", 
                   status_code=500, text="Internal Server Error")
            
            response = client.post("/rag", json={
                "query": "valid query for testing server error"
            })
            
            assert response.status_code == 500  # Internal server error
            data = response.json()
            assert "Document retrieval failed" in data["detail"]
    
    def test_all_endpoints_invalid_http_methods(self):
        """Test that endpoints reject invalid HTTP methods."""
        # Test GET on POST-only endpoints
        response = client.get("/ingest")
        assert response.status_code == 405  # Method not allowed
        
        response = client.get("/search")
        assert response.status_code == 405  # Method not allowed
        
        response = client.get("/rag")
        assert response.status_code == 405  # Method not allowed
        
        # Test PUT on POST-only endpoints
        response = client.put("/ingest", json={"content": "test"})
        assert response.status_code == 405  # Method not allowed
        
        response = client.put("/search", json={"query": "test"})
        assert response.status_code == 405  # Method not allowed
        
        response = client.put("/rag", json={"query": "test"})
        assert response.status_code == 405  # Method not allowed
        
        # Test DELETE on POST-only endpoints
        response = client.delete("/ingest")
        assert response.status_code == 405  # Method not allowed
        
        response = client.delete("/search")
        assert response.status_code == 405  # Method not allowed
        
        response = client.delete("/rag")
        assert response.status_code == 405  # Method not allowed
    
    def test_endpoints_with_no_content_type(self):
        """Test endpoints with missing Content-Type header."""
        # Test with raw data instead of JSON
        response = client.post("/ingest", data="not json")
        assert response.status_code == 422  # Unprocessable entity
        
        response = client.post("/search", data="not json")
        assert response.status_code == 422  # Unprocessable entity
        
        response = client.post("/rag", data="not json")
        assert response.status_code == 422  # Unprocessable entity
    
    def test_endpoints_with_invalid_content_type(self):
        """Test endpoints with invalid Content-Type header."""
        # Test with XML content type
        response = client.post("/ingest", 
                             data='{"content": "test"}',
                             headers={"Content-Type": "application/xml"})
        assert response.status_code == 422  # Unprocessable entity
        
        response = client.post("/search", 
                             data='{"query": "test"}',
                             headers={"Content-Type": "text/plain"})
        assert response.status_code == 422  # Unprocessable entity
        
        response = client.post("/rag", 
                             data='{"query": "test"}',
                             headers={"Content-Type": "application/xml"})
        assert response.status_code == 422  # Unprocessable entity
    
    def test_nonexistent_endpoints(self):
        """Test requests to nonexistent endpoints."""
        response = client.get("/nonexistent")
        assert response.status_code == 404  # Not found
        
        response = client.post("/invalid-endpoint", json={"test": "data"})
        assert response.status_code == 404  # Not found
        
        response = client.get("/api/v1/search")  # Wrong path
        assert response.status_code == 404  # Not found
    
    def test_error_response_format(self):
        """Test that error responses follow consistent format."""
        # Test with a simple validation error
        response = client.post("/ingest", json={
            "content": ""  # Too short
        })
        
        assert response.status_code == 422
        data = response.json()
        
        # Check that response has expected structure
        assert "detail" in data
        # Pydantic validation errors have a specific structure
        assert isinstance(data["detail"], list)
        
        # Test with service unavailable error
        with requests_mock.Mocker() as m:
            import requests
            m.post("http://localhost:8081/vectors/add", 
                   exc=requests.exceptions.ConnectionError)
            
            response = client.post("/ingest", json={
                "content": "Valid content for testing error format"
            })
            
            assert response.status_code == 503
            data = response.json()
            
            # Check custom error response format
            assert "error" in data
            assert "detail" in data
            assert "timestamp" in data
            assert data["error"] == "HTTP Error"
            assert "Vector database unavailable" in data["detail"]


if __name__ == "__main__":
    pytest.main([__file__])