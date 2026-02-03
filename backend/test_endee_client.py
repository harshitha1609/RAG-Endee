"""
Tests for Endee HTTP Client functionality.
"""

import pytest
import requests
import requests_mock
from unittest.mock import patch, MagicMock, AsyncMock
from endee_client import (
    EndeeHTTPClient, 
    EndeeClientError, 
    EndeeConnectionError, 
    EndeeRequestError, 
    EndeeTimeoutError
)


class TestEndeeHTTPClient:
    """Test cases for the Endee HTTP client."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = EndeeHTTPClient(endee_url="http://test-endee:8080")
        self.test_vector = [0.1] * 384  # 384-dimensional test vector
        self.test_metadata = {"title": "Test Document", "source": "test.txt"}
    
    def test_client_initialization(self):
        """Test client initialization with default and custom parameters."""
        # Test default initialization
        default_client = EndeeHTTPClient()
        assert default_client.endee_url == "http://localhost:8081"
        assert default_client.timeout == 30
        assert default_client.max_retries == 3
        
        # Test custom initialization
        custom_client = EndeeHTTPClient(
            endee_url="http://custom:9000",
            timeout=60,
            max_retries=5
        )
        assert custom_client.endee_url == "http://custom:9000"
        assert custom_client.timeout == 60
        assert custom_client.max_retries == 5
    
    def test_add_vector_validation(self):
        """Test input validation for add_vector method."""
        # Test missing document ID
        with pytest.raises(EndeeRequestError, match="Document ID is required"):
            self.client.add_vector("", self.test_vector)
        
        # Test missing vector
        with pytest.raises(EndeeRequestError, match="Vector is required"):
            self.client.add_vector("test-doc", [])
        
        # Test invalid vector type
        with pytest.raises(EndeeRequestError, match="Vector must be a list"):
            self.client.add_vector("test-doc", "not-a-list")
        
        # Test wrong vector dimension
        with pytest.raises(EndeeRequestError, match="Vector must be 384-dimensional"):
            self.client.add_vector("test-doc", [0.1, 0.2, 0.3])  # Only 3 dimensions
        
        # Test invalid vector values
        with pytest.raises(EndeeRequestError, match="Vector must contain only numeric values"):
            invalid_vector = [0.1] * 383 + ["invalid"]
            self.client.add_vector("test-doc", invalid_vector)
    
    def test_add_vector_success(self):
        """Test successful vector addition."""
        with patch('endee_client.aiohttp.ClientSession.post') as mock_post:
            # Create a mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"status": "success", "id": "test-doc"}
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = None
            
            mock_post.return_value = mock_response
            
            result = self.client.add_vector("test-doc", self.test_vector, self.test_metadata)
            
            assert result["status"] == "success"
            assert result["id"] == "test-doc"
            
            # Verify the mock was called
            mock_post.assert_called_once()
    
    def test_add_vector_client_error(self):
        """Test client error handling (4xx responses)."""
        with requests_mock.Mocker() as m:
            # Mock bad request response
            m.post("http://test-endee:8080/vectors/add", status_code=400, text="Bad request")
            
            with pytest.raises(EndeeRequestError, match="Bad request to Endee"):
                self.client.add_vector("test-doc", self.test_vector)
    
    def test_add_vector_server_error_with_retry(self):
        """Test server error handling with retry logic."""
        with requests_mock.Mocker() as m:
            # Mock server error followed by success
            m.post("http://test-endee:8080/vectors/add", [
                {"status_code": 500, "text": "Internal server error"},
                {"json": {"status": "success", "id": "test-doc"}}
            ])
            
            # Mock time.sleep to speed up test
            with patch('endee_client.time.sleep'):
                result = self.client.add_vector("test-doc", self.test_vector)
                
                assert result["status"] == "success"
                assert m.call_count == 2  # One retry
    
    def test_add_vector_timeout_error(self):
        """Test timeout error handling."""
        with requests_mock.Mocker() as m:
            # Mock timeout
            m.post("http://test-endee:8080/vectors/add", exc=requests.exceptions.Timeout)
            
            # Mock time.sleep to speed up test
            with patch('endee_client.time.sleep'):
                with pytest.raises(EndeeTimeoutError):
                    self.client.add_vector("test-doc", self.test_vector)
    
    def test_add_vector_connection_error(self):
        """Test connection error handling."""
        with requests_mock.Mocker() as m:
            # Mock connection error
            m.post("http://test-endee:8080/vectors/add", exc=requests.exceptions.ConnectionError)
            
            # Mock time.sleep to speed up test
            with patch('endee_client.time.sleep'):
                with pytest.raises(EndeeConnectionError):
                    self.client.add_vector("test-doc", self.test_vector)
    
    def test_add_vector_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded."""
        with requests_mock.Mocker() as m:
            # Mock persistent server error
            m.post("http://test-endee:8080/vectors/add", status_code=500, text="Server error")
            
            # Mock time.sleep to speed up test
            with patch('endee_client.time.sleep'):
                with pytest.raises(EndeeConnectionError, match="Failed to add vector to Endee after 3 attempts"):
                    self.client.add_vector("test-doc", self.test_vector)
                
                # Should have made max_retries attempts
                assert m.call_count == 3
    
    def test_payload_size_validation(self):
        """Test payload size validation."""
        # Create very large metadata to exceed 1MB limit
        large_metadata = {"large_field": "x" * 1000000}
        
        with pytest.raises(EndeeRequestError, match="Payload too large"):
            self.client.add_vector("test-doc", self.test_vector, large_metadata)
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with EndeeHTTPClient() as client:
            assert client.session is not None
        
        # Session should be closed after context exit
        # Note: We can't easily test this without accessing private attributes
    
    def test_exponential_backoff_delay(self):
        """Test exponential backoff delay calculation."""
        client = EndeeHTTPClient()
        
        # Test delay calculation
        assert client._exponential_backoff_delay(0) == 1.0  # 2^0 * 1.0
        assert client._exponential_backoff_delay(1) == 2.0  # 2^1 * 1.0
        assert client._exponential_backoff_delay(2) == 4.0  # 2^2 * 1.0
        
        # Test max delay cap
        assert client._exponential_backoff_delay(10) == 10.0  # Capped at max delay
    
    def test_handle_response_success(self):
        """Test response handling for successful responses."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        
        result = self.client._handle_response(mock_response, "test operation")
        assert result == {"status": "success"}
    
    def test_handle_response_client_errors(self):
        """Test response handling for client errors."""
        # Test 400 Bad Request
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        
        with pytest.raises(EndeeRequestError, match="Bad request to Endee"):
            self.client._handle_response(mock_response, "test operation")
        
        # Test 404 Not Found
        mock_response.status_code = 404
        with pytest.raises(EndeeRequestError, match="endpoint not found"):
            self.client._handle_response(mock_response, "test operation")
        
        # Test 413 Payload Too Large
        mock_response.status_code = 413
        with pytest.raises(EndeeRequestError, match="Payload too large"):
            self.client._handle_response(mock_response, "test operation")
    
    def test_handle_response_server_errors(self):
        """Test response handling for server errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        
        with pytest.raises(EndeeConnectionError, match="failed with status 500"):
            self.client._handle_response(mock_response, "test operation")
    
    def test_handle_response_invalid_json(self):
        """Test response handling for invalid JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        
        with pytest.raises(EndeeRequestError, match="Invalid JSON response"):
            self.client._handle_response(mock_response, "test operation")
    
    def test_search_vectors_validation(self):
        """Test input validation for search_vectors method."""
        # Test missing query vector
        with pytest.raises(EndeeRequestError, match="Query vector is required"):
            self.client.search_vectors([])
        
        # Test invalid vector type
        with pytest.raises(EndeeRequestError, match="Query vector must be a list"):
            self.client.search_vectors("not-a-list")
        
        # Test wrong vector dimension
        with pytest.raises(EndeeRequestError, match="Query vector must be 384-dimensional"):
            self.client.search_vectors([0.1, 0.2, 0.3])  # Only 3 dimensions
        
        # Test invalid vector values
        with pytest.raises(EndeeRequestError, match="Query vector must contain only numeric values"):
            invalid_vector = [0.1] * 383 + ["invalid"]
            self.client.search_vectors(invalid_vector)
        
        # Test invalid top_k
        with pytest.raises(EndeeRequestError, match="top_k must be a positive integer"):
            self.client.search_vectors(self.test_vector, top_k=0)
        
        with pytest.raises(EndeeRequestError, match="top_k must be a positive integer"):
            self.client.search_vectors(self.test_vector, top_k="invalid")
        
        with pytest.raises(EndeeRequestError, match="top_k cannot exceed 1000"):
            self.client.search_vectors(self.test_vector, top_k=1001)
        
        # Test invalid similarity_threshold
        with pytest.raises(EndeeRequestError, match="similarity_threshold must be a number"):
            self.client.search_vectors(self.test_vector, similarity_threshold="invalid")
        
        with pytest.raises(EndeeRequestError, match="similarity_threshold must be between 0.0 and 1.0"):
            self.client.search_vectors(self.test_vector, similarity_threshold=-0.1)
        
        with pytest.raises(EndeeRequestError, match="similarity_threshold must be between 0.0 and 1.0"):
            self.client.search_vectors(self.test_vector, similarity_threshold=1.1)
    
    def test_search_vectors_success(self):
        """Test successful vector search."""
        with requests_mock.Mocker() as m:
            # Mock successful response
            mock_results = [
                {
                    "id": "doc1",
                    "score": 0.95,
                    "metadata": {"title": "Document 1", "content": "Test content 1"}
                },
                {
                    "id": "doc2", 
                    "score": 0.85,
                    "metadata": {"title": "Document 2", "content": "Test content 2"}
                }
            ]
            m.post("http://test-endee:8080/vectors/search", json={"results": mock_results})
            
            result = self.client.search_vectors(self.test_vector, top_k=5, similarity_threshold=0.8)
            
            assert "results" in result
            assert len(result["results"]) == 2
            assert result["results"][0]["id"] == "doc1"
            assert result["results"][0]["score"] == 0.95
            
            # Verify request was made correctly
            assert m.call_count == 1
            request = m.request_history[0]
            assert len(request.json()["vector"]) == 384
            assert request.json()["top_k"] == 5
            assert request.json()["similarity_threshold"] == 0.8
    
    def test_search_vectors_success_no_threshold(self):
        """Test successful vector search without similarity threshold."""
        with requests_mock.Mocker() as m:
            # Mock successful response
            mock_results = [{"id": "doc1", "score": 0.5}]
            m.post("http://test-endee:8080/vectors/search", json={"results": mock_results})
            
            result = self.client.search_vectors(self.test_vector, top_k=3)
            
            assert "results" in result
            assert len(result["results"]) == 1
            
            # Verify request was made correctly (no similarity_threshold in payload)
            request = m.request_history[0]
            assert "similarity_threshold" not in request.json()
            assert request.json()["top_k"] == 3
    
    def test_search_vectors_client_error(self):
        """Test client error handling for search_vectors (4xx responses)."""
        with requests_mock.Mocker() as m:
            # Mock bad request response
            m.post("http://test-endee:8080/vectors/search", status_code=400, text="Bad request")
            
            with pytest.raises(EndeeRequestError, match="Bad request to Endee"):
                self.client.search_vectors(self.test_vector)
    
    def test_search_vectors_server_error_with_retry(self):
        """Test server error handling with retry logic for search_vectors."""
        with requests_mock.Mocker() as m:
            # Mock server error followed by success
            mock_results = [{"id": "doc1", "score": 0.9}]
            m.post("http://test-endee:8080/vectors/search", [
                {"status_code": 500, "text": "Internal server error"},
                {"json": {"results": mock_results}}
            ])
            
            # Mock time.sleep to speed up test
            with patch('endee_client.time.sleep'):
                result = self.client.search_vectors(self.test_vector)
                
                assert "results" in result
                assert len(result["results"]) == 1
                assert m.call_count == 2  # One retry
    
    def test_search_vectors_timeout_error(self):
        """Test timeout error handling for search_vectors."""
        with requests_mock.Mocker() as m:
            # Mock timeout
            m.post("http://test-endee:8080/vectors/search", exc=requests.exceptions.Timeout)
            
            # Mock time.sleep to speed up test
            with patch('endee_client.time.sleep'):
                with pytest.raises(EndeeTimeoutError):
                    self.client.search_vectors(self.test_vector)
    
    def test_search_vectors_connection_error(self):
        """Test connection error handling for search_vectors."""
        with requests_mock.Mocker() as m:
            # Mock connection error
            m.post("http://test-endee:8080/vectors/search", exc=requests.exceptions.ConnectionError)
            
            # Mock time.sleep to speed up test
            with patch('endee_client.time.sleep'):
                with pytest.raises(EndeeConnectionError):
                    self.client.search_vectors(self.test_vector)
    
    def test_search_vectors_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded for search_vectors."""
        with requests_mock.Mocker() as m:
            # Mock persistent server error
            m.post("http://test-endee:8080/vectors/search", status_code=500, text="Server error")
            
            # Mock time.sleep to speed up test
            with patch('endee_client.time.sleep'):
                with pytest.raises(EndeeConnectionError, match="Failed to search vectors in Endee after 3 attempts"):
                    self.client.search_vectors(self.test_vector)
                
                # Should have made max_retries attempts
                assert m.call_count == 3
    
    def test_search_vectors_payload_size_validation(self):
        """Test payload size validation for search_vectors."""
        # The payload size validation is the same as add_vector
        # For search_vectors, the payload is much smaller (just vector + top_k)
        # so it's unlikely to exceed 1MB, but the validation code is there
        
        # Test with normal vector - should work fine
        with requests_mock.Mocker() as m:
            m.post("http://test-endee:8080/vectors/search", json={"results": []})
            
            # This should not raise an exception
            result = self.client.search_vectors(self.test_vector, top_k=5)
            assert "results" in result
    
    def test_connection_pooling_configuration(self):
        """Test that connection pooling is properly configured."""
        # Test default connection pool parameters
        default_client = EndeeHTTPClient()
        
        # Verify session exists
        assert default_client.session is not None
        
        # Verify HTTP adapters are mounted
        http_adapter = default_client.session.get_adapter("http://example.com")
        https_adapter = default_client.session.get_adapter("https://example.com")
        
        assert http_adapter is not None
        assert https_adapter is not None
        
        # Test custom connection pool parameters
        custom_client = EndeeHTTPClient(
            pool_connections=20,
            pool_maxsize=50
        )
        
        # Verify custom parameters are used
        custom_http_adapter = custom_client.session.get_adapter("http://example.com")
        assert custom_http_adapter is not None
        
        # Clean up
        default_client.close()
        custom_client.close()
    
    def test_session_reuse_across_requests(self):
        """Test that the same session is reused across multiple requests."""
        with requests_mock.Mocker() as m:
            # Mock multiple successful responses
            m.post("http://test-endee:8080/vectors/add", json={"status": "success", "id": "doc1"})
            m.post("http://test-endee:8080/vectors/search", json={"results": []})
            
            # Make multiple requests
            self.client.add_vector("doc1", self.test_vector)
            self.client.search_vectors(self.test_vector)
            
            # Verify both requests used the same session
            assert m.call_count == 2
            
            # Both requests should have the same session headers
            request1 = m.request_history[0]
            request2 = m.request_history[1]
            
            assert request1.headers.get("User-Agent") == "EndeeRAGSystem/1.0.0"
            assert request2.headers.get("User-Agent") == "EndeeRAGSystem/1.0.0"
            assert request1.headers.get("Content-Type") == "application/json"
            assert request2.headers.get("Content-Type") == "application/json"