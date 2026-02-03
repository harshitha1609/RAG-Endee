"""
Test detailed error messages without exposing internals.
Tests the enhanced error handling implementation that provides helpful error messages
while protecting sensitive internal system details.
"""

import pytest
import requests_mock
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app import app, sanitize_error_message, create_detailed_error_message
from ingest import DocumentIngestionService, ValidationError, EmbeddingError
from search import VectorSearchService, SearchError
from endee_client import EndeeConnectionError, ServiceUnavailableError, EndeeTimeoutError

client = TestClient(app)


class TestDetailedErrorMessages:
    """Test suite for detailed error messages without exposing internals."""
    
    def test_sanitize_error_message_basic_patterns(self):
        """Test that basic sensitive patterns are sanitized."""
        test_cases = [
            ("password=secret123", "password=***"),
            ("user=admin", "user=***"),
            ("host=internal-server.company.com", "host=***"),
            ("token=abc123def456", "token=***"),
            ("api_key=sk-1234567890", "api_key=***"),
            ("secret=mysecret", "secret=***"),
        ]
        
        for original, expected in test_cases:
            result = sanitize_error_message(original)
            assert expected in result
            # Ensure original sensitive data is not present
            sensitive_part = original.split('=')[1]
            assert sensitive_part not in result
    
    def test_sanitize_error_message_advanced_patterns(self):
        """Test that advanced sensitive patterns are sanitized."""
        test_cases = [
            ("Connection failed to internal-db-server:5432", "internal-***"),
            ("postgresql://user:pass@localhost:5432/db", "postgresql://***"),
            ("File \"/home/user/app.py\", line 123", "File \"***\", line ***"),
            ("Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9", "bearer ***"),
            ("localhost:8080", "localhost:***"),
            ("192.168.1.100:3000", "192.168.*.*:***"),
        ]
        
        for original, expected_pattern in test_cases:
            result = sanitize_error_message(original)
            assert expected_pattern in result or "***" in result
    
    def test_sanitize_error_message_preserves_useful_info(self):
        """Test that sanitization preserves useful error information."""
        original = "Connection refused by server=internal-host, check network connectivity"
        result = sanitize_error_message(original)
        
        # Should preserve useful context
        assert "Connection refused" in result
        assert "check network connectivity" in result
        # Should sanitize sensitive info
        assert "server=***" in result
        assert "internal-host" not in result
    
    def test_create_detailed_error_message_validation(self):
        """Test detailed error message creation for validation errors."""
        result = create_detailed_error_message(
            "validation", 
            "Field 'content' is required", 
            {"operation": "document_ingestion", "field": "content"}
        )
        
        assert "Input validation failed" in result
        assert "Field: content" in result
        assert "Operation: document_ingestion" in result
        assert "Please check that all required fields are provided" in result
        assert "Suggestions:" in result
    
    def test_create_detailed_error_message_connection(self):
        """Test detailed error message creation for connection errors."""
        result = create_detailed_error_message(
            "connection", 
            "Connection timeout", 
            {"operation": "semantic_search"}
        )
        
        assert "Service connection failed" in result
        assert "Operation: semantic_search" in result
        assert "vector database service may be temporarily unavailable" in result
        assert "try again in a few moments" in result
    
    def test_create_detailed_error_message_timeout(self):
        """Test detailed error message creation for timeout errors."""
        result = create_detailed_error_message(
            "timeout", 
            "Request timed out after 30 seconds", 
            {"operation": "document_ingestion"}
        )
        
        assert "Request timed out" in result
        assert "Operation: document_ingestion" in result
        assert "took longer than expected" in result
        assert "high system load or network issues" in result
    
    def test_create_detailed_error_message_embedding(self):
        """Test detailed error message creation for embedding errors."""
        result = create_detailed_error_message(
            "embedding", 
            "Text processing failed", 
            {"operation": "document_ingestion"}
        )
        
        assert "Text processing failed" in result
        assert "Operation: document_ingestion" in result
        assert "unsupported characters or format" in result
        assert "reducing the text length" in result
    
    def test_ingest_endpoint_detailed_validation_error(self):
        """Test that ingest endpoint returns detailed validation error messages."""
        # Test with whitespace-only content
        response = client.post("/ingest", json={"content": "   \n\t   "})
        assert response.status_code == 400
        
        data = response.json()
        assert "Input validation failed" in data["detail"]
        assert "Suggestions:" in data["detail"]
        assert "check that all required fields are provided" in data["detail"]
    
    def test_ingest_endpoint_detailed_connection_error(self):
        """Test that ingest endpoint returns detailed connection error messages."""
        with patch('ingest.DocumentIngestionService.ingest_document') as mock_ingest:
            mock_ingest.side_effect = EndeeConnectionError("Vector database is temporarily unavailable")
            
            response = client.post("/ingest", json={
                "content": "Valid content for testing connection error"
            })
            
            assert response.status_code == 503
            data = response.json()
            assert "Service connection failed" in data["detail"]
            assert "vector database service may be temporarily unavailable" in data["detail"]
            assert "try again in a few moments" in data["detail"]
    
    def test_search_endpoint_detailed_embedding_error(self):
        """Test that search endpoint returns detailed embedding error messages."""
        with patch('search.VectorSearchService.search') as mock_search:
            mock_search.side_effect = EmbeddingError("Query processing failed")
            
            response = client.post("/search", json={"query": "test query"})
            
            assert response.status_code == 422
            data = response.json()
            assert "Text processing failed" in data["detail"]
            assert "unsupported characters or format" in data["detail"]
            assert "simplifying the content" in data["detail"]
    
    def test_rag_endpoint_detailed_timeout_error(self):
        """Test that RAG endpoint returns detailed timeout error messages."""
        with patch('search.VectorSearchService.search') as mock_search:
            mock_search.side_effect = EndeeTimeoutError("Request timed out", timeout_duration=30.0)
            
            response = client.post("/rag", json={"query": "test query"})
            
            assert response.status_code == 504
            data = response.json()
            assert "Request timed out" in data["detail"]
            assert "took longer than expected" in data["detail"]
            assert "high system load or network issues" in data["detail"]
    
    def test_service_unavailable_detailed_error(self):
        """Test detailed error messages for service unavailable scenarios."""
        with patch('search.VectorSearchService.search') as mock_search:
            mock_search.side_effect = ServiceUnavailableError(
                "Service unavailable - circuit breaker is OPEN",
                consecutive_failures=5
            )
            
            response = client.post("/search", json={"query": "test query"})
            
            assert response.status_code == 503
            data = response.json()
            assert "Service temporarily unavailable" in data["detail"]
            assert "experiencing high load or maintenance" in data["detail"]
            assert "try again in a few minutes" in data["detail"]
    
    def test_error_messages_do_not_expose_stack_traces(self):
        """Test that error messages don't expose stack traces or internal paths."""
        with patch('ingest.DocumentIngestionService.ingest_document') as mock_ingest:
            # Simulate an error with stack trace information
            mock_ingest.side_effect = Exception(
                'File "/usr/local/app/backend/ingest.py", line 123, in _generate_embedding\n'
                'Traceback (most recent call last):\n'
                'RuntimeError: Model loading failed at /opt/models/sentence-transformers'
            )
            
            response = client.post("/ingest", json={
                "content": "Valid content for testing error sanitization"
            })
            
            assert response.status_code == 500
            data = response.json()
            
            # Should not contain internal paths or stack traces
            assert "/usr/local/app" not in data["detail"]
            assert "/opt/models" not in data["detail"]
            assert "Traceback" not in data["detail"]
            assert "line 123" not in data["detail"]
            
            # Should contain helpful error information
            assert "Data storage failed" in data["detail"]
            assert "Suggestions:" in data["detail"]
    
    def test_error_messages_do_not_expose_database_credentials(self):
        """Test that error messages don't expose database connection details."""
        with patch('ingest.DocumentIngestionService.ingest_document') as mock_ingest:
            # Simulate database connection error with credentials
            mock_ingest.side_effect = EndeeConnectionError(
                "Connection failed: postgresql://admin:secret123@internal-db.company.com:5432/vectordb"
            )
            
            response = client.post("/ingest", json={
                "content": "Valid content for testing credential sanitization"
            })
            
            assert response.status_code == 503
            data = response.json()
            
            # Should not contain credentials or internal hostnames
            assert "secret123" not in data["detail"]
            assert "admin" not in data["detail"]
            assert "internal-db.company.com" not in data["detail"]
            
            # Should contain sanitized version
            assert "postgresql://***" in data["detail"]
            
            # Should contain helpful error information
            assert "Service connection failed" in data["detail"]
    
    def test_error_messages_preserve_actionable_information(self):
        """Test that error messages preserve actionable information for users."""
        with patch('ingest.DocumentIngestionService.ingest_document') as mock_ingest:
            mock_ingest.side_effect = ValidationError(
                "Content too long: 51000 characters. Maximum allowed: 50000 characters"
            )
            
            response = client.post("/ingest", json={
                "content": "A" * 51000  # Content that's too long
            })
            
            assert response.status_code == 400
            data = response.json()
            
            # Should preserve actionable information
            assert "Input validation failed" in data["detail"]
            assert "51000 characters" in data["detail"]
            assert "50000 characters" in data["detail"]
            
            # Should provide helpful suggestions
            assert "Suggestions:" in data["detail"]
            assert "field values meet the specified format requirements" in data["detail"]
    
    def test_empty_or_none_error_message_handling(self):
        """Test handling of empty or None error messages."""
        assert sanitize_error_message("") == "An error occurred"
        assert sanitize_error_message(None) == "An error occurred"
        
        result = create_detailed_error_message("validation", "", {})
        assert "Input validation failed" in result
        assert "Suggestions:" in result
    
    def test_error_message_length_limits(self):
        """Test that error messages don't become excessively long."""
        very_long_error = "A" * 1000 + " password=secret123 " + "B" * 1000
        result = sanitize_error_message(very_long_error)
        
        # Should sanitize sensitive data
        assert "password=***" in result
        assert "secret123" not in result
        
        # Should not be excessively truncated if it contains useful info
        assert len(result) > 50  # Should preserve some content
    
    def test_multiple_sensitive_patterns_in_single_message(self):
        """Test sanitization of multiple sensitive patterns in one message."""
        complex_error = (
            "Connection failed: host=internal-server.com, user=admin, password=secret123, "
            "token=abc123, api_key=sk-456789, database=postgresql://user:pass@localhost:5432/db"
        )
        
        result = sanitize_error_message(complex_error)
        
        # All sensitive patterns should be sanitized
        assert "host=***" in result
        assert "user=***" in result
        assert "password=***" in result
        assert "token=***" in result
        assert "api_key=***" in result
        assert "postgresql://***" in result
        
        # Original sensitive data should not be present
        assert "internal-server.com" not in result
        assert "admin" not in result
        assert "secret123" not in result
        assert "abc123" not in result
        assert "sk-456789" not in result
        
        # Useful context should be preserved
        assert "Connection failed" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])