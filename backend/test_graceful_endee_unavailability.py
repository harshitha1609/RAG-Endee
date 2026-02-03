# Test graceful handling of Endee service unavailability
# Tests for circuit breaker pattern and service unavailability handling

import asyncio
import pytest
import requests
import requests_mock
import aiohttp
import time
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app import app
from endee_client import EndeeHTTPClient, ServiceUnavailableError, CircuitBreakerState
from ingest import DocumentIngestionService
from search import VectorSearchService

client = TestClient(app)

class TestGracefulEndeeUnavailability:
    """Test suite for graceful handling of Endee service unavailability."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_vector = [0.1] * 384
        self.test_document = {
            "content": "Test document for unavailability testing",
            "metadata": {"title": "Test Document"}
        }
        self.test_query = "test query for unavailability"
    
    def test_circuit_breaker_initialization(self):
        """Test that circuit breaker is properly initialized."""
        client = EndeeHTTPClient(
            endee_url="http://test-endee:8080",
            circuit_breaker_enabled=True,
            circuit_breaker_failure_threshold=3,
            circuit_breaker_recovery_timeout=30.0
        )
        
        assert client.circuit_breaker_enabled is True
        assert client.circuit_breaker is not None
        assert client.circuit_breaker.failure_threshold == 3
        assert client.circuit_breaker.recovery_timeout == 30.0
        assert client.circuit_breaker.state == "CLOSED"
    
    def test_circuit_breaker_disabled(self):
        """Test circuit breaker can be disabled."""
        client = EndeeHTTPClient(
            endee_url="http://test-endee:8080",
            circuit_breaker_enabled=False
        )
        
        assert client.circuit_breaker_enabled is False
        assert client.circuit_breaker is None
    
    def test_circuit_breaker_state_transitions(self):
        """Test circuit breaker state transitions."""
        breaker = CircuitBreakerState(failure_threshold=2, recovery_timeout=1.0)
        
        # Initial state should be CLOSED
        assert breaker.state == "CLOSED"
        assert breaker.can_execute() is True
        
        # Record failures to open circuit
        breaker.record_failure()
        assert breaker.state == "CLOSED"  # Still closed after 1 failure
        
        breaker.record_failure()
        assert breaker.state == "OPEN"  # Open after 2 failures
        assert breaker.can_execute() is False
        
        # Wait for recovery timeout and check HALF_OPEN state
        import time
        time.sleep(1.1)  # Wait longer than recovery timeout
        assert breaker.can_execute() is True
        assert breaker.state == "HALF_OPEN"
        
        # Record success to close circuit
        breaker.record_success()
        breaker.record_success()
        breaker.record_success()  # Need 3 successes in half-open
        assert breaker.state == "CLOSED"
    
    @pytest.mark.asyncio
    async def test_service_unavailable_error_raised(self):
        """Test that ServiceUnavailableError is raised when circuit is open."""
        client = EndeeHTTPClient(
            endee_url="http://test-endee:8080",
            circuit_breaker_enabled=True,
            circuit_breaker_failure_threshold=1
        )
        
        # Force circuit breaker to open state
        client.circuit_breaker.state = "OPEN"
        client.circuit_breaker.failure_count = 5
        client.circuit_breaker.last_failure_time = time.time() - 10  # 10 seconds ago
        
        with pytest.raises(ServiceUnavailableError) as exc_info:
            await client.add_vector_async("test-doc", self.test_vector)
        
        assert "circuit breaker is OPEN" in str(exc_info.value)
        assert exc_info.value.consecutive_failures == 5
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_records_failures(self):
        """Test that circuit breaker records failures correctly."""
        client = EndeeHTTPClient(
            endee_url="http://test-endee:8080",
            circuit_breaker_enabled=True,
            circuit_breaker_failure_threshold=2
        )
        
        with patch('endee_client.aiohttp.ClientSession.post') as mock_post:
            # Mock connection errors
            mock_post.side_effect = aiohttp.ClientConnectorError(
                connection_key=None, os_error=OSError("Connection refused")
            )
            
            # First failure - circuit should remain closed
            with pytest.raises(Exception):
                await client.add_vector_async("test-doc", self.test_vector)
            
            assert client.circuit_breaker.state == "CLOSED"
            assert client.circuit_breaker.failure_count == 1
            
            # Second failure - circuit should open
            with pytest.raises(Exception):
                await client.add_vector_async("test-doc", self.test_vector)
            
            assert client.circuit_breaker.state == "OPEN"
            assert client.circuit_breaker.failure_count == 2
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self):
        """Test the Endee health check endpoint."""
        with patch('app.ingestion_service.endee_client.check_service_availability') as mock_health:
            # Mock successful health check
            mock_health.return_value = {
                "available": True,
                "response_time_ms": 50.0,
                "status_code": 200,
                "endee_url": "http://localhost:8081",
                "check_time": 1234567890.0
            }
            
            response = client.get("/health/endee")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert "endee_service" in data
            assert "circuit_breaker" in data
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint_service_down(self):
        """Test health check endpoint when Endee service is down."""
        with patch('app.ingestion_service.endee_client.check_service_availability') as mock_health:
            # Mock service unavailable
            mock_health.side_effect = ServiceUnavailableError(
                "Endee service health check failed: Connection refused",
                consecutive_failures=3
            )
            
            response = client.get("/health/endee")
            assert response.status_code == 200  # Endpoint should still respond
            
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "error" in data
            assert "circuit_breaker" in data
    
    def test_circuit_breaker_reset_endpoint(self):
        """Test the circuit breaker reset endpoint."""
        response = client.post("/admin/endee/circuit-breaker/reset")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "ingestion_service" in data
        assert "search_service" in data
        assert "message" in data
    
    def test_ingest_endpoint_service_unavailable(self):
        """Test ingest endpoint with service unavailable error."""
        with patch('ingest.DocumentIngestionService.ingest_document') as mock_ingest:
            mock_ingest.side_effect = ServiceUnavailableError(
                "Service unavailable - circuit breaker is OPEN",
                consecutive_failures=5
            )
            
            response = client.post("/ingest", json=self.test_document)
            assert response.status_code == 503
            
            data = response.json()
            assert "temporarily unavailable" in data["detail"]
    
    def test_search_endpoint_service_unavailable(self):
        """Test search endpoint with service unavailable error."""
        with patch('search.VectorSearchService.search') as mock_search:
            mock_search.side_effect = ServiceUnavailableError(
                "Service unavailable - circuit breaker is OPEN",
                consecutive_failures=3
            )
            
            response = client.post("/search", json={"query": self.test_query})
            assert response.status_code == 503
            
            data = response.json()
            assert "temporarily unavailable" in data["detail"]
    
    def test_rag_endpoint_service_unavailable(self):
        """Test RAG endpoint with service unavailable error."""
        with patch('search.VectorSearchService.search') as mock_search:
            mock_search.side_effect = ServiceUnavailableError(
                "Service unavailable - circuit breaker is OPEN",
                consecutive_failures=4
            )
            
            response = client.post("/rag", json={"query": self.test_query})
            assert response.status_code == 500  # RAG converts search errors to 500
            
            data = response.json()
            assert "RAG pipeline failed" in data["detail"]
    
    def test_performance_stats_include_circuit_breaker(self):
        """Test that performance stats include circuit breaker information."""
        client = EndeeHTTPClient(
            endee_url="http://test-endee:8080",
            circuit_breaker_enabled=True
        )
        
        stats = client.get_performance_stats()
        assert "circuit_breaker" in stats
        assert "service_health" in stats
        
        circuit_stats = stats["circuit_breaker"]
        assert "state" in circuit_stats
        assert "failure_count" in circuit_stats
        assert "failure_threshold" in circuit_stats
    
    def test_service_health_information(self):
        """Test service health information retrieval."""
        client = EndeeHTTPClient(
            endee_url="http://test-endee:8080",
            circuit_breaker_enabled=True
        )
        
        health = client.get_service_health()
        assert "endee_url" in health
        assert "circuit_breaker_enabled" in health
        assert "service_available" in health
        assert "last_check_time" in health
        assert "circuit_breaker" in health
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_multiple_failures(self):
        """Test graceful degradation with multiple consecutive failures."""
        client = EndeeHTTPClient(
            endee_url="http://test-endee:8080",
            circuit_breaker_enabled=True,
            circuit_breaker_failure_threshold=3
        )
        
        with patch('endee_client.aiohttp.ClientSession.post') as mock_post:
            # Mock connection errors for multiple attempts
            mock_post.side_effect = aiohttp.ClientConnectorError(
                connection_key=None, os_error=OSError("Connection refused")
            )
            
            # Make multiple requests that should fail
            for i in range(5):
                try:
                    await client.add_vector_async(f"test-doc-{i}", self.test_vector)
                except Exception:
                    pass  # Expected to fail
            
            # Circuit should be open after 3 failures
            assert client.circuit_breaker.state == "OPEN"
            
            # Next request should be blocked by circuit breaker
            with pytest.raises(ServiceUnavailableError):
                await client.add_vector_async("blocked-doc", self.test_vector)
    
    def test_error_messages_do_not_expose_internals(self):
        """Test that error messages don't expose internal system details."""
        with patch('ingest.DocumentIngestionService.ingest_document') as mock_ingest:
            # Simulate internal error with sensitive information
            mock_ingest.side_effect = ServiceUnavailableError(
                "Database connection failed: host=internal-db-server, user=admin, password=secret123"
            )
            
            response = client.post("/ingest", json=self.test_document)
            assert response.status_code == 503
            
            data = response.json()
            # Error message should be sanitized
            assert "temporarily unavailable" in data["detail"]
            # Should not contain sensitive internal details
            assert "secret123" not in data["detail"]
            assert "internal-db-server" not in data["detail"]
            assert "admin" not in data["detail"]
            # Should contain sanitized versions
            assert "password=***" in data["detail"]
            assert "host=***" in data["detail"]
            assert "user=***" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_after_service_restoration(self):
        """Test circuit breaker recovery when service comes back online."""
        client = EndeeHTTPClient(
            endee_url="http://test-endee:8080",
            circuit_breaker_enabled=True,
            circuit_breaker_failure_threshold=2,
            circuit_breaker_recovery_timeout=0.1  # Short timeout for testing
        )
        
        with patch('endee_client.aiohttp.ClientSession.post') as mock_post:
            # First, cause failures to open circuit
            mock_post.side_effect = aiohttp.ClientConnectorError(
                connection_key=None, os_error=OSError("Connection refused")
            )
            
            # Cause 2 failures to open circuit
            for _ in range(2):
                try:
                    await client.add_vector_async("test-doc", self.test_vector)
                except Exception:
                    pass
            
            assert client.circuit_breaker.state == "OPEN"
            
            # Wait for recovery timeout
            await asyncio.sleep(0.2)
            
            # Now mock successful responses
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"status": "success", "id": "test-doc"}
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = None
            
            mock_post.side_effect = None
            mock_post.return_value = mock_response
            
            # Circuit should allow requests in HALF_OPEN state
            result = await client.add_vector_async("recovery-doc", self.test_vector)
            assert result["status"] == "success"
            
            # After enough successes, circuit should close
            for _ in range(2):  # Need multiple successes
                await client.add_vector_async(f"success-doc", self.test_vector)
            
            assert client.circuit_breaker.state == "CLOSED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])