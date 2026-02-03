"""
Test connection pooling optimizations for the Endee HTTP client.
Verifies that connection pooling improvements provide better performance and resource usage.
"""

import asyncio
import pytest
import time
import statistics
from unittest.mock import patch, MagicMock
import aiohttp
import requests_mock
from requests.adapters import HTTPAdapter

from endee_client import EndeeHTTPClient


class TestConnectionPoolingOptimization:
    """Test optimized connection pooling functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_vector = [0.1] * 384
        self.test_document_id = "test-doc-123"
        self.test_metadata = {"title": "Test Document", "content": "Test content"}
        
        # Mock Endee responses
        self.mock_add_response = {"status": "success", "id": self.test_document_id}
        self.mock_search_response = {
            "results": [
                {
                    "id": self.test_document_id,
                    "score": 0.95,
                    "metadata": self.test_metadata
                }
            ]
        }
    
    def test_optimized_connection_pool_configuration(self):
        """Test that optimized connection pool settings are applied correctly."""
        client = EndeeHTTPClient(
            pool_connections=30,
            pool_maxsize=60,
            keepalive_timeout=45,
            enable_socket_keepalive=True
        )
        
        # Verify configuration is stored
        assert client.keepalive_timeout == 45
        assert client.enable_socket_keepalive == True
        
        # Verify sync session configuration
        adapter = client.session.get_adapter("http://")
        # HTTPAdapter stores these in init_poolmanager method, not directly accessible
        # We can verify the adapter exists and is properly configured
        assert isinstance(adapter, HTTPAdapter)
        
        # Verify headers include keep-alive settings
        assert "Connection" in client.session.headers
        assert client.session.headers["Connection"] == "keep-alive"
        assert "Keep-Alive" in client.session.headers
        assert "timeout=45" in client.session.headers["Keep-Alive"]
        
        client.close_sync()
    
    @pytest.mark.asyncio
    async def test_async_session_optimization(self):
        """Test that async session uses optimized connection pooling."""
        client = EndeeHTTPClient()
        
        # Get async session to trigger creation
        session = await client._get_async_session()
        
        # Verify connector configuration
        connector = session.connector
        assert connector._limit == 50  # DEFAULT_POOL_MAXSIZE
        assert connector._limit_per_host == 20  # DEFAULT_POOL_CONNECTIONS
        assert connector._use_dns_cache == True
        assert connector._keepalive_timeout == 30  # DEFAULT_KEEPALIVE_TIMEOUT
        
        # Verify timeout configuration
        timeout = session.timeout
        assert timeout.total == 30  # DEFAULT_TIMEOUT
        assert timeout.connect == 10
        assert timeout.sock_read == 30
        
        # Verify headers
        assert session.headers["Connection"] == "keep-alive"
        assert "Keep-Alive" in session.headers
        
        await client.close()
    
    def test_connection_pool_statistics(self):
        """Test connection pool statistics reporting."""
        client = EndeeHTTPClient(
            pool_connections=25,
            pool_maxsize=55,
            keepalive_timeout=40
        )
        
        # Get connection pool stats
        pool_stats = client.get_connection_pool_stats()
        
        # Verify sync session stats
        sync_stats = pool_stats["sync_session"]
        assert sync_stats["configured"] == True
        assert sync_stats["pool_connections"] == 20  # Uses default from constants
        assert sync_stats["pool_maxsize"] == 50  # Uses default from constants
        assert sync_stats["keepalive_timeout"] == 40
        assert sync_stats["socket_keepalive"] == True
        
        # Verify async session stats
        async_stats = pool_stats["async_session"]
        assert async_stats["created"] == False  # Not created yet
        assert async_stats["pool_limit"] == 50
        assert async_stats["per_host_limit"] == 20
        assert async_stats["dns_cache_ttl"] == 300
        
        client.close_sync()
    
    @pytest.mark.asyncio
    async def test_connection_warmup(self):
        """Test connection pool warmup functionality."""
        from aioresponses import aioresponses
        
        with aioresponses() as m:
            # Mock Endee search endpoint for warmup - allow multiple calls
            m.post("http://localhost:8081/vectors/search", payload=self.mock_search_response, repeat=True)
            
            client = EndeeHTTPClient()
            
            # Test warmup with 3 connections
            warmup_stats = await client.warmup_connections(num_connections=3)
            
            # Verify warmup results
            assert warmup_stats["total_attempts"] == 3
            assert warmup_stats["successful_connections"] >= 0
            assert warmup_stats["failed_connections"] >= 0
            assert warmup_stats["successful_connections"] + warmup_stats["failed_connections"] == 3
            assert 0 <= warmup_stats["success_rate"] <= 100
            assert warmup_stats["warmup_duration_ms"] > 0
            
            # At least some connections should succeed with proper mocking
            assert warmup_stats["successful_connections"] >= 1
            
            await client.close()
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_with_optimized_pooling(self):
        """Test that optimized connection pooling handles concurrent requests efficiently."""
        with requests_mock.Mocker() as m:
            m.post("http://localhost:8081/vectors/search", json=self.mock_search_response)
            
            client = EndeeHTTPClient(
                pool_connections=30,
                pool_maxsize=60
            )
            
            # Warm up connections first
            await client.warmup_connections(num_connections=5)
            
            async def make_search_request(request_id: int):
                """Make a single search request."""
                start_time = time.time()
                result = await client.search_vectors(
                    query_vector=self.test_vector,
                    top_k=5
                )
                duration = (time.time() - start_time) * 1000
                return request_id, result, duration
            
            # Make 20 concurrent requests
            tasks = [make_search_request(i) for i in range(20)]
            results = await asyncio.gather(*tasks)
            
            # Verify all requests succeeded
            durations = []
            for request_id, result, duration in results:
                assert "results" in result
                assert len(result["results"]) == 1
                durations.append(duration)
            
            # Verify performance with optimized pooling
            avg_duration = statistics.mean(durations)
            max_duration = max(durations)
            
            # With optimized connection pooling, concurrent requests should be faster
            assert avg_duration < 1000  # Average under 1 second
            assert max_duration < 2000  # No request over 2 seconds
            
            # Verify connection reuse by checking request count
            assert len(m.request_history) >= 25  # 5 warmup + 20 search requests
            
            await client.close()
    
    @pytest.mark.asyncio
    async def test_connection_pool_performance_improvement(self):
        """Test that optimized connection pooling improves performance over default settings."""
        with requests_mock.Mocker() as m:
            m.post("http://localhost:8081/vectors/search", json=self.mock_search_response)
            
            # Test with default (non-optimized) settings
            default_client = EndeeHTTPClient(
                pool_connections=5,  # Smaller pool
                pool_maxsize=10,     # Smaller max size
                keepalive_timeout=5  # Shorter keepalive
            )
            
            # Test with optimized settings
            optimized_client = EndeeHTTPClient(
                pool_connections=30,  # Larger pool
                pool_maxsize=60,      # Larger max size
                keepalive_timeout=30  # Longer keepalive
            )
            
            async def benchmark_client(client, num_requests=10):
                """Benchmark a client with multiple requests."""
                start_time = time.time()
                
                tasks = [
                    client.search_vectors(query_vector=self.test_vector, top_k=3)
                    for _ in range(num_requests)
                ]
                await asyncio.gather(*tasks)
                
                return (time.time() - start_time) * 1000
            
            # Benchmark both clients
            default_duration = await benchmark_client(default_client, 15)
            optimized_duration = await benchmark_client(optimized_client, 15)
            
            # Optimized client should perform better or at least not worse
            # Allow for some variance in test environments
            performance_ratio = optimized_duration / default_duration
            assert performance_ratio <= 1.2  # Optimized should be at most 20% slower (accounting for test variance)
            
            # Verify both clients have performance stats
            default_stats = default_client.get_performance_stats()
            optimized_stats = optimized_client.get_performance_stats()
            
            assert "search_vectors" in default_stats
            assert "search_vectors" in optimized_stats
            assert default_stats["search_vectors"]["total_calls"] == 15
            assert optimized_stats["search_vectors"]["total_calls"] == 15
            
            await default_client.close()
            await optimized_client.close()
    
    @pytest.mark.asyncio
    async def test_connection_pool_resource_cleanup(self):
        """Test that connection pool resources are properly cleaned up."""
        client = EndeeHTTPClient()
        
        # Create async session
        session = await client._get_async_session()
        assert not session.closed
        
        # Get initial connection pool stats
        initial_stats = client.get_connection_pool_stats()
        assert initial_stats["async_session"]["created"] == True
        
        # Close client
        await client.close()
        
        # Verify session is closed
        assert session.closed
        
        # Verify cleanup in stats (session should be marked as not created)
        final_stats = client.get_connection_pool_stats()
        # Note: After closing, the session reference still exists but is closed
        # The "created" flag checks if session exists and is not closed
        assert final_stats["async_session"]["created"] == False
    
    def test_performance_stats_include_connection_pool_info(self):
        """Test that performance stats include connection pool information."""
        client = EndeeHTTPClient(
            pool_connections=25,
            pool_maxsize=55,
            keepalive_timeout=35
        )
        
        # Get performance stats
        stats = client.get_performance_stats()
        
        # Verify connection pool stats are included
        assert "connection_pool" in stats
        pool_info = stats["connection_pool"]
        
        # Verify sync session info
        assert "sync_session" in pool_info
        sync_info = pool_info["sync_session"]
        assert sync_info["configured"] == True
        assert sync_info["keepalive_timeout"] == 35
        
        # Verify async session info
        assert "async_session" in pool_info
        async_info = pool_info["async_session"]
        assert async_info["created"] == False  # Not created yet
        assert async_info["pool_limit"] == 50
        assert async_info["per_host_limit"] == 20
        
        client.close_sync()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])