"""
Integration tests for all API endpoints with valid inputs.
Tests that all endpoints respond correctly to valid inputs.
"""

import pytest
import requests_mock
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


class TestAPIEndpointsValidInputs:
    """Test all API endpoints with valid inputs to ensure they respond correctly."""
    
    def test_root_endpoint_valid(self):
        """Test root endpoint responds correctly."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "Endee RAG System API"
        assert data["version"] == "1.0.0"
        assert data["docs"] == "/docs"
    
    def test_health_endpoint_valid(self):
        """Test health endpoint responds correctly."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data
        # Verify timestamp is a valid ISO format string
        from datetime import datetime
        datetime.fromisoformat(data["timestamp"])
    
    def test_ingest_endpoint_valid_minimal(self):
        """Test ingest endpoint with minimal valid input."""
        with requests_mock.Mocker() as m:
            # Mock successful Endee add vector response
            m.post("http://localhost:8081/vectors/add", json={
                "status": "success",
                "id": "test-doc-id"
            })
            
            response = client.post("/ingest", json={
                "content": "This is a test document with valid content."
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert "document_id" in data
            assert data["embedding_dimension"] == 384
            assert data["content_length"] > 0
            assert isinstance(data["metadata"], dict)
    
    def test_ingest_endpoint_valid_with_metadata(self):
        """Test ingest endpoint with valid content and metadata."""
        with requests_mock.Mocker() as m:
            # Mock successful Endee add vector response
            m.post("http://localhost:8081/vectors/add", json={
                "status": "success",
                "id": "test-doc-id"
            })
            
            response = client.post("/ingest", json={
                "content": "This is a comprehensive test document with detailed content for testing the ingestion endpoint.",
                "metadata": {
                    "title": "Test Document",
                    "author": "Test Author",
                    "category": "Testing",
                    "tags": "test,document,api",  # Use string instead of list
                    "word_count": 15,
                    "published": True
                }
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert "document_id" in data
            assert data["embedding_dimension"] == 384
            assert data["content_length"] > 0
            assert data["metadata"]["title"] == "Test Document"
            assert data["metadata"]["author"] == "Test Author"
            assert data["metadata"]["category"] == "Testing"
            assert data["metadata"]["tags"] == "test,document,api"
            assert data["metadata"]["word_count"] == 15
            assert data["metadata"]["published"] == True
    
    def test_ingest_endpoint_valid_long_content(self):
        """Test ingest endpoint with valid long content."""
        with requests_mock.Mocker() as m:
            # Mock successful Endee add vector response
            m.post("http://localhost:8081/vectors/add", json={
                "status": "success",
                "id": "test-doc-id"
            })
            
            # Create long but valid content (under 50000 chars)
            long_content = "This is a test document. " * 500  # ~12500 chars
            
            response = client.post("/ingest", json={
                "content": long_content,
                "metadata": {
                    "title": "Long Test Document",
                    "type": "comprehensive"
                }
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert "document_id" in data
            assert data["embedding_dimension"] == 384
            assert data["content_length"] == len(long_content.strip())
            assert data["metadata"]["title"] == "Long Test Document"
    
    def test_search_endpoint_valid_minimal(self):
        """Test search endpoint with minimal valid input."""
        with requests_mock.Mocker() as m:
            # Mock successful Endee search response
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "metadata": {
                            "content": "Machine learning is a subset of artificial intelligence.",
                            "title": "ML Introduction"
                        }
                    }
                ]
            }
            m.post("http://localhost:8081/vectors/search", json=mock_response)
            
            response = client.post("/search", json={
                "query": "machine learning"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["query"] == "machine learning"
            assert len(data["results"]) == 1
            assert data["total_results"] == 1
            assert data["similarity_threshold"] == 0.0
            
            # Verify result structure
            result = data["results"][0]
            assert result["document_id"] == "doc1"
            assert result["score"] == 0.95
            assert result["content"] == "Machine learning is a subset of artificial intelligence."
            assert result["metadata"]["title"] == "ML Introduction"
    
    def test_search_endpoint_valid_with_parameters(self):
        """Test search endpoint with valid parameters."""
        with requests_mock.Mocker() as m:
            # Mock successful Endee search response with multiple results
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "metadata": {
                            "content": "Deep learning is a subset of machine learning.",
                            "title": "Deep Learning Guide",
                            "category": "AI"
                        }
                    },
                    {
                        "id": "doc2",
                        "score": 0.85,
                        "metadata": {
                            "content": "Neural networks are the foundation of deep learning.",
                            "title": "Neural Networks",
                            "category": "AI"
                        }
                    },
                    {
                        "id": "doc3",
                        "score": 0.75,
                        "metadata": {
                            "content": "Artificial intelligence encompasses many techniques.",
                            "title": "AI Overview",
                            "category": "AI"
                        }
                    }
                ]
            }
            m.post("http://localhost:8081/vectors/search", json=mock_response)
            
            response = client.post("/search", json={
                "query": "deep learning neural networks",
                "top_k": 10,
                "similarity_threshold": 0.7
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["query"] == "deep learning neural networks"
            assert len(data["results"]) == 3  # All above threshold
            assert data["total_results"] == 3
            assert data["similarity_threshold"] == 0.7
            
            # Verify all results are above threshold
            for result in data["results"]:
                assert result["score"] >= 0.7
                assert "document_id" in result
                assert "content" in result
                assert "metadata" in result
    
    def test_search_endpoint_valid_with_threshold_filtering(self):
        """Test search endpoint with similarity threshold filtering."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with mixed scores
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "metadata": {
                            "content": "High relevance content about machine learning.",
                            "title": "ML Guide"
                        }
                    },
                    {
                        "id": "doc2",
                        "score": 0.65,
                        "metadata": {
                            "content": "Lower relevance content about programming.",
                            "title": "Programming Guide"
                        }
                    }
                ]
            }
            m.post("http://localhost:8081/vectors/search", json=mock_response)
            
            response = client.post("/search", json={
                "query": "machine learning",
                "top_k": 5,
                "similarity_threshold": 0.8
            })
            
            assert response.status_code == 200
            data = response.json()
            
            # Only high-scoring result should be included
            assert len(data["results"]) == 1
            assert data["results"][0]["document_id"] == "doc1"
            assert data["results"][0]["score"] == 0.95
    
    def test_rag_endpoint_valid_minimal(self):
        """Test RAG endpoint with minimal valid input."""
        with requests_mock.Mocker() as m:
            # Mock successful Endee search response
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "metadata": {
                            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
                            "title": "ML Introduction"
                        }
                    }
                ]
            }
            m.post("http://localhost:8081/vectors/search", json=mock_response)
            
            response = client.post("/rag", json={
                "query": "What is machine learning?"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["query"] == "What is machine learning?"
            assert "answer" in data
            assert len(data["sources"]) == 1
            assert data["sources_count"] == 1
            assert data["context_length"] > 0
            assert data["truncated"] == False
            assert data["max_context_length"] == 4000  # Default value
            
            # Verify source structure
            source = data["sources"][0]
            assert source["document_id"] == "doc1"
            assert source["score"] == 0.95
            assert "content_preview" in source
            assert source["metadata"]["title"] == "ML Introduction"
    
    def test_rag_endpoint_valid_with_parameters(self):
        """Test RAG endpoint with valid parameters."""
        with requests_mock.Mocker() as m:
            # Mock successful Endee search response with multiple results
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "metadata": {
                            "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
                            "title": "Deep Learning Guide",
                            "category": "AI"
                        }
                    },
                    {
                        "id": "doc2",
                        "score": 0.88,
                        "metadata": {
                            "content": "Neural networks are computational models inspired by biological neural networks.",
                            "title": "Neural Networks Explained",
                            "category": "AI"
                        }
                    },
                    {
                        "id": "doc3",
                        "score": 0.82,
                        "metadata": {
                            "content": "Convolutional neural networks are particularly effective for image recognition tasks.",
                            "title": "CNN Guide",
                            "category": "Computer Vision"
                        }
                    }
                ]
            }
            m.post("http://localhost:8081/vectors/search", json=mock_response)
            
            response = client.post("/rag", json={
                "query": "How do neural networks work in deep learning?",
                "top_k": 5,
                "similarity_threshold": 0.8,
                "max_context_length": 2000
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["query"] == "How do neural networks work in deep learning?"
            assert "answer" in data
            assert len(data["sources"]) == 3  # All above threshold
            assert data["sources_count"] == 3
            assert data["context_length"] > 0
            assert data["max_context_length"] == 2000
            
            # Verify all sources are above threshold
            for source in data["sources"]:
                assert source["score"] >= 0.8
                assert "document_id" in source
                assert "content_preview" in source
                assert "metadata" in source
    
    def test_rag_endpoint_valid_with_context_truncation(self):
        """Test RAG endpoint with context length limits."""
        with requests_mock.Mocker() as m:
            # Mock response with long content that should trigger truncation
            long_content = "This is a very long document content. " * 100  # ~3800 chars
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "metadata": {
                            "content": long_content,
                            "title": "Long Document 1"
                        }
                    },
                    {
                        "id": "doc2",
                        "score": 0.90,
                        "metadata": {
                            "content": long_content,
                            "title": "Long Document 2"
                        }
                    }
                ]
            }
            m.post("http://localhost:8081/vectors/search", json=mock_response)
            
            response = client.post("/rag", json={
                "query": "Tell me about these documents",
                "top_k": 5,
                "max_context_length": 1000  # Small limit to trigger truncation
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["query"] == "Tell me about these documents"
            assert "answer" in data
            assert data["context_length"] <= 1000  # Should respect limit
            assert data["max_context_length"] == 1000
            # May or may not be truncated depending on actual content size
            assert isinstance(data["truncated"], bool)
    
    def test_all_endpoints_return_json_content_type(self):
        """Test that all endpoints return proper JSON content type."""
        with requests_mock.Mocker() as m:
            # Mock Endee responses for endpoints that need them
            m.post("http://localhost:8081/vectors/add", json={"status": "success", "id": "test"})
            m.post("http://localhost:8081/vectors/search", json={"results": []})
            
            # Test root endpoint
            response = client.get("/")
            assert response.status_code == 200
            assert "application/json" in response.headers["content-type"]
            
            # Test health endpoint
            response = client.get("/health")
            assert response.status_code == 200
            assert "application/json" in response.headers["content-type"]
            
            # Test ingest endpoint
            response = client.post("/ingest", json={"content": "test content"})
            assert response.status_code == 200
            assert "application/json" in response.headers["content-type"]
            
            # Test search endpoint
            response = client.post("/search", json={"query": "test query"})
            assert response.status_code == 200
            assert "application/json" in response.headers["content-type"]
            
            # Test RAG endpoint
            response = client.post("/rag", json={"query": "test query"})
            assert response.status_code == 200
            assert "application/json" in response.headers["content-type"]
    
    def test_endpoints_handle_unicode_content(self):
        """Test that endpoints properly handle Unicode content."""
        with requests_mock.Mocker() as m:
            # Mock Endee responses
            m.post("http://localhost:8081/vectors/add", json={"status": "success", "id": "test"})
            m.post("http://localhost:8081/vectors/search", json={
                "results": [{
                    "id": "doc1",
                    "score": 0.95,
                    "metadata": {
                        "content": "Contenu en français avec des caractères spéciaux: àéèùç",
                        "title": "Document Français"
                    }
                }]
            })
            
            # Test ingest with Unicode content
            unicode_content = "This document contains Unicode: 你好世界, café, naïve, résumé, Москва"
            response = client.post("/ingest", json={
                "content": unicode_content,
                "metadata": {"title": "Unicode Test Document"}
            })
            assert response.status_code == 200
            
            # Test search with Unicode query
            response = client.post("/search", json={
                "query": "café résumé français"
            })
            assert response.status_code == 200
            
            # Test RAG with Unicode query
            response = client.post("/rag", json={
                "query": "Qu'est-ce que l'intelligence artificielle?"
            })
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])