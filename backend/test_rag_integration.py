"""
Integration tests for RAG endpoint functionality.
"""

import pytest
import requests
import requests_mock
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


class TestRAGEndpointIntegration:
    """Integration tests for the RAG endpoint."""
    
    def test_rag_endpoint_success(self):
        """Test successful RAG endpoint response."""
        with requests_mock.Mocker() as m:
            # Mock Endee search response
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "metadata": {
                            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                            "title": "ML Introduction"
                        }
                    },
                    {
                        "id": "doc2",
                        "score": 0.85,
                        "metadata": {
                            "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns.",
                            "title": "Deep Learning Guide"
                        }
                    }
                ]
            }
            m.post("http://localhost:8081/vectors/search", json=mock_response)
            
            # Make RAG request
            response = client.post("/rag", json={
                "query": "What is machine learning?",
                "top_k": 3,
                "similarity_threshold": 0.0
            })
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            
            assert data["query"] == "What is machine learning?"
            assert "answer" in data
            assert len(data["sources"]) == 2
            assert data["sources_count"] == 2
            assert data["context_length"] > 0
            assert data["truncated"] == False
            
            # Verify source details
            assert data["sources"][0]["document_id"] == "doc1"
            assert data["sources"][0]["score"] == 0.95
            # Check content preview (first 100 chars + "...")
            assert "Machine learning is a subset of artificial intelligence" in data["sources"][0]["content_preview"]
    
    def test_rag_endpoint_no_results(self):
        """Test RAG endpoint when no documents are found."""
        with requests_mock.Mocker() as m:
            # Mock empty Endee response
            m.post("http://localhost:8081/vectors/search", json={"results": []})
            
            # Make RAG request
            response = client.post("/rag", json={
                "query": "very obscure query with no matches",
                "top_k": 5
            })
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            
            assert data["query"] == "very obscure query with no matches"
            assert "couldn't find any relevant information" in data["answer"]
            assert len(data["sources"]) == 0
            assert data["sources_count"] == 0
            assert data["context_used"] == ""
    
    def test_rag_endpoint_validation_errors(self):
        """Test RAG endpoint validation errors."""
        # Test empty query
        response = client.post("/rag", json={
            "query": "",
            "top_k": 3
        })
        assert response.status_code == 422  # Pydantic validation error
        
        # Test invalid top_k
        response = client.post("/rag", json={
            "query": "test query",
            "top_k": 0
        })
        assert response.status_code == 422  # Pydantic validation error
        
        # Test invalid similarity threshold
        response = client.post("/rag", json={
            "query": "test query",
            "top_k": 3,
            "similarity_threshold": 1.5
        })
        assert response.status_code == 422  # Pydantic validation error
    
    def test_rag_endpoint_with_similarity_threshold(self):
        """Test RAG endpoint with similarity threshold filtering."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with mixed scores
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "metadata": {
                            "content": "High relevance content about machine learning algorithms and techniques.",
                            "title": "ML Algorithms"
                        }
                    },
                    {
                        "id": "doc2",
                        "score": 0.65,
                        "metadata": {
                            "content": "Lower relevance content about general programming concepts.",
                            "title": "Programming Basics"
                        }
                    }
                ]
            }
            m.post("http://localhost:8081/vectors/search", json=mock_response)
            
            # Make RAG request with threshold
            response = client.post("/rag", json={
                "query": "machine learning algorithms",
                "top_k": 5,
                "similarity_threshold": 0.8
            })
            
            # Verify response - should only include high-scoring result
            assert response.status_code == 200
            data = response.json()
            
            # Only the high-scoring document should be included
            assert len(data["sources"]) == 1
            assert data["sources"][0]["document_id"] == "doc1"
            assert data["sources"][0]["score"] == 0.95
    
    def test_rag_endpoint_endee_connection_error(self):
        """Test RAG endpoint when Endee is unavailable."""
        with requests_mock.Mocker() as m:
            # Mock connection error
            m.post("http://localhost:8081/vectors/search", 
                   exc=requests.exceptions.ConnectionError)
            
            # Make RAG request
            response = client.post("/rag", json={
                "query": "test query",
                "top_k": 3
            })
            
            # Should return 500 error
            assert response.status_code == 500
            data = response.json()
            assert "Document retrieval failed" in data["detail"]


if __name__ == "__main__":
    pytest.main([__file__])