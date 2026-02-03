"""
Full API integration test with running mock Endee service.
This test verifies the complete API functionality end-to-end.
"""

import pytest
import time
import requests
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


class TestFullAPIIntegration:
    """Test complete API integration with real mock Endee service."""
    
    def setup_method(self):
        """Set up test fixtures."""
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
    
    def test_health_endpoint(self):
        """Test the API health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_complete_workflow_ingest_search_rag(self):
        """Test the complete workflow: ingest -> search -> RAG."""
        try:
            # Step 1: Ingest a document
            ingest_response = client.post("/ingest", json={
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions.",
                "metadata": {
                    "title": "Machine Learning Introduction",
                    "author": "AI Expert",
                    "category": "AI/ML",
                    "source": "educational_content.txt"
                }
            })
            
            assert ingest_response.status_code == 200
            ingest_data = ingest_response.json()
            assert ingest_data["status"] == "success"
            assert "document_id" in ingest_data
            assert ingest_data["embedding_dimension"] == 384
            
            document_id = ingest_data["document_id"]
            
            # Step 2: Search for the document
            search_response = client.post("/search", json={
                "query": "What is machine learning?",
                "top_k": 5
            })
            
            assert search_response.status_code == 200
            search_data = search_response.json()
            assert "results" in search_data
            assert len(search_data["results"]) >= 1
            
            # Find our document in the results
            found_doc = None
            for result in search_data["results"]:
                if result["document_id"] == document_id:
                    found_doc = result
                    break
            
            assert found_doc is not None, "Ingested document not found in search results"
            assert found_doc["metadata"]["title"] == "Machine Learning Introduction"
            assert "score" in found_doc
            
            # Step 3: Use RAG to get an answer
            rag_response = client.post("/rag", json={
                "query": "What is machine learning and how does it work?",
                "top_k": 3
            })
            
            assert rag_response.status_code == 200
            rag_data = rag_response.json()
            assert rag_data["query"] == "What is machine learning and how does it work?"
            assert "answer" in rag_data
            assert len(rag_data["sources"]) >= 1
            assert rag_data["sources_count"] >= 1
            
            # Verify our document is in the RAG sources
            found_in_rag = False
            for source in rag_data["sources"]:
                if source["document_id"] == document_id:
                    found_in_rag = True
                    break
            
            assert found_in_rag, "Ingested document not found in RAG sources"
            
        except requests.exceptions.RequestException:
            pytest.skip("Mock Endee service is not available")
    
    def test_multiple_documents_workflow(self):
        """Test workflow with multiple documents."""
        try:
            # Ingest multiple documents
            documents = [
                {
                    "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.",
                    "metadata": {"title": "Deep Learning Basics", "category": "AI/ML"}
                },
                {
                    "content": "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language.",
                    "metadata": {"title": "NLP Introduction", "category": "AI/ML"}
                },
                {
                    "content": "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world.",
                    "metadata": {"title": "Computer Vision Overview", "category": "AI/ML"}
                }
            ]
            
            document_ids = []
            
            # Ingest all documents
            for doc in documents:
                response = client.post("/ingest", json=doc)
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                document_ids.append(data["document_id"])
            
            # Search for AI-related content
            search_response = client.post("/search", json={
                "query": "artificial intelligence and neural networks",
                "top_k": 5
            })
            
            assert search_response.status_code == 200
            search_data = search_response.json()
            assert len(search_data["results"]) >= 2  # Should find multiple relevant docs
            
            # Use RAG with the multiple documents
            rag_response = client.post("/rag", json={
                "query": "Explain the different areas of artificial intelligence",
                "top_k": 5
            })
            
            assert rag_response.status_code == 200
            rag_data = rag_response.json()
            assert "answer" in rag_data
            assert len(rag_data["sources"]) >= 2  # Should use multiple sources
            
        except requests.exceptions.RequestException:
            pytest.skip("Mock Endee service is not available")
    
    def test_search_with_similarity_threshold(self):
        """Test search with similarity threshold filtering."""
        try:
            # Ingest a specific document
            ingest_response = client.post("/ingest", json={
                "content": "Quantum computing uses quantum mechanical phenomena to perform calculations that would be impossible for classical computers.",
                "metadata": {"title": "Quantum Computing", "category": "Quantum Physics"}
            })
            
            assert ingest_response.status_code == 200
            document_id = ingest_response.json()["document_id"]
            
            # Search with high similarity threshold
            search_response = client.post("/search", json={
                "query": "quantum computing and quantum mechanics",
                "top_k": 5,
                "similarity_threshold": 0.7
            })
            
            assert search_response.status_code == 200
            search_data = search_response.json()
            
            # All results should have high similarity scores
            for result in search_data["results"]:
                assert result["score"] >= 0.7
            
            # Our document should be in the results (if any)
            if len(search_data["results"]) > 0:
                found_doc = any(r["document_id"] == document_id for r in search_data["results"])
                # Note: We don't assert this because similarity depends on the embedding model
                
        except requests.exceptions.RequestException:
            pytest.skip("Mock Endee service is not available")
    
    def test_rag_with_no_relevant_documents(self):
        """Test RAG when no relevant documents are found."""
        try:
            # Search for something very specific that likely won't match existing docs
            rag_response = client.post("/rag", json={
                "query": "How to bake a chocolate cake with unicorn sprinkles",
                "top_k": 3,
                "similarity_threshold": 0.9  # High threshold to filter out irrelevant docs
            })
            
            assert rag_response.status_code == 200
            rag_data = rag_response.json()
            
            # Should handle no results gracefully
            assert "answer" in rag_data
            if len(rag_data["sources"]) == 0:
                assert "couldn't find any relevant information" in rag_data["answer"].lower()
            
        except requests.exceptions.RequestException:
            pytest.skip("Mock Endee service is not available")
    
    def test_api_error_handling(self):
        """Test API error handling for invalid inputs."""
        # Test invalid ingest request
        response = client.post("/ingest", json={
            "content": "",  # Empty content
            "metadata": {}
        })
        assert response.status_code == 422  # Validation error
        
        # Test invalid search request
        response = client.post("/search", json={
            "query": "",  # Empty query
            "top_k": 5
        })
        assert response.status_code == 422  # Validation error
        
        # Test invalid RAG request
        response = client.post("/rag", json={
            "query": "valid query",
            "top_k": 0  # Invalid top_k
        })
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])