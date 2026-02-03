"""
Tests for the vector search service query embedding functionality.
"""

import pytest
import requests
import requests_mock
from search import VectorSearchService, ValidationError, EmbeddingError, EndeeConnectionError


class TestQueryEmbedding:
    """Test cases for query embedding generation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.search_service = VectorSearchService()
    
    def test_valid_query_embedding_generation(self):
        """Test that valid queries generate 384-dimensional embeddings."""
        query = "What is machine learning?"
        
        # Generate embedding
        embedding = self.search_service._generate_query_embedding(query)
        
        # Verify embedding properties
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, (int, float)) for x in embedding)
        
    def test_query_validation_success(self):
        """Test that valid queries pass validation."""
        valid_queries = [
            "machine learning",
            "What is AI?",
            "How does neural network work?",
            "a" * 100,  # Long but valid query
        ]
        
        for query in valid_queries:
            processed = self.search_service._validate_query(query)
            assert isinstance(processed, str)
            assert len(processed.strip()) >= 2
    
    def test_query_validation_failures(self):
        """Test that invalid queries fail validation."""
        invalid_queries = [
            "",           # Empty string
            "   ",        # Only whitespace
            None,         # None value
            123,          # Non-string
            "a",          # Too short
            "a" * 1001,   # Too long
        ]
        
        for query in invalid_queries:
            with pytest.raises(ValidationError):
                self.search_service._validate_query(query)
    
    def test_embedding_consistency(self):
        """Test that same query generates consistent embeddings."""
        query = "test query for consistency"
        
        embedding1 = self.search_service._generate_query_embedding(query)
        embedding2 = self.search_service._generate_query_embedding(query)
        
        # Embeddings should be identical for same input
        assert embedding1 == embedding2
    
    def test_different_queries_different_embeddings(self):
        """Test that different queries generate different embeddings."""
        query1 = "machine learning algorithms"
        query2 = "natural language processing"
        
        embedding1 = self.search_service._generate_query_embedding(query1)
        embedding2 = self.search_service._generate_query_embedding(query2)
        
        # Embeddings should be different for different inputs
        assert embedding1 != embedding2


class TestEndeeVectorSearch:
    """Test cases for Endee vector similarity search."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.search_service = VectorSearchService()
        self.test_embedding = [0.1] * 384  # Valid 384-dimensional vector
    
    def test_endee_search_success(self):
        """Test successful Endee vector search."""
        with requests_mock.Mocker() as m:
            # Mock successful Endee response
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "metadata": {"content": "Machine learning content", "title": "ML Doc"}
                    },
                    {
                        "id": "doc2", 
                        "score": 0.85,
                        "metadata": {"content": "AI content", "title": "AI Doc"}
                    }
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Perform search
            results = self.search_service._perform_endee_search(self.test_embedding, 5)
            
            # Verify results
            assert len(results) == 2
            assert results[0]["id"] == "doc1"
            assert results[0]["score"] == 0.95
            assert results[1]["id"] == "doc2"
            assert results[1]["score"] == 0.85
    
    def test_endee_search_invalid_dimensions(self):
        """Test Endee search with invalid embedding dimensions."""
        invalid_embedding = [0.1] * 100  # Wrong dimensions
        
        with pytest.raises(EndeeConnectionError, match="Expected 384 dimensions"):
            self.search_service._perform_endee_search(invalid_embedding, 5)
    
    def test_endee_search_invalid_top_k(self):
        """Test Endee search with invalid top_k parameter."""
        with pytest.raises(EndeeConnectionError, match="top_k must be greater than 0"):
            self.search_service._perform_endee_search(self.test_embedding, 0)
    
    def test_endee_search_invalid_similarity_threshold(self):
        """Test Endee search with invalid similarity threshold."""
        # Test negative threshold
        with pytest.raises(EndeeConnectionError, match="similarity_threshold must be between 0.0 and 1.0"):
            self.search_service._perform_endee_search(self.test_embedding, 5, -0.1)
        
        # Test threshold > 1.0
        with pytest.raises(EndeeConnectionError, match="similarity_threshold must be between 0.0 and 1.0"):
            self.search_service._perform_endee_search(self.test_embedding, 5, 1.1)
    
    def test_endee_search_with_similarity_threshold(self):
        """Test Endee search with similarity threshold filtering."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with mixed scores
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "metadata": {"content": "High relevance content", "title": "High Doc"}
                    },
                    {
                        "id": "doc2", 
                        "score": 0.85,
                        "metadata": {"content": "Medium relevance content", "title": "Medium Doc"}
                    },
                    {
                        "id": "doc3",
                        "score": 0.65,
                        "metadata": {"content": "Low relevance content", "title": "Low Doc"}
                    },
                    {
                        "id": "doc4",
                        "score": 0.45,
                        "metadata": {"content": "Very low relevance content", "title": "Very Low Doc"}
                    }
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Test with threshold 0.7 - should return only first 2 results
            results = self.search_service._perform_endee_search(self.test_embedding, 5, 0.7)
            
            # Verify only results above threshold are returned
            assert len(results) == 2
            assert results[0]["id"] == "doc1"
            assert results[0]["score"] == 0.95
            assert results[1]["id"] == "doc2"
            assert results[1]["score"] == 0.85
    
    def test_endee_search_no_results_above_threshold(self):
        """Test Endee search when no results meet the similarity threshold."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with low scores
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.45,
                        "metadata": {"content": "Low relevance content", "title": "Low Doc"}
                    },
                    {
                        "id": "doc2", 
                        "score": 0.35,
                        "metadata": {"content": "Very low relevance content", "title": "Very Low Doc"}
                    }
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Test with high threshold 0.8 - should return no results
            results = self.search_service._perform_endee_search(self.test_embedding, 5, 0.8)
            
            # Verify no results returned
            assert len(results) == 0
    
    def test_endee_search_connection_error(self):
        """Test Endee search with connection error."""
        with requests_mock.Mocker() as m:
            # Mock connection error
            m.post(f"{self.search_service.endee_url}/vectors/search", 
                   exc=requests.exceptions.ConnectionError)
            
            with pytest.raises(EndeeConnectionError, match="Failed to connect to Endee service"):
                self.search_service._perform_endee_search(self.test_embedding, 5)
    
    def test_endee_search_http_error(self):
        """Test Endee search with HTTP error response."""
        with requests_mock.Mocker() as m:
            # Mock HTTP error
            m.post(f"{self.search_service.endee_url}/vectors/search", 
                   status_code=500, text="Internal Server Error")
            
            with pytest.raises(EndeeConnectionError, match="Endee search failed with status 500"):
                self.search_service._perform_endee_search(self.test_embedding, 5)


if __name__ == "__main__":
    pytest.main([__file__])


class TestSearchWithSimilarityThreshold:
    """Test cases for search functionality with similarity threshold."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.search_service = VectorSearchService()
    
    @pytest.mark.asyncio
    async def test_search_with_similarity_threshold(self):
        """Test search method with similarity threshold parameter."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with mixed scores
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "metadata": {"content": "Highly relevant machine learning content", "title": "ML Guide"}
                    },
                    {
                        "id": "doc2", 
                        "score": 0.85,
                        "metadata": {"content": "Moderately relevant AI content", "title": "AI Basics"}
                    },
                    {
                        "id": "doc3",
                        "score": 0.65,
                        "metadata": {"content": "Somewhat relevant data content", "title": "Data Science"}
                    },
                    {
                        "id": "doc4",
                        "score": 0.45,
                        "metadata": {"content": "Barely relevant programming content", "title": "Programming"}
                    }
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Test with threshold 0.7
            results = await self.search_service.search("machine learning", top_k=5, similarity_threshold=0.7)
            
            # Verify only results above threshold are returned
            assert len(results) == 2
            assert results[0]["document_id"] == "doc1"
            assert results[0]["score"] == 0.95
            assert results[1]["document_id"] == "doc2"
            assert results[1]["score"] == 0.85
    
    @pytest.mark.asyncio
    async def test_search_default_threshold_zero(self):
        """Test that default similarity threshold is 0.0 (no filtering)."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with low scores
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.25,
                        "metadata": {"content": "Low relevance content", "title": "Low Doc"}
                    },
                    {
                        "id": "doc2", 
                        "score": 0.15,
                        "metadata": {"content": "Very low relevance content", "title": "Very Low Doc"}
                    }
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Test without specifying threshold (should default to 0.0)
            results = await self.search_service.search("test query", top_k=5)
            
            # Verify all results are returned (no filtering)
            assert len(results) == 2
            assert results[0]["score"] == 0.25
            assert results[1]["score"] == 0.15
    
    @pytest.mark.asyncio
    async def test_search_threshold_validation(self):
        """Test validation of similarity threshold parameter."""
        # Test negative threshold
        with pytest.raises(ValidationError, match="similarity_threshold must be between 0.0 and 1.0"):
            await self.search_service.search("test query", top_k=5, similarity_threshold=-0.1)
        
        # Test threshold > 1.0
        with pytest.raises(ValidationError, match="similarity_threshold must be between 0.0 and 1.0"):
            await self.search_service.search("test query", top_k=5, similarity_threshold=1.1)
    
    @pytest.mark.asyncio
    async def test_search_high_threshold_no_results(self):
        """Test search with high threshold that filters out all results."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with moderate scores
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.75,
                        "metadata": {"content": "Moderate relevance content", "title": "Moderate Doc"}
                    },
                    {
                        "id": "doc2", 
                        "score": 0.65,
                        "metadata": {"content": "Lower relevance content", "title": "Lower Doc"}
                    }
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Test with very high threshold
            results = await self.search_service.search("test query", top_k=5, similarity_threshold=0.9)
            
            # Verify no results returned
            assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_search_exact_threshold_boundary(self):
        """Test search with threshold exactly matching a result score."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with specific scores
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.8,
                        "metadata": {"content": "Exact threshold content", "title": "Exact Doc"}
                    },
                    {
                        "id": "doc2", 
                        "score": 0.7,
                        "metadata": {"content": "Below threshold content", "title": "Below Doc"}
                    }
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Test with threshold exactly matching first result
            results = await self.search_service.search("test query", top_k=5, similarity_threshold=0.8)
            
            # Verify only the exact match is returned (>= threshold)
            assert len(results) == 1
            assert results[0]["document_id"] == "doc1"
            assert results[0]["score"] == 0.8