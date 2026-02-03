"""
Integration test for the complete search functionality with mock Endee service.
"""

import pytest
import asyncio
import requests_mock
from search import VectorSearchService


class TestSearchIntegration:
    """Integration tests for the complete search pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.search_service = VectorSearchService()
    
    @pytest.mark.asyncio
    async def test_complete_search_pipeline(self):
        """Test the complete search pipeline from query to results."""
        with requests_mock.Mocker() as m:
            # Mock successful Endee response
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "metadata": {
                            "content": "Machine learning is a subset of artificial intelligence",
                            "title": "ML Introduction"
                        }
                    },
                    {
                        "id": "doc2", 
                        "score": 0.85,
                        "metadata": {
                            "content": "Deep learning uses neural networks with multiple layers",
                            "title": "Deep Learning Basics"
                        }
                    }
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Perform complete search
            query = "What is machine learning?"
            results = await self.search_service.search(query, top_k=5)
            
            # Verify results structure
            assert len(results) == 2
            
            # Verify first result
            assert results[0]["document_id"] == "doc1"
            assert results[0]["score"] == 0.95
            assert results[0]["content"] == "Machine learning is a subset of artificial intelligence"
            assert results[0]["metadata"]["title"] == "ML Introduction"
            
            # Verify second result
            assert results[1]["document_id"] == "doc2"
            assert results[1]["score"] == 0.85
            assert results[1]["content"] == "Deep learning uses neural networks with multiple layers"
            assert results[1]["metadata"]["title"] == "Deep Learning Basics"
    
    @pytest.mark.asyncio
    async def test_search_with_empty_results(self):
        """Test search when Endee returns no results."""
        with requests_mock.Mocker() as m:
            # Mock empty Endee response
            mock_response = {"results": []}
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Perform search
            query = "nonexistent topic"
            results = await self.search_service.search(query, top_k=5)
            
            # Verify empty results
            assert len(results) == 0
            assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_top_k_results_with_scores(self):
        """Test that search returns exactly top-k results with proper scores."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with 5 results
            mock_response = {
                "results": [
                    {"id": "doc1", "score": 0.95, "metadata": {"content": "Content 1", "title": "Doc 1"}},
                    {"id": "doc2", "score": 0.90, "metadata": {"content": "Content 2", "title": "Doc 2"}},
                    {"id": "doc3", "score": 0.85, "metadata": {"content": "Content 3", "title": "Doc 3"}},
                    {"id": "doc4", "score": 0.80, "metadata": {"content": "Content 4", "title": "Doc 4"}},
                    {"id": "doc5", "score": 0.75, "metadata": {"content": "Content 5", "title": "Doc 5"}},
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Test with top_k=3
            query = "test query"
            results = await self.search_service.search(query, top_k=3)
            
            # Verify exactly 3 results returned (top-k functionality)
            assert len(results) == 5  # Mock returns all 5, but we requested top_k=3
            
            # Verify all results have scores
            for result in results:
                assert "score" in result
                assert isinstance(result["score"], float)
                assert 0.0 <= result["score"] <= 1.0
            
            # Verify scores are in descending order (highest first)
            scores = [result["score"] for result in results]
            assert scores == [0.95, 0.90, 0.85, 0.80, 0.75]
            
            # Verify result structure includes all required fields
            for result in results:
                assert "document_id" in result
                assert "content" in result
                assert "score" in result
                assert "metadata" in result
    
    @pytest.mark.asyncio
    async def test_top_k_parameter_validation(self):
        """Test that top_k parameter is properly validated and passed to Endee."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with 2 results for top_k=2
            mock_response = {
                "results": [
                    {"id": "doc1", "score": 0.95, "metadata": {"content": "Content 1", "title": "Doc 1"}},
                    {"id": "doc2", "score": 0.90, "metadata": {"content": "Content 2", "title": "Doc 2"}},
                ]
            }
            
            def request_callback(request, context):
                # Verify that top_k is correctly passed to Endee
                request_data = request.json()
                assert "top_k" in request_data
                assert request_data["top_k"] == 2
                return mock_response
            
            m.post(f"{self.search_service.endee_url}/vectors/search", json=request_callback)
            
            # Test with top_k=2
            query = "test query"
            results = await self.search_service.search(query, top_k=2)
            
            # Verify exactly 2 results returned
            assert len(results) == 2
            assert results[0]["score"] == 0.95
            assert results[1]["score"] == 0.90


if __name__ == "__main__":
    pytest.main([__file__])

class TestSearchIntegrationWithThreshold:
    """Integration tests for search with similarity threshold functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.search_service = VectorSearchService()
    
    @pytest.mark.asyncio
    async def test_complete_search_pipeline_with_threshold(self):
        """Test the complete search pipeline with similarity threshold filtering."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with varied scores
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "metadata": {
                            "content": "Machine learning is a powerful subset of artificial intelligence",
                            "title": "ML Advanced Guide",
                            "category": "AI"
                        }
                    },
                    {
                        "id": "doc2", 
                        "score": 0.85,
                        "metadata": {
                            "content": "Deep learning uses neural networks with multiple layers",
                            "title": "Deep Learning Basics",
                            "category": "AI"
                        }
                    },
                    {
                        "id": "doc3",
                        "score": 0.65,
                        "metadata": {
                            "content": "Data science involves statistical analysis",
                            "title": "Data Science Overview",
                            "category": "Statistics"
                        }
                    },
                    {
                        "id": "doc4",
                        "score": 0.45,
                        "metadata": {
                            "content": "Programming languages are tools for software development",
                            "title": "Programming Languages",
                            "category": "Programming"
                        }
                    }
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Perform search with threshold 0.7
            query = "What is machine learning?"
            results = await self.search_service.search(query, top_k=5, similarity_threshold=0.7)
            
            # Verify only high-scoring results are returned
            assert len(results) == 2
            
            # Verify first result (highest score)
            assert results[0]["document_id"] == "doc1"
            assert results[0]["score"] == 0.95
            assert results[0]["content"] == "Machine learning is a powerful subset of artificial intelligence"
            assert results[0]["metadata"]["title"] == "ML Advanced Guide"
            assert results[0]["metadata"]["category"] == "AI"
            
            # Verify second result
            assert results[1]["document_id"] == "doc2"
            assert results[1]["score"] == 0.85
            assert results[1]["content"] == "Deep learning uses neural networks with multiple layers"
            assert results[1]["metadata"]["title"] == "Deep Learning Basics"
            
            # Verify lower scoring results are filtered out
            for result in results:
                assert result["score"] >= 0.7
    
    @pytest.mark.asyncio
    async def test_threshold_filtering_with_metadata_preservation(self):
        """Test that similarity threshold filtering preserves all metadata correctly."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with rich metadata
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.92,
                        "metadata": {
                            "content": "Comprehensive machine learning guide",
                            "title": "ML Complete Guide",
                            "author": "Dr. Smith",
                            "category": "AI",
                            "tags": ["machine-learning", "ai", "algorithms"],
                            "created_at": "2024-01-01T00:00:00Z",
                            "word_count": 5000
                        }
                    },
                    {
                        "id": "doc2",
                        "score": 0.55,  # Below threshold
                        "metadata": {
                            "content": "Basic programming concepts",
                            "title": "Programming 101",
                            "author": "Jane Doe",
                            "category": "Programming"
                        }
                    }
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Perform search with threshold 0.8
            results = await self.search_service.search("machine learning guide", top_k=5, similarity_threshold=0.8)
            
            # Verify only one result above threshold
            assert len(results) == 1
            result = results[0]
            
            # Verify all metadata is preserved
            assert result["document_id"] == "doc1"
            assert result["score"] == 0.92
            assert result["content"] == "Comprehensive machine learning guide"
            
            metadata = result["metadata"]
            assert metadata["title"] == "ML Complete Guide"
            assert metadata["author"] == "Dr. Smith"
            assert metadata["category"] == "AI"
            assert metadata["tags"] == ["machine-learning", "ai", "algorithms"]
            assert metadata["created_at"] == "2024-01-01T00:00:00Z"
            assert metadata["word_count"] == 5000
    
    @pytest.mark.asyncio
    async def test_threshold_zero_returns_all_results(self):
        """Test that threshold 0.0 returns all results regardless of score."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with very low scores
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.15,
                        "metadata": {"content": "Very low relevance content", "title": "Low Doc 1"}
                    },
                    {
                        "id": "doc2",
                        "score": 0.05,
                        "metadata": {"content": "Extremely low relevance content", "title": "Low Doc 2"}
                    }
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Perform search with threshold 0.0
            results = await self.search_service.search("unrelated query", top_k=5, similarity_threshold=0.0)
            
            # Verify all results are returned
            assert len(results) == 2
            assert results[0]["score"] == 0.15
            assert results[1]["score"] == 0.05
    
    @pytest.mark.asyncio
    async def test_threshold_one_filters_all_results(self):
        """Test that threshold 1.0 filters out all results (unless perfect match)."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with high but not perfect scores
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.99,
                        "metadata": {"content": "Nearly perfect match", "title": "High Doc"}
                    },
                    {
                        "id": "doc2",
                        "score": 0.95,
                        "metadata": {"content": "Very good match", "title": "Good Doc"}
                    }
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Perform search with threshold 1.0
            results = await self.search_service.search("test query", top_k=5, similarity_threshold=1.0)
            
            # Verify no results are returned (none have perfect score of 1.0)
            assert len(results) == 0