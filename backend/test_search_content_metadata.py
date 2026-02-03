"""
Test to verify that search results include document content and metadata correctly.
This test specifically validates the task: "Results include document content and metadata"
"""

import pytest
import asyncio
import requests_mock
from search import VectorSearchService


class TestSearchContentAndMetadata:
    """Test that search results properly include document content and metadata."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.search_service = VectorSearchService()
    
    @pytest.mark.asyncio
    async def test_results_include_content_and_metadata(self):
        """Test that search results include both content and metadata fields."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with rich metadata
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "metadata": {
                            "content": "This is the document content for testing",
                            "title": "Test Document",
                            "author": "Test Author",
                            "category": "Technology",
                            "created_at": "2024-01-01T00:00:00Z"
                        }
                    }
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Perform search
            results = await self.search_service.search("test query", top_k=1)
            
            # Verify result structure
            assert len(results) == 1
            result = results[0]
            
            # Verify all required fields are present
            assert "document_id" in result
            assert "content" in result
            assert "score" in result
            assert "metadata" in result
            
            # Verify content is correctly extracted
            assert result["content"] == "This is the document content for testing"
            
            # Verify metadata is correctly preserved
            metadata = result["metadata"]
            assert metadata["title"] == "Test Document"
            assert metadata["author"] == "Test Author"
            assert metadata["category"] == "Technology"
            assert metadata["created_at"] == "2024-01-01T00:00:00Z"
            
            # Verify content is also available in metadata (as stored by ingestion)
            assert metadata["content"] == "This is the document content for testing"
    
    @pytest.mark.asyncio
    async def test_results_with_minimal_metadata(self):
        """Test search results when metadata contains only content."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with minimal metadata
            mock_response = {
                "results": [
                    {
                        "id": "doc2",
                        "score": 0.80,
                        "metadata": {
                            "content": "Minimal content document"
                        }
                    }
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Perform search
            results = await self.search_service.search("minimal query", top_k=1)
            
            # Verify result structure
            assert len(results) == 1
            result = results[0]
            
            # Verify content is extracted
            assert result["content"] == "Minimal content document"
            
            # Verify metadata contains only content
            metadata = result["metadata"]
            assert metadata["content"] == "Minimal content document"
            assert len(metadata) == 1  # Only content field
    
    @pytest.mark.asyncio
    async def test_results_with_missing_content(self):
        """Test search results when content is missing from metadata."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with metadata but no content field
            mock_response = {
                "results": [
                    {
                        "id": "doc3",
                        "score": 0.70,
                        "metadata": {
                            "title": "Document without content field",
                            "author": "Test Author"
                        }
                    }
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Perform search
            results = await self.search_service.search("missing content query", top_k=1)
            
            # Verify result structure
            assert len(results) == 1
            result = results[0]
            
            # Verify content defaults to empty string when missing
            assert result["content"] == ""
            
            # Verify other metadata is preserved
            metadata = result["metadata"]
            assert metadata["title"] == "Document without content field"
            assert metadata["author"] == "Test Author"
    
    @pytest.mark.asyncio
    async def test_results_with_empty_metadata(self):
        """Test search results when metadata is empty."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with empty metadata
            mock_response = {
                "results": [
                    {
                        "id": "doc4",
                        "score": 0.60,
                        "metadata": {}
                    }
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Perform search
            results = await self.search_service.search("empty metadata query", top_k=1)
            
            # Verify result structure
            assert len(results) == 1
            result = results[0]
            
            # Verify content defaults to empty string
            assert result["content"] == ""
            
            # Verify metadata is empty dict
            assert result["metadata"] == {}
    
    @pytest.mark.asyncio
    async def test_multiple_results_content_and_metadata(self):
        """Test that multiple search results all include content and metadata."""
        with requests_mock.Mocker() as m:
            # Mock Endee response with multiple results
            mock_response = {
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "metadata": {
                            "content": "First document content",
                            "title": "First Document",
                            "category": "Tech"
                        }
                    },
                    {
                        "id": "doc2",
                        "score": 0.85,
                        "metadata": {
                            "content": "Second document content",
                            "title": "Second Document",
                            "category": "Science"
                        }
                    },
                    {
                        "id": "doc3",
                        "score": 0.75,
                        "metadata": {
                            "content": "Third document content",
                            "title": "Third Document"
                        }
                    }
                ]
            }
            m.post(f"{self.search_service.endee_url}/vectors/search", json=mock_response)
            
            # Perform search
            results = await self.search_service.search("multiple results query", top_k=3)
            
            # Verify all results have content and metadata
            assert len(results) == 3
            
            for i, result in enumerate(results):
                # Verify structure
                assert "document_id" in result
                assert "content" in result
                assert "score" in result
                assert "metadata" in result
                
                # Verify content matches expected
                expected_content = f"{['First', 'Second', 'Third'][i]} document content"
                assert result["content"] == expected_content
                
                # Verify metadata is preserved
                expected_title = f"{['First', 'Second', 'Third'][i]} Document"
                assert result["metadata"]["title"] == expected_title


if __name__ == "__main__":
    pytest.main([__file__])