"""
Tests for the RAG (Retrieval Augmented Generation) service.
"""

import pytest
import requests_mock
from unittest.mock import Mock, AsyncMock
from rag import (
    RAGService, RAGError, ContextCombinationError, LLMError,
    BaseLLMProvider, PlaceholderLLMProvider, LLMIntegrationManager
)
from search import VectorSearchService, ValidationError, SearchError


class TestRAGService:
    """Test cases for RAG service functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create mock search service
        self.mock_search_service = Mock(spec=VectorSearchService)
        self.mock_llm_manager = Mock(spec=LLMIntegrationManager)
        self.rag_service = RAGService(self.mock_search_service, self.mock_llm_manager)
    
    def test_rag_service_initialization(self):
        """Test RAG service initialization."""
        # Test with provided services
        rag = RAGService(self.mock_search_service, self.mock_llm_manager)
        assert rag.search_service == self.mock_search_service
        assert rag.llm_manager == self.mock_llm_manager
        assert rag.max_context_length == 4000
        assert rag.context_separator == "\n\n---\n\n"
        
        # Test with custom max_context_length
        rag_custom = RAGService(self.mock_search_service, self.mock_llm_manager, max_context_length=8000)
        assert rag_custom.max_context_length == 8000
        
        # Test with default services
        rag_default = RAGService()
        assert isinstance(rag_default.search_service, VectorSearchService)
        assert isinstance(rag_default.llm_manager, LLMIntegrationManager)
        assert rag_default.max_context_length == 4000
    
    def test_llm_provider_management(self):
        """Test LLM provider management methods."""
        # Test setting LLM provider
        mock_provider = Mock(spec=BaseLLMProvider)
        mock_provider.get_provider_name.return_value = "TestProvider"
        
        self.rag_service.set_llm_provider(mock_provider)
        self.mock_llm_manager.set_provider.assert_called_once_with(mock_provider)
        
        # Test getting provider info
        expected_info = {"provider_name": "TestProvider", "model_info": {"test": "data"}}
        self.mock_llm_manager.get_provider_info.return_value = expected_info
        
        info = self.rag_service.get_llm_provider_info()
        assert info == expected_info
        self.mock_llm_manager.get_provider_info.assert_called_once()
        
        # Test configuring LLM defaults
        config = {"temperature": 0.5, "max_tokens": 2000}
        self.mock_llm_manager.default_config = {}  # Add the attribute to the mock
        self.rag_service.configure_llm_defaults(config)
        # Should update the manager's default config
        assert hasattr(self.mock_llm_manager, 'default_config')
    
    def test_context_length_configuration(self):
        """Test context length configuration methods."""
        # Test setting valid context length
        self.rag_service.set_max_context_length(8000)
        assert self.rag_service.get_max_context_length() == 8000
        
        # Test setting another valid context length
        self.rag_service.set_max_context_length(2000)
        assert self.rag_service.get_max_context_length() == 2000
        
        # Test invalid context lengths
        with pytest.raises(ValidationError, match="max_context_length must be a positive integer"):
            self.rag_service.set_max_context_length(0)
        
        with pytest.raises(ValidationError, match="max_context_length must be a positive integer"):
            self.rag_service.set_max_context_length(-100)
        
        with pytest.raises(ValidationError, match="max_context_length cannot exceed 50000"):
            self.rag_service.set_max_context_length(60000)
        
        with pytest.raises(ValidationError, match="max_context_length must be a positive integer"):
            self.rag_service.set_max_context_length("invalid")
    
    def test_validate_inputs_success(self):
        """Test successful input validation."""
        # Valid inputs
        self.rag_service._validate_inputs("What is machine learning?", 5)
        self.rag_service._validate_inputs("AI", 1)
        self.rag_service._validate_inputs("a" * 100, 20)
    
    def test_validate_inputs_failures(self):
        """Test input validation failures."""
        # Invalid query
        with pytest.raises(ValidationError, match="Query must be a non-empty string"):
            self.rag_service._validate_inputs("", 5)
        
        with pytest.raises(ValidationError, match="Query must be a non-empty string"):
            self.rag_service._validate_inputs(None, 5)
        
        with pytest.raises(ValidationError, match="Query cannot be empty or contain only whitespace"):
            self.rag_service._validate_inputs("   ", 5)
        
        # Invalid top_k
        with pytest.raises(ValidationError, match="top_k must be a positive integer"):
            self.rag_service._validate_inputs("test", 0)
        
        with pytest.raises(ValidationError, match="top_k must be a positive integer"):
            self.rag_service._validate_inputs("test", -1)
        
        with pytest.raises(ValidationError, match="top_k cannot exceed 20"):
            self.rag_service._validate_inputs("test", 25)


class TestContextCombination:
    """Test cases for context combination functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_search_service = Mock(spec=VectorSearchService)
        self.mock_llm_manager = Mock(spec=LLMIntegrationManager)
        self.rag_service = RAGService(self.mock_search_service, self.mock_llm_manager)
    
    def test_combine_context_empty_results(self):
        """Test context combination with empty search results."""
        result = self.rag_service._combine_retrieved_context([])
        
        assert result["combined_context"] == ""
        assert result["sources_used"] == []
        assert result["total_sources"] == 0
        assert result["context_length"] == 0
        assert result["truncated"] == False
    
    def test_combine_context_single_document(self):
        """Test context combination with single document."""
        search_results = [
            {
                "document_id": "doc1",
                "content": "Machine learning is a subset of artificial intelligence.",
                "score": 0.95,
                "metadata": {"title": "ML Basics", "author": "John Doe"}
            }
        ]
        
        result = self.rag_service._combine_retrieved_context(search_results)
        
        assert "Source 1 (ML Basics):" in result["combined_context"]
        assert "Machine learning is a subset of artificial intelligence." in result["combined_context"]
        assert result["total_sources"] == 1
        assert result["truncated"] == False
        assert len(result["sources_used"]) == 1
        assert result["sources_used"][0]["document_id"] == "doc1"
        assert result["sources_used"][0]["score"] == 0.95
    
    def test_combine_context_multiple_documents(self):
        """Test context combination with multiple documents."""
        search_results = [
            {
                "document_id": "doc1",
                "content": "Machine learning is a subset of AI.",
                "score": 0.95,
                "metadata": {"title": "ML Basics"}
            },
            {
                "document_id": "doc2",
                "content": "Deep learning uses neural networks.",
                "score": 0.85,
                "metadata": {"title": "Deep Learning"}
            },
            {
                "document_id": "doc3",
                "content": "Natural language processing handles text.",
                "score": 0.75,
                "metadata": {"title": "NLP Guide"}
            }
        ]
        
        result = self.rag_service._combine_retrieved_context(search_results)
        
        # Check that all sources are included
        assert "Source 1 (ML Basics):" in result["combined_context"]
        assert "Source 2 (Deep Learning):" in result["combined_context"]
        assert "Source 3 (NLP Guide):" in result["combined_context"]
        assert result["total_sources"] == 3
        assert result["truncated"] == False
        
        # Check content is included
        assert "Machine learning is a subset of AI." in result["combined_context"]
        assert "Deep learning uses neural networks." in result["combined_context"]
        assert "Natural language processing handles text." in result["combined_context"]
        
        # Check sources are sorted by score (highest first)
        sources = result["sources_used"]
        assert sources[0]["score"] == 0.95
        assert sources[1]["score"] == 0.85
        assert sources[2]["score"] == 0.75
    
    def test_combine_context_with_truncation(self):
        """Test context combination with length-based truncation."""
        # Create a document with very long content to trigger truncation
        long_content = "A" * 2000  # 2000 characters
        search_results = [
            {
                "document_id": "doc1",
                "content": long_content,
                "score": 0.95,
                "metadata": {"title": "Long Doc 1"}
            },
            {
                "document_id": "doc2",
                "content": long_content,
                "score": 0.85,
                "metadata": {"title": "Long Doc 2"}
            },
            {
                "document_id": "doc3",
                "content": long_content,
                "score": 0.75,
                "metadata": {"title": "Long Doc 3"}
            }
        ]
        
        result = self.rag_service._combine_retrieved_context(search_results)
        
        # Should be truncated due to max_context_length limit
        assert result["truncated"] == True
        assert result["context_length"] <= self.rag_service.max_context_length
        assert result["total_sources"] < 3  # Not all sources should be included
    
    def test_combine_context_with_custom_length_limit(self):
        """Test context combination with custom length limit."""
        # Set a smaller context limit
        self.rag_service.set_max_context_length(500)
        
        # Create documents that would exceed the limit
        search_results = [
            {
                "document_id": "doc1",
                "content": "A" * 200,  # 200 characters
                "score": 0.95,
                "metadata": {"title": "Doc 1"}
            },
            {
                "document_id": "doc2",
                "content": "B" * 200,  # 200 characters
                "score": 0.85,
                "metadata": {"title": "Doc 2"}
            },
            {
                "document_id": "doc3",
                "content": "C" * 200,  # 200 characters
                "score": 0.75,
                "metadata": {"title": "Doc 3"}
            }
        ]
        
        result = self.rag_service._combine_retrieved_context(search_results)
        
        # Should be truncated due to custom limit
        assert result["context_length"] <= 500
        assert result["truncated"] == True
        assert result["total_sources"] < 3  # Not all sources should be included
    
    def test_combine_context_empty_content_skipped(self):
        """Test that documents with empty content are skipped."""
        search_results = [
            {
                "document_id": "doc1",
                "content": "Valid content here.",
                "score": 0.95,
                "metadata": {"title": "Valid Doc"}
            },
            {
                "document_id": "doc2",
                "content": "",  # Empty content
                "score": 0.85,
                "metadata": {"title": "Empty Doc"}
            },
            {
                "document_id": "doc3",
                "content": "   ",  # Whitespace only
                "score": 0.75,
                "metadata": {"title": "Whitespace Doc"}
            }
        ]
        
        result = self.rag_service._combine_retrieved_context(search_results)
        
        # Only the valid document should be included
        assert result["total_sources"] == 1
        assert "Valid content here." in result["combined_context"]
        assert "Source 1 (Valid Doc):" in result["combined_context"]
    
    def test_combine_context_no_title_metadata(self):
        """Test context combination when documents have no title metadata."""
        search_results = [
            {
                "document_id": "doc1",
                "content": "Content without title.",
                "score": 0.95,
                "metadata": {"author": "John Doe"}  # No title
            }
        ]
        
        result = self.rag_service._combine_retrieved_context(search_results)
        
        # Should use "Source 1:" without title
        assert "Source 1:" in result["combined_context"]
        assert "Content without title." in result["combined_context"]
        assert result["total_sources"] == 1
    
    def test_optimize_context_for_llm(self):
        """Test context optimization for LLM consumption."""
        combined_context = "Source 1:\nMachine learning content.\n\n---\n\nSource 2:\nAI content."
        query = "What is machine learning?"
        
        optimized = self.rag_service._optimize_context_for_llm(combined_context, query)
        
        # Check for enhanced format elements
        assert "=== USER QUESTION ===" in optimized
        assert "Question: What is machine learning?" in optimized
        assert "=== RETRIEVED INFORMATION ===" in optimized
        assert "=== RESPONSE GUIDELINES ===" in optimized
        assert "MACHINE LEARNING content." in optimized  # Should be highlighted
        assert "AI content." in optimized
        assert "comprehensive and accurate answer" in optimized
    
    def test_optimize_context_empty(self):
        """Test context optimization with empty context."""
        optimized = self.rag_service._optimize_context_for_llm("", "test query")
        assert optimized == ""


class TestRAGPipeline:
    """Test cases for the complete RAG pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_search_service = Mock(spec=VectorSearchService)
        self.mock_llm_manager = Mock(spec=LLMIntegrationManager)
        self.rag_service = RAGService(self.mock_search_service, self.mock_llm_manager)
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self):
        """Test successful RAG response generation."""
        # Mock search results
        mock_search_results = [
            {
                "document_id": "doc1",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.",
                "score": 0.95,
                "metadata": {"title": "ML Introduction", "author": "Expert"}
            },
            {
                "document_id": "doc2",
                "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
                "score": 0.85,
                "metadata": {"title": "Deep Learning Basics"}
            }
        ]
        
        # Configure mocks
        self.mock_search_service.search = AsyncMock(return_value=mock_search_results)
        self.mock_llm_manager.generate_response = AsyncMock(return_value={
            "response": "Generated LLM response about machine learning",
            "provider_name": "TestLLM",
            "model_info": {"name": "test-model"},
            "config_used": {},
            "prompt_length": 25,
            "context_length": 100,
            "response_length": 50
        })
        
        # Generate response
        result = await self.rag_service.generate_response("What is machine learning?", top_k=3)
        
        # Verify response structure
        assert result["query"] == "What is machine learning?"
        assert result["answer"] == "Generated LLM response about machine learning"
        assert "sources" in result
        assert "context_used" in result
        assert result["sources_count"] == 2
        assert result["truncated"] == False
        assert result["max_context_length"] == 4000  # Default value
        assert result["llm_provider"] == "TestLLM"
        assert result["llm_model"] == {"name": "test-model"}
        
        # Verify sources
        assert len(result["sources"]) == 2
        assert result["sources"][0]["document_id"] == "doc1"
        assert result["sources"][0]["score"] == 0.95
        
        # Verify LLM manager was called correctly
        self.mock_llm_manager.generate_response.assert_called_once()
        call_args = self.mock_llm_manager.generate_response.call_args
        assert call_args[1]["prompt"] == "What is machine learning?"
        assert "context" in call_args[1]
        assert "config" in call_args[1]
    
    @pytest.mark.asyncio
    async def test_generate_response_no_results(self):
        """Test RAG response when no relevant documents are found."""
        # Mock empty search results
        self.mock_search_service.search = AsyncMock(return_value=[])
        
        # Generate response
        result = await self.rag_service.generate_response("obscure query", top_k=3)
        
        # Verify response for no results
        assert result["query"] == "obscure query"
        assert "couldn't find any relevant information" in result["answer"]
        assert result["sources"] == []
        assert result["context_used"] == ""
        assert result["sources_count"] == 0
        assert result["max_context_length"] == 4000  # Default value
        
        # LLM manager should not be called when no results
        self.mock_llm_manager.generate_response.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_generate_response_with_similarity_threshold(self):
        """Test RAG response with similarity threshold."""
        mock_search_results = [
            {
                "document_id": "doc1",
                "content": "Relevant content.",
                "score": 0.95,
                "metadata": {"title": "Relevant Doc"}
            }
        ]
        
        self.mock_search_service.search = AsyncMock(return_value=mock_search_results)
        self.mock_llm_manager.generate_response = AsyncMock(return_value={
            "response": "Test response",
            "provider_name": "TestLLM",
            "model_info": {"name": "test-model"},
            "config_used": {},
            "prompt_length": 10,
            "context_length": 50,
            "response_length": 13
        })
        
        # Generate response with threshold
        result = await self.rag_service.generate_response(
            "test query", 
            top_k=5, 
            similarity_threshold=0.8
        )
        
        # Verify search was called with threshold
        self.mock_search_service.search.assert_called_once_with(
            query="test query",
            top_k=5,
            similarity_threshold=0.8
        )
        
        assert result["sources_count"] == 1
    
    @pytest.mark.asyncio
    async def test_generate_response_with_custom_context_length(self):
        """Test RAG response with custom context length parameter."""
        # Create mock search results with long content
        mock_search_results = [
            {
                "document_id": "doc1",
                "content": "A" * 1000,  # 1000 characters
                "score": 0.95,
                "metadata": {"title": "Long Doc 1"}
            },
            {
                "document_id": "doc2",
                "content": "B" * 1000,  # 1000 characters
                "score": 0.85,
                "metadata": {"title": "Long Doc 2"}
            }
        ]
        
        self.mock_search_service.search = AsyncMock(return_value=mock_search_results)
        self.mock_llm_manager.generate_response = AsyncMock(return_value={
            "response": "Test response with custom context length",
            "provider_name": "TestLLM",
            "model_info": {"name": "test-model"},
            "config_used": {},
            "prompt_length": 10,
            "context_length": 500,
            "response_length": 40
        })
        
        # Generate response with custom context length
        result = await self.rag_service.generate_response(
            "test query", 
            top_k=5, 
            max_context_length=800  # Custom context length
        )
        
        # Verify custom context length is used and returned
        assert result["max_context_length"] == 800
        assert result["context_length"] <= 800  # Should respect the limit
        
        # Verify original service context length is restored
        assert self.rag_service.get_max_context_length() == 4000  # Should be back to default
    
    @pytest.mark.asyncio
    async def test_generate_response_invalid_custom_context_length(self):
        """Test RAG response with invalid custom context length."""
        # Test with invalid context length values
        with pytest.raises(ValidationError, match="max_context_length must be a positive integer"):
            await self.rag_service.generate_response("test", top_k=3, max_context_length=0)
        
        with pytest.raises(ValidationError, match="max_context_length must be a positive integer"):
            await self.rag_service.generate_response("test", top_k=3, max_context_length=-100)
        
        with pytest.raises(ValidationError, match="max_context_length cannot exceed 50000"):
            await self.rag_service.generate_response("test", top_k=3, max_context_length=60000)
    
    @pytest.mark.asyncio
    async def test_generate_response_validation_error(self):
        """Test RAG response with validation errors."""
        # Test empty query
        with pytest.raises(ValidationError):
            await self.rag_service.generate_response("", top_k=3)
        
        # Test invalid top_k
        with pytest.raises(ValidationError):
            await self.rag_service.generate_response("test", top_k=0)
    
    @pytest.mark.asyncio
    async def test_generate_response_search_error(self):
        """Test RAG response when search service fails."""
        # Mock search service to raise error
        self.mock_search_service.search = AsyncMock(side_effect=SearchError("Search failed"))
        
        # Should propagate search error
        with pytest.raises(SearchError, match="Document retrieval failed"):
            await self.rag_service.generate_response("test query", top_k=3)
    
    @pytest.mark.asyncio
    async def test_generate_response_llm_error(self):
        """Test RAG response when LLM generation fails."""
        # Mock search to return results
        mock_search_results = [{"document_id": "doc1", "content": "test", "score": 0.9, "metadata": {}}]
        self.mock_search_service.search = AsyncMock(return_value=mock_search_results)
        
        # Mock LLM manager to raise error
        self.mock_llm_manager.generate_response = AsyncMock(side_effect=LLMError("LLM failed"))
        
        # Should propagate LLM error
        with pytest.raises(LLMError, match="Response generation failed"):
            await self.rag_service.generate_response("test query", top_k=3)
    
    @pytest.mark.asyncio
    async def test_generate_response_context_combination_error(self):
        """Test RAG response when context combination fails."""
        # Mock search to return results, but patch context combination to fail
        self.mock_search_service.search = AsyncMock(return_value=[{"doc": "test"}])
        
        # Patch the context combination method to raise error
        original_method = self.rag_service._combine_retrieved_context
        self.rag_service._combine_retrieved_context = Mock(
            side_effect=ContextCombinationError("Context combination failed")
        )
        
        try:
            with pytest.raises(ContextCombinationError):
                await self.rag_service.generate_response("test query", top_k=3)
        finally:
            # Restore original method
            self.rag_service._combine_retrieved_context = original_method


class TestLLMIntegration:
    """Test cases for LLM integration components."""
    
    def test_placeholder_llm_provider_initialization(self):
        """Test PlaceholderLLMProvider initialization."""
        # Test with default template
        provider = PlaceholderLLMProvider()
        assert provider.get_provider_name() == "PlaceholderLLM"
        assert provider.get_model_info()["name"] == "placeholder-model-v1"
        assert provider.get_model_info()["type"] == "mock"
        
        # Test with custom template
        custom_template = "Custom response: {prompt}"
        provider_custom = PlaceholderLLMProvider(custom_template)
        assert provider_custom.response_template == custom_template
    
    @pytest.mark.asyncio
    async def test_placeholder_llm_provider_generate_response(self):
        """Test PlaceholderLLMProvider response generation."""
        provider = PlaceholderLLMProvider()
        
        # Test basic response generation
        response = await provider.generate_response(
            prompt="What is AI?",
            context="AI is artificial intelligence.",
            sources_count=2,
            context_length=100
        )
        
        assert "What is AI?" in response
        assert "2 relevant sources" in response
        assert "100 characters of context" in response
        assert "placeholder response" in response.lower()
        assert "Context preview: AI is artificial intelligence." in response
    
    @pytest.mark.asyncio
    async def test_placeholder_llm_provider_empty_context(self):
        """Test PlaceholderLLMProvider with empty context."""
        provider = PlaceholderLLMProvider()
        
        response = await provider.generate_response(
            prompt="Test query",
            context="",
            sources_count=0,
            context_length=0
        )
        
        assert "Test query" in response
        assert "0 relevant sources" in response
        assert "Context preview:" not in response
    
    def test_llm_integration_manager_initialization(self):
        """Test LLMIntegrationManager initialization."""
        # Test with default provider
        manager = LLMIntegrationManager()
        assert isinstance(manager.provider, PlaceholderLLMProvider)
        assert manager.default_config["max_tokens"] == 1000
        assert manager.default_config["temperature"] == 0.7
        
        # Test with custom provider
        custom_provider = PlaceholderLLMProvider()
        manager_custom = LLMIntegrationManager(custom_provider)
        assert manager_custom.provider == custom_provider
    
    def test_llm_integration_manager_provider_management(self):
        """Test LLM provider management in integration manager."""
        manager = LLMIntegrationManager()
        
        # Test getting provider info
        info = manager.get_provider_info()
        assert info["provider_name"] == "PlaceholderLLM"
        assert "model_info" in info
        
        # Test setting new provider
        new_provider = PlaceholderLLMProvider("New template: {prompt}")
        manager.set_provider(new_provider)
        assert manager.provider == new_provider
        
        # Verify provider info updated
        new_info = manager.get_provider_info()
        assert new_info["provider_name"] == "PlaceholderLLM"
    
    @pytest.mark.asyncio
    async def test_llm_integration_manager_generate_response(self):
        """Test LLM integration manager response generation."""
        manager = LLMIntegrationManager()
        
        result = await manager.generate_response(
            prompt="Test prompt",
            context="Test context",
            config={"temperature": 0.5}
        )
        
        # Verify response structure
        assert "response" in result
        assert result["provider_name"] == "PlaceholderLLM"
        assert "model_info" in result
        assert result["config_used"]["temperature"] == 0.5
        assert result["config_used"]["max_tokens"] == 1000  # Default merged
        assert result["prompt_length"] == len("Test prompt")
        assert result["context_length"] == len("Test context")
        assert result["response_length"] > 0
    
    @pytest.mark.asyncio
    async def test_llm_integration_manager_error_handling(self):
        """Test LLM integration manager error handling."""
        # Create a mock provider that raises an error
        mock_provider = Mock(spec=BaseLLMProvider)
        mock_provider.generate_response = AsyncMock(side_effect=Exception("Provider error"))
        mock_provider.get_provider_name.return_value = "ErrorProvider"
        mock_provider.get_model_info.return_value = {"name": "error-model"}
        
        manager = LLMIntegrationManager(mock_provider)
        
        # Should raise LLMError
        with pytest.raises(LLMError, match="Unexpected error in LLM integration"):
            await manager.generate_response("test", "context")


if __name__ == "__main__":
    pytest.main([__file__])