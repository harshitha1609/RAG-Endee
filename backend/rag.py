# RAG logic
# Handles retrieval augmented generation pipeline

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol

from search import (
    EmbeddingError,
    EndeeConnectionError,
    SearchError,
    ValidationError,
    VectorSearchService,
)

logger = logging.getLogger(__name__)

class RAGError(Exception):
    """Base exception for RAG-related errors."""
    pass

class ContextCombinationError(RAGError):
    """Exception for context combination failures."""
    pass

class LLMError(RAGError):
    """Exception for LLM-related errors."""
    pass

class LLMInterface(Protocol):
    """Protocol defining the interface for LLM integration."""
    
    async def generate_response(
        self, prompt: str, context: str, **kwargs: Any
    ) -> str:
        """Generate a response using the LLM.
        
        Args:
            prompt: The user's original query/prompt
            context: The retrieved and optimized context
            **kwargs: Additional parameters for LLM configuration
            
        Returns:
            Generated response string
            
        Raises:
            LLMError: If response generation fails
        """
        ...

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate_response(
        self, prompt: str, context: str, **kwargs: Any
    ) -> str:
        """Generate a response using the LLM.
        
        Args:
            prompt: The user's original query/prompt
            context: The retrieved and optimized context
            **kwargs: Additional parameters for LLM configuration
            
        Returns:
            Generated response string
            
        Raises:
            LLMError: If response generation fails
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the LLM provider."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the model being used."""
        pass

class PlaceholderLLMProvider(BaseLLMProvider):
    """Placeholder LLM provider for development and testing.
    
    This provider generates mock responses and serves as a template
    for implementing actual LLM integrations (OpenAI, Anthropic, etc.).
    """
    
    def __init__(self, response_template: Optional[str] = None) -> None:
        """Initialize the placeholder LLM provider.
        
        Args:
            response_template: Optional template for generating responses
        """
        self.response_template = response_template or (
            "Based on the retrieved information, I can provide the following "
            "response to your query: '{prompt}'. The system found {sources_count} "
            "relevant sources with {context_length} characters of context. "
            "[This is a placeholder response - actual LLM integration not yet implemented]"
        )
        self.provider_name = "PlaceholderLLM"
        self.model_info = {
            "name": "placeholder-model-v1",
            "type": "mock",
            "max_tokens": 4096,
            "temperature": 0.7
        }
    
    async def generate_response(
        self, prompt: str, context: str, **kwargs: Any
    ) -> str:
        """Generate a placeholder response.
        
        Args:
            prompt: The user's original query/prompt
            context: The retrieved and optimized context
            **kwargs: Additional parameters (sources_count, context_length, etc.)
            
        Returns:
            Generated placeholder response string
            
        Raises:
            LLMError: If response generation fails
        """
        try:
            query_preview = (
                f"'{prompt[:50]}{'...' if len(prompt) > 50 else ''}'"
            )
            logger.debug(f"Generating placeholder response for prompt: {query_preview}")
            
            # Extract metadata from kwargs
            sources_count = kwargs.get('sources_count', 0)
            context_length = kwargs.get('context_length', len(context))
            
            # Generate response using template
            response = self.response_template.format(
                prompt=prompt,
                sources_count=sources_count,
                context_length=context_length
            )
            
            # Add context preview if available
            if context and len(context.strip()) > 0:
                context_preview = (
                    context[:200] + "..." if len(context) > 200 else context
                )
                response += f"\n\nContext preview: {context_preview}"
            
            logger.debug(f"Generated placeholder response: {len(response)} characters")
            return response
            
        except Exception as e:
            error_msg = f"Failed to generate placeholder response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise LLMError(error_msg)
    
    def get_provider_name(self) -> str:
        """Return the name of the LLM provider."""
        return self.provider_name
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the model being used."""
        return self.model_info.copy()

class LLMIntegrationManager:
    """Manager class for LLM integration that handles provider selection and configuration.
    
    This class provides a unified interface for different LLM providers and handles
    provider switching, configuration, and error handling.
    """
    
    def __init__(self, provider: Optional[BaseLLMProvider] = None) -> None:
        """Initialize the LLM integration manager.
        
        Args:
            provider: LLM provider instance. If None, uses PlaceholderLLMProvider.
        """
        self.provider = provider or PlaceholderLLMProvider()
        self.default_config = {
            "max_tokens": 1000,
            "temperature": 0.7,
            "timeout": 30.0
        }
        logger.info(
            f"Initialized LLM integration with provider: "
            f"{self.provider.get_provider_name()}"
        )
    
    def set_provider(self, provider: BaseLLMProvider) -> None:
        """Set a new LLM provider.
        
        Args:
            provider: New LLM provider instance
        """
        old_provider = self.provider.get_provider_name()
        self.provider = provider
        logger.info(
            f"Switched LLM provider from {old_provider} to "
            f"{provider.get_provider_name()}"
        )
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current LLM provider.
        
        Returns:
            Dictionary containing provider name and model information
        """
        return {
            "provider_name": self.provider.get_provider_name(),
            "model_info": self.provider.get_model_info()
        }
    
    async def generate_response(
        self,
        prompt: str,
        context: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a response using the configured LLM provider.
        
        Args:
            prompt: The user's original query/prompt
            context: The retrieved and optimized context
            config: Optional configuration overrides
            
        Returns:
            Dictionary containing the response and metadata
            
        Raises:
            LLMError: If response generation fails
        """
        try:
            # Merge configuration
            final_config = {**self.default_config}
            if config:
                final_config.update(config)
            
            logger.debug(
                f"Generating response with provider: "
                f"{self.provider.get_provider_name()}"
            )
            
            # Generate response using provider
            response_text = await self.provider.generate_response(
                prompt=prompt,
                context=context,
                **final_config
            )
            
            # Return structured response
            result = {
                "response": response_text,
                "provider_name": self.provider.get_provider_name(),
                "model_info": self.provider.get_model_info(),
                "config_used": final_config,
                "prompt_length": len(prompt),
                "context_length": len(context),
                "response_length": len(response_text)
            }
            
            logger.info(
                f"Generated response: {len(response_text)} chars using "
                f"{self.provider.get_provider_name()}"
            )
            return result
            
        except LLMError:
            # Re-raise LLM errors as-is
            raise
        except Exception as e:
            error_msg = f"Unexpected error in LLM integration: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise LLMError(error_msg)

class RAGService:
    """Service for Retrieval Augmented Generation using retrieved context."""
    
    def __init__(
        self,
        search_service: Optional[VectorSearchService] = None,
        llm_manager: Optional[LLMIntegrationManager] = None,
        max_context_length: int = 4000
    ) -> None:
        """Initialize the RAG service.
        
        Args:
            search_service: Optional search service instance. If None, creates a new one.
            llm_manager: Optional LLM integration manager. If None, creates one with placeholder provider.
            max_context_length: Maximum context length in characters (default: 4000)
        """
        self.search_service = search_service or VectorSearchService()
        self.llm_manager = llm_manager or LLMIntegrationManager()
        self.max_context_length = max_context_length  # Maximum context length in characters
        self.context_separator = "\n\n---\n\n"  # Separator between document chunks
        self.source_prefix = "Source"  # Prefix for source identification
    
    def _validate_inputs(self, query: str, top_k: int) -> None:
        """
        Validate RAG input parameters.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Raises:
            ValidationError: If inputs are invalid
        """
        if not query or not isinstance(query, str):
            raise ValidationError("Query must be a non-empty string")
        
        if not query.strip():
            raise ValidationError("Query cannot be empty or contain only whitespace")
        
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValidationError("top_k must be a positive integer")
        
        if top_k > 20:
            raise ValidationError("top_k cannot exceed 20 for RAG context")
    
    def _combine_retrieved_context(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine retrieved documents into coherent context for LLM consumption with memory optimization.
        
        This method implements a memory-efficient context combination algorithm that:
        1. Sorts documents by relevance score (highest first)
        2. Combines documents while respecting context length limits
        3. Uses streaming approach to avoid loading all content into memory at once
        4. Maintains source attribution for each document chunk
        5. Handles truncation gracefully when context limit is exceeded
        6. Provides detailed metadata about the combination process
        
        Memory optimizations:
        - Processes documents one at a time instead of loading all content
        - Uses string builders for efficient concatenation
        - Releases processed document content from memory immediately
        - Limits intermediate string allocations
        
        The algorithm prioritizes the most relevant documents and stops adding
        content when the context length limit would be exceeded, ensuring
        the LLM receives the most valuable information within its constraints.
        
        Args:
            search_results: List of search results from vector search
            
        Returns:
            Dictionary containing combined context and source information
            
        Raises:
            ContextCombinationError: If context combination fails
        """
        try:
            if not search_results:
                logger.warning("No search results provided for context combination")
                return {
                    "combined_context": "",
                    "sources_used": [],
                    "total_sources": 0,
                    "context_length": 0,
                    "truncated": False
                }
            
            logger.debug(f"Combining context from {len(search_results)} search results")
            
            # Initialize context combination state with memory-efficient approach
            combined_chunks = []
            sources_used = []
            current_length = 0
            truncated = False
            separator_length = len(self.context_separator)
            
            # CRITICAL: Sort results by relevance score (highest first)
            # This ensures the most relevant documents are included first
            # when context length limits force truncation
            sorted_results = sorted(search_results, key=lambda x: x.get('score', 0.0), reverse=True)
            
            # Process each document in order of relevance with memory optimization
            for i, result in enumerate(sorted_results):
                try:
                    # Extract document content and metadata (memory-efficient access)
                    content = result.get('content', '').strip()
                    document_id = result.get('document_id', f'unknown_{i}')
                    score = result.get('score', 0.0)
                    metadata = result.get('metadata', {})
                    
                    # Skip empty documents to avoid polluting context
                    if not content:
                        logger.warning(f"Empty content in search result {i}, skipping")
                        continue
                    
                    # Create source attribution for traceability (memory-efficient)
                    # This allows the LLM to reference specific sources in its response
                    source_info = {
                        'document_id': document_id,
                        'score': score,
                        'metadata': metadata,
                        'content_preview': content[:100] + '...' if len(content) > 100 else content
                    }
                    
                    # Format content chunk with clear source attribution
                    # The source label helps the LLM understand document boundaries
                    source_label = f"{self.source_prefix} {i + 1}"
                    if 'title' in metadata:
                        source_label += f" ({metadata['title']})"
                    
                    formatted_chunk = f"{source_label}:\n{content}"
                    
                    # CRITICAL: Check context length limit before adding chunk
                    # This prevents exceeding LLM context windows and ensures
                    # we can always fit the most important documents
                    chunk_length = len(formatted_chunk)
                    total_addition = chunk_length + (separator_length if combined_chunks else 0)
                    
                    if current_length + total_addition > self.max_context_length:
                        logger.info(f"Context length limit reached. Truncating at {i} sources out of {len(sorted_results)}")
                        truncated = True
                        break
                    
                    # Add chunk to combined context (memory-efficient)
                    combined_chunks.append(formatted_chunk)
                    sources_used.append(source_info)
                    current_length += total_addition
                    
                    logger.debug(f"Added source {i + 1}: {len(content)} chars, score: {score:.3f}")
                    
                    # Memory optimization: Clear large variables to free memory immediately
                    del content, formatted_chunk
                    
                except Exception as e:
                    logger.warning(f"Error processing search result {i}: {str(e)}")
                    continue
            
            # Combine all chunks into final context with memory-efficient join
            # The separator helps the LLM distinguish between different sources
            if combined_chunks:
                combined_context = self.context_separator.join(combined_chunks)
            else:
                logger.warning("No valid content found in search results")
                combined_context = ""
            
            # Memory optimization: Clear intermediate data structures
            del combined_chunks
            
            result = {
                "combined_context": combined_context,
                "sources_used": sources_used,
                "total_sources": len(sources_used),
                "context_length": len(combined_context),
                "truncated": truncated
            }
            
            logger.info(f"Context combination completed: {len(sources_used)} sources, {len(combined_context)} chars, truncated: {truncated}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to combine retrieved context: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ContextCombinationError(error_msg)
    
    def _optimize_context_for_llm(self, combined_context: str, query: str, sources_info: List[Dict[str, Any]] = None) -> str:
        """
        Optimize combined context for LLM consumption.
        
        This method implements a sophisticated context optimization strategy that
        transforms raw document chunks into a structured, LLM-friendly format.
        The optimization includes:
        
        1. Query term extraction and highlighting for better LLM attention
        2. Structured formatting with clear sections and instructions
        3. Enhanced readability through improved paragraph formatting
        4. Clear response guidelines to improve LLM output quality
        5. Fallback handling for optimization failures
        
        The structured format helps the LLM understand:
        - What task it needs to perform
        - What information is available
        - How to structure its response
        - What to do if information is insufficient
        
        Args:
            combined_context: Raw combined context from documents
            query: Original user query for context optimization
            sources_info: Optional list of source information for enhanced formatting
            
        Returns:
            Optimized context string formatted for optimal LLM processing
        """
        if not combined_context.strip():
            return ""
        
        try:
            # STEP 1: Extract key terms from query for context highlighting
            # This helps the LLM focus on the most relevant parts of the context
            query_terms = self._extract_key_terms(query)
            
            # STEP 2: Build structured context with clear sections
            # The structured approach helps LLMs understand the task better
            optimized_parts = []
            
            # Section 1: Clear instruction header
            # Provides explicit instructions to guide LLM behavior
            optimized_parts.append("=== INSTRUCTION ===")
            optimized_parts.append(f"Answer the user's question using ONLY the information provided below.")
            optimized_parts.append(f"If the provided information is insufficient, clearly state what additional information is needed.")
            optimized_parts.append("")
            
            # Section 2: User question with emphasis and key terms
            # Reinforces the original query and highlights important terms
            optimized_parts.append("=== USER QUESTION ===")
            optimized_parts.append(f"Question: {query}")
            if query_terms:
                optimized_parts.append(f"Key terms: {', '.join(query_terms)}")
            optimized_parts.append("")
            
            # Section 3: Enhanced context with better formatting
            # The core information section with improved readability
            optimized_parts.append("=== RETRIEVED INFORMATION ===")
            
            # CRITICAL: Process and enhance the combined context
            # This step improves readability and highlights relevant terms
            enhanced_context = self._enhance_context_formatting(combined_context, query_terms)
            optimized_parts.append(enhanced_context)
            optimized_parts.append("")
            
            # Section 4: Response guidelines
            # Explicit guidelines help ensure consistent, high-quality responses
            optimized_parts.append("=== RESPONSE GUIDELINES ===")
            optimized_parts.append("- Provide a comprehensive and accurate answer")
            optimized_parts.append("- Cite specific sources when referencing information")
            optimized_parts.append("- If information conflicts between sources, acknowledge the discrepancy")
            optimized_parts.append("- If the question cannot be fully answered, explain what's missing")
            optimized_parts.append("- Use clear, concise language appropriate for the user's question")
            
            # STEP 3: Combine all sections into final optimized context
            optimized_context = "\n".join(optimized_parts)
            
            logger.debug(f"Context optimized for LLM: {len(optimized_context)} characters, {len(query_terms)} key terms identified")
            return optimized_context
            
        except Exception as e:
            # FALLBACK: If optimization fails, use basic format to ensure functionality
            logger.warning(f"Error optimizing context for LLM, using basic format: {str(e)}")
            return self._create_basic_llm_context(combined_context, query)
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """
        Extract key terms from the user query for context highlighting.
        
        Args:
            query: User query string
            
        Returns:
            List of key terms extracted from the query
        """
        try:
            import re
            
            # Remove common stop words and extract meaningful terms
            stop_words = {
                'what', 'is', 'are', 'how', 'why', 'when', 'where', 'who', 'which', 'that',
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
                'after', 'above', 'below', 'between', 'among', 'can', 'could', 'should',
                'would', 'will', 'do', 'does', 'did', 'have', 'has', 'had', 'be', 'been',
                'being', 'am', 'is', 'are', 'was', 'were'
            }
            
            # Extract words (alphanumeric sequences)
            words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', query.lower())
            
            # Filter out stop words and short words, keep meaningful terms
            key_terms = [word for word in words if len(word) > 2 and word not in stop_words]
            
            # Limit to most relevant terms (max 10)
            return key_terms[:10]
            
        except Exception as e:
            logger.debug(f"Error extracting key terms: {str(e)}")
            return []
    
    def _enhance_context_formatting(self, combined_context: str, query_terms: List[str]) -> str:
        """
        Enhance the formatting of combined context for better LLM comprehension.
        
        This method implements advanced text formatting techniques to improve
        how the LLM processes and understands the retrieved context:
        
        1. Splits context by source separators to handle each document individually
        2. Extracts and formats source headers for clear attribution
        3. Breaks long paragraphs into readable chunks
        4. Highlights query terms to draw LLM attention to relevant content
        5. Adds visual separators to distinguish between sources
        
        The enhanced formatting helps the LLM:
        - Distinguish between different sources
        - Focus on query-relevant content
        - Process information in digestible chunks
        - Maintain source attribution in responses
        
        Args:
            combined_context: Raw combined context from documents
            query_terms: Key terms from the user query for highlighting
            
        Returns:
            Enhanced formatted context with improved structure and readability
        """
        try:
            if not combined_context.strip():
                return "No relevant information found."
            
            # STEP 1: Split context by source separators
            # This allows us to process each document individually
            sources = combined_context.split(self.context_separator)
            enhanced_sources = []
            
            # STEP 2: Process each source individually
            for i, source in enumerate(sources):
                if not source.strip():
                    continue
                
                # Extract source header and content
                lines = source.strip().split('\n')
                if not lines:
                    continue
                
                # CRITICAL: Separate header from content for proper formatting
                # The header contains source attribution, content has the actual text
                source_header = lines[0] if lines else f"Source {i+1}:"
                source_content = '\n'.join(lines[1:]) if len(lines) > 1 else ""
                
                if not source_content.strip():
                    continue
                
                # Format the source with improved structure
                enhanced_source = f"{source_header}\n"
                
                # STEP 3: Add content with improved formatting
                if source_content:
                    # Break long paragraphs for better readability
                    # This helps LLMs process information in manageable chunks
                    paragraphs = source_content.split('\n\n')
                    formatted_paragraphs = []
                    
                    for paragraph in paragraphs:
                        if paragraph.strip():
                            # CRITICAL: Highlight query terms for better LLM attention
                            # This helps the LLM focus on the most relevant parts
                            highlighted = self._highlight_query_terms(paragraph.strip(), query_terms)
                            # Add indentation to visually separate content from headers
                            formatted_paragraphs.append(f"  {highlighted}")
                    
                    enhanced_source += '\n'.join(formatted_paragraphs)
                
                enhanced_sources.append(enhanced_source)
            
            if not enhanced_sources:
                return "No relevant information found."
            
            # STEP 4: Join enhanced sources with clear visual separators
            # The separator line helps LLMs distinguish between different sources
            separator_line = '─' * 50
            return '\n\n' + f'\n\n{separator_line}\n\n'.join([''] + enhanced_sources + [''])
            
        except Exception as e:
            logger.debug(f"Error enhancing context formatting: {str(e)}")
            # Fallback to original context if enhancement fails
            return combined_context
    
    def _highlight_query_terms(self, text: str, query_terms: List[str]) -> str:
        """
        Highlight query terms in the text for better LLM attention.
        
        This method implements intelligent term highlighting that helps LLMs
        focus on the most relevant parts of the retrieved context. The highlighting
        strategy uses:
        
        1. Case-insensitive matching to catch variations
        2. Word boundary detection to avoid partial matches
        3. Uppercase transformation for LLM-friendly highlighting
        4. Length filtering to focus on meaningful terms
        
        The highlighting helps the LLM:
        - Quickly identify relevant information
        - Focus on query-related content
        - Make better connections between query and context
        
        Args:
            text: Text content to process and highlight
            query_terms: Terms extracted from the user query to highlight
            
        Returns:
            Text with query terms highlighted using uppercase formatting
        """
        try:
            if not query_terms:
                return text
            
            import re
            highlighted_text = text
            
            # CRITICAL: Process each query term for highlighting
            # We use uppercase highlighting as it's LLM-friendly and doesn't
            # interfere with text processing or tokenization
            for term in query_terms:
                if len(term) > 2:  # Only highlight meaningful terms (3+ characters)
                    # Use word boundaries to avoid partial matches
                    # Example: "test" won't match "testing" or "contest"
                    pattern = r'\b' + re.escape(term) + r'\b'
                    
                    # Case-insensitive replacement with uppercase highlighting
                    # This preserves the original word while making it stand out
                    highlighted_text = re.sub(
                        pattern, 
                        lambda m: m.group().upper(),  # Convert matched term to uppercase
                        highlighted_text, 
                        flags=re.IGNORECASE
                    )
            
            return highlighted_text
            
        except Exception as e:
            logger.debug(f"Error highlighting query terms: {str(e)}")
            # Fallback to original text if highlighting fails
            return text
    
    def set_llm_provider(self, provider: BaseLLMProvider) -> None:
        """
        Set a new LLM provider for the RAG service.
        
        Args:
            provider: New LLM provider instance
        """
        self.llm_manager.set_provider(provider)
        logger.info(f"RAG service updated to use LLM provider: {provider.get_provider_name()}")
    
    def get_llm_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the current LLM provider.
        
        Returns:
            Dictionary containing provider name and model information
        """
        return self.llm_manager.get_provider_info()
    
    def configure_llm_defaults(self, config: Dict[str, Any]) -> None:
        """
        Configure default LLM settings.
        
        Args:
            config: Dictionary of configuration parameters
        """
        self.llm_manager.default_config.update(config)
        logger.info(f"Updated LLM default configuration: {config}")
    
    def set_max_context_length(self, max_length: int) -> None:
        """
        Set the maximum context length for document combination.
        
        Args:
            max_length: Maximum context length in characters
            
        Raises:
            ValidationError: If max_length is invalid
        """
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValidationError("max_context_length must be a positive integer")
        
        if max_length > 50000:  # Reasonable upper limit
            raise ValidationError("max_context_length cannot exceed 50000 characters")
        
        old_length = self.max_context_length
        self.max_context_length = max_length
        logger.info(f"Updated max context length from {old_length} to {max_length} characters")
    
    def get_max_context_length(self) -> int:
        """
        Get the current maximum context length setting.
        
        Returns:
            Current maximum context length in characters
        """
        return self.max_context_length
    
    def _create_basic_llm_context(self, combined_context: str, query: str) -> str:
        """
        Create a basic LLM context as fallback when optimization fails.
        
        Args:
            combined_context: Raw combined context
            query: User query
            
        Returns:
            Basic formatted context for LLM
        """
        return f"""Based on the following retrieved information, please answer the user's question.

User Question: {query}

Retrieved Information:
{combined_context}

Please provide a comprehensive answer based on the retrieved information above. If the information is insufficient to fully answer the question, please indicate what additional information might be needed."""
    
    async def generate_response(self, query: str, top_k: int = 3, similarity_threshold: float = 0.0, max_context_length: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a response using RAG pipeline.
        
        This method implements the complete RAG pipeline:
        1. Retrieve relevant documents using vector search
        2. Combine retrieved documents into coherent context
        3. Optimize context for LLM consumption
        4. Return structured response with sources
        
        Args:
            query: User question/query
            top_k: Number of documents to retrieve for context
            similarity_threshold: Minimum similarity score for retrieved documents
            max_context_length: Optional override for maximum context length
            
        Returns:
            Dictionary containing generated response and sources
            
        Raises:
            ValidationError: If input validation fails
            SearchError: If document retrieval fails
            ContextCombinationError: If context combination fails
            RAGError: For other RAG pipeline failures
        """
        try:
            logger.info(f"Starting RAG pipeline for query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
            
            # Validate inputs
            self._validate_inputs(query, top_k)
            
            # Validate and set context length for this request
            original_max_length = self.max_context_length
            if max_context_length is not None:
                if not isinstance(max_context_length, int) or max_context_length <= 0:
                    raise ValidationError("max_context_length must be a positive integer")
                if max_context_length > 50000:
                    raise ValidationError("max_context_length cannot exceed 50000 characters")
                self.max_context_length = max_context_length
                logger.debug(f"Using custom context length: {max_context_length} characters")
            
            try:
                # Step 1: Retrieve relevant documents using vector search
                logger.debug("Step 1: Retrieving relevant documents")
                search_results = await self.search_service.search(
                    query=query,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold
                )
                
                if not search_results:
                    logger.warning("No relevant documents found for query")
                    return {
                        "query": query,
                        "answer": "I couldn't find any relevant information to answer your question. Please try rephrasing your query or check if documents have been ingested into the system.",
                        "sources": [],
                        "context_used": "",
                        "context_length": 0,
                        "sources_count": 0,
                        "truncated": False,
                        "max_context_length": self.max_context_length
                    }
                
                logger.debug(f"Retrieved {len(search_results)} relevant documents")
                
                # Step 2: Combine retrieved documents into coherent context
                logger.debug("Step 2: Combining retrieved context")
                context_info = self._combine_retrieved_context(search_results)
                
                # Step 3: Optimize context for LLM consumption
                logger.debug("Step 3: Optimizing context for LLM")
                optimized_context = self._optimize_context_for_llm(
                    context_info["combined_context"], 
                    query,
                    context_info["sources_used"]
                )
                
                # Step 4: Generate response using LLM integration
                logger.debug("Step 4: Generating response using LLM integration")
                
                # Prepare LLM configuration
                llm_config = {
                    "sources_count": context_info["total_sources"],
                    "context_length": context_info["context_length"],
                    "truncated": context_info["truncated"]
                }
                
                # Generate response using LLM integration manager
                llm_result = await self.llm_manager.generate_response(
                    prompt=query,
                    context=optimized_context,
                    config=llm_config
                )
                
                response = {
                    "query": query,
                    "answer": llm_result["response"],
                    "sources": context_info["sources_used"],
                    "context_used": optimized_context,
                    "context_length": context_info["context_length"],
                    "sources_count": context_info["total_sources"],
                    "truncated": context_info["truncated"],
                    "max_context_length": self.max_context_length,
                    "llm_provider": llm_result["provider_name"],
                    "llm_model": llm_result["model_info"]
                }
                
                logger.info(f"RAG pipeline completed successfully: {context_info['total_sources']} sources, {context_info['context_length']} context chars")
                return response
                
            finally:
                # Restore original context length if it was temporarily changed
                if max_context_length is not None:
                    self.max_context_length = original_max_length
            
        except ValidationError as e:
            logger.error(f"Validation error in RAG pipeline: {str(e)}")
            raise ValidationError(str(e))  # Keep original validation messages as they're user-facing
            
        except (SearchError, EmbeddingError, EndeeConnectionError) as e:
            logger.error(f"Search error in RAG pipeline: {str(e)}")
            raise SearchError("Document retrieval failed. The search service is temporarily unavailable. Please try again in a few moments.")
            
        except ContextCombinationError as e:
            logger.error(f"Context combination error in RAG pipeline: {str(e)}")
            raise ContextCombinationError("Context processing failed. Please try with a simpler query or contact support if the problem persists.")
            
        except LLMError as e:
            logger.error(f"LLM error in RAG pipeline: {str(e)}")
            raise LLMError("Response generation failed. The language model service is temporarily unavailable. Please try again in a few moments.")
            
        except Exception as e:
            error_msg = f"Unexpected error in RAG pipeline: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RAGError("RAG pipeline failed due to an unexpected error. Please try again or contact support if the problem persists.")