# LLM Integration Interface Guide

This guide explains how to use the new LLM integration interface in the RAG system.

## Overview

The LLM integration interface provides a flexible, extensible way to integrate various Large Language Model (LLM) providers with the RAG system. It includes:

- **Protocol-based interface**: Ensures consistent API across providers
- **Placeholder provider**: Ready-to-use mock LLM for development/testing
- **Integration manager**: Unified provider management and configuration
- **Seamless RAG integration**: Works directly with the existing RAG pipeline

## Core Components

### 1. LLMInterface Protocol

Defines the contract that all LLM providers must implement:

```python
class LLMInterface(Protocol):
    async def generate_response(self, prompt: str, context: str, **kwargs) -> str:
        """Generate a response using the LLM."""
        ...
```

### 2. BaseLLMProvider (Abstract Base Class)

Base class for implementing LLM providers:

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate_response(self, prompt: str, context: str, **kwargs) -> str:
        """Generate a response using the LLM."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the LLM provider."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the model being used."""
        pass
```

### 3. PlaceholderLLMProvider

Ready-to-use mock LLM provider for development and testing:

```python
from rag import PlaceholderLLMProvider

# Create with default template
provider = PlaceholderLLMProvider()

# Create with custom template
provider = PlaceholderLLMProvider(
    response_template="Custom response for '{prompt}' with {sources_count} sources"
)

# Generate response
response = await provider.generate_response(
    prompt="What is AI?",
    context="AI context here...",
    sources_count=3,
    context_length=150
)
```

### 4. LLMIntegrationManager

Manages LLM providers and provides unified interface:

```python
from rag import LLMIntegrationManager, PlaceholderLLMProvider

# Create with default provider
manager = LLMIntegrationManager()

# Create with custom provider
custom_provider = PlaceholderLLMProvider()
manager = LLMIntegrationManager(custom_provider)

# Generate response
result = await manager.generate_response(
    prompt="User question",
    context="Retrieved context",
    config={"temperature": 0.7, "max_tokens": 1000}
)

# Switch providers
new_provider = MyCustomLLMProvider()
manager.set_provider(new_provider)
```

## Creating Custom LLM Providers

### Example: OpenAI Provider

```python
import openai
from rag import BaseLLMProvider, LLMError

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def generate_response(self, prompt: str, context: str, **kwargs) -> str:
        try:
            # Create messages for OpenAI API
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": prompt}
            ]
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise LLMError(f"OpenAI API error: {str(e)}")
    
    def get_provider_name(self) -> str:
        return "OpenAI"
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model,
            "type": "openai",
            "provider": "OpenAI"
        }
```

### Example: Anthropic Provider

```python
import anthropic
from rag import BaseLLMProvider, LLMError

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
    
    async def generate_response(self, prompt: str, context: str, **kwargs) -> str:
        try:
            # Format prompt for Anthropic
            full_prompt = f"{context}\n\nHuman: {prompt}\n\nAssistant:"
            
            # Call Anthropic API
            response = await self.client.completions.create(
                model=self.model,
                prompt=full_prompt,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens_to_sample=kwargs.get("max_tokens", 1000)
            )
            
            return response.completion
            
        except Exception as e:
            raise LLMError(f"Anthropic API error: {str(e)}")
    
    def get_provider_name(self) -> str:
        return "Anthropic"
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model,
            "type": "anthropic",
            "provider": "Anthropic"
        }
```

## Integration with RAG Service

### Basic Usage

```python
from rag import RAGService, LLMIntegrationManager, PlaceholderLLMProvider
from search import VectorSearchService

# Create components
search_service = VectorSearchService()
llm_manager = LLMIntegrationManager(PlaceholderLLMProvider())
rag_service = RAGService(search_service, llm_manager)

# Generate RAG response
response = await rag_service.generate_response(
    query="What is machine learning?",
    top_k=5,
    similarity_threshold=0.7
)

print(f"Answer: {response['answer']}")
print(f"Sources: {len(response['sources'])}")
print(f"LLM Provider: {response['llm_provider']}")
```

### Provider Management

```python
# Get current provider info
provider_info = rag_service.get_llm_provider_info()
print(f"Current provider: {provider_info['provider_name']}")

# Switch to custom provider
openai_provider = OpenAIProvider(api_key="your-api-key")
rag_service.set_llm_provider(openai_provider)

# Configure LLM defaults
rag_service.configure_llm_defaults({
    "temperature": 0.3,
    "max_tokens": 1500,
    "timeout": 45.0
})
```

## Configuration Options

### LLM Manager Default Configuration

```python
manager = LLMIntegrationManager()
manager.default_config = {
    "max_tokens": 1000,      # Maximum tokens to generate
    "temperature": 0.7,      # Sampling temperature
    "timeout": 30.0          # Request timeout in seconds
}
```

### Per-Request Configuration

```python
# Override defaults for specific requests
result = await manager.generate_response(
    prompt="Question",
    context="Context",
    config={
        "temperature": 0.5,   # Override default
        "max_tokens": 2000,   # Override default
        "custom_param": "value"  # Provider-specific parameter
    }
)
```

## Error Handling

The interface provides structured error handling:

```python
from rag import LLMError, RAGError

try:
    response = await rag_service.generate_response("query")
except LLMError as e:
    print(f"LLM generation failed: {e}")
except RAGError as e:
    print(f"RAG pipeline failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

### 1. Provider Selection

- Use `PlaceholderLLMProvider` for development and testing
- Implement custom providers for production LLM services
- Consider fallback providers for reliability

### 2. Configuration Management

- Set appropriate defaults in the LLM manager
- Use per-request configuration for specific needs
- Monitor token usage and costs

### 3. Error Handling

- Always handle `LLMError` exceptions
- Implement retry logic for transient failures
- Log errors for debugging and monitoring

### 4. Performance Optimization

- Cache LLM providers to avoid re-initialization
- Use connection pooling for HTTP-based providers
- Monitor response times and adjust timeouts

## Testing

### Unit Testing with Mock Providers

```python
import pytest
from unittest.mock import Mock, AsyncMock
from rag import BaseLLMProvider, LLMIntegrationManager

class MockLLMProvider(BaseLLMProvider):
    def __init__(self, response: str = "Mock response"):
        self.response = response
    
    async def generate_response(self, prompt: str, context: str, **kwargs) -> str:
        return self.response
    
    def get_provider_name(self) -> str:
        return "MockLLM"
    
    def get_model_info(self) -> Dict[str, Any]:
        return {"name": "mock-model", "type": "test"}

@pytest.mark.asyncio
async def test_rag_with_mock_llm():
    mock_provider = MockLLMProvider("Test response")
    llm_manager = LLMIntegrationManager(mock_provider)
    
    result = await llm_manager.generate_response("test", "context")
    assert result["response"] == "Test response"
    assert result["provider_name"] == "MockLLM"
```

## Migration from Previous Implementation

If you're upgrading from the previous hardcoded LLM implementation:

1. **Replace direct LLM calls** with the integration manager
2. **Update RAG service initialization** to include LLM manager
3. **Update tests** to use the new interface
4. **Configure providers** according to your needs

### Before (Old Implementation)

```python
# Old hardcoded approach
llm_response = f"Hardcoded response for: {query}"
```

### After (New Implementation)

```python
# New flexible approach
llm_result = await self.llm_manager.generate_response(
    prompt=query,
    context=optimized_context,
    config=llm_config
)
response_text = llm_result["response"]
```

## Demo Script

Run the included demo to see all features in action:

```bash
python backend/llm_integration_demo.py
```

This demonstrates:
- PlaceholderLLMProvider usage
- Custom provider implementation
- LLMIntegrationManager features
- RAG service integration
- Error handling

## Next Steps

1. **Implement production providers** for your chosen LLM services
2. **Configure monitoring** for LLM usage and performance
3. **Set up fallback strategies** for reliability
4. **Optimize configurations** based on your use cases
5. **Add custom providers** for specialized models or services

The LLM integration interface is designed to be extensible and production-ready. You can now easily integrate with any LLM service while maintaining a consistent, testable interface throughout your RAG system.