"""Base LLM provider interface for DoDHaluEval.

This module defines the abstract base class for all LLM providers,
ensuring consistent interfaces across different API providers.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from dodhalueval.utils.logger import get_logger
from dodhalueval.models.config import APIConfig


@dataclass
class GenerationParameters:
    """Parameters for text generation."""
    
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    """Result from text generation."""
    
    text: str
    model: str
    provider: str
    tokens_used: int
    finish_reason: str
    generation_params: Dict[str, Any]
    metadata: Dict[str, Any]
    generated_at: datetime
    
    @classmethod
    def from_response(
        cls,
        text: str,
        model: str,
        provider: str,
        tokens_used: int = 0,
        finish_reason: str = "stop",
        generation_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'GenerationResult':
        """Create result from response data."""
        return cls(
            text=text,
            model=model,
            provider=provider,
            tokens_used=tokens_used,
            finish_reason=finish_reason,
            generation_params=generation_params or {},
            metadata=metadata or {},
            generated_at=datetime.now()
        )


@dataclass
class RateLimitInfo:
    """Rate limiting information."""
    
    requests_per_minute: int = 0
    tokens_per_minute: int = 0
    remaining_requests: int = 0
    remaining_tokens: int = 0
    reset_time: Optional[datetime] = None


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class RateLimitError(LLMProviderError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(LLMProviderError):
    """Raised when authentication fails."""
    pass


class ModelNotFoundError(LLMProviderError):
    """Raised when requested model is not available."""
    pass


class ContentFilterError(LLMProviderError):
    """Raised when content is filtered by the provider."""
    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._rate_limit_info = RateLimitInfo()
        self._session = None
    
    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self.__class__.__name__.lower().replace('provider', '')
    
    @property
    def rate_limit_info(self) -> RateLimitInfo:
        """Get current rate limit information."""
        return self._rate_limit_info
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        params: Optional[GenerationParameters] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate a single response from a prompt.
        
        Args:
            prompt: The input prompt
            params: Generation parameters
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generation result
            
        Raises:
            LLMProviderError: On provider-specific errors
            RateLimitError: When rate limits are exceeded
            AuthenticationError: When authentication fails
        """
        pass
    
    @abstractmethod
    async def batch_generate(
        self,
        prompts: List[str],
        params: Optional[GenerationParameters] = None,
        max_concurrent: int = 5,
        **kwargs
    ) -> List[GenerationResult]:
        """Generate responses for multiple prompts with rate limiting.
        
        Args:
            prompts: List of input prompts
            params: Generation parameters
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of generation results
            
        Raises:
            LLMProviderError: On provider-specific errors
        """
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[str]:
        """Get list of available models for this provider.
        
        Returns:
            List of model names
        """
        pass
    
    @abstractmethod
    async def validate_model(self, model_name: str) -> bool:
        """Validate that a model is available.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            True if model is available
        """
        pass
    
    async def health_check(self) -> bool:
        """Check if the provider is healthy and responsive.
        
        Returns:
            True if provider is healthy
        """
        try:
            # Simple test generation
            result = await self.generate(
                "Test",
                GenerationParameters(max_tokens=1, temperature=0.0)
            )
            return len(result.text) > 0
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False
    
    async def estimate_cost(
        self,
        prompts: List[str],
        params: Optional[GenerationParameters] = None
    ) -> Dict[str, float]:
        """Estimate the cost for generating responses to prompts.
        
        Args:
            prompts: List of prompts
            params: Generation parameters
            
        Returns:
            Dictionary with cost estimates
        """
        # Default implementation - providers should override with actual pricing
        return {
            'estimated_input_tokens': sum(len(p.split()) * 1.3 for p in prompts),
            'estimated_output_tokens': len(prompts) * (params.max_tokens if params else 1000),
            'estimated_cost_usd': 0.0
        }
    
    async def close(self):
        """Clean up resources."""
        if self._session:
            await self._session.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class BaseLLMProvider(LLMProvider):
    """Base implementation with common functionality."""
    
    def __init__(self, config: APIConfig):
        super().__init__(config)
        self._semaphore = asyncio.Semaphore(config.max_retries or 5)
    
    async def batch_generate(
        self,
        prompts: List[str],
        params: Optional[GenerationParameters] = None,
        max_concurrent: int = 5,
        **kwargs
    ) -> List[GenerationResult]:
        """Default batch generation implementation."""
        if not prompts:
            return []
        
        # Limit concurrency
        semaphore = asyncio.Semaphore(min(max_concurrent, len(prompts)))
        
        async def generate_with_semaphore(prompt: str) -> GenerationResult:
            async with semaphore:
                return await self.generate(prompt, params, **kwargs)
        
        # Execute all generations
        tasks = [generate_with_semaphore(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to generate response for prompt {i}: {result}")
                # Create error result
                error_result = GenerationResult.from_response(
                    text="",
                    model=self.config.model,
                    provider=self.provider_name,
                    finish_reason="error",
                    metadata={"error": str(result)}
                )
                successful_results.append(error_result)
            else:
                successful_results.append(result)
        
        return successful_results
    
    def _create_generation_params(
        self,
        params: Optional[GenerationParameters] = None
    ) -> Dict[str, Any]:
        """Convert GenerationParameters to provider-specific format."""
        if params is None:
            params = GenerationParameters()
        
        return {
            'temperature': params.temperature,
            'max_tokens': params.max_tokens,
            'top_p': params.top_p,
            'frequency_penalty': params.frequency_penalty,
            'presence_penalty': params.presence_penalty,
            'stop': params.stop_sequences,
            'seed': params.seed
        }
    
    def _extract_rate_limit_info(self, headers: Dict[str, str]) -> RateLimitInfo:
        """Extract rate limit information from response headers."""
        return RateLimitInfo(
            requests_per_minute=int(headers.get('x-ratelimit-limit-requests', 0)),
            tokens_per_minute=int(headers.get('x-ratelimit-limit-tokens', 0)),
            remaining_requests=int(headers.get('x-ratelimit-remaining-requests', 0)),
            remaining_tokens=int(headers.get('x-ratelimit-remaining-tokens', 0))
        )


class MockLLMProvider(BaseLLMProvider):
    """Mock provider for testing purposes."""
    
    def __init__(self, config: Optional[APIConfig] = None):
        if config is None:
            from dodhalueval.models.config import APIConfig
            config = APIConfig(provider="mock", model="mock-model")
        super().__init__(config)
    
    async def generate(
        self,
        prompt: str,
        params: Optional[GenerationParameters] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate a mock response."""
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        # Detect if this is a JSON request (G-Eval, structured output, etc.)
        if any(keyword in prompt.lower() for keyword in ['json', 'evaluate', '"score"', 'hallucination']):
            # Return a proper JSON response for evaluation prompts
            mock_response = '''```json
{
    "score": 0.3,
    "confidence": 0.8,
    "explanation": "Mock evaluation response. This appears to be a reasonable response with minimal hallucination indicators.",
    "hallucination_types": [],
    "specific_issues": []
}
```'''
            self.logger.debug(f"Mock provider returning JSON response: {mock_response[:100]}...")
        else:
            # Generate a simple mock response for other prompts
            mock_response = f"Mock response to: {prompt[:50]}..."
        
        return GenerationResult.from_response(
            text=mock_response,
            model=self.config.model,
            provider=self.provider_name,
            tokens_used=len(mock_response.split()),
            finish_reason="stop",
            generation_params=self._create_generation_params(params)
        )
    
    async def get_available_models(self) -> List[str]:
        """Get mock available models."""
        return ["mock-model", "mock-large", "mock-small"]
    
    async def validate_model(self, model_name: str) -> bool:
        """Validate mock model."""
        return model_name in await self.get_available_models()