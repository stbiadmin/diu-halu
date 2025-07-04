"""Unit tests for base provider."""

import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from dodhalueval.providers.base import BaseLLMProvider
from dodhalueval.models.config import APIConfig


@pytest.mark.unit
class TestBaseLLMProvider:
    """Test cases for BaseLLMProvider."""

    def test_base_provider_is_abstract(self):
        """Test that BaseLLMProvider cannot be instantiated directly."""
        config = APIConfig(
            provider="mock",
            model="test-model",
            api_key="test-key"
        )
        with pytest.raises(TypeError):
            BaseLLMProvider(config)

    def test_concrete_provider_implementation(self):
        """Test concrete implementation of BaseLLMProvider."""
        
        class ConcreteProvider(BaseLLMProvider):
            def __init__(self, config: APIConfig):
                super().__init__(config)
                self.name = "concrete_provider"
            
            async def generate(self, prompt: str, **kwargs) -> Any:
                class MockResponse:
                    def __init__(self, text: str):
                        self.text = text
                        self.usage = {"prompt_tokens": 10, "completion_tokens": 20}
                return MockResponse(f"Response to: {prompt}")
            
            async def batch_generate(self, prompts, **kwargs):
                return [await self.generate(p, **kwargs) for p in prompts]
            
            async def get_available_models(self):
                return ["test-model"]
            
            async def validate_model(self, model_name: str):
                return model_name == "test-model"
        
        config = APIConfig(
            provider="mock",  # Use valid provider
            model="test-model",
            api_key="test-key"
        )
        
        provider = ConcreteProvider(config)
        assert provider.config == config
        assert provider.name == "concrete_provider"

    def test_provider_config_validation(self):
        """Test provider configuration validation."""
        
        class ValidatingProvider(BaseLLMProvider):
            def __init__(self, config: APIConfig):
                super().__init__(config)
                self.name = "validating_provider"
            
            async def generate(self, prompt: str, **kwargs) -> Any:
                return Mock(text="test response")
            
            async def get_available_models(self):
                return ["test-model"]
            
            async def validate_model(self, model_name: str):
                return model_name == "test-model"
            
            def has_valid_config(self) -> bool:
                required_fields = ["api_key", "model"]
                for field in required_fields:
                    if not getattr(self.config, field, None):
                        return False
                return True
        
        # Valid config
        valid_config = APIConfig(
            provider="openai",
            model="test-model",
            api_key="test-key"
        )
        provider = ValidatingProvider(valid_config)
        assert provider.has_valid_config() is True
        
        # Invalid config - missing API key
        invalid_config = APIConfig(
            provider="openai",
            model="test-model",
            api_key=""
        )
        invalid_provider = ValidatingProvider(invalid_config)
        assert invalid_provider.has_valid_config() is False

    @pytest.mark.asyncio
    async def test_provider_generate_method(self):
        """Test provider generate method."""
        
        class TestProvider(BaseLLMProvider):
            def __init__(self, config: APIConfig):
                super().__init__(config)
                self.name = "test_provider"
            
            async def generate(self, prompt: str, **kwargs) -> Any:
                class Response:
                    def __init__(self, prompt: str, provider_config, **params):
                        self.text = f"Generated response for: {prompt[:20]}..."
                        self.model = params.get("model", provider_config.model)
                        self.usage = {
                            "prompt_tokens": len(prompt.split()),
                            "completion_tokens": len(self.text.split())
                        }
                        self.params = params
                
                return Response(prompt, self.config, **kwargs)
            
            async def get_available_models(self):
                return ["test-model"]
            
            async def validate_model(self, model_name: str):
                return model_name == "test-model"
        
        config = APIConfig(provider="openai", model="test-model", api_key="test-key")
        provider = TestProvider(config)
        
        prompt = "What is the capital of France?"
        response = await provider.generate(prompt, temperature=0.7, max_tokens=100)
        
        assert hasattr(response, 'text')
        assert "Generated response for:" in response.text
        assert response.model == "test-model"
        assert response.usage["prompt_tokens"] > 0
        assert hasattr(response, 'params')
        assert response.params["temperature"] == 0.7

    def test_provider_rate_limiting(self):
        """Test provider rate limiting functionality."""
        
        class RateLimitedProvider(BaseLLMProvider):
            def __init__(self, config: APIConfig):
                super().__init__(config)
                self.name = "rate_limited_provider"
                self.call_count = 0
                self.max_calls_per_minute = 10
            
            async def generate(self, prompt: str, **kwargs) -> Any:
                import time
                current_time = time.time()
                
                # Simple rate limiting logic
                self.call_count += 1
                if self.call_count > self.max_calls_per_minute:
                    raise Exception("Rate limit exceeded")
                
                class Response:
                    def __init__(self, text: str):
                        self.text = text
                        self.usage = {"prompt_tokens": 5, "completion_tokens": 10}
                
                return Response(f"Response #{self.call_count}")
            
            async def get_available_models(self):
                return ["test-model"]
            
            async def validate_model(self, model_name: str):
                return model_name == "test-model"
        
        config = APIConfig(provider="openai", model="test-model", api_key="test-key")
        provider = RateLimitedProvider(config)
        
        # Test normal operation
        import asyncio
        for i in range(5):
            response = asyncio.run(provider.generate(f"Prompt {i}"))
            assert f"Response #{i+1}" == response.text
        
        # Test rate limit
        provider.call_count = 15  # Exceed limit
        with pytest.raises(Exception, match="Rate limit exceeded"):
            asyncio.run(provider.generate("Too many requests"))

    @pytest.mark.asyncio
    async def test_provider_error_handling(self):
        """Test provider error handling."""
        
        class ErrorProneProvider(BaseLLMProvider):
            def __init__(self, config: APIConfig, fail_mode: str = None):
                super().__init__(config)
                self.name = "error_prone_provider"
                self.fail_mode = fail_mode
            
            async def generate(self, prompt: str, **kwargs) -> Any:
                if self.fail_mode == "network":
                    raise ConnectionError("Network connection failed")
                elif self.fail_mode == "auth":
                    raise PermissionError("Authentication failed")
                elif self.fail_mode == "quota":
                    raise RuntimeError("API quota exceeded")
                
                return Mock(text="Success response", usage={"prompt_tokens": 1, "completion_tokens": 1})
            
            async def get_available_models(self):
                return ["test-model"]
            
            async def validate_model(self, model_name: str):
                return model_name == "test-model"
        
        config = APIConfig(provider="openai", model="test-model", api_key="test-key")
        
        # Test successful generation
        good_provider = ErrorProneProvider(config)
        response = await good_provider.generate("test prompt")
        assert response.text == "Success response"
        
        # Test network error
        network_provider = ErrorProneProvider(config, fail_mode="network")
        with pytest.raises(ConnectionError, match="Network connection failed"):
            await network_provider.generate("test prompt")
        
        # Test auth error
        auth_provider = ErrorProneProvider(config, fail_mode="auth")
        with pytest.raises(PermissionError, match="Authentication failed"):
            await auth_provider.generate("test prompt")
        
        # Test quota error
        quota_provider = ErrorProneProvider(config, fail_mode="quota")
        with pytest.raises(RuntimeError, match="API quota exceeded"):
            await quota_provider.generate("test prompt")

    def test_provider_retry_logic(self):
        """Test provider retry functionality."""
        
        class RetryableProvider(BaseLLMProvider):
            def __init__(self, config: APIConfig):
                super().__init__(config)
                self.name = "retryable_provider"
                self.attempt_count = 0
                self.max_retries = 3
            
            async def generate(self, prompt: str, **kwargs) -> Any:
                self.attempt_count += 1
                
                # Fail first two attempts, succeed on third
                if self.attempt_count < 3:
                    raise ConnectionError(f"Temporary failure (attempt {self.attempt_count})")
                
                return Mock(
                    text=f"Success after {self.attempt_count} attempts",
                    usage={"prompt_tokens": 1, "completion_tokens": 1}
                )
            
            async def get_available_models(self):
                return ["test-model"]
            
            async def validate_model(self, model_name: str):
                return model_name == "test-model"
            
            async def generate_with_retry(self, prompt: str, **kwargs) -> Any:
                """Implement retry logic."""
                import asyncio
                for attempt in range(self.max_retries):
                    try:
                        return await self.generate(prompt, **kwargs)
                    except ConnectionError as e:
                        if attempt == self.max_retries - 1:
                            raise e
                        await asyncio.sleep(0.1)  # Brief delay between retries
        
        config = APIConfig(provider="openai", model="test-model", api_key="test-key")
        provider = RetryableProvider(config)
        
        import asyncio
        response = asyncio.run(provider.generate_with_retry("test prompt"))
        assert "Success after 3 attempts" == response.text
        assert provider.attempt_count == 3

    def test_provider_batch_support(self):
        """Test provider batch generation support."""
        
        class BatchProvider(BaseLLMProvider):
            def __init__(self, config: APIConfig):
                super().__init__(config)
                self.name = "batch_provider"
            
            async def generate(self, prompt: str, **kwargs) -> Any:
                return Mock(
                    text=f"Single response to: {prompt[:10]}...",
                    usage={"prompt_tokens": 5, "completion_tokens": 10}
                )
            
            async def generate_batch(self, prompts: list, **kwargs) -> list:
                """Batch generation for efficiency."""
                import asyncio
                tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
                return await asyncio.gather(*tasks)
            
            async def get_available_models(self):
                return ["test-model"]
            
            async def validate_model(self, model_name: str):
                return model_name == "test-model"
        
        config = APIConfig(provider="openai", model="test-model", api_key="test-key")
        provider = BatchProvider(config)
        
        prompts = [f"Prompt {i}" for i in range(3)]
        
        import asyncio
        responses = asyncio.run(provider.generate_batch(prompts))
        
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert f"Prompt {i}" in response.text

    def test_provider_context_management(self):
        """Test provider context management and cleanup."""
        
        class ContextManagedProvider(BaseLLMProvider):
            def __init__(self, config: APIConfig):
                super().__init__(config)
                self.name = "context_managed_provider"
                self.is_connected = False
                self.cleanup_called = False
            
            async def __aenter__(self):
                self.is_connected = True
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.cleanup_called = True
                self.is_connected = False
            
            async def generate(self, prompt: str, **kwargs) -> Any:
                if not self.is_connected:
                    raise RuntimeError("Provider not connected")
                
                return Mock(text="Connected response", usage={"prompt_tokens": 1, "completion_tokens": 1})
            
            async def get_available_models(self):
                return ["test-model"]
            
            async def validate_model(self, model_name: str):
                return model_name == "test-model"
        
        config = APIConfig(provider="openai", model="test-model", api_key="test-key")
        
        async def test_context():
            async with ContextManagedProvider(config) as provider:
                assert provider.is_connected is True
                response = await provider.generate("test")
                assert response.text == "Connected response"
            assert provider.cleanup_called is True
            assert provider.is_connected is False
        
        import asyncio
        asyncio.run(test_context())