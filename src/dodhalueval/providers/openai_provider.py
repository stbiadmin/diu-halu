"""OpenAI provider implementation for DoDHaluEval.

This module implements the OpenAI API provider with retry logic,
rate limiting, and error handling.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from dodhalueval.providers.base import (
    BaseLLMProvider,
    GenerationParameters,
    GenerationResult,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
    ContentFilterError,
    LLMProviderError
)
from dodhalueval.models.config import APIConfig


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, config: APIConfig):
        super().__init__(config)
        
        if not config.api_key:
            raise AuthenticationError("OpenAI API key is required")
        
        # Initialize OpenAI client
        self.client = openai.AsyncClient(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=0  # We handle retries ourselves
        )
        
        # Model mappings
        self.model_mappings = {
            'gpt-4': 'gpt-4',
            'gpt-4-turbo': 'gpt-4-turbo-preview',
            'gpt-3.5-turbo': 'gpt-3.5-turbo',
            'gpt-3.5': 'gpt-3.5-turbo',
            'gpt-4o': 'gpt-4o',
            'gpt-4o-mini': 'gpt-4o-mini'
        }
        
        # Pricing information (per 1K tokens)
        self.pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-turbo-preview': {'input': 0.01, 'output': 0.03},
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002}
        }
    
    def _get_model_name(self, model: Optional[str] = None) -> str:
        """Get the actual model name to use."""
        model_name = model or self.config.model
        return self.model_mappings.get(model_name, model_name)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.InternalServerError
        ))
    )
    async def generate(
        self,
        prompt: str,
        params: Optional[GenerationParameters] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate a response using OpenAI's API."""
        if params is None:
            params = GenerationParameters()
        
        model_name = self._get_model_name(model)
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare parameters
        api_params = {
            'model': model_name,
            'messages': messages,
            'temperature': params.temperature,
            'max_tokens': params.max_tokens,
            'top_p': params.top_p,
            'frequency_penalty': params.frequency_penalty,
            'presence_penalty': params.presence_penalty
        }
        
        if params.stop_sequences:
            api_params['stop'] = params.stop_sequences
        
        if params.seed is not None:
            api_params['seed'] = params.seed
        
        # Add any additional parameters
        api_params.update(kwargs)
        
        try:
            response = await self.client.chat.completions.create(**api_params)
            
            # Extract response data
            choice = response.choices[0]
            content = choice.message.content or ""
            
            # Update rate limit info if available
            if hasattr(response, 'headers'):
                self._rate_limit_info = self._extract_rate_limit_info(response.headers)
            
            return GenerationResult.from_response(
                text=content,
                model=model_name,
                provider=self.provider_name,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                finish_reason=choice.finish_reason or "stop",
                generation_params=self._create_generation_params(params),
                metadata={
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                    'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                    'response_id': response.id,
                    'model_used': response.model
                }
            )
        
        except openai.AuthenticationError as e:
            raise AuthenticationError(f"OpenAI authentication failed: {e}")
        
        except openai.PermissionDeniedError as e:
            raise AuthenticationError(f"OpenAI permission denied: {e}")
        
        except openai.NotFoundError as e:
            raise ModelNotFoundError(f"OpenAI model not found: {e}")
        
        except openai.RateLimitError as e:
            # Extract retry-after from headers if available
            retry_after = getattr(e.response, 'headers', {}).get('retry-after')
            if retry_after:
                retry_after = int(retry_after)
            raise RateLimitError(f"OpenAI rate limit exceeded: {e}", retry_after)
        
        except openai.BadRequestError as e:
            # Check if it's a content filter issue
            if "content_filter" in str(e).lower():
                raise ContentFilterError(f"Content filtered by OpenAI: {e}")
            raise LLMProviderError(f"OpenAI bad request: {e}")
        
        except Exception as e:
            raise LLMProviderError(f"OpenAI API error: {e}")
    
    async def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models."""
        try:
            models = await self.client.models.list()
            # Filter for chat models
            chat_models = [
                model.id for model in models.data
                if any(prefix in model.id for prefix in ['gpt-3.5', 'gpt-4'])
            ]
            return sorted(chat_models)
        except Exception as e:
            self.logger.warning(f"Failed to fetch OpenAI models: {e}")
            # Return known models as fallback
            return list(self.model_mappings.keys())
    
    async def validate_model(self, model_name: str) -> bool:
        """Validate that an OpenAI model is available."""
        try:
            available_models = await self.get_available_models()
            actual_model = self._get_model_name(model_name)
            return actual_model in available_models or model_name in self.model_mappings
        except Exception:
            # If we can't fetch models, assume common models are available
            return model_name in self.model_mappings
    
    async def estimate_cost(
        self,
        prompts: List[str],
        params: Optional[GenerationParameters] = None
    ) -> Dict[str, float]:
        """Estimate cost for OpenAI API calls."""
        if params is None:
            params = GenerationParameters()
        
        model_name = self._get_model_name()
        pricing = self.pricing.get(model_name, {'input': 0.01, 'output': 0.03})
        
        # Rough token estimation (1 token ≈ 0.75 words)
        total_input_tokens = sum(len(prompt.split()) * 1.33 for prompt in prompts)
        total_output_tokens = len(prompts) * params.max_tokens
        
        input_cost = (total_input_tokens / 1000) * pricing['input']
        output_cost = (total_output_tokens / 1000) * pricing['output']
        total_cost = input_cost + output_cost
        
        return {
            'estimated_input_tokens': total_input_tokens,
            'estimated_output_tokens': total_output_tokens,
            'estimated_input_cost_usd': input_cost,
            'estimated_output_cost_usd': output_cost,
            'estimated_total_cost_usd': total_cost,
            'model': model_name,
            'pricing_per_1k_tokens': pricing
        }
    
    async def batch_generate_with_smart_batching(
        self,
        prompts: List[str],
        params: Optional[GenerationParameters] = None,
        batch_size: int = 10,
        delay_between_batches: float = 1.0,
        **kwargs
    ) -> List[GenerationResult]:
        """Generate responses with smart batching to avoid rate limits."""
        if not prompts:
            return []
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            self.logger.info(f"Processing batch {i // batch_size + 1}/{(len(prompts) - 1) // batch_size + 1}")
            
            # Generate batch
            batch_results = await self.batch_generate(
                batch,
                params,
                max_concurrent=min(5, len(batch)),  # Limit concurrency
                **kwargs
            )
            all_results.extend(batch_results)
            
            # Delay between batches if not the last batch
            if i + batch_size < len(prompts):
                await asyncio.sleep(delay_between_batches)
        
        return all_results
    
    async def generate_with_system_prompt(
        self,
        prompt: str,
        system_prompt: str,
        params: Optional[GenerationParameters] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate response with a system prompt."""
        return await self.generate(
            prompt=prompt,
            params=params,
            system_prompt=system_prompt,
            **kwargs
        )
    
    async def generate_structured_output(
        self,
        prompt: str,
        response_format: Dict[str, Any],
        params: Optional[GenerationParameters] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate structured output (JSON mode)."""
        # Add response format to kwargs
        kwargs['response_format'] = response_format
        
        return await self.generate(prompt, params, **kwargs)
    
    async def close(self):
        """Close the OpenAI client."""
        if self.client:
            await self.client.close()
        await super().close()