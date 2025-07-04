"""Fireworks AI provider implementation for DoDHaluEval.

This module implements the Fireworks AI API provider for accessing
open-source models like Llama-2, Mixtral, and others.
"""

import asyncio
from typing import List, Dict, Any, Optional

import fireworks.client
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
from dodhalueval.models.model_registry import get_model_registry
from dodhalueval.utils.logger import get_logger

logger = get_logger(__name__)


class FireworksProvider(BaseLLMProvider):
    """Fireworks AI API provider implementation."""
    
    def __init__(self, config: APIConfig):
        super().__init__(config)
        
        if not config.api_key:
            raise AuthenticationError("Fireworks API key is required")
        
        # Set up Fireworks client
        fireworks.client.api_key = config.api_key
        if config.base_url:
            fireworks.client.api_base = config.base_url
        
        # Get model registry for validation
        self.registry = get_model_registry()
        
        # Validate model exists in registry
        is_valid, message = self.registry.validate_model(config.model, "fireworks")
        if not is_valid:
            logger.warning(f"Model validation: {message}")
            # Don't fail initialization for backward compatibility, just warn
        
        # Legacy model mappings for backward compatibility
        self.model_mappings = {
            'llama-2-7b': 'accounts/fireworks/models/llama-v2-7b-chat',
            'llama-2-13b': 'accounts/fireworks/models/llama-v2-13b-chat',
            'llama-2-70b': 'accounts/fireworks/models/llama-v2-70b-chat',
            'mixtral-8x7b': 'accounts/fireworks/models/mixtral-8x7b-instruct',
            'mistral-7b': 'accounts/fireworks/models/mistral-7b-instruct-4k',
            'zephyr-7b': 'accounts/fireworks/models/zephyr-7b-beta',
            'code-llama-7b': 'accounts/fireworks/models/code-llama-7b-instruct',
            'code-llama-13b': 'accounts/fireworks/models/code-llama-13b-instruct',
            'code-llama-34b': 'accounts/fireworks/models/code-llama-34b-instruct'
        }
        
        # Get pricing from registry if available
        model_info = self.registry.get_model(config.model)
        if model_info:
            self.input_cost_per_token = model_info.input_cost_per_token
            self.output_cost_per_token = model_info.output_cost_per_token
        else:
            # Fallback pricing for unknown models
            self.input_cost_per_token = 0.0005
            self.output_cost_per_token = 0.0005
        
        # Legacy pricing dict for backward compatibility
        self.pricing = {
            'llama-v2-7b': {'input': 0.0002, 'output': 0.0002},
            'llama-v2-13b': {'input': 0.0003, 'output': 0.0003},
            'llama-v2-70b': {'input': 0.0009, 'output': 0.0009},
            'mixtral-8x7b': {'input': 0.0005, 'output': 0.0005},
            'mistral-7b': {'input': 0.0002, 'output': 0.0002}
        }
    
    def _get_model_name(self, model: Optional[str] = None) -> str:
        """Get the actual model name to use."""
        model_name = model or self.config.model
        return self.model_mappings.get(model_name, model_name)
    
    def _format_prompt_for_chat_model(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt for chat models that expect specific formatting."""
        if system_prompt:
            return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        else:
            return f"<s>[INST] {prompt} [/INST]"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((
            Exception,  # Fireworks may raise various exceptions
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
        """Generate a response using Fireworks AI API."""
        if params is None:
            params = GenerationParameters()
        
        model_name = self._get_model_name(model)
        
        # Format prompt for chat models
        if any(chat_indicator in model_name.lower() for chat_indicator in ['chat', 'instruct']):
            formatted_prompt = self._format_prompt_for_chat_model(prompt, system_prompt)
        else:
            formatted_prompt = prompt
        
        # Prepare parameters
        api_params = {
            'model': model_name,
            'prompt': formatted_prompt,
            'temperature': params.temperature,
            'max_tokens': params.max_tokens,
            'top_p': params.top_p,
            'frequency_penalty': params.frequency_penalty,
            'presence_penalty': params.presence_penalty
        }
        
        if params.stop_sequences:
            api_params['stop'] = params.stop_sequences
        
        # Add any additional parameters
        api_params.update(kwargs)
        
        try:
            # Use asyncio to run the synchronous Fireworks client
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: fireworks.client.Completion.create(**api_params)
            )
            
            # Extract response data
            if response.choices:
                choice = response.choices[0]
                content = choice.text or ""
                finish_reason = getattr(choice, 'finish_reason', 'stop')
            else:
                content = ""
                finish_reason = "error"
            
            return GenerationResult.from_response(
                text=content,
                model=model_name,
                provider=self.provider_name,
                tokens_used=getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0,
                finish_reason=finish_reason,
                generation_params=self._create_generation_params(params),
                metadata={
                    'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') else 0,
                    'completion_tokens': getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') else 0,
                    'response_id': getattr(response, 'id', ''),
                    'model_used': getattr(response, 'model', model_name)
                }
            )
        
        except Exception as e:
            error_msg = str(e).lower()
            
            if 'authentication' in error_msg or 'api key' in error_msg:
                raise AuthenticationError(f"Fireworks authentication failed: {e}")
            
            elif 'model' in error_msg and 'not found' in error_msg:
                raise ModelNotFoundError(f"Fireworks model not found: {e}")
            
            elif 'rate limit' in error_msg or 'too many requests' in error_msg:
                raise RateLimitError(f"Fireworks rate limit exceeded: {e}")
            
            elif 'content' in error_msg and 'filter' in error_msg:
                raise ContentFilterError(f"Content filtered by Fireworks: {e}")
            
            else:
                raise LLMProviderError(f"Fireworks API error: {e}")
    
    async def get_available_models(self) -> List[str]:
        """Get list of available Fireworks models."""
        try:
            # Use asyncio to run the synchronous Fireworks client
            models = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: fireworks.client.Models.list()
            )
            
            # Extract model names
            if hasattr(models, 'data'):
                model_names = [model.id for model in models.data]
            else:
                model_names = list(self.model_mappings.values())
            
            return sorted(model_names)
        
        except Exception as e:
            self.logger.warning(f"Failed to fetch Fireworks models: {e}")
            # Return known models as fallback
            return list(self.model_mappings.values())
    
    async def validate_model(self, model_name: str) -> bool:
        """Validate that a Fireworks model is available."""
        try:
            available_models = await self.get_available_models()
            actual_model = self._get_model_name(model_name)
            return actual_model in available_models or model_name in self.model_mappings
        except Exception:
            # If we can't fetch models, assume mapped models are available
            return model_name in self.model_mappings
    
    async def estimate_cost(
        self,
        prompts: List[str],
        params: Optional[GenerationParameters] = None
    ) -> Dict[str, float]:
        """Estimate cost for Fireworks AI API calls."""
        if params is None:
            params = GenerationParameters()
        
        model_name = self._get_model_name()
        
        # Extract base model name for pricing
        base_model = model_name.split('/')[-1] if '/' in model_name else model_name
        pricing = self.pricing.get(base_model, {'input': 0.0005, 'output': 0.0005})
        
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
    
    async def batch_generate_with_delay(
        self,
        prompts: List[str],
        params: Optional[GenerationParameters] = None,
        delay_between_requests: float = 0.5,
        **kwargs
    ) -> List[GenerationResult]:
        """Generate responses with delays to avoid rate limits."""
        if not prompts:
            return []
        
        results = []
        
        for i, prompt in enumerate(prompts):
            self.logger.info(f"Processing prompt {i + 1}/{len(prompts)}")
            
            try:
                result = await self.generate(prompt, params, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to generate response for prompt {i}: {e}")
                # Create error result
                error_result = GenerationResult.from_response(
                    text="",
                    model=self._get_model_name(),
                    provider=self.provider_name,
                    finish_reason="error",
                    metadata={"error": str(e)}
                )
                results.append(error_result)
            
            # Delay between requests if not the last prompt
            if i < len(prompts) - 1:
                await asyncio.sleep(delay_between_requests)
        
        return results
    
    async def close(self):
        """Close the Fireworks client."""
        # Fireworks client doesn't need explicit closing
        await super().close()