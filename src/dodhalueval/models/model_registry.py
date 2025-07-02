"""
Model Registry for DoDHaluEval

Centralized management of supported models across different providers
with validation, discovery, and configuration utilities.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelCapability(Enum):
    """Capabilities that models can have."""
    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    INSTRUCTION_FOLLOWING = "instruction_following"


@dataclass
class ModelInfo:
    """Information about a specific model."""
    id: str
    display_name: str
    provider: str
    capabilities: List[ModelCapability]
    context_length: int
    input_cost_per_token: float  # USD per 1000 tokens
    output_cost_per_token: float  # USD per 1000 tokens
    recommended_for: List[str]  # Use cases
    description: str
    is_default: bool = False
    is_deprecated: bool = False


class ModelRegistry:
    """Central registry for all supported models across providers."""
    
    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._provider_models: Dict[str, List[str]] = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the model registry with current supported models."""
        
        # OpenAI Models
        openai_models = [
            ModelInfo(
                id="gpt-4",
                display_name="GPT-4",
                provider="openai",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT_COMPLETION,
                    ModelCapability.REASONING,
                    ModelCapability.INSTRUCTION_FOLLOWING
                ],
                context_length=8192,
                input_cost_per_token=0.03,
                output_cost_per_token=0.06,
                recommended_for=["complex reasoning", "detailed analysis", "high-quality generation"],
                description="Most capable GPT-4 model for complex tasks",
                is_default=True
            ),
            ModelInfo(
                id="gpt-4-turbo",
                display_name="GPT-4 Turbo",
                provider="openai",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT_COMPLETION,
                    ModelCapability.REASONING,
                    ModelCapability.INSTRUCTION_FOLLOWING
                ],
                context_length=128000,
                input_cost_per_token=0.01,
                output_cost_per_token=0.03,
                recommended_for=["long documents", "large context tasks", "cost-effective analysis"],
                description="Latest GPT-4 with larger context and lower cost"
            ),
            ModelInfo(
                id="gpt-3.5-turbo",
                display_name="GPT-3.5 Turbo",
                provider="openai",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT_COMPLETION,
                    ModelCapability.INSTRUCTION_FOLLOWING
                ],
                context_length=16384,
                input_cost_per_token=0.0015,
                output_cost_per_token=0.002,
                recommended_for=["fast generation", "cost-effective tasks", "simple analysis"],
                description="Fast and cost-effective model for simpler tasks"
            ),
            ModelInfo(
                id="gpt-4o",
                display_name="GPT-4o",
                provider="openai",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT_COMPLETION,
                    ModelCapability.REASONING,
                    ModelCapability.INSTRUCTION_FOLLOWING
                ],
                context_length=128000,
                input_cost_per_token=0.005,
                output_cost_per_token=0.015,
                recommended_for=["balanced performance", "multimodal tasks", "general purpose"],
                description="Optimized GPT-4 model with good performance/cost balance"
            )
        ]
        
        # Fireworks Models (Based on current serverless offerings)
        fireworks_models = [
            ModelInfo(
                id="accounts/fireworks/models/llama-v3p1-8b-instruct",
                display_name="Llama 3.1 8B Instruct",
                provider="fireworks",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT_COMPLETION,
                    ModelCapability.INSTRUCTION_FOLLOWING
                ],
                context_length=8192,
                input_cost_per_token=0.0002,
                output_cost_per_token=0.0002,
                recommended_for=["fast generation", "cost-effective tasks", "instruction following"],
                description="Fast and efficient Llama 3.1 8B model",
                is_default=True
            ),
            ModelInfo(
                id="accounts/fireworks/models/llama-v3p1-70b-instruct",
                display_name="Llama 3.1 70B Instruct",
                provider="fireworks",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT_COMPLETION,
                    ModelCapability.REASONING,
                    ModelCapability.INSTRUCTION_FOLLOWING
                ],
                context_length=8192,
                input_cost_per_token=0.0009,
                output_cost_per_token=0.0009,
                recommended_for=["complex reasoning", "high-quality generation", "detailed analysis"],
                description="Large Llama 3.1 model with strong reasoning capabilities"
            ),
            ModelInfo(
                id="accounts/fireworks/models/llama-v3p1-405b-instruct",
                display_name="Llama 3.1 405B Instruct",
                provider="fireworks",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT_COMPLETION,
                    ModelCapability.REASONING,
                    ModelCapability.INSTRUCTION_FOLLOWING
                ],
                context_length=8192,
                input_cost_per_token=0.003,
                output_cost_per_token=0.003,
                recommended_for=["highest quality", "complex reasoning", "research tasks"],
                description="Largest Llama 3.1 model with maximum capabilities"
            ),
            ModelInfo(
                id="accounts/fireworks/models/mixtral-8x7b-instruct",
                display_name="Mixtral 8x7B Instruct",
                provider="fireworks",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT_COMPLETION,
                    ModelCapability.REASONING,
                    ModelCapability.INSTRUCTION_FOLLOWING
                ],
                context_length=32768,
                input_cost_per_token=0.0005,
                output_cost_per_token=0.0005,
                recommended_for=["long context", "multilingual tasks", "balanced performance"],
                description="Mixture of experts model with large context window"
            ),
            ModelInfo(
                id="accounts/fireworks/models/llama-v3-8b-instruct",
                display_name="Llama 3 8B Instruct",
                provider="fireworks",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT_COMPLETION,
                    ModelCapability.INSTRUCTION_FOLLOWING
                ],
                context_length=8192,
                input_cost_per_token=0.0002,
                output_cost_per_token=0.0002,
                recommended_for=["fast generation", "cost-effective tasks"],
                description="Previous generation Llama 3 8B model"
            ),
            ModelInfo(
                id="accounts/fireworks/models/llama-v3-70b-instruct",
                display_name="Llama 3 70B Instruct",
                provider="fireworks",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT_COMPLETION,
                    ModelCapability.REASONING,
                    ModelCapability.INSTRUCTION_FOLLOWING
                ],
                context_length=8192,
                input_cost_per_token=0.0009,
                output_cost_per_token=0.0009,
                recommended_for=["complex reasoning", "high-quality generation"],
                description="Previous generation Llama 3 70B model"
            ),
            # Legacy model mappings for backward compatibility
            ModelInfo(
                id="accounts/fireworks/models/llama-v2-7b-chat",
                display_name="Llama 2 7B Chat",
                provider="fireworks",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT_COMPLETION
                ],
                context_length=4096,
                input_cost_per_token=0.0002,
                output_cost_per_token=0.0002,
                recommended_for=["legacy compatibility", "basic chat"],
                description="Legacy Llama 2 7B model",
                is_deprecated=True
            ),
            ModelInfo(
                id="accounts/fireworks/models/llama-v2-13b-chat",
                display_name="Llama 2 13B Chat",
                provider="fireworks",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT_COMPLETION
                ],
                context_length=4096,
                input_cost_per_token=0.0003,
                output_cost_per_token=0.0003,
                recommended_for=["legacy compatibility"],
                description="Legacy Llama 2 13B model",
                is_deprecated=True
            ),
            ModelInfo(
                id="accounts/fireworks/models/llama-v2-70b-chat",
                display_name="Llama 2 70B Chat",
                provider="fireworks",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT_COMPLETION,
                    ModelCapability.REASONING
                ],
                context_length=4096,
                input_cost_per_token=0.0009,
                output_cost_per_token=0.0009,
                recommended_for=["legacy compatibility"],
                description="Legacy Llama 2 70B model",
                is_deprecated=True
            )
        ]
        
        # Mock models for testing
        mock_models = [
            ModelInfo(
                id="mock-gpt-4",
                display_name="Mock GPT-4",
                provider="mock",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT_COMPLETION,
                    ModelCapability.REASONING,
                    ModelCapability.INSTRUCTION_FOLLOWING
                ],
                context_length=8192,
                input_cost_per_token=0.0,
                output_cost_per_token=0.0,
                recommended_for=["testing", "development", "debugging"],
                description="Mock model for testing purposes",
                is_default=True
            )
        ]
        
        # Register all models
        all_models = openai_models + fireworks_models + mock_models
        
        for model in all_models:
            self._models[model.id] = model
            
            # Track models by provider
            if model.provider not in self._provider_models:
                self._provider_models[model.provider] = []
            self._provider_models[model.provider].append(model.id)
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information by ID."""
        return self._models.get(model_id)
    
    def get_models_by_provider(self, provider: str, include_deprecated: bool = False) -> List[ModelInfo]:
        """Get all models for a specific provider."""
        model_ids = self._provider_models.get(provider, [])
        models = [self._models[model_id] for model_id in model_ids]
        
        if not include_deprecated:
            models = [m for m in models if not m.is_deprecated]
            
        return models
    
    def get_default_model(self, provider: str) -> Optional[ModelInfo]:
        """Get the default model for a provider."""
        models = self.get_models_by_provider(provider)
        defaults = [m for m in models if m.is_default]
        return defaults[0] if defaults else (models[0] if models else None)
    
    def get_models_by_capability(self, capability: ModelCapability, provider: Optional[str] = None) -> List[ModelInfo]:
        """Get models that have a specific capability."""
        models = []
        
        if provider:
            candidate_models = self.get_models_by_provider(provider)
        else:
            candidate_models = list(self._models.values())
        
        for model in candidate_models:
            if capability in model.capabilities and not model.is_deprecated:
                models.append(model)
        
        return models
    
    def validate_model(self, model_id: str, provider: str) -> Tuple[bool, str]:
        """Validate if a model exists and is available for the provider."""
        model = self.get_model(model_id)
        
        if not model:
            return False, f"Model '{model_id}' not found in registry"
        
        if model.provider != provider:
            return False, f"Model '{model_id}' belongs to provider '{model.provider}', not '{provider}'"
        
        if model.is_deprecated:
            return True, f"Model '{model_id}' is deprecated but still available"
        
        return True, f"Model '{model_id}' is valid and available"
    
    def get_recommended_models(self, use_case: str, provider: Optional[str] = None, max_cost: Optional[float] = None) -> List[ModelInfo]:
        """Get models recommended for a specific use case."""
        if provider:
            models = self.get_models_by_provider(provider)
        else:
            models = [m for m in self._models.values() if not m.is_deprecated]
        
        # Filter by use case
        recommended = []
        for model in models:
            if any(use_case.lower() in rec.lower() for rec in model.recommended_for):
                recommended.append(model)
        
        # Filter by cost if specified
        if max_cost is not None:
            recommended = [m for m in recommended if m.output_cost_per_token <= max_cost]
        
        # Sort by cost (cheapest first)
        recommended.sort(key=lambda x: x.output_cost_per_token)
        
        return recommended
    
    def list_providers(self) -> List[str]:
        """Get list of all supported providers."""
        return list(self._provider_models.keys())
    
    def get_model_summary(self, model_id: str) -> str:
        """Get a human-readable summary of a model."""
        model = self.get_model(model_id)
        if not model:
            return f"Model '{model_id}' not found"
        
        capabilities = ", ".join([cap.value for cap in model.capabilities])
        cost_info = f"${model.output_cost_per_token:.4f}/1k tokens"
        status = ""
        
        if model.is_default:
            status += " [DEFAULT]"
        if model.is_deprecated:
            status += " [DEPRECATED]"
        
        return f"{model.display_name} ({model.provider}){status}: {model.description}. Capabilities: {capabilities}. Cost: {cost_info}"


# Global registry instance
model_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    return model_registry