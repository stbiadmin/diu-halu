"""Configuration models for DoDHaluEval."""

from typing import Dict, List, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ConfigDict


class PDFProcessingConfig(BaseModel):
    """Configuration for PDF processing."""
    
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    cache_enabled: bool = Field(default=True)
    cache_dir: str = Field(default="/tmp/dodhalueval/pdf_cache")
    max_pages: Optional[int] = Field(default=None, ge=1)
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Ensure chunk_overlap is less than chunk_size."""
        if info.data.get('chunk_size') and v >= info.data['chunk_size']:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v


class APIConfig(BaseModel):
    """Configuration for API providers."""
    
    provider: str = Field(..., description="Provider name (openai, fireworks, etc.)")
    api_key: Optional[str] = Field(default=None, description="API key")
    model: str = Field(..., description="Model name")
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout: int = Field(default=30, ge=5, le=300)
    rate_limit: Optional[int] = Field(default=None, ge=1)
    base_url: Optional[str] = Field(default=None)
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider is supported."""
        supported = {'openai', 'fireworks', 'local', 'mock'}
        if v not in supported:
            raise ValueError(f'Provider must be one of: {supported}')
        return v


class EvaluationConfig(BaseModel):
    """Configuration for evaluation methods."""
    
    method: str = Field(..., description="Evaluation method name")
    enabled: bool = Field(default=True)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    batch_size: int = Field(default=10, ge=1, le=100)
    config: Dict = Field(default_factory=dict)
    
    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate evaluation method is supported."""
        supported = {'vectara_hhem', 'g_eval', 'self_check_gpt', 'consistency_check'}
        if v not in supported:
            raise ValueError(f'Method must be one of: {supported}')
        return v


class PromptGenerationConfig(BaseModel):
    """Configuration for prompt generation."""
    
    strategies: List[str] = Field(default_factory=lambda: ['template', 'llm_based'])
    max_prompts_per_document: int = Field(default=100, ge=1, le=1000)
    hallucination_types: List[str] = Field(
        default_factory=lambda: ['factual', 'logical', 'context']
    )
    template_file: Optional[str] = Field(default=None)
    perturbation_enabled: bool = Field(default=True)
    
    @field_validator('strategies')
    @classmethod
    def validate_strategies(cls, v: List[str]) -> List[str]:
        """Validate prompt generation strategies."""
        supported = {'template', 'llm_based', 'heuristic', 'hybrid'}
        invalid = set(v) - supported
        if invalid:
            raise ValueError(f'Invalid strategies: {invalid}. Supported: {supported}')
        return v


class OutputConfig(BaseModel):
    """Configuration for output generation."""
    
    format: str = Field(default='jsonl', pattern=r'^(json|jsonl|csv)$')
    output_dir: str = Field(default='data/evaluations')
    filename_prefix: str = Field(default='dod_halueval')
    include_metadata: bool = Field(default=True)
    compress: bool = Field(default=False)
    
    @field_validator('output_dir')
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        """Ensure output directory exists or can be created."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class CacheConfig(BaseModel):
    """Configuration for caching."""
    
    enabled: bool = Field(default=True)
    pdf_cache_dir: str = Field(default="/tmp/dodhalueval/pdf_cache")
    llm_cache_dir: str = Field(default="/tmp/dodhalueval/llm_cache")
    max_cache_size_mb: int = Field(default=1000, ge=100, le=10000)
    ttl_hours: int = Field(default=24, ge=1, le=168)  # 1 hour to 1 week
    
    def __post_init__(self) -> None:
        """Create cache directories if they don't exist."""
        if self.enabled:
            Path(self.pdf_cache_dir).mkdir(parents=True, exist_ok=True)
            Path(self.llm_cache_dir).mkdir(parents=True, exist_ok=True)


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: str = Field(default='INFO', pattern=r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$')
    log_file: Optional[str] = Field(default=None)
    log_dir: str = Field(default='logs')
    max_file_size_mb: int = Field(default=10, ge=1, le=100)
    backup_count: int = Field(default=5, ge=1, le=10)
    console_output: bool = Field(default=True)
    structured_logging: bool = Field(default=True)


class DoDHaluEvalConfig(BaseModel):
    """Main configuration for DoDHaluEval."""
    
    # Core settings
    version: str = Field(default='0.1.0')
    environment: str = Field(default='development', pattern=r'^(development|production|testing)$')
    
    # Component configurations
    pdf_processing: PDFProcessingConfig = Field(default_factory=PDFProcessingConfig)
    api_configs: Dict[str, APIConfig] = Field(default_factory=dict)
    evaluation_methods: List[EvaluationConfig] = Field(default_factory=list)
    prompt_generation: PromptGenerationConfig = Field(default_factory=PromptGenerationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Processing settings
    batch_size: int = Field(default=50, ge=1, le=500)
    max_concurrent_requests: int = Field(default=10, ge=1, le=50)
    enable_progress_bar: bool = Field(default=True)
    
    # Data paths
    data_dir: str = Field(default='data')
    source_documents_dir: str = Field(default='data/CSC')
    processed_documents_dir: str = Field(default='data/processed')
    
    @field_validator('api_configs')
    @classmethod
    def validate_api_configs(cls, v: Dict[str, APIConfig]) -> Dict[str, APIConfig]:
        """Validate API configurations."""
        if not v:
            # Provide default configurations
            return {
                'openai': APIConfig(
                    provider='openai',
                    model='gpt-4',
                    api_key=None
                ),
                'fireworks': APIConfig(
                    provider='fireworks',
                    model='llama-v2-70b-chat',
                    api_key=None
                )
            }
        return v
    
    @field_validator('evaluation_methods')
    @classmethod
    def validate_evaluation_methods(cls, v: List[EvaluationConfig]) -> List[EvaluationConfig]:
        """Validate evaluation method configurations."""
        if not v:
            # Provide default evaluation methods
            return [
                EvaluationConfig(method='vectara_hhem'),
                EvaluationConfig(method='g_eval'),
                EvaluationConfig(method='self_check_gpt')
            ]
        return v
    
    def get_api_config(self, provider: str) -> Optional[APIConfig]:
        """Get API configuration for a specific provider."""
        return self.api_configs.get(provider)
    
    def get_evaluation_config(self, method: str) -> Optional[EvaluationConfig]:
        """Get evaluation configuration for a specific method."""
        for config in self.evaluation_methods:
            if config.method == method:
                return config
        return None
    
    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.cache.enabled
    
    def get_cache_dir(self, cache_type: str) -> str:
        """Get cache directory for specific cache type."""
        if cache_type == 'pdf':
            return self.cache.pdf_cache_dir
        elif cache_type == 'llm':
            return self.cache.llm_cache_dir
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
    
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        use_enum_values=True,
        json_encoders={
            Path: str
        }
    )