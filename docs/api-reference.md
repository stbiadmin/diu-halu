# API Reference

This document provides comprehensive API documentation for all public interfaces in the DoDHaluEval framework.

## Core Modules

### dodhalueval.core

#### HallucinationDetector

Main interface for hallucination detection and evaluation.

```python
class HallucinationDetector:
    """Comprehensive hallucination detection using multiple methods."""
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        enable_huggingface_hhem: bool = True,
        enable_g_eval: bool = True,
        enable_selfcheck: bool = True,
        confidence_threshold: float = 0.7,
        ensemble_weights: Optional[Dict[str, float]] = None
    ):
        """Initialize hallucination detector.
        
        Args:
            llm_provider: LLM provider for evaluation methods
            enable_huggingface_hhem: Enable HuggingFace HHEM evaluator
            enable_g_eval: Enable G-Eval method
            enable_selfcheck: Enable SelfCheckGPT method
            confidence_threshold: Minimum confidence for positive detection
            ensemble_weights: Custom weights for ensemble voting
        """
        
    async def evaluate_batch(
        self,
        responses: List[Response],
        prompts: List[Prompt],
        batch_size: int = 10,
        include_details: bool = True
    ) -> List[EvaluationResult]:
        """Evaluate batch of responses for hallucinations.
        
        Args:
            responses: List of response objects to evaluate
            prompts: Corresponding prompts for context
            batch_size: Number of responses to process simultaneously
            include_details: Include detailed evaluation information
            
        Returns:
            List of evaluation results with scores and classifications
        """
        
    async def evaluate_single(
        self,
        response: Response,
        prompt: Prompt,
        methods: Optional[List[str]] = None
    ) -> EvaluationResult:
        """Evaluate single response for hallucinations.
        
        Args:
            response: Response object to evaluate
            prompt: Corresponding prompt for context
            methods: Specific detection methods to use
            
        Returns:
            Evaluation result with aggregated score and classification
        """
```

#### PromptGenerator

Generate hallucination-prone prompts using multiple strategies.

```python
class PromptGenerator:
    """Generate prompts using template-based and heuristic methods."""
    
    def __init__(self, config: PromptGenerationConfig):
        """Initialize prompt generator.
        
        Args:
            config: Configuration for prompt generation strategies
        """
        
    def generate_from_chunks(
        self,
        chunks: List[DocumentChunk],
        max_prompts: Optional[int] = None,
        filter_quality: bool = True
    ) -> List[Prompt]:
        """Generate prompts from document chunks.
        
        Args:
            chunks: Document chunks to use as source material
            max_prompts: Maximum number of prompts to generate
            filter_quality: Apply quality filtering to results
            
        Returns:
            List of generated prompt objects
        """
        
    def generate_from_template(
        self,
        template: str,
        chunk: DocumentChunk,
        variables: Dict[str, Any]
    ) -> Prompt:
        """Generate prompt from template with variable substitution.
        
        Args:
            template: Template string with placeholders
            chunk: Source chunk for context and variables
            variables: Additional variables for template substitution
            
        Returns:
            Generated prompt object
        """
```

#### ResponseGenerator

Generate responses with controlled hallucination injection.

```python
class ResponseGenerator:
    """Generate responses with hallucination injection capabilities."""
    
    def __init__(
        self,
        providers: Dict[str, BaseLLMProvider],
        config: ResponseConfig
    ):
        """Initialize response generator.
        
        Args:
            providers: Dictionary of LLM providers by name
            config: Configuration for response generation
        """
        
    async def generate_responses(
        self,
        prompts: List[Prompt],
        providers: List[str],
        inject_hallucinations: bool = True
    ) -> List[Response]:
        """Generate responses for prompts using specified providers.
        
        Args:
            prompts: List of prompts to generate responses for
            providers: List of provider names to use
            inject_hallucinations: Whether to inject hallucinations
            
        Returns:
            List of generated response objects
        """
        
    async def generate_single_response(
        self,
        prompt: Prompt,
        provider_name: str,
        **generation_kwargs
    ) -> Response:
        """Generate single response for a prompt.
        
        Args:
            prompt: Prompt object to generate response for
            provider_name: Name of LLM provider to use
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated response object
        """
```

### dodhalueval.data

#### PDFProcessor

Process PDF documents and extract structured content.

```python
class PDFProcessor:
    """Process PDF documents with text extraction and chunking."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_pages: Optional[int] = None,
        cache_enabled: bool = True,
        cache_dir: str = "/tmp/dodhalueval/pdf_cache"
    ):
        """Initialize PDF processor.
        
        Args:
            chunk_size: Target size for text chunks
            chunk_overlap: Overlap between consecutive chunks
            max_pages: Maximum pages to process (None for all)
            cache_enabled: Enable caching of processed documents
            cache_dir: Directory for cache storage
        """
        
    def process_document(
        self,
        pdf_path: str,
        extract_metadata: bool = True,
        detect_structure: bool = True
    ) -> Dict[str, Any]:
        """Process single PDF document.
        
        Args:
            pdf_path: Path to PDF file
            extract_metadata: Extract document metadata
            detect_structure: Detect document structure (headings, sections)
            
        Returns:
            Dictionary containing processed document data:
            - 'pages': List of page content objects
            - 'chunks': List of document chunks
            - 'metadata': Document metadata
            - 'structure': Document structure information
        """
        
    def extract_text(
        self,
        pdf_path: str,
        max_pages: Optional[int] = None
    ) -> List[PageContent]:
        """Extract raw text from PDF pages.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum pages to extract (None for all)
            
        Returns:
            List of page content objects with text and metadata
        """
```

#### DatasetBuilder

Build and export evaluation datasets in standard formats.

```python
class DatasetBuilder:
    """Build evaluation datasets from pipeline results."""
    
    def __init__(self, output_dir: str):
        """Initialize dataset builder.
        
        Args:
            output_dir: Directory for dataset output files
        """
        
    def build_halueval_format(
        self,
        prompts: List[Prompt],
        responses: List[Response],
        evaluations: Optional[List[EvaluationResult]] = None,
        documents: Optional[List[Document]] = None,
        dataset_name: str = "dod_halueval",
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> HaluEvalDataset:
        """Build dataset in HaluEval-compatible format.
        
        Args:
            prompts: List of prompt objects
            responses: List of response objects
            evaluations: Optional evaluation results
            documents: Optional source documents
            dataset_name: Name for the dataset
            additional_metadata: Additional metadata to include
            
        Returns:
            HaluEvalDataset object with standardized format
        """
        
    def export_jsonl(
        self,
        dataset: HaluEvalDataset,
        filename: str
    ) -> str:
        """Export dataset to JSONL format.
        
        Args:
            dataset: Dataset to export
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        
    def create_train_test_split(
        self,
        dataset: HaluEvalDataset,
        train_ratio: float = 0.8,
        stratify_by: str = "label",
        random_seed: int = 42
    ) -> Tuple[HaluEvalDataset, HaluEvalDataset]:
        """Split dataset into training and test sets.
        
        Args:
            dataset: Dataset to split
            train_ratio: Proportion for training set
            stratify_by: Field to stratify split by
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
```

### dodhalueval.providers

#### BaseLLMProvider

Abstract base class for LLM provider implementations.

```python
class BaseLLMProvider:
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: APIConfig):
        """Initialize provider with configuration.
        
        Args:
            config: API configuration object
        """
        
    async def generate_response(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """Generate response for given prompt.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        
    async def generate_batch(
        self,
        prompts: List[str],
        batch_size: int = 10,
        **kwargs
    ) -> List[str]:
        """Generate responses for batch of prompts.
        
        Args:
            prompts: List of prompt texts
            batch_size: Number of prompts to process simultaneously
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated response texts
        """
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary with model information (name, version, capabilities)
        """
```

#### OpenAIProvider

OpenAI API provider implementation.

```python
class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider for GPT models."""
    
    def __init__(self, config: APIConfig):
        """Initialize OpenAI provider.
        
        Args:
            config: API configuration with OpenAI-specific settings
        """
        
    async def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> str:
        """Generate response using OpenAI API.
        
        Args:
            prompt: Input prompt text
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional OpenAI API parameters
            
        Returns:
            Generated response text
        """
```

#### FireworksProvider

Fireworks AI provider implementation.

```python
class FireworksProvider(BaseLLMProvider):
    """Fireworks AI provider for Llama and other models."""
    
    def __init__(self, config: APIConfig):
        """Initialize Fireworks provider.
        
        Args:
            config: API configuration with Fireworks-specific settings
        """
        
    async def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> str:
        """Generate response using Fireworks API.
        
        Args:
            prompt: Input prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional Fireworks API parameters
            
        Returns:
            Generated response text
        """
```

## Data Models

### Core Schemas

#### Document

```python
class Document(BaseModel):
    """Document with metadata and content chunks."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    source_path: str
    file_hash: Optional[str] = None
    page_count: int
    content: List[DocumentChunk] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def total_chunks(self) -> int:
        """Total number of content chunks."""
        return len(self.content)
    
    @property
    def total_words(self) -> int:
        """Total word count across all chunks."""
        return sum(chunk.word_count for chunk in self.content)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get chunk by ID."""
        return next((c for c in self.content if c.id == chunk_id), None)
    
    def get_chunks_by_page(self, page_number: int) -> List[DocumentChunk]:
        """Get all chunks from specific page."""
        return [c for c in self.content if c.page_number == page_number]
```

#### DocumentChunk

```python
class DocumentChunk(BaseModel):
    """Individual chunk of document content."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str
    page_number: int
    chunk_index: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def word_count(self) -> int:
        """Word count of chunk content."""
        return len(self.content.split())
    
    @property
    def char_count(self) -> int:
        """Character count of chunk content."""
        return len(self.content)
    
    @field_validator('end_char')
    @classmethod
    def validate_char_positions(cls, v: Optional[int], info) -> Optional[int]:
        """Validate character position ordering."""
        if v is not None and info.data.get('start_char') is not None:
            if v <= info.data['start_char']:
                raise ValueError('end_char must be greater than start_char')
        return v
```

#### Prompt

```python
class Prompt(BaseModel):
    """Generated prompt with metadata."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source_document_id: str
    source_chunk_id: Optional[str] = None
    expected_answer: Optional[str] = None
    hallucination_type: Optional[str] = None
    generation_strategy: str
    difficulty_level: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def word_count(self) -> int:
        """Word count of prompt text."""
        return len(self.text.split())
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate prompt text."""
        if not v.strip():
            raise ValueError('Prompt text cannot be empty')
        if len(v) > 10000:
            raise ValueError('Prompt text too long (max 10000 characters)')
        return v.strip()
```

#### Response

```python
class Response(BaseModel):
    """Generated response with metadata."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt_id: str
    text: str
    model: str
    provider: str
    generation_params: Dict[str, Any] = Field(default_factory=dict)
    contains_hallucination: Optional[bool] = None
    hallucination_types: List[str] = Field(default_factory=list)
    hallucination_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def word_count(self) -> int:
        """Word count of response text."""
        return len(self.text.split())
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate response text."""
        if not v.strip():
            raise ValueError('Response text cannot be empty')
        return v.strip()
```

#### EvaluationResult

```python
class EvaluationResult(BaseModel):
    """Result of hallucination detection evaluation."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    response_id: str
    method: str
    is_hallucinated: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    hallucination_type: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate evaluation method."""
        valid_methods = {
            'vectara_hhem', 'huggingface_hhem', 'g_eval', 
            'self_check_gpt', 'consistency_check', 'ensemble'
        }
        if v not in valid_methods:
            raise ValueError(f'Invalid evaluation method: {v}')
        return v
```

## Configuration Models

### APIConfig

```python
class APIConfig(BaseModel):
    """Configuration for API providers."""
    
    provider: str
    api_key: Optional[str] = None
    model: str
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout: int = Field(default=30, ge=5, le=300)
    rate_limit: Optional[int] = Field(default=None, ge=1)
    base_url: Optional[str] = None
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider name."""
        supported = {'openai', 'fireworks', 'local', 'mock'}
        if v not in supported:
            raise ValueError(f'Provider must be one of: {supported}')
        return v
```

### PromptGenerationConfig

```python
class PromptGenerationConfig(BaseModel):
    """Configuration for prompt generation."""
    
    strategies: List[str] = Field(default_factory=lambda: ['template', 'llm_based'])
    max_prompts_per_document: int = Field(default=100, ge=1, le=1000)
    hallucination_types: List[str] = Field(
        default_factory=lambda: ['factual', 'logical', 'context']
    )
    template_file: Optional[str] = None
    perturbation_enabled: bool = True
    
    @field_validator('strategies')
    @classmethod
    def validate_strategies(cls, v: List[str]) -> List[str]:
        """Validate prompt generation strategies."""
        supported = {'template', 'llm_based', 'heuristic', 'hybrid'}
        invalid = set(v) - supported
        if invalid:
            raise ValueError(f'Invalid strategies: {invalid}')
        return v
```

## Utility Functions

### Metrics Calculation

```python
from dodhalueval.utils.metrics import MetricsCalculator

calculator = MetricsCalculator()

# Calculate detection metrics
metrics = calculator.calculate_detection_metrics(
    predictions: List[float],
    ground_truth: List[bool],
    threshold: float = 0.5
) -> DetectionMetrics

# Calculate per-category metrics
category_metrics = calculator.calculate_per_category_metrics(
    evaluations: List[EvaluationResult],
    ground_truth: List[bool],
    categories: List[str]
) -> Dict[str, DetectionMetrics]

# Find optimal threshold
optimal_threshold, optimal_metrics = calculator.find_optimal_threshold(
    predictions: List[float],
    ground_truth: List[bool],
    metric: str = "f1_score"
) -> Tuple[float, DetectionMetrics]
```

### Configuration Loading

```python
from dodhalueval.utils.config import load_config, get_default_config

# Load configuration from file
config = load_config(
    config_file: Optional[str] = None,
    environment: str = "development",
    config_dir: str = "configs"
) -> DoDHaluEvalConfig

# Get default configuration
default_config = get_default_config() -> DoDHaluEvalConfig
```

### Logging

```python
from dodhalueval.utils.logger import get_logger, setup_logging

# Get logger instance
logger = get_logger(name: str, config: Optional[LoggingConfig] = None)

# Setup logging configuration
setup_logging(config: LoggingConfig)
```

## Exception Hierarchy

```python
from dodhalueval.utils.exceptions import (
    DoDHaluEvalError,          # Base exception
    ConfigurationError,        # Configuration issues
    APIError,                  # API-related errors
    ProcessingError,           # Document processing errors
    ValidationError,           # Data validation errors
    ModelError,                # Model-related errors
    CacheError                 # Caching errors
)
```

## CLI Interface

### Main Commands

```bash
# Process documents
dodhalueval process-docs --input <input_dir> --output <output_dir>

# Generate prompts
dodhalueval generate-prompts --input <input_dir> --output <output_dir> [options]

# Generate responses
dodhalueval generate-responses --input <input_dir> --output <output_dir> [options]

# Evaluate responses
dodhalueval evaluate --input <input_dir> --output <output_dir> [options]

# Build dataset
dodhalueval build-dataset --input <input_dir> --output <output_file> [options]

# Validate configuration
dodhalueval validate-config --config <config_file>

# Show system information
dodhalueval info

# Generate default configuration
dodhalueval generate-config --output <output_file>
```

### Command Options

Most commands support these common options:

- `--config, -c`: Configuration file path
- `--environment, -e`: Environment (development, production, testing)
- `--verbose, -v`: Enable verbose logging
- `--quiet, -q`: Suppress output except errors
- `--help`: Show command help

For detailed command-specific options, use `dodhalueval <command> --help`.

This API reference provides comprehensive documentation for all public interfaces. For implementation examples, see the [Usage Guide](usage.md).