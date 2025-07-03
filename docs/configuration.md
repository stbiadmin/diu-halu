# Configuration Reference

This comprehensive guide covers all configuration options available in DoDHaluEval, including generation method selection, parameter tuning, and environment variable overrides.

## Overview

DoDHaluEval uses YAML configuration files for reproducible experiments and supports environment variable overrides for runtime customization. The configuration system is built around Pydantic models providing type safety and validation.

## Configuration File Structure

```yaml
# configs/example.yaml
version: "0.1.0"
environment: "production"  # Options: development, testing, production

# Global processing settings
batch_size: 50
max_concurrent_requests: 10
enable_caching: true

# API provider configurations
api_configs:
  openai:
    provider: "openai"
    model: "gpt-4"
    max_retries: 3
    timeout: 30
  fireworks:
    provider: "fireworks" 
    model: "llama-v3p1-70b-instruct"
    max_retries: 3
    timeout: 45

# Document processing
pdf_processing:
  chunk_size: 1000
  chunk_overlap: 200
  min_chunk_length: 100
  max_chunks_per_document: 100

# Prompt generation
prompt_generation:
  strategies: ["template", "llm_based"]
  num_prompts: 100
  max_prompts_per_document: 50
  hallucination_types: ["factual", "logical", "context"]

# Response generation (method selection)
response_generation:
  generation_method: "halueval"  # Options: dodhalueval, halueval, hybrid
  providers: ["openai", "fireworks"]
  hallucination_rate: 0.3
  
  # Method-specific settings
  halueval_settings:
    use_two_stage_generation: true
    enable_filtering: true
    hallucination_patterns:
      - factual_contradiction
      - context_misunderstanding
  
  dodhalueval_settings:
    injection_strategies: ["factual", "logical", "context"]
    system_prompt_strategy: "hallucination_prone"
  
  hybrid_settings:
    primary_method: "halueval"
    fallback_method: "dodhalueval"

# Hallucination detection
hallucination_detection:
  methods: ["hhem", "g_eval", "selfcheck"]
  ensemble_evaluation: true
  consensus_method: "majority_vote"

# Output settings
output:
  directory: "output/"
  save_intermediate: true
  generate_report: true
  export_formats: ["jsonl", "json"]

# Processing control
processing:
  bypass_cache: false
  bypass_cache_steps:
    pdf_extraction: false
    document_chunking: false
    prompt_generation: false
    response_generation: false
    hallucination_detection: false
```

## Generation Method Configuration

### DoDHaluEval Method Settings

```yaml
response_generation:
  generation_method: "dodhalueval"
  
  dodhalueval_settings:
    # Hallucination injection
    hallucination_rate: 0.3
    injection_strategies:
      - factual      # Equipment specs, numbers, dates
      - logical      # Contradictory statements
      - context      # Cross-domain information
    
    # System prompt strategy
    system_prompt_strategy: "hallucination_prone"  # Options: conservative, hallucination_prone, mixed
    conservative_prompt_probability: 0.7
    hallucination_prone_prompt_probability: 0.3
    
    # Post-processing injection
    post_processing_injection: true
    factual_injection_probability: 0.3
    logical_injection_probability: 0.2
    contextual_injection_probability: 0.2
    
    # Military-specific patterns
    enable_equipment_substitution: true
    enable_branch_confusion: true
    enable_temporal_confusion: true
    
    # Response processing
    max_response_length: 2000
    temperature: 0.7
    enable_response_validation: true
```

### HaluEval Method Settings

```yaml
response_generation:
  generation_method: "halueval"
  
  halueval_settings:
    # Generation control
    use_two_stage_generation: true
    enable_filtering: true
    generation_schema: "one_pass"  # Options: one_pass, conversational, both
    
    # Hallucination patterns
    hallucination_patterns:
      - factual_contradiction      # Direct factual errors
      - context_misunderstanding   # Question context errors
      - specificity_mismatch       # Wrong detail level
      - invalid_inference          # Incorrect reasoning
      - equipment_substitution     # Military equipment errors
      - branch_confusion           # Service branch mixing
      - temporal_confusion         # Time/era errors
    
    # Knowledge context building
    max_knowledge_length: 2000
    semantic_similarity_threshold: 0.3
    use_sentence_transformers: true
    context_relevance_threshold: 0.5
    
    # Generation parameters
    generation_temperature: 0.7
    filtering_temperature: 0.3
    max_response_tokens: 200
    response_length_constraint: "similar"  # Options: similar, flexible, strict
    
    # Template configuration
    template_file: "data/prompts/halueval_templates.yaml"
    enable_custom_patterns: true
    pattern_selection_strategy: "random"  # Options: random, weighted, sequential
    
    # Quality control
    enable_response_validation: true
    minimum_hallucination_confidence: 0.6
    enable_plausibility_checking: true
```

### Hybrid Method Settings

```yaml
response_generation:
  generation_method: "hybrid"
  
  hybrid_settings:
    # Method selection
    primary_method: "halueval"
    fallback_method: "dodhalueval"
    
    # Operation modes
    comparison_mode: false          # Generate with both methods and compare
    enable_fallback: true           # Enable automatic fallback
    fallback_on_error: true         # Fallback on generation errors
    fallback_on_low_confidence: true # Fallback on low confidence scores
    
    # Selection criteria
    selection_criteria: "confidence_score"  # Options: confidence_score, length, primary_method, custom
    confidence_threshold: 0.3
    length_preference: "longer"     # Options: longer, shorter, balanced
    
    # Performance optimization
    enable_parallel_generation: false  # Generate with both methods simultaneously
    timeout_primary_method: 30      # Timeout before fallback (seconds)
    cache_method_results: true      # Cache results by method
    
    # Quality assurance
    enable_method_comparison: true  # Track method performance
    log_method_selection: true      # Log which method was used
    validate_method_consistency: false  # Check consistency between methods
  
  # Include settings for both methods
  halueval_settings:
    use_two_stage_generation: true
    enable_filtering: true
    
  dodhalueval_settings:
    injection_strategies: ["factual", "logical", "context"]
    system_prompt_strategy: "hallucination_prone"
```

## API Provider Configuration

### OpenAI Provider

```yaml
api_configs:
  openai:
    provider: "openai"
    model: "gpt-4"                # Options: gpt-3.5-turbo, gpt-4, gpt-4-turbo
    api_key: "${OPENAI_API_KEY}"  # Environment variable reference
    api_base: null                # Custom API base URL (optional)
    organization: null            # OpenAI organization ID (optional)
    
    # Request configuration
    max_retries: 3
    timeout: 30
    retry_delay: 1.0
    backoff_multiplier: 2.0
    
    # Rate limiting
    requests_per_minute: 60
    tokens_per_minute: 90000
    
    # Generation parameters
    temperature: 0.7
    max_tokens: 2000
    top_p: 1.0
    frequency_penalty: 0.0
    presence_penalty: 0.0
    
    # Cost management
    enable_cost_tracking: true
    max_cost_per_request: 1.0     # USD
    warn_on_high_cost: true
```

### Fireworks Provider

```yaml
api_configs:
  fireworks:
    provider: "fireworks"
    model: "llama-v3p1-70b-instruct"  # Options: llama-v3p1-70b-instruct, mixtral-8x7b-instruct
    api_key: "${FIREWORKS_API_KEY}"
    api_base: "https://api.fireworks.ai/inference/v1"
    
    # Request configuration
    max_retries: 3
    timeout: 45
    retry_delay: 2.0
    
    # Generation parameters
    temperature: 0.7
    max_tokens: 2000
    top_p: 0.9
    top_k: 40
    
    # Model-specific settings
    enable_system_prompt: true
    supports_functions: false
    context_length: 4096
```

### Mock Provider (Testing)

```yaml
api_configs:
  mock:
    provider: "mock"
    model: "mock-gpt-4"
    
    # Mock behavior
    response_delay: 0.5           # Simulate API delay
    failure_rate: 0.05            # Simulate 5% failure rate
    
    # Response generation
    response_templates:
      - "Based on the document, {topic} involves..."
      - "According to military doctrine, {topic} requires..."
      - "The document specifies that {topic} must..."
    
    # Error simulation
    simulate_rate_limits: true
    simulate_timeouts: true
    simulate_server_errors: false
```

## Document Processing Configuration

### PDF Processing

```yaml
pdf_processing:
  # Chunking strategy
  chunk_size: 1000              # Characters per chunk
  chunk_overlap: 200            # Overlap between chunks
  min_chunk_length: 100         # Minimum viable chunk size
  max_chunks_per_document: 100  # Limit chunks per document
  
  # Content filtering
  filter_low_quality_chunks: true
  min_text_density: 0.5         # Minimum text/total character ratio
  exclude_headers_footers: true
  exclude_page_numbers: true
  
  # Text extraction
  preserve_formatting: false
  normalize_whitespace: true
  extract_tables: false
  extract_images: false
  
  # Processing optimization
  enable_parallel_processing: true
  max_workers: 4
  memory_limit_mb: 1024
  
  # Quality control
  validate_extracted_text: true
  min_words_per_chunk: 20
  max_special_char_ratio: 0.3
```

### Document Validation

```yaml
document_validation:
  # Content requirements
  min_document_length: 1000     # Minimum characters
  max_document_length: 1000000  # Maximum characters
  min_chunks_required: 5        # Minimum viable chunks
  
  # Language detection
  expected_language: "en"
  language_confidence_threshold: 0.8
  
  # Content type validation
  allowed_content_types: ["application/pdf"]
  validate_pdf_structure: true
  require_text_content: true
  
  # Security checks
  scan_for_malicious_content: false
  check_file_signatures: true
  max_file_size_mb: 50
```

## Prompt Generation Configuration

### Template-Based Generation

```yaml
prompt_generation:
  strategies: ["template", "llm_based"]
  
  template_settings:
    template_file: "data/prompts/templates.yaml"
    categories:
      - equipment_specifications
      - operational_procedures  
      - doctrine_principles
      - leadership_ethics
      - tactical_operations
    
    # Template selection
    selection_strategy: "weighted"  # Options: random, weighted, balanced
    category_weights:
      equipment_specifications: 0.3
      operational_procedures: 0.25
      doctrine_principles: 0.2
      leadership_ethics: 0.15
      tactical_operations: 0.1
    
    # Prompt customization
    enable_perturbation: true
    perturbation_strategies:
      - paraphrase
      - question_type_change
      - complexity_variation
      - context_expansion
    
    # Quality control
    validate_generated_prompts: true
    min_prompt_length: 20
    max_prompt_length: 500
    filter_duplicate_prompts: true
```

### LLM-Based Generation

```yaml
prompt_generation:
  llm_settings:
    provider: "openai"
    model: "gpt-3.5-turbo"
    
    # Generation parameters
    temperature: 0.8
    max_tokens: 300
    batch_size: 10
    
    # Prompt engineering
    generation_prompts:
      - "Generate a question about military equipment specifications from this text:"
      - "Create a question about operational procedures based on this content:"
      - "Formulate a question about military doctrine from this passage:"
    
    # Content filtering
    filter_content_rich_chunks: true
    min_chunk_words: 50
    content_richness_threshold: 0.6
    
    # Quality assurance
    validate_question_quality: true
    check_answerability: true
    filter_overly_specific: true
    filter_too_general: true
```

### Perturbation Settings

```yaml
perturbation:
  # Available strategies
  strategies:
    - paraphrase
    - question_type_change
    - complexity_variation
    - context_expansion
    - negation_introduction
    - temporal_shift
    - scope_modification
    - specificity_change
    - perspective_change
    - assumption_challenge
  
  # Strategy weights
  strategy_weights:
    paraphrase: 0.2
    question_type_change: 0.15
    complexity_variation: 0.15
    context_expansion: 0.1
    negation_introduction: 0.1
    temporal_shift: 0.1
    scope_modification: 0.05
    specificity_change: 0.05
    perspective_change: 0.05
    assumption_challenge: 0.05
  
  # Perturbation parameters
  perturbation_rate: 0.3          # Probability of applying perturbation
  max_perturbations_per_prompt: 2
  preserve_intent: true
  validate_perturbed_prompts: true
```

## Hallucination Detection Configuration

### Evaluation Methods

```yaml
hallucination_detection:
  methods: ["hhem", "g_eval", "selfcheck"]
  
  # HuggingFace HHEM
  hhem_settings:
    model_name: "vectara/hallucination_evaluation_model"
    confidence_threshold: 0.5
    batch_size: 16
    enable_gpu: true
    cache_predictions: true
  
  # G-Eval
  g_eval_settings:
    provider: "openai"
    model: "gpt-4"
    evaluation_criteria:
      - factual_accuracy
      - logical_consistency
      - contextual_relevance
    temperature: 0.0
    enable_caching: true
  
  # SelfCheckGPT
  selfcheck_settings:
    provider: "openai"
    model: "gpt-3.5-turbo"
    num_samples: 5
    temperature: 0.8
    similarity_threshold: 0.7
    enable_sentence_level: true
  
  # Ensemble evaluation
  ensemble_evaluation:
    enabled: true
    consensus_method: "majority_vote"  # Options: majority_vote, weighted_average, threshold_based
    method_weights:
      hhem: 0.4
      g_eval: 0.4
      selfcheck: 0.2
    confidence_threshold: 0.5
    require_unanimous: false
```

### Custom Evaluators

```yaml
custom_evaluators:
  military_domain_evaluator:
    type: "rule_based"
    rules:
      - check_equipment_consistency
      - validate_doctrine_alignment
      - verify_procedural_accuracy
    
  semantic_consistency_checker:
    type: "embedding_based"
    model: "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: 0.8
    
  fact_verification:
    type: "external_api"
    api_endpoint: "https://factcheck.api.example.com"
    timeout: 10
    enable_caching: true
```

## Output and Export Configuration

### Output Settings

```yaml
output:
  # Directory structure
  base_directory: "output/"
  experiment_name: "experiment_$(date)"
  create_timestamped_dirs: true
  
  # File formats
  export_formats: ["jsonl", "json", "csv"]
  compression: "gzip"           # Options: none, gzip, bz2
  
  # Content control
  save_intermediate: true
  intermediate_formats: ["json"]
  save_metadata: true
  save_configuration: true
  
  # Report generation
  generate_report: true
  report_format: "markdown"     # Options: markdown, html, pdf
  include_visualizations: true
  include_statistics: true
  
  # Quality assurance
  validate_outputs: true
  backup_results: true
  cleanup_temp_files: true
```

### Dataset Export

```yaml
dataset_export:
  # HaluEval compatibility
  halueval_format:
    enabled: true
    include_metadata: true
    validate_schema: true
    
  # Custom formats
  dod_benchmark_format:
    enabled: true
    include_source_documents: false
    include_generation_details: true
    
  # Export options
  split_datasets: true
  train_split: 0.8
  validation_split: 0.1
  test_split: 0.1
  stratify_by: "hallucination_type"
  
  # File naming
  filename_template: "{dataset_name}_{split}_{timestamp}"
  include_version: true
  include_checksum: true
```

## Environment Variables

### Global Settings

```bash
# Framework configuration
export DODHALUEVAL_CONFIG_FILE="configs/production.yaml"
export DODHALUEVAL_LOG_LEVEL="INFO"
export DODHALUEVAL_CACHE_DIR="cache/"
export DODHALUEVAL_OUTPUT_DIR="output/"

# Performance settings
export DODHALUEVAL_BATCH_SIZE=50
export DODHALUEVAL_MAX_WORKERS=4
export DODHALUEVAL_MEMORY_LIMIT_MB=2048
```

### API Keys

```bash
# Provider API keys
export OPENAI_API_KEY="sk-your-openai-key"
export FIREWORKS_API_KEY="your-fireworks-key"
export HUGGINGFACE_TOKEN="your-hf-token"

# Organization settings
export OPENAI_ORG_ID="org-your-org-id"
export WANDB_API_KEY="your-wandb-key"
```

### Generation Method Overrides

```bash
# Method selection
export DODHALUEVAL_GENERATION_METHOD="halueval"
export DODHALUEVAL_HALLUCINATION_RATE=0.3

# HaluEval settings
export DODHALUEVAL_HALUEVAL_PATTERNS="factual_contradiction,context_misunderstanding"
export DODHALUEVAL_USE_TWO_STAGE=true
export DODHALUEVAL_MAX_KNOWLEDGE_LENGTH=2000

# DoDHaluEval settings
export DODHALUEVAL_INJECTION_STRATEGIES="factual,logical,context"
export DODHALUEVAL_SYSTEM_PROMPT_STRATEGY="hallucination_prone"

# Hybrid settings
export DODHALUEVAL_PRIMARY_METHOD="halueval"
export DODHALUEVAL_FALLBACK_METHOD="dodhalueval"
export DODHALUEVAL_COMPARISON_MODE=false
```

### Processing Control

```bash
# Cache control
export DODHALUEVAL_BYPASS_CACHE=false
export DODHALUEVAL_BYPASS_PDF_CACHE=false
export DODHALUEVAL_BYPASS_RESPONSE_CACHE=true

# Document processing
export DODHALUEVAL_PDF_CHUNK_SIZE=1000
export DODHALUEVAL_PDF_CHUNK_OVERLAP=200
export DODHALUEVAL_MAX_CHUNKS_PER_DOC=100

# Quality control
export DODHALUEVAL_VALIDATE_OUTPUTS=true
export DODHALUEVAL_REQUIRE_HUMAN_REVIEW=false
```

## Configuration Validation

### Schema Validation

DoDHaluEval validates all configuration using Pydantic models:

```python
from dodhalueval.models.config import PipelineConfig

# Load and validate configuration
try:
    config = PipelineConfig.from_yaml("configs/experiment.yaml")
    print("Configuration valid!")
except ValidationError as e:
    print(f"Configuration error: {e}")
```

### Configuration Testing

```bash
# Test configuration validity
dodhalueval validate-config --config configs/experiment.yaml

# Check provider connectivity
dodhalueval test-providers --config configs/experiment.yaml

# Validate templates and patterns
dodhalueval validate-templates --config configs/experiment.yaml
```

## Best Practices

### Configuration Management

1. **Version Control**: Store all configurations in version control
2. **Environment Separation**: Use different configs for dev/test/prod
3. **Parameter Documentation**: Document all custom parameters
4. **Validation**: Always validate configurations before running

### Security

1. **API Key Management**: Never commit API keys to version control
2. **Environment Variables**: Use environment variables for sensitive data
3. **Access Control**: Restrict access to production configurations
4. **Audit Trail**: Log configuration changes and usage

### Performance Optimization

1. **Resource Limits**: Set appropriate memory and CPU limits
2. **Batch Sizing**: Optimize batch sizes for your hardware
3. **Caching**: Enable caching for repeated operations
4. **Provider Selection**: Choose cost-effective providers for your use case

### Reproducibility

1. **Fixed Seeds**: Use fixed random seeds for reproducible results
2. **Configuration Freezing**: Save exact configurations with results
3. **Dependency Tracking**: Track library versions and dependencies
4. **Environment Documentation**: Document runtime environment details

This comprehensive configuration reference provides complete control over all aspects of the DoDHaluEval framework, enabling fine-tuned customization for specific research needs and operational requirements.