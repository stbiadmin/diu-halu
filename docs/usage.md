# Usage Guide

This comprehensive guide demonstrates how to use DoDHaluEval for creating and evaluating hallucination benchmarks in the Department of Defense knowledge domain using multiple generation methodologies.

## Quick Start

### Basic Pipeline Execution with Method Selection

Run the complete pipeline with configurable generation methodology:

```bash
# Run with HaluEval methodology
python scripts/run_pipeline.py --config configs/halueval_method.yaml

# Run with original DoDHaluEval methodology
python scripts/run_pipeline.py --config configs/dodhalueval_method.yaml

# Run with hybrid approach
python scripts/run_pipeline.py --config configs/hybrid_method.yaml

# Run with default configuration
python scripts/run_pipeline.py

# Run with specific output directory
python scripts/run_pipeline.py --config configs/halueval_method.yaml --output output/halueval_experiment/
```

### CLI Commands

DoDHaluEval provides a comprehensive command-line interface with generation method support:

```bash
# Show available commands
dodhalueval --help

# Process PDF documents
dodhalueval process-docs --input data/CSC/ --output data/processed/

# Generate prompts
dodhalueval generate-prompts --input data/processed/ --output data/prompts/ --count 1000

# Generate responses with method selection
dodhalueval generate-responses --input data/prompts/ --output data/responses/ --provider openai --generation-method halueval

# Generate responses with hybrid method
dodhalueval generate-responses --input data/prompts/ --output data/responses/ --provider openai --generation-method hybrid --primary-method halueval --fallback-method dodhalueval

# Evaluate responses
dodhalueval evaluate --input data/responses/ --output data/evaluations/ --methods hhem,g_eval,selfcheck

# Build final dataset with HaluEval compatibility
dodhalueval build-dataset --input data/evaluations/ --output benchmark_v1.jsonl --format halueval

# Validate configuration and methods
dodhalueval validate-config --config configs/halueval_method.yaml
dodhalueval test-methods --config configs/hybrid_method.yaml
```

## Pipeline Components

### 1. Document Processing

Process DoD PDF documents to extract structured knowledge:

```python
from dodhalueval.data import PDFProcessor

# Initialize processor
processor = PDFProcessor(
    chunk_size=1000,
    chunk_overlap=200,
    max_pages=None,  # Process all pages
    cache_enabled=True
)

# Process single document
result = processor.process_document("data/CSC/MCDP1_Warfighting.pdf")
print(f"Extracted {len(result['chunks'])} chunks from {len(result['pages'])} pages")

# Process multiple documents
import glob
for pdf_path in glob.glob("data/CSC/*.pdf"):
    try:
        result = processor.process_document(pdf_path)
        print(f"Processed {pdf_path}: {result['metadata']['title']}")
    except Exception as e:
        print(f"Failed to process {pdf_path}: {e}")
```

#### Advanced Document Processing

```python
from dodhalueval.data import PDFProcessor, DocumentMetadata

# Custom processing configuration
processor = PDFProcessor(
    chunk_size=800,           # Smaller chunks for detailed analysis
    chunk_overlap=150,        # Overlap for context preservation
    max_pages=50,            # Limit for testing
    min_chunk_length=100,    # Filter short chunks
    preserve_structure=True   # Maintain document hierarchy
)

# Process with metadata extraction
result = processor.process_document(
    "data/CSC/MCDP1_Warfighting.pdf",
    extract_metadata=True,
    detect_structure=True
)

# Access structured content
for chunk in result['chunks']:
    print(f"Section: {chunk.section}")
    print(f"Content: {chunk.content[:200]}...")
    print(f"Concepts: {chunk.metadata.get('concepts', [])}")
```

### 2. Prompt Generation

Generate hallucination-prone prompts using multiple strategies:

#### Template-Based Generation

```python
from dodhalueval.core import PromptGenerator
from dodhalueval.models.config import PromptGenerationConfig

# Configure generation
config = PromptGenerationConfig(
    strategies=['template'],
    max_prompts_per_document=100,
    hallucination_types=['factual', 'logical', 'context'],
    template_file='data/prompts/templates.yaml'
)

# Initialize generator
generator = PromptGenerator(config)

# Generate from processed chunks
chunks = result['chunks']
prompts = generator.generate_from_chunks(chunks)

print(f"Generated {len(prompts)} prompts")
for prompt in prompts[:3]:
    print(f"Type: {prompt.hallucination_type}")
    print(f"Text: {prompt.text}")
    print(f"Expected: {prompt.expected_answer}")
```

#### LLM-Based Generation

```python
from dodhalueval.core import LLMPromptGenerator
from dodhalueval.providers import OpenAIProvider
from dodhalueval.models.config import APIConfig

# Setup LLM provider
api_config = APIConfig(
    provider='openai',
    model='gpt-4',
    api_key=os.getenv('OPENAI_API_KEY')
)
provider = OpenAIProvider(api_config)

# Initialize LLM generator
llm_generator = LLMPromptGenerator(provider, config)

# Generate sophisticated prompts
import asyncio

async def generate_llm_prompts():
    prompts = await llm_generator.generate_hallucination_prone_prompts(
        source_content=chunks[0].content,
        chunk=chunks[0],
        num_prompts=10,
        strategy='factual_probing'
    )
    return prompts

llm_prompts = asyncio.run(generate_llm_prompts())
```

#### Perturbation Strategies

```python
from dodhalueval.core import PromptPerturbator

# Initialize perturbator
perturbator = PromptPerturbator()

# Apply specific perturbation
base_prompt = prompts[0]
perturbed = perturbator.perturb(
    base_prompt, 
    'entity_substitution', 
    chunks[0]
)

# Apply multiple strategies
variations = perturbator.apply_multiple_strategies(
    base_prompt, 
    chunks[0], 
    max_strategies=3
)

# Available strategies
strategies = [
    'entity_substitution',     # Replace equipment names
    'numerical_manipulation',  # Change numbers/quantities
    'temporal_confusion',      # Mix time periods
    'negation_injection',      # Add logical negations
    'authority_confusion',     # Mix up regulations
    'causal_reversal',        # Reverse cause-effect
    'multi_hop_reasoning',    # Add complexity
    'scope_expansion',        # Go beyond source
    'conditional_complexity', # Add conditions
    'quantifier_manipulation' # Change all/some/none
]
```

### 3. Response Generation

Generate responses with controlled hallucination injection:

```python
from dodhalueval.core import ResponseGenerator
from dodhalueval.models.config import ResponseConfig

# Configure response generation
response_config = ResponseConfig(
    hallucination_rate=0.3,      # 30% injection rate
    concurrent_requests=5,        # Parallel processing
    temperature=0.7,             # Response creativity
    max_tokens=500              # Response length limit
)

# Setup multiple providers
providers = {
    'openai': OpenAIProvider(openai_config),
    'fireworks': FireworksProvider(fireworks_config)
}

# Initialize generator
response_generator = ResponseGenerator(providers, response_config)

# Generate responses
async def generate_responses():
    responses = await response_generator.generate_responses(
        prompts=prompts,
        providers=['openai', 'fireworks']
    )
    return responses

responses = asyncio.run(generate_responses())

# Analyze generation results
hallucinated_count = sum(1 for r in responses if r.contains_hallucination)
print(f"Generated {len(responses)} responses, {hallucinated_count} with hallucinations")
```

#### Custom Hallucination Injection

```python
from dodhalueval.core import HallucinationInjector

# Initialize injector
injector = HallucinationInjector()

# Configure injection types
injection_config = {
    'factual': {
        'probability': 0.4,
        'strategies': ['false_facts', 'incorrect_numbers', 'wrong_entities']
    },
    'logical': {
        'probability': 0.3,
        'strategies': ['contradiction', 'non_sequitur', 'circular_reasoning']
    },
    'context': {
        'probability': 0.3,
        'strategies': ['context_mixing', 'scope_violation', 'source_confusion']
    }
}

# Apply custom injection
for prompt in prompts:
    if injector.should_inject_hallucination(prompt, injection_config):
        hallucinated_response = injector.inject_hallucination(
            prompt, 
            base_response, 
            prompt.hallucination_type
        )
```

### 4. Hallucination Detection

Evaluate responses using multiple detection methods:

```python
from dodhalueval.core import HallucinationDetector

# Configure detection methods
detector = HallucinationDetector(
    llm_provider=provider,
    enable_huggingface_hhem=True,
    enable_g_eval=True,
    enable_selfcheck=True,
    confidence_threshold=0.7
)

# Run evaluation
async def evaluate_responses():
    evaluations = await detector.evaluate_batch(
        responses=responses,
        prompts=prompts,
        batch_size=10
    )
    return evaluations

evaluations = asyncio.run(evaluate_responses())

# Analyze results
for eval_result in evaluations[:5]:
    print(f"Response ID: {eval_result.response_id}")
    print(f"Detected Hallucination: {eval_result.is_hallucinated}")
    print(f"Confidence: {eval_result.confidence_score:.3f}")
    print(f"Method: {eval_result.method}")
```

#### Individual Detection Methods

```python
from dodhalueval.core.evaluators import (
    HuggingFaceHHEMEvaluator,
    GEvalEvaluator,
    SelfCheckGPTEvaluator
)

# HuggingFace HHEM
hhem = HuggingFaceHHEMEvaluator()
hhem_results = await hhem.evaluate_batch(responses, prompts)

# G-Eval with custom criteria
g_eval = GEvalEvaluator(llm_provider=provider)
g_eval_results = await g_eval.evaluate_batch(
    responses, 
    prompts,
    criteria=['consistency', 'factuality', 'fluency']
)

# SelfCheckGPT
selfcheck = SelfCheckGPTEvaluator(llm_provider=provider)
selfcheck_results = await selfcheck.evaluate_batch(responses, prompts)
```

### 5. Dataset Building

Compile evaluation results into benchmark datasets:

```python
from dodhalueval.data import DatasetBuilder

# Initialize builder
builder = DatasetBuilder(output_dir="output/benchmarks/")

# Build HaluEval-compatible dataset
dataset = builder.build_halueval_format(
    prompts=prompts,
    responses=responses,
    evaluations=evaluations,
    documents=processed_documents,
    dataset_name="dod_halueval_v1",
    additional_metadata={
        "created_by": "DoDHaluEval Pipeline",
        "domain": "Department of Defense",
        "version": "1.0.0"
    }
)

# Export in multiple formats
jsonl_path = builder.export_jsonl(dataset, "dod_benchmark.jsonl")
json_path = builder.export_json(dataset, "dod_benchmark.json")
csv_path = builder.export_csv(dataset, "dod_benchmark.csv")

print(f"Dataset exported to:")
print(f"  JSONL: {jsonl_path}")
print(f"  JSON: {json_path}")
print(f"  CSV: {csv_path}")
```

#### Dataset Analysis

```python
from dodhalueval.utils import MetricsCalculator

# Calculate comprehensive metrics
calculator = MetricsCalculator()

# Extract ground truth and predictions
ground_truth = [sample.label for sample in dataset.samples]
predictions = [sample.ensemble_score for sample in dataset.samples]

# Calculate detection metrics
metrics = calculator.calculate_detection_metrics(predictions, ground_truth)

print(f"Detection Performance:")
print(f"  Accuracy: {metrics.accuracy:.3f}")
print(f"  Precision: {metrics.precision:.3f}")
print(f"  Recall: {metrics.recall:.3f}")
print(f"  F1-Score: {metrics.f1_score:.3f}")

# Per-category analysis
category_labels = [sample.hallucination_type for sample in dataset.samples]
category_metrics = calculator.calculate_per_category_metrics(
    evaluations, ground_truth, category_labels
)

for category, cat_metrics in category_metrics.items():
    print(f"{category}: F1={cat_metrics.f1_score:.3f}")
```

## Configuration Management

### Configuration Files

DoDHaluEval uses YAML configuration files for reproducible experiments:

```yaml
# configs/experiment.yaml
version: "0.1.0"
environment: "production"

# Processing settings
batch_size: 50
max_concurrent_requests: 10

# PDF processing
pdf_processing:
  chunk_size: 1000
  chunk_overlap: 200
  max_pages: null
  cache_enabled: true

# API configurations
api_configs:
  openai:
    provider: "openai"
    model: "gpt-4"
    max_retries: 3
    timeout: 30
  fireworks:
    provider: "fireworks"
    model: "llama-v2-70b-chat"
    max_retries: 3
    timeout: 30

# Prompt generation
prompt_generation:
  strategies: ["template", "llm_based"]
  max_prompts_per_document: 100
  hallucination_types: ["factual", "logical", "context"]
  template_file: "data/prompts/templates.yaml"

# Response generation
response_generation:
  hallucination_rate: 0.3
  concurrent_requests: 5
  temperature: 0.7
  max_tokens: 500

# Evaluation methods
evaluation_methods:
  - method: "vectara_hhem"
    enabled: true
    confidence_threshold: 0.5
  - method: "g_eval"
    enabled: true
    confidence_threshold: 0.6
  - method: "self_check_gpt"
    enabled: true
    confidence_threshold: 0.7

# Output settings
output:
  format: "jsonl"
  output_dir: "output/experiments/"
  include_metadata: true

# Caching
cache:
  enabled: true
  pdf_cache_dir: "/tmp/dodhalueval/pdf_cache"
  llm_cache_dir: "/tmp/dodhalueval/llm_cache"
  ttl_hours: 24

# Logging
logging:
  level: "INFO"
  log_file: "logs/experiment.log"
  console_output: true
```

### Environment Variables

Override configuration with environment variables:

```bash
# Core settings
export DODHALUEVAL_BATCH_SIZE=100
export DODHALUEVAL_ENVIRONMENT=production

# API keys
export OPENAI_API_KEY=your-openai-key
export FIREWORKS_API_KEY=your-fireworks-key

# Processing settings
export DODHALUEVAL_PDF_PROCESSING_CHUNK_SIZE=1200
export DODHALUEVAL_PDF_PROCESSING_MAX_PAGES=10

# Cache settings
export DODHALUEVAL_CACHE_ENABLED=true
export DODHALUEVAL_CACHE_TTL_HOURS=48

# Logging
export DODHALUEVAL_LOGGING_LEVEL=DEBUG
export DODHALUEVAL_LOGGING_CONSOLE_OUTPUT=false
```

## Advanced Usage Patterns

### Batch Processing

Process large document collections efficiently:

```python
import asyncio
from pathlib import Path
from dodhalueval.core import PipelineOrchestrator

async def process_document_collection():
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(config_path="configs/batch_processing.yaml")
    
    # Find all PDF files
    pdf_files = list(Path("data/CSC/").glob("*.pdf"))
    
    # Process in batches
    batch_size = 10
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:i + batch_size]
        
        # Process batch
        results = await orchestrator.process_document_batch(batch)
        
        # Save intermediate results
        batch_id = f"batch_{i//batch_size + 1}"
        orchestrator.save_intermediate_results(results, f"output/{batch_id}/")
        
        print(f"Completed batch {batch_id}: {len(batch)} documents")

# Run batch processing
asyncio.run(process_document_collection())
```

### Custom Evaluation Metrics

Implement domain-specific evaluation criteria:

```python
from dodhalueval.core.evaluators.base import BaseEvaluator

class DoDSpecificEvaluator(BaseEvaluator):
    """Custom evaluator for DoD domain-specific criteria."""
    
    def __init__(self, llm_provider, dod_knowledge_base):
        super().__init__(llm_provider)
        self.knowledge_base = dod_knowledge_base
    
    async def evaluate_response(self, response, prompt, context=None):
        """Evaluate response against DoD-specific criteria."""
        
        # Check factual accuracy against DoD sources
        factual_score = await self._check_factual_accuracy(response, context)
        
        # Verify regulatory compliance
        compliance_score = await self._check_regulatory_compliance(response)
        
        # Assess operational relevance
        relevance_score = await self._assess_operational_relevance(response, prompt)
        
        # Calculate weighted composite score
        composite_score = (
            factual_score * 0.4 +
            compliance_score * 0.3 +
            relevance_score * 0.3
        )
        
        return EvaluationResult(
            response_id=response.id,
            method="dod_specific",
            is_hallucinated=composite_score < 0.7,
            confidence_score=composite_score,
            details={
                "factual_accuracy": factual_score,
                "regulatory_compliance": compliance_score,
                "operational_relevance": relevance_score
            }
        )

# Use custom evaluator
dod_evaluator = DoDSpecificEvaluator(provider, knowledge_base)
custom_evaluations = await dod_evaluator.evaluate_batch(responses, prompts)
```

### Pipeline Orchestration

Coordinate complex workflows:

```python
from dodhalueval.core import PipelineOrchestrator

# Initialize with configuration
orchestrator = PipelineOrchestrator("configs/full_pipeline.yaml")

# Define workflow stages
workflow = [
    {
        "stage": "document_processing",
        "inputs": {"pdf_directory": "data/CSC/"},
        "outputs": {"processed_docs": "output/processed/"}
    },
    {
        "stage": "prompt_generation", 
        "inputs": {"processed_docs": "output/processed/"},
        "outputs": {"prompts": "output/prompts/"}
    },
    {
        "stage": "response_generation",
        "inputs": {"prompts": "output/prompts/"},
        "outputs": {"responses": "output/responses/"}
    },
    {
        "stage": "evaluation",
        "inputs": {"responses": "output/responses/"},
        "outputs": {"evaluations": "output/evaluations/"}
    },
    {
        "stage": "dataset_building",
        "inputs": {"evaluations": "output/evaluations/"},
        "outputs": {"benchmark": "output/benchmark.jsonl"}
    }
]

# Execute workflow
async def run_full_pipeline():
    results = await orchestrator.execute_workflow(workflow)
    return results

# Run with monitoring
results = asyncio.run(run_full_pipeline())
print(f"Pipeline completed: {results.summary}")
```

## Performance Optimization

### Memory Management

Optimize memory usage for large datasets:

```python
# Use streaming processing for large documents
from dodhalueval.data import StreamingPDFProcessor

processor = StreamingPDFProcessor(
    chunk_size=800,
    memory_limit_mb=4096,  # 4GB memory limit
    batch_size=50
)

# Process documents in memory-efficient batches
for batch_result in processor.process_documents_streaming("data/CSC/"):
    # Process batch immediately
    batch_prompts = generator.generate_from_chunks(batch_result.chunks)
    
    # Save intermediate results
    save_batch_results(batch_prompts, batch_result.batch_id)
    
    # Clear memory
    del batch_result, batch_prompts
```

### Parallel Processing

Maximize throughput with concurrent operations:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_pipeline_execution():
    # Setup thread pool for CPU-bound tasks
    with ThreadPoolExecutor(max_workers=4) as executor:
        
        # Process documents in parallel
        pdf_files = list(Path("data/CSC/").glob("*.pdf"))
        
        # Submit document processing tasks
        processing_tasks = [
            executor.submit(processor.process_document, pdf_file)
            for pdf_file in pdf_files
        ]
        
        # Await completion
        processed_docs = []
        for task in asyncio.as_completed(processing_tasks):
            result = await task
            processed_docs.append(result)
        
        return processed_docs

# Run parallel processing
results = asyncio.run(parallel_pipeline_execution())
```

### Caching Strategies

Optimize performance with intelligent caching:

```python
from dodhalueval.utils import CacheManager

# Configure multi-level caching
cache_manager = CacheManager(
    pdf_cache_enabled=True,
    pdf_cache_ttl_hours=24,
    llm_cache_enabled=True,
    llm_cache_ttl_hours=12,
    prompt_cache_enabled=True,
    max_cache_size_mb=2048
)

# Use cached operations
cached_processor = cache_manager.wrap_processor(processor)
cached_generator = cache_manager.wrap_generator(generator)

# Cache will automatically handle storage and retrieval
result = cached_processor.process_document("data/CSC/document.pdf")
prompts = cached_generator.generate_from_chunks(result['chunks'])
```

## Error Handling and Recovery

### Robust Error Handling

Implement comprehensive error recovery:

```python
from dodhalueval.utils.exceptions import (
    DoDHaluEvalError,
    APIError,
    ProcessingError,
    ValidationError
)

async def robust_pipeline_execution():
    try:
        # Execute pipeline with error recovery
        results = await orchestrator.execute_with_recovery(
            workflow,
            max_retries=3,
            retry_delay=5.0,
            continue_on_error=True
        )
        
    except APIError as e:
        # Handle API-specific errors
        logger.error(f"API error: {e}")
        await handle_api_error(e)
        
    except ProcessingError as e:
        # Handle processing errors
        logger.error(f"Processing error: {e}")
        await handle_processing_error(e)
        
    except ValidationError as e:
        # Handle validation errors
        logger.error(f"Validation error: {e}")
        await handle_validation_error(e)
        
    except DoDHaluEvalError as e:
        # Handle general framework errors
        logger.error(f"Framework error: {e}")
        await handle_framework_error(e)
        
    finally:
        # Cleanup resources
        await orchestrator.cleanup()

# Custom error handlers
async def handle_api_error(error):
    """Handle API-related errors with appropriate recovery."""
    if "rate_limit" in str(error):
        # Implement exponential backoff
        await asyncio.sleep(60)  # Wait 1 minute
        return "retry"
    elif "authentication" in str(error):
        # Check API keys
        logger.error("API authentication failed - check API keys")
        return "abort"
    else:
        # Generic API error
        return "retry_with_fallback"
```

This usage guide provides comprehensive examples for all major use cases of DoDHaluEval. For specific implementation details, refer to the [API Reference](api-reference.md) and [Architecture Documentation](architecture.md).