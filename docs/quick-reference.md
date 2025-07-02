# Quick Reference Guide

This quick reference provides essential commands, configuration options, and code snippets for common DoDHaluEval operations.

## Common Commands

### Pipeline Execution

```bash
# Complete pipeline with default settings
python scripts/run_pipeline.py

# Custom configuration
python scripts/run_pipeline.py --config configs/experiment.yaml

# Specify output directory
python scripts/run_pipeline.py --output output/my_experiment/

# Bypass cache for fresh run
python scripts/run_pipeline.py --bypass-cache
```

### CLI Commands

```bash
# Document processing
dodhalueval process-docs --input data/CSC/ --output data/processed/

# Prompt generation
dodhalueval generate-prompts --input data/processed/ --count 1000 --strategies template,llm_based

# Response generation
dodhalueval generate-responses --input data/prompts/ --provider openai --hallucination-rate 0.3

# Evaluation
dodhalueval evaluate --input data/responses/ --methods hhem,g_eval,selfcheck

# Dataset building
dodhalueval build-dataset --input data/evaluations/ --output benchmark.jsonl --format jsonl

# System information
dodhalueval info

# Configuration validation
dodhalueval validate-config --config configs/experiment.yaml
```

### Testing Commands

```bash
# Quick functionality test
python scripts/test_prompt_generation.py --mode quick

# Test with sample data
python scripts/test_prompt_generation.py --mode demo --verbose

# Test with CSC documents
python scripts/test_prompt_generation.py --mode csc --max-prompts 20

# Test LLM integration
python scripts/test_prompt_generation.py --mode llm --provider openai

# Run full test suite
pytest tests/ -v
```

## Configuration Templates

### Basic Configuration

```yaml
# configs/basic.yaml
version: "0.1.0"
environment: "development"
batch_size: 20

api_configs:
  openai:
    provider: "openai"
    model: "gpt-4"
    max_retries: 3

prompt_generation:
  strategies: ["template"]
  max_prompts_per_document: 50

evaluation_methods:
  - method: "vectara_hhem"
    enabled: true
```

### Production Configuration

```yaml
# configs/production.yaml
version: "0.1.0"
environment: "production"
batch_size: 100
max_concurrent_requests: 10

api_configs:
  openai:
    provider: "openai"
    model: "gpt-4"
    max_retries: 5
    timeout: 60
  fireworks:
    provider: "fireworks"
    model: "llama-v2-70b-chat"
    max_retries: 3

prompt_generation:
  strategies: ["template", "llm_based"]
  max_prompts_per_document: 200
  hallucination_types: ["factual", "logical", "context"]

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

cache:
  enabled: true
  ttl_hours: 24

logging:
  level: "INFO"
  log_file: "logs/production.log"
```

## Code Snippets

### Basic Pipeline Usage

```python
from dodhalueval.core import HallucinationDetector, PromptGenerator, ResponseGenerator
from dodhalueval.data import PDFProcessor, DatasetBuilder
from dodhalueval.providers import OpenAIProvider
from dodhalueval.models.config import APIConfig, PromptGenerationConfig, ResponseConfig

# Setup
api_config = APIConfig(provider='openai', model='gpt-4', api_key='your-key')
provider = OpenAIProvider(api_config)

# Process documents
processor = PDFProcessor(chunk_size=1000)
result = processor.process_document("document.pdf")

# Generate prompts
prompt_config = PromptGenerationConfig()
generator = PromptGenerator(prompt_config)
prompts = generator.generate_from_chunks(result['chunks'])

# Generate responses
response_config = ResponseConfig(hallucination_rate=0.3)
response_gen = ResponseGenerator({'openai': provider}, response_config)
responses = await response_gen.generate_responses(prompts, ['openai'])

# Evaluate
detector = HallucinationDetector(provider)
evaluations = await detector.evaluate_batch(responses, prompts)

# Build dataset
builder = DatasetBuilder("output/")
dataset = builder.build_halueval_format(prompts, responses, evaluations)
```

### Configuration Loading

```python
from dodhalueval.utils.config import load_config

# Load from file
config = load_config("configs/experiment.yaml")

# Load with environment override
config = load_config(environment="production")

# Load with explicit config directory
config = load_config("custom.yaml", config_dir="my_configs/")
```

### Batch Processing

```python
import asyncio
from pathlib import Path

async def process_multiple_documents():
    pdf_files = list(Path("data/CSC/").glob("*.pdf"))
    
    for pdf_file in pdf_files:
        try:
            result = processor.process_document(pdf_file)
            prompts = generator.generate_from_chunks(result['chunks'])
            responses = await response_gen.generate_responses(prompts, ['openai'])
            evaluations = await detector.evaluate_batch(responses, prompts)
            
            # Save results
            dataset = builder.build_halueval_format(prompts, responses, evaluations)
            output_file = f"output/{pdf_file.stem}_benchmark.jsonl"
            builder.export_jsonl(dataset, output_file)
            
        except Exception as e:
            print(f"Failed to process {pdf_file}: {e}")

asyncio.run(process_multiple_documents())
```

### Custom Evaluation

```python
from dodhalueval.core.evaluators import HuggingFaceHHEMEvaluator, GEvalEvaluator

# Individual evaluators
hhem = HuggingFaceHHEMEvaluator()
g_eval = GEvalEvaluator(llm_provider=provider)

# Run evaluations
hhem_results = await hhem.evaluate_batch(responses, prompts)
g_eval_results = await g_eval.evaluate_batch(responses, prompts)

# Custom ensemble
def custom_ensemble(results_dict):
    weights = {'hhem': 0.4, 'g_eval': 0.6}
    return sum(weights[method] * score for method, score in results_dict.items())
```

## Environment Variables

### Required Variables

```bash
# API Keys
export OPENAI_API_KEY="your-openai-api-key"
export FIREWORKS_API_KEY="your-fireworks-api-key"  # Optional
export HUGGINGFACE_TOKEN="your-hf-token"  # Optional
```

### Configuration Overrides

```bash
# Processing settings
export DODHALUEVAL_BATCH_SIZE=50
export DODHALUEVAL_MAX_CONCURRENT_REQUESTS=5

# PDF processing
export DODHALUEVAL_PDF_PROCESSING_CHUNK_SIZE=1200
export DODHALUEVAL_PDF_PROCESSING_CHUNK_OVERLAP=250
export DODHALUEVAL_PDF_PROCESSING_MAX_PAGES=10

# Cache settings
export DODHALUEVAL_CACHE_ENABLED=true
export DODHALUEVAL_CACHE_TTL_HOURS=48

# Logging
export DODHALUEVAL_LOGGING_LEVEL=DEBUG
export DODHALUEVAL_LOGGING_CONSOLE_OUTPUT=true
```

## File Structure Reference

```
dodhalueval/
├── src/dodhalueval/           # Source code
│   ├── core/                  # Core processing components
│   ├── data/                  # Data processing and storage
│   ├── providers/             # LLM provider integrations
│   ├── models/                # Data models and schemas
│   ├── utils/                 # Utilities and helpers
│   └── cli/                   # Command-line interface
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── configs/                   # Configuration files
├── data/                      # Data directory
│   ├── CSC/                   # Source documents
│   ├── processed/             # Processed documents
│   ├── prompts/               # Generated prompts
│   ├── responses/             # Generated responses
│   └── evaluations/           # Evaluation results
├── output/                    # Pipeline outputs
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
└── logs/                      # Log files
```

## Troubleshooting

### Common Issues

**Import Error**:
```bash
# Ensure you're in the correct environment
source dodhalueval-env/bin/activate
pip install -e ".[dev]"
```

**API Connection Issues**:
```bash
# Test API connectivity
python scripts/test_api_providers.py --provider openai
```

**Memory Issues**:
```python
# Reduce batch sizes
config.batch_size = 10
config.pdf_processing.chunk_size = 500
```

**Cache Issues**:
```bash
# Clear cache
rm -rf /tmp/dodhalueval/
# Or disable cache
export DODHALUEVAL_CACHE_ENABLED=false
```

### Performance Optimization

```python
# Optimize for large documents
processor = PDFProcessor(
    chunk_size=800,           # Smaller chunks
    max_pages=50,            # Limit pages for testing
    cache_enabled=True       # Enable caching
)

# Optimize response generation
response_config = ResponseConfig(
    concurrent_requests=3,    # Reduce concurrency
    timeout=60,              # Increase timeout
    batch_size=5             # Smaller batches
)
```

## Quick Validation

### Test Installation

```bash
# Check imports
python -c "import dodhalueval; print('✓ Installation successful')"

# Test basic functionality
python scripts/test_prompt_generation.py --mode quick

# Validate configuration
dodhalueval validate-config --config configs/default.yaml
```

### Test Pipeline

```bash
# Quick end-to-end test
python scripts/run_pipeline.py --config configs/test_pipeline.yaml

# Check outputs
ls -la output/pipeline_results/
```

### Verify API Connections

```bash
# Test OpenAI
python -c "from dodhalueval.providers import OpenAIProvider; print('✓ OpenAI provider available')"

# Test all providers
python scripts/test_api_providers.py --provider all
```

This quick reference covers the most common operations and configurations. For detailed information, see the full documentation in the `docs/` directory.