# DoDHaluEval: Department of Defense Hallucination Evaluation Framework

DoDHaluEval is a comprehensive, production-ready framework for creating and evaluating hallucination benchmarks specifically designed for Department of Defense knowledge domains. The system provides end-to-end capabilities for processing DoD documents, generating hallucination-prone prompts, creating controlled responses using multiple methodologies, and evaluating detection methods using state-of-the-art techniques.

## Overview

Large language models (LLMs) are increasingly deployed in critical applications, including military and defense contexts where factual accuracy is paramount. However, these models are prone to generating hallucinations - content that conflicts with source material or cannot be verified against factual knowledge. DoDHaluEval addresses this challenge by providing a domain-specific evaluation framework that enables rigorous assessment of hallucination detection capabilities across DoD knowledge areas.

### Key Features

- **Multi-Method Generation**: Supports three distinct hallucination generation approaches: DoDHaluEval (post-hoc injection), HaluEval (direct generation), and Hybrid (intelligent combination)
- **Domain-Specific Focus**: Tailored for Department of Defense knowledge areas including military doctrine, procedures, equipment specifications, and operational guidelines
- **HaluEval Compatibility**: Full implementation of the HaluEval research methodology (ArXiv:2305.11747) with DoD-specific enhancements
- **Advanced Hallucination Patterns**: Seven distinct patterns including factual contradiction, context misunderstanding, and military-specific equipment substitution
- **Multi-Strategy Prompt Generation**: Combines template-based, LLM-based, and perturbation methods with 92+ military domain templates
- **Controlled Hallucination Injection**: Systematic injection with configurable rates, types, and sophisticated knowledge context building
- **Multi-Method Detection**: Integrates HuggingFace HHEM, G-Eval, and SelfCheckGPT detection methods with ensemble evaluation
- **LLM Provider Agnostic**: Supports OpenAI GPT models, Fireworks AI, and extensible provider architecture
- **Production Ready**: Comprehensive testing, caching, error handling, and performance optimization with async processing

## Architecture

DoDHaluEval follows a modular pipeline architecture with sophisticated generation methodology selection:

```
Document Processing → Prompt Generation → Response Generation → Hallucination Detection → Dataset Building
                                            ↓
                      Method Selection: DoDHaluEval | HaluEval | Hybrid
```

### Core Components

1. **Document Processing Pipeline**: Extracts and structures knowledge from DoD PDF documents with intelligent chunking and metadata preservation
2. **Prompt Generation Engine**: Creates hallucination-prone prompts using 92+ template categories, LLM-based generation, and 10 perturbation strategies
3. **Multi-Method Response Generation System**: 
   - **DoDHaluEval Method**: Post-hoc hallucination injection with sophisticated system prompt manipulation
   - **HaluEval Method**: Direct hallucination generation following research paper methodology exactly
   - **Hybrid Method**: Intelligent combination with fallback strategies and comparison modes
4. **Knowledge Context Builder**: Semantic similarity-based context extraction with SentenceTransformers integration
5. **Hallucination Detection Framework**: Employs ensemble detection using multiple state-of-the-art methods
6. **Dataset Builder**: Compiles results into standardized benchmark formats with comprehensive metadata and HaluEval compatibility

For detailed architectural information, see [Architecture Documentation](docs/architecture.md).

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/stbiadmin/diu-halu.git
cd dodhalueval

# Create virtual environment
python -m venv dodhalueval-env
source dodhalueval-env/bin/activate  # On Windows: dodhalueval-env\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

### System Requirements

- **Python**: 3.10 or higher
- **Memory**: 8GB RAM (16GB recommended for large document processing)
- **Storage**: 5GB free disk space
- **Dependencies**: OpenAI API key, Fireworks AI API key (optional), HuggingFace token (optional)

For detailed installation instructions including platform-specific setup, Docker deployment, and troubleshooting, see [Installation Guide](docs/installation.md).

## Quick Usage

### Complete Pipeline with Method Selection

Run the entire pipeline with configurable generation methodology:

```bash
# Run with HaluEval methodology
python scripts/run_pipeline.py --config configs/halueval_method.yaml

# Run with original DoDHaluEval methodology  
python scripts/run_pipeline.py --config configs/dodhalueval_method.yaml

# Run with hybrid approach
python scripts/run_pipeline.py --config configs/hybrid_method.yaml

# Run with default configuration
python scripts/run_pipeline.py
```

### Step-by-Step Usage with Method Selection

```python
import asyncio
from dodhalueval.core import HallucinationDetector, PromptGenerator, ResponseGenerator
from dodhalueval.core.halueval_generator import HaluEvalGenerator
from dodhalueval.core.knowledge_builder import KnowledgeContextBuilder
from dodhalueval.data import PDFProcessor, DatasetBuilder
from dodhalueval.providers import OpenAIProvider

# 1. Process DoD documents
processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
result = processor.process_document("data/CSC/8906_AY_24_coursebook.pdf")

# 2. Generate hallucination-prone prompts
generator = PromptGenerator(config)
prompts = generator.generate_from_chunks(result['chunks'])

# 3. Configure generation method and generate responses
config = {
    'generation_method': 'halueval',  # Options: 'dodhalueval', 'halueval', 'hybrid'
    'halueval_settings': {
        'use_two_stage_generation': True,
        'enable_filtering': True,
        'hallucination_patterns': ['factual_contradiction', 'context_misunderstanding']
    }
}

provider = OpenAIProvider(api_config)
response_gen = ResponseGenerator({'openai': provider}, config)
responses = await response_gen.generate_responses(
    prompts, ['openai'], chunks=result['chunks']
)

# 4. Detect hallucinations
detector = HallucinationDetector(provider, enable_huggingface_hhem=True)
evaluations = await detector.evaluate_batch(responses, prompts)

# 5. Build benchmark dataset
builder = DatasetBuilder("output/")
dataset = builder.build_halueval_format(prompts, responses, evaluations)
builder.export_jsonl(dataset, "dod_benchmark.jsonl")
```

### CLI Interface

```bash
# Process documents
dodhalueval process-docs --input data/CSC/ --output data/processed/

# Generate prompts  
dodhalueval generate-prompts --input data/processed/ --count 1000

# Evaluate responses with method selection
dodhalueval evaluate --input data/responses/ --methods hhem,g_eval,selfcheck --generation-method halueval

# System information
dodhalueval info
```

For comprehensive usage examples and advanced patterns, see [Usage Guide](docs/usage.md) and [Generation Methods Guide](docs/generation-methods.md).

## Configuration

DoDHaluEval uses YAML configuration files for reproducible experiments with full generation method control:

### Generation Method Selection

```yaml
# configs/halueval_method.yaml
version: "0.1.0"
environment: "production"

response_generation:
  generation_method: "halueval"  # Options: dodhalueval, halueval, hybrid
  
  halueval_settings:
    use_two_stage_generation: true
    enable_filtering: true
    hallucination_patterns:
      - factual_contradiction
      - context_misunderstanding 
      - specificity_mismatch
      - invalid_inference
    max_knowledge_length: 2000
    
  providers: ["openai", "fireworks"]
  hallucination_rate: 0.3
```

### Hybrid Method Configuration

```yaml
# configs/hybrid_method.yaml
response_generation:
  generation_method: "hybrid"
  
  hybrid_settings:
    primary_method: "halueval"
    fallback_method: "dodhalueval"
    comparison_mode: false
    selection_criteria: "confidence_score"
    
  halueval_settings:
    use_two_stage_generation: true
    enable_filtering: true
    
  dodhalueval_settings:
    injection_strategies: ["factual", "logical", "context"]
    system_prompt_strategy: "hallucination_prone"
```

### Complete Configuration Example

```yaml
# configs/comprehensive_experiment.yaml
version: "0.1.0"
environment: "production"

# Processing settings
batch_size: 50
max_concurrent_requests: 10

# API configurations
api_configs:
  openai:
    provider: "openai"
    model: "gpt-4"
    max_retries: 3
  fireworks:
    provider: "fireworks"
    model: "llama-v3p1-70b-instruct"
    max_retries: 3

# Document processing
pdf_processing:
  chunk_size: 1000
  chunk_overlap: 200
  min_chunk_length: 100

# Prompt generation
prompt_generation:
  strategies: ["template", "llm_based"]
  max_prompts_per_document: 100
  hallucination_types: ["factual", "logical", "context"]

# Response generation with method selection
response_generation:
  generation_method: "halueval"
  providers: ["openai", "fireworks"]
  hallucination_rate: 0.3
  
  halueval_settings:
    use_two_stage_generation: true
    enable_filtering: true
    hallucination_patterns:
      - factual_contradiction
      - context_misunderstanding
      - specificity_mismatch
      - invalid_inference
      - equipment_substitution
      - branch_confusion
    max_knowledge_length: 2000

# Evaluation methods
evaluation_methods:
  - method: "vectara_hhem"
    enabled: true
    confidence_threshold: 0.5
  - method: "g_eval"
    enabled: true
  - method: "self_check_gpt"
    enabled: true
    
ensemble_evaluation:
  enabled: true
  consensus_method: "majority_vote"
```

Environment variables provide runtime configuration overrides:

```bash
export OPENAI_API_KEY="your-openai-key"
export FIREWORKS_API_KEY="your-fireworks-key"
export DODHALUEVAL_GENERATION_METHOD="halueval"
export DODHALUEVAL_BATCH_SIZE=100
```

## Generation Methods

DoDHaluEval supports three distinct hallucination generation methodologies:

### 1. DoDHaluEval Method (Original)
- **Approach**: Post-hoc hallucination injection
- **Strengths**: Domain-specific patterns, sophisticated system prompt manipulation
- **Use Case**: Military-specific hallucination patterns, existing pipeline compatibility

### 2. HaluEval Method (Research-Based)
- **Approach**: Direct generation following HaluEval paper (ArXiv:2305.11747)
- **Strengths**: Research validation, controlled generation patterns
- **Use Case**: Academic benchmarking, methodology comparison

### 3. Hybrid Method (Intelligent Combination)
- **Approach**: Combines both methods with fallback and comparison modes
- **Strengths**: Robustness, method validation, performance optimization
- **Use Case**: Production deployments, comprehensive evaluation

For detailed comparison and usage guidance, see [Generation Methods Guide](docs/generation-methods.md).

## Performance and Quality

### Validated Performance Metrics

- **Detection Accuracy**: 80% accuracy, 67% F1 score on mixed content with 35% hallucination injection rate
- **Generation Success Rate**: 100% knowledge context integration with semantic similarity-based extraction
- **Processing Throughput**: 20 prompts processed in approximately 2 minutes with complete evaluation pipeline
- **Method Coverage**: All three generation methods fully implemented and operational
- **Document Processing**: 50+ PDFs per hour with intelligent caching

### Quality Assurance

- **Multi-Method Validation**: All generation methods tested with real DoD documents
- **HaluEval Compliance**: Full compatibility with HaluEval benchmark format and evaluation metrics
- **Schema Validation**: Type-safe data models using Pydantic with comprehensive validation
- **Error Recovery**: Robust error handling with exponential backoff, circuit breaker patterns, and method fallback
- **Reproducibility**: Deterministic processing with cache control and configuration management

## Project Status

**Status: Production Ready - Multi-Method Implementation Complete**

The DoDHaluEval framework has successfully completed its implementation of multiple generation methodologies, delivering a sophisticated system that bridges academic research (HaluEval) with domain-specific military content processing.

### Completed Features

- **Complete HaluEval Integration**: Full implementation of HaluEval research methodology with DoD-specific enhancements
- **Multi-Method Architecture**: Three distinct generation approaches with seamless switching
- **Knowledge Context Building**: Semantic similarity-based context extraction with SentenceTransformers
- **Advanced Pipeline**: End-to-end processing with real document integration and chunk flow
- **Production Quality**: Comprehensive error handling, performance optimization, and deployment readiness

### Recently Resolved

- **Knowledge Context Pipeline**: Document chunks now properly flow through all generation methods
- **HaluEval Template System**: Complete template implementation with seven hallucination patterns
- **Configuration Management**: Full YAML-based configuration with method selection and validation
- **Async Integration**: Performance-optimized concurrent processing across all components

### Ready-to-Use Components

- **Multi-Method Pipeline**: `scripts/run_pipeline.py` with configurable generation methodology
- **Generation Method Configs**: Pre-configured YAML files for each approach
- **HaluEval Compatibility**: Full format conversion and benchmark compliance
- **Test Suite**: Comprehensive validation with real DoD document processing

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Quick Reference](docs/quick-reference.md)**: Essential commands and code snippets for common operations
- **[Installation Guide](docs/installation.md)**: Detailed setup instructions for all platforms
- **[Usage Guide](docs/usage.md)**: Comprehensive examples and advanced usage patterns
- **[Generation Methods Guide](docs/generation-methods.md)**: Detailed comparison and usage for all three methods
- **[API Reference](docs/api-reference.md)**: Complete API documentation for all public interfaces
- **[Architecture Documentation](docs/architecture.md)**: Detailed system architecture and component design
- **[HaluEval Integration Guide](docs/halueval-integration.md)**: Specific guidance for HaluEval compatibility and usage
- **[Configuration Reference](docs/configuration.md)**: Complete configuration options and examples

## Research Foundation

DoDHaluEval is built upon rigorous research foundations and established evaluation frameworks:

### Based on HaluEval Framework

This project implements and extends the research outcomes from the HaluEval paper (ArXiv:2305.11747), providing full compatibility with the HaluEval methodology while adding domain-specific enhancements for DoD knowledge evaluation.

### Multi-Method Innovation

- **Research Validation**: First implementation supporting both HaluEval research methodology and domain-specific approaches
- **DoD Knowledge Focus**: Comprehensive hallucination evaluation framework specifically designed for Department of Defense knowledge domains
- **Method Comparison**: Enables rigorous comparison between different hallucination generation approaches
- **Production Integration**: Enterprise-grade system design with performance optimization and error recovery

### Data Sources

The framework processes unclassified US Marine Corps Command and Staff College educational documents as source material, including Marine Corps Doctrinal Publications (MCDPs) and related educational materials. These documents provide authentic DoD knowledge for benchmark generation while maintaining appropriate security classifications.

## Contributing

Contributions are welcome! The project follows established development practices:

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run with coverage
pytest --cov=src/dodhalueval --cov-report=html
```

### Code Quality Standards

- **Linting**: flake8, black, isort with PEP 8 compliance
- **Type Checking**: mypy with strict mode and comprehensive type hints
- **Testing**: 90%+ coverage target with unit and integration tests
- **Documentation**: Google-style docstrings for all public interfaces

## Security and Compliance

DoDHaluEval is designed with security considerations appropriate for Department of Defense applications:

- **Unclassified Processing**: System processes only unclassified educational materials
- **API Key Security**: Secure API key management through environment variables
- **Data Handling**: No sensitive data persistence in logs or cache files
- **Audit Trail**: Comprehensive logging and traceability for all processing operations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use DoDHaluEval in your research, please cite:

```bibtex
@software{dodhalueval2025,
  title={DoDHaluEval: Department of Defense Hallucination Evaluation Framework},
  author={Norman, Justin D},
  year={2025},
  url={https://github.com/stbiadmin/diu-halu},
  version={2.0.0},
  note={Multi-method hallucination evaluation framework with HaluEval compatibility}
}
```

### Related Publications

DoDHaluEval builds upon and extends the following research:

```bibtex
@inproceedings{li2023halueval,
  title={HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models},
  author={Li, Junyi and Cheng, Xiaoxue and Zhao, Wayne Xin and Nie, Jian-Yun and Wen, Ji-Rong},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  year={2023},
  url={https://arxiv.org/abs/2305.11747},
  note={arXiv:2305.11747}
}

@article{huang2025survey,
  title={A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions},
  author={Huang, Lei and Yu, Weijiang and Ma, Weitao and Zhong, Weihong and Feng, Zhangyin and Wang, Haotian and Chen, Qianglong and Peng, Weihua and Feng, Xiaocheng and Qin, Bing and Liu, Ting},
  journal={ACM Transactions on Information Systems},
  volume={43},
  number={2},
  year={2025},
  publisher={ACM},
  doi={10.1145/3703155}
}

@article{norman2025language,
  title={Language models should be subject to repeatable, open, domain-contextualized hallucination benchmarking},
  author={Norman, Justin D and Rivera, Michael U and Hughes, D Alex},
  journal={arXiv preprint arXiv:2505.17345},
  year={2025},
  url={https://arxiv.org/abs/2505.17345}
}

@misc{bao2024hhem,
  title={{HHEM-2.1-Open: Hughes Hallucination Evaluation Model}},
  author={Bao, Forrest and Li, Miaoran and Luo, Rogger and Mendelevitch, Ofer},
  year={2024},
  url={https://huggingface.co/vectara/hallucination_evaluation_model},
  doi={10.57967/hf/3240},
  publisher={Hugging Face}
}
```

## Contact and Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/stbiadmin/diu-halu/issues)
- **Documentation**: [Project Documentation](docs/)
- **API Reference**: [API Documentation](docs/api-reference.md)

## Acknowledgments

- **HaluEval Research Team**: Foundation research and evaluation framework design (Li et al., 2023)
- **Marine Corps University**: US Marine Corps Command and Staff College educational documents used as source material
- **Research Contributors**: Norman, Rivera, and Hughes for foundational work on domain-contextualized hallucination benchmarking
- **Survey Research Team**: Huang et al. for comprehensive survey of hallucination principles and taxonomy
- **Vectara Research Team**: Hughes Hallucination Evaluation Model (HHEM) development and open-source release (Bao et al., 2024)
- **Open Source Community**: Contributing libraries and frameworks including HuggingFace, OpenAI, and PyTorch ecosystems

---

**DoDHaluEval** provides the first comprehensive, multi-method framework for evaluating hallucination detection in Department of Defense knowledge domains. The system combines rigorous research foundations with enterprise-grade implementation to enable reliable assessment of LLM performance in critical applications across multiple generation methodologies.