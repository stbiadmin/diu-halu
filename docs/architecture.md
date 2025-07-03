# DoDHaluEval Architecture

This document provides a comprehensive overview of the DoDHaluEval system architecture, component interactions, and design principles including the multi-method generation capabilities.

## System Overview

DoDHaluEval is a modular hallucination evaluation framework designed to create domain-specific benchmarks for Department of Defense knowledge areas. The system follows a pipeline architecture with four main processing stages, three distinct generation methodologies, and supporting infrastructure components.

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │    Prompt       │    │   Response      │    │   Evaluation    │
│  Processing     │───▶│  Generation     │───▶│  Generation     │───▶│ & Detection     │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Structured     │    │  Hallucination  │    │   Multi-Method  │    │   Benchmark     │
│   Knowledge     │    │ Prone Prompts   │    │   Responses     │    │    Dataset      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │ Method Selection│
                                              │ DoDHaluEval |   │
                                              │ HaluEval |      │
                                              │ Hybrid          │
                                              └─────────────────┘
```

## Core Components

### 1. Document Processing Pipeline

**Location**: `src/dodhalueval/data/`

The document processing pipeline extracts and structures knowledge from DoD PDF documents.

#### Components:
- **PDFProcessor**: Extracts text from PDF files using PyPDF2
- **TextChunker**: Splits documents into manageable, contextual chunks
- **DocumentCache**: Provides caching layer for processed documents
- **KnowledgeExtractor**: Identifies key concepts and entities

#### Data Flow:
```
PDF Documents → Text Extraction → Chunking → Metadata Enrichment → Structured Knowledge
```

#### Key Features:
- Preserves document structure (headings, sections, paragraphs)
- Configurable chunk size and overlap
- Intelligent paragraph-based splitting
- Metadata preservation (source, page numbers, sections)
- Caching for performance optimization

### 2. Prompt Generation Engine

**Location**: `src/dodhalueval/core/prompt_generator.py`

The prompt generation engine creates hallucination-prone prompts using multiple strategies.

#### Generation Strategies:

**Template-Based Generation**:
- 92 pre-built templates across military domains
- Categories: factual_recall, procedural, technical, tactical, regulatory
- Variable substitution from document content

**LLM-Based Generation**:
- Uses GPT-4 or Llama models for context-aware prompt creation
- Strategies: factual_probing, logical_reasoning, adversarial, contextual_confusion
- Adaptive generation based on document analysis

**Perturbation Engine**:
- 10 distinct perturbation strategies
- Entity substitution, numerical manipulation, temporal confusion
- Negation injection, authority confusion, causal reversal
- Multi-hop reasoning, scope expansion, conditional complexity

#### Quality Assurance:
- Rule-based validation for completeness and clarity
- LLM-based validation for domain relevance
- Scoring system (0.0-1.0) for prompt quality
- Filtering mechanisms for high-quality outputs

### 3. Multi-Method Response Generation System

**Location**: `src/dodhalueval/core/response_generator.py`

The response generation system supports three distinct hallucination generation methodologies with seamless switching and intelligent fallback capabilities.

#### Generation Methods:

**1. DoDHaluEval Method (Original)**:
- Post-hoc hallucination injection approach
- System prompt manipulation for hallucination-prone responses
- Domain-specific patterns: equipment substitution, branch confusion, temporal confusion
- Multi-stage pipeline: prompt → system prompt → response → post-processing injection

**2. HaluEval Method (Research-Based)**:
- Direct generation following HaluEval paper methodology (ArXiv:2305.11747)
- Structured instruction design with knowledge-question-answer format
- Seven hallucination patterns: factual contradiction, context misunderstanding, specificity mismatch, invalid inference, plus DoD-specific extensions
- Two-stage generation: one-pass + conversational schema with filtering

**3. Hybrid Method (Intelligent Combination)**:
- Combines both methods with configurable primary/fallback strategy
- Comparison mode: generates with both methods and selects best response
- Adaptive selection based on content type, confidence scores, or custom criteria
- Performance optimization with timeout and caching mechanisms

#### LLM Provider Support:
- **OpenAI**: GPT-4, GPT-3.5-turbo integration with full async support
- **Fireworks AI**: Llama-v3p1-70b-instruct, Mixtral models
- **Mock Provider**: Testing and development support with configurable behavior

#### Knowledge Context Building:
**Location**: `src/dodhalueval/core/knowledge_builder.py`
- Semantic similarity-based context extraction using SentenceTransformers
- Multi-chunk knowledge combination with relevance scoring
- Quality filtering and content truncation
- Fallback to text-based similarity when embeddings unavailable

#### HaluEval Generator:
**Location**: `src/dodhalueval/core/halueval_generator.py`
- Complete implementation of HaluEval research methodology
- Template-based prompt construction with pattern-specific demonstrations
- Two-stage generation with filtering for optimal hallucination selection
- Async and sync generation methods for pipeline compatibility

#### Features:
- Method selection through configuration with runtime switching
- Batch processing with configurable concurrency across all methods
- Rate limiting and retry logic with provider-specific parameters
- Response cleaning and normalization with method-specific processing
- Comprehensive metadata preservation and traceability
- Chunk-aware generation with document context integration

### 4. Hallucination Detection Framework

**Location**: `src/dodhalueval/core/hallucination_detector.py`

The detection framework employs multiple methods for comprehensive hallucination identification.

#### Detection Methods:

**HuggingFace HHEM**:
- Vectara hallucination evaluation model
- Binary classification with confidence scores
- Optimized for factual consistency

**G-Eval**:
- LLM-based evaluation using structured criteria
- Multi-dimensional scoring (consistency, fluency, relevance)
- Customizable evaluation prompts

**SelfCheckGPT**:
- Self-consistency checking through multiple sampling
- Statistical analysis of response variations
- Effective for detecting knowledge gaps

#### Ensemble Approach:
- Weighted voting across multiple detectors
- Configurable confidence thresholds
- Consensus scoring for final classifications
- Uncertainty quantification

### 5. Dataset Builder

**Location**: `src/dodhalueval/data/dataset_builder.py`

The dataset builder compiles evaluation results into standardized benchmark formats.

#### Output Formats:
- **HaluEval Compatible**: Direct compatibility with existing evaluation frameworks
- **JSONL**: Streaming JSON format for large datasets
- **JSON**: Structured format for smaller datasets
- **CSV**: Tabular format for analysis tools

#### Dataset Features:
- Train/validation/test splits with stratification
- Metadata preservation and enrichment
- Quality metrics and statistics
- Export validation and integrity checks

## Supporting Infrastructure

### Configuration Management

**Location**: `src/dodhalueval/utils/config.py`

Centralized configuration system using Pydantic models with environment variable support.

**Features**:
- Type-safe configuration with validation
- Environment-specific configurations
- Runtime configuration overrides
- Configuration file validation and generation

### Model Registry

**Location**: `src/dodhalueval/models/model_registry.py`

Centralized model discovery and validation system.

**Features**:
- Automatic model discovery across providers
- Model capability validation
- Performance benchmarking
- Provider health monitoring

### Logging and Monitoring

**Location**: `src/dodhalueval/utils/logger.py`

Structured logging system with configurable output formats.

**Features**:
- JSON structured logging
- Configurable log levels and destinations
- Performance metrics collection
- Error tracking and reporting

### Caching System

**Locations**: 
- `src/dodhalueval/data/pdf_processor.py` (PDF cache)
- `src/dodhalueval/providers/` (LLM response cache)

Multi-level caching for performance optimization.

**Features**:
- Document processing cache
- LLM response cache with TTL
- Configurable cache policies
- Cache validation and cleanup

## Data Models

### Core Schemas

**Location**: `src/dodhalueval/models/schemas.py`

Pydantic models for type-safe data handling throughout the pipeline.

#### Key Models:
- **Document**: PDF document with metadata and content chunks
- **DocumentChunk**: Text segment with position and context information
- **Prompt**: Generated question with metadata and difficulty rating
- **Response**: LLM response with generation parameters and metadata
- **EvaluationResult**: Hallucination detection result with confidence scores
- **BenchmarkDataset**: Complete dataset with statistics and export capabilities

### Validation Framework

**Location**: `src/dodhalueval/models/validators.py`

Comprehensive validation system for data integrity and quality assurance.

**Features**:
- Schema validation with custom rules
- Content quality assessment
- Domain-specific validation rules
- Validation reporting and suggestions

### HaluEval Compatibility Layer

**Location**: `src/dodhalueval/evaluation/halueval_compatibility.py`

Complete compatibility system for HaluEval benchmark format and academic research integration.

#### Key Components:

**Format Conversion**:
- Bidirectional conversion between DoDHaluEval and HaluEval formats
- Schema validation and compliance checking
- Metadata preservation and enrichment
- Export functionality for multiple academic formats

**Validation System**:
- HaluEval schema compliance verification
- Dataset quality metrics calculation
- Pattern distribution analysis
- Academic publication readiness assessment

**Research Integration**:
- Methodology validation against HaluEval paper specifications
- Performance benchmarking with original results
- Pattern extension for domain-specific evaluation
- Publication-ready dataset generation

**Features**:
- Complete HaluEval methodology implementation
- Academic benchmark compatibility
- Research validation and comparison tools
- Extended pattern support for military domains

## CLI Interface

**Location**: `src/dodhalueval/cli/commands.py`

Rich command-line interface for pipeline execution and management.

#### Available Commands:
- `process-docs`: Document processing and text extraction
- `generate-prompts`: Prompt generation with multiple strategies
- `generate-responses`: Response generation with hallucination injection
- `evaluate`: Hallucination detection and scoring
- `build-dataset`: Dataset compilation and export
- `validate-config`: Configuration validation and testing
- `info`: System information and model discovery

## Integration Patterns

### API Integration

The system provides multiple integration patterns for external APIs:

**Synchronous Integration**:
- Direct API calls with retry logic
- Rate limiting and error handling
- Response validation and normalization

**Asynchronous Integration**:
- Concurrent processing for improved throughput
- Batch processing capabilities
- Resource pool management

### Plugin Architecture

The system supports extensible plugin architecture for:

**Custom LLM Providers**:
- Standardized provider interface
- Automatic provider discovery
- Health monitoring and failover

**Custom Detection Methods**:
- Pluggable detection algorithms
- Standardized evaluation interface
- Performance benchmarking

### Data Pipeline Integration

**Input Sources**:
- Local PDF document collections
- Remote document repositories
- Streaming document feeds

**Output Destinations**:
- Local file system storage
- Cloud storage integration
- Database persistence
- Streaming data pipelines

## Performance Characteristics

### Throughput Metrics

- **Document Processing**: 50+ PDFs per hour
- **Prompt Generation**: 1000+ prompts per hour
- **Response Generation**: 100+ responses per hour (API dependent)
- **Hallucination Detection**: 500+ evaluations per hour

### Resource Requirements

- **Memory**: 8GB RAM for standard operations
- **Storage**: 1GB for caches, variable for datasets
- **Network**: API bandwidth dependent
- **CPU**: Multi-core recommended for parallel processing

### Scalability Considerations

- Horizontal scaling through batch processing
- Configurable concurrency limits
- Resource pooling and connection management
- Graceful degradation under load

## Security and Compliance

### Data Security

- API key management through environment variables
- No sensitive data in logs or cache files
- Secure temporary file handling
- Memory-safe data processing

### DoD Compliance

- Unclassified document processing only
- Audit trail preservation
- Reproducible processing pipelines
- Version control for all configurations

## Error Handling and Recovery

### Error Classification

- **Transient Errors**: Network issues, rate limits, temporary API failures
- **Permanent Errors**: Invalid configurations, missing dependencies, malformed data
- **Resource Errors**: Memory exhaustion, disk space, quota limits

### Recovery Strategies

- Exponential backoff for transient errors
- Circuit breaker pattern for API failures
- Graceful degradation with fallback modes
- Comprehensive error reporting and logging

## Testing Strategy

### Unit Testing

- Component isolation with mocking
- Comprehensive edge case coverage
- Property-based testing for data models
- Performance regression testing

### Integration Testing

- End-to-end pipeline validation
- API integration verification
- Cross-component compatibility testing
- Error injection and recovery testing

### System Testing

- Large-scale processing validation
- Performance benchmarking
- Resource usage monitoring
- Compliance verification

## Development Guidelines

### Code Quality Standards

- PEP 8 compliance with Black formatting
- Type hints for all public interfaces
- Comprehensive docstring documentation
- 90%+ test coverage target

### Contribution Workflow

- Feature development in isolated branches
- Comprehensive test coverage for new features
- Documentation updates for user-facing changes
- Performance impact assessment

### Release Management

- Semantic versioning for releases
- Changelog maintenance
- Backward compatibility preservation
- Migration guides for breaking changes

This architecture provides a robust, scalable foundation for DoD-specific hallucination evaluation while maintaining flexibility for future enhancements and integrations.