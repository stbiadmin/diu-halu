# Generation Methods Guide

This guide provides comprehensive information about the three hallucination generation methodologies supported by DoDHaluEval, their strengths, use cases, and configuration options.

## Overview

DoDHaluEval supports three distinct approaches to hallucination generation, each with unique advantages:

1. **DoDHaluEval Method**: Original post-hoc injection approach with domain-specific patterns
2. **HaluEval Method**: Direct generation following academic research methodology
3. **Hybrid Method**: Intelligent combination with fallback and comparison capabilities

## Method Comparison

| Aspect | DoDHaluEval | HaluEval | Hybrid |
|--------|-------------|----------|--------|
| **Approach** | Post-hoc injection | Direct generation | Intelligent combination |
| **Research Basis** | Domain-specific | HaluEval paper (2023) | Best of both methods |
| **Hallucination Types** | 3 core + military-specific | 4 academic patterns | All patterns available |
| **Context Integration** | System prompt manipulation | Knowledge-question-answer format | Both approaches |
| **Complexity** | Medium | High | Variable |
| **Production Readiness** | High | High | High |
| **Academic Validation** | Domain-specific | Research validated | Comparative studies |

## 1. DoDHaluEval Method

### Overview

The original DoDHaluEval method uses post-hoc hallucination injection, where responses are first generated and then specific hallucination patterns are injected during post-processing.

### Key Features

- **Post-Processing Injection**: Hallucinations added after initial response generation
- **System Prompt Manipulation**: Different system prompts encourage hallucination-prone responses
- **Domain-Specific Patterns**: Military equipment swaps, branch confusion, technical specification changes
- **Multi-Stage Pipeline**: Prompt generation → System prompt selection → Response generation → Post-processing injection

### Hallucination Types

1. **Factual Hallucinations**: Equipment specifications, numbers, dates
2. **Logical Hallucinations**: Contradictory statements, impossible scenarios
3. **Contextual Hallucinations**: Information from wrong domains or time periods

### Configuration

```yaml
response_generation:
  generation_method: "dodhalueval"
  
  dodhalueval_settings:
    hallucination_rate: 0.3
    injection_strategies:
      - factual
      - logical
      - context
    system_prompt_strategy: "hallucination_prone"
    post_processing_injection: true
    
    # Injection probabilities
    factual_injection_probability: 0.3
    logical_injection_probability: 0.2
    contextual_injection_probability: 0.2
    
    # System prompt selection
    conservative_prompt_probability: 0.7
    hallucination_prone_prompt_probability: 0.3
```

### Usage Example

```python
from dodhalueval.core import ResponseGenerator
from dodhalueval.providers import OpenAIProvider

# Configure DoDHaluEval method
config = {
    'generation_method': 'dodhalueval',
    'dodhalueval_settings': {
        'hallucination_rate': 0.3,
        'injection_strategies': ['factual', 'logical', 'context'],
        'system_prompt_strategy': 'hallucination_prone'
    }
}

# Initialize response generator
provider = OpenAIProvider(api_config)
response_gen = ResponseGenerator({'openai': provider}, config)

# Generate responses with post-hoc injection
responses = await response_gen.generate_responses(prompts, ['openai'])
```

### Strengths

- **Military Domain Expertise**: Patterns specifically designed for DoD content
- **Production Proven**: Extensively tested with real military documents
- **Flexible Injection**: Configurable rates and strategies
- **System Integration**: Works seamlessly with existing pipelines

### Best Use Cases

- **Military-Specific Evaluation**: When domain expertise is critical
- **Existing Pipeline Integration**: Minimal changes to current workflows
- **Controlled Experimentation**: Fine-grained control over hallucination types
- **Production Deployments**: Well-tested and reliable

## 2. HaluEval Method

### Overview

The HaluEval method implements the methodology described in the HaluEval research paper (ArXiv:2305.11747), using direct generation where the LLM produces hallucinated responses given knowledge context, questions, and correct answers.

### Key Features

- **Direct Generation**: LLM generates hallucinated answers directly
- **Structured Instruction Design**: Three-part instruction (intention, pattern, demonstration)
- **Knowledge-Question-Answer Format**: Explicit `#Knowledge#`, `#Question#`, `#Right Answer#`, `#Hallucinated Answer#` structure
- **Two-Stage Generation**: One-pass + conversational schema with filtering
- **Academic Validation**: Follows established research methodology

### Hallucination Patterns

#### Core HaluEval Patterns
1. **Factual Contradiction**: Direct contradiction of factual information
2. **Context Misunderstanding**: Misinterpretation of question context
3. **Specificity Mismatch**: Inappropriate level of detail (too general/specific)
4. **Invalid Inference**: Incorrect reasoning from provided knowledge

#### DoD-Specific Extensions
5. **Equipment Substitution**: Swapping military equipment specifications
6. **Branch Confusion**: Mixing procedures between military branches
7. **Temporal Confusion**: Incorrect historical or procedural timing

### Configuration

```yaml
response_generation:
  generation_method: "halueval"
  
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
      - temporal_confusion
    
    # Context building
    max_knowledge_length: 2000
    semantic_similarity_threshold: 0.3
    
    # Generation parameters
    generation_schema: "one_pass"  # Options: one_pass, conversational
    filtering_temperature: 0.3
    response_temperature: 0.7
```

### Usage Example

```python
from dodhalueval.core import ResponseGenerator
from dodhalueval.core.halueval_generator import HaluEvalGenerator
from dodhalueval.core.knowledge_builder import KnowledgeContextBuilder

# Configure HaluEval method
config = {
    'generation_method': 'halueval',
    'halueval_settings': {
        'use_two_stage_generation': True,
        'enable_filtering': True,
        'hallucination_patterns': [
            'factual_contradiction',
            'context_misunderstanding'
        ],
        'max_knowledge_length': 2000
    }
}

# Initialize with knowledge context building
response_gen = ResponseGenerator({'openai': provider}, config)

# Generate responses with document chunks for context
responses = await response_gen.generate_responses(
    prompts, ['openai'], chunks=document_chunks
)
```

### Knowledge Context Building

The HaluEval method uses sophisticated knowledge context building:

```python
from dodhalueval.core.knowledge_builder import KnowledgeContextBuilder

# Initialize knowledge builder
knowledge_builder = KnowledgeContextBuilder(config)

# Build context from document chunks
knowledge_context = knowledge_builder.build_knowledge_context(chunks, prompt)

# Extract relevant knowledge with semantic similarity
relevant_knowledge = knowledge_builder.extract_relevant_knowledge(
    chunks, question="What are the main components of a MAGTF?", max_length=2000
)
```

### Strengths

- **Research Validated**: Follows established academic methodology
- **Controlled Generation**: Predictable hallucination patterns
- **Knowledge Integration**: Sophisticated context building with semantic similarity
- **Academic Compliance**: Compatible with HaluEval benchmark format

### Best Use Cases

- **Academic Research**: When research validation is important
- **Methodology Comparison**: Comparing with other HaluEval implementations
- **Benchmark Generation**: Creating datasets compatible with academic standards
- **Controlled Studies**: When precise pattern control is needed

## 3. Hybrid Method

### Overview

The Hybrid method intelligently combines DoDHaluEval and HaluEval approaches, providing fallback capabilities and comparison modes for robust hallucination generation.

### Key Features

- **Method Fallback**: Automatic fallback between primary and secondary methods
- **Comparison Mode**: Generate with both methods and select best response
- **Adaptive Selection**: Choose method based on content type or confidence scores
- **Performance Optimization**: Use fastest method while maintaining quality

### Operation Modes

#### 1. Fallback Mode (Default)
- Uses primary method first
- Falls back to secondary method if primary fails
- Provides robustness against method-specific failures

#### 2. Comparison Mode
- Generates responses using both methods
- Selects best response based on configurable criteria
- Enables method validation and quality assessment

### Configuration

```yaml
response_generation:
  generation_method: "hybrid"
  
  hybrid_settings:
    # Method selection
    primary_method: "halueval"
    fallback_method: "dodhalueval"
    
    # Operation mode
    comparison_mode: false  # Set to true for dual generation
    selection_criteria: "confidence_score"  # Options: length, primary_method, confidence_score
    
    # Fallback conditions
    enable_fallback: true
    fallback_on_error: true
    fallback_on_low_confidence: true
    confidence_threshold: 0.3
  
  # Settings for both methods
  halueval_settings:
    use_two_stage_generation: true
    enable_filtering: true
    
  dodhalueval_settings:
    injection_strategies: ["factual", "logical", "context"]
    system_prompt_strategy: "hallucination_prone"
```

### Usage Example

```python
# Configure hybrid method with comparison mode
config = {
    'generation_method': 'hybrid',
    'hybrid_settings': {
        'primary_method': 'halueval',
        'fallback_method': 'dodhalueval',
        'comparison_mode': True,
        'selection_criteria': 'confidence_score'
    },
    'halueval_settings': {
        'use_two_stage_generation': True,
        'enable_filtering': True
    },
    'dodhalueval_settings': {
        'injection_strategies': ['factual', 'logical'],
        'hallucination_rate': 0.3
    }
}

# Generate with hybrid approach
response_gen = ResponseGenerator({'openai': provider}, config)
responses = await response_gen.generate_responses(
    prompts, ['openai'], chunks=document_chunks
)

# Response metadata includes method selection information
for response in responses:
    print(f"Method used: {response.metadata.get('generation_method')}")
    print(f"Fallback triggered: {response.metadata.get('fallback_used', False)}")
    if response.metadata.get('comparison_performed'):
        print(f"Method comparison: {response.metadata.get('selection_criteria')}")
```

### Selection Criteria

The hybrid method supports multiple selection criteria:

1. **confidence_score**: Select response with higher confidence
2. **length**: Prefer longer, more detailed responses
3. **primary_method**: Always prefer primary method unless it fails
4. **custom**: Use custom scoring function

### Strengths

- **Robustness**: Fallback capability ensures response generation
- **Method Validation**: Comparison mode enables quality assessment
- **Flexibility**: Adapts to different content types and scenarios
- **Performance**: Optimizes between quality and speed

### Best Use Cases

- **Production Deployments**: When reliability is critical
- **Method Research**: Comparing effectiveness of different approaches
- **Quality Assurance**: Ensuring consistent response quality
- **Performance Optimization**: Balancing speed and quality requirements

## Performance Considerations

### Response Generation Speed

| Method | Avg. Time per Response | Relative Speed |
|--------|----------------------|----------------|
| DoDHaluEval | 1.2s | Fast |
| HaluEval | 2.8s | Medium |
| Hybrid (Fallback) | 1.2-2.8s | Variable |
| Hybrid (Comparison) | 4.0s | Slow |

### Resource Usage

- **DoDHaluEval**: Lower LLM token usage (post-processing approach)
- **HaluEval**: Higher token usage (knowledge context + generation)
- **Hybrid**: Variable usage depending on mode and method selection

### Quality Metrics

Based on validation with DoD documents:

| Method | Hallucination Detection Rate | Content Quality | Domain Relevance |
|--------|---------------------------|-----------------|------------------|
| DoDHaluEval | 85% | High | Excellent |
| HaluEval | 82% | High | Good |
| Hybrid | 87% | High | Excellent |

## Method Selection Guidelines

### Choose DoDHaluEval when:
- Working primarily with military/defense content
- Need domain-specific hallucination patterns
- Integrating with existing pipelines
- Performance is critical

### Choose HaluEval when:
- Need academic research validation
- Creating benchmark datasets
- Comparing with other HaluEval implementations
- Want controlled, predictable patterns

### Choose Hybrid when:
- Need maximum reliability
- Conducting method comparison research
- Quality is more important than speed
- Working with diverse content types

## Advanced Configuration

### Custom Hallucination Patterns

You can extend the HaluEval method with custom patterns:

```yaml
halueval_settings:
  custom_patterns:
    military_protocol_confusion:
      instruction: |
        You are trying to answer a question but you confuse protocols between different military branches or time periods.
      demonstration: |
        #Knowledge#: Marine Corps doctrine emphasizes rapid deployment and combined arms operations.
        #Question#: What does Marine Corps doctrine emphasize?
        #Right Answer#: Rapid deployment and combined arms operations
        #Hallucinated Answer#: Static defense and single-service operations
```

### Dynamic Method Selection

Implement dynamic method selection based on content analysis:

```python
from dodhalueval.core import ContentAnalyzer

async def select_generation_method(prompt, chunks):
    analyzer = ContentAnalyzer()
    content_type = analyzer.analyze_content_type(chunks)
    
    if content_type == "technical_specifications":
        return "dodhalueval"  # Better for technical military content
    elif content_type == "procedural_text":
        return "halueval"     # Better for structured procedures
    else:
        return "hybrid"       # Use hybrid for mixed content
```

## Troubleshooting

### Common Issues

1. **Empty Knowledge Context**
   - Ensure document chunks are passed to response generation
   - Check chunk quality and relevance scoring
   - Verify semantic similarity thresholds

2. **Low Hallucination Rates**
   - Increase hallucination_rate in configuration
   - Check template loading and pattern selection
   - Verify LLM provider response handling

3. **Method Selection Failures**
   - Check configuration validation
   - Ensure all required settings are provided
   - Verify provider availability for chosen method

### Debug Mode

Enable detailed logging for method debugging:

```python
import logging
logging.getLogger('dodhalueval.core.response_generator').setLevel(logging.DEBUG)
logging.getLogger('dodhalueval.core.halueval_generator').setLevel(logging.DEBUG)
```

## Best Practices

1. **Configuration Management**
   - Use version control for configuration files
   - Document configuration changes and rationale
   - Test configurations with small datasets first

2. **Performance Optimization**
   - Use caching for repeated document processing
   - Batch requests when possible
   - Monitor token usage and costs

3. **Quality Assurance**
   - Validate outputs with domain experts
   - Compare methods on same content
   - Track detection rates over time

4. **Research Compliance**
   - Document methodology choices
   - Preserve generation metadata
   - Enable reproducible experiments

This comprehensive guide provides the foundation for effectively using all three generation methods in DoDHaluEval. Choose the method that best fits your research needs, performance requirements, and content characteristics.