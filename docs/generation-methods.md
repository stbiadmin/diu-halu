# Generation Methods Guide

This guide provides comprehensive information about the three hallucination generation methodologies supported by DoDHaluEval, with complete configuration templates and usage examples for each method.

## Table of Contents

1. [Overview](#overview)
2. [Quick Comparison](#quick-comparison)
3. [DoDHaluEval Method](#1-dodhalueval-method)
4. [HaluEval Method](#2-halueval-method)
5. [Hybrid Method](#3-hybrid-method)
6. [Performance Considerations](#performance-considerations)
7. [Method Selection Guidelines](#method-selection-guidelines)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

## Overview

DoDHaluEval supports three distinct approaches to hallucination generation:

1. **DoDHaluEval Method**: Post-hoc injection with military domain expertise
2. **HaluEval Method**: Direct generation following academic research methodology
3. **Hybrid Method**: Intelligent combination with fallback and comparison capabilities

## Quick Comparison

| Feature | DoDHaluEval | HaluEval | Hybrid |
|---------|-------------|----------|--------|
| **Approach** | Post-hoc injection | Direct generation | Combined approach |
| **Best For** | Military/DoD content | Academic research | Production systems |
| **Speed** | Fast (1.2s/response) | Medium (2.8s/response) | Variable (1.2-4.0s) |
| **Configuration Complexity** | Medium | High | High |
| **Hallucination Control** | Pattern-based injection | Instruction-based | Both methods |
| **Research Validation** | Domain-specific | Academic standard | Comparative |

## 1. DoDHaluEval Method

### Overview

The DoDHaluEval method uses post-hoc hallucination injection, where responses are generated normally and then specific hallucination patterns are injected during post-processing.

### Key Features

- Post-processing injection of hallucinations
- Domain-specific patterns for military content
- System prompt manipulation strategies
- Configurable injection rates by pattern type

### Supported Hallucination Patterns

1. **Factual Hallucinations**
   - Equipment specification changes
   - Numerical alterations
   - Date/timeline modifications
   
2. **Logical Hallucinations**
   - Contradictory statements
   - Impossible scenarios
   - Circular reasoning
   
3. **Contextual Hallucinations**
   - Wrong domain information
   - Temporal confusion
   - Geographic misplacement

### Complete Configuration Template

```yaml
response_generation:
  generation_method: "dodhalueval"
  
  dodhalueval_settings:
    # Overall hallucination control
    hallucination_enabled: true
    hallucination_rate: 0.3  # 30% of responses will contain hallucinations
    
    # Injection strategies (all that apply)
    injection_strategies:
      - factual
      - logical
      - contextual
    
    # Pattern-specific injection rates
    injection_probabilities:
      factual: 0.4      # 40% chance when factual strategy selected
      logical: 0.3      # 30% chance when logical strategy selected
      contextual: 0.3   # 30% chance when contextual strategy selected
    
    # System prompt configuration
    system_prompt_strategy: "mixed"  # Options: conservative, hallucination_prone, mixed
    system_prompt_probabilities:
      conservative: 0.7
      hallucination_prone: 0.3
    
    # Post-processing options
    post_processing:
      enabled: true
      preserve_structure: true
      maintain_length: true
      
    # Domain-specific patterns
    military_patterns:
      equipment_swap: true
      branch_confusion: true
      rank_alteration: true
      acronym_modification: true
    
    # Advanced settings
    perturbation_settings:
      min_changes: 1
      max_changes: 3
      preserve_keywords: ["MAGTF", "MEF", "classified"]
```

### Usage Example

```python
from dodhalueval.core import ResponseGenerator
from dodhalueval.providers import OpenAIProvider
from dodhalueval.models.config import APIConfig

# Initialize provider
api_config = APIConfig(
    provider="openai",
    api_key="your-api-key",
    model="gpt-4",
    temperature=0.7
)
provider = OpenAIProvider(api_config)

# Complete configuration
config = {
    'generation_method': 'dodhalueval',
    'dodhalueval_settings': {
        'hallucination_enabled': True,
        'hallucination_rate': 0.3,
        'injection_strategies': ['factual', 'logical', 'contextual'],
        'injection_probabilities': {
            'factual': 0.4,
            'logical': 0.3,
            'contextual': 0.3
        },
        'system_prompt_strategy': 'mixed',
        'system_prompt_probabilities': {
            'conservative': 0.7,
            'hallucination_prone': 0.3
        },
        'post_processing': {
            'enabled': True,
            'preserve_structure': True,
            'maintain_length': True
        },
        'military_patterns': {
            'equipment_swap': True,
            'branch_confusion': True,
            'rank_alteration': True,
            'acronym_modification': True
        }
    }
}

# Initialize response generator
response_gen = ResponseGenerator(
    providers={'openai': provider},
    config=config
)

# Generate responses with hallucinations
prompts = [...]  # Your prompts
responses = await response_gen.generate_responses(
    prompts=prompts,
    provider_names=['openai'],
    include_metadata=True  # Get injection details
)

# Access response details
for response in responses:
    print(f"Text: {response.text}")
    print(f"Hallucinated: {response.hallucinated}")
    print(f"Injection type: {response.metadata.get('injection_type')}")
```

### CLI Usage

```bash
# Generate with DoDHaluEval method
dodhalueval generate \
    --method dodhalueval \
    --hallucination-rate 0.3 \
    --injection-strategies factual logical contextual \
    --input prompts.json \
    --output responses.json
```

## 2. HaluEval Method

### Overview

The HaluEval method implements the methodology from the HaluEval research paper, using direct generation where the LLM produces hallucinated responses given structured context.

### Key Features

- Direct hallucination generation
- Knowledge-Question-Answer format
- Academic research compliance
- Two-stage generation with filtering
- Semantic similarity for context selection

### Supported Hallucination Patterns

1. **Core HaluEval Patterns**
   - Comprehension: Misunderstanding the question
   - Factuality: Contradicting provided facts
   - Specificity: Wrong level of detail
   - Inference: Invalid logical conclusions

2. **DoD Extensions**
   - Equipment substitution
   - Branch confusion
   - Temporal confusion
   - Classification errors

### Complete Configuration Template

```yaml
response_generation:
  generation_method: "halueval"
  
  halueval_settings:
    # Generation approach
    generation_schema: "one_pass"  # Options: one_pass, conversational
    use_two_stage_generation: true
    enable_filtering: true
    
    # Hallucination patterns to use
    hallucination_patterns:
      # Core patterns
      - comprehension
      - factuality
      - specificity
      - inference
      # DoD extensions
      - equipment_substitution
      - branch_confusion
      - temporal_confusion
      - classification_error
    
    # Pattern selection
    pattern_selection:
      mode: "weighted"  # Options: random, weighted, sequential
      weights:
        comprehension: 0.2
        factuality: 0.3
        specificity: 0.2
        inference: 0.3
    
    # Knowledge context building
    knowledge_context:
      max_length: 2000
      min_length: 100
      semantic_similarity_threshold: 0.3
      use_chunk_overlap: true
      chunk_selection_method: "top_k"  # Options: top_k, threshold, hybrid
      top_k_chunks: 3
    
    # Instruction template configuration
    instruction_template:
      include_demonstrations: true
      num_demonstrations: 2
      demonstration_selection: "pattern_specific"
      
    # Generation parameters
    generation_params:
      temperature: 0.7
      max_tokens: 500
      frequency_penalty: 0.3
      presence_penalty: 0.3
      
    # Filtering and validation
    filtering:
      temperature: 0.3
      reject_threshold: 0.8
      max_retries: 3
      validation_prompt: "default"  # Options: default, strict, lenient
    
    # Advanced settings
    advanced:
      cache_instructions: true
      batch_size: 10
      parallel_generation: true
      track_pattern_distribution: true
```

### Usage Example

```python
from dodhalueval.core import ResponseGenerator
from dodhalueval.core.knowledge_builder import KnowledgeContextBuilder
from dodhalueval.data import DocumentChunk
from dodhalueval.providers import OpenAIProvider

# Initialize provider and knowledge builder
provider = OpenAIProvider(api_config)
knowledge_builder = KnowledgeContextBuilder({
    'max_length': 2000,
    'semantic_similarity_threshold': 0.3,
    'chunk_selection_method': 'top_k',
    'top_k_chunks': 3
})

# Complete configuration
config = {
    'generation_method': 'halueval',
    'halueval_settings': {
        'generation_schema': 'one_pass',
        'use_two_stage_generation': True,
        'enable_filtering': True,
        'hallucination_patterns': [
            'comprehension', 'factuality', 
            'specificity', 'inference',
            'equipment_substitution'
        ],
        'pattern_selection': {
            'mode': 'weighted',
            'weights': {
                'comprehension': 0.2,
                'factuality': 0.3,
                'specificity': 0.2,
                'inference': 0.3
            }
        },
        'knowledge_context': {
            'max_length': 2000,
            'semantic_similarity_threshold': 0.3,
            'chunk_selection_method': 'top_k',
            'top_k_chunks': 3
        },
        'generation_params': {
            'temperature': 0.7,
            'max_tokens': 500
        },
        'filtering': {
            'temperature': 0.3,
            'reject_threshold': 0.8,
            'max_retries': 3
        }
    }
}

# Initialize response generator
response_gen = ResponseGenerator(
    providers={'openai': provider},
    config=config
)

# Prepare document chunks
chunks = [
    DocumentChunk(
        text="Marine Expeditionary Forces (MEF) are the principal warfighting...",
        metadata={"source": "MCDP 1-0", "page": 15}
    ),
    # ... more chunks
]

# Generate with HaluEval method
responses = await response_gen.generate_responses(
    prompts=prompts,
    provider_names=['openai'],
    chunks=chunks,  # Required for knowledge context
    include_metadata=True
)

# Access detailed information
for response in responses:
    print(f"Text: {response.text}")
    print(f"Pattern used: {response.metadata.get('hallucination_pattern')}")
    print(f"Knowledge context: {response.metadata.get('knowledge_used')}")
    print(f"Generation attempts: {response.metadata.get('generation_attempts', 1)}")
```

### CLI Usage

```bash
# Generate with HaluEval method
dodhalueval generate \
    --method halueval \
    --patterns comprehension factuality specificity inference \
    --knowledge-source documents/ \
    --two-stage-generation \
    --enable-filtering \
    --input prompts.json \
    --output responses.json
```

## 3. Hybrid Method

### Overview

The Hybrid method intelligently combines both DoDHaluEval and HaluEval approaches, providing fallback capabilities, comparison modes, and adaptive selection.

### Key Features

- Automatic method fallback
- Dual generation and comparison
- Configurable selection criteria
- Performance optimization
- Method-specific overrides

### Operation Modes

1. **Fallback Mode**: Use primary method, fall back to secondary on failure
2. **Comparison Mode**: Generate with both methods, select best
3. **Adaptive Mode**: Choose method based on content analysis
4. **Sequential Mode**: Apply methods in sequence

### Complete Configuration Template

```yaml
response_generation:
  generation_method: "hybrid"
  
  hybrid_settings:
    # Method configuration
    primary_method: "halueval"
    secondary_method: "dodhalueval"
    
    # Operation mode
    mode: "comparison"  # Options: fallback, comparison, adaptive, sequential
    
    # Fallback configuration
    fallback:
      enabled: true
      triggers:
        - error
        - low_confidence
        - empty_response
        - timeout
      confidence_threshold: 0.3
      timeout_seconds: 30
    
    # Comparison configuration
    comparison:
      generate_both: true
      selection_criteria: "weighted_score"  # Options: confidence, length, weighted_score, custom
      scoring_weights:
        confidence: 0.4
        length_appropriateness: 0.2
        pattern_quality: 0.2
        domain_relevance: 0.2
      prefer_primary: true
      primary_bonus: 0.1
    
    # Adaptive configuration
    adaptive:
      content_analysis: true
      method_mapping:
        technical_manual: "dodhalueval"
        procedural_text: "halueval"
        mixed_content: "comparison"
      confidence_required: 0.7
    
    # Sequential configuration
    sequential:
      order: ["halueval", "dodhalueval"]
      combination_strategy: "merge"  # Options: merge, overlay, selective
      preserve_best_features: true
    
    # Method-specific overrides
    method_overrides:
      halueval:
        generation_params:
          temperature: 0.8
        filtering:
          reject_threshold: 0.7
      dodhalueval:
        hallucination_rate: 0.4
        injection_strategies: ["factual", "logical"]
    
    # Performance optimization
    optimization:
      cache_results: true
      parallel_generation: true
      early_termination: true
      max_total_time: 60
    
    # Logging and debugging
    debug:
      log_method_selection: true
      log_comparison_scores: true
      save_all_candidates: false
      
  # Full settings for both methods
  halueval_settings:
    # ... (full halueval configuration as above)
    
  dodhalueval_settings:
    # ... (full dodhalueval configuration as above)
```

### Usage Example

```python
from dodhalueval.core import ResponseGenerator
from dodhalueval.providers import OpenAIProvider, FireworksProvider

# Initialize multiple providers
openai_provider = OpenAIProvider(openai_config)
fireworks_provider = FireworksProvider(fireworks_config)

# Complete hybrid configuration
config = {
    'generation_method': 'hybrid',
    'hybrid_settings': {
        'primary_method': 'halueval',
        'secondary_method': 'dodhalueval',
        'mode': 'comparison',
        'comparison': {
            'generate_both': True,
            'selection_criteria': 'weighted_score',
            'scoring_weights': {
                'confidence': 0.4,
                'length_appropriateness': 0.2,
                'pattern_quality': 0.2,
                'domain_relevance': 0.2
            }
        },
        'fallback': {
            'enabled': True,
            'triggers': ['error', 'low_confidence'],
            'confidence_threshold': 0.3
        },
        'optimization': {
            'parallel_generation': True,
            'cache_results': True
        },
        'debug': {
            'log_method_selection': True,
            'log_comparison_scores': True
        }
    },
    # Include full configurations for both methods
    'halueval_settings': {
        'use_two_stage_generation': True,
        'enable_filtering': True,
        'hallucination_patterns': ['comprehension', 'factuality'],
        # ... rest of halueval config
    },
    'dodhalueval_settings': {
        'hallucination_rate': 0.3,
        'injection_strategies': ['factual', 'logical'],
        # ... rest of dodhalueval config
    }
}

# Initialize response generator with multiple providers
response_gen = ResponseGenerator(
    providers={
        'openai': openai_provider,
        'fireworks': fireworks_provider
    },
    config=config
)

# Generate with hybrid method
responses = await response_gen.generate_responses(
    prompts=prompts,
    provider_names=['openai'],  # Can use different providers for each method
    chunks=chunks,  # Required if using halueval
    include_metadata=True
)

# Access comprehensive metadata
for response in responses:
    print(f"Text: {response.text}")
    print(f"Final method: {response.metadata.get('selected_method')}")
    print(f"Selection reason: {response.metadata.get('selection_reason')}")
    
    if response.metadata.get('comparison_performed'):
        scores = response.metadata.get('method_scores', {})
        print(f"HaluEval score: {scores.get('halueval', 'N/A')}")
        print(f"DoDHaluEval score: {scores.get('dodhalueval', 'N/A')}")
    
    if response.metadata.get('fallback_used'):
        print(f"Fallback reason: {response.metadata.get('fallback_reason')}")
```

### Advanced Usage: Custom Selection Function

```python
# Define custom selection criteria
def custom_selection_function(responses, metadata):
    """Select response based on domain-specific criteria."""
    halueval_response = responses.get('halueval')
    dodhalueval_response = responses.get('dodhalueval')
    
    # Check for military keywords
    military_keywords = ['MAGTF', 'MEF', 'tactical', 'operational']
    
    halueval_keywords = sum(1 for kw in military_keywords 
                           if kw in halueval_response.text)
    dodhalueval_keywords = sum(1 for kw in military_keywords 
                              if kw in dodhalueval_response.text)
    
    # Prefer response with more domain keywords
    if dodhalueval_keywords > halueval_keywords:
        return 'dodhalueval', {'reason': 'higher_domain_relevance'}
    elif halueval_keywords > dodhalueval_keywords:
        return 'halueval', {'reason': 'higher_domain_relevance'}
    else:
        # Fall back to confidence scores
        return 'halueval', {'reason': 'equal_relevance_default'}

# Use custom selection
config['hybrid_settings']['comparison']['selection_function'] = custom_selection_function
```

### CLI Usage

```bash
# Generate with hybrid method in comparison mode
dodhalueval generate \
    --method hybrid \
    --mode comparison \
    --primary halueval \
    --secondary dodhalueval \
    --selection-criteria weighted_score \
    --enable-fallback \
    --input prompts.json \
    --output responses.json \
    --save-comparison-data
```

## Performance Considerations

### Response Time Comparison

| Method | Mode | Avg. Time | Token Usage | Quality Score |
|--------|------|-----------|-------------|---------------|
| DoDHaluEval | Standard | 1.2s | ~800 tokens | 85% |
| HaluEval | One-pass | 2.3s | ~1500 tokens | 82% |
| HaluEval | Two-stage | 3.5s | ~2200 tokens | 88% |
| Hybrid | Fallback | 1.2-3.5s | Variable | 86% |
| Hybrid | Comparison | 4.5s | ~3000 tokens | 90% |

### Resource Optimization Tips

1. **Batch Processing**
   ```python
   # Process in batches for efficiency
   responses = await response_gen.generate_responses_batch(
       prompts_list=prompts,
       batch_size=20,
       provider_names=['openai']
   )
   ```

2. **Caching**
   ```python
   # Enable caching for repeated content
   config['cache_settings'] = {
       'enabled': True,
       'ttl_seconds': 3600,
       'max_entries': 1000
   }
   ```

3. **Provider Selection**
   ```python
   # Use appropriate providers for each method
   providers = {
       'openai': openai_provider,      # For HaluEval
       'fireworks': fireworks_provider  # For DoDHaluEval
   }
   ```

## Method Selection Guidelines

### Decision Matrix

| Criteria | Recommended Method |
|----------|--------------------|
| Military/DoD content exclusively | DoDHaluEval |
| Academic research compliance | HaluEval |
| Production system reliability | Hybrid (Fallback) |
| Quality comparison needed | Hybrid (Comparison) |
| Mixed content types | Hybrid (Adaptive) |
| Speed critical | DoDHaluEval |
| Highest quality required | Hybrid (Comparison) |

### Quick Selection Guide

```python
def select_method(requirements):
    """Quick method selection based on requirements."""
    if requirements['domain'] == 'military' and requirements['speed'] == 'critical':
        return 'dodhalueval'
    elif requirements['validation'] == 'academic':
        return 'halueval'
    elif requirements['reliability'] == 'critical':
        return 'hybrid'
    else:
        # Default to hybrid for flexibility
        return 'hybrid'
```

## Advanced Usage

### Custom Hallucination Patterns

```python
# Define custom pattern for HaluEval
custom_pattern = {
    'name': 'acronym_confusion',
    'instruction': """
    You are trying to answer a question but you confuse military acronyms 
    with similar-sounding ones from different contexts or branches.
    """,
    'demonstrations': [
        {
            'knowledge': 'MAGTF stands for Marine Air-Ground Task Force',
            'question': 'What does MAGTF stand for?',
            'correct': 'Marine Air-Ground Task Force',
            'hallucinated': 'Mobile Air-Ground Tactical Formation'
        }
    ]
}

# Add to configuration
config['halueval_settings']['custom_patterns'] = [custom_pattern]
```

### Pipeline Integration

```python
from dodhalueval.pipeline import HallucinationPipeline

# Create full pipeline with generation method
pipeline = HallucinationPipeline(
    generation_method='hybrid',
    generation_config=config,
    evaluation_methods=['hhem', 'selfcheck', 'geval']
)

# Run end-to-end
results = await pipeline.run(
    documents=documents,
    num_prompts=100,
    hallucination_rate=0.3
)
```

### Monitoring and Metrics

```python
from dodhalueval.utils.metrics import GenerationMetrics

# Track generation metrics
metrics = GenerationMetrics()

# Generate with metric tracking
responses = await response_gen.generate_responses(
    prompts=prompts,
    provider_names=['openai'],
    metrics_collector=metrics
)

# Get performance report
report = metrics.get_report()
print(f"Average generation time: {report['avg_time']}")
print(f"Method distribution: {report['method_distribution']}")
print(f"Fallback rate: {report['fallback_rate']}")
```

## Troubleshooting

### Common Issues and Solutions

1. **Empty Responses**
   ```python
   # Check configuration
   if not response.text:
       print(f"Empty response - Method: {response.metadata.get('method')}")
       print(f"Error: {response.metadata.get('error')}")
   ```

2. **Low Hallucination Quality**
   - Increase temperature in generation parameters
   - Check pattern selection weights
   - Verify knowledge context quality

3. **Slow Generation**
   - Enable parallel processing
   - Use caching for repeated content
   - Consider using DoDHaluEval for speed

4. **Method Selection Issues**
   ```python
   # Enable debug logging
   import logging
   logging.getLogger('dodhalueval.core.response_generator').setLevel(logging.DEBUG)
   ```

### Configuration Validation

```python
from dodhalueval.utils.config import validate_generation_config

# Validate configuration before use
is_valid, errors = validate_generation_config(config)
if not is_valid:
    print(f"Configuration errors: {errors}")
```

### Debug Output

Enable comprehensive debugging:

```yaml
response_generation:
  debug_settings:
    log_level: "DEBUG"
    save_intermediate_results: true
    trace_method_selection: true
    log_prompts: true
    log_responses: true
    save_path: "./debug_output/"
```

## Conclusion

This guide provides complete configuration templates and usage examples for all three generation methods in DoDHaluEval. Choose the appropriate method based on your:

- Content domain (military vs. general)
- Quality requirements
- Performance constraints
- Research compliance needs

For production systems, we recommend starting with the Hybrid method in fallback mode, then optimizing based on your specific requirements and performance metrics.