# How to Use DoDHaluEval Prompt Generation

This guide shows you how to test and use the different prompt generation approaches with your CSC data using the consolidated test script.

## 🚀 Quick Start

### 1. Environment Setup
The API keys are already configured in `.env` file in the project root:
```bash
# Keys are pre-configured in .env:
# OPENAI_API_KEY
# FIREWORKS_API_KEY
# HUGGINGFACE_TOKEN
```

### 2. Test Options (run from project root)

**Quick Test (Recommended First Step)**
```bash
python scripts/test_prompt_generation.py --mode quick
```

**Full Demo with Rich Content**
```bash
python scripts/test_prompt_generation.py --mode demo --verbose
```

**Test with Your CSC Data**
```bash
python scripts/test_prompt_generation.py --mode csc --max-prompts 20 --verbose
```

**Test with Real LLM Providers**
```bash
python scripts/test_prompt_generation.py --mode llm --provider openai --verbose
python scripts/test_prompt_generation.py --mode llm --provider fireworks
```

**Comprehensive Test (All Modes)**
```bash
python scripts/test_prompt_generation.py --mode all --provider openai --validate --verbose
```

### 3. Command Options

| Option | Values | Description |
|--------|--------|-------------|
| `--mode` | quick, demo, csc, llm, all | Test mode to run |
| `--provider` | openai, fireworks, mock | LLM provider to use |
| `--max-prompts` | integer | Maximum prompts to generate |
| `--validate` | flag | Run prompt validation |
| `--verbose` | flag | Enable detailed output |

## 📋 Available Approaches

### 1. Template-Based Generation
- **92 pre-built templates** covering military domains
- Categories: factual_recall, question_answer, procedural, technical, tactical, etc.
- **Best for**: Consistent, structured prompts
- **Example**: "What are the specifications of {equipment}?"

### 2. LLM-Based Generation  
- Uses GPT-4, Llama-2, or other LLMs to create sophisticated prompts
- Strategies: factual_probing, logical_reasoning, adversarial, contextual_confusion
- **Best for**: Creative, context-aware prompts
- **Example**: Generated based on document analysis

### 3. Perturbation Strategies (10 Types)
- **entity_substitution**: Replace M1A1 → M2A3
- **numerical_manipulation**: Change 47 items → 48 items  
- **temporal_confusion**: Mix time periods
- **negation_injection**: Add "not" to test logic
- **authority_confusion**: Mix up regulations
- **causal_reversal**: Reverse cause-effect
- **multi_hop_reasoning**: Chain multiple steps
- **scope_expansion**: Go beyond source material
- **conditional_complexity**: Add if-then conditions
- **quantifier_manipulation**: Change all → some

## 🔧 Integration Examples

### Basic Usage in Your Code

```python
import asyncio
from dodhalueval.core import PromptGenerator, PromptPerturbator
from dodhalueval.data.pdf_processor import PDFProcessor
from dodhalueval.models.config import PromptGenerationConfig

async def generate_prompts_for_document(pdf_path):
    # 1. Process PDF
    processor = PDFProcessor(chunk_size=800, max_pages=10)
    result = processor.process_document(pdf_path)
    chunks = result['chunks']
    
    # 2. Generate template-based prompts
    config = PromptGenerationConfig(
        template_file='data/prompts/templates.yaml',
        max_prompts_per_document=20
    )
    generator = PromptGenerator(config)
    prompts = generator.generate_from_chunks(chunks)
    
    # 3. Apply perturbations
    perturbator = PromptPerturbator()
    all_prompts = []
    for prompt in prompts:
        variations = perturbator.apply_multiple_strategies(
            prompt, chunks[0], max_strategies=2
        )
        all_prompts.extend(variations)
    
    return all_prompts
```

### With Real LLM Providers

```python
import os
from dotenv import load_dotenv
from dodhalueval.providers import OpenAIProvider
from dodhalueval.core import LLMPromptGenerator
from dodhalueval.models.config import APIConfig

# Load environment variables
load_dotenv()

# Setup OpenAI provider (API key from .env)
api_config = APIConfig(
    provider='openai',
    model='gpt-4',
    api_key=os.getenv('OPENAI_API_KEY')
)
provider = OpenAIProvider(api_config)

# Generate sophisticated prompts
llm_generator = LLMPromptGenerator(provider, config)
llm_prompts = await llm_generator.generate_hallucination_prone_prompts(
    source_content, chunk, num_prompts=15, strategy='factual_probing'
)
```

## 📊 Understanding the Output

### Prompt Objects
Each generated prompt includes:
- **text**: The actual question
- **hallucination_type**: factual, logical, or context
- **generation_strategy**: How it was created
- **difficulty_level**: easy, medium, hard
- **metadata**: Additional details about generation

### Validation Results
- **is_valid**: Passes quality checks
- **score**: 0.0-1.0 quality score
- **issues**: Problems found
- **suggestions**: How to improve

## 🎯 Customization Options

### Template Categories Available
- `question_answer`: Basic Q&A format
- `factual_recall`: Specific facts and numbers
- `procedural`: Step-by-step processes  
- `technical`: Equipment specifications
- `tactical`: Military tactics and strategy
- `regulatory`: Policies and regulations
- `logistical`: Supply chain and support
- `training`: Education and exercises
- `intelligence`: Information gathering
- `hallucination_prone`: Designed to elicit errors

### Perturbation Control
```python
# Apply specific strategy
variations = perturbator.perturb(prompt, 'entity_substitution', chunk)

# Apply multiple strategies
variations = perturbator.apply_multiple_strategies(
    prompt, chunk, max_strategies=3
)

# Apply random strategy
variations = perturbator.apply_random_strategy(prompt, chunk)
```

### Validation Tuning
```python
# Strict validation
validator = PromptValidator(llm_provider=openai_provider)
result = await validator.validate_prompt(
    prompt, chunk, use_llm_validation=True
)

# Filter high-quality prompts only
valid_prompts = validator.filter_valid_prompts(
    all_prompts, validation_results, min_score=0.8
)
```

## 📈 Performance Tips

### For Large Document Sets
1. **Limit pages**: Set `max_pages=5` for testing
2. **Batch processing**: Process documents in chunks
3. **Cache enabled**: Use `cache_enabled=True` for PDFs
4. **Concurrent generation**: Use asyncio for LLM calls

### For Quality Results
1. **Use multiple strategies**: Combine template + LLM + perturbation
2. **Validate thoroughly**: Use both rule-based and LLM validation
3. **Filter by score**: Keep only high-quality prompts (score > 0.6)
4. **Domain-specific**: Ensure military/defense relevance

## 🔍 Troubleshooting

### Common Issues

**Low validation scores**:
- Content may lack military-specific terminology
- Try different template categories
- Use LLM-based generation for better context awareness

**Few prompt variations**:
- Content may lack entities for substitution
- Try scope_expansion or conditional_complexity strategies
- Use richer source documents

**PDF processing slow**:
- Reduce `max_pages` for testing
- Enable caching with `cache_enabled=True`
- Use smaller `chunk_size` values

### Getting Better Results

1. **Rich source content**: Use documents with specific details, numbers, equipment names
2. **Multiple approaches**: Combine template, LLM, and perturbation methods
3. **Validation feedback**: Use validation suggestions to improve templates
4. **Domain expertise**: Customize templates for your specific military domain

## 📞 Next Steps

1. **Start with quick test**: `python scripts/test_prompt_generation.py --mode quick`
2. **Test with your CSC data**: `python scripts/test_prompt_generation.py --mode csc --verbose`  
3. **Use real LLM providers**: `python scripts/test_prompt_generation.py --mode llm --provider openai`
4. **Customize** templates in `data/prompts/templates.yaml`
5. **Integrate** into your workflow using the code examples above
6. **Scale up** with comprehensive testing: `python scripts/test_prompt_generation.py --mode all`

## 🎯 Quick Test Commands

```bash
# Basic functionality test
python scripts/test_prompt_generation.py --mode quick

# Rich demo with sample military content  
python scripts/test_prompt_generation.py --mode demo --verbose

# Test with your CSC PDFs
python scripts/test_prompt_generation.py --mode csc --max-prompts 15

# Test OpenAI integration
python scripts/test_prompt_generation.py --mode llm --provider openai

# Full comprehensive test
python scripts/test_prompt_generation.py --mode all --provider openai --validate --verbose
```

The system is designed to be modular and extensible - you can use any combination of approaches based on your needs!