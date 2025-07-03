#!/usr/bin/env python3
"""
DoDHaluEval Pipeline Driver Script

This script provides a unified interface to run the complete hallucination evaluation pipeline.
All configuration is done through a Python dictionary or an external YAML config file.

Features:
- Document processing from PDFs
- Prompt generation (template-based and LLM-based)
- Response generation with multiple LLM providers
- Hallucination detection with multiple methods
- HaluEval-compatible dataset export
- Comprehensive metrics reporting

Usage:
    # Using inline configuration
    python3 scripts/run_pipeline.py
    
    # Using external config file
    python3 scripts/run_pipeline.py --config configs/pipeline_example.yaml
    
    # Override specific settings
    python3 scripts/run_pipeline.py --num-prompts 50 --providers openai
"""

import asyncio
import json
import sys
import os
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

# Add the src directory to the path so we can import dodhalueval
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from dodhalueval.data.pdf_processor import PDFProcessor
from dodhalueval.data.dataset_builder import DatasetBuilder
from dodhalueval.data.dataset_validator import DatasetValidator
from dodhalueval.core import (
    PromptGenerator,
    LLMPromptGenerator,
    PromptPerturbator,
    PromptValidator,
    ResponseGenerator,
    HallucinationDetector
)
from dodhalueval.core.evaluators import (
    HuggingFaceHHEMEvaluator,
    GEvalEvaluator,
    SelfCheckGPTEvaluator
)
from dodhalueval.providers import (
    OpenAIProvider,
    FireworksProvider,
    MockLLMProvider
)
from dodhalueval.models.config import (
    PDFProcessingConfig,
    PromptGenerationConfig,
    APIConfig
)
from dodhalueval.models.schemas import (
    DocumentChunk,
    Prompt,
    Response,
    EvaluationResult
)
from dodhalueval.utils.logger import get_logger
from dodhalueval.utils.metrics import MetricsCalculator

logger = get_logger(__name__)


# Default pipeline configuration
#
# CACHE CONTROL:
# 1. Global bypass: processing.bypass_cache = True (regenerates everything)
# 2. Selective bypass: processing.bypass_cache_steps.<step> = True
#    Available steps: pdf_extraction, document_chunking, prompt_generation,
#                     response_generation, hallucination_detection
#
# USAGE EXAMPLES:
# - After code improvements: bypass relevant steps only
# - New output directory: all caches are automatically bypassed
# - Testing specific fixes: enable bypass for affected components
#
DEFAULT_CONFIG = {
    "dataset": {
        "path": "data/CSC",
        "file_pattern": "*.pdf",
        "max_files": None,  # Process all files
        "max_pages_per_file": 10  # Limit pages for faster processing
    },
    
    "prompt_generation": {
        "num_prompts": 100,
        "methods": ["template", "llm"],  # Can be "template", "llm", or both
        "llm_provider": "openai",  # Provider for LLM-based generation
        "strategies": ["factual", "logical", "context"],  # Hallucination types
        "perturbation_enabled": True,
        "validation_enabled": True
    },
    
    "response_generation": {
        "providers": ["openai", "fireworks"],  # Can include "mock" for testing
        "models": {
            "openai": "gpt-4",
            "fireworks": "accounts/fireworks/models/llama-v3p1-70b-instruct"
        },
        "hallucination_rate": 0.3,  # 30% of responses will have injected hallucinations
        "max_concurrent": 5,
        "timeout": 30
    },
    
    "hallucination_detection": {
        "methods": ["hhem", "g_eval", "selfcheck"],  # Detection methods to use
        "ensemble": True,  # Use ensemble voting
        "providers": {
            "g_eval": "openai",
            "selfcheck": "openai"
        }
    },
    
    "output": {
        "directory": "output/pipeline_results",
        "format": "halueval",  # Output format
        "save_intermediate": True,  # Save intermediate results
        "generate_report": True  # Generate summary report
    },
    
    "processing": {
        "batch_size": 10,
        "show_progress": True,
        "verbose": False,
        "bypass_cache": False,  # Set to True to regenerate all intermediate files
        "bypass_cache_steps": {  # Granular control over which steps to bypass
            "pdf_extraction": False,  # Re-extract text from PDFs
            "document_chunking": False,  # Re-chunk documents
            "prompt_generation": False,  # Re-generate prompts
            "response_generation": False,  # Re-generate responses
            "hallucination_detection": False  # Re-run evaluations
        }
    }
}


class DoDHaluEvalPipeline:
    """Main pipeline orchestrator for DoDHaluEval."""
    
    def __init__(self, config: Dict[str, Any]):
        # Use defaults as base, then overlay loaded config
        self.config = self._merge_configs(DEFAULT_CONFIG, config)
        self.project_root = project_root
        self.start_time = time.time()
        self.stats = defaultdict(int)
        
        # Setup output directory
        self.output_dir = self.project_root / self.config["output"]["directory"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
    
    def _merge_configs(self, default: Dict, override: Dict) -> Dict:
        """Deep merge configurations, with override taking precedence."""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                # Override always takes precedence, even if default is not None
                result[key] = value
        return result
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")
        
        # PDF Processor - respect bypass_cache flags
        pdf_cache_bypass = (
            self.config["processing"].get("bypass_cache", False) or
            self.config["processing"].get("bypass_cache_steps", {}).get("pdf_extraction", False)
        )
        self.pdf_processor = PDFProcessor(
            chunk_size=1000,
            chunk_overlap=200,
            cache_enabled=not pdf_cache_bypass,
            max_pages=self.config["dataset"].get("max_pages_per_file")
        )
        
        # Prompt Generation
        prompt_config = PromptGenerationConfig(
            template_file=str(self.project_root / "data" / "prompts" / "templates.yaml"),
            max_prompts_per_document=self.config["prompt_generation"]["num_prompts"],
            perturbation_enabled=self.config["prompt_generation"]["perturbation_enabled"],
            hallucination_types=self.config["prompt_generation"]["strategies"]
        )
        
        self.template_generator = PromptGenerator(prompt_config)
        self.perturbator = PromptPerturbator()
        self.validator = PromptValidator()
        
        # LLM Providers
        self.providers = self._initialize_providers()
        
        # LLM-based prompt generator
        if "llm" in self.config["prompt_generation"]["methods"]:
            provider_name = self.config["prompt_generation"]["llm_provider"]
            if provider_name in self.providers:
                self.llm_generator = LLMPromptGenerator(
                    self.providers[provider_name], 
                    prompt_config
                )
            else:
                logger.warning(f"LLM provider '{provider_name}' not available for prompt generation")
                self.llm_generator = None
        else:
            self.llm_generator = None
        
        # Response Generator with generation config
        response_config = self.config.get('response_generation', {})
        self.response_generator = ResponseGenerator(
            self.providers,
            generation_config=response_config
        )
        
        # Hallucination Detectors
        self.evaluators = self._initialize_evaluators()
        self.hallucination_detector = HallucinationDetector(list(self.evaluators.values()))
        
        # Dataset Builder
        self.dataset_builder = DatasetBuilder()
        self.dataset_validator = DatasetValidator()
        
        # Metrics Calculator
        self.metrics_calculator = MetricsCalculator()
    
    def _initialize_providers(self) -> Dict[str, Any]:
        """Initialize LLM providers based on configuration."""
        providers = {}
        
        # Always include mock provider for testing
        providers["mock"] = MockLLMProvider()
        
        # OpenAI
        if "openai" in self.config["response_generation"]["providers"]:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                model = self.config["response_generation"]["models"].get("openai", "gpt-4")
                api_config = APIConfig(
                    provider="openai",
                    api_key=api_key,
                    model=model
                )
                providers["openai"] = OpenAIProvider(api_config)
                logger.info(f"Initialized OpenAI provider with model: {model}")
            else:
                logger.warning("OpenAI API key not found, skipping OpenAI provider")
        
        # Fireworks
        if "fireworks" in self.config["response_generation"]["providers"]:
            api_key = os.getenv("FIREWORKS_API_KEY")
            if api_key:
                model = self.config["response_generation"]["models"].get(
                    "fireworks", 
                    "accounts/fireworks/models/llama-v3p1-70b-instruct"
                )
                api_config = APIConfig(
                    provider="fireworks",
                    api_key=api_key,
                    model=model,
                    base_url="https://api.fireworks.ai/inference/v1"
                )
                providers["fireworks"] = FireworksProvider(api_config)
                logger.info(f"Initialized Fireworks provider with model: {model}")
            else:
                logger.warning("Fireworks API key not found, skipping Fireworks provider")
        
        return providers
    
    def _initialize_evaluators(self) -> Dict[str, Any]:
        """Initialize hallucination detection evaluators."""
        evaluators = {}
        methods = self.config["hallucination_detection"]["methods"]
        
        # HuggingFace HHEM
        if "hhem" in methods:
            evaluators["hhem"] = HuggingFaceHHEMEvaluator()
            logger.info("Initialized HuggingFace HHEM evaluator")
        
        # G-Eval
        if "g_eval" in methods:
            provider_name = self.config["hallucination_detection"]["providers"].get("g_eval", "openai")
            if provider_name in self.providers:
                evaluators["g_eval"] = GEvalEvaluator(self.providers[provider_name])
                logger.info(f"Initialized G-Eval detector with {provider_name}")
            else:
                logger.warning(f"Provider '{provider_name}' not available for G-Eval")
        
        # SelfCheckGPT
        if "selfcheck" in methods:
            provider_name = self.config["hallucination_detection"]["providers"].get("selfcheck", "openai")
            if provider_name in self.providers:
                evaluators["selfcheck"] = SelfCheckGPTEvaluator(self.providers[provider_name])
                logger.info(f"Initialized SelfCheckGPT with {provider_name}")
            else:
                logger.warning(f"Provider '{provider_name}' not available for SelfCheckGPT")
        
        return evaluators
    
    def _should_bypass_cache_for_step(self, step_name: str) -> bool:
        """Check if cache should be bypassed for a specific step."""
        # Global bypass
        if self.config["processing"].get("bypass_cache", False):
            return True
        
        # Step-specific bypass
        step_bypass_config = self.config["processing"].get("bypass_cache_steps", {})
        
        # Map internal names to config names
        step_mapping = {
            "chunks": "document_chunking",
            "prompts": "prompt_generation", 
            "responses": "response_generation",
            "evaluations": "hallucination_detection"
        }
        
        config_step_name = step_mapping.get(step_name, step_name)
        return step_bypass_config.get(config_step_name, False)
    
    def _load_intermediate_if_exists(self, name: str) -> Optional[List[Any]]:
        """Load intermediate results if they exist and caching is enabled."""
        # Check if cache bypass is enabled for this step
        if self._should_bypass_cache_for_step(name):
            logger.info(f"Cache bypass enabled for {name} - regenerating")
            return None
            
        file_path = self.output_dir / f"intermediate_{name}.json"
        if file_path.exists():
            logger.info(f"Loading cached {name} from {file_path}")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                logger.warning(f"Failed to load cached {name}: {e}")
                return None
        return None
    
    async def _process_documents_with_cache(self) -> List[DocumentChunk]:
        """Process documents or load from cache."""
        cached_data = self._load_intermediate_if_exists("chunks")
        if cached_data:
            # Convert back to DocumentChunk objects
            chunks = []
            for chunk_data in cached_data:
                chunk = DocumentChunk(**chunk_data)
                chunks.append(chunk)
            logger.info(f"Loaded {len(chunks)} chunks from cache")
            # Store chunks for document store building
            self._current_chunks = chunks
            return chunks
        return await self._process_documents()
    
    async def _generate_prompts_with_cache(self, chunks: List[DocumentChunk]) -> List[Prompt]:
        """Generate prompts or load from cache."""
        cached_data = self._load_intermediate_if_exists("prompts")
        if cached_data:
            # Convert back to Prompt objects
            prompts = []
            for prompt_data in cached_data:
                prompt = Prompt(**prompt_data)
                prompts.append(prompt)
            logger.info(f"Loaded {len(prompts)} prompts from cache")
            self._original_prompts = prompts  # Store for later use
            return prompts
        prompts = await self._generate_prompts(chunks)
        self._original_prompts = prompts  # Store for later use
        return prompts
    
    async def _generate_responses_with_cache(self, prompts: List[Prompt], chunks: List[DocumentChunk] = None) -> List[Response]:
        """Generate responses or load from cache."""
        cached_data = self._load_intermediate_if_exists("responses")
        if cached_data:
            # Convert back to Response objects
            responses = []
            for response_data in cached_data:
                response = Response(**response_data)
                responses.append(response)
            logger.info(f"Loaded {len(responses)} responses from cache")
            return responses
        return await self._generate_responses(prompts, chunks)
    
    async def _detect_hallucinations_with_cache(self, responses: List[Response], prompts: List[Prompt]) -> List[Any]:
        """Detect hallucinations or load from cache."""
        cached_data = self._load_intermediate_if_exists("evaluations")
        if cached_data:
            logger.info(f"Loaded {len(cached_data)} evaluations from cache")
            # Convert cached data back to proper objects
            evaluations = []
            for eval_data in cached_data:
                if isinstance(eval_data, dict):
                    # Check if this looks like an EnsembleResult (has ensemble_score)
                    if 'ensemble_score' in eval_data:
                        # Reconstruct EnsembleResult with all required attributes
                        class CachedEnsembleResult:
                            def __init__(self, data):
                                self.ensemble_score = data.get('ensemble_score', 0.0)
                                self.ensemble_confidence = data.get('ensemble_confidence', 0.0)
                                self.is_hallucinated = data.get('ensemble_score', 0.0) > 0.5
                                self.metadata = data.get('metadata', {})
                                self.response_id = data.get('response_id', 'unknown')
                                
                                # Reconstruct individual_results
                                self.individual_results = []
                                individual_data = data.get('individual_results', [])
                                for individual in individual_data:
                                    if isinstance(individual, dict):
                                        # Create minimal individual result object
                                        class CachedIndividualResult:
                                            def __init__(self, ind_data):
                                                self.evaluator_name = ind_data.get('method', 'unknown')
                                                # Create hallucination_score object
                                                class CachedHallucinationScore:
                                                    def __init__(self, score_data):
                                                        if isinstance(score_data, dict):
                                                            self.score = score_data.get('score', 0.0)
                                                            self.confidence = score_data.get('confidence', 0.0)
                                                        else:
                                                            self.score = float(score_data) if score_data else 0.0
                                                            self.confidence = 0.0
                                                
                                                self.hallucination_score = CachedHallucinationScore(
                                                    ind_data.get('hallucination_score', ind_data.get('score', 0.0))
                                                )
                                        
                                        self.individual_results.append(CachedIndividualResult(individual))
                        
                        evaluations.append(CachedEnsembleResult(eval_data))
                    else:
                        # Try to reconstruct as EvaluationResult
                        try:
                            from dodhalueval.models.schemas import EvaluationResult
                            evaluations.append(EvaluationResult(**eval_data))
                        except:
                            # Fallback: create simple object with required attributes
                            class CachedEvaluation:
                                def __init__(self, data):
                                    self.is_hallucinated = data.get('is_hallucinated', False)
                                    self.metadata = data.get('metadata', {})
                            evaluations.append(CachedEvaluation(eval_data))
                else:
                    evaluations.append(eval_data)
            return evaluations
        return await self._detect_hallucinations(responses, prompts)
    
    async def run(self):
        """Run the complete pipeline."""
        logger.info("=" * 80)
        logger.info("Starting DoDHaluEval Pipeline")
        logger.info("=" * 80)
        
        try:
            # Step 1: Process documents (or load from cache)
            chunks = await self._process_documents_with_cache()
            
            # Step 2: Generate prompts (or load from cache)
            prompts = await self._generate_prompts_with_cache(chunks)
            
            # Step 3: Generate responses (or load from cache)
            responses = await self._generate_responses_with_cache(prompts, chunks)
            
            # Step 4: Detect hallucinations (or load from cache)
            evaluations = await self._detect_hallucinations_with_cache(responses, prompts)
            
            # Step 5: Build dataset
            dataset = self._build_dataset(responses, evaluations)
            
            # Step 6: Calculate metrics
            metrics = self._calculate_metrics(evaluations, responses)
            
            # Step 7: Save results
            self._save_results(dataset, metrics)
            
            # Step 8: Generate report
            if self.config["output"]["generate_report"]:
                self._generate_report(metrics)
            
            elapsed_time = time.time() - self.start_time
            logger.info(f"\nPipeline completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    async def _process_documents(self) -> List[DocumentChunk]:
        """Process PDF documents into chunks."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 1: Processing Documents")
        logger.info("=" * 50)
        
        dataset_path = self.project_root / self.config["dataset"]["path"]
        file_pattern = self.config["dataset"]["file_pattern"]
        max_files = self.config["dataset"]["max_files"]
        
        pdf_files = list(dataset_path.glob(file_pattern))
        if max_files:
            pdf_files = pdf_files[:max_files]
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_chunks = []
        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            result = self.pdf_processor.process_document(str(pdf_file))
            chunks_data = result.get('chunks', [])
            
            # Handle both DocumentChunk objects and dict chunks
            for chunk_item in chunks_data:
                if isinstance(chunk_item, DocumentChunk):
                    # Already a DocumentChunk object
                    all_chunks.append(chunk_item)
                else:
                    # Convert dict to DocumentChunk object
                    doc_chunk = DocumentChunk(
                        document_id=str(pdf_file.stem),
                        content=chunk_item.get('text', chunk_item.get('content', '')),
                        page_number=chunk_item.get('page_number', 1),
                        chunk_index=chunk_item.get('chunk_index', 0),
                        metadata=chunk_item.get('metadata', {})
                    )
                    all_chunks.append(doc_chunk)
            
            self.stats["documents_processed"] += 1
            self.stats["chunks_created"] += len(chunks_data)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(pdf_files)} documents")
        
        # Store chunks for document store building
        self._current_chunks = all_chunks
        
        if self.config["output"]["save_intermediate"]:
            self._save_intermediate("chunks", [chunk.model_dump() for chunk in all_chunks])
        
        return all_chunks
    
    async def _generate_prompts(self, chunks: List[DocumentChunk]) -> List[Prompt]:
        """Generate prompts from document chunks."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 2: Generating Prompts")
        logger.info("=" * 50)
        
        all_prompts = []
        methods = self.config["prompt_generation"]["methods"]
        
        # Template-based generation
        if "template" in methods:
            logger.info("Generating template-based prompts...")
            template_prompts = self.template_generator.generate_from_chunks(chunks)
            
            # Apply perturbations if enabled
            if self.config["prompt_generation"]["perturbation_enabled"]:
                perturbed_prompts = []
                for prompt in template_prompts[:20]:  # Limit perturbations
                    perturbed = self.perturbator.perturb(prompt, strategy="entity_substitution")
                    perturbed_prompts.extend(perturbed)
                template_prompts.extend(perturbed_prompts)
            
            all_prompts.extend(template_prompts)
            logger.info(f"Generated {len(template_prompts)} template-based prompts")
        
        # LLM-based generation
        if "llm" in methods and self.llm_generator:
            logger.info("Generating LLM-based prompts...")
            
            # Enhanced content filtering for meaningful chunks
            content_chunks = []
            for chunk in chunks:
                content = chunk.content.strip()
                words = content.split()
                
                # Basic length and content checks
                if (len(content) > 20 
                    and not content.isdigit() 
                    and len(words) > 5 
                    and any(word.isalpha() for word in words)):
                    
                    # Advanced quality checks
                    # Exclude header-only content (too many uppercase, too repetitive)
                    uppercase_ratio = sum(1 for c in content if c.isupper()) / len(content)
                    if uppercase_ratio > 0.7:  # Skip if >70% uppercase (likely headers)
                        continue
                    
                    # Exclude chunks with too much repetition
                    unique_words = set(word.lower() for word in words if word.isalpha())
                    if len(unique_words) < len(words) * 0.3:  # Skip if <30% unique words
                        continue
                    
                    # Require sentence-like structure (contains periods, commas, or paragraph indicators)
                    if not any(punct in content for punct in ['.', ',', ':', ';', '•', '▪', '-']):
                        continue
                    
                    # Require substantial content (not just course titles/headers)
                    substantive_indicators = [
                        'provides', 'includes', 'covers', 'discusses', 'describes', 'explains',
                        'operations', 'procedures', 'requirements', 'guidelines', 'protocols',
                        'training', 'doctrine', 'planning', 'execution', 'strategy', 'tactics'
                    ]
                    if any(indicator in content.lower() for indicator in substantive_indicators):
                        content_chunks.append(chunk)
                    elif len(words) > 20:  # Allow longer chunks even without indicators
                        content_chunks.append(chunk)
            
            logger.info(f"Filtered to {len(content_chunks)} content-rich chunks from {len(chunks)} total chunks")
            
            if content_chunks:
                # Select a subset of content-rich chunks for LLM generation
                selected_chunks = content_chunks[:min(10, len(content_chunks))]  # Limit for cost/time
                
                for chunk in selected_chunks:
                    logger.info(f"Generating prompts from chunk with {len(chunk.content)} characters: {chunk.content[:100]}...")
                    try:
                        # Calculate prompts per chunk to reach total target
                        remaining_prompts = self.config["prompt_generation"]["num_prompts"] - len(all_prompts)
                        chunks_remaining = len(selected_chunks) - selected_chunks.index(chunk)
                        prompts_per_chunk = max(1, remaining_prompts // chunks_remaining)
                        
                        llm_prompts = await self.llm_generator.generate_hallucination_prone_prompts(
                            source_content=chunk.content,
                            source_chunk=chunk,
                            num_prompts=prompts_per_chunk
                        )
                        all_prompts.extend(llm_prompts)
                    except Exception as e:
                        logger.warning(f"LLM generation failed for chunk: {e}")
                        continue
                
                logger.info(f"Generated {len(all_prompts) - len(template_prompts) if 'template' in methods else len(all_prompts)} LLM-based prompts")
            else:
                logger.warning("No content-rich chunks found for LLM generation")
        
        # Validate prompts if enabled
        if self.config["prompt_generation"]["validation_enabled"]:
            logger.info("Validating prompts...")
            # For validation, we need the source chunk, so we'll use a simple heuristic
            # In a real implementation, you'd track which chunk each prompt came from
            validated_prompts = []
            for prompt in all_prompts:
                # Simple validation: check if prompt has basic required fields and isn't empty
                if (hasattr(prompt, 'text') and prompt.text and 
                    len(prompt.text.strip()) > 10 and 
                    '?' in prompt.text):
                    validated_prompts.append(prompt)
            all_prompts = validated_prompts
            logger.info(f"Validated {len(all_prompts)} prompts")
        
        # If no prompts were generated, create fallback prompts
        if not all_prompts and chunks:
            logger.warning("No prompts generated from template or LLM methods, creating fallback prompts")
            fallback_prompts = self._create_fallback_prompts(chunks)
            all_prompts.extend(fallback_prompts)
            logger.info(f"Created {len(fallback_prompts)} fallback prompts")
        
        # Limit to configured number
        max_prompts = self.config["prompt_generation"]["num_prompts"]
        if len(all_prompts) > max_prompts:
            all_prompts = all_prompts[:max_prompts]
        
        self.stats["prompts_generated"] = len(all_prompts)
        
        if self.config["output"]["save_intermediate"]:
            self._save_intermediate("prompts", [prompt.model_dump() for prompt in all_prompts])
        
        return all_prompts
    
    def _create_fallback_prompts(self, chunks: List[DocumentChunk]) -> List[Prompt]:
        """Create fallback prompts when template and LLM generation fail."""
        fallback_prompts = []
        
        # Create simple question templates for military content
        question_templates = [
            "What does this document say about {topic}?",
            "According to this text, what are the procedures for {topic}?", 
            "What requirements does this document specify for {topic}?",
            "How does this document describe {topic}?",
            "What guidelines does this text provide for {topic}?"
        ]
        
        # Military topics to use as fallback
        military_topics = [
            "operations", "equipment", "training", "procedures", "safety",
            "command", "personnel", "maintenance", "security", "protocols"
        ]
        
        # Create prompts from each chunk
        for i, chunk in enumerate(chunks[:5]):  # Limit to 5 chunks to avoid too many prompts
            # Extract a relevant topic from the chunk content if possible
            chunk_text = chunk.content.lower()
            topic = "military procedures"  # default
            
            # Try to find a relevant topic in the chunk
            for mil_topic in military_topics:
                if mil_topic in chunk_text:
                    topic = mil_topic
                    break
            
            # Create a prompt using a template
            template = question_templates[i % len(question_templates)]
            question_text = template.format(topic=topic)
            
            # Add context from the chunk 
            context_snippet = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            full_question = f"{question_text} Context: {context_snippet}"
            
            prompt = Prompt(
                text=full_question,
                source_document_id=chunk.document_id,
                source_chunk_id=getattr(chunk, 'id', f"chunk_{i}"),
                generation_strategy="fallback_template",
                hallucination_type=["factual", "contextual"][i % 2],  # Alternate hallucination types
                metadata={
                    "fallback_reason": "template_and_llm_generation_failed",
                    "source_chunk_index": chunk.chunk_index,
                    "topic_extracted": topic,
                    "template_name": f"fallback_{i % len(question_templates)}"
                }
            )
            fallback_prompts.append(prompt)
        
        # If no chunks available, create at least one basic prompt
        if not chunks and not fallback_prompts:
            basic_prompt = Prompt(
                text="What are the key procedures outlined in this military document?",
                source_document_id="unknown",
                source_chunk_id="fallback_basic",
                generation_strategy="fallback_basic",
                hallucination_type="factual",
                metadata={
                    "fallback_reason": "no_chunks_available",
                    "template_name": "basic_fallback"
                }
            )
            fallback_prompts.append(basic_prompt)
        
        return fallback_prompts
    
    async def _generate_responses(self, prompts: List[Prompt], chunks: List[DocumentChunk] = None) -> List[Response]:
        """Generate responses for prompts."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 3: Generating Responses")
        logger.info("=" * 50)
        
        # Build document store for context-aware response generation
        # Pass the chunks we just processed to ensure context is available
        document_store = self._build_document_store(prompts, chunks)
        
        # Create context-aware response generator with generation config
        response_config = self.config.get('response_generation', {})
        context_response_generator = ResponseGenerator(
            providers=self.providers, 
            document_store=document_store,
            generation_config=response_config
        )
        
        providers_to_use = []
        for provider_name in self.config["response_generation"]["providers"]:
            if provider_name in self.providers:
                providers_to_use.append(provider_name)
        
        logger.info(f"Using providers: {providers_to_use}")
        
        responses = await context_response_generator.generate_responses(
            prompts=prompts,
            models=providers_to_use,
            hallucination_rate=self.config["response_generation"]["hallucination_rate"],
            chunks=chunks
        )
        
        self.stats["responses_generated"] = len(responses)
        
        # Count hallucinations by type
        for response in responses:
            if response.contains_hallucination:
                # Count by each hallucination type if multiple types exist
                for hall_type in response.hallucination_types:
                    self.stats[f"hallucinations_{hall_type}"] += 1
        
        logger.info(f"Generated {len(responses)} responses")
        logger.info(f"Hallucination injection rate: {self.config['response_generation']['hallucination_rate']}")
        
        if self.config["output"]["save_intermediate"]:
            self._save_intermediate("responses", [response.model_dump() for response in responses])
        
        return responses
    
    def _build_document_store(self, prompts: List[Prompt], chunks: List[DocumentChunk] = None) -> Dict[str, Any]:
        """Build document store from chunks for context-aware generation."""
        document_store = {}
        chunks_data = []
        
        # First try to use provided chunks
        if chunks:
            logger.info(f"Using provided chunks: {len(chunks)} chunks")
            chunks_data = [chunk.model_dump() for chunk in chunks]
        else:
            # Fall back to cached chunks if available
            chunks_cache_path = self.output_dir / "intermediate_chunks.json"
            if chunks_cache_path.exists():
                try:
                    with open(chunks_cache_path, 'r') as f:
                        chunks_data = json.load(f)
                    logger.info(f"Loaded {len(chunks_data)} chunks from cache")
                except Exception as e:
                    logger.warning(f"Failed to load chunks cache: {e}")
                    return document_store
            else:
                logger.warning("No chunks available - responses will be generated without document context")
                return document_store
        
        # Build chunk lookup
        for chunk_data in chunks_data:
            chunk_id = chunk_data.get('id')
            if chunk_id:
                document_store[chunk_id] = {
                    'text': chunk_data.get('content', ''),
                    'document_id': chunk_data.get('document_id', ''),
                    'page_number': chunk_data.get('page_number', 1)
                }
        
        # Build document lookup (aggregate chunks per document)
        doc_chunks = {}
        for chunk_data in chunks_data:
            doc_id = chunk_data.get('document_id')
            if doc_id:
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                doc_chunks[doc_id].append(chunk_data.get('content', ''))
        
        # Add document-level entries
        for doc_id, chunk_texts in doc_chunks.items():
            document_store[doc_id] = {
                'title': doc_id,
                'content': '\\n\\n'.join(chunk_texts[:5])  # First 5 chunks as sample
            }
        
        logger.info(f"Built document store with {len(document_store)} entries from {len(chunks_data)} chunks")
        return document_store

    async def _detect_hallucinations(self, responses: List[Response], prompts: List[Prompt]) -> List[EvaluationResult]:
        """Detect hallucinations in responses."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 4: Detecting Hallucinations")
        logger.info("=" * 50)
        
        logger.info(f"Using evaluators: {list(self.evaluators.keys())}")
        logger.info(f"Ensemble evaluation: {self.config['hallucination_detection']['ensemble']}")
        
        # Create a mapping of responses to their prompts
        # Since we may have multiple responses per prompt (different models)
        # we need to create corresponding prompt list
        response_prompts = []
        prompt_dict = {prompt.id: prompt for prompt in prompts}
        
        for response in responses:
            # Find the corresponding prompt for this response
            corresponding_prompt = prompt_dict.get(response.prompt_id)
            if corresponding_prompt:
                response_prompts.append(corresponding_prompt)
            else:
                # Fallback: use first prompt if mapping fails
                logger.warning(f"Could not find prompt for response {response.id}, using fallback")
                response_prompts.append(prompts[0] if prompts else None)
        
        logger.info(f"Evaluating {len(responses)} responses with {len(response_prompts)} matched prompts")
        
        evaluations = await self.hallucination_detector.evaluate_batch(
            responses=responses,
            prompts=response_prompts
        )
        
        self.stats["evaluations_completed"] = len(evaluations)
        
        # Count detections
        for evaluation in evaluations:
            if evaluation.is_hallucinated:
                self.stats["hallucinations_detected"] += 1
        
        logger.info(f"Completed {len(evaluations)} evaluations")
        
        if self.config["output"]["save_intermediate"]:
            # Handle EnsembleResult objects (dataclasses) differently from Pydantic models
            eval_data = []
            for eval_result in evaluations:
                if hasattr(eval_result, 'model_dump'):
                    # Pydantic model
                    eval_data.append(eval_result.model_dump())
                elif hasattr(eval_result, '__dataclass_fields__'):
                    # Dataclass
                    from dataclasses import asdict
                    eval_data.append(asdict(eval_result))
                else:
                    # Fallback
                    eval_data.append(str(eval_result))
            self._save_intermediate("evaluations", eval_data)
        
        return evaluations
    
    def _build_dataset(self, responses: List[Response], evaluations: List[Any]) -> Any:
        """Build HaluEval-compatible dataset."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 5: Building Dataset")
        logger.info("=" * 50)
        
        logger.info(f"Building dataset with {len(responses)} responses, {len(evaluations)} evaluations")
        
        # Create corresponding prompts list by mapping response.prompt_id to actual prompts
        # Since we have multiple responses per prompt, we need to duplicate prompts accordingly
        prompt_dict = {}
        if hasattr(self, '_original_prompts'):
            prompt_dict = {p.id: p for p in self._original_prompts}
        
        corresponding_prompts = []
        for response in responses:
            # Find the prompt that generated this response
            prompt = prompt_dict.get(response.prompt_id)
            if prompt:
                corresponding_prompts.append(prompt)
            else:
                # Create a dummy prompt if we can't find the original
                from dodhalueval.models.schemas import Prompt
                dummy_prompt = Prompt(
                    id=response.prompt_id,
                    text=f"[Original prompt for response {response.id}]",
                    expected_answer="",
                    source_document="unknown",
                    metadata={}
                )
                corresponding_prompts.append(dummy_prompt)
        
        logger.info(f"Mapped {len(corresponding_prompts)} prompts to {len(responses)} responses")
        
        # Now we have 1:1 mapping: corresponding_prompts, responses, evaluations
        dataset = self.dataset_builder.build_halueval_format(
            prompts=corresponding_prompts,
            responses=responses,
            evaluations=evaluations,
            dataset_name="dod_halueval"
        )
        
        # Validate dataset
        validation_result = self.dataset_validator.validate_dataset(dataset)
        if validation_result.is_valid:
            logger.info("Dataset validation passed")
        else:
            logger.warning(f"Dataset validation failed with {len(validation_result.issues)} issues")
        
        self.stats["dataset_size"] = len(dataset.samples)
        
        return dataset
    
    def _calculate_metrics(self, evaluations: List[EvaluationResult], responses: List[Response]) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 6: Calculating Metrics")
        logger.info("=" * 50)
        
        # Extract predictions and ground truth
        predictions = []
        ground_truth = []
        
        # Create response lookup by response_id for accurate ground truth mapping
        response_lookup = {resp.id: resp for resp in responses}
        
        for evaluation in evaluations:
            # Prediction from evaluation
            predictions.append(evaluation.is_hallucinated)
            
            # Ground truth from corresponding response (CORRECT - Academic Standard)
            response = response_lookup.get(evaluation.response_id)
            if response and response.contains_hallucination is not None:
                ground_truth.append(response.contains_hallucination)
            else:
                # Fallback: assume no hallucination if injection info missing
                ground_truth.append(False)
                logger.warning(f"No ground truth available for evaluation {evaluation.response_id}")
        
        # Calculate overall metrics
        metrics = self.metrics_calculator.calculate_detection_metrics(predictions, ground_truth)
        
        # Calculate per-category metrics
        category_results = defaultdict(lambda: {"predictions": [], "ground_truth": []})
        for i, evaluation in enumerate(evaluations):
            # Get category from corresponding response's hallucination types
            response = response_lookup.get(evaluation.response_id)
            if response and response.hallucination_types:
                category = response.hallucination_types[0] if response.hallucination_types else "unknown"
            else:
                category = evaluation.metadata.get("hallucination_type", "unknown")
            
            category_results[category]["predictions"].append(predictions[i])
            category_results[category]["ground_truth"].append(ground_truth[i])
        
        per_category_metrics = {}
        for category, data in category_results.items():
            per_category_metrics[category] = self.metrics_calculator.calculate_detection_metrics(
                data["predictions"], 
                data["ground_truth"]
            )
        
        # Handle metrics serialization (might be Pydantic models or dataclasses)
        def serialize_metrics(obj):
            if hasattr(obj, 'model_dump'):
                return obj.model_dump()
            elif hasattr(obj, '__dataclass_fields__'):
                from dataclasses import asdict
                return asdict(obj)
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)
        
        return {
            "overall": serialize_metrics(metrics),
            "per_category": {k: serialize_metrics(v) for k, v in per_category_metrics.items()},
            "statistics": dict(self.stats)
        }
    
    def _save_results(self, dataset: Any, metrics: Dict[str, Any]):
        """Save final results."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 7: Saving Results")
        logger.info("=" * 50)
        
        # Save dataset in HaluEval format
        output_file = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.dataset_builder.export_jsonl(dataset, str(output_file))
        logger.info(f"Saved dataset to: {output_file}")
        
        # Save metrics
        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to: {metrics_file}")
        
        # Save configuration
        config_file = self.output_dir / "pipeline_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Saved configuration to: {config_file}")
    
    def _save_intermediate(self, name: str, data: List[Dict]):
        """Save intermediate results."""
        file_path = self.output_dir / f"intermediate_{name}.json"
        
        # Custom JSON encoder to handle datetime objects
        def json_serializer(obj):
            if hasattr(obj, 'isoformat'):  # datetime objects
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):  # Other objects
                return obj.__dict__
            else:
                return str(obj)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=json_serializer)
        logger.info(f"Saved intermediate results to: {file_path}")
    
    def _generate_report(self, metrics: Dict[str, Any]):
        """Generate a human-readable report."""
        report_file = self.output_dir / "report.txt"
        
        with open(report_file, 'w') as f:
            f.write("DoDHaluEval Pipeline Report\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Runtime: {time.time() - self.start_time:.2f} seconds\n\n")
            
            f.write("Configuration Summary:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Dataset Path: {self.config['dataset']['path']}\n")
            f.write(f"Number of Prompts: {self.config['prompt_generation']['num_prompts']}\n")
            f.write(f"Prompt Methods: {', '.join(self.config['prompt_generation']['methods'])}\n")
            f.write(f"Response Providers: {', '.join(self.config['response_generation']['providers'])}\n")
            f.write(f"Detection Methods: {', '.join(self.config['hallucination_detection']['methods'])}\n")
            f.write(f"Hallucination Rate: {self.config['response_generation']['hallucination_rate']}\n\n")
            
            f.write("Pipeline Statistics:\n")
            f.write("-" * 40 + "\n")
            for key, value in metrics["statistics"].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("Overall Metrics:\n")
            f.write("-" * 40 + "\n")
            overall = metrics["overall"]
            f.write(f"Accuracy: {overall['accuracy']:.3f}\n")
            f.write(f"Precision: {overall['precision']:.3f}\n")
            f.write(f"Recall: {overall['recall']:.3f}\n")
            f.write(f"F1 Score: {overall['f1_score']:.3f}\n\n")
            
            f.write("Per-Category Metrics:\n")
            f.write("-" * 40 + "\n")
            for category, cat_metrics in metrics["per_category"].items():
                f.write(f"\n{category.upper()}:\n")
                f.write(f"  Accuracy: {cat_metrics['accuracy']:.3f}\n")
                f.write(f"  Precision: {cat_metrics['precision']:.3f}\n")
                f.write(f"  Recall: {cat_metrics['recall']:.3f}\n")
                f.write(f"  F1 Score: {cat_metrics['f1_score']:.3f}\n")
        
        logger.info(f"Generated report: {report_file}")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DoDHaluEval Pipeline Driver")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--num-prompts", type=int, help="Override number of prompts")
    parser.add_argument("--providers", nargs="+", help="Override response providers")
    parser.add_argument("--output-dir", help="Override output directory")
    
    args = parser.parse_args()
    
    # Load configuration - default to example config  
    config_path = "configs/pipeline_example_test.yaml"
    # Resolve path relative to project root
    full_config_path = project_root / config_path
    print(f"DEBUG: Project root: {project_root}")
    print(f"DEBUG: Full config path: {full_config_path}")
    print(f"DEBUG: Config file exists: {full_config_path.exists()}")
    config = load_config(str(full_config_path))
    
    # Debug: Print the loaded config
    print(f"DEBUG: Loaded config from {config_path}")
    print(f"DEBUG: max_files = {config.get('dataset', {}).get('max_files', 'NOT SET')}")
    
    # Apply command-line overrides
    if args.num_prompts:
        config.setdefault("prompt_generation", {})["num_prompts"] = args.num_prompts
    if args.providers:
        config.setdefault("response_generation", {})["providers"] = args.providers
    if args.output_dir:
        config.setdefault("output", {})["directory"] = args.output_dir
    
    # Create and run pipeline
    pipeline = DoDHaluEvalPipeline(config)
    
    # Run the async pipeline
    asyncio.run(pipeline.run())


def run_with_fresh_cache(config_path: str = None, output_dir: str = None):
    """Run pipeline with cache bypass to regenerate everything.
    
    Args:
        config_path: Path to config file (optional)
        output_dir: Output directory (optional)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Enable cache bypass
    config["processing"]["bypass_cache"] = True
    
    # Set output directory if provided
    if output_dir:
        config["output"]["directory"] = output_dir
    
    print("Running pipeline with cache bypass enabled - regenerating all intermediate files")
    
    # Create and run pipeline
    pipeline = DoDHaluEvalPipeline(config)
    asyncio.run(pipeline.run())


def run_with_selective_cache_bypass(config_path: str = None, bypass_steps: list = None, output_dir: str = None):
    """Run pipeline with selective cache bypass for specific steps.
    
    Args:
        config_path: Path to config file (optional)
        bypass_steps: List of steps to bypass cache for (optional)
                     Options: ['pdf_extraction', 'document_chunking', 'prompt_generation', 
                              'response_generation', 'hallucination_detection']
        output_dir: Output directory (optional)
    
    Example:
        # Only regenerate responses and evaluations
        run_with_selective_cache_bypass(
            config_path="configs/pipeline_example_test.yaml",
            bypass_steps=["response_generation", "hallucination_detection"]
        )
    """
    # Load configuration
    config = load_config(config_path)
    
    # Configure selective bypass
    if bypass_steps:
        config["processing"]["bypass_cache"] = False  # Disable global bypass
        
        # Reset all step bypasses to False first
        step_config = config["processing"].setdefault("bypass_cache_steps", {})
        for step in ["pdf_extraction", "document_chunking", "prompt_generation", 
                    "response_generation", "hallucination_detection"]:
            step_config[step] = False
        
        # Enable bypass for specified steps
        for step in bypass_steps:
            if step in step_config:
                step_config[step] = True
                print(f"Cache bypass enabled for: {step}")
            else:
                print(f"Warning: Unknown step '{step}' ignored")
    
    # Set output directory if provided
    if output_dir:
        config["output"]["directory"] = output_dir
    
    # Create and run pipeline
    pipeline = DoDHaluEvalPipeline(config)
    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()