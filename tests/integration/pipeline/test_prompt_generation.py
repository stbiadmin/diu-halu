#!/usr/bin/env python3
"""
Consolidated DoDHaluEval Prompt Generation Test Script

This script provides various testing modes for the prompt generation system:
- Quick test with sample content
- Demo with rich military content  
- Test with CSC PDF data
- Test with real LLM providers
- Comprehensive end-to-end testing

Usage:
    python test_prompt_generation.py --mode quick
    python test_prompt_generation.py --mode demo --verbose
    python test_prompt_generation.py --mode csc --max-prompts 20
    python test_prompt_generation.py --mode llm --provider openai
    python test_prompt_generation.py --mode all --validate
"""

import asyncio
import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

import click
# from dotenv import load_dotenv  # Removed for CI/CD compatibility

# Add the src directory to the path so we can import dodhalueval
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

# Load environment variables
# load_dotenv(project_root / ".env")  # Removed for CI/CD compatibility

from dodhalueval.data.pdf_processor import PDFProcessor
from dodhalueval.core import (
    PromptGenerator, 
    LLMPromptGenerator, 
    PromptPerturbator, 
    PromptValidator
)
from dodhalueval.providers import OpenAIProvider, FireworksProvider, MockLLMProvider
from dodhalueval.models.config import (
    PDFProcessingConfig, 
    PromptGenerationConfig, 
    APIConfig
)
from dodhalueval.models.schemas import DocumentChunk, Prompt
from dodhalueval.utils.logger import get_logger

logger = get_logger(__name__)


class PromptGenerationTester:
    """Consolidated tester for all prompt generation approaches."""
    
    def __init__(self, provider_type: str = "mock", max_prompts: int = 10, verbose: bool = False, output_dir: str = "test_outputs"):
        self.provider_type = provider_type
        self.max_prompts = max_prompts
        self.verbose = verbose
        self.project_root = project_root
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Timing and statistics tracking
        self.start_time = time.time()
        self.stage_times = {}
        self.all_generated_prompts = []
        
        # Initialize configurations
        self.prompt_config = PromptGenerationConfig(
            template_file=str(project_root / 'data' / 'prompts' / 'templates.yaml'),
            max_prompts_per_document=max_prompts,
            perturbation_enabled=True
        )
        
        # Initialize components
        self.template_generator = PromptGenerator(self.prompt_config)
        self.perturbator = PromptPerturbator()
        self.validator = PromptValidator()
        
        # Initialize LLM provider
        self.llm_provider = self._create_llm_provider()
        self.llm_generator = LLMPromptGenerator(self.llm_provider, self.prompt_config) if self.llm_provider else None
    
    def _create_llm_provider(self):
        """Create LLM provider based on type and available API keys."""
        try:
            if self.provider_type == "openai":
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    self._print("❌ OPENAI_API_KEY not found in environment")
                    return None
                
                config = APIConfig(
                    provider='openai',
                    model=os.getenv('DEFAULT_MODEL_OPENAI', 'gpt-4'),
                    api_key=api_key,
                    max_retries=3,
                    timeout=30
                )
                return OpenAIProvider(config)
            
            elif self.provider_type == "fireworks":
                api_key = os.getenv('FIREWORKS_API_KEY')
                if not api_key:
                    self._print("❌ FIREWORKS_API_KEY not found in environment")
                    return None
                
                config = APIConfig(
                    provider='fireworks',
                    model=os.getenv('DEFAULT_MODEL_FIREWORKS', 'accounts/fireworks/models/llama-v3p1-8b-instruct'),
                    api_key=api_key,
                    max_retries=3,
                    timeout=30
                )
                return FireworksProvider(config)
            
            else:  # mock
                config = APIConfig(provider='mock', model='mock-gpt-4')
                return MockLLMProvider(config)
        
        except Exception as e:
            self._print(f"❌ Error creating {self.provider_type} provider: {e}")
            return None
    
    def _print(self, message: str, level: str = "info"):
        """Print message with optional verbose control."""
        if level == "verbose" and not self.verbose:
            return
        print(message)
    
    def _time_stage(self, stage_name: str):
        """Record timing for a processing stage."""
        current_time = time.time()
        self.stage_times[stage_name] = current_time - self.start_time
        self.start_time = current_time
    
    def _save_prompts(self, prompts: List[Any], filename: str, metadata: Dict[str, Any] = None):
        """Save prompts to JSON file with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"{filename}_{timestamp}.json"
        
        # Convert prompts to serializable format
        prompt_data = []
        for prompt in prompts:
            prompt_dict = {
                "text": prompt.text,
                "source_document_id": prompt.source_document_id,
                "expected_answer": prompt.expected_answer,
                "hallucination_type": prompt.hallucination_type,
                "generation_strategy": prompt.generation_strategy,
                "metadata": getattr(prompt, 'metadata', {})
            }
            prompt_data.append(prompt_dict)
        
        # Add metadata
        output_data = {
            "metadata": metadata or {},
            "timestamp": timestamp,
            "total_prompts": len(prompts),
            "prompts": prompt_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self._print(f"💾 Saved {len(prompts)} prompts to: {filepath}")
        return filepath
    
    def _analyze_prompt_distribution(self, prompts: List[Any]) -> Dict[str, Any]:
        """Analyze the distribution of prompts across different dimensions."""
        analysis = {
            "total_prompts": len(prompts),
            "by_hallucination_type": defaultdict(int),
            "by_generation_strategy": defaultdict(int),
            "by_source_document": defaultdict(int),
            "average_length": 0,
            "length_distribution": {"short": 0, "medium": 0, "long": 0}
        }
        
        if not prompts:
            return analysis
        
        total_length = 0
        for prompt in prompts:
            # Count by hallucination type
            h_type = prompt.hallucination_type or 'unknown'
            analysis["by_hallucination_type"][h_type] += 1
            
            # Count by generation strategy
            g_strategy = prompt.generation_strategy or 'unknown'
            analysis["by_generation_strategy"][g_strategy] += 1
            
            # Count by source document
            doc_id = prompt.source_document_id or 'unknown'
            analysis["by_source_document"][doc_id] += 1
            
            # Length analysis
            length = len(prompt.text)
            total_length += length
            
            if length < 50:
                analysis["length_distribution"]["short"] += 1
            elif length < 150:
                analysis["length_distribution"]["medium"] += 1
            else:
                analysis["length_distribution"]["long"] += 1
        
        analysis["average_length"] = total_length / len(prompts)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        for key in ["by_hallucination_type", "by_generation_strategy", "by_source_document"]:
            analysis[key] = dict(analysis[key])
        
        return analysis
    
    def _print_prompt_summary(self, prompts: List[Any], title: str = "Prompt Summary"):
        """Print a comprehensive summary of generated prompts."""
        if not prompts:
            self._print(f"❌ No prompts generated for {title}")
            return
        
        analysis = self._analyze_prompt_distribution(prompts)
        
        self._print(f"\n📊 {title.upper()}")
        self._print("=" * 50)
        self._print(f"Total prompts: {analysis['total_prompts']}")
        self._print(f"Average length: {analysis['average_length']:.1f} characters")
        
        # Hallucination type distribution
        self._print("\nBy Hallucination Type:")
        for h_type, count in analysis["by_hallucination_type"].items():
            percentage = (count / analysis['total_prompts']) * 100
            self._print(f"  {h_type}: {count} ({percentage:.1f}%)")
        
        # Generation strategy distribution
        self._print("\nBy Generation Strategy:")
        for strategy, count in analysis["by_generation_strategy"].items():
            percentage = (count / analysis['total_prompts']) * 100
            self._print(f"  {strategy}: {count} ({percentage:.1f}%)")
        
        # Source document distribution (important for balance)
        if len(analysis["by_source_document"]) > 1:
            self._print("\nBy Source Document (Balance Check):")
            for doc_id, count in analysis["by_source_document"].items():
                percentage = (count / analysis['total_prompts']) * 100
                self._print(f"  {doc_id}: {count} ({percentage:.1f}%)")
        
        # Length distribution
        self._print("\nBy Length:")
        for length_cat, count in analysis["length_distribution"].items():
            percentage = (count / analysis['total_prompts']) * 100
            self._print(f"  {length_cat}: {count} ({percentage:.1f}%)")
        
        # Show sample prompts if verbose
        if self.verbose and prompts:
            self._print("\nSample Prompts:", "verbose")
            for i, prompt in enumerate(prompts[:3]):
                self._print(f"  {i+1}. [{prompt.hallucination_type}] {prompt.text[:80]}...", "verbose")
    
    def _create_sample_content(self) -> DocumentChunk:
        """Create rich sample military content for testing."""
        content = """
        CHAPTER 7: ADVANCED COMBAT SYSTEMS OPERATIONS
        
        The M1A2 Abrams main battle tank represents the pinnacle of armored warfare technology. 
        Each tank requires a crew of four specialists: tank commander, gunner, loader, and driver. 
        The primary armament consists of a 120mm M256A1 smoothbore cannon capable of engaging 
        targets at ranges exceeding 4,000 meters with high-explosive anti-tank (HEAT) rounds.
        
        Standard operating procedures mandate pre-combat checks every 12 hours during extended 
        operations. The inspection protocol includes 52 critical systems: engine performance, 
        track tension (optimal range 15-20 inches), hydraulic fluid levels, fire control systems, 
        and communications equipment functionality.
        
        During offensive operations, tank platoons advance in echelon formation with 100-meter 
        intervals between vehicles. The lead element maintains overwatch while supporting tanks 
        advance by bounds. Rules of engagement authorize engagement of hostile armor at maximum 
        effective range using depleted uranium penetrator rounds.
        
        Logistical requirements include Class III POL (petroleum, oils, lubricants) consumption 
        of approximately 2.1 gallons per mile, Class V ammunition loads of 42 main gun rounds, 
        and Class IX maintenance parts with 95% availability requirements.
        """
        
        return DocumentChunk(
            document_id="sample-combat-systems",
            content=content.strip(),
            page_number=7,
            chunk_index=0,
            section="Chapter 7: Advanced Combat Systems Operations"
        )
    
    async def quick_test(self) -> Dict[str, Any]:
        """Quick functionality test."""
        self._print("\n🚀 QUICK TEST MODE")
        self._print("=" * 50)
        
        start_time = time.time()
        
        # Use sample content
        chunk = self._create_sample_content()
        self._print(f"📄 Using sample content: {chunk.word_count} words")
        
        # Generate prompts
        prompts = self.template_generator.generate_from_chunks([chunk])
        self._print(f"✅ Generated {len(prompts)} template prompts")
        
        # Add to collection and save
        self.all_generated_prompts.extend(prompts)
        
        metadata = {
            "mode": "quick",
            "processing_time_seconds": time.time() - start_time,
            "content_words": chunk.word_count
        }
        
        if prompts:
            self._save_prompts(prompts, "quick_test_prompts", metadata)
            self._print_prompt_summary(prompts, "Quick Test Results")
        
        return {
            "mode": "quick",
            "prompts_generated": len(prompts),
            "content_words": chunk.word_count,
            "processing_time_seconds": metadata["processing_time_seconds"]
        }
    
    async def demo_test(self) -> Dict[str, Any]:
        """Comprehensive demo with rich content."""
        self._print("\n🎯 DEMO MODE - Full Capabilities")
        self._print("=" * 50)
        
        chunk = self._create_sample_content()
        self._print(f"📄 Processing content: {chunk.word_count} words")
        
        # Step 1: Template generation
        self._print("\n🔧 Step 1: Template-based Generation")
        prompts = self.template_generator.generate_from_chunks([chunk])
        
        # Analyze distribution
        by_type = {}
        for prompt in prompts:
            ptype = prompt.hallucination_type or 'unknown'
            by_type[ptype] = by_type.get(ptype, 0) + 1
        
        self._print(f"✅ Generated {len(prompts)} prompts:")
        for ptype, count in sorted(by_type.items()):
            self._print(f"   {ptype}: {count}")
        
        # Step 2: Perturbation
        self._print("\n🔄 Step 2: Perturbation Strategies")
        all_variations = []
        
        if prompts:
            strategies = ['entity_substitution', 'numerical_manipulation', 'negation_injection']
            for strategy in strategies:
                variations = self.perturbator.perturb(prompts[0], strategy, chunk)
                new_variations = [v for v in variations if v.text != prompts[0].text]
                all_variations.extend(new_variations)
                self._print(f"   {strategy}: {len(new_variations)} variations")
        
        # Step 3: LLM generation (if available)
        llm_prompts = []
        if self.llm_generator:
            self._print(f"\n🤖 Step 3: LLM Generation ({self.provider_type})")
            try:
                llm_prompts = await self.llm_generator.generate_hallucination_prone_prompts(
                    chunk.content, chunk, num_prompts=3, strategy='factual_probing'
                )
                self._print(f"✅ Generated {len(llm_prompts)} LLM prompts")
            except Exception as e:
                self._print(f"❌ LLM generation failed: {e}")
        
        # Step 4: Validation
        self._print("\n✅ Step 4: Quality Validation")
        all_prompts = prompts + all_variations + llm_prompts
        
        validation_results = None
        if all_prompts:
            chunks_for_validation = [chunk] * len(all_prompts)
            validation_results = await self.validator.validate_batch(
                all_prompts, chunks_for_validation, use_llm_validation=False
            )
            
            valid_count = sum(1 for r in validation_results if r.is_valid)
            scores = [r.score for r in validation_results]
            
            self._print(f"   Valid: {valid_count}/{len(all_prompts)} ({valid_count/len(all_prompts)*100:.1f}%)")
            self._print(f"   Avg Score: {sum(scores)/len(scores):.2f}")
            self._print(f"   Best Score: {max(scores):.2f}")
        
        # Add to collection and save
        self.all_generated_prompts.extend(all_prompts)
        
        metadata = {
            "mode": "demo",
            "content_words": chunk.word_count,
            "template_prompts": len(prompts),
            "perturbation_variations": len(all_variations),
            "llm_prompts": len(llm_prompts),
            "validation_performed": validation_results is not None,
            "validation_stats": {
                "valid_count": sum(1 for r in validation_results if r.is_valid) if validation_results else 0,
                "avg_score": sum(r.score for r in validation_results) / len(validation_results) if validation_results else 0,
                "max_score": max(r.score for r in validation_results) if validation_results else 0
            } if validation_results else None
        }
        
        if all_prompts:
            self._save_prompts(all_prompts, "demo_test_prompts", metadata)
            self._print_prompt_summary(all_prompts, "Demo Test Results")
        
        return {
            "mode": "demo",
            "template_prompts": len(prompts),
            "perturbation_variations": len(all_variations),
            "llm_prompts": len(llm_prompts),
            "total_prompts": len(all_prompts),
            "validation_performed": len(all_prompts) > 0
        }
    
    async def csc_test(self) -> Dict[str, Any]:
        """Test with actual CSC PDF data with balanced sampling."""
        self._print("\n📚 CSC DATA TEST MODE")
        self._print("=" * 50)
        
        start_time = time.time()
        
        csc_dir = self.project_root / "data" / "CSC"
        pdf_files = list(csc_dir.glob("*.pdf"))
        
        if not pdf_files:
            self._print("❌ No PDF files found in data/CSC")
            return {"mode": "csc", "error": "No PDF files found"}
        
        self._print(f"📄 Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            size_mb = pdf.stat().st_size / (1024 * 1024)
            self._print(f"   - {pdf.name} ({size_mb:.1f}MB)")
        
        # Process first PDF with timing
        test_pdf = pdf_files[0]
        self._print(f"\n🔄 Processing: {test_pdf.name}")
        
        try:
            # Configure processor with balanced sampling
            max_pages = int(os.getenv('MAX_PAGES_FOR_TESTING', '5'))  # Keep at 5 for speed
            processor = PDFProcessor(
                chunk_size=int(os.getenv('CHUNK_SIZE', '800')),
                chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '100')),
                cache_enabled=True,
                max_pages=max_pages
            )
            
            pdf_start = time.time()
            self._print(f"🕐 Starting PDF processing (max {max_pages} pages)...")
            
            result = processor.process_document(test_pdf)
            chunks = result.get('chunks', [])
            
            pdf_time = time.time() - pdf_start
            self._print(f"✅ PDF processing completed in {pdf_time:.2f}s")
            self._print(f"✅ Extracted {len(chunks)} chunks")
            
            if not chunks:
                self._print("❌ No text chunks extracted")
                return {"mode": "csc", "error": "No chunks extracted"}
            
            # Ensure balanced sampling across document
            # Instead of just taking first 3 chunks, sample evenly
            if len(chunks) > 6:
                # Sample from beginning, middle, and end for balance
                chunk_indices = [
                    0,  # Beginning
                    len(chunks) // 4,  # Early middle
                    len(chunks) // 2,  # Middle
                    3 * len(chunks) // 4,  # Late middle
                    len(chunks) - 1  # End
                ]
                sampled_chunks = [chunks[i] for i in chunk_indices if i < len(chunks)]
            else:
                sampled_chunks = chunks[:6]  # Use more chunks for small documents
            
            self._print(f"🎯 Selected {len(sampled_chunks)} chunks for balanced sampling")
            
            # Generate prompts with timing
            prompt_start = time.time()
            prompts = self.template_generator.generate_from_chunks(sampled_chunks)
            prompt_time = time.time() - prompt_start
            
            self._print(f"✅ Generated {len(prompts)} prompts in {prompt_time:.2f}s")
            
            # Add to collection and save
            self.all_generated_prompts.extend(prompts)
            
            metadata = {
                "mode": "csc",
                "pdf_file": test_pdf.name,
                "pdf_size_mb": test_pdf.stat().st_size / (1024 * 1024),
                "max_pages_processed": max_pages,
                "chunks_extracted": len(chunks),
                "chunks_used": len(sampled_chunks),
                "processing_time_seconds": time.time() - start_time,
                "pdf_processing_time_seconds": pdf_time,
                "prompt_generation_time_seconds": prompt_time
            }
            
            if prompts:
                self._save_prompts(prompts, "csc_test_prompts", metadata)
                self._print_prompt_summary(prompts, "CSC Test Results")
                
                # Check document balance
                doc_balance = self._analyze_prompt_distribution(prompts)
                if len(doc_balance["by_source_document"]) > 1:
                    self._print("\n⚙️ Document Balance Analysis:")
                    for doc_id, count in doc_balance["by_source_document"].items():
                        percentage = (count / len(prompts)) * 100
                        self._print(f"   {doc_id}: {count} prompts ({percentage:.1f}%)")
            
            return {
                "mode": "csc",
                "pdf_processed": test_pdf.name,
                "chunks_extracted": len(chunks),
                "chunks_used": len(sampled_chunks),
                "prompts_generated": len(prompts),
                "processing_time_seconds": metadata["processing_time_seconds"],
                "pdf_processing_time_seconds": pdf_time
            }
            
        except Exception as e:
            self._print(f"❌ Error processing CSC data: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return {"mode": "csc", "error": str(e), "processing_time_seconds": time.time() - start_time}
    
    async def llm_test(self) -> Dict[str, Any]:
        """Test LLM-based generation with real providers."""
        self._print(f"\n🤖 LLM TEST MODE - {self.provider_type.upper()}")
        self._print("=" * 50)
        
        if not self.llm_generator:
            self._print("❌ No LLM provider available")
            return {"mode": "llm", "error": "No LLM provider"}
        
        chunk = self._create_sample_content()
        
        # Test different strategies
        strategies = [
            ('factual_probing', 3),
            ('logical_reasoning', 2),
            ('adversarial', 2)
        ]
        
        all_llm_prompts = []
        strategy_results = {}
        
        for strategy, num_prompts in strategies:
            self._print(f"\n🔄 Testing {strategy}...")
            
            try:
                prompts = await self.llm_generator.generate_hallucination_prone_prompts(
                    chunk.content, chunk, num_prompts=num_prompts, strategy=strategy
                )
                
                all_llm_prompts.extend(prompts)
                strategy_results[strategy] = len(prompts)
                self._print(f"✅ Generated {len(prompts)} {strategy} prompts")
                
                if self.verbose and prompts:
                    self._print(f"   Sample: {prompts[0].text[:80]}...", "verbose")
                
            except Exception as e:
                self._print(f"❌ {strategy} failed: {e}")
                strategy_results[strategy] = 0
        
        # Test cost estimation
        if self.provider_type in ["openai", "fireworks"]:
            self._print(f"\n💰 Cost Estimation")
            try:
                sample_prompts = [p.text for p in all_llm_prompts[:5]]
                cost_estimate = await self.llm_provider.estimate_cost(sample_prompts)
                self._print(f"   Estimated cost: ${cost_estimate.get('estimated_total_cost_usd', 0):.4f}")
            except Exception as e:
                self._print(f"❌ Cost estimation failed: {e}")
        
        # Add to collection and save
        self.all_generated_prompts.extend(all_llm_prompts)
        
        metadata = {
            "mode": "llm",
            "provider": self.provider_type,
            "content_words": chunk.word_count,
            "strategy_results": strategy_results,
            "total_strategies_tested": len(strategies)
        }
        
        if all_llm_prompts:
            self._save_prompts(all_llm_prompts, "llm_test_prompts", metadata)
            self._print_prompt_summary(all_llm_prompts, "LLM Test Results")
        
        return {
            "mode": "llm",
            "provider": self.provider_type,
            "total_prompts": len(all_llm_prompts),
            "strategy_results": strategy_results
        }
    
    async def comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests comprehensively."""
        self._print("\n🎯 COMPREHENSIVE TEST MODE - All Capabilities")
        self._print("=" * 60)
        
        results = {}
        
        # Run all test modes
        self._print("Running quick test...")
        results["quick"] = await self.quick_test()
        
        self._print("\nRunning demo test...")
        results["demo"] = await self.demo_test()
        
        self._print("\nRunning CSC test...")
        results["csc"] = await self.csc_test()
        
        if self.provider_type != "mock":
            self._print("\nRunning LLM test...")
            results["llm"] = await self.llm_test()
        
        # Summary
        self._print("\n" + "=" * 60)
        self._print("🎉 COMPREHENSIVE TEST SUMMARY")
        self._print("=" * 60)
        
        total_prompts = 0
        for test_name, test_results in results.items():
            if isinstance(test_results, dict) and 'prompts_generated' in test_results:
                prompts = test_results['prompts_generated']
                total_prompts += prompts
                self._print(f"✅ {test_name.upper()}: {prompts} prompts")
            elif isinstance(test_results, dict) and 'total_prompts' in test_results:
                prompts = test_results['total_prompts']
                total_prompts += prompts
                self._print(f"✅ {test_name.upper()}: {prompts} prompts")
        
        self._print(f"\n🎯 TOTAL PROMPTS GENERATED: {total_prompts}")
        
        results["summary"] = {
            "total_prompts": total_prompts,
            "tests_completed": len(results)
        }
        
        return results
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.llm_provider:
            await self.llm_provider.close()


async def run_tests(mode: str, provider: str, max_prompts: int, validate: bool, verbose: bool, output_dir: str = "test_outputs"):
    """
    DoDHaluEval Prompt Generation Test Suite
    
    Test different approaches for generating hallucination-prone prompts
    from military/defense documents.
    """
    
    # Print header
    print("🚀 DoDHaluEval Prompt Generation Test Suite")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Provider: {provider}")
    print(f"Max Prompts: {max_prompts}")
    print(f"Validation: {'Enabled' if validate else 'Disabled'}")
    print(f"Verbose: {'Enabled' if verbose else 'Disabled'}")
    
    # Check environment
    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        print("❌ .env file not found. Please ensure API keys are configured.")
        return
    
    print(f"✅ Environment loaded from {env_file}")
    
    # Initialize tester
    tester = PromptGenerationTester(
        provider_type=provider,
        max_prompts=max_prompts,
        verbose=verbose,
        output_dir=output_dir
    )
    
    try:
        # Run appropriate test mode
        if mode == 'quick':
            results = await tester.quick_test()
        elif mode == 'demo':
            results = await tester.demo_test()
        elif mode == 'csc':
            results = await tester.csc_test()
        elif mode == 'llm':
            results = await tester.llm_test()
        elif mode == 'all':
            results = await tester.comprehensive_test()
        
        print(f"\n✅ {mode.upper()} test completed successfully!")
        
        # Always show timing summary
        if 'processing_time_seconds' in results:
            print(f"⏱️ Total processing time: {results['processing_time_seconds']:.2f}s")
        
        # Show final summary of all prompts
        if tester.all_generated_prompts:
            print(f"\n📊 FINAL SUMMARY - ALL GENERATED PROMPTS")
            tester._print_prompt_summary(tester.all_generated_prompts, "All Generated Prompts")
            
            # Save comprehensive results
            final_metadata = {
                "test_mode": mode,
                "provider": provider,
                "max_prompts": max_prompts,
                "validation_enabled": validate,
                "total_processing_time": results.get('processing_time_seconds', 0),
                "individual_results": results
            }
            tester._save_prompts(tester.all_generated_prompts, f"{mode}_final_results", final_metadata)
        
        if verbose:
            print(f"\n📊 Results: {results}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
    
    finally:
        await tester.cleanup()


@click.command()
@click.option('--mode', 
              type=click.Choice(['quick', 'demo', 'csc', 'llm', 'all']), 
              default='quick',
              help='Test mode to run')
@click.option('--provider', 
              type=click.Choice(['openai', 'fireworks', 'mock']), 
              default='mock',
              help='LLM provider to use')
@click.option('--max-prompts', 
              type=int, 
              default=10,
              help='Maximum prompts to generate')
@click.option('--validate', 
              is_flag=True,
              help='Run prompt validation')
@click.option('--verbose', 
              is_flag=True,
              help='Enable verbose output')
@click.option('--output-dir',
              type=str,
              default='test_outputs',
              help='Directory to save results (default: test_outputs)')
def main(mode: str, provider: str, max_prompts: int, validate: bool, verbose: bool, output_dir: str):
    """
    DoDHaluEval Prompt Generation Test Suite
    
    Test different approaches for generating hallucination-prone prompts
    from military/defense documents.
    """
    asyncio.run(run_tests(mode, provider, max_prompts, validate, verbose, output_dir))


if __name__ == "__main__":
    main()