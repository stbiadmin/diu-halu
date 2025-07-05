#!/usr/bin/env python3
"""
Test script for Response Generation functionality.

This script tests the response generation system with different providers
and hallucination injection capabilities.
"""

import asyncio
import sys
import os
import json
import csv
import pytest
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add the src directory to the path so we can import dodhalueval
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

# Dotenv import removed for CI/CD compatibility
# from dotenv import load_dotenv
# load_dotenv(project_root / ".env")

from dodhalueval.core import ResponseGenerator, ResponseConfig, HallucinationInjector
from dodhalueval.providers import OpenAIProvider, FireworksProvider, MockLLMProvider
from dodhalueval.models.schemas import Prompt
from dodhalueval.models.config import APIConfig
from dodhalueval.utils.logger import get_logger

logger = get_logger(__name__)

# Output directory for storing results
OUTPUT_DIR = Path("test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_sample_prompts() -> List[Prompt]:
    """Create sample prompts for testing."""
    prompts = [
        Prompt(
            text="What are the main specifications of the M1A2 Abrams tank?",
            source_document_id="doc_1",
            expected_answer="The M1A2 Abrams has a 120mm cannon, crew of 4, and weighs 62 tons.",
            hallucination_type="factual",
            generation_strategy="template_based"
        ),
        Prompt(
            text="Explain the tactical deployment procedures for armored units.",
            source_document_id="doc_1", 
            expected_answer="Armored units deploy in formation with overwatch and bounding movement.",
            hallucination_type="context",
            generation_strategy="template_based"
        ),
        Prompt(
            text="How many rounds does an M1A2 Abrams carry?",
            source_document_id="doc_1",
            expected_answer="The M1A2 Abrams carries 42 main gun rounds.",
            hallucination_type="factual",
            generation_strategy="template_based"
        )
    ]
    return prompts


def save_prompts_to_json(prompts: List[Prompt], filepath: Path):
    """Save prompts to JSON file."""
    prompt_data = []
    for prompt in prompts:
        prompt_data.append({
            "text": prompt.text,
            "source_document_id": prompt.source_document_id,
            "expected_answer": prompt.expected_answer,
            "hallucination_type": prompt.hallucination_type,
            "generation_strategy": prompt.generation_strategy
        })
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(prompt_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Prompts saved to: {filepath}")


def save_results_to_csv(responses: List[Any], filepath: Path):
    """Save response results to CSV file."""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Provider', 'Model', 'Prompt_ID', 'Response_Text', 
            'Contains_Hallucination', 'Hallucination_Types', 
            'Word_Count', 'Processing_Time'
        ])
        
        for response in responses:
            writer.writerow([
                response.provider,
                response.model,
                response.prompt_id,
                response.text[:200] + "..." if len(response.text) > 200 else response.text,
                response.contains_hallucination,
                "; ".join(response.hallucination_types) if response.hallucination_types else "",
                response.word_count,
                response.metadata.get('processing_time_seconds', 0)
            ])
    print(f"💾 Results saved to: {filepath}")


def print_results_summary(responses: List[Any]):
    """Print a formatted summary of results."""
    print("\n📊 RESULTS SUMMARY")
    print("=" * 50)
    
    total_responses = len(responses)
    hallucinated_responses = sum(1 for r in responses if r.contains_hallucination)
    
    print(f"Total responses: {total_responses}")
    print(f"Responses with hallucinations: {hallucinated_responses} ({hallucinated_responses/total_responses*100:.1f}%)")
    
    # Group by provider
    provider_stats = {}
    for response in responses:
        provider = response.provider
        if provider not in provider_stats:
            provider_stats[provider] = {'total': 0, 'hallucinated': 0, 'avg_time': 0}
        
        provider_stats[provider]['total'] += 1
        if response.contains_hallucination:
            provider_stats[provider]['hallucinated'] += 1
        provider_stats[provider]['avg_time'] += response.metadata.get('processing_time_seconds', 0)
    
    print("\nBy Provider:")
    for provider, stats in provider_stats.items():
        avg_time = stats['avg_time'] / stats['total']
        halluc_rate = stats['hallucinated'] / stats['total'] * 100
        print(f"  {provider}: {stats['total']} responses, {halluc_rate:.1f}% hallucinated, {avg_time:.2f}s avg")
    
    # Show hallucination types
    halluc_types = {}
    for response in responses:
        for h_type in response.hallucination_types:
            halluc_types[h_type] = halluc_types.get(h_type, 0) + 1
    
    if halluc_types:
        print("\nHallucination Types:")
        for h_type, count in halluc_types.items():
            print(f"  {h_type}: {count}")


def setup_providers() -> Dict[str, Any]:
    """Setup LLM providers for testing."""
    providers = {}
    
    # Always include mock provider
    mock_config = APIConfig(provider='mock', model='mock-gpt-4')
    providers['mock'] = MockLLMProvider(mock_config)
    
    # Add OpenAI if available
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        try:
            openai_config = APIConfig(
                provider='openai',
                model='gpt-4',
                api_key=openai_key,
                max_retries=2,
                timeout=30
            )
            providers['openai'] = OpenAIProvider(openai_config)
            print("✅ OpenAI provider configured")
        except Exception as e:
            print(f"❌ OpenAI provider failed: {e}")
    
    # Add Fireworks if available
    fireworks_key = os.getenv('FIREWORKS_API_KEY')
    if fireworks_key:
        try:
            fireworks_config = APIConfig(
                provider='fireworks',
                model=os.getenv('DEFAULT_MODEL_FIREWORKS', 'accounts/fireworks/models/llama-v3p1-8b-instruct'),
                api_key=fireworks_key,
                max_retries=2,
                timeout=30
            )
            providers['fireworks'] = FireworksProvider(fireworks_config)
            print("✅ Fireworks provider configured")
        except Exception as e:
            print(f"❌ Fireworks provider failed: {e}")
    
    return providers


@pytest.mark.asyncio
async def test_hallucination_injector():
    """Test the hallucination injection functionality."""
    print("\n🔬 Testing Hallucination Injector")
    print("=" * 50)
    
    injector = HallucinationInjector()
    
    original_response = """
    The M1A2 Abrams tank is equipped with a 120mm smoothbore cannon and has a crew of four specialists.
    It can engage targets at ranges exceeding 4000 meters using depleted uranium rounds.
    The tank follows standard operating procedures for maintenance checks every 12 hours.
    """
    
    # Test factual injection
    factual_result, factual_types = injector.inject_hallucinations(
        original_response, ['factual'], 1.0  # Force injection
    )
    
    print("📝 Original response:")
    print(f"   {original_response.strip()}")
    
    if factual_types:
        print(f"\n✅ Factual injection applied: {factual_types}")
        print("📝 Modified response:")
        print(f"   {factual_result.strip()}")
    else:
        print("❌ No factual injection occurred")
    
    # Test logical injection
    logical_result, logical_types = injector.inject_hallucinations(
        original_response, ['logical'], 1.0
    )
    
    if logical_types:
        print(f"\n✅ Logical injection applied: {logical_types}")
        print("📝 Modified response:")
        print(f"   {logical_result.strip()}")
    
    # Test context injection
    context_result, context_types = injector.inject_hallucinations(
        original_response, ['context'], 1.0
    )
    
    if context_types:
        print(f"\n✅ Context injection applied: {context_types}")
        print("📝 Modified response:")
        print(f"   {context_result.strip()}")


@pytest.mark.asyncio
async def test_response_generation():
    """Test basic response generation."""
    print("\n🤖 Testing Response Generation")
    print("=" * 50)
    
    # Setup
    providers = setup_providers()
    if not providers:
        print("❌ No providers available for testing")
        return
    
    config = ResponseConfig(
        hallucination_rate=0.5,  # 50% chance for testing
        concurrent_requests=3,
        request_delay=0.1
    )
    
    generator = ResponseGenerator(providers, config)
    prompts = create_sample_prompts()
    
    # Save prompts for reference
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompts_file = OUTPUT_DIR / f"test_prompts_{timestamp}.json"
    save_prompts_to_json(prompts, prompts_file)
    
    print(f"📊 Testing with {len(providers)} providers and {len(prompts)} prompts")
    
    # Test single response generation
    print("\n🔹 Testing single response...")
    first_prompt = prompts[0]
    provider_name = list(providers.keys())[0]
    
    try:
        response = await generator.generate_single_response(
            first_prompt, provider_name, inject_hallucination=True
        )
        
        print(f"✅ Single response generated:")
        print(f"   Provider: {response.provider}")
        print(f"   Model: {response.model}")
        print(f"   Contains hallucination: {response.contains_hallucination}")
        print(f"   Hallucination types: {response.hallucination_types}")
        print(f"   Response: {response.text[:100]}...")
        print(f"   Word count: {response.word_count}")
        
    except Exception as e:
        print(f"❌ Single response generation failed: {e}")
    
    # Test batch generation
    print("\n🔹 Testing batch generation...")
    try:
        responses = await generator.generate_responses(
            prompts[:2],  # Test with 2 prompts
            models=list(providers.keys())[:1],  # Use first provider
            hallucination_rate=0.3
        )
        
        print(f"✅ Batch generation completed: {len(responses)} responses")
        
        # Save results
        results_file = OUTPUT_DIR / f"test_results_{timestamp}.csv"
        save_results_to_csv(responses, results_file)
        
        # Print summary
        print_results_summary(responses)
        
        for i, response in enumerate(responses[:3]):  # Show first 3
            print(f"   Response {i+1}: {response.text[:60]}...")
            
    except Exception as e:
        print(f"❌ Batch generation failed: {e}")
    
    # Cleanup
    await generator.cleanup()


@pytest.mark.asyncio
async def test_batch_with_progress():
    """Test batch generation with progress callback."""
    print("\n📊 Testing Batch Generation with Progress")
    print("=" * 50)
    
    providers = setup_providers()
    if not providers:
        print("❌ No providers available")
        return
    
    generator = ResponseGenerator(providers)
    prompts = create_sample_prompts()
    
    # Progress callback
    def progress_callback(completed: int, total: int):
        percent = (completed / total) * 100
        print(f"   Progress: {completed}/{total} ({percent:.1f}%)")
    
    try:
        responses = await generator.generate_batch_with_progress(
            prompts,
            models=['mock'],  # Use mock for speed
            progress_callback=progress_callback
        )
        
        print(f"✅ Batch with progress completed: {len(responses)} responses")
        
        # Save batch results
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = OUTPUT_DIR / f"batch_results_{batch_timestamp}.csv"
        save_results_to_csv(responses, batch_file)
        print_results_summary(responses)
        
    except Exception as e:
        print(f"❌ Batch with progress failed: {e}")
    
    await generator.cleanup()


@pytest.mark.asyncio
async def test_multi_provider():
    """Test generation across multiple providers."""
    print("\n🌐 Testing Multi-Provider Generation")
    print("=" * 50)
    
    providers = setup_providers()
    available_providers = list(providers.keys())
    
    if len(available_providers) < 2:
        print("❌ Need at least 2 providers for multi-provider test")
        print(f"   Available: {available_providers}")
        return
    
    generator = ResponseGenerator(providers)
    prompt = create_sample_prompts()[0]
    
    print(f"🔄 Generating responses across {len(available_providers)} providers...")
    
    try:
        responses = await generator.generate_responses(
            [prompt],
            models=available_providers,
            hallucination_rate=0.0  # No hallucination for comparison
        )
        
        print(f"✅ Multi-provider generation completed: {len(responses)} responses")
        
        # Save multi-provider results
        multi_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        multi_file = OUTPUT_DIR / f"multi_provider_results_{multi_timestamp}.csv"
        save_results_to_csv(responses, multi_file)
        print_results_summary(responses)
        
        # Compare responses
        for response in responses:
            print(f"\n📝 {response.provider} ({response.model}):")
            print(f"   {response.text[:80]}...")
            print(f"   Processing time: {response.metadata.get('processing_time_seconds', 0):.2f}s")
            
    except Exception as e:
        print(f"❌ Multi-provider generation failed: {e}")
    
    await generator.cleanup()


async def main():
    """Run all response generation tests."""
    print("🚀 DoDHaluEval Response Generation Test Suite")
    print("=" * 60)
    
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    print(f"🕐 Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test sequence
    await test_hallucination_injector()
    await test_response_generation()
    await test_batch_with_progress()
    await test_multi_provider()
    
    print("\n🎉 All tests completed!")
    print(f"📁 Check {OUTPUT_DIR} for saved prompts and results")


if __name__ == "__main__":
    asyncio.run(main())