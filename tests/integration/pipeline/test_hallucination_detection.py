#!/usr/bin/env python3
"""
Test script for Hallucination Detection functionality.

This script tests the hallucination detection system including:
- HuggingFace HHEM integration
- G-Eval implementation
- SelfCheckGPT implementation
- Ensemble evaluation orchestrator
"""

import asyncio
import sys
import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add the src directory to the path so we can import dodhalueval
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

# Load environment variables manually
def load_env_file():
    """Simple .env file loader."""
    env_path = project_root / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env_file()

from dodhalueval.core.hallucination_detector import HallucinationDetector
from dodhalueval.core.evaluators import (
    HuggingFaceHHEMEvaluator, 
    GEvalEvaluator, 
    SelfCheckGPTEvaluator
)
from dodhalueval.core.response_generator import ResponseGenerator, ResponseConfig
from dodhalueval.providers.openai_provider import OpenAIProvider
from dodhalueval.providers import MockLLMProvider
from dodhalueval.models.schemas import Prompt, Response
from dodhalueval.models.config import APIConfig
from dodhalueval.utils.logger import get_logger

logger = get_logger(__name__)

# Output directory for storing results
OUTPUT_DIR = Path("test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_test_prompts() -> List[Prompt]:
    """Create test prompts with known hallucination types."""
    return [
        Prompt(
            id="prompt_1",
            text="What are the specifications of the M1A2 Abrams tank?",
            source_document_id="military_doc_1",
            expected_answer="120mm cannon, crew of 4, weighs 62 tons",
            hallucination_type="factual",
            generation_strategy="template_based"
        ),
        Prompt(
            id="prompt_2",
            text="Explain the logical reasoning behind armored unit deployment.",
            source_document_id="military_doc_1", 
            expected_answer="Tactical positioning for maximum effectiveness",
            hallucination_type="logical",
            generation_strategy="template_based"
        ),
        Prompt(
            id="prompt_3",
            text="How do military communication protocols work?",
            source_document_id="military_doc_1",
            expected_answer="Secure channels with authentication",
            hallucination_type="context",
            generation_strategy="template_based"
        )
    ]


def create_test_responses() -> List[Response]:
    """Create test responses with varying levels of hallucination."""
    return [
        # Factually accurate response
        Response(
            id="response_1",
            prompt_id="prompt_1",
            text="The M1A2 Abrams tank has a 120mm smoothbore cannon, requires a crew of 4 specialists, and weighs approximately 62 tons.",
            model="test-model",
            provider="mock",
            contains_hallucination=False,
            hallucination_types=[]
        ),
        # Response with factual hallucinations
        Response(
            id="response_2",
            prompt_id="prompt_1", 
            text="The M1A2 Abrams tank has a 150mm cannon, requires a crew of 6 specialists, and weighs approximately 85 tons.",
            model="test-model",
            provider="mock",
            contains_hallucination=True,
            hallucination_types=["factual"]
        ),
        # Response with logical issues
        Response(
            id="response_3",
            prompt_id="prompt_2",
            text="Armored units should be deployed for maximum stealth, which is why they use the heaviest possible armor while remaining completely invisible.",
            model="test-model",
            provider="mock",
            contains_hallucination=True,
            hallucination_types=["logical"]
        ),
        # Response with context mixing
        Response(
            id="response_4",
            prompt_id="prompt_3",
            text="Military communication follows civilian air traffic control protocols, using commercial radio frequencies for all tactical operations.",
            model="test-model",
            provider="mock",
            contains_hallucination=True,
            hallucination_types=["context"]
        )
    ]


def save_evaluation_results(results, filename: str):
    """Save evaluation results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = OUTPUT_DIR / f"{filename}_{timestamp}.json"
    
    # Convert results to serializable format
    serializable_results = []
    for result in results:
        if isinstance(result, dict):
            # Already a dictionary
            result_data = result
        elif hasattr(result, 'individual_results'):  # EnsembleResult
            result_data = {
                "ensemble_score": result.ensemble_score,
                "ensemble_confidence": result.ensemble_confidence,
                "consensus_level": result.consensus_level,
                "explanation": result.explanation,
                "individual_evaluations": [
                    {
                        "evaluator_name": r.evaluator_name,
                        "score": r.hallucination_score.score,
                        "confidence": r.hallucination_score.confidence,
                        "reasoning": r.hallucination_score.explanation
                    }
                    for r in result.individual_results
                ],
                "metadata": result.metadata
            }
        else:  # Individual EvaluationResult
            result_data = {
                "evaluator_name": result.evaluator_name,
                "score": result.hallucination_score.score,
                "confidence": result.hallucination_score.confidence,
                "reasoning": result.hallucination_score.explanation,
                "metadata": result.metadata
            }
        
        serializable_results.append(result_data)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Evaluation results saved to: {filepath}")
    return filepath


async def test_huggingface_hhem_evaluator():
    """Test HuggingFace HHEM evaluator."""
    print("\n🤗 Testing HuggingFace HHEM Evaluator")
    print("=" * 50)
    
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    evaluator = HuggingFaceHHEMEvaluator(hf_token=hf_token)
    prompts = create_test_prompts()
    responses = create_test_responses()
    
    # Test single evaluation
    print("🔹 Testing single evaluation...")
    try:
        result = await evaluator.evaluate(
            text=responses[0].text,
            reference_text="The M1A2 Abrams tank specifications include a 120mm cannon and crew of 4."
        )
        
        print(f"✅ HuggingFace HHEM evaluation completed:")
        print(f"   Score: {result.hallucination_score.score:.3f}")
        print(f"   Confidence: {result.hallucination_score.confidence:.3f}")
        print(f"   Reasoning: {result.hallucination_score.explanation}")
        print(f"   Metadata: {result.metadata}")
        
    except Exception as e:
        print(f"❌ HuggingFace HHEM single evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test batch-like evaluation
    print("\n🔹 Testing multiple evaluations...")
    try:
        results = []
        for i, (response, prompt) in enumerate(zip(responses[:2], prompts[:2])):
            result = await evaluator.evaluate(
                text=response.text,
                reference_text=prompt.expected_answer
            )
            results.append(result)
            print(f"   Result {i+1}: score={result.hallucination_score.score:.3f}")
        
        print(f"✅ HuggingFace HHEM multiple evaluations completed: {len(results)} results")
            
        # Save results - the results are already EvaluationResult objects
        save_evaluation_results(results, "huggingface_hhem_test_results")
        
    except Exception as e:
        print(f"❌ HuggingFace HHEM multiple evaluations failed: {e}")
        import traceback
        traceback.print_exc()


async def test_g_eval_evaluator():
    """Test G-Eval evaluator."""
    print("\n🤖 Testing G-Eval Evaluator")
    print("=" * 50)
    
    # Setup LLM provider
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY not found - using mock provider")
        config = APIConfig(provider='mock', model='mock-gpt-4')
        llm_provider = MockLLMProvider(config)
    else:
        config = APIConfig(
            provider='openai',
            model='gpt-4',
            api_key=openai_key,
            max_retries=2,
            timeout=30
        )
        llm_provider = OpenAIProvider(config)
    
    evaluator = GEvalEvaluator(llm_provider)
    prompts = create_test_prompts()
    responses = create_test_responses()
    
    # Test single evaluation
    print("🔹 Testing single evaluation...")
    try:
        result = await evaluator.evaluate_single(
            responses[1], prompts[0],  # Use hallucinated response
            source_text="The M1A2 Abrams tank has a 120mm cannon, crew of 4, and weighs 62 tons."
        )
        
        print(f"✅ G-Eval evaluation completed:")
        print(f"   Score: {result.hallucination_score.score:.3f}")
        print(f"   Confidence: {result.hallucination_score.confidence:.3f}")
        print(f"   Explanation: {result.hallucination_score.explanation[:100]}...")
        print(f"   Evaluation type: {result.metadata.get('evaluation_type', 'unknown')}")
        
    except Exception as e:
        print(f"❌ G-Eval single evaluation failed: {e}")
    
    # Test with different hallucination types
    print("\n🔹 Testing different hallucination types...")
    test_cases = [
        (responses[1], prompts[0], "factual"),  # Factual hallucination
        (responses[2], prompts[1], "logical"),  # Logical hallucination
        (responses[3], prompts[2], "context")   # Context hallucination
    ]
    
    results = []
    for response, prompt, expected_type in test_cases:
        try:
            result = await evaluator.evaluate_single(response, prompt)
            results.append(result)
            print(f"   {expected_type}: score={result.hallucination_score.score:.3f}")
        except Exception as e:
            print(f"   {expected_type}: failed - {e}")
    
    if results:
        save_evaluation_results(results, "g_eval_test_results")
    
    await evaluator.cleanup()


async def test_selfcheck_evaluator():
    """Test SelfCheckGPT evaluator."""
    print("\n🔄 Testing SelfCheckGPT Evaluator")
    print("=" * 50)
    
    # Setup LLM provider (use mock for speed)
    config = APIConfig(provider='mock', model='mock-gpt-4')
    llm_provider = MockLLMProvider(config)
    
    evaluator = SelfCheckGPTEvaluator(
        llm_provider, 
        num_samples=3,  # Reduced for testing speed
        temperature=0.8
    )
    
    prompts = create_test_prompts()
    responses = create_test_responses()
    
    # Test single evaluation
    print("🔹 Testing single evaluation...")
    try:
        result = await evaluator.evaluate_single(
            responses[0], prompts[0],
            source_text="The M1A2 Abrams tank specifications."
        )
        
        print(f"✅ SelfCheckGPT evaluation completed:")
        print(f"   Score: {result.hallucination_score.score:.3f}")
        print(f"   Confidence: {result.hallucination_score.confidence:.3f}")
        print(f"   Explanation: {result.hallucination_score.explanation}")
        print(f"   Samples generated: {result.metadata.get('num_samples', 0)}")
        
        # Save individual result
        save_evaluation_results([result], "selfcheck_test_results")
        
    except Exception as e:
        print(f"❌ SelfCheckGPT evaluation failed: {e}")
    
    await evaluator.cleanup()


async def test_ensemble_detector():
    """Test the ensemble hallucination detector."""
    print("\n🎯 Testing Ensemble Hallucination Detector")
    print("=" * 50)
    
    # Setup LLM provider
    config = APIConfig(provider='mock', model='mock-gpt-4')
    llm_provider = MockLLMProvider(config)
    
    # Initialize detector with all evaluators
    detector = HallucinationDetector(
        llm_provider=llm_provider,
        enable_huggingface_hhem=True,
        enable_g_eval=True,
        enable_selfcheck=True,
        ensemble_method="weighted_average"
    )
    
    # Print detector info
    info = detector.get_evaluator_info()
    print(f"📊 Detector configured with {info['total_evaluators']} evaluators:")
    for evaluator_info in info['evaluators']:
        print(f"   - {evaluator_info['name']} ({evaluator_info['type']})")
    
    prompts = create_test_prompts()
    responses = create_test_responses()
    
    # Test single ensemble evaluation
    print("\n🔹 Testing single ensemble evaluation...")
    try:
        result = await detector.evaluate_single(
            responses[1], prompts[0],  # Use hallucinated response
            source_text="The M1A2 Abrams tank has a 120mm cannon, crew of 4, and weighs 62 tons."
        )
        
        print(f"✅ Ensemble evaluation completed:")
        print(f"   Ensemble Score: {result.ensemble_score:.3f}")
        print(f"   Ensemble Confidence: {result.ensemble_confidence:.3f}")
        print(f"   Consensus Level: {result.consensus_level}")
        print(f"   Evaluator Agreement: {result.evaluator_agreement:.3f}")
        print(f"   Explanation: {result.explanation}")
        print(f"   Individual evaluators: {len(result.individual_results)}")
        
        # Show individual results
        for individual in result.individual_results:
            print(f"     {individual.evaluator_name}: {individual.hallucination_score.score:.3f}")
        
    except Exception as e:
        print(f"❌ Ensemble single evaluation failed: {e}")
    
    # Test batch ensemble evaluation
    print("\n🔹 Testing batch ensemble evaluation...")
    try:
        def progress_callback(completed, total):
            print(f"   Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        
        results = await detector.evaluate_batch(
            responses[:3], prompts[:3],
            progress_callback=progress_callback
        )
        
        print(f"✅ Ensemble batch evaluation completed: {len(results)} results")
        
        # Analyze results
        ensemble_scores = [r.ensemble_score for r in results]
        avg_score = sum(ensemble_scores) / len(ensemble_scores)
        hallucinated_count = sum(1 for r in results if r.is_hallucinated)
        
        print(f"   Average ensemble score: {avg_score:.3f}")
        print(f"   Responses flagged as hallucinated: {hallucinated_count}/{len(results)}")
        
        # Save ensemble results
        save_evaluation_results(results, "ensemble_test_results")
        
    except Exception as e:
        print(f"❌ Ensemble batch evaluation failed: {e}")
    
    await detector.cleanup()


async def test_integration_with_response_generator():
    """Test integration between response generator and hallucination detection."""
    print("\n🔗 Testing Integration: Response Generation + Hallucination Detection")
    print("=" * 50)
    
    # Setup providers
    mock_config = APIConfig(provider='mock', model='mock-gpt-4')
    mock_provider = MockLLMProvider(mock_config)
    
    # Setup response generator
    response_config = ResponseConfig(
        hallucination_rate=0.5,  # 50% injection rate for testing
        concurrent_requests=2
    )
    response_generator = ResponseGenerator({"mock": mock_provider}, response_config)
    
    # Setup hallucination detector
    detector = HallucinationDetector(
        llm_provider=mock_provider,
        enable_huggingface_hhem=True,
        enable_g_eval=False,  # Skip G-Eval for speed
        enable_selfcheck=False  # Skip SelfCheck for speed
    )
    
    prompts = create_test_prompts()[:2]  # Use first 2 prompts
    
    print(f"🔄 Generating responses for {len(prompts)} prompts...")
    
    try:
        # Generate responses
        responses = await response_generator.generate_responses(
            prompts, ["mock"], hallucination_rate=0.7
        )
        
        print(f"✅ Generated {len(responses)} responses")
        
        # Evaluate responses for hallucinations
        print(f"🔍 Evaluating responses for hallucinations...")
        
        evaluation_results = await detector.evaluate_batch(
            responses, prompts[:len(responses)]
        )
        
        print(f"✅ Evaluated {len(evaluation_results)} responses")
        
        # Analyze end-to-end pipeline
        print("\n📊 End-to-End Pipeline Analysis:")
        for i, (response, evaluation) in enumerate(zip(responses, evaluation_results)):
            injected = response.contains_hallucination
            detected_score = evaluation.ensemble_score
            detected = evaluation.is_hallucinated
            
            print(f"   Response {i+1}:")
            print(f"     Hallucination injected: {injected}")
            print(f"     Detection score: {detected_score:.3f}")
            print(f"     Detected as hallucination: {detected}")
            print(f"     Agreement: {'✅' if injected == detected else '❌'}")
        
        # Save integrated results
        integrated_data = []
        for response, evaluation in zip(responses, evaluation_results):
            integrated_data.append({
                "response_text": response.text[:100] + "...",
                "injected_hallucination": response.contains_hallucination,
                "injected_types": response.hallucination_types,
                "detected_score": evaluation.ensemble_score,
                "detected_hallucination": evaluation.is_hallucinated,
                "detection_confidence": evaluation.ensemble_confidence,
                "consensus_level": evaluation.consensus_level
            })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = OUTPUT_DIR / f"integration_test_results_{timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(integrated_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Integration results saved to: {filepath}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    await response_generator.cleanup()
    await detector.cleanup()


async def main():
    """Run all hallucination detection tests."""
    print("🚀 DoDHaluEval Hallucination Detection Test Suite")
    print("=" * 60)
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    print(f"🕐 Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    await test_huggingface_hhem_evaluator()
    await test_g_eval_evaluator()
    await test_selfcheck_evaluator()
    await test_ensemble_detector()
    await test_integration_with_response_generator()
    
    print("\n🎉 All hallucination detection tests completed!")
    print(f"📁 Check {OUTPUT_DIR} for detailed results")


if __name__ == "__main__":
    asyncio.run(main())