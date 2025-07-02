"""
SelfCheckGPT implementation for hallucination detection.

This module implements SelfCheckGPT, which detects hallucinations by checking
consistency across multiple samples generated from the same prompt.
"""

import time
import asyncio
from typing import List, Dict, Any, Optional
from collections import Counter
import re

from .base import BaseEvaluator, EvaluationResult
from ...models.schemas import Response, Prompt
from ...providers.base import LLMProvider, GenerationParameters
from ...utils.logger import get_logger

logger = get_logger(__name__)


class SelfCheckGPTEvaluator(BaseEvaluator):
    """
    SelfCheckGPT implementation for hallucination detection.
    
    Generates multiple responses to the same prompt and checks for consistency.
    Inconsistent information across samples is likely to be hallucinated.
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        num_samples: int = 5,
        temperature: float = 0.8,
        max_tokens: int = 1000,
        consistency_threshold: float = 0.6
    ):
        super().__init__("SelfCheckGPT")
        self.llm_provider = llm_provider
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.consistency_threshold = consistency_threshold
        
        # Consistency checking prompts
        self.consistency_prompts = {
            "factual": self._get_factual_consistency_prompt(),
            "general": self._get_general_consistency_prompt()
        }
    
    def _get_factual_consistency_prompt(self) -> str:
        """Get prompt for checking factual consistency."""
        return """You are an expert fact-checker. Compare these responses to determine if they contain consistent factual information.

**Original Prompt:** {original_prompt}

**Response 1 (Original):** {original_response}

**Response 2 (Sample):** {sample_response}

**Instructions:**
Compare the factual claims in both responses. Focus on:
1. Specific numbers, dates, measurements, quantities
2. Names of people, places, equipment, organizations  
3. Technical specifications and details
4. Procedures and processes described

**Output Format:**
Provide your analysis as a JSON object:
- "consistency_score": Float 0.0-1.0 (1.0 = fully consistent, 0.0 = completely inconsistent)
- "factual_agreements": List of facts both responses agree on
- "factual_disagreements": List of facts where responses contradict each other
- "explanation": Brief explanation of the consistency analysis

Example:
```json
{{
    "consistency_score": 0.7,
    "factual_agreements": ["Both mention 120mm cannon", "Both state crew of 4"],
    "factual_disagreements": ["Weight: 62 tons vs 65 tons", "Range: 4000m vs 3500m"],
    "explanation": "Responses agree on major specifications but differ on specific measurements"
}}
```

Analyze the consistency:"""
    
    def _get_general_consistency_prompt(self) -> str:
        """Get prompt for general consistency checking."""
        return """You are an expert evaluator comparing AI responses for consistency.

**Original Prompt:** {original_prompt}

**Response 1 (Original):** {original_response}

**Response 2 (Sample):** {sample_response}

**Instructions:**
Compare these responses for overall consistency in:
1. Main points and conclusions
2. Factual information and details
3. Logical reasoning and explanations
4. Tone and approach to the topic

**Output Format:**
Provide your analysis as a JSON object:
- "consistency_score": Float 0.0-1.0 (1.0 = fully consistent, 0.0 = completely inconsistent)
- "consistent_elements": List of elements both responses agree on
- "inconsistent_elements": List of elements where responses differ significantly  
- "explanation": Brief explanation of the consistency analysis

Example:
```json
{{
    "consistency_score": 0.8,
    "consistent_elements": ["Main conclusion about effectiveness", "General approach to problem"],
    "inconsistent_elements": ["Specific procedure details", "Risk assessment"],
    "explanation": "Responses agree on high-level points but differ in implementation details"
}}
```

Analyze the consistency:"""
    
    async def _generate_sample_responses(
        self,
        prompt: Prompt,
        source_text: str = None
    ) -> List[str]:
        """
        Generate multiple sample responses to the same prompt.
        
        Args:
            prompt: The prompt to generate responses for
            source_text: Optional source text for context
            
        Returns:
            List of generated response texts
        """
        # Modify prompt slightly for sampling if source text is available
        sample_prompt = prompt.text
        if source_text:
            sample_prompt = f"Based on the following information, {prompt.text.lower()}\n\nSource: {source_text}"
        
        params = GenerationParameters(
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Generate samples concurrently
        async def generate_single():
            try:
                result = await self.llm_provider.generate(sample_prompt, params)
                return result.text
            except Exception as e:
                logger.warning(f"Failed to generate sample response: {e}")
                return f"Generation failed: {str(e)}"
        
        tasks = [generate_single() for _ in range(self.num_samples)]
        sample_responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid responses
        valid_responses = []
        for response in sample_responses:
            if isinstance(response, Exception):
                logger.warning(f"Sample generation exception: {response}")
            elif isinstance(response, str) and not response.startswith("Generation failed:"):
                valid_responses.append(response)
        
        return valid_responses
    
    async def _check_consistency_with_llm(
        self,
        original_prompt: str,
        original_response: str,
        sample_response: str,
        check_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Use LLM to check consistency between original and sample response.
        
        Args:
            original_prompt: The original prompt
            original_response: The original response
            sample_response: A sample response to compare
            check_type: Type of consistency check
            
        Returns:
            Dictionary with consistency analysis
        """
        prompt_template = self.consistency_prompts.get(check_type, self.consistency_prompts["general"])
        
        consistency_prompt = prompt_template.format(
            original_prompt=original_prompt,
            original_response=original_response,
            sample_response=sample_response
        )
        
        try:
            params = GenerationParameters(
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=500
            )
            
            result = await self.llm_provider.generate(consistency_prompt, params)
            response_text = result.text
            
            # Parse JSON response
            import json
            import re
            
            # Try to extract JSON from response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try to find JSON without code blocks
            json_match = re.search(r'\{[^}]*"consistency_score"[^}]*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            # Fallback if no JSON found
            logger.warning("Could not parse consistency check response")
            return {
                "consistency_score": 0.5,
                "explanation": "Could not parse consistency analysis",
                "raw_response": response_text
            }
            
        except Exception as e:
            logger.error(f"Consistency check with LLM failed: {e}")
            return {
                "consistency_score": 0.5,
                "explanation": f"Consistency check failed: {str(e)}",
                "error": str(e)
            }
    
    def _calculate_simple_consistency(
        self,
        original_response: str,
        sample_responses: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate consistency using simple text-based metrics.
        
        Args:
            original_response: The original response
            sample_responses: List of sample responses
            
        Returns:
            Dictionary with consistency metrics
        """
        if not sample_responses:
            return {
                "consistency_score": 0.0,
                "explanation": "No sample responses available for comparison",
                "metrics": {"num_samples": 0}
            }
        
        # Extract key terms from responses
        def extract_key_terms(text: str) -> set:
            # Simple extraction: numbers, capitalized words, technical terms
            numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', text))
            capitalized = set(re.findall(r'\b[A-Z][a-z]+\b', text))
            technical = set(re.findall(r'\b[A-Z0-9-]+\b', text))
            return numbers | capitalized | technical
        
        original_terms = extract_key_terms(original_response)
        
        # Calculate consistency with each sample
        consistency_scores = []
        for sample in sample_responses:
            sample_terms = extract_key_terms(sample)
            
            if not original_terms and not sample_terms:
                consistency_scores.append(1.0)
            elif not original_terms or not sample_terms:
                consistency_scores.append(0.0)
            else:
                # Jaccard similarity
                intersection = len(original_terms & sample_terms)
                union = len(original_terms | sample_terms)
                consistency = intersection / union if union > 0 else 0.0
                consistency_scores.append(consistency)
        
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        
        return {
            "consistency_score": avg_consistency,
            "explanation": f"Average consistency across {len(sample_responses)} samples: {avg_consistency:.3f}",
            "metrics": {
                "num_samples": len(sample_responses),
                "individual_scores": consistency_scores,
                "original_terms_count": len(original_terms),
                "avg_sample_terms": sum(len(extract_key_terms(s)) for s in sample_responses) / len(sample_responses)
            }
        }
    
    async def evaluate_single(
        self,
        response: Response,
        prompt: Prompt,
        source_text: str = None
    ) -> EvaluationResult:
        """
        Evaluate a single response using SelfCheckGPT.
        
        Args:
            response: The response to evaluate
            prompt: The original prompt
            source_text: Optional source text for context
            
        Returns:
            EvaluationResult with SelfCheckGPT score
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Starting SelfCheckGPT evaluation for response {response.id}")
            
            # Generate sample responses
            sample_responses = await self._generate_sample_responses(prompt, source_text)
            
            if not sample_responses:
                logger.warning("No sample responses generated for SelfCheckGPT")
                return EvaluationResult.create(
                    response=response,
                    prompt=prompt,
                    evaluator_name=self.name,
                    score=0.5,
                    confidence=0.1,
                    explanation="No sample responses could be generated",
                    evaluation_time=time.time() - start_time,
                    metadata={"num_samples": 0, "error": "No samples generated"}
                )
            
            # Use simple consistency check as primary method
            simple_consistency = self._calculate_simple_consistency(response.text, sample_responses)
            
            # If we have samples, also try LLM-based consistency check with first sample
            llm_consistency = None
            if sample_responses:
                try:
                    llm_consistency = await self._check_consistency_with_llm(
                        prompt.text,
                        response.text,
                        sample_responses[0],
                        "factual" if prompt.hallucination_type == "factual" else "general"
                    )
                except Exception as e:
                    logger.warning(f"LLM consistency check failed: {e}")
            
            # Combine scores (prefer LLM result if available, otherwise use simple)
            if llm_consistency and "consistency_score" in llm_consistency:
                consistency_score = llm_consistency["consistency_score"]
                explanation = llm_consistency.get("explanation", "LLM-based consistency check")
                raw_output = {
                    "llm_consistency": llm_consistency,
                    "simple_consistency": simple_consistency,
                    "sample_responses": sample_responses[:3]  # Store first 3 samples
                }
            else:
                consistency_score = simple_consistency["consistency_score"]
                explanation = simple_consistency["explanation"]
                raw_output = {
                    "simple_consistency": simple_consistency,
                    "sample_responses": sample_responses[:3]
                }
            
            # Convert consistency to hallucination score (high consistency = low hallucination)
            hallucination_score = 1.0 - consistency_score
            
            # Calculate confidence based on number of samples and consistency variance
            if len(sample_responses) >= self.num_samples - 1:
                confidence = 0.8
            elif len(sample_responses) >= 2:
                confidence = 0.6
            else:
                confidence = 0.3
            
            evaluation_time = time.time() - start_time
            
            result = EvaluationResult.create(
                response=response,
                prompt=prompt,
                evaluator_name=self.name,
                score=hallucination_score,
                confidence=confidence,
                explanation=f"SelfCheck consistency: {consistency_score:.3f}. {explanation}",
                evaluation_time=evaluation_time,
                raw_output=raw_output,
                metadata={
                    "num_samples": len(sample_responses),
                    "consistency_score": consistency_score,
                    "temperature": self.temperature,
                    "source_text_provided": source_text is not None
                }
            )
            
            logger.debug(
                f"SelfCheckGPT completed: "
                f"consistency={consistency_score:.3f}, "
                f"hallucination_score={hallucination_score:.3f}, "
                f"samples={len(sample_responses)}, "
                f"time={evaluation_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            evaluation_time = time.time() - start_time
            logger.error(f"SelfCheckGPT evaluation failed for response {response.id}: {e}")
            
            return EvaluationResult.create(
                response=response,
                prompt=prompt,
                evaluator_name=self.name,
                score=0.5,
                confidence=0.1,
                explanation=f"SelfCheckGPT evaluation failed: {str(e)}",
                evaluation_time=evaluation_time,
                metadata={"error": str(e)}
            )
    
    async def cleanup(self):
        """Cleanup LLM provider resources."""
        if hasattr(self.llm_provider, 'close'):
            await self.llm_provider.close()