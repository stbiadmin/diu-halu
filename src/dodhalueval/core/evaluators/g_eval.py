"""
G-Eval implementation for hallucination detection.

This module implements G-Eval, which uses GPT models as judges to evaluate
whether responses contain hallucinations based on prompts and source text.
"""

import time
import json
import re
from typing import List, Dict, Any, Optional

from .base import BaseEvaluator, EvaluationResult
from ...models.schemas import Response, Prompt
from ...providers.base import LLMProvider, GenerationParameters
from ...utils.logger import get_logger

logger = get_logger(__name__)


class GEvalEvaluator(BaseEvaluator):
    """
    G-Eval implementation using LLM as judge for hallucination detection.
    
    Uses a language model (typically GPT-4) to evaluate whether responses
    contain hallucinations by analyzing them against prompts and source text.
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        temperature: float = 0.1,
        max_tokens: int = 500
    ):
        super().__init__("G-Eval")
        self.llm_provider = llm_provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Pre-defined evaluation prompts for different aspects
        self.evaluation_prompts = {
            "factual": self._get_factual_evaluation_prompt(),
            "logical": self._get_logical_evaluation_prompt(),
            "context": self._get_context_evaluation_prompt(),
            "general": self._get_general_evaluation_prompt()
        }
    
    def _get_factual_evaluation_prompt(self) -> str:
        """Get prompt for evaluating factual hallucinations."""
        return """You are an expert evaluator tasked with detecting factual hallucinations in AI responses.

Your task is to evaluate whether the response contains factual errors or made-up information that contradicts the source text or general knowledge.

**Source Text:**
{source_text}

**Original Prompt:**
{prompt}

**Response to Evaluate:**
{response}

**Evaluation Instructions:**
1. Check if the response contains any factual claims that contradict the source text
2. Identify any specific numbers, dates, names, or technical details that are incorrect
3. Look for plausible but fabricated information (e.g., wrong specifications, fake statistics)
4. Consider whether missing information is filled in with made-up details

**Output Format:**
Provide your evaluation as a JSON object with these fields:
- "score": A float between 0.0 (no hallucination) and 1.0 (definite hallucination)
- "confidence": A float between 0.0 (uncertain) and 1.0 (very confident)
- "explanation": A detailed explanation of your evaluation
- "specific_issues": A list of specific factual errors found (if any)

Example:
```json
{{
    "score": 0.7,
    "confidence": 0.9,
    "explanation": "The response claims the M1A2 Abrams has a crew of 5, but the source text clearly states it has a crew of 4. Also mentions 150mm cannon instead of 120mm.",
    "specific_issues": ["Incorrect crew size: 5 instead of 4", "Wrong cannon caliber: 150mm instead of 120mm"]
}}
```

Evaluate the response:"""
    
    def _get_logical_evaluation_prompt(self) -> str:
        """Get prompt for evaluating logical hallucinations."""
        return """You are an expert evaluator tasked with detecting logical inconsistencies and reasoning errors in AI responses.

Your task is to evaluate whether the response contains logical contradictions, invalid reasoning, or inconsistent statements.

**Source Text:**
{source_text}

**Original Prompt:**
{prompt}

**Response to Evaluate:**
{response}

**Evaluation Instructions:**
1. Check for internal contradictions within the response
2. Look for invalid cause-and-effect relationships
3. Identify reasoning that doesn't follow logically from the premises
4. Check if conclusions are supported by the evidence presented
5. Look for impossible or contradictory statements

**Output Format:**
Provide your evaluation as a JSON object with these fields:
- "score": A float between 0.0 (logically sound) and 1.0 (major logical issues)
- "confidence": A float between 0.0 (uncertain) and 1.0 (very confident)  
- "explanation": A detailed explanation of your evaluation
- "logical_issues": A list of specific logical problems found (if any)

Example:
```json
{{
    "score": 0.6,
    "confidence": 0.8,
    "explanation": "The response states that tanks are both 'extremely mobile' and 'unable to move quickly' in the same paragraph, creating a logical contradiction.",
    "logical_issues": ["Contradictory statements about mobility", "Invalid conclusion about effectiveness"]
}}
```

Evaluate the response:"""
    
    def _get_context_evaluation_prompt(self) -> str:
        """Get prompt for evaluating contextual hallucinations."""
        return """You are an expert evaluator tasked with detecting context-related hallucinations in AI responses.

Your task is to evaluate whether the response inappropriately mixes contexts, uses information from wrong domains, or applies irrelevant knowledge.

**Source Text:**
{source_text}

**Original Prompt:**
{prompt}

**Response to Evaluate:**
{response}

**Evaluation Instructions:**
1. Check if the response mixes information from different contexts inappropriately
2. Look for application of rules/procedures from wrong domains (e.g., civilian vs military)
3. Identify use of outdated or irrelevant information
4. Check if the response stays within the appropriate scope and context
5. Look for anachronistic references or temporal confusion

**Output Format:**
Provide your evaluation as a JSON object with these fields:
- "score": A float between 0.0 (appropriate context) and 1.0 (major context issues)
- "confidence": A float between 0.0 (uncertain) and 1.0 (very confident)
- "explanation": A detailed explanation of your evaluation  
- "context_issues": A list of specific context problems found (if any)

Example:
```json
{{
    "score": 0.5,
    "confidence": 0.7,
    "explanation": "The response mentions civilian air traffic control procedures when discussing military aviation, which is an inappropriate context mix.",
    "context_issues": ["Mixed civilian and military procedures", "Referenced outdated regulation"]
}}
```

Evaluate the response:"""
    
    def _get_general_evaluation_prompt(self) -> str:
        """Get prompt for general hallucination evaluation."""
        return """You are an expert evaluator tasked with detecting hallucinations in AI responses.

Your task is to comprehensively evaluate whether the response contains any form of hallucination, including factual errors, logical inconsistencies, or contextual inappropriateness.

**Source Text:**
{source_text}

**Original Prompt:**
{prompt}

**Response to Evaluate:**
{response}

**Evaluation Instructions:**
1. **Factual Accuracy**: Check for incorrect facts, numbers, names, specifications
2. **Logical Consistency**: Look for contradictions and invalid reasoning
3. **Contextual Appropriateness**: Ensure information is relevant and properly contextualized
4. **Source Fidelity**: Verify the response aligns with the source text when applicable
5. **Plausibility**: Identify made-up but plausible-sounding information

**Output Format:**
Provide your evaluation as a JSON object with these fields:
- "score": A float between 0.0 (no hallucination) and 1.0 (definite hallucination)
- "confidence": A float between 0.0 (uncertain) and 1.0 (very confident)
- "explanation": A detailed explanation covering all aspects of your evaluation
- "hallucination_types": A list of types found (e.g., ["factual", "logical"])
- "specific_issues": A list of specific problems identified

Example:
```json
{{
    "score": 0.8,
    "confidence": 0.9,
    "explanation": "The response contains several factual errors and one logical contradiction. The crew size is wrong, and the response contradicts itself about operational capabilities.",
    "hallucination_types": ["factual", "logical"],
    "specific_issues": ["Incorrect crew size", "Self-contradictory statements about capabilities"]
}}
```

Evaluate the response:"""
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling various formats."""
        logger.debug(f"G-Eval extracting JSON from response (full): {repr(response_text)}")
        try:
            # Try to find JSON in code blocks
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try to find JSON without code blocks - look for complete JSON objects
            # Find JSON that contains "score" field
            json_match = re.search(r'\{.*?"score".*?\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                # Try to fix common JSON extraction issues by finding balanced braces
                brace_count = 0
                end_pos = 0
                for i, char in enumerate(json_text):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                
                if end_pos > 0:
                    balanced_json = json_text[:end_pos]
                    return json.loads(balanced_json)
                else:
                    return json.loads(json_text)
            
            # If no JSON found, create default response
            logger.warning("Could not extract JSON from G-Eval response, using fallback")
            return {
                "score": 0.5,
                "confidence": 0.3,
                "explanation": "Could not parse evaluation response",
                "specific_issues": [],
                "raw_response": response_text
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in G-Eval response: {e}")
            return {
                "score": 0.5,
                "confidence": 0.1,
                "explanation": f"JSON parsing failed: {str(e)}",
                "specific_issues": [],
                "raw_response": response_text
            }
    
    async def _evaluate_with_llm(
        self,
        prompt_text: str,
        response_text: str,
        source_text: str = None,
        evaluation_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Use LLM to evaluate response for hallucinations.
        
        Args:
            prompt_text: Original prompt
            response_text: Response to evaluate
            source_text: Source text for context
            evaluation_type: Type of evaluation (factual, logical, context, general)
            
        Returns:
            Dictionary with evaluation results
        """
        # Get appropriate evaluation prompt
        eval_prompt_template = self.evaluation_prompts.get(evaluation_type, self.evaluation_prompts["general"])
        
        # Format the prompt
        eval_prompt = eval_prompt_template.format(
            source_text=source_text or "No source text provided",
            prompt=prompt_text,
            response=response_text
        )
        
        try:
            # Generate evaluation using LLM
            params = GenerationParameters(
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result = await self.llm_provider.generate(eval_prompt, params)
            evaluation_response = result.text
            
            # Parse the response
            parsed_result = self._extract_json_from_response(evaluation_response)
            
            # Ensure required fields exist with defaults
            parsed_result.setdefault("score", 0.5)
            parsed_result.setdefault("confidence", 0.5)
            parsed_result.setdefault("explanation", "No explanation provided")
            
            # Clamp values to valid ranges
            parsed_result["score"] = max(0.0, min(1.0, float(parsed_result["score"])))
            parsed_result["confidence"] = max(0.0, min(1.0, float(parsed_result["confidence"])))
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"G-Eval LLM call failed: {e}")
            return {
                "score": 0.5,
                "confidence": 0.1,
                "explanation": f"G-Eval failed: {str(e)}",
                "specific_issues": [],
                "error": str(e)
            }
    
    async def evaluate_single(
        self,
        response: Response,
        prompt: Prompt,
        source_text: str = None
    ) -> EvaluationResult:
        """
        Evaluate a single response using G-Eval.
        
        Args:
            response: The response to evaluate
            prompt: The original prompt  
            source_text: Optional source text for context
            
        Returns:
            EvaluationResult with G-Eval score
        """
        start_time = time.time()
        
        try:
            # Determine evaluation type based on prompt hallucination type
            eval_type = "general"
            if prompt.hallucination_type in ["factual", "logical", "context"]:
                eval_type = prompt.hallucination_type
            
            # Run evaluation
            eval_result = await self._evaluate_with_llm(
                prompt.text,
                response.text,
                source_text,
                eval_type
            )
            
            evaluation_time = time.time() - start_time
            
            # Create evaluation result
            result = EvaluationResult.create(
                response=response,
                prompt=prompt,
                evaluator_name=self.name,
                score=eval_result["score"],
                confidence=eval_result["confidence"],
                explanation=eval_result["explanation"],
                evaluation_time=evaluation_time,
                raw_output=eval_result,
                metadata={
                    "evaluation_type": eval_type,
                    "llm_model": self.llm_provider.config.model,
                    "temperature": self.temperature,
                    "source_text_provided": source_text is not None
                }
            )
            
            logger.debug(
                f"G-Eval completed: "
                f"score={eval_result['score']:.3f}, "
                f"confidence={eval_result['confidence']:.3f}, "
                f"type={eval_type}, "
                f"time={evaluation_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            evaluation_time = time.time() - start_time
            logger.error(f"G-Eval evaluation failed for response {response.id}: {e}")
            logger.error(f"Full exception details: {type(e).__name__}: {repr(str(e))}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return EvaluationResult.create(
                response=response,
                prompt=prompt,
                evaluator_name=self.name,
                score=0.5,
                confidence=0.1,
                explanation=f"G-Eval evaluation failed: {str(e)}",
                evaluation_time=evaluation_time,
                metadata={"error": str(e)}
            )
    
    async def cleanup(self):
        """Cleanup LLM provider resources."""
        if hasattr(self.llm_provider, 'close'):
            await self.llm_provider.close()