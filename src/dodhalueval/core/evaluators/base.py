"""
Base evaluator interface for hallucination detection.

This module defines the abstract base class for all hallucination evaluators,
ensuring consistent interfaces across different detection methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from ...models.schemas import Response, Prompt
from ...utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HallucinationScore:
    """Score indicating the likelihood of hallucination."""
    
    score: float  # 0.0 = no hallucination, 1.0 = definitely hallucinated
    confidence: float  # 0.0 = no confidence, 1.0 = very confident
    explanation: Optional[str] = None  # Human-readable explanation
    raw_output: Optional[Dict[str, Any]] = None  # Raw evaluator output
    
    @property
    def is_hallucinated(self) -> bool:
        """Return True if score indicates hallucination (score > 0.5)."""
        return self.score > 0.5
    
    @property
    def certainty_level(self) -> str:
        """Get human-readable certainty level."""
        if self.confidence >= 0.8:
            return "very confident"
        elif self.confidence >= 0.6:
            return "confident"
        elif self.confidence >= 0.4:
            return "somewhat confident"
        else:
            return "uncertain"


@dataclass
class EvaluationResult:
    """Result of evaluating a response for hallucinations."""
    
    id: str
    response_id: str
    prompt_id: str
    evaluator_name: str
    hallucination_score: HallucinationScore
    evaluation_time_seconds: float
    metadata: Dict[str, Any]
    evaluated_at: datetime
    
    @classmethod
    def create(
        cls,
        response: Response,
        prompt: Prompt,
        evaluator_name: str,
        score: float,
        confidence: float,
        explanation: str = None,
        evaluation_time: float = 0.0,
        raw_output: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> 'EvaluationResult':
        """Create an evaluation result."""
        from uuid import uuid4
        
        hallucination_score = HallucinationScore(
            score=score,
            confidence=confidence,
            explanation=explanation,
            raw_output=raw_output
        )
        
        # Include injection metadata for analysis
        enhanced_metadata = metadata or {}
        enhanced_metadata.update({
            "injected_hallucination": response.contains_hallucination,
            "hallucination_types": getattr(response, 'hallucination_types', []),
            "source_document": response.metadata.get("source_document"),
            "evaluation_method": evaluator_name
        })
        
        return cls(
            id=str(uuid4()),
            response_id=response.id,
            prompt_id=prompt.id,
            evaluator_name=evaluator_name,
            hallucination_score=hallucination_score,
            evaluation_time_seconds=evaluation_time,
            metadata=enhanced_metadata,
            evaluated_at=datetime.now()
        )
    
    @property
    def is_hallucinated(self) -> bool:
        """Return True if this evaluation indicates hallucination."""
        return self.hallucination_score.is_hallucinated


class BaseEvaluator(ABC):
    """Abstract base class for hallucination evaluators."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.logger = get_logger(f"{__name__}.{self.name}")
    
    @abstractmethod
    async def evaluate_single(
        self,
        response: Response,
        prompt: Prompt,
        source_text: str = None
    ) -> EvaluationResult:
        """
        Evaluate a single response for hallucinations.
        
        Args:
            response: The response to evaluate
            prompt: The original prompt
            source_text: Optional source text for context
            
        Returns:
            EvaluationResult with hallucination score
        """
        pass
    
    async def evaluate_batch(
        self,
        responses: List[Response],
        prompts: List[Prompt],
        source_texts: List[str] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple responses for hallucinations.
        
        Args:
            responses: List of responses to evaluate
            prompts: List of corresponding prompts
            source_texts: Optional list of source texts
            
        Returns:
            List of EvaluationResult objects
        """
        if len(responses) != len(prompts):
            raise ValueError("Number of responses must match number of prompts")
            
        if source_texts and len(source_texts) != len(responses):
            raise ValueError("Number of source texts must match number of responses")
        
        results = []
        for i, (response, prompt) in enumerate(zip(responses, prompts)):
            source_text = source_texts[i] if source_texts else None
            try:
                result = await self.evaluate_single(response, prompt, source_text)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to evaluate response {response.id}: {e}")
                # Create error result
                error_result = EvaluationResult.create(
                    response=response,
                    prompt=prompt,
                    evaluator_name=self.name,
                    score=0.0,
                    confidence=0.0,
                    explanation=f"Evaluation failed: {str(e)}",
                    metadata={'error': str(e)}
                )
                results.append(error_result)
        
        return results
    
    @property
    def supports_batch_evaluation(self) -> bool:
        """Return True if this evaluator has optimized batch evaluation."""
        return False
    
    async def cleanup(self):
        """Cleanup any resources used by the evaluator."""
        pass