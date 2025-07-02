"""
Hallucination Detection Orchestrator for DoDHaluEval.

This module coordinates multiple hallucination detection methods and provides
a unified interface for evaluating responses across different evaluators.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from statistics import mean, median, stdev

from .evaluators.base import BaseEvaluator, EvaluationResult
from .evaluators.huggingface_hhem_evaluator import HuggingFaceHHEMEvaluator
from .evaluators.g_eval import GEvalEvaluator
from .evaluators.selfcheck import SelfCheckGPTEvaluator
from ..models.schemas import Response, Prompt
from ..providers.base import LLMProvider
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleResult:
    """Result from ensemble of multiple evaluators."""
    
    individual_results: List[EvaluationResult]
    ensemble_score: float
    ensemble_confidence: float
    consensus_level: str  # "high", "medium", "low"
    explanation: str
    metadata: Dict[str, Any]
    response_id: str = "unknown"  # ID of the response being evaluated
    
    @property
    def is_hallucinated(self) -> bool:
        """Return True if ensemble indicates hallucination."""
        return self.ensemble_score > 0.5
    
    @property
    def evaluator_agreement(self) -> float:
        """Calculate agreement level between evaluators (0.0-1.0)."""
        if len(self.individual_results) < 2:
            return 1.0
        
        scores = [r.hallucination_score.score for r in self.individual_results]
        
        # Calculate coefficient of variation (lower = more agreement)
        if mean(scores) == 0:
            return 1.0
        
        cv = stdev(scores) / mean(scores) if len(scores) > 1 else 0.0
        agreement = max(0.0, 1.0 - cv)  # Convert to agreement score
        return agreement


class HallucinationDetector:
    """
    Orchestrates multiple hallucination detection methods.
    
    Coordinates evaluation across different detectors and provides ensemble results.
    """
    
    def __init__(
        self,
        evaluators: List[BaseEvaluator] = None,
        llm_provider: LLMProvider = None,
        enable_huggingface_hhem: bool = True,
        enable_g_eval: bool = True,
        enable_selfcheck: bool = True,
        ensemble_method: str = "weighted_average"
    ):
        self.evaluators = evaluators or []
        self.llm_provider = llm_provider
        self.ensemble_method = ensemble_method
        
        # Auto-initialize evaluators if none provided
        if not self.evaluators:
            self._initialize_default_evaluators(
                enable_huggingface_hhem, enable_g_eval, enable_selfcheck
            )
        
        logger.info(f"HallucinationDetector initialized with {len(self.evaluators)} evaluators")
    
    def _initialize_default_evaluators(
        self,
        enable_huggingface_hhem: bool,
        enable_g_eval: bool,
        enable_selfcheck: bool
    ):
        """Initialize default set of evaluators."""
        
        # HuggingFace HHEM evaluator
        if enable_huggingface_hhem:
            try:
                import os
                hf_token = os.getenv('HUGGINGFACE_TOKEN')
                hf_hhem_evaluator = HuggingFaceHHEMEvaluator(hf_token=hf_token)
                self.evaluators.append(hf_hhem_evaluator)
                logger.info("Added HuggingFace HHEM evaluator")
            except Exception as e:
                logger.warning(f"Failed to initialize HuggingFace HHEM evaluator: {e}")
        
        # G-Eval evaluator (requires LLM provider)
        if enable_g_eval and self.llm_provider:
            try:
                g_eval_evaluator = GEvalEvaluator(self.llm_provider)
                self.evaluators.append(g_eval_evaluator)
                logger.info("Added G-Eval evaluator")
            except Exception as e:
                logger.warning(f"Failed to initialize G-Eval evaluator: {e}")
        elif enable_g_eval:
            logger.warning("G-Eval evaluator requires LLM provider")
        
        # SelfCheckGPT evaluator (requires LLM provider)
        if enable_selfcheck and self.llm_provider:
            try:
                selfcheck_evaluator = SelfCheckGPTEvaluator(self.llm_provider)
                self.evaluators.append(selfcheck_evaluator)
                logger.info("Added SelfCheckGPT evaluator")
            except Exception as e:
                logger.warning(f"Failed to initialize SelfCheckGPT evaluator: {e}")
        elif enable_selfcheck:
            logger.warning("SelfCheckGPT evaluator requires LLM provider")
        
        if not self.evaluators:
            logger.warning("No evaluators could be initialized")
    
    def add_evaluator(self, evaluator: BaseEvaluator):
        """Add an evaluator to the detector."""
        self.evaluators.append(evaluator)
        logger.info(f"Added evaluator: {evaluator.name}")
    
    def remove_evaluator(self, evaluator_name: str):
        """Remove an evaluator by name."""
        self.evaluators = [e for e in self.evaluators if e.name != evaluator_name]
        logger.info(f"Removed evaluator: {evaluator_name}")
    
    async def evaluate_single(
        self,
        response: Response,
        prompt: Prompt,
        source_text: str = None,
        evaluators: List[str] = None
    ) -> EnsembleResult:
        """
        Evaluate a single response using multiple evaluators.
        
        Args:
            response: Response to evaluate
            prompt: Original prompt
            source_text: Optional source text for context
            evaluators: Optional list of evaluator names to use
            
        Returns:
            EnsembleResult with combined evaluation
        """
        start_time = time.time()
        
        # Filter evaluators if specific ones requested
        active_evaluators = self.evaluators
        if evaluators:
            active_evaluators = [e for e in self.evaluators if e.name in evaluators]
        
        if not active_evaluators:
            logger.warning("No active evaluators for evaluation")
            return self._create_empty_result(response, prompt, "No evaluators available")
        
        logger.debug(f"Evaluating response {response.id} with {len(active_evaluators)} evaluators")
        
        # Run all evaluators concurrently
        evaluation_tasks = [
            evaluator.evaluate_single(response, prompt, source_text)
            for evaluator in active_evaluators
        ]
        
        try:
            individual_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error during concurrent evaluation: {e}")
            return self._create_empty_result(response, prompt, f"Evaluation failed: {str(e)}")
        
        # Process results, handling exceptions
        valid_results = []
        for i, result in enumerate(individual_results):
            if isinstance(result, Exception):
                logger.error(f"Evaluator {active_evaluators[i].name} failed: {result}")
            elif isinstance(result, EvaluationResult):
                valid_results.append(result)
        
        if not valid_results:
            logger.warning("No valid evaluation results")
            return self._create_empty_result(response, prompt, "All evaluations failed")
        
        # Calculate ensemble result
        ensemble_result = self._calculate_ensemble(valid_results, response, prompt)
        ensemble_result.metadata["evaluation_time_seconds"] = time.time() - start_time
        ensemble_result.metadata["num_evaluators"] = len(valid_results)
        ensemble_result.metadata["source_text_provided"] = source_text is not None
        
        logger.debug(
            f"Ensemble evaluation completed: "
            f"score={ensemble_result.ensemble_score:.3f}, "
            f"confidence={ensemble_result.ensemble_confidence:.3f}, "
            f"evaluators={len(valid_results)}"
        )
        
        return ensemble_result
    
    async def evaluate_batch(
        self,
        responses: List[Response],
        prompts: List[Prompt],
        source_texts: List[str] = None,
        evaluators: List[str] = None,
        progress_callback: callable = None
    ) -> List[EnsembleResult]:
        """
        Evaluate multiple responses using multiple evaluators.
        
        Args:
            responses: List of responses to evaluate
            prompts: List of corresponding prompts
            source_texts: Optional list of source texts
            evaluators: Optional list of evaluator names to use
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of EnsembleResult objects
        """
        if len(responses) != len(prompts):
            raise ValueError("Number of responses must match number of prompts")
        
        if source_texts and len(source_texts) != len(responses):
            raise ValueError("Number of source texts must match number of responses")
        
        logger.info(f"Starting batch evaluation of {len(responses)} responses")
        
        # Process in smaller batches to manage memory and provide progress updates
        batch_size = 10
        all_results = []
        
        for i in range(0, len(responses), batch_size):
            batch_end = min(i + batch_size, len(responses))
            batch_responses = responses[i:batch_end]
            batch_prompts = prompts[i:batch_end]
            batch_source_texts = source_texts[i:batch_end] if source_texts else None
            
            # Evaluate batch
            batch_tasks = []
            for j, (response, prompt) in enumerate(zip(batch_responses, batch_prompts)):
                source_text = batch_source_texts[j] if batch_source_texts else None
                task = self.evaluate_single(response, prompt, source_text, evaluators)
                batch_tasks.append(task)
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process batch results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch evaluation exception: {result}")
                    # Create error result
                    error_result = self._create_empty_result(
                        batch_responses[0], batch_prompts[0], f"Batch error: {str(result)}"
                    )
                    all_results.append(error_result)
                else:
                    all_results.append(result)
            
            # Progress callback
            if progress_callback:
                progress_callback(batch_end, len(responses))
        
        logger.info(f"Batch evaluation completed: {len(all_results)} results")
        return all_results
    
    def _calculate_ensemble(
        self,
        individual_results: List[EvaluationResult],
        response: Response,
        prompt: Prompt
    ) -> EnsembleResult:
        """Calculate ensemble result from individual evaluator results."""
        
        if not individual_results:
            return self._create_empty_result(response, prompt, "No individual results")
        
        scores = [r.hallucination_score.score for r in individual_results]
        confidences = [r.hallucination_score.confidence for r in individual_results]
        
        # Calculate ensemble score based on method
        if self.ensemble_method == "weighted_average":
            # Weight by confidence
            total_weight = sum(confidences)
            if total_weight > 0:
                ensemble_score = sum(s * c for s, c in zip(scores, confidences)) / total_weight
            else:
                ensemble_score = mean(scores)
        elif self.ensemble_method == "median":
            ensemble_score = median(scores)
        elif self.ensemble_method == "majority_vote":
            # Simple majority vote (>0.5 = hallucination)
            hallucination_votes = sum(1 for s in scores if s > 0.5)
            ensemble_score = 1.0 if hallucination_votes > len(scores) / 2 else 0.0
        else:  # "simple_average"
            ensemble_score = mean(scores)
        
        # Calculate ensemble confidence
        ensemble_confidence = mean(confidences)
        
        # Adjust confidence based on agreement
        if len(scores) > 1:
            score_variance = stdev(scores) if len(scores) > 1 else 0.0
            agreement_factor = max(0.5, 1.0 - score_variance)  # Higher variance = lower confidence
            ensemble_confidence *= agreement_factor
        
        # Determine consensus level
        if len(scores) > 1:
            score_range = max(scores) - min(scores)
            if score_range < 0.2:
                consensus_level = "high"
            elif score_range < 0.4:
                consensus_level = "medium"
            else:
                consensus_level = "low"
        else:
            consensus_level = "single_evaluator"
        
        # Create explanation
        evaluator_names = [r.evaluator_name for r in individual_results]
        explanation = f"Ensemble of {len(individual_results)} evaluators: {', '.join(evaluator_names)}. "
        explanation += f"Consensus: {consensus_level}. "
        
        if ensemble_score > 0.7:
            explanation += "Strong indication of hallucination."
        elif ensemble_score > 0.5:
            explanation += "Likely hallucination detected."
        elif ensemble_score > 0.3:
            explanation += "Some signs of potential hallucination."
        else:
            explanation += "Low likelihood of hallucination."
        
        # Propagate injection metadata for analysis
        ensemble_metadata = {
            "ensemble_method": self.ensemble_method,
            "individual_scores": scores,
            "individual_confidences": confidences,
            "score_variance": stdev(scores) if len(scores) > 1 else 0.0,
            "injected_hallucination": response.contains_hallucination,
            "hallucination_types": getattr(response, 'hallucination_types', []),
            "source_document": response.metadata.get("source_document"),
            "evaluation_methods": [r.evaluator_name for r in individual_results]
        }
        
        return EnsembleResult(
            individual_results=individual_results,
            ensemble_score=ensemble_score,
            ensemble_confidence=ensemble_confidence,
            consensus_level=consensus_level,
            explanation=explanation,
            metadata=ensemble_metadata,
            response_id=response.id
        )
    
    def _create_empty_result(
        self,
        response: Response,
        prompt: Prompt,
        error_message: str
    ) -> EnsembleResult:
        """Create an empty/error result when evaluation fails."""
        return EnsembleResult(
            individual_results=[],
            ensemble_score=0.5,  # Neutral score
            ensemble_confidence=0.1,  # Very low confidence
            consensus_level="error",
            explanation=f"Evaluation error: {error_message}",
            metadata={"error": error_message},
            response_id=response.id
        )
    
    def get_evaluator_info(self) -> Dict[str, Any]:
        """Get information about available evaluators."""
        info = {
            "total_evaluators": len(self.evaluators),
            "evaluators": [],
            "ensemble_method": self.ensemble_method
        }
        
        for evaluator in self.evaluators:
            evaluator_info = {
                "name": evaluator.name,
                "supports_batch": evaluator.supports_batch_evaluation,
                "type": evaluator.__class__.__name__
            }
            
            # Add specific info for different evaluator types
            if isinstance(evaluator, HuggingFaceHHEMEvaluator):
                evaluator_info["initialized"] = evaluator._initialized
                evaluator_info["use_direct_model"] = getattr(evaluator, "use_direct_model", None)
            elif isinstance(evaluator, GEvalEvaluator):
                evaluator_info["llm_model"] = evaluator.llm_provider.config.model
            elif isinstance(evaluator, SelfCheckGPTEvaluator):
                evaluator_info["num_samples"] = evaluator.num_samples
                evaluator_info["llm_model"] = evaluator.llm_provider.config.model
            
            info["evaluators"].append(evaluator_info)
        
        return info
    
    async def cleanup(self):
        """Cleanup all evaluators."""
        for evaluator in self.evaluators:
            try:
                await evaluator.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up evaluator {evaluator.name}: {e}")
        
        logger.info("HallucinationDetector cleanup completed")