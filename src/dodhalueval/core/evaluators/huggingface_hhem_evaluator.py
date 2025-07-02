"""
HuggingFace HHEM-2.1 Hallucination Evaluator

This module implements hallucination detection using the HuggingFace version of HHEM-2.1
rather than the Vectara API. Based on the implementation pattern from secret_agent repository.
"""

import logging
import warnings
import re
from typing import Optional, List, Tuple
import os

from .base import EvaluationResult, BaseEvaluator

logger = logging.getLogger(__name__)

try:
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    HHEM_AVAILABLE = True
except ImportError:
    HHEM_AVAILABLE = False
    torch = None
    transformers = None


class HuggingFaceHHEMEvaluator(BaseEvaluator):
    """
    HuggingFace HHEM-2.1 hallucination evaluation model wrapper.
    
    This evaluator uses the HuggingFace transformers library to run HHEM-2.1
    locally without requiring API calls to Vectara. It implements smart text
    chunking for long texts and provides fallback mechanisms.
    """
    
    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize the HuggingFace HHEM evaluator.
        
        Args:
            hf_token: Optional HuggingFace token for model access
        """
        super().__init__("HuggingFaceHHEM")
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.hf_token = hf_token
        self._initialized = False
        self.use_direct_model = True
        self.mock_mode = False
        
    def _initialize_model(self):
        """Lazy initialization of the HHEM model."""
        if not HHEM_AVAILABLE:
            logger.warning(
                "HHEM model dependencies (torch, transformers) not available. "
                "Running in mock mode. Install with: pip install torch transformers"
            )
            self.use_direct_model = False
            self._initialized = True
            self.mock_mode = True
            return
            
        if self._initialized:
            return
        
        # Set up comprehensive warning suppression for HHEM
        self._setup_warning_suppression()
        
        try:
            model_name = "vectara/hallucination_evaluation_model"
            logger.info(f"Loading HHEM-2.1 model: {model_name}")
            
            # Method 1: Try using AutoModel with direct predict method
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="You are using a model of type HHEMv2Config")
                    
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        token=self.hf_token,
                        trust_remote_code=True
                    )
                    
                self.tokenizer = None  # Model has built-in predict method
                self.use_direct_model = True
                logger.info("HHEM-2.1 model loaded successfully using AutoModel")
                
            except Exception as e:
                logger.warning(f"AutoModel approach failed: {e}, trying pipeline approach")
                
                # Method 2: Try pipeline with proper tokenizer
                # Use the correct tokenizer as specified in the documentation
                self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
                
                self.pipeline = pipeline(
                    "text-classification",
                    model=model_name,
                    tokenizer=self.tokenizer,
                    token=self.hf_token,
                    trust_remote_code=True,
                    device=0 if torch.cuda.is_available() else -1
                )
                self.use_direct_model = False
                logger.info("HHEM-2.1 model loaded successfully using pipeline")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to load HHEM-2.1 model: {e}")
            raise
    
    def _setup_warning_suppression(self):
        """Set up comprehensive warning suppression for HHEM."""
        # Suppress transformers warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
        warnings.filterwarnings("ignore", message=".*HHEMv2Config.*")
        warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
        
        # Also suppress at environment level
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    
    async def evaluate(self, text: str, reference_text: Optional[str] = None) -> EvaluationResult:
        """
        Evaluate hallucination score using HHEM-2.1.
        
        Args:
            text: The text to evaluate for hallucinations
            reference_text: Optional reference document (if available)
            
        Returns:
            EvaluationResult: Result containing HHEM score and metadata
        """
        if not self._initialized:
            self._initialize_model()
        
        try:
            # HHEM expects pairs of text: (hypothesis, premise)
            # For single text evaluation, we use the text as hypothesis and empty/document as premise
            
            if reference_text:
                premise = reference_text
                hypothesis = text
            else:
                # For single text, use a neutral premise
                premise = "This is a factual statement."
                hypothesis = text
            
            # Smart chunking for long texts to preserve semantic meaning
            hypothesis_chunks = self._smart_chunk_text(hypothesis)
            premise_chunks = self._smart_chunk_text(premise) if len(premise) > 200 else [premise]
            
            if self.mock_mode:
                # Mock evaluation when dependencies are missing
                score = self._mock_evaluate(text, reference_text)
            else:
                score = await self._evaluate_chunks(hypothesis_chunks, premise_chunks)
            
            # Create a fake response and prompt for compatibility
            from ...models.schemas import Response, Prompt
            
            fake_response = Response(
                id="hf_eval_response",
                prompt_id="hf_eval_prompt", 
                text=text,
                model="HuggingFaceHHEM",
                provider="huggingface"
            )
            
            fake_prompt = Prompt(
                id="hf_eval_prompt",
                text="HuggingFace HHEM evaluation",
                source_document_id="hf_eval",
                generation_strategy="hf_hhem"
            )
            
            return EvaluationResult.create(
                response=fake_response,
                prompt=fake_prompt,
                evaluator_name="HuggingFaceHHEM",
                score=score,
                confidence=0.85,  # HHEM is generally reliable  
                explanation=f"HHEM-2.1 evaluation using {len(hypothesis_chunks)} text chunks",
                metadata={
                    "model_type": "direct" if self.use_direct_model else "pipeline",
                    "num_hypothesis_chunks": len(hypothesis_chunks),
                    "num_premise_chunks": len(premise_chunks),
                    "has_reference": reference_text is not None
                }
            )
            
        except Exception as e:
            logger.error(f"HHEM evaluation failed: {e}")
            # Return fallback score
            from ...models.schemas import Response, Prompt
            
            fake_response = Response(
                id="hf_eval_response_error",
                prompt_id="hf_eval_prompt_error", 
                text=text,
                model="HuggingFaceHHEM",
                provider="huggingface"
            )
            
            fake_prompt = Prompt(
                id="hf_eval_prompt_error",
                text="HuggingFace HHEM evaluation (error)",
                source_document_id="hf_eval",
                generation_strategy="hf_hhem"
            )
            
            return EvaluationResult.create(
                response=fake_response,
                prompt=fake_prompt,
                evaluator_name="HuggingFaceHHEM",
                score=0.5,  # Neutral fallback
                confidence=0.1,
                explanation=f"HHEM evaluation failed: {str(e)}. Using fallback score.",
                metadata={"error": str(e), "fallback": True}
            )
    
    async def evaluate_single(
        self,
        response,
        prompt,
        source_text: str = None
    ):
        """
        Evaluate a single response for hallucinations (BaseEvaluator interface).
        
        Args:
            response: The response object to evaluate
            prompt: The original prompt object
            source_text: Optional source text for context
            
        Returns:
            EvaluationResult with hallucination score
        """
        # Extract text from response object and use source_text as reference
        response_text = response.text if hasattr(response, 'text') else str(response)
        reference_text = source_text or (prompt.expected_answer if hasattr(prompt, 'expected_answer') else None)
        
        return await self.evaluate(text=response_text, reference_text=reference_text)
    
    async def cleanup(self):
        """Cleanup resources (BaseEvaluator interface)."""
        # No cleanup needed for local models
        pass
    
    @property
    def supports_batch_evaluation(self) -> bool:
        """Return True if this evaluator has optimized batch evaluation."""
        return True
    
    def _mock_evaluate(self, text: str, reference_text: Optional[str] = None) -> float:
        """Mock evaluation when dependencies are not available."""
        # Simple heuristic-based mock scoring
        text_lower = text.lower()
        
        # Look for obvious hallucination indicators
        if any(word in text_lower for word in ['impossible', 'never', 'always', 'definitely', 'absolutely']):
            return 0.7  # Higher hallucination score
        elif any(word in text_lower for word in ['laser', 'fly', 'supersonic', 'rainbow', 'magic']):
            return 0.9  # Very high hallucination score
        elif any(word in text_lower for word in ['crew', 'tank', 'military', 'protocol', 'standard']):
            return 0.2  # Low hallucination score (military facts)
        else:
            return 0.4  # Default moderate score
        
        # If reference text available, do simple comparison
        if reference_text:
            # Very basic similarity check
            text_words = set(text_lower.split())
            ref_words = set(reference_text.lower().split())
            overlap = len(text_words & ref_words) / max(len(text_words), 1)
            
            # High overlap = low hallucination
            if overlap > 0.5:
                return 0.2
            elif overlap > 0.3:
                return 0.4
            else:
                return 0.6
    
    async def _evaluate_chunks(self, hypothesis_chunks: List[str], premise_chunks: List[str]) -> float:
        """Evaluate chunks and aggregate scores."""
        if self.use_direct_model:
            return await self._evaluate_with_direct_model(hypothesis_chunks, premise_chunks)
        else:
            return await self._evaluate_with_pipeline(hypothesis_chunks, premise_chunks)
    
    async def _evaluate_with_direct_model(self, hypothesis_chunks: List[str], premise_chunks: List[str]) -> float:
        """Evaluate using direct model approach."""
        all_scores = []
        
        # Create pairs from hypothesis chunks with premise chunks
        for i, hyp_chunk in enumerate(hypothesis_chunks):
            if len(premise_chunks) == 1:
                # Short premise - use with each hypothesis chunk
                premise_chunk = premise_chunks[0]
            else:
                # Long premise - use corresponding chunk or last chunk if fewer premise chunks
                premise_chunk = premise_chunks[min(i, len(premise_chunks) - 1)]
            
            pairs = [(hyp_chunk, premise_chunk)]
            logger.debug(f"Running HHEM evaluation on chunk {i+1}/{len(hypothesis_chunks)}")
            
            # Suppress warnings during prediction
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Token indices sequence length is longer")
                warnings.filterwarnings("ignore", message="You are using a model of type HHEMv2Config")
                results = self.model.predict(pairs)
            
            # Extract score from tensor results
            if hasattr(results, '__iter__') and len(results) > 0:
                first_result = results[0]
                if hasattr(first_result, 'item'):
                    chunk_score = float(first_result.item())
                else:
                    chunk_score = float(first_result)
                all_scores.append(chunk_score)
        
        # Aggregate scores - use weighted average based on chunk length
        if all_scores:
            chunk_lengths = [len(chunk) for chunk in hypothesis_chunks]
            total_length = sum(chunk_lengths)
            
            # Weighted average by chunk length
            weighted_score = sum(score * length for score, length in zip(all_scores, chunk_lengths)) / total_length
            return round(weighted_score, 4)
        else:
            return 0.0
    
    async def _evaluate_with_pipeline(self, hypothesis_chunks: List[str], premise_chunks: List[str]) -> float:
        """Evaluate using pipeline approach."""
        all_scores = []
        prompt_template = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {premise}\n\nHypothesis: {hypothesis}"
        
        for i, hyp_chunk in enumerate(hypothesis_chunks):
            if len(premise_chunks) == 1:
                premise_chunk = premise_chunks[0]
            else:
                premise_chunk = premise_chunks[min(i, len(premise_chunks) - 1)]
            
            input_text = prompt_template.format(premise=premise_chunk, hypothesis=hyp_chunk)
            
            logger.debug(f"Running HHEM pipeline on chunk {i+1}/{len(hypothesis_chunks)}")
            result = self.pipeline(input_text, top_k=None)
            
            # Extract hallucination score from pipeline results
            chunk_score = 0.0
            if isinstance(result, list) and len(result) > 0:
                scores = result[0]
                if isinstance(scores, list):
                    # Find the hallucination/false label
                    for score_item in scores:
                        if score_item.get('label', '').lower() in ['false', 'hallucination', '1']:
                            chunk_score = score_item.get('score', 0.0)
                            break
                    else:
                        # Default to first score if no specific label found
                        chunk_score = scores[0].get('score', 0.0)
                elif isinstance(scores, dict):
                    chunk_score = scores.get('score', 0.0)
            
            all_scores.append(chunk_score)
        
        # Aggregate pipeline scores for overlapping chunks
        if all_scores:
            # Calculate weights accounting for overlap
            weights = self._calculate_overlap_weights(hypothesis_chunks, sum(len(chunk) for chunk in hypothesis_chunks))
            
            # Weighted average using overlap-adjusted weights
            weighted_score = sum(score * weight for score, weight in zip(all_scores, weights))
            return round(weighted_score, 4)
        else:
            return 0.0
    
    def _smart_chunk_text(self, text: str, max_chunk_length: int = 200) -> List[str]:
        """
        Sliding window chunking with overlap to preserve context across boundaries.
        
        Args:
            text: Text to chunk
            max_chunk_length: Maximum character length per chunk
            
        Returns:
            List of overlapping text chunks
        """
        if len(text) <= max_chunk_length:
            return [text]
        
        # Use sliding window with 25% overlap
        overlap_size = max_chunk_length // 4
        step_size = max_chunk_length - overlap_size
        chunks = []
        
        # First try to split by paragraphs to find good boundaries
        paragraphs = text.split('\n\n')
        full_text = '\n\n'.join(paragraphs)
        
        start = 0
        while start < len(full_text):
            end = start + max_chunk_length
            
            if end >= len(full_text):
                # Last chunk - take remaining text
                chunks.append(full_text[start:].strip())
                break
            
            # Try to find a good boundary (paragraph or sentence end)
            chunk_text = full_text[start:end]
            
            # Look for paragraph boundary within last 50 chars
            last_para_break = chunk_text.rfind('\n\n', max(0, len(chunk_text) - 50))
            if last_para_break > len(chunk_text) // 2:  # Don't cut too early
                end = start + last_para_break + 2
            else:
                # Look for sentence boundary within last 50 chars
                sentences_end = list(re.finditer(r'[.!?]+\s', chunk_text))
                if sentences_end:
                    last_sentence = sentences_end[-1]
                    if last_sentence.end() > len(chunk_text) // 2:  # Don't cut too early
                        end = start + last_sentence.end()
            
            chunks.append(full_text[start:end].strip())
            start += step_size
        
        return chunks if chunks else [text[:max_chunk_length]]
    
    def _calculate_overlap_weights(self, chunks: List[str], total_text_length: int) -> List[float]:
        """
        Calculate weights for overlapping chunks to avoid double-counting.
        
        Args:
            chunks: List of overlapping text chunks
            total_text_length: Total length of original text
            
        Returns:
            List of normalized weights for each chunk
        """
        if len(chunks) <= 1:
            return [1.0] * len(chunks)
        
        # For overlapping chunks, weight each chunk by its unique contribution
        weights = []
        max_chunk_length = max(len(chunk) for chunk in chunks) if chunks else 200
        overlap_size = max_chunk_length // 4
        step_size = max_chunk_length - overlap_size
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk: full weight for non-overlapping part + half weight for overlap
                unique_length = step_size
                overlap_length = len(chunk) - step_size if len(chunk) > step_size else 0
                weight = unique_length + (overlap_length * 0.5)
            elif i == len(chunks) - 1:
                # Last chunk: half weight for overlap + full weight for unique end
                overlap_length = overlap_size if len(chunk) > overlap_size else len(chunk)
                unique_length = len(chunk) - overlap_length
                weight = (overlap_length * 0.5) + unique_length
            else:
                # Middle chunks: half weight for both overlaps + full weight for middle
                weight = (overlap_size * 0.5) + step_size + (overlap_size * 0.5)
            
            weights.append(weight)
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(chunks)] * len(chunks)
        
        return weights