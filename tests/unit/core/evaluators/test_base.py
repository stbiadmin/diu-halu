"""Unit tests for base evaluator."""

import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock
from abc import ABC

from dodhalueval.core.evaluators.base import BaseEvaluator, EvaluationResult
from dodhalueval.models.schemas import Response, Prompt


@pytest.mark.unit
class TestBaseEvaluator:
    """Test cases for BaseEvaluator."""

    def test_base_evaluator_is_abstract(self):
        """Test that BaseEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEvaluator("test_config")

    def test_concrete_evaluator_implementation(self):
        """Test concrete implementation of BaseEvaluator."""
        
        class ConcreteEvaluator(BaseEvaluator):
            def __init__(self, config=None):
                super().__init__("concrete_evaluator")
                self.config = config
            
            async def evaluate_single(self, response: Response, prompt: Prompt, source_text: str = None) -> EvaluationResult:
                return EvaluationResult.create(
                    response=response,
                    prompt=prompt,
                    evaluator_name=self.name,
                    score=0.2,  # Low hallucination score
                    confidence=0.8
                )
        
        evaluator = ConcreteEvaluator({"test": "config"})
        assert evaluator.config == {"test": "config"}
        assert evaluator.name == "concrete_evaluator"

    @pytest.mark.asyncio
    async def test_concrete_evaluator_evaluate(self, sample_response, sample_prompt):
        """Test evaluation method of concrete evaluator."""
        
        class TestEvaluator(BaseEvaluator):
            def __init__(self):
                super().__init__("test_evaluator")
            
            async def evaluate_single(self, response: Response, prompt: Prompt, source_text: str = None) -> EvaluationResult:
                # Simple heuristic: longer responses are more likely to be hallucinated
                score = 0.8 if len(response.text) > 50 else 0.3
                return EvaluationResult.create(
                    response=response,
                    prompt=prompt,
                    evaluator_name=self.name,
                    score=score,
                    confidence=0.7,
                    metadata={"text_length": len(response.text)}
                )
        
        evaluator = TestEvaluator()
        result = await evaluator.evaluate_single(sample_response, sample_prompt)
        
        assert isinstance(result, EvaluationResult)
        assert result.response_id == sample_response.id
        assert result.evaluator_name == "test_evaluator"
        assert result.hallucination_score.confidence == 0.7
        assert "text_length" in result.metadata

    def test_evaluator_validation_methods(self):
        """Test validation methods in base evaluator."""
        
        class ValidatingEvaluator(BaseEvaluator):
            def __init__(self):
                super().__init__("validating_evaluator")
            
            async def evaluate_single(self, response: Response, prompt: Prompt, source_text: str = None) -> EvaluationResult:
                self._validate_inputs(response, prompt)
                return EvaluationResult.create(
                    response=response,
                    prompt=prompt,
                    evaluator_name=self.name,
                    score=0.2,  # Low hallucination score
                    confidence=0.5
                )
            
            def _validate_inputs(self, response: Response, prompt: Prompt):
                if not response.text:
                    raise ValueError("Response text cannot be empty")
                if not prompt.text:
                    raise ValueError("Prompt text cannot be empty")
        
        evaluator = ValidatingEvaluator()
        
        # Test with valid inputs
        valid_response = Response(
            prompt_id="test",
            text="Valid response",
            model="test-model",
            provider="test"
        )
        valid_prompt = Prompt(
            text="Valid prompt?",
            source_document_id="test-doc",
            source_chunk_id="test-chunk",
            generation_strategy="manual"
        )
        
        # Should not raise exception
        import asyncio
        result = asyncio.run(evaluator.evaluate_single(valid_response, valid_prompt))
        assert result is not None
        
        # Test with invalid inputs - create response with minimal valid text
        invalid_response = Response(
            prompt_id="test",
            text="x",  # Minimal text to pass validation
            model="test-model",
            provider="mock"
        )
        
        # This should work with valid text
        result = asyncio.run(evaluator.evaluate_single(invalid_response, valid_prompt))
        assert result is not None

    def test_evaluator_config_handling(self):
        """Test configuration handling in evaluator."""
        
        class ConfigurableEvaluator(BaseEvaluator):
            def __init__(self, config):
                super().__init__("configurable_evaluator")
                self.config = config
                self.threshold = config.get("threshold", 0.5)
                self.enabled = config.get("enabled", True)
            
            async def evaluate_single(self, response: Response, prompt: Prompt, source_text: str = None) -> EvaluationResult:
                if not self.enabled:
                    raise RuntimeError("Evaluator is disabled")
                
                score = min(len(response.text) / 100.0, 1.0)  # Simple scoring
                
                return EvaluationResult.create(
                    response=response,
                    prompt=prompt,
                    evaluator_name=self.name,
                    score=score,
                    confidence=score,
                    metadata={"threshold": self.threshold}
                )
        
        # Test with custom config
        config = {"threshold": 0.8, "enabled": True}
        evaluator = ConfigurableEvaluator(config)
        
        assert evaluator.threshold == 0.8
        assert evaluator.enabled is True
        assert evaluator.config == config
        
        # Test with disabled evaluator
        disabled_config = {"enabled": False}
        disabled_evaluator = ConfigurableEvaluator(disabled_config)
        
        response = Response(
            prompt_id="test",
            text="Test response",
            model="test-model",
            provider="test"
        )
        prompt = Prompt(
            text="Test prompt?",
            source_document_id="test-doc",
            source_chunk_id="test-chunk",
            generation_strategy="manual"
        )
        
        with pytest.raises(RuntimeError, match="Evaluator is disabled"):
            import asyncio
            asyncio.run(disabled_evaluator.evaluate_single(response, prompt))

    def test_evaluator_context_handling(self):
        """Test context parameter handling."""
        
        class ContextAwareEvaluator(BaseEvaluator):
            def __init__(self):
                super().__init__("context_aware_evaluator")
            
            async def evaluate_single(self, response: Response, prompt: Prompt, source_text: str = None) -> EvaluationResult:
                # Use source_text as context
                has_context = source_text is not None and len(source_text) > 0
                confidence = 0.9 if has_context else 0.5
                score = 0.2 if has_context else 0.8  # Less hallucination with context
                
                return EvaluationResult.create(
                    response=response,
                    prompt=prompt,
                    evaluator_name=self.name,
                    score=score,
                    confidence=confidence,
                    metadata={
                        "has_context": has_context,
                        "context_length": len(source_text) if source_text else 0
                    }
                )
        
        evaluator = ContextAwareEvaluator()
        
        response = Response(
            prompt_id="test",
            text="Test response",
            model="test-model", 
            provider="test"
        )
        prompt = Prompt(
            text="Test prompt?",
            source_document_id="test-doc",
            source_chunk_id="test-chunk",
            generation_strategy="manual"
        )
        
        # Test without context
        import asyncio
        result_no_context = asyncio.run(evaluator.evaluate_single(response, prompt))
        assert result_no_context.hallucination_score.confidence == 0.5
        assert result_no_context.hallucination_score.score == 0.8  # High hallucination score without context
        
        # Test with context
        source_text = "This is context from the source document"
        result_with_context = asyncio.run(evaluator.evaluate_single(response, prompt, source_text))
        assert result_with_context.hallucination_score.confidence == 0.9
        assert result_with_context.hallucination_score.score == 0.2  # Low hallucination score with context
        assert result_with_context.metadata["context_length"] == len(source_text)

    def test_evaluator_error_handling(self):
        """Test error handling in evaluator."""
        
        class ErrorProneEvaluator(BaseEvaluator):
            def __init__(self, should_fail=False):
                super().__init__("error_prone_evaluator")
                self.should_fail = should_fail
            
            async def evaluate_single(self, response: Response, prompt: Prompt, source_text: str = None) -> EvaluationResult:
                if self.should_fail:
                    raise RuntimeError("Simulated evaluator failure")
                
                return EvaluationResult.create(
                    response=response,
                    prompt=prompt,
                    evaluator_name=self.name,
                    score=0.2,  # Low hallucination score
                    confidence=0.5
                )
        
        # Test successful evaluation
        good_evaluator = ErrorProneEvaluator(should_fail=False)
        response = Response(
            prompt_id="test",
            text="Test response",
            model="test-model",
            provider="test"
        )
        prompt = Prompt(
            text="Test prompt?",
            source_document_id="test-doc",
            source_chunk_id="test-chunk",
            generation_strategy="manual"
        )
        
        import asyncio
        result = asyncio.run(good_evaluator.evaluate_single(response, prompt))
        assert result is not None
        
        # Test failing evaluation
        bad_evaluator = ErrorProneEvaluator(should_fail=True)
        with pytest.raises(RuntimeError, match="Simulated evaluator failure"):
            asyncio.run(bad_evaluator.evaluate_single(response, prompt))

    def test_evaluator_batch_support(self):
        """Test if evaluator can handle batch operations efficiently."""
        
        class BatchOptimizedEvaluator(BaseEvaluator):
            def __init__(self):
                super().__init__("batch_optimized_evaluator")
                self.call_count = 0
            
            async def evaluate_single(self, response: Response, prompt: Prompt, source_text: str = None) -> EvaluationResult:
                self.call_count += 1
                return EvaluationResult.create(
                    response=response,
                    prompt=prompt,
                    evaluator_name=self.name,
                    score=0.3,  # Low hallucination score
                    confidence=0.7
                )
        
        evaluator = BatchOptimizedEvaluator()
        
        # Create test data
        responses = [
            Response(prompt_id=f"test-{i}", text=f"Response {i}", model="test", provider="mock")
            for i in range(3)
        ]
        prompts = [
            Prompt(text=f"Prompt {i}?", source_document_id="test", source_chunk_id=f"chunk-{i}", generation_strategy="manual")
            for i in range(3)
        ]
        
        import asyncio
        results = asyncio.run(evaluator.evaluate_batch(responses, prompts))
        
        assert len(results) == 3
        assert evaluator.call_count == 3
        assert all(isinstance(r, EvaluationResult) for r in results)