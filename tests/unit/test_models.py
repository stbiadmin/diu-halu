"""Unit tests for data models and schemas."""

import json
from datetime import datetime
from typing import Dict, Any

import pytest
from pydantic import ValidationError

from dodhalueval.models.schemas import (
    Document,
    DocumentChunk,
    Prompt,
    Response,
    EvaluationResult,
    HumanAnnotation,
    PromptResponsePair,
    BenchmarkDataset
)


class TestDocumentChunk:
    """Test DocumentChunk model."""
    
    def test_valid_chunk_creation(self):
        """Test creating a valid document chunk."""
        chunk = DocumentChunk(
            document_id="test-doc-1",
            content="This is test content.",
            page_number=1,
            chunk_index=0
        )
        
        assert chunk.document_id == "test-doc-1"
        assert chunk.content == "This is test content."
        assert chunk.page_number == 1
        assert chunk.chunk_index == 0
        assert chunk.word_count == 4
        assert chunk.char_count == 21
    
    def test_chunk_with_char_positions(self):
        """Test chunk with character positions."""
        chunk = DocumentChunk(
            document_id="test-doc-1",
            content="Test content",
            page_number=1,
            chunk_index=0,
            start_char=0,
            end_char=12
        )
        
        assert chunk.start_char == 0
        assert chunk.end_char == 12
    
    def test_invalid_char_positions(self):
        """Test validation of character positions."""
        with pytest.raises(ValidationError, match="end_char must be greater than start_char"):
            DocumentChunk(
                document_id="test-doc-1",
                content="Test content",
                page_number=1,
                chunk_index=0,
                start_char=10,
                end_char=5
            )
    
    def test_chunk_properties(self):
        """Test chunk computed properties."""
        chunk = DocumentChunk(
            document_id="test-doc-1",
            content="This is a test chunk with multiple words.",
            page_number=1,
            chunk_index=0
        )
        
        assert chunk.word_count == 8
        assert chunk.char_count == 41
    
    def test_chunk_serialization(self):
        """Test chunk serialization."""
        chunk = DocumentChunk(
            document_id="test-doc-1",
            content="Test content",
            page_number=1,
            chunk_index=0
        )
        
        chunk_dict = chunk.to_dict()
        assert "id" in chunk_dict
        assert chunk_dict["content"] == "Test content"
        
        # Test JSON serialization
        chunk_json = chunk.to_json()
        assert isinstance(chunk_json, str)
        
        # Test deserialization
        restored_chunk = DocumentChunk.from_dict(chunk_dict)
        assert restored_chunk.content == chunk.content


class TestDocument:
    """Test Document model."""
    
    def test_valid_document_creation(self, sample_pdf_file):
        """Test creating a valid document."""
        doc = Document(
            title="Test Document",
            source_path=str(sample_pdf_file),
            page_count=1
        )
        
        assert doc.title == "Test Document"
        assert doc.page_count == 1
        assert doc.total_chunks == 0
        assert doc.total_words == 0
    
    def test_document_with_chunks(self, sample_pdf_file):
        """Test document with chunks."""
        chunk1 = DocumentChunk(
            document_id="test-doc",
            content="First chunk content",
            page_number=1,
            chunk_index=0
        )
        chunk2 = DocumentChunk(
            document_id="test-doc",
            content="Second chunk content",
            page_number=1,
            chunk_index=1
        )
        
        doc = Document(
            title="Test Document",
            source_path=str(sample_pdf_file),
            page_count=1,
            content=[chunk1, chunk2]
        )
        
        assert doc.total_chunks == 2
        assert doc.total_words == 6  # 3 words per chunk
    
    def test_document_chunk_retrieval(self, sample_pdf_file):
        """Test document chunk retrieval methods."""
        chunk1 = DocumentChunk(
            id="chunk-1",
            document_id="test-doc",
            content="Page 1 content",
            page_number=1,
            chunk_index=0
        )
        chunk2 = DocumentChunk(
            id="chunk-2",
            document_id="test-doc",
            content="Page 2 content",
            page_number=2,
            chunk_index=1
        )
        
        doc = Document(
            title="Test Document",
            source_path=str(sample_pdf_file),
            page_count=2,
            content=[chunk1, chunk2]
        )
        
        # Test get chunk by ID
        retrieved_chunk = doc.get_chunk_by_id("chunk-1")
        assert retrieved_chunk is not None
        assert retrieved_chunk.content == "Page 1 content"
        
        # Test get chunks by page
        page1_chunks = doc.get_chunks_by_page(1)
        assert len(page1_chunks) == 1
        assert page1_chunks[0].content == "Page 1 content"
    
    def test_invalid_source_path(self):
        """Test validation of source path."""
        with pytest.raises(ValidationError, match="Source file not found"):
            Document(
                title="Test Document",
                source_path="/nonexistent/file.pdf",
                page_count=1
            )


class TestPrompt:
    """Test Prompt model."""
    
    def test_valid_prompt_creation(self):
        """Test creating a valid prompt."""
        prompt = Prompt(
            text="What are the key principles?",
            source_document_id="doc-1",
            generation_strategy="template"
        )
        
        assert prompt.text == "What are the key principles?"
        assert prompt.source_document_id == "doc-1"
        assert prompt.generation_strategy == "template"
        assert prompt.word_count == 5
    
    def test_prompt_validation(self):
        """Test prompt validation rules."""
        # Empty text should fail
        with pytest.raises(ValidationError, match="Prompt text cannot be empty"):
            Prompt(
                text="   ",
                source_document_id="doc-1",
                generation_strategy="template"
            )
        
        # Very long text should fail
        long_text = "x" * 10001
        with pytest.raises(ValidationError, match="Prompt text too long"):
            Prompt(
                text=long_text,
                source_document_id="doc-1",
                generation_strategy="template"
            )
    
    def test_prompt_with_metadata(self):
        """Test prompt with optional fields."""
        prompt = Prompt(
            text="What are the procedures?",
            source_document_id="doc-1",
            source_chunk_id="chunk-1",
            expected_answer="The procedures are...",
            hallucination_type="factual",
            generation_strategy="llm_based",
            difficulty_level="hard",
            metadata={"template_id": "factual_1"}
        )
        
        assert prompt.expected_answer == "The procedures are..."
        assert prompt.hallucination_type == "factual"
        assert prompt.difficulty_level == "hard"
        assert prompt.metadata["template_id"] == "factual_1"


class TestResponse:
    """Test Response model."""
    
    def test_valid_response_creation(self):
        """Test creating a valid response."""
        response = Response(
            prompt_id="prompt-1",
            text="The key principles are leadership and training.",
            model="gpt-4",
            provider="openai"
        )
        
        assert response.prompt_id == "prompt-1"
        assert response.text == "The key principles are leadership and training."
        assert response.model == "gpt-4"
        assert response.provider == "openai"
        assert response.word_count == 7
    
    def test_response_validation(self):
        """Test response validation."""
        # Empty text should fail
        with pytest.raises(ValidationError, match="Response text cannot be empty"):
            Response(
                prompt_id="prompt-1",
                text="   ",
                model="gpt-4",
                provider="openai"
            )
    
    def test_response_with_metadata(self):
        """Test response with additional fields."""
        response = Response(
            prompt_id="prompt-1",
            text="Response text",
            model="gpt-4",
            provider="openai",
            generation_params={"temperature": 0.7},
            contains_hallucination=True,
            hallucination_score=0.8,
            metadata={"api_version": "2023-05-15"}
        )
        
        assert response.generation_params["temperature"] == 0.7
        assert response.contains_hallucination is True
        assert response.hallucination_score == 0.8


class TestEvaluationResult:
    """Test EvaluationResult model."""
    
    def test_valid_evaluation_creation(self):
        """Test creating a valid evaluation result."""
        evaluation = EvaluationResult(
            response_id="response-1",
            method="vectara_hhem",
            is_hallucinated=False,
            confidence_score=0.85
        )
        
        assert evaluation.response_id == "response-1"
        assert evaluation.method == "vectara_hhem"
        assert evaluation.is_hallucinated is False
        assert evaluation.confidence_score == 0.85
    
    def test_evaluation_method_validation(self):
        """Test evaluation method validation."""
        # Valid method
        evaluation = EvaluationResult(
            response_id="response-1",
            method="g_eval",
            is_hallucinated=True,
            confidence_score=0.7
        )
        assert evaluation.method == "g_eval"
        
        # Invalid method
        with pytest.raises(ValidationError, match="Invalid evaluation method"):
            EvaluationResult(
                response_id="response-1",
                method="invalid_method",
                is_hallucinated=False,
                confidence_score=0.5
            )
    
    def test_confidence_score_validation(self):
        """Test confidence score validation."""
        # Valid scores
        EvaluationResult(
            response_id="response-1",
            method="vectara_hhem",
            is_hallucinated=False,
            confidence_score=0.0
        )
        
        EvaluationResult(
            response_id="response-1",
            method="vectara_hhem",
            is_hallucinated=False,
            confidence_score=1.0
        )
        
        # Invalid scores
        with pytest.raises(ValidationError):
            EvaluationResult(
                response_id="response-1",
                method="vectara_hhem",
                is_hallucinated=False,
                confidence_score=-0.1
            )
        
        with pytest.raises(ValidationError):
            EvaluationResult(
                response_id="response-1",
                method="vectara_hhem",
                is_hallucinated=False,
                confidence_score=1.1
            )


class TestHumanAnnotation:
    """Test HumanAnnotation model."""
    
    def test_valid_annotation_creation(self):
        """Test creating a valid human annotation."""
        annotation = HumanAnnotation(
            response_id="response-1",
            annotator_id="expert-1",
            is_hallucinated=True,
            severity="major",
            confidence=0.9
        )
        
        assert annotation.response_id == "response-1"
        assert annotation.annotator_id == "expert-1"
        assert annotation.is_hallucinated is True
        assert annotation.severity == "major"
        assert annotation.confidence == 0.9
    
    def test_annotation_with_details(self):
        """Test annotation with optional details."""
        annotation = HumanAnnotation(
            response_id="response-1",
            annotator_id="expert-1",
            is_hallucinated=True,
            severity="critical",
            explanation="This contains factual errors.",
            highlighted_text="incorrect statement",
            correction="correct statement",
            confidence=0.95,
            time_spent_seconds=300
        )
        
        assert annotation.explanation == "This contains factual errors."
        assert annotation.highlighted_text == "incorrect statement"
        assert annotation.correction == "correct statement"
        assert annotation.time_spent_seconds == 300


class TestPromptResponsePair:
    """Test PromptResponsePair model."""
    
    def test_valid_pair_creation(self, sample_prompt: Prompt, sample_response: Response):
        """Test creating a valid prompt-response pair."""
        # Ensure response references the prompt
        sample_response.prompt_id = sample_prompt.id
        
        pair = PromptResponsePair(
            prompt=sample_prompt,
            response=sample_response
        )
        
        assert pair.prompt.id == sample_prompt.id
        assert pair.response.id == sample_response.id
        assert len(pair.evaluations) == 0
        assert len(pair.human_annotations) == 0
    
    def test_pair_validation(self, sample_prompt: Prompt, sample_response: Response):
        """Test pair validation."""
        # Mismatched prompt-response should fail
        sample_response.prompt_id = "different-prompt-id"
        
        with pytest.raises(ValidationError, match="Response prompt_id does not match"):
            PromptResponsePair(
                prompt=sample_prompt,
                response=sample_response
            )
    
    def test_pair_with_evaluations(
        self,
        sample_prompt: Prompt,
        sample_response: Response,
        sample_evaluation_result: EvaluationResult
    ):
        """Test pair with evaluations."""
        sample_response.prompt_id = sample_prompt.id
        sample_evaluation_result.response_id = sample_response.id
        
        pair = PromptResponsePair(
            prompt=sample_prompt,
            response=sample_response,
            evaluations=[sample_evaluation_result]
        )
        
        assert len(pair.evaluations) == 1
        assert pair.consensus_hallucination_score == 0.85
        
        # Test get evaluation by method
        eval_result = pair.get_evaluation_by_method("vectara_hhem")
        assert eval_result is not None
        assert eval_result.method == "vectara_hhem"
    
    def test_pair_with_annotations(
        self,
        sample_prompt: Prompt,
        sample_response: Response,
        sample_human_annotation: HumanAnnotation
    ):
        """Test pair with human annotations."""
        sample_response.prompt_id = sample_prompt.id
        sample_human_annotation.response_id = sample_response.id
        
        pair = PromptResponsePair(
            prompt=sample_prompt,
            response=sample_response,
            human_annotations=[sample_human_annotation]
        )
        
        assert len(pair.human_annotations) == 1
        assert pair.has_human_annotation is True


class TestBenchmarkDataset:
    """Test BenchmarkDataset model."""
    
    def test_valid_dataset_creation(self):
        """Test creating a valid benchmark dataset."""
        dataset = BenchmarkDataset(
            name="Test Dataset",
            version="1.0",
            description="A test dataset"
        )
        
        assert dataset.name == "Test Dataset"
        assert dataset.version == "1.0"
        assert dataset.total_pairs == 0
        assert dataset.total_documents == 0
        assert dataset.hallucination_rate == 0.0
    
    def test_dataset_with_data(
        self,
        sample_document: Document,
        sample_prompt_response_pair: PromptResponsePair
    ):
        """Test dataset with documents and pairs."""
        dataset = BenchmarkDataset(
            name="Test Dataset",
            version="1.0",
            documents=[sample_document],
            pairs=[sample_prompt_response_pair]
        )
        
        assert dataset.total_pairs == 1
        assert dataset.total_documents == 1
    
    def test_dataset_filtering(
        self,
        sample_document: Document,
        sample_prompt_response_pair: PromptResponsePair
    ):
        """Test dataset filtering methods."""
        # Create pairs with different types
        sample_prompt_response_pair.prompt.hallucination_type = "factual"
        sample_prompt_response_pair.prompt.difficulty_level = "medium"
        
        dataset = BenchmarkDataset(
            name="Test Dataset",
            version="1.0",
            documents=[sample_document],
            pairs=[sample_prompt_response_pair]
        )
        
        # Test filtering by hallucination type
        factual_pairs = dataset.get_pairs_by_hallucination_type("factual")
        assert len(factual_pairs) == 1
        
        logical_pairs = dataset.get_pairs_by_hallucination_type("logical")
        assert len(logical_pairs) == 0
        
        # Test filtering by difficulty
        medium_pairs = dataset.get_pairs_by_difficulty("medium")
        assert len(medium_pairs) == 1
    
    def test_halueval_export(
        self,
        sample_document: Document,
        sample_prompt_response_pair: PromptResponsePair
    ):
        """Test exporting to HaluEval format."""
        dataset = BenchmarkDataset(
            name="Test Dataset",
            version="1.0",
            documents=[sample_document],
            pairs=[sample_prompt_response_pair]
        )
        
        halueval_data = dataset.export_halueval_format()
        
        assert len(halueval_data) == 1
        assert "id" in halueval_data[0]
        assert "prompt" in halueval_data[0]
        assert "response" in halueval_data[0]
        assert "metadata" in halueval_data[0]
    
    def test_dataset_serialization(self, tmp_path):
        """Test dataset file operations."""
        dataset = BenchmarkDataset(
            name="Test Dataset",
            version="1.0",
            description="Test serialization"
        )
        
        # Test save to file
        output_file = tmp_path / "test_dataset.jsonl"
        dataset.save_to_file(output_file, format="jsonl")
        
        assert output_file.exists()
        
        # Test JSON format
        json_file = tmp_path / "test_dataset.json"
        dataset.save_to_file(json_file, format="json")
        
        assert json_file.exists()
        
        # Test invalid format
        with pytest.raises(ValueError, match="Unsupported format"):
            dataset.save_to_file(tmp_path / "test.txt", format="txt")