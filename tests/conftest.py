"""Pytest configuration and fixtures for DoDHaluEval tests."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
from datetime import datetime
from unittest.mock import Mock

import pytest
from PyPDF2 import PdfWriter

from dodhalueval.models.config import DoDHaluEvalConfig, LoggingConfig
from dodhalueval.models.schemas import (
    Document,
    DocumentChunk,
    Prompt,
    Response,
    EvaluationResult,
    HumanAnnotation,
    PromptResponsePair
)
from dodhalueval.data.pdf_processor import PDFProcessor
from dodhalueval.utils.logger import get_logger


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_config(temp_dir: Path) -> DoDHaluEvalConfig:
    """Create a test configuration."""
    return DoDHaluEvalConfig(
        environment="testing",
        pdf_processing={
            "chunk_size": 100,
            "chunk_overlap": 20,
            "cache_enabled": False,
            "cache_dir": str(temp_dir / "pdf_cache"),
            "max_pages": 2
        },
        cache={
            "enabled": False,
            "pdf_cache_dir": str(temp_dir / "pdf_cache"),
            "llm_cache_dir": str(temp_dir / "llm_cache"),
            "max_cache_size_mb": 10,
            "ttl_hours": 1
        },
        logging=LoggingConfig(
            level="ERROR",
            console_output=False,
            log_file=None
        ),
        batch_size=2,
        max_concurrent_requests=1,
        data_dir=str(temp_dir / "data"),
        source_documents_dir=str(temp_dir / "docs"),
        processed_documents_dir=str(temp_dir / "processed")
    )


@pytest.fixture
def sample_document_chunk() -> DocumentChunk:
    """Create a sample document chunk for testing."""
    return DocumentChunk(
        document_id="test-doc-1",
        content="This is a sample chunk of text from a DoD document. It contains information about military procedures and guidelines.",
        page_number=1,
        chunk_index=0,
        start_char=0,
        end_char=100,
        section="Chapter 1",
        metadata={
            "source": "test_document.pdf",
            "extraction_method": "PyPDF2"
        }
    )


@pytest.fixture
def sample_document(sample_document_chunk: DocumentChunk, temp_dir: Path) -> Document:
    """Create a sample document for testing."""
    # Create a dummy PDF file
    pdf_path = temp_dir / "test_document.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create minimal PDF content
    pdf_writer = PdfWriter()
    # Add a blank page
    from PyPDF2.generic import RectangleObject
    pdf_writer.add_blank_page(width=612, height=792)  # Standard letter size
    
    with open(pdf_path, 'wb') as f:
        pdf_writer.write(f)
    
    return Document(
        title="Test DoD Document",
        source_path=str(pdf_path),
        file_hash="test-hash-123",
        page_count=1,
        content=[sample_document_chunk],
        metadata={
            "author": "DoD Test Division",
            "creation_date": "2024-01-01",
            "classification": "Unclassified"
        }
    )


@pytest.fixture
def sample_prompt(sample_document: Document) -> Prompt:
    """Create a sample prompt for testing."""
    return Prompt(
        text="What are the key principles mentioned in this document?",
        source_document_id=sample_document.id,
        source_chunk_id=sample_document.content[0].id,
        expected_answer="The key principles include leadership, training, and operational readiness.",
        hallucination_type="factual",
        generation_strategy="template",
        difficulty_level="medium",
        metadata={
            "template_id": "factual_question_1",
            "keywords": ["principles", "leadership", "training"]
        }
    )


@pytest.fixture
def sample_response(sample_prompt: Prompt) -> Response:
    """Create a sample response for testing."""
    return Response(
        prompt_id=sample_prompt.id,
        text="According to the document, the key principles are leadership, comprehensive training programs, and maintaining high levels of operational readiness at all times.",
        model="gpt-3.5-turbo",
        provider="openai",
        generation_params={
            "temperature": 0.7,
            "max_tokens": 150
        },
        contains_hallucination=False,
        hallucination_score=0.2,
        metadata={
            "api_version": "2023-05-15",
            "request_id": "test-req-123"
        }
    )


@pytest.fixture
def sample_evaluation_result(sample_response: Response) -> EvaluationResult:
    """Create a sample evaluation result for testing."""
    return EvaluationResult(
        response_id=sample_response.id,
        method="vectara_hhem",
        is_hallucinated=False,
        confidence_score=0.85,
        hallucination_type=None,
        details={
            "raw_score": 0.85,
            "threshold": 0.5,
            "model_version": "hhem-v1.0"
        },
        metadata={
            "evaluation_time_ms": 250,
            "api_version": "v1"
        }
    )


@pytest.fixture
def sample_human_annotation(sample_response: Response) -> HumanAnnotation:
    """Create a sample human annotation for testing."""
    return HumanAnnotation(
        response_id=sample_response.id,
        annotator_id="expert-001",
        is_hallucinated=False,
        severity="minor",
        explanation="The response accurately reflects the document content with minor stylistic differences.",
        highlighted_text=None,
        correction=None,
        confidence=0.9,
        time_spent_seconds=120,
        metadata={
            "annotator_expertise": "military_doctrine",
            "annotation_round": 1
        }
    )


@pytest.fixture
def sample_prompt_response_pair(
    sample_prompt: Prompt,
    sample_response: Response,
    sample_evaluation_result: EvaluationResult,
    sample_human_annotation: HumanAnnotation
) -> PromptResponsePair:
    """Create a sample prompt-response pair for testing."""
    return PromptResponsePair(
        prompt=sample_prompt,
        response=sample_response,
        evaluations=[sample_evaluation_result],
        human_annotations=[sample_human_annotation]
    )


@pytest.fixture
def mock_pdf_processor(test_config: DoDHaluEvalConfig) -> Mock:
    """Create a mock PDF processor for testing."""
    mock_processor = Mock(spec=PDFProcessor)
    mock_processor.chunk_size = test_config.pdf_processing.chunk_size
    mock_processor.chunk_overlap = test_config.pdf_processing.chunk_overlap
    mock_processor.cache_enabled = test_config.pdf_processing.cache_enabled
    
    # Mock extract_text method
    mock_processor.extract_text.return_value = [
        type('PageContent', (), {
            'page_number': 1,
            'text': 'Sample text from page 1',
            'raw_text': 'Sample text from page 1',
            'metadata': {'page_number': 1},
            'structure': {'headings': [], 'sections': []}
        })()
    ]
    
    return mock_processor


@pytest.fixture
def sample_pdf_file(temp_dir: Path) -> Path:
    """Create a sample PDF file for testing."""
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a simple PDF with some text content
    pdf_writer = PdfWriter()
    pdf_writer.add_blank_page(width=612, height=792)
    
    with open(pdf_path, 'wb') as f:
        pdf_writer.write(f)
    
    return pdf_path


@pytest.fixture
def sample_config_data() -> Dict[str, Any]:
    """Sample configuration data for testing."""
    return {
        "version": "0.1.0",
        "environment": "testing",
        "pdf_processing": {
            "chunk_size": 500,
            "chunk_overlap": 100,
            "cache_enabled": False
        },
        "api_configs": {
            "openai": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "api_key": "test-key",
                "max_retries": 2,
                "timeout": 10
            }
        },
        "evaluation_methods": [
            {
                "method": "vectara_hhem",
                "enabled": True,
                "confidence_threshold": 0.5,
                "batch_size": 5
            }
        ],
        "batch_size": 10,
        "max_concurrent_requests": 2
    }


@pytest.fixture
def logger_for_tests():
    """Get a logger configured for testing."""
    return get_logger("test", LoggingConfig(level="ERROR", console_output=False))


@pytest.fixture(autouse=True)
def setup_test_environment(temp_dir: Path):
    """Automatically setup test environment for each test."""
    # Set environment variables for testing
    os.environ["DODHALUEVAL_ENVIRONMENT"] = "testing"
    os.environ["DODHALUEVAL_CACHE_ENABLED"] = "false"
    os.environ["DODHALUEVAL_LOGGING_LEVEL"] = "ERROR"
    
    # Ensure test directories exist
    (temp_dir / "data").mkdir(exist_ok=True)
    (temp_dir / "docs").mkdir(exist_ok=True)
    (temp_dir / "processed").mkdir(exist_ok=True)
    (temp_dir / "cache").mkdir(exist_ok=True)
    
    yield
    
    # Clean up environment variables
    test_env_vars = [k for k in os.environ.keys() if k.startswith("DODHALUEVAL_")]
    for var in test_env_vars:
        os.environ.pop(var, None)


# Pytest markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring external API access"
    )


# Skip slow tests by default
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runapi", action="store_true", default=False, help="run tests requiring API access"
    )