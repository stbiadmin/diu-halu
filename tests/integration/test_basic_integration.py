"""Basic integration tests to verify core functionality."""

import pytest
from pathlib import Path

from dodhalueval.models.schemas import DocumentChunk, Prompt, Response
from dodhalueval.models.config import DoDHaluEvalConfig


@pytest.mark.integration
class TestBasicIntegration:
    """Basic integration tests."""

    def test_schema_integration(self):
        """Test that schemas work together correctly."""
        # Create a document chunk
        chunk = DocumentChunk(
            document_id="test-doc",
            content="This is test content for integration testing.",
            page_number=1,
            chunk_index=0
        )
        
        # Create a prompt using the chunk
        prompt = Prompt(
            text="What is this content about?",
            source_document_id=chunk.document_id,
            source_chunk_id=chunk.id,
            expected_answer="Integration testing",
            generation_strategy="manual"
        )
        
        # Create a response to the prompt
        response = Response(
            prompt_id=prompt.id,
            text="This content is about integration testing.",
            model="test-model",
            provider="mock"
        )
        
        # Verify relationships
        assert prompt.source_document_id == chunk.document_id
        assert response.prompt_id == prompt.id
        
        # Verify all objects are valid
        assert chunk.id is not None
        assert prompt.id is not None
        assert response.id is not None

    def test_config_loading(self):
        """Test that configuration loading works."""
        config = DoDHaluEvalConfig()
        
        # Verify default configuration is valid
        assert config.environment == "development"
        assert config.batch_size > 0
        assert config.max_concurrent_requests > 0

    def test_test_fixtures_work(self, test_data_generator):
        """Test that our test fixtures work correctly."""
        # Test the test data generator
        chunk = test_data_generator.generate_document_chunk(seed=42)
        assert isinstance(chunk, DocumentChunk)
        assert chunk.content is not None
        assert len(chunk.content) > 0
        
        prompt = test_data_generator.generate_prompt(seed=42)
        assert isinstance(prompt, Prompt)
        assert prompt.text is not None
        assert len(prompt.text) > 0
        
        response = test_data_generator.generate_response(seed=42)
        assert isinstance(response, Response)
        assert response.text is not None
        assert len(response.text) > 0

    def test_batch_fixtures(self, batch_document_chunks, batch_prompts):
        """Test batch fixtures work correctly."""
        assert len(batch_document_chunks) == 10
        assert all(isinstance(chunk, DocumentChunk) for chunk in batch_document_chunks)
        
        assert len(batch_prompts) == 5
        assert all(isinstance(prompt, Prompt) for prompt in batch_prompts)

    def test_environment_setup(self, temp_dir):
        """Test that test environment is set up correctly."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test we can create files in temp directory
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"


@pytest.mark.integration
class TestConfigurationIntegration:
    """Test configuration system integration."""

    def test_config_validation_integration(self):
        """Test configuration validation with real data."""
        # Test with valid configuration
        config_data = {
            "environment": "testing",
            "batch_size": 10,
            "max_concurrent_requests": 2,
            "pdf_processing": {
                "chunk_size": 500,
                "chunk_overlap": 100
            },
            "api_configs": {
                "openai": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key"
                }
            }
        }
        
        config = DoDHaluEvalConfig(**config_data)
        assert config.environment == "testing"
        assert config.batch_size == 10
        assert config.pdf_processing.chunk_size == 500

    def test_config_with_test_fixtures(self, test_config):
        """Test configuration with test fixtures."""
        assert test_config.environment == "testing"
        assert test_config.pdf_processing.cache_enabled is False
        assert test_config.logging.level == "ERROR"


@pytest.mark.integration  
class TestSchemaValidation:
    """Test schema validation integration."""

    def test_document_chunk_validation(self):
        """Test document chunk validation."""
        # Valid chunk
        chunk = DocumentChunk(
            document_id="doc-123",
            content="Valid content",
            page_number=1,
            chunk_index=0
        )
        assert chunk.document_id == "doc-123"
        assert chunk.page_number == 1
        
        # Test with char positions
        chunk_with_positions = DocumentChunk(
            document_id="doc-123",
            content="Content with positions",
            page_number=1,
            chunk_index=0,
            start_char=0,
            end_char=22
        )
        assert chunk_with_positions.start_char == 0
        assert chunk_with_positions.end_char == 22

    def test_prompt_validation(self):
        """Test prompt validation."""
        prompt = Prompt(
            text="What is the main topic?",
            source_document_id="doc-123",
            source_chunk_id="chunk-456",
            expected_answer="The main topic is testing",
            generation_strategy="template"
        )
        
        assert prompt.text == "What is the main topic?"
        assert prompt.source_document_id == "doc-123"
        assert prompt.hallucination_type is None  # Default value

    def test_response_validation(self):
        """Test response validation."""
        response = Response(
            prompt_id="prompt-123",
            text="The main topic is testing and validation.",
            model="gpt-3.5-turbo",
            provider="openai"
        )
        
        assert response.prompt_id == "prompt-123"
        assert response.model == "gpt-3.5-turbo"
        assert response.contains_hallucination is None  # Default


@pytest.mark.integration
@pytest.mark.slow
class TestDataFlow:
    """Test data flow through the system."""

    def test_document_to_prompt_to_response_flow(self, test_data_generator):
        """Test the complete data flow."""
        # Create a document chunk
        chunk = test_data_generator.generate_document_chunk(
            doc_id="flow-test",
            content_length=100,
            seed=12345
        )
        
        # Generate a prompt from the chunk
        prompt = test_data_generator.generate_prompt(
            prompt_type="qa",
            seed=12345
        )
        
        # Make the prompt reference the chunk
        prompt.source_document_id = chunk.document_id
        prompt.source_chunk_id = chunk.id
        
        # Generate a response to the prompt
        response = test_data_generator.generate_response(
            has_hallucination=False,
            prompt=prompt,
            seed=12345
        )
        
        # Verify the flow
        assert response.prompt_id == prompt.id
        assert prompt.source_document_id == chunk.document_id
        assert prompt.source_chunk_id == chunk.id
        
        # Verify consistency
        assert all(obj.id is not None for obj in [chunk, prompt, response])
        assert all(len(obj.id) > 0 for obj in [chunk, prompt, response])

    def test_deterministic_generation(self, test_data_generator):
        """Test that generation is deterministic with same seed."""
        seed = 98765
        
        # Generate same data twice
        chunk1 = test_data_generator.generate_document_chunk(seed=seed)
        chunk2 = test_data_generator.generate_document_chunk(seed=seed)
        
        prompt1 = test_data_generator.generate_prompt(seed=seed)
        prompt2 = test_data_generator.generate_prompt(seed=seed)
        
        # Should be identical (except IDs which are auto-generated)
        assert chunk1.content == chunk2.content
        assert chunk1.document_id == chunk2.document_id
        
        assert prompt1.text == prompt2.text
        assert prompt1.hallucination_type == prompt2.hallucination_type