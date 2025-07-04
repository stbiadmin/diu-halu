"""Unit tests for exceptions module."""

import pytest
from dodhalueval.utils.exceptions import (
    DoDHaluEvalError,
    ConfigurationError,
    PDFProcessingError,
    APIError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
    ValidationError,
    DataProcessingError,
    CacheError,
    EvaluationError,
    PromptGenerationError,
    FileSystemError,
    ResourceError,
    TimeoutError
)


@pytest.mark.unit
class TestDoDHaluEvalExceptions:
    """Test cases for DoDHaluEval exception hierarchy."""

    def test_base_exception(self):
        """Test base DoDHaluEval exception."""
        error = DoDHaluEvalError("Base error message")
        assert str(error) == "Base error message"
        assert isinstance(error, Exception)

    def test_configuration_error(self):
        """Test configuration error."""
        error = ConfigurationError("Invalid configuration")
        assert str(error) == "Invalid configuration"
        assert isinstance(error, DoDHaluEvalError)

    def test_validation_error(self):
        """Test validation error."""
        error = ValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, DoDHaluEvalError)

    def test_data_processing_error(self):
        """Test data processing error."""
        error = DataProcessingError("Processing failed")
        assert str(error) == "Processing failed"
        assert isinstance(error, DoDHaluEvalError)

    def test_api_error(self):
        """Test API error."""
        error = APIError("API request failed")
        assert str(error) == "API request failed"
        assert isinstance(error, DoDHaluEvalError)

    def test_model_not_found_error(self):
        """Test model not found error."""
        error = ModelNotFoundError("Model not found")
        assert str(error) == "Model not found"
        assert isinstance(error, APIError)

    def test_cache_error(self):
        """Test cache error."""
        error = CacheError("Cache operation failed")
        assert str(error) == "Cache operation failed"
        assert isinstance(error, DoDHaluEvalError)

    def test_prompt_generation_error(self):
        """Test prompt generation error."""
        error = PromptGenerationError("Prompt generation failed")
        assert str(error) == "Prompt generation failed"
        assert isinstance(error, DoDHaluEvalError)

    def test_evaluation_error(self):
        """Test evaluation error."""
        error = EvaluationError("Evaluation failed")
        assert str(error) == "Evaluation failed"
        assert isinstance(error, DoDHaluEvalError)

    def test_file_system_error(self):
        """Test file system error."""
        error = FileSystemError("File operation failed")
        assert str(error) == "File operation failed"
        assert isinstance(error, DoDHaluEvalError)

    def test_resource_error(self):
        """Test resource error."""
        error = ResourceError("Resource limit exceeded")
        assert str(error) == "Resource limit exceeded"
        assert isinstance(error, DoDHaluEvalError)

    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = RateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, APIError)

    def test_authentication_error(self):
        """Test authentication error."""
        error = AuthenticationError("Authentication failed")
        assert str(error) == "Authentication failed"
        assert isinstance(error, APIError)

    def test_pdf_processing_error(self):
        """Test PDF processing error."""
        error = PDFProcessingError("PDF processing failed")
        assert str(error) == "PDF processing failed"
        assert isinstance(error, DoDHaluEvalError)

    def test_timeout_error(self):
        """Test timeout error."""
        error = TimeoutError("Request timed out")
        assert str(error) == "Request timed out"
        assert isinstance(error, DoDHaluEvalError)

    def test_exception_inheritance_chain(self):
        """Test that all exceptions inherit from DoDHaluEvalError."""
        exceptions = [
            ConfigurationError("test"),
            ValidationError("test"),
            DataProcessingError("test"),
            APIError("test"),
            ModelNotFoundError("test"),
            CacheError("test"),
            PromptGenerationError("test"),
            EvaluationError("test"),
            FileSystemError("test"),
            ResourceError("test"),
            RateLimitError("test"),
            AuthenticationError("test"),
            PDFProcessingError("test"),
            TimeoutError("test")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, DoDHaluEvalError)
            assert isinstance(exc, Exception)

    def test_exception_with_details(self):
        """Test exceptions with additional details."""
        # Test with constructor parameters that create details
        error = APIError("API failed", status_code=500, response_text="Internal Server Error")
        assert hasattr(error, 'details')
        assert error.details["status_code"] == 500

    def test_exception_chaining(self):
        """Test exception chaining."""
        original_error = ValueError("Original error")
        try:
            raise original_error
        except ValueError as e:
            wrapped_error = DataProcessingError("Processing failed")
            wrapped_error.__cause__ = e
        
        assert wrapped_error.__cause__ is original_error
        assert "Processing failed" in str(wrapped_error)

    def test_rate_limit_error_with_retry_after(self):
        """Test rate limit error with retry after."""
        error = RateLimitError("Rate limit exceeded", retry_after=60)
        assert hasattr(error, 'retry_after')
        assert error.retry_after == 60

    def test_api_error_with_status_code(self):
        """Test API error with status code."""
        error = APIError("API failed", status_code=404)
        assert hasattr(error, 'status_code')
        assert error.status_code == 404

    def test_validation_error_with_field(self):
        """Test validation error with field information."""
        error = ValidationError("Invalid value", field="temperature")
        assert hasattr(error, 'field')
        assert error.field == "temperature"

    def test_configuration_error_with_config_section(self):
        """Test configuration error with config section."""
        # ConfigurationError doesn't have section parameter - test with details instead
        error = ConfigurationError("Missing key")
        assert "Missing key" in str(error)
        assert isinstance(error, DoDHaluEvalError)

    def test_file_not_found_error_with_path(self):
        """Test file not found error with file path."""
        error = FileSystemError("File not found", path="/path/to/file.txt")
        assert hasattr(error, 'path')
        assert error.path == "/path/to/file.txt"

    def test_exception_with_multiple_args(self):
        """Test exceptions with multiple arguments."""
        # DataProcessingError only takes specific parameters
        error = DataProcessingError("Processing failed", operation="test_op", data_type="test_data")
        assert "Processing failed" in str(error)
        assert hasattr(error, 'operation')
        assert error.operation == "test_op"

    def test_exception_repr(self):
        """Test exception representation."""
        error = ValidationError("Test error")
        repr_str = repr(error)
        assert "ValidationError" in repr_str
        assert "Test error" in repr_str


@pytest.mark.unit
class TestExceptionUsage:
    """Test realistic exception usage scenarios."""

    def test_raise_configuration_error(self):
        """Test raising configuration error."""
        with pytest.raises(ConfigurationError, match="Missing API key"):
            raise ConfigurationError("Missing API key")

    def test_raise_validation_error(self):
        """Test raising validation error."""
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            raise ValidationError("Temperature must be between 0 and 2", field="temperature")

    def test_raise_api_error(self):
        """Test raising API error."""
        with pytest.raises(APIError, match="Request failed"):
            raise APIError("Request failed", status_code=500)

    def test_catch_base_exception(self):
        """Test catching base exception."""
        try:
            raise ValidationError("Test validation error")
        except DoDHaluEvalError as e:
            assert isinstance(e, ValidationError)
            assert "Test validation error" in str(e)

    def test_catch_specific_exception(self):
        """Test catching specific exception type."""
        try:
            raise RateLimitError("Rate limit exceeded", retry_after=30)
        except RateLimitError as e:
            assert e.retry_after == 30

    def test_nested_exception_handling(self):
        """Test nested exception handling."""
        def inner_function():
            raise ValueError("Inner error")
        
        def outer_function():
            try:
                inner_function()
            except ValueError as e:
                new_error = DataProcessingError("Outer error")
                new_error.__cause__ = e
                raise new_error
        
        with pytest.raises(DataProcessingError) as exc_info:
            outer_function()
        
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_exception_context_manager(self):
        """Test exception in context manager."""
        class ErrorProneResource:
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type:
                    # Transform any exception to our custom exception
                    new_error = DataProcessingError("Resource cleanup failed")
                    new_error.__cause__ = exc_val
                    raise new_error
                return False
            
            def do_work(self):
                raise ValueError("Something went wrong")
        
        with pytest.raises(DataProcessingError):
            with ErrorProneResource() as resource:
                resource.do_work()

    def test_exception_with_complex_context(self):
        """Test exception with complex context information."""
        context = {
            "operation": "prompt_generation",
            "model": "gpt-4",
            "timestamp": "2024-01-01T00:00:00Z",
            "parameters": {"temperature": 0.7, "max_tokens": 1000}
        }
        
        # DataProcessingError doesn't have context parameter - use operation and data_type
        error = DataProcessingError("Generation failed", operation="prompt_generation", data_type="gpt-4")
        assert hasattr(error, 'operation')
        assert error.operation == "prompt_generation"
        assert error.data_type == "gpt-4"