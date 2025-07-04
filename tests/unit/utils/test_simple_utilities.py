"""Simple unit tests for utility functions that don't require complex mocking."""

import pytest
import tempfile
import os
from pathlib import Path

from dodhalueval.utils.exceptions import (
    DoDHaluEvalError,
    ConfigurationError,
    ValidationError
)


@pytest.mark.unit
class TestSimpleExceptions:
    """Test basic exception functionality."""

    def test_base_exception_creation(self):
        """Test basic exception creation."""
        error = DoDHaluEvalError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_configuration_error_inheritance(self):
        """Test configuration error inheritance."""
        error = ConfigurationError("Config error")
        assert isinstance(error, DoDHaluEvalError)
        assert isinstance(error, Exception)
        assert str(error) == "Config error"

    def test_validation_error_inheritance(self):
        """Test validation error inheritance."""
        error = ValidationError("Validation error")
        assert isinstance(error, DoDHaluEvalError)
        assert isinstance(error, Exception)
        assert str(error) == "Validation error"

    def test_exception_with_details(self):
        """Test exception with details dictionary."""
        details = {"field": "temperature", "value": 2.5}
        error = DoDHaluEvalError("Invalid value", details=details)
        assert error.details == details
        assert "Invalid value" in str(error)

    def test_raise_and_catch_exception(self):
        """Test raising and catching custom exceptions."""
        with pytest.raises(ConfigurationError, match="Missing API key"):
            raise ConfigurationError("Missing API key")

    def test_exception_chaining(self):
        """Test exception chaining works properly."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            wrapped_error = ConfigurationError("Wrapped error")
            wrapped_error.__cause__ = e
            assert wrapped_error.__cause__ is e


@pytest.mark.unit 
class TestFileSystemOperations:
    """Test file system related operations."""

    def test_temp_directory_creation(self):
        """Test temporary directory handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            assert temp_path.exists()
            assert temp_path.is_dir()
            
            # Create a test file
            test_file = temp_path / "test.txt"
            test_file.write_text("test content")
            assert test_file.exists()
            assert test_file.read_text() == "test content"

    def test_path_operations(self):
        """Test basic path operations."""
        test_path = Path("/test/path/file.txt")
        assert test_path.name == "file.txt"
        assert test_path.suffix == ".txt"
        assert test_path.parent == Path("/test/path")

    def test_environment_variable_handling(self):
        """Test environment variable operations."""
        test_var = "DODHALUEVAL_TEST_VAR"
        test_value = "test_value"
        
        # Set environment variable
        os.environ[test_var] = test_value
        assert os.environ.get(test_var) == test_value
        
        # Clean up
        os.environ.pop(test_var, None)
        assert os.environ.get(test_var) is None


@pytest.mark.unit
class TestDataStructures:
    """Test basic data structure operations."""

    def test_list_operations(self):
        """Test list operations."""
        test_list = ["item1", "item2", "item3"]
        assert len(test_list) == 3
        assert "item2" in test_list
        assert test_list[0] == "item1"

    def test_dict_operations(self):
        """Test dictionary operations."""
        test_dict = {"key1": "value1", "key2": "value2"}
        assert len(test_dict) == 2
        assert test_dict.get("key1") == "value1"
        assert test_dict.get("nonexistent") is None
        assert test_dict.get("nonexistent", "default") == "default"

    def test_string_operations(self):
        """Test string operations."""
        test_string = "DoDHaluEval Test String"
        assert test_string.lower() == "dodhalueval test string"
        assert test_string.upper() == "DODHALUEVAL TEST STRING"
        assert "Test" in test_string
        assert test_string.startswith("DoD")
        assert test_string.endswith("String")

    def test_type_checking(self):
        """Test type checking operations."""
        assert isinstance("string", str)
        assert isinstance(42, int)
        assert isinstance(3.14, float)
        assert isinstance([], list)
        assert isinstance({}, dict)
        assert isinstance(True, bool)


@pytest.mark.unit
class TestPythonFeatures:
    """Test Python language features work correctly."""

    def test_list_comprehension(self):
        """Test list comprehension."""
        numbers = [1, 2, 3, 4, 5]
        squares = [x**2 for x in numbers]
        assert squares == [1, 4, 9, 16, 25]

    def test_dict_comprehension(self):
        """Test dictionary comprehension."""
        numbers = [1, 2, 3, 4, 5]
        squares_dict = {x: x**2 for x in numbers}
        expected = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
        assert squares_dict == expected

    def test_lambda_functions(self):
        """Test lambda functions."""
        add_one = lambda x: x + 1
        assert add_one(5) == 6
        
        numbers = [1, 2, 3, 4, 5]
        incremented = list(map(add_one, numbers))
        assert incremented == [2, 3, 4, 5, 6]

    def test_generator_expression(self):
        """Test generator expressions."""
        numbers = [1, 2, 3, 4, 5]
        squares_gen = (x**2 for x in numbers)
        squares_list = list(squares_gen)
        assert squares_list == [1, 4, 9, 16, 25]

    def test_context_manager(self):
        """Test context manager functionality."""
        class TestContextManager:
            def __init__(self):
                self.entered = False
                self.exited = False
            
            def __enter__(self):
                self.entered = True
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.exited = True
                return False
        
        manager = TestContextManager()
        with manager as cm:
            assert cm.entered is True
            assert cm.exited is False
        
        assert manager.exited is True