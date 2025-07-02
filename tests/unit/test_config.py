"""Unit tests for configuration management."""

import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import yaml

from dodhalueval.models.config import DoDHaluEvalConfig, PDFProcessingConfig, APIConfig
from dodhalueval.utils.config import ConfigLoader, load_config, get_default_config
from dodhalueval.utils.exceptions import ConfigurationError


class TestDoDHaluEvalConfig:
    """Test the main configuration model."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = DoDHaluEvalConfig()
        
        assert config.version == "0.1.0"
        assert config.environment == "development"
        assert config.batch_size == 50
        assert config.pdf_processing.chunk_size == 1000
        assert config.cache.enabled is True
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config_data = {
            "version": "0.1.0",
            "environment": "testing",
            "batch_size": 25,
            "pdf_processing": {
                "chunk_size": 800,
                "chunk_overlap": 150
            }
        }
        
        config = DoDHaluEvalConfig(**config_data)
        assert config.batch_size == 25
        assert config.pdf_processing.chunk_size == 800
    
    def test_invalid_environment(self):
        """Test validation of invalid environment."""
        with pytest.raises(ValueError, match="environment"):
            DoDHaluEvalConfig(environment="invalid")
    
    def test_invalid_batch_size(self):
        """Test validation of invalid batch size."""
        with pytest.raises(ValueError):
            DoDHaluEvalConfig(batch_size=0)
        
        with pytest.raises(ValueError):
            DoDHaluEvalConfig(batch_size=1000)
    
    def test_api_config_validation(self):
        """Test API configuration validation."""
        # Valid API config
        api_config = APIConfig(
            provider="openai",
            model="gpt-4",
            api_key="test-key"
        )
        assert api_config.provider == "openai"
        assert api_config.max_retries == 3
        
        # Invalid provider
        with pytest.raises(ValueError, match="Provider must be one of"):
            APIConfig(provider="invalid", model="test")
    
    def test_pdf_processing_config_validation(self):
        """Test PDF processing configuration validation."""
        # Valid config
        pdf_config = PDFProcessingConfig(
            chunk_size=1000,
            chunk_overlap=200
        )
        assert pdf_config.chunk_size == 1000
        
        # Invalid: overlap >= chunk_size
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            PDFProcessingConfig(chunk_size=1000, chunk_overlap=1000)
    
    def test_config_methods(self):
        """Test configuration utility methods."""
        config = DoDHaluEvalConfig()
        
        # Test API config getter - should return None for empty default config
        openai_config = config.get_api_config("openai")
        assert openai_config is None  # Default config has no API configs
        
        # Test non-existent provider
        invalid_config = config.get_api_config("invalid")
        assert invalid_config is None
        
        # Test evaluation config getter - should return None for empty default config
        eval_config = config.get_evaluation_config("vectara_hhem")
        assert eval_config is None  # Default config has no evaluation methods
        
        # Test cache methods
        assert config.is_cache_enabled() is True
        pdf_cache_dir = config.get_cache_dir("pdf")
        assert "pdf_cache" in pdf_cache_dir


class TestConfigLoader:
    """Test the configuration loader."""
    
    def test_config_loader_initialization(self):
        """Test ConfigLoader initialization."""
        loader = ConfigLoader("test_configs")
        assert loader.config_dir == Path("test_configs")
        assert loader.env_prefix == "DODHALUEVAL_"
    
    def test_load_default_config(self, sample_config_data: Dict[str, Any], tmp_path: Path):
        """Test loading default configuration."""
        # Create test config file
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        # Create development.yaml since that's the default environment
        config_file = config_dir / "development.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_data, f)
        
        loader = ConfigLoader(str(config_dir))
        config = loader.load_config()
        
        assert config.environment == "testing"
        assert config.batch_size == 10
    
    def test_load_environment_config(self, sample_config_data: Dict[str, Any], tmp_path: Path):
        """Test loading environment-specific configuration."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        # Create default config
        default_config = sample_config_data.copy()
        default_config["batch_size"] = 50
        
        with open(config_dir / "development.yaml", 'w') as f:
            yaml.dump(default_config, f)
        
        # Create environment-specific config
        env_config = {"batch_size": 25}
        with open(config_dir / "testing.yaml", 'w') as f:
            yaml.dump(env_config, f)
        
        loader = ConfigLoader(str(config_dir))
        config = loader.load_config(environment="testing")
        
        # Should merge default and environment configs
        assert config.batch_size == 25  # From environment config
        assert config.version == "0.1.0"  # From default config
    
    def test_environment_variable_overrides(self, sample_config_data: Dict[str, Any], tmp_path: Path, monkeypatch):
        """Test environment variable overrides."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        with open(config_dir / "development.yaml", 'w') as f:
            yaml.dump(sample_config_data, f)
        
        # Set environment variables
        monkeypatch.setenv("DODHALUEVAL_BATCH_SIZE", "100")
        monkeypatch.setenv("DODHALUEVAL_PDF_PROCESSING_CHUNK_SIZE", "2000")
        
        loader = ConfigLoader(str(config_dir))
        config = loader.load_config()
        
        assert config.batch_size == 100
        assert config.pdf_processing.chunk_size == 2000
    
    def test_missing_config_file(self, tmp_path: Path):
        """Test error handling for missing config file."""
        loader = ConfigLoader(str(tmp_path))
        
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            loader.load_config("nonexistent.yaml")
    
    def test_invalid_yaml(self, tmp_path: Path):
        """Test error handling for invalid YAML."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        config_file = config_dir / "invalid.yaml"
        
        # Write invalid YAML
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        loader = ConfigLoader(str(config_dir))
        
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            loader.load_config("invalid.yaml")
    
    def test_save_config(self, tmp_path: Path):
        """Test saving configuration to file."""
        config = DoDHaluEvalConfig()
        output_file = tmp_path / "saved_config.yaml"
        
        loader = ConfigLoader()
        loader.save_config(config, str(output_file))
        
        assert output_file.exists()
        
        # Verify saved content
        with open(output_file, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["version"] == "0.1.0"
        assert saved_data["environment"] == "development"
    
    def test_generate_default_config(self, tmp_path: Path):
        """Test generating default configuration."""
        output_file = tmp_path / "generated_config.yaml"
        
        loader = ConfigLoader()
        loader.generate_default_config(str(output_file))
        
        assert output_file.exists()
        
        # Verify generated config can be loaded
        config = loader.load_config(str(output_file))
        assert isinstance(config, DoDHaluEvalConfig)
    
    def test_validate_config_file(self, sample_config_data: Dict[str, Any], tmp_path: Path):
        """Test configuration file validation."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        # Create valid config file
        valid_config = config_dir / "valid.yaml"
        with open(valid_config, 'w') as f:
            yaml.dump(sample_config_data, f)
        
        # Create invalid config file
        invalid_config = config_dir / "invalid.yaml"
        invalid_data = sample_config_data.copy()
        invalid_data["batch_size"] = -1  # Invalid value
        with open(invalid_config, 'w') as f:
            yaml.dump(invalid_data, f)
        
        loader = ConfigLoader(str(config_dir))
        
        assert loader.validate_config_file("valid.yaml") is True
        assert loader.validate_config_file("invalid.yaml") is False


class TestConfigUtilityFunctions:
    """Test utility functions for configuration."""
    
    def test_load_config_function(self, sample_config_data: Dict[str, Any], tmp_path: Path):
        """Test the load_config convenience function."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        with open(config_dir / "testing.yaml", 'w') as f:
            yaml.dump(sample_config_data, f)
        
        config = load_config(environment="testing", config_dir=str(config_dir))
        
        assert isinstance(config, DoDHaluEvalConfig)
        assert config.environment == "testing"
    
    def test_get_default_config_function(self):
        """Test the get_default_config convenience function."""
        config = get_default_config()
        
        assert isinstance(config, DoDHaluEvalConfig)
        assert config.version == "0.1.0"
        assert config.environment == "development"
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        original_config = DoDHaluEvalConfig()
        
        # Convert to dict and back
        config_dict = original_config.dict()
        restored_config = DoDHaluEvalConfig(**config_dict)
        
        assert original_config.version == restored_config.version
        assert original_config.environment == restored_config.environment
        assert original_config.batch_size == restored_config.batch_size