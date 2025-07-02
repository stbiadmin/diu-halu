"""Configuration loading and management utilities."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from pydantic import ValidationError

from dodhalueval.models.config import DoDHaluEvalConfig
from dodhalueval.utils.exceptions import ConfigurationError


class ConfigLoader:
    """Loads and manages configuration from YAML files and environment variables."""
    
    def __init__(self, config_dir: str = "configs"):
        """Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.env_prefix = "DODHALUEVAL_"
    
    def load_config(
        self, 
        config_file: Optional[str] = None,
        environment: str = "development"
    ) -> DoDHaluEvalConfig:
        """Load configuration from file and environment variables.
        
        Args:
            config_file: Specific config file to load. If None, loads default for environment
            environment: Environment name (development, production, testing)
            
        Returns:
            Validated configuration object (falls back to defaults if files don't exist)
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Determine config file path
            explicit_config_file = config_file is not None
            if config_file is None:
                config_file = f"{environment}.yaml"
            
            config_path = self.config_dir / config_file
            
            # Load base configuration
            if config_path.exists():
                config_data = self._load_yaml_file(config_path)
            elif not explicit_config_file:
                # Fall back to default configuration only when no explicit file was requested
                config_data = {}
            else:
                # Explicit config file was requested but doesn't exist - raise error
                config_data = self._load_yaml_file(config_path)
            
            # Load default configuration if not using default file
            if config_file != "default.yaml":
                default_path = self.config_dir / "default.yaml"
                if default_path.exists():
                    default_data = self._load_yaml_file(default_path)
                    config_data = self._merge_configs(default_data, config_data)
            
            # Override with environment variables
            config_data = self._apply_env_overrides(config_data)
            
            # Validate and create configuration object
            return DoDHaluEvalConfig(**config_data)
            
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}") from e
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigurationError: If file cannot be loaded
        """
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {file_path}: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Failed to read {file_path}: {e}") from e
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries.
        
        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration.
        
        Environment variables should be in the format:
        DODHALUEVAL_SECTION_SUBSECTION_KEY=value
        
        Examples:
        DODHALUEVAL_API_CONFIGS_OPENAI_API_KEY=sk-...
        DODHALUEVAL_PDF_PROCESSING_CHUNK_SIZE=1500
        DODHALUEVAL_BATCH_SIZE=100
        
        Args:
            config_data: Configuration dictionary to override
            
        Returns:
            Configuration dictionary with environment overrides applied
        """
        config_copy = config_data.copy()
        
        for env_var, value in os.environ.items():
            if not env_var.startswith(self.env_prefix):
                continue
            
            # Remove prefix and convert to lowercase
            key_path = env_var[len(self.env_prefix):].lower()
            
            # Handle known nested configurations
            nested_configs = {
                'pdf_processing': 'pdf_processing',
                'api_configs': 'api_configs', 
                'prompt_generation': 'prompt_generation',
                'output': 'output',
                'cache': 'cache',
                'logging': 'logging'
            }
            
            # Check if this is a nested config variable
            for nested_prefix, nested_field in nested_configs.items():
                if key_path.startswith(nested_prefix + '_'):
                    # This is a nested config: pdf_processing_chunk_size -> pdf_processing.chunk_size
                    remaining_path = key_path[len(nested_prefix) + 1:]
                    if nested_field not in config_copy:
                        config_copy[nested_field] = {}
                    config_copy[nested_field][remaining_path] = self._convert_env_value(value)
                    break
            else:
                # This is a top-level config: batch_size -> batch_size
                config_copy[key_path] = self._convert_env_value(value)
        
        return config_copy
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type.
        
        Args:
            value: Environment variable value as string
            
        Returns:
            Converted value
        """
        # Handle boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Handle numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def save_config(self, config: DoDHaluEvalConfig, file_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: Configuration object to save
            file_path: Path to save configuration file
            
        Raises:
            ConfigurationError: If file cannot be saved
        """
        try:
            config_dict = config.dict()
            
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config_dict,
                    f,
                    default_flow_style=False,
                    sort_keys=True,
                    indent=2
                )
                
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {file_path}: {e}") from e
    
    def generate_default_config(self, output_path: str) -> None:
        """Generate a default configuration file.
        
        Args:
            output_path: Path to save the default configuration
        """
        default_config = DoDHaluEvalConfig()
        self.save_config(default_config, output_path)
    
    def validate_config_file(self, config_file: str) -> bool:
        """Validate a configuration file without loading it.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            self.load_config(config_file)
            return True
        except ConfigurationError:
            return False


def load_config(
    config_file: Optional[str] = None,
    environment: str = "development",
    config_dir: str = "configs"
) -> DoDHaluEvalConfig:
    """Convenience function to load configuration.
    
    Args:
        config_file: Specific config file to load
        environment: Environment name
        config_dir: Configuration directory
        
    Returns:
        Loaded configuration object
    """
    loader = ConfigLoader(config_dir)
    return loader.load_config(config_file, environment)


def get_default_config() -> DoDHaluEvalConfig:
    """Get default configuration object.
    
    Returns:
        Default configuration
    """
    return DoDHaluEvalConfig()