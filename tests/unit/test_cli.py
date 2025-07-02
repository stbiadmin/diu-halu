"""Unit tests for CLI functionality."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from dodhalueval.cli.commands import cli, process_docs, validate_config, info, version


class TestCLI:
    """Test CLI command functionality."""
    
    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert "DoDHaluEval" in result.output
        assert "DoD Hallucination Evaluation Benchmark" in result.output
    
    def test_cli_version(self):
        """Test version command."""
        runner = CliRunner()
        result = runner.invoke(version)
        
        assert result.exit_code == 0
        assert "DoDHaluEval version" in result.output
    
    def test_cli_with_config_options(self, tmp_path):
        """Test CLI with configuration options."""
        # Create a test config file
        config_data = {
            "version": "0.1.0",
            "environment": "testing",
            "batch_size": 10
        }
        config_file = tmp_path / "test_config.yaml"
        
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        runner = CliRunner()
        result = runner.invoke(cli, ['--config', str(config_file), 'info'])
        
        assert result.exit_code == 0
    
    def test_cli_verbose_and_quiet_options(self):
        """Test verbose and quiet options."""
        runner = CliRunner()
        
        # Test verbose
        result = runner.invoke(cli, ['--verbose', 'info'])
        assert result.exit_code == 0
        
        # Test quiet
        result = runner.invoke(cli, ['--quiet', 'info'])
        assert result.exit_code == 0


class TestProcessDocsCommand:
    """Test the process-docs command."""
    
    @patch('dodhalueval.cli.commands.PDFProcessor')
    def test_process_docs_basic(self, mock_processor_class, tmp_path):
        """Test basic document processing command."""
        # Setup directories
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        
        # Create a dummy PDF file
        pdf_file = input_dir / "test.pdf"
        pdf_file.write_bytes(b"dummy pdf content")
        
        # Mock processor
        mock_processor = Mock()
        mock_processor.process_document.return_value = {
            'pages': [{'page': 1}],
            'chunks': [{'chunk': 1}, {'chunk': 2}],
            'metadata': {'title': 'Test'}
        }
        mock_processor_class.return_value = mock_processor
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'process-docs',
            '--input', str(input_dir),
            '--output', str(output_dir)
        ])
        
        assert result.exit_code == 0
        assert "Processing Complete" in result.output
        assert output_dir.exists()
    
    def test_process_docs_no_pdfs(self, tmp_path):
        """Test process-docs with no PDF files."""
        input_dir = tmp_path / "empty"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'process-docs',
            '--input', str(input_dir),
            '--output', str(output_dir)
        ])
        
        assert result.exit_code == 0
        assert "No PDF files found" in result.output
    
    def test_process_docs_with_options(self, tmp_path):
        """Test process-docs with various options."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        
        # Create a dummy PDF file
        pdf_file = input_dir / "test.pdf"
        pdf_file.write_bytes(b"dummy pdf content")
        
        with patch('dodhalueval.cli.commands.PDFProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor.process_document.return_value = {
                'pages': [],
                'chunks': [],
                'metadata': {}
            }
            mock_processor_class.return_value = mock_processor
            
            runner = CliRunner()
            result = runner.invoke(cli, [
                'process-docs',
                '--input', str(input_dir),
                '--output', str(output_dir),
                '--max-pages', '5',
                '--chunk-size', '800',
                '--chunk-overlap', '150',
                '--force'
            ])
            
            assert result.exit_code == 0
            
            # Verify processor was called with correct parameters
            mock_processor_class.assert_called_once()
            call_args = mock_processor_class.call_args[1]
            assert call_args['chunk_size'] == 800
            assert call_args['chunk_overlap'] == 150
            assert call_args['max_pages'] == 5


class TestValidateConfigCommand:
    """Test the validate-config command."""
    
    def test_validate_valid_config(self, tmp_path):
        """Test validating a valid configuration file."""
        config_data = {
            "version": "0.1.0",
            "environment": "testing",
            "batch_size": 10,
            "pdf_processing": {
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
        }
        config_file = tmp_path / "valid_config.yaml"
        
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        runner = CliRunner()
        result = runner.invoke(validate_config, ['--config-file', str(config_file)])
        
        assert result.exit_code == 0
        assert "Configuration file" in result.output
        assert "is valid" in result.output
    
    def test_validate_environment_config(self):
        """Test validating environment configuration."""
        runner = CliRunner()
        result = runner.invoke(validate_config, ['--environment', 'testing'])
        
        assert result.exit_code == 0
        assert "Environment 'testing' configuration is valid" in result.output
    
    def test_validate_invalid_config(self, tmp_path):
        """Test validating an invalid configuration file."""
        # Create invalid config (negative batch size)
        config_data = {
            "version": "0.1.0",
            "environment": "testing",
            "batch_size": -1  # Invalid
        }
        config_file = tmp_path / "invalid_config.yaml"
        
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        runner = CliRunner()
        result = runner.invoke(validate_config, ['--config-file', str(config_file)])
        
        assert result.exit_code == 1
        assert "Configuration validation failed" in result.output


class TestInfoCommand:
    """Test the info command."""
    
    def test_info_command(self):
        """Test info command output."""
        runner = CliRunner()
        result = runner.invoke(cli, ['info'])
        
        assert result.exit_code == 0
        assert "DoDHaluEval System Information" in result.output
        assert "Python Version" in result.output
        assert "Platform" in result.output
        assert "Environment" in result.output
    
    def test_info_with_api_providers(self):
        """Test info command shows API providers."""
        runner = CliRunner()
        result = runner.invoke(cli, ['info'])
        
        assert result.exit_code == 0
        # Should show default API providers
        assert "openai" in result.output.lower()


class TestGenerateConfigCommand:
    """Test the generate-config command."""
    
    def test_generate_config_basic(self, tmp_path):
        """Test basic config generation."""
        output_file = tmp_path / "generated.yaml"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'generate-config',
            '--output', str(output_file)
        ])
        
        assert result.exit_code == 0
        assert "Generated development configuration" in result.output
        assert output_file.exists()
        
        # Verify generated config is valid YAML
        import yaml
        with open(output_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        assert 'version' in config_data
        assert 'environment' in config_data
    
    def test_generate_config_different_environment(self, tmp_path):
        """Test generating config for different environment."""
        output_file = tmp_path / "production.yaml"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'generate-config',
            '--output', str(output_file),
            '--environment', 'production'
        ])
        
        assert result.exit_code == 0
        assert "Generated production configuration" in result.output
        assert output_file.exists()
    
    def test_generate_config_overwrite_protection(self, tmp_path):
        """Test overwrite protection."""
        output_file = tmp_path / "existing.yaml"
        output_file.write_text("existing content")
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'generate-config',
            '--output', str(output_file)
        ])
        
        assert result.exit_code == 0
        assert "already exists" in result.output
        
        # Test with overwrite flag
        result = runner.invoke(cli, [
            'generate-config',
            '--output', str(output_file),
            '--overwrite'
        ])
        
        assert result.exit_code == 0
        assert "Generated" in result.output


class TestAnalyzeDatasetCommand:
    """Test the analyze-dataset command."""
    
    def test_analyze_jsonl_dataset(self, tmp_path):
        """Test analyzing a JSONL dataset."""
        dataset_file = tmp_path / "test_dataset.jsonl"
        
        # Create sample JSONL data
        sample_data = [
            {"id": "1", "prompt": "Test prompt 1", "response": "Test response 1"},
            {"id": "2", "prompt": "Test prompt 2", "response": "Test response 2"}
        ]
        
        with open(dataset_file, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'analyze-dataset',
            '--dataset', str(dataset_file),
            '--format', 'jsonl'
        ])
        
        assert result.exit_code == 0
        assert "Analyzing dataset" in result.output
        assert "Number of entries: 2" in result.output
        assert "Dataset analysis complete" in result.output
    
    def test_analyze_no_dataset(self):
        """Test analyze command with no dataset file."""
        runner = CliRunner()
        result = runner.invoke(cli, ['analyze-dataset'])
        
        assert result.exit_code == 0
        assert "No dataset file specified" in result.output


class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    def test_missing_input_directory(self, tmp_path):
        """Test error handling for missing input directory."""
        nonexistent_dir = tmp_path / "nonexistent"
        output_dir = tmp_path / "output"
        
        runner = CliRunner()
        result = runner.invoke(process_docs, [
            '--input', str(nonexistent_dir),
            '--output', str(output_dir)
        ])
        
        assert result.exit_code != 0
    
    def test_invalid_config_file(self, tmp_path):
        """Test error handling for invalid config file."""
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content: [")
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            '--config', str(invalid_config),
            'info'
        ])
        
        assert result.exit_code == 1
        assert "Failed to initialize" in result.output