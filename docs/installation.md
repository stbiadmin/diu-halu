# Installation Guide

This guide provides detailed installation instructions for DoDHaluEval across different environments and use cases.

## System Requirements

### Minimum Requirements
- **Operating System**: Linux, macOS, or Windows 10+
- **Python**: 3.10 or higher
- **Memory**: 8GB RAM
- **Storage**: 5GB free disk space
- **Network**: Internet connection for API access

### Recommended Requirements
- **Memory**: 16GB RAM for large document processing
- **Storage**: 10GB+ for caches and datasets
- **CPU**: Multi-core processor for parallel processing

## Installation Methods

### 1. Development Installation (Recommended)

For development, testing, or customization:

```bash
# Clone the repository
git clone https://github.com/yourusername/dodhalueval.git
cd dodhalueval

# Create and activate virtual environment
python -m venv dodhalueval-env
source dodhalueval-env/bin/activate  # On Windows: dodhalueval-env\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

### 2. Production Installation

For production deployment:

```bash
# Clone the repository
git clone https://github.com/yourusername/dodhalueval.git
cd dodhalueval

# Create and activate virtual environment
python -m venv dodhalueval-env
source dodhalueval-env/bin/activate

# Install production dependencies only
pip install -e .
```

### 3. Conda Installation

For conda environments:

```bash
# Create conda environment from file
conda env create -f environment.yml
conda activate dodhalueval

# Or create manually
conda create -n dodhalueval python=3.10
conda activate dodhalueval
pip install -e ".[dev]"
```

## Dependencies

### Core Dependencies

DoDHaluEval requires the following core packages:

```bash
# Core framework dependencies
pydantic>=2.0.0          # Data validation and settings
pandas>=2.0.0            # Data manipulation
numpy>=1.24.0            # Numerical operations
scikit-learn>=1.3.0      # Evaluation metrics
tqdm>=4.65.0            # Progress bars
jsonlines>=3.0.0        # Data storage format

# Document processing
PyPDF2>=3.0.0           # PDF text extraction
langchain>=0.0.200      # Text processing utilities

# API integrations
openai>=1.0.0           # OpenAI GPT models
fireworks-ai>=0.9.0     # Fireworks AI models
transformers>=4.30.0    # HuggingFace models
torch>=2.0.0            # PyTorch for model inference

# CLI and utilities
click>=8.1.0            # Command-line interface
rich>=13.0.0            # Rich terminal output
python-dotenv>=1.0.0    # Environment variable management
```

### Development Dependencies

Additional packages for development and testing:

```bash
# Testing
pytest>=7.0.0           # Testing framework
pytest-asyncio>=0.21.0  # Async test support
pytest-cov>=4.0.0       # Coverage reporting
pytest-mock>=3.10.0     # Mocking utilities

# Code quality
black>=23.0.0           # Code formatting
flake8>=6.0.0           # Linting
isort>=5.12.0           # Import sorting
mypy>=1.0.0             # Type checking

# Documentation
sphinx>=6.0.0           # Documentation generation
sphinx-rtd-theme>=1.2.0 # Documentation theme
```

## Environment Setup

### 1. API Keys Configuration

Create a `.env` file in the project root:

```bash
# Required API keys
OPENAI_API_KEY=your-openai-api-key-here
FIREWORKS_API_KEY=your-fireworks-api-key-here

# Optional API keys
HUGGINGFACE_TOKEN=your-huggingface-token-here

# Optional configuration
DODHALUEVAL_ENVIRONMENT=development
DODHALUEVAL_LOG_LEVEL=INFO
```

### 2. Directory Structure Setup

The installation automatically creates the following directory structure:

```
dodhalueval/
├── src/                    # Source code
├── tests/                  # Test suite
├── docs/                   # Documentation
├── configs/                # Configuration files
├── data/                   # Data directory
│   ├── CSC/               # DoD source documents
│   ├── processed/         # Processed documents
│   ├── prompts/           # Generated prompts
│   ├── responses/         # Generated responses
│   └── evaluations/       # Evaluation results
├── output/                # Pipeline outputs
├── cache/                 # Temporary cache files
└── logs/                  # Log files
```

### 3. Configuration Files

Initialize default configuration:

```bash
# Generate default configuration
python -m dodhalueval.cli.commands generate-config --output configs/default.yaml

# Validate configuration
python -m dodhalueval.cli.commands validate-config --config configs/default.yaml
```

## Verification

### 1. Installation Verification

Test the installation:

```bash
# Check version and dependencies
python -c "import dodhalueval; print(dodhalueval.__version__)"

# Run system info
python -m dodhalueval.cli.commands info

# Run basic tests
pytest tests/unit/ -v
```

### 2. API Integration Testing

Test API connectivity:

```bash
# Test OpenAI integration
python scripts/test_api_providers.py --provider openai

# Test Fireworks integration  
python scripts/test_api_providers.py --provider fireworks

# Test all providers
python scripts/test_api_providers.py --provider all
```

### 3. Pipeline Testing

Run a quick end-to-end test:

```bash
# Quick pipeline test with mock data
python scripts/run_pipeline.py --config configs/pipeline_test.yaml

# Test with sample documents
python scripts/test_prompt_generation.py --mode quick --verbose
```

## Platform-Specific Instructions

### Linux/Ubuntu

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3.10 python3.10-venv python3.10-dev build-essential

# Follow standard installation
python3.10 -m venv dodhalueval-env
source dodhalueval-env/bin/activate
pip install -e ".[dev]"
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10

# Follow standard installation
python3.10 -m venv dodhalueval-env
source dodhalueval-env/bin/activate
pip install -e ".[dev]"
```

### Windows

```powershell
# Install Python 3.10 from python.org or Microsoft Store

# Create virtual environment
python -m venv dodhalueval-env
dodhalueval-env\Scripts\activate

# Install package
pip install -e ".[dev]"
```

## Docker Installation

### Using Docker

Build and run with Docker:

```bash
# Build Docker image
docker build -t dodhalueval .

# Run container with mounted data
docker run -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -e OPENAI_API_KEY=your-key-here \
  -e FIREWORKS_API_KEY=your-key-here \
  dodhalueval
```

### Docker Compose

Use Docker Compose for development:

```yaml
# docker-compose.yml
version: '3.8'
services:
  dodhalueval:
    build: .
    volumes:
      - ./data:/app/data
      - ./output:/app/output
      - ./configs:/app/configs
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - FIREWORKS_API_KEY=${FIREWORKS_API_KEY}
    command: python scripts/run_pipeline.py --config configs/default.yaml
```

## Troubleshooting

### Common Installation Issues

**ImportError: No module named 'dodhalueval'**
```bash
# Ensure you're in the correct directory and virtual environment
cd /path/to/dodhalueval
source dodhalueval-env/bin/activate
pip install -e .
```

**Permission denied during installation**
```bash
# On Linux/macOS, avoid sudo with pip
# Use virtual environments instead
python -m venv dodhalueval-env
source dodhalueval-env/bin/activate
pip install -e .
```

**PyTorch installation issues**
```bash
# Install PyTorch separately with CUDA support if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

**SSL certificate errors**
```bash
# Update certificates and pip
pip install --upgrade pip certifi
pip install -e . --trusted-host pypi.org --trusted-host pypi.python.org
```

### Dependency Conflicts

**Resolve version conflicts**:
```bash
# Check for conflicts
pip check

# Update conflicting packages
pip install --upgrade package-name

# Or use constraint files
pip install -e . -c constraints.txt
```

### Performance Issues

**Slow PDF processing**:
- Ensure sufficient RAM (8GB minimum)
- Enable caching in configuration
- Reduce batch sizes for large documents

**API timeout errors**:
- Check internet connectivity
- Verify API keys are valid
- Increase timeout values in configuration

### Environment Variables

**Required variables**:
```bash
export OPENAI_API_KEY="your-openai-key"
export FIREWORKS_API_KEY="your-fireworks-key"

# Optional variables
export DODHALUEVAL_LOG_LEVEL="DEBUG"
export DODHALUEVAL_CACHE_ENABLED="true"
export DODHALUEVAL_MAX_WORKERS="4"
```

## Development Setup

### Pre-commit Hooks

Install pre-commit hooks for code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks on all files
pre-commit run --all-files
```

### IDE Configuration

**VSCode settings** (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./dodhalueval-env/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"]
}
```

### Testing Setup

Configure testing environment:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=src/dodhalueval --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

## Next Steps

After successful installation:

1. **Configure API keys** in `.env` file
2. **Validate installation** with test commands
3. **Review documentation** in `docs/` directory
4. **Run quick test** with `python scripts/test_prompt_generation.py --mode quick`
5. **Explore examples** in the `docs/usage.md` guide

For detailed usage instructions, see the [Usage Guide](usage.md) and [API Reference](api-reference.md).