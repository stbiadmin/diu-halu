# Contributing to DoDHaluEval

Thank you for your interest in contributing to DoDHaluEval! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/diu-halu.git
   cd diu-halu
   ```
3. Create a virtual environment and install development dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests and linting:
   ```bash
   pytest tests/
   flake8 src/ tests/
   black --check src/ tests/
   isort --check-only src/ tests/
   ```
4. Commit your changes with a descriptive message
5. Push to your fork and open a pull request

## Code Standards

- **Style**: PEP 8 compliance enforced via `black` and `flake8`
- **Imports**: Sorted with `isort`
- **Type Hints**: Use type annotations for all public function signatures
- **Docstrings**: Google-style docstrings for all public classes and functions
- **Testing**: All new features must include tests; aim for 90%+ coverage

## Pull Request Process

1. Update documentation if your change affects public APIs or behavior
2. Add tests covering your changes
3. Ensure CI passes (linting, type checking, tests)
4. Provide a clear description of what your PR does and why
5. Link any related issues

## Reporting Bugs

Use the [bug report template](https://github.com/stbiadmin/diu-halu/issues/new?template=bug_report.md) and include:

- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Relevant configuration or logs

## Requesting Features

Use the [feature request template](https://github.com/stbiadmin/diu-halu/issues/new?template=feature_request.md) and describe:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

## Questions?

Open a [Discussion](https://github.com/stbiadmin/diu-halu/discussions) for general questions or ideas.
