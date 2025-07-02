#!/usr/bin/env python3
"""Setup script for DoDHaluEval package."""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="dodhalueval",
    version="0.1.0",
    author="DoD Research Team",
    author_email="research@dod.mil",
    description="DoD Hallucination Evaluation Benchmark for Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dod/dodhalueval",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.0.0",
        "PyPDF2>=3.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "rich>=13.0.0",
        "click>=8.0.0",
        "pyyaml>=6.0",
        "jsonlines>=3.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "isort",
            "pre-commit",
        ],
        "llm": [
            "openai>=1.0.0",
            "transformers>=4.30.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dodhalueval=dodhalueval.cli.commands:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)