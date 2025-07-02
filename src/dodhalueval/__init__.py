"""DoDHaluEval - DoD Hallucination Evaluation Benchmark

A comprehensive benchmark for evaluating large language model hallucinations
in the Department of Defense knowledge domain.
"""

__version__ = "0.1.0"
__author__ = "DoD Research Team"

# Import only the modules that are currently implemented
try:
    from dodhalueval.data.pdf_processor import PDFProcessor
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from dodhalueval.models.schemas import Document, DocumentChunk, Prompt, Response
from dodhalueval.utils.config import load_config, get_default_config

__all__ = [
    "Document",
    "DocumentChunk", 
    "Prompt",
    "Response",
    "load_config",
    "get_default_config",
]

if PDF_AVAILABLE:
    __all__.append("PDFProcessor")