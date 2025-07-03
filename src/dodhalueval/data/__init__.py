"""Data processing and management for DoDHaluEval."""

try:
    from .pdf_processor import PDFProcessor
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
from .dataset_builder import DatasetBuilder, HaluEvalDataset, HaluEvalSample
from .dataset_validator import DatasetValidator, ValidationReport, ValidationIssue

__all__ = [
    "DatasetBuilder",
    "HaluEvalDataset",
    "HaluEvalSample",
    "DatasetValidator",
    "ValidationReport",
    "ValidationIssue"
]

if PDF_AVAILABLE:
    __all__.append("PDFProcessor")
