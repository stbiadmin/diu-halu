"""
Dataset Validator for DoDHaluEval.

This module provides validation functionality for datasets to ensure
schema compliance, data completeness, and statistical quality.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .dataset_builder import HaluEvalDataset, HaluEvalSample
from ..utils.logger import get_logger

logger = get_logger(__name__)


class IssueType(Enum):
    """Types of validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A validation issue found in the dataset."""
    type: IssueType
    message: str
    sample_id: Optional[str] = None
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    """Report containing all validation results."""
    is_valid: bool
    total_samples: int
    issues: List[ValidationIssue]
    statistics: Dict[str, Any]
    
    @property
    def error_count(self) -> int:
        return len([i for i in self.issues if i.type == IssueType.ERROR])
    
    @property
    def warning_count(self) -> int:
        return len([i for i in self.issues if i.type == IssueType.WARNING])


class DatasetValidator:
    """Validates HaluEval datasets for quality and compliance."""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        
    def validate_dataset(self, dataset: HaluEvalDataset) -> ValidationReport:
        """Validate a complete dataset."""
        issues = []
        
        # Basic structure validation
        issues.extend(self._validate_structure(dataset))
        
        # Sample validation
        for sample in dataset.samples:
            issues.extend(self._validate_sample(sample))
        
        # Statistical validation
        stats = self._compute_statistics(dataset)
        issues.extend(self._validate_statistics(stats))
        
        # Dataset is valid if no errors (warnings are ok)
        is_valid = not any(issue.type == IssueType.ERROR for issue in issues)
        
        return ValidationReport(
            is_valid=is_valid,
            total_samples=len(dataset.samples),
            issues=issues,
            statistics=stats
        )
    
    def validate_file(self, file_path: Path) -> ValidationReport:
        """Validate a dataset file."""
        try:
            if file_path.suffix == '.json':
                return self._validate_json_file(file_path)
            elif file_path.suffix == '.jsonl':
                return self._validate_jsonl_file(file_path)
            else:
                issues = [ValidationIssue(
                    type=IssueType.ERROR,
                    message=f"Unsupported file format: {file_path.suffix}"
                )]
                return ValidationReport(
                    is_valid=False,
                    total_samples=0,
                    issues=issues,
                    statistics={}
                )
        except Exception as e:
            issues = [ValidationIssue(
                type=IssueType.ERROR,
                message=f"Failed to read file: {str(e)}"
            )]
            return ValidationReport(
                is_valid=False,
                total_samples=0,
                issues=issues,
                statistics={}
            )
    
    def _validate_structure(self, dataset: HaluEvalDataset) -> List[ValidationIssue]:
        """Validate dataset structure."""
        issues = []
        
        if not dataset.samples:
            issues.append(ValidationIssue(
                type=IssueType.ERROR,
                message="Dataset contains no samples"
            ))
        
        if not dataset.metadata:
            issues.append(ValidationIssue(
                type=IssueType.WARNING,
                message="Dataset metadata is missing"
            ))
        
        return issues
    
    def _validate_sample(self, sample: HaluEvalSample) -> List[ValidationIssue]:
        """Validate a single sample."""
        issues = []
        
        # Required fields
        if not sample.id:
            issues.append(ValidationIssue(
                type=IssueType.ERROR,
                message="Sample ID is missing",
                sample_id=sample.id,
                field="id"
            ))
        
        if not sample.question or not sample.question.strip():
            issues.append(ValidationIssue(
                type=IssueType.ERROR,
                message="Question is empty",
                sample_id=sample.id,
                field="question"
            ))
        
        if not sample.answer or not sample.answer.strip():
            issues.append(ValidationIssue(
                type=IssueType.ERROR,
                message="Answer is empty", 
                sample_id=sample.id,
                field="answer"
            ))
        
        # Content quality checks
        if sample.question and len(sample.question) < 10:
            issues.append(ValidationIssue(
                type=IssueType.WARNING,
                message="Question is very short",
                sample_id=sample.id,
                field="question"
            ))
        
        if sample.answer and len(sample.answer) < 5:
            issues.append(ValidationIssue(
                type=IssueType.WARNING,
                message="Answer is very short",
                sample_id=sample.id,
                field="answer"
            ))
        
        # Confidence score validation
        if sample.confidence_score is not None:
            if not 0 <= sample.confidence_score <= 1:
                issues.append(ValidationIssue(
                    type=IssueType.ERROR,
                    message="Confidence score must be between 0 and 1",
                    sample_id=sample.id,
                    field="confidence_score"
                ))
        
        return issues
    
    def _compute_statistics(self, dataset: HaluEvalDataset) -> Dict[str, Any]:
        """Compute dataset statistics."""
        if not dataset.samples:
            return {}
        
        stats = {
            "total_samples": len(dataset.samples),
            "hallucinated_samples": sum(1 for s in dataset.samples if s.is_hallucinated),
            "non_hallucinated_samples": sum(1 for s in dataset.samples if not s.is_hallucinated),
            "avg_question_length": sum(len(s.question) for s in dataset.samples) / len(dataset.samples),
            "avg_answer_length": sum(len(s.answer) for s in dataset.samples) / len(dataset.samples),
            "samples_with_confidence": sum(1 for s in dataset.samples if s.confidence_score is not None),
            "samples_with_metadata": sum(1 for s in dataset.samples if s.metadata),
            "unique_hallucination_types": len(set(s.hallucination_type for s in dataset.samples if s.hallucination_type))
        }
        
        # Hallucination rate
        stats["hallucination_rate"] = stats["hallucinated_samples"] / stats["total_samples"]
        
        # Average confidence scores
        confidence_scores = [s.confidence_score for s in dataset.samples if s.confidence_score is not None]
        if confidence_scores:
            stats["avg_confidence_score"] = sum(confidence_scores) / len(confidence_scores)
            stats["min_confidence_score"] = min(confidence_scores)
            stats["max_confidence_score"] = max(confidence_scores)
        
        return stats
    
    def _validate_statistics(self, stats: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate dataset statistics."""
        issues = []
        
        if not stats:
            return issues
        
        # Check for reasonable distribution
        hallucination_rate = stats.get("hallucination_rate", 0)
        if hallucination_rate < 0.1:
            issues.append(ValidationIssue(
                type=IssueType.WARNING,
                message=f"Low hallucination rate: {hallucination_rate:.2%}"
            ))
        elif hallucination_rate > 0.9:
            issues.append(ValidationIssue(
                type=IssueType.WARNING,
                message=f"Very high hallucination rate: {hallucination_rate:.2%}"
            ))
        
        # Check content lengths
        avg_question_length = stats.get("avg_question_length", 0)
        if avg_question_length < 20:
            issues.append(ValidationIssue(
                type=IssueType.WARNING,
                message=f"Average question length is very short: {avg_question_length:.1f} characters"
            ))
        
        avg_answer_length = stats.get("avg_answer_length", 0)
        if avg_answer_length < 10:
            issues.append(ValidationIssue(
                type=IssueType.WARNING,
                message=f"Average answer length is very short: {avg_answer_length:.1f} characters"
            ))
        
        return issues
    
    def _validate_json_file(self, file_path: Path) -> ValidationReport:
        """Validate a JSON dataset file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to HaluEvalDataset
        samples = []
        for sample_data in data.get('samples', []):
            sample = HaluEvalSample(**sample_data)
            samples.append(sample)
        
        dataset = HaluEvalDataset(
            samples=samples,
            metadata=data.get('metadata', {})
        )
        
        return self.validate_dataset(dataset)
    
    def _validate_jsonl_file(self, file_path: Path) -> ValidationReport:
        """Validate a JSONL dataset file."""
        samples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample_data = json.loads(line.strip())
                    sample = HaluEvalSample(**sample_data)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    issues = [ValidationIssue(
                        type=IssueType.ERROR,
                        message=f"Invalid JSON on line {line_num}: {str(e)}"
                    )]
                    return ValidationReport(
                        is_valid=False,
                        total_samples=0,
                        issues=issues,
                        statistics={}
                    )
                except Exception as e:
                    issues = [ValidationIssue(
                        type=IssueType.ERROR,
                        message=f"Invalid sample on line {line_num}: {str(e)}"
                    )]
                    return ValidationReport(
                        is_valid=False,
                        total_samples=0,
                        issues=issues,
                        statistics={}
                    )
        
        dataset = HaluEvalDataset(samples=samples, metadata={})
        return self.validate_dataset(dataset)