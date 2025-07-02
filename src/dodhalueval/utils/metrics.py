"""
Metrics Calculator for DoDHaluEval.

This module provides comprehensive metrics calculation for evaluating
hallucination detection performance across different methods and datasets.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import math
from statistics import mean, median, stdev

from ..core.evaluators.base import EvaluationResult
from ..core.hallucination_detector import EnsembleResult
from ..models.schemas import Response, Prompt
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricsReport:
    """Comprehensive metrics report for hallucination detection."""
    
    # Basic classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    
    # Threshold-based metrics
    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    # Advanced metrics
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    average_precision: Optional[float] = None
    
    # Score distribution metrics
    score_statistics: Dict[str, float] = None
    
    # Per-category metrics
    category_metrics: Dict[str, 'MetricsReport'] = None
    
    # Additional information
    sample_size: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.score_statistics is None:
            self.score_statistics = {}
        if self.category_metrics is None:
            self.category_metrics = {}
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def confusion_matrix(self) -> Dict[str, int]:
        """Get confusion matrix as dictionary."""
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives
        }
    
    @property
    def balanced_accuracy(self) -> float:
        """Calculate balanced accuracy (average of sensitivity and specificity)."""
        sensitivity = self.recall  # Same as recall
        return (sensitivity + self.specificity) / 2.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics report to dictionary."""
        result = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "specificity": self.specificity,
            "balanced_accuracy": self.balanced_accuracy,
            "threshold": self.threshold,
            "confusion_matrix": self.confusion_matrix,
            "sample_size": self.sample_size,
            "score_statistics": self.score_statistics,
            "metadata": self.metadata
        }
        
        if self.auc_roc is not None:
            result["auc_roc"] = self.auc_roc
        if self.auc_pr is not None:
            result["auc_pr"] = self.auc_pr
        if self.average_precision is not None:
            result["average_precision"] = self.average_precision
        
        if self.category_metrics:
            result["category_metrics"] = {
                cat: metrics.to_dict() for cat, metrics in self.category_metrics.items()
            }
        
        return result


class MetricsCalculator:
    """
    Calculator for hallucination detection metrics.
    
    Provides comprehensive evaluation metrics including classification metrics,
    threshold analysis, ROC curves, and category-specific performance.
    """
    
    def __init__(self, default_threshold: float = 0.5):
        self.default_threshold = default_threshold
        self.logger = get_logger(__name__)
    
    def calculate_detection_metrics(
        self,
        predictions: List[float],
        ground_truth: List[bool],
        threshold: float = None,
        positive_label: Union[bool, str] = True
    ) -> MetricsReport:
        """
        Calculate detection metrics for binary classification.
        
        Args:
            predictions: List of prediction scores (0.0-1.0)
            ground_truth: List of true labels (True=hallucination, False=no hallucination)
            threshold: Classification threshold (default: 0.5)
            positive_label: What constitutes a positive case
            
        Returns:
            MetricsReport with comprehensive metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        if not predictions:
            raise ValueError("Cannot calculate metrics for empty predictions")
        
        threshold = threshold or self.default_threshold
        
        # Convert predictions to binary using threshold
        binary_predictions = [score > threshold for score in predictions]
        
        # Calculate confusion matrix
        tp = sum(1 for pred, true in zip(binary_predictions, ground_truth) 
                if pred and true)
        fp = sum(1 for pred, true in zip(binary_predictions, ground_truth) 
                if pred and not true)
        tn = sum(1 for pred, true in zip(binary_predictions, ground_truth) 
                if not pred and not true)
        fn = sum(1 for pred, true in zip(binary_predictions, ground_truth) 
                if not pred and true)
        
        # Calculate basic metrics
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate score statistics
        score_stats = self._calculate_score_statistics(predictions, ground_truth)
        
        # Calculate advanced metrics if we have enough data
        auc_roc = None
        auc_pr = None
        average_precision = None
        
        try:
            if len(set(ground_truth)) > 1 and len(predictions) > 1:  # Need both classes
                auc_roc = self._calculate_auc_roc(predictions, ground_truth)
                auc_pr = self._calculate_auc_pr(predictions, ground_truth)
                average_precision = self._calculate_average_precision(predictions, ground_truth)
        except Exception as e:
            self.logger.warning(f"Could not calculate advanced metrics: {e}")
        
        return MetricsReport(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            specificity=specificity,
            threshold=threshold,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            average_precision=average_precision,
            score_statistics=score_stats,
            sample_size=len(predictions),
            metadata={
                "positive_label": positive_label,
                "threshold_used": threshold
            }
        )
    
    def calculate_per_category_metrics(
        self,
        evaluation_results: List[Union[EvaluationResult, EnsembleResult]],
        ground_truth: List[bool],
        category_labels: List[str],
        threshold: float = None
    ) -> Dict[str, MetricsReport]:
        """
        Calculate metrics broken down by categories (e.g., hallucination type).
        
        Args:
            evaluation_results: List of evaluation results
            ground_truth: List of true labels
            category_labels: List of category labels for each sample
            threshold: Classification threshold
            
        Returns:
            Dictionary mapping category names to MetricsReport objects
        """
        if len(evaluation_results) != len(ground_truth) or len(evaluation_results) != len(category_labels):
            raise ValueError("All input lists must have same length")
        
        threshold = threshold or self.default_threshold
        
        # Group by category
        categories = defaultdict(lambda: {"predictions": [], "truth": []})
        
        for result, truth, category in zip(evaluation_results, ground_truth, category_labels):
            # Extract score from result
            if hasattr(result, 'ensemble_score'):  # EnsembleResult
                score = result.ensemble_score
            elif hasattr(result, 'hallucination_score'):  # EvaluationResult
                score = result.hallucination_score.score
            else:
                self.logger.warning(f"Unknown result type: {type(result)}")
                continue
            
            categories[category]["predictions"].append(score)
            categories[category]["truth"].append(truth)
        
        # Calculate metrics for each category
        category_metrics = {}
        for category, data in categories.items():
            if len(data["predictions"]) > 0:
                try:
                    metrics = self.calculate_detection_metrics(
                        data["predictions"],
                        data["truth"],
                        threshold
                    )
                    metrics.metadata["category"] = category
                    category_metrics[category] = metrics
                except Exception as e:
                    self.logger.error(f"Failed to calculate metrics for category {category}: {e}")
        
        return category_metrics
    
    def calculate_evaluator_comparison(
        self,
        ensemble_results: List[EnsembleResult],
        ground_truth: List[bool],
        threshold: float = None
    ) -> Dict[str, MetricsReport]:
        """
        Compare performance of different evaluators within ensemble results.
        
        Args:
            ensemble_results: List of ensemble evaluation results
            ground_truth: List of true labels
            threshold: Classification threshold
            
        Returns:
            Dictionary mapping evaluator names to MetricsReport objects
        """
        threshold = threshold or self.default_threshold
        
        # Group results by evaluator
        evaluator_data = defaultdict(lambda: {"predictions": [], "truth": []})
        
        for ensemble_result, truth in zip(ensemble_results, ground_truth):
            for individual_result in ensemble_result.individual_results:
                evaluator_name = individual_result.evaluator_name
                score = individual_result.hallucination_score.score
                
                evaluator_data[evaluator_name]["predictions"].append(score)
                evaluator_data[evaluator_name]["truth"].append(truth)
        
        # Calculate metrics for each evaluator
        evaluator_metrics = {}
        for evaluator_name, data in evaluator_data.items():
            if len(data["predictions"]) > 0:
                try:
                    metrics = self.calculate_detection_metrics(
                        data["predictions"],
                        data["truth"], 
                        threshold
                    )
                    metrics.metadata["evaluator"] = evaluator_name
                    evaluator_metrics[evaluator_name] = metrics
                except Exception as e:
                    self.logger.error(f"Failed to calculate metrics for evaluator {evaluator_name}: {e}")
        
        return evaluator_metrics
    
    def find_optimal_threshold(
        self,
        predictions: List[float],
        ground_truth: List[bool],
        metric: str = "f1_score"
    ) -> Tuple[float, MetricsReport]:
        """
        Find optimal classification threshold for a given metric.
        
        Args:
            predictions: List of prediction scores
            ground_truth: List of true labels
            metric: Metric to optimize ("f1_score", "accuracy", "balanced_accuracy")
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_optimal_threshold)
        """
        if not predictions or not ground_truth:
            raise ValueError("Cannot find threshold for empty data")
        
        # Test thresholds from 0.1 to 0.9 in steps of 0.1
        thresholds = np.arange(0.1, 1.0, 0.1)
        best_threshold = 0.5
        best_score = 0.0
        best_metrics = None
        
        for threshold in thresholds:
            try:
                metrics = self.calculate_detection_metrics(predictions, ground_truth, threshold)
                
                # Get the score for the target metric
                if metric == "f1_score":
                    score = metrics.f1_score
                elif metric == "accuracy":
                    score = metrics.accuracy
                elif metric == "balanced_accuracy":
                    score = metrics.balanced_accuracy
                elif metric == "precision":
                    score = metrics.precision
                elif metric == "recall":
                    score = metrics.recall
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_metrics = metrics
                    
            except Exception as e:
                self.logger.warning(f"Failed to calculate metrics for threshold {threshold}: {e}")
        
        if best_metrics is None:
            # Fallback to default threshold
            best_metrics = self.calculate_detection_metrics(predictions, ground_truth, 0.5)
        
        return best_threshold, best_metrics
    
    def _calculate_score_statistics(
        self,
        predictions: List[float],
        ground_truth: List[bool]
    ) -> Dict[str, float]:
        """Calculate statistics about prediction scores."""
        stats = {
            "mean_score": mean(predictions),
            "median_score": median(predictions),
            "min_score": min(predictions),
            "max_score": max(predictions),
            "score_range": max(predictions) - min(predictions)
        }
        
        if len(predictions) > 1:
            stats["std_score"] = stdev(predictions)
        else:
            stats["std_score"] = 0.0
        
        # Separate statistics for positive and negative cases
        positive_scores = [p for p, t in zip(predictions, ground_truth) if t]
        negative_scores = [p for p, t in zip(predictions, ground_truth) if not t]
        
        if positive_scores:
            stats["mean_positive_score"] = mean(positive_scores)
            stats["std_positive_score"] = stdev(positive_scores) if len(positive_scores) > 1 else 0.0
        
        if negative_scores:
            stats["mean_negative_score"] = mean(negative_scores)
            stats["std_negative_score"] = stdev(negative_scores) if len(negative_scores) > 1 else 0.0
        
        # Score separation
        if positive_scores and negative_scores:
            stats["score_separation"] = abs(mean(positive_scores) - mean(negative_scores))
        
        return stats
    
    def _calculate_auc_roc(self, predictions: List[float], ground_truth: List[bool]) -> float:
        """Calculate Area Under ROC Curve using trapezoidal rule."""
        # Sort by prediction score (descending)
        sorted_pairs = sorted(zip(predictions, ground_truth), reverse=True)
        
        n_pos = sum(ground_truth)
        n_neg = len(ground_truth) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return float('nan')
        
        tp = fp = 0
        fpr_prev = tpr_prev = 0
        auc = 0
        
        for score, is_positive in sorted_pairs:
            if is_positive:
                tp += 1
            else:
                fp += 1
            
            fpr = fp / n_neg
            tpr = tp / n_pos
            
            # Trapezoidal rule
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
            fpr_prev, tpr_prev = fpr, tpr
        
        return auc
    
    def _calculate_auc_pr(self, predictions: List[float], ground_truth: List[bool]) -> float:
        """Calculate Area Under Precision-Recall Curve."""
        # Sort by prediction score (descending)
        sorted_pairs = sorted(zip(predictions, ground_truth), reverse=True)
        
        n_pos = sum(ground_truth)
        if n_pos == 0:
            return float('nan')
        
        tp = fp = 0
        precision_prev = 1.0
        recall_prev = 0.0
        auc = 0
        
        for score, is_positive in sorted_pairs:
            if is_positive:
                tp += 1
            else:
                fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / n_pos
            
            # Trapezoidal rule
            auc += (recall - recall_prev) * (precision + precision_prev) / 2
            precision_prev, recall_prev = precision, recall
        
        return auc
    
    def _calculate_average_precision(self, predictions: List[float], ground_truth: List[bool]) -> float:
        """Calculate Average Precision (AP) score."""
        # Sort by prediction score (descending)
        sorted_pairs = sorted(zip(predictions, ground_truth), reverse=True)
        
        n_pos = sum(ground_truth)
        if n_pos == 0:
            return 0.0
        
        tp = fp = 0
        average_precision = 0
        
        for score, is_positive in sorted_pairs:
            if is_positive:
                tp += 1
            else:
                fp += 1
            
            if is_positive:  # Only update when we have a positive sample
                precision = tp / (tp + fp)
                average_precision += precision
        
        return average_precision / n_pos
    
    def generate_summary_report(
        self,
        metrics_dict: Dict[str, MetricsReport],
        title: str = "Metrics Summary"
    ) -> str:
        """
        Generate a human-readable summary report.
        
        Args:
            metrics_dict: Dictionary of metrics reports
            title: Title for the report
            
        Returns:
            Formatted summary report as string
        """
        lines = [
            "=" * 60,
            f"  {title}",
            "=" * 60,
            ""
        ]
        
        if not metrics_dict:
            lines.append("No metrics to report.")
            return "\n".join(lines)
        
        # Overall summary
        all_accuracies = [m.accuracy for m in metrics_dict.values()]
        all_f1_scores = [m.f1_score for m in metrics_dict.values()]
        all_precisions = [m.precision for m in metrics_dict.values()]
        all_recalls = [m.recall for m in metrics_dict.values()]
        
        lines.extend([
            "OVERALL SUMMARY:",
            f"  Average Accuracy: {mean(all_accuracies):.3f} (±{stdev(all_accuracies):.3f})" if len(all_accuracies) > 1 else f"  Average Accuracy: {mean(all_accuracies):.3f}",
            f"  Average F1-Score: {mean(all_f1_scores):.3f} (±{stdev(all_f1_scores):.3f})" if len(all_f1_scores) > 1 else f"  Average F1-Score: {mean(all_f1_scores):.3f}",
            f"  Average Precision: {mean(all_precisions):.3f} (±{stdev(all_precisions):.3f})" if len(all_precisions) > 1 else f"  Average Precision: {mean(all_precisions):.3f}",
            f"  Average Recall: {mean(all_recalls):.3f} (±{stdev(all_recalls):.3f})" if len(all_recalls) > 1 else f"  Average Recall: {mean(all_recalls):.3f}",
            ""
        ])
        
        # Detailed breakdown
        lines.append("DETAILED BREAKDOWN:")
        for name, metrics in metrics_dict.items():
            lines.extend([
                f"  {name}:",
                f"    Accuracy:  {metrics.accuracy:.3f}",
                f"    Precision: {metrics.precision:.3f}",
                f"    Recall:    {metrics.recall:.3f}",
                f"    F1-Score:  {metrics.f1_score:.3f}",
                f"    Threshold: {metrics.threshold:.3f}",
                f"    Samples:   {metrics.sample_size}",
                ""
            ])
            
            if metrics.auc_roc is not None:
                lines.append(f"    AUC-ROC:   {metrics.auc_roc:.3f}")
            
            if metrics.auc_pr is not None:
                lines.append(f"    AUC-PR:    {metrics.auc_pr:.3f}")
            
            lines.append("")
        
        return "\n".join(lines)