"""
HaluEval compatibility layer for format conversion and validation.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from ..models.schemas import Response, PromptResponsePair, BenchmarkDataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetEntry:
    """Dataset entry for compatibility with existing code."""
    id: str
    question: str
    answer: str
    is_hallucinated: bool = False
    hallucination_type: Optional[str] = None
    confidence_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class HaluEvalEntry:
    """HaluEval-compatible dataset entry."""
    id: str
    question: str
    answer: str
    knowledge: str
    right_answer: Optional[str] = None
    is_hallucinated: bool = False
    hallucination_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class HaluEvalValidationReport:
    """Validation report for HaluEval compatibility."""
    is_valid: bool
    total_entries: int
    valid_entries: int
    missing_fields: List[str]
    format_errors: List[str]
    warnings: List[str]
    compliance_score: float


class HaluEvalCompatibilityLayer:
    """Ensures datasets are compatible with HaluEval evaluation metrics."""
    
    def __init__(self):
        """Initialize the compatibility layer."""
        self.required_fields = {'id', 'question', 'answer', 'knowledge'}
        self.optional_fields = {'right_answer', 'is_hallucinated', 'hallucination_type', 'metadata'}
        
    def convert_to_halueval_format(self, dataset: List[DatasetEntry]) -> List[Dict[str, Any]]:
        """Convert DoDHaluEval format to HaluEval benchmark format.
        
        Args:
            dataset: List of DoDHaluEval dataset entries
            
        Returns:
            List of HaluEval-compatible dictionaries
        """
        if not dataset:
            logger.warning("Empty dataset provided for conversion")
            return []
        
        converted_entries = []
        
        for entry in dataset:
            try:
                halueval_entry = self._convert_single_entry(entry)
                if halueval_entry:
                    converted_entries.append(halueval_entry)
            except Exception as e:
                logger.error(f"Failed to convert entry {entry.id}: {e}")
                continue
        
        logger.info(f"Converted {len(converted_entries)}/{len(dataset)} entries to HaluEval format")
        return converted_entries
    
    def _convert_single_entry(self, entry: DatasetEntry) -> Optional[Dict[str, Any]]:
        """Convert a single dataset entry to HaluEval format.
        
        Args:
            entry: DoDHaluEval dataset entry
            
        Returns:
            HaluEval-compatible dictionary or None if conversion fails
        """
        try:
            # Extract knowledge context from metadata
            knowledge_context = ""
            if hasattr(entry, 'metadata') and entry.metadata:
                knowledge_context = entry.metadata.get('knowledge_context', '')
                if not knowledge_context:
                    knowledge_context = entry.metadata.get('document_context', '')
                if not knowledge_context:
                    knowledge_context = entry.metadata.get('source_document', '')
            
            # Build HaluEval-compatible entry
            halueval_entry = {
                'id': entry.id,
                'question': entry.question,
                'answer': entry.answer,
                'knowledge': knowledge_context,
                'is_hallucinated': entry.is_hallucinated,
                'hallucination_type': self._map_hallucination_type(entry.hallucination_type),
                'metadata': {
                    'original_id': entry.id,
                    'generation_method': entry.metadata.get('generation_method', 'unknown') if entry.metadata else 'unknown',
                    'source_document': entry.metadata.get('source_document', '') if entry.metadata else '',
                    'confidence_score': entry.confidence_score,
                    'model': entry.metadata.get('model', '') if entry.metadata else '',
                    'provider': entry.metadata.get('provider', '') if entry.metadata else ''
                }
            }
            
            # Add right_answer if available in metadata
            if hasattr(entry, 'metadata') and entry.metadata:
                right_answer = entry.metadata.get('correct_answer', '')
                if right_answer:
                    halueval_entry['right_answer'] = right_answer
            
            return halueval_entry
            
        except Exception as e:
            logger.error(f"Error converting entry {entry.id}: {e}")
            return None
    
    def _map_hallucination_type(self, dod_type: Optional[str]) -> Optional[str]:
        """Map DoDHaluEval hallucination types to HaluEval types.
        
        Args:
            dod_type: DoDHaluEval hallucination type
            
        Returns:
            HaluEval-compatible hallucination type
        """
        if not dod_type:
            return None
            
        # Mapping from DoDHaluEval types to HaluEval types
        type_mapping = {
            'factual': 'factual_contradiction',
            'logical': 'invalid_inference', 
            'context': 'context_misunderstanding',
            'halueval_generated': 'factual_contradiction',
            'equipment_substitution': 'factual_contradiction',
            'branch_confusion': 'context_misunderstanding',
            'temporal_confusion': 'factual_contradiction'
        }
        
        return type_mapping.get(dod_type, dod_type)
    
    def convert_from_halueval_format(self, halueval_data: List[Dict[str, Any]]) -> List[DatasetEntry]:
        """Convert HaluEval format to DoDHaluEval format.
        
        Args:
            halueval_data: List of HaluEval-compatible dictionaries
            
        Returns:
            List of DoDHaluEval dataset entries
        """
        if not halueval_data:
            logger.warning("Empty HaluEval data provided for conversion")
            return []
        
        converted_entries = []
        
        for data in halueval_data:
            try:
                dod_entry = self._convert_from_halueval_single(data)
                if dod_entry:
                    converted_entries.append(dod_entry)
            except Exception as e:
                logger.error(f"Failed to convert HaluEval entry {data.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Converted {len(converted_entries)}/{len(halueval_data)} entries from HaluEval format")
        return converted_entries
    
    def _convert_from_halueval_single(self, data: Dict[str, Any]) -> Optional[DatasetEntry]:
        """Convert a single HaluEval entry to DoDHaluEval format.
        
        Args:
            data: HaluEval-compatible dictionary
            
        Returns:
            DoDHaluEval dataset entry or None if conversion fails
        """
        try:
            # Extract required fields
            entry_id = data.get('id', '')
            question = data.get('question', '')
            answer = data.get('answer', '')
            knowledge = data.get('knowledge', '')
            
            if not all([entry_id, question, answer]):
                logger.warning(f"Missing required fields for entry {entry_id}")
                return None
            
            # Extract optional fields
            is_hallucinated = data.get('is_hallucinated', False)
            hallucination_type = self._reverse_map_hallucination_type(data.get('hallucination_type'))
            confidence_score = data.get('metadata', {}).get('confidence_score') if data.get('metadata') else None
            
            # Build metadata
            metadata = {
                'knowledge_context': knowledge,
                'right_answer': data.get('right_answer', ''),
                'generation_method': data.get('metadata', {}).get('generation_method', 'halueval') if data.get('metadata') else 'halueval',
                'halueval_converted': True
            }
            
            # Add original metadata if available
            if data.get('metadata'):
                metadata.update(data['metadata'])
            
            # Create DoDHaluEval entry
            dod_entry = DatasetEntry(
                id=entry_id,
                question=question,
                answer=answer,
                is_hallucinated=is_hallucinated,
                hallucination_type=hallucination_type,
                confidence_score=confidence_score,
                metadata=metadata
            )
            
            return dod_entry
            
        except Exception as e:
            logger.error(f"Error converting HaluEval entry: {e}")
            return None
    
    def _reverse_map_hallucination_type(self, halueval_type: Optional[str]) -> Optional[str]:
        """Map HaluEval hallucination types back to DoDHaluEval types.
        
        Args:
            halueval_type: HaluEval hallucination type
            
        Returns:
            DoDHaluEval-compatible hallucination type
        """
        if not halueval_type:
            return None
            
        # Reverse mapping from HaluEval types to DoDHaluEval types
        reverse_mapping = {
            'factual_contradiction': 'factual',
            'invalid_inference': 'logical',
            'context_misunderstanding': 'context',
            'specificity_mismatch': 'context'
        }
        
        return reverse_mapping.get(halueval_type, halueval_type)
    
    def validate_halueval_compliance(self, dataset: List[Dict[str, Any]]) -> HaluEvalValidationReport:
        """Validate dataset meets HaluEval benchmark requirements.
        
        Args:
            dataset: List of dataset entries to validate
            
        Returns:
            Validation report with compliance details
        """
        if not dataset:
            return HaluEvalValidationReport(
                is_valid=False,
                total_entries=0,
                valid_entries=0,
                missing_fields=['dataset'],
                format_errors=['Empty dataset'],
                warnings=[],
                compliance_score=0.0
            )
        
        total_entries = len(dataset)
        valid_entries = 0
        missing_fields = []
        format_errors = []
        warnings = []
        
        # Track missing fields across all entries
        all_missing_fields = set()
        
        for i, entry in enumerate(dataset):
            entry_id = entry.get('id', f'entry_{i}')
            
            # Check required fields
            entry_missing_fields = []
            for field in self.required_fields:
                if field not in entry or not entry[field]:
                    entry_missing_fields.append(f"{entry_id}.{field}")
                    all_missing_fields.add(field)
            
            # Check field types and formats
            if 'id' in entry and not isinstance(entry['id'], str):
                format_errors.append(f"{entry_id}: 'id' must be string")
            
            if 'question' in entry and not isinstance(entry['question'], str):
                format_errors.append(f"{entry_id}: 'question' must be string")
            
            if 'answer' in entry and not isinstance(entry['answer'], str):
                format_errors.append(f"{entry_id}: 'answer' must be string")
            
            if 'knowledge' in entry and not isinstance(entry['knowledge'], str):
                format_errors.append(f"{entry_id}: 'knowledge' must be string")
            
            if 'is_hallucinated' in entry and not isinstance(entry['is_hallucinated'], bool):
                format_errors.append(f"{entry_id}: 'is_hallucinated' must be boolean")
            
            # Check for warnings
            if 'knowledge' in entry and len(entry['knowledge']) < 50:
                warnings.append(f"{entry_id}: knowledge context is very short (< 50 chars)")
            
            if 'answer' in entry and len(entry['answer']) < 10:
                warnings.append(f"{entry_id}: answer is very short (< 10 chars)")
            
            # Entry is valid if it has all required fields and no format errors for this entry
            if not entry_missing_fields and not any(entry_id in error for error in format_errors):
                valid_entries += 1
        
        missing_fields = list(all_missing_fields)
        compliance_score = valid_entries / total_entries if total_entries > 0 else 0.0
        is_valid = compliance_score >= 0.8 and len(format_errors) == 0
        
        return HaluEvalValidationReport(
            is_valid=is_valid,
            total_entries=total_entries,
            valid_entries=valid_entries,
            missing_fields=missing_fields,
            format_errors=format_errors,
            warnings=warnings,
            compliance_score=compliance_score
        )
    
    def generate_halueval_metrics(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate metrics compatible with HaluEval benchmark.
        
        Args:
            dataset: List of HaluEval-compatible dataset entries
            
        Returns:
            Dictionary with HaluEval-compatible metrics
        """
        if not dataset:
            return {
                'total_samples': 0,
                'hallucinated_samples': 0,
                'hallucination_rate': 0.0,
                'avg_question_length': 0.0,
                'avg_answer_length': 0.0,
                'avg_knowledge_length': 0.0,
                'hallucination_types': {},
                'quality_metrics': {}
            }
        
        total_samples = len(dataset)
        hallucinated_samples = sum(1 for entry in dataset if entry.get('is_hallucinated', False))
        
        # Calculate average lengths
        question_lengths = [len(entry.get('question', '')) for entry in dataset]
        answer_lengths = [len(entry.get('answer', '')) for entry in dataset]
        knowledge_lengths = [len(entry.get('knowledge', '')) for entry in dataset]
        
        avg_question_length = sum(question_lengths) / len(question_lengths) if question_lengths else 0.0
        avg_answer_length = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0.0
        avg_knowledge_length = sum(knowledge_lengths) / len(knowledge_lengths) if knowledge_lengths else 0.0
        
        # Count hallucination types
        hallucination_types = {}
        for entry in dataset:
            if entry.get('is_hallucinated', False):
                hal_type = entry.get('hallucination_type', 'unknown')
                hallucination_types[hal_type] = hallucination_types.get(hal_type, 0) + 1
        
        # Quality metrics
        quality_metrics = {
            'entries_with_knowledge': sum(1 for entry in dataset if entry.get('knowledge', '')),
            'entries_with_right_answer': sum(1 for entry in dataset if entry.get('right_answer', '')),
            'avg_knowledge_quality': self._calculate_knowledge_quality(dataset),
            'response_diversity': self._calculate_response_diversity(dataset)
        }
        
        return {
            'total_samples': total_samples,
            'hallucinated_samples': hallucinated_samples,
            'hallucination_rate': hallucinated_samples / total_samples if total_samples > 0 else 0.0,
            'avg_question_length': avg_question_length,
            'avg_answer_length': avg_answer_length,
            'avg_knowledge_length': avg_knowledge_length,
            'hallucination_types': hallucination_types,
            'quality_metrics': quality_metrics
        }
    
    def _calculate_knowledge_quality(self, dataset: List[Dict[str, Any]]) -> float:
        """Calculate average knowledge context quality score.
        
        Args:
            dataset: List of dataset entries
            
        Returns:
            Average quality score (0.0 to 1.0)
        """
        if not dataset:
            return 0.0
        
        quality_scores = []
        for entry in dataset:
            knowledge = entry.get('knowledge', '')
            if not knowledge:
                quality_scores.append(0.0)
                continue
            
            # Simple quality heuristics
            score = 0.0
            
            # Length scoring (optimal range 100-800 chars)
            if 100 <= len(knowledge) <= 800:
                score += 0.4
            elif 50 <= len(knowledge) < 100 or 800 < len(knowledge) <= 1200:
                score += 0.2
            
            # Word count scoring
            word_count = len(knowledge.split())
            if 20 <= word_count <= 150:
                score += 0.3
            elif 10 <= word_count < 20 or 150 < word_count <= 200:
                score += 0.15
            
            # Sentence structure scoring
            sentence_count = len([s for s in knowledge.split('.') if s.strip()])
            if sentence_count >= 2:
                score += 0.3
            elif sentence_count == 1:
                score += 0.15
            
            quality_scores.append(min(score, 1.0))
        
        return sum(quality_scores) / len(quality_scores)
    
    def _calculate_response_diversity(self, dataset: List[Dict[str, Any]]) -> float:
        """Calculate response diversity score.
        
        Args:
            dataset: List of dataset entries
            
        Returns:
            Diversity score (0.0 to 1.0)
        """
        if not dataset:
            return 0.0
        
        answers = [entry.get('answer', '') for entry in dataset]
        unique_answers = set(answers)
        
        # Diversity based on unique vs total responses
        diversity_ratio = len(unique_answers) / len(answers) if answers else 0.0
        
        return diversity_ratio
    
    def export_halueval_format(self, dataset: List[Dict[str, Any]], output_path: Path) -> bool:
        """Export dataset in HaluEval format to file.
        
        Args:
            dataset: HaluEval-compatible dataset
            output_path: Path to save the dataset
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                import json
                for entry in dataset:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            logger.info(f"Exported {len(dataset)} entries to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export HaluEval dataset: {e}")
            return False