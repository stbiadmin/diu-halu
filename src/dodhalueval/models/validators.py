"""Validation utilities and custom validators for DoDHaluEval models."""

import re
import json
import jsonlines
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
from datetime import datetime

from pydantic import ValidationError

from dodhalueval.models.schemas import (
    BaseSchema,
    Document,
    DocumentChunk,
    Prompt,
    Response,
    EvaluationResult,
    HumanAnnotation,
    PromptResponsePair,
    BenchmarkDataset
)
from dodhalueval.utils.exceptions import ValidationError as DoDValidationError
from dodhalueval.utils.logger import get_logger

logger = get_logger(__name__)
T = TypeVar('T', bound=BaseSchema)


class DataValidator:
    """Comprehensive data validation utilities."""
    
    @staticmethod
    def validate_text_quality(text: str, min_length: int = 10, max_length: int = 10000) -> bool:
        """Validate text quality and length.
        
        Args:
            text: Text to validate
            min_length: Minimum character length
            max_length: Maximum character length
            
        Returns:
            True if text passes quality checks
            
        Raises:
            DoDValidationError: If validation fails
        """
        if not isinstance(text, str):
            raise DoDValidationError("Text must be a string", field="text", value=type(text))
        
        # Remove whitespace for length check
        clean_text = text.strip()
        
        if len(clean_text) < min_length:
            raise DoDValidationError(
                f"Text too short (minimum {min_length} characters)",
                field="text",
                value=len(clean_text)
            )
        
        if len(clean_text) > max_length:
            raise DoDValidationError(
                f"Text too long (maximum {max_length} characters)",
                field="text",
                value=len(clean_text)
            )
        
        # Check for reasonable character distribution
        if not DataValidator._has_reasonable_character_distribution(clean_text):
            raise DoDValidationError(
                "Text contains unusual character distribution",
                field="text",
                value=clean_text[:100]
            )
        
        return True
    
    @staticmethod
    def _has_reasonable_character_distribution(text: str) -> bool:
        """Check if text has reasonable character distribution."""
        if len(text) < 10:
            return True
        
        # Count different character types
        letters = sum(1 for c in text if c.isalpha())
        digits = sum(1 for c in text if c.isdigit())
        spaces = sum(1 for c in text if c.isspace())
        punct = sum(1 for c in text if c in '.,!?;:-')
        
        total_chars = len(text)
        
        # Text should be mostly letters
        if letters / total_chars < 0.5:
            return False
        
        # Should have some spaces (not a single word)
        if spaces / total_chars < 0.1:
            return False
        
        # Shouldn't be mostly digits
        if digits / total_chars > 0.5:
            return False
        
        return True
    
    @staticmethod
    def validate_hallucination_score(score: float) -> bool:
        """Validate hallucination confidence score.
        
        Args:
            score: Score to validate (should be 0.0-1.0)
            
        Returns:
            True if score is valid
            
        Raises:
            DoDValidationError: If score is invalid
        """
        if not isinstance(score, (int, float)):
            raise DoDValidationError(
                "Hallucination score must be numeric",
                field="hallucination_score",
                value=type(score)
            )
        
        if not (0.0 <= score <= 1.0):
            raise DoDValidationError(
                "Hallucination score must be between 0.0 and 1.0",
                field="hallucination_score",
                value=score
            )
        
        return True
    
    @staticmethod
    def validate_model_name(model_name: str) -> bool:
        """Validate model name format.
        
        Args:
            model_name: Model name to validate
            
        Returns:
            True if model name is valid
            
        Raises:
            DoDValidationError: If model name is invalid
        """
        if not isinstance(model_name, str):
            raise DoDValidationError(
                "Model name must be a string",
                field="model",
                value=type(model_name)
            )
        
        # Model name should not be empty
        if not model_name.strip():
            raise DoDValidationError(
                "Model name cannot be empty",
                field="model",
                value=model_name
            )
        
        # Check for reasonable model name patterns
        valid_pattern = r'^[a-zA-Z0-9\-_.]+$'
        if not re.match(valid_pattern, model_name):
            raise DoDValidationError(
                "Model name contains invalid characters",
                field="model",
                value=model_name
            )
        
        return True
    
    @staticmethod
    def validate_prompt_response_consistency(prompt: Prompt, response: Response) -> bool:
        """Validate consistency between prompt and response.
        
        Args:
            prompt: Prompt object
            response: Response object
            
        Returns:
            True if prompt and response are consistent
            
        Raises:
            DoDValidationError: If validation fails
        """
        # Check that response references the correct prompt
        if response.prompt_id != prompt.id:
            raise DoDValidationError(
                "Response prompt_id does not match prompt id",
                field="prompt_id",
                value=f"response: {response.prompt_id}, prompt: {prompt.id}"
            )
        
        # Check temporal consistency (response should be after prompt)
        if response.generated_at < prompt.created_at:
            raise DoDValidationError(
                "Response timestamp is before prompt timestamp",
                field="generated_at",
                value=f"response: {response.generated_at}, prompt: {prompt.created_at}"
            )
        
        # Check response relevance (basic checks)
        if len(response.text.strip()) < 10:
            raise DoDValidationError(
                "Response text is too short to be meaningful",
                field="text",
                value=len(response.text)
            )
        
        return True
    
    @staticmethod
    def validate_evaluation_consistency(evaluation: EvaluationResult, response: Response) -> bool:
        """Validate consistency between evaluation and response.
        
        Args:
            evaluation: Evaluation result
            response: Response being evaluated
            
        Returns:
            True if evaluation is consistent
            
        Raises:
            DoDValidationError: If validation fails
        """
        # Check that evaluation references the correct response
        if evaluation.response_id != response.id:
            raise DoDValidationError(
                "Evaluation response_id does not match response id",
                field="response_id",
                value=f"evaluation: {evaluation.response_id}, response: {response.id}"
            )
        
        # Check temporal consistency
        if evaluation.evaluated_at < response.generated_at:
            raise DoDValidationError(
                "Evaluation timestamp is before response timestamp",
                field="evaluated_at",
                value=f"evaluation: {evaluation.evaluated_at}, response: {response.generated_at}"
            )
        
        # Validate score consistency
        DataValidator.validate_hallucination_score(evaluation.confidence_score)
        
        return True


class ModelSerializer:
    """Serialization utilities for DoDHaluEval models."""
    
    @staticmethod
    def serialize_to_dict(obj: BaseSchema) -> Dict[str, Any]:
        """Serialize object to dictionary with type information.
        
        Args:
            obj: Object to serialize
            
        Returns:
            Dictionary representation with metadata
        """
        data = obj.dict()
        data['_type'] = obj.__class__.__name__
        data['_version'] = getattr(obj, '__version__', '1.0')
        data['_serialized_at'] = datetime.now().isoformat()
        
        return data
    
    @staticmethod
    def deserialize_from_dict(data: Dict[str, Any]) -> BaseSchema:
        """Deserialize object from dictionary.
        
        Args:
            data: Dictionary to deserialize
            
        Returns:
            Deserialized object
            
        Raises:
            DoDValidationError: If deserialization fails
        """
        if '_type' not in data:
            raise DoDValidationError("Missing type information in serialized data")
        
        obj_type = data.pop('_type')
        data.pop('_version', None)  # Remove metadata
        data.pop('_serialized_at', None)
        
        # Map type names to classes
        type_mapping = {
            'Document': Document,
            'DocumentChunk': DocumentChunk,
            'Prompt': Prompt,
            'Response': Response,
            'EvaluationResult': EvaluationResult,
            'HumanAnnotation': HumanAnnotation,
            'PromptResponsePair': PromptResponsePair,
            'BenchmarkDataset': BenchmarkDataset
        }
        
        if obj_type not in type_mapping:
            raise DoDValidationError(f"Unknown object type: {obj_type}")
        
        target_class = type_mapping[obj_type]
        
        try:
            return target_class(**data)
        except ValidationError as e:
            raise DoDValidationError(f"Failed to deserialize {obj_type}: {e}")
    
    @staticmethod
    def save_to_json(obj: BaseSchema, file_path: Union[str, Path]) -> None:
        """Save object to JSON file.
        
        Args:
            obj: Object to save
            file_path: Output file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = ModelSerializer.serialize_to_dict(obj)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved {obj.__class__.__name__} to {file_path}")
    
    @staticmethod
    def load_from_json(file_path: Union[str, Path]) -> BaseSchema:
        """Load object from JSON file.
        
        Args:
            file_path: Input file path
            
        Returns:
            Loaded object
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DoDValidationError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        obj = ModelSerializer.deserialize_from_dict(data)
        logger.info(f"Loaded {obj.__class__.__name__} from {file_path}")
        
        return obj
    
    @staticmethod
    def save_list_to_jsonl(objects: List[BaseSchema], file_path: Union[str, Path]) -> None:
        """Save list of objects to JSONL file.
        
        Args:
            objects: List of objects to save
            file_path: Output file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with jsonlines.open(file_path, 'w') as writer:
            for obj in objects:
                data = ModelSerializer.serialize_to_dict(obj)
                writer.write(data)
        
        logger.info(f"Saved {len(objects)} objects to {file_path}")
    
    @staticmethod
    def load_list_from_jsonl(file_path: Union[str, Path]) -> List[BaseSchema]:
        """Load list of objects from JSONL file.
        
        Args:
            file_path: Input file path
            
        Returns:
            List of loaded objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DoDValidationError(f"File not found: {file_path}")
        
        objects = []
        with jsonlines.open(file_path, 'r') as reader:
            for data in reader:
                obj = ModelSerializer.deserialize_from_dict(data)
                objects.append(obj)
        
        logger.info(f"Loaded {len(objects)} objects from {file_path}")
        return objects


class DatasetValidator:
    """Validation utilities specifically for datasets."""
    
    @staticmethod
    def validate_dataset_completeness(dataset: BenchmarkDataset) -> bool:
        """Validate that dataset is complete and consistent.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            True if dataset is valid
            
        Raises:
            DoDValidationError: If validation fails
        """
        if not dataset.pairs:
            raise DoDValidationError("Dataset must contain at least one prompt-response pair")
        
        if not dataset.documents:
            raise DoDValidationError("Dataset must contain at least one source document")
        
        # Check that all prompts reference valid documents
        document_ids = {doc.id for doc in dataset.documents}
        
        for i, pair in enumerate(dataset.pairs):
            if pair.prompt.source_document_id not in document_ids:
                raise DoDValidationError(
                    f"Prompt {i} references non-existent document",
                    field="source_document_id",
                    value=pair.prompt.source_document_id
                )
        
        # Check for duplicate IDs
        DatasetValidator._check_duplicate_ids(dataset)
        
        # Validate individual pairs
        for i, pair in enumerate(dataset.pairs):
            try:
                DataValidator.validate_prompt_response_consistency(pair.prompt, pair.response)
            except DoDValidationError as e:
                raise DoDValidationError(f"Pair {i} validation failed: {e.message}")
        
        return True
    
    @staticmethod
    def _check_duplicate_ids(dataset: BenchmarkDataset) -> None:
        """Check for duplicate IDs in dataset."""
        # Check document IDs
        doc_ids = [doc.id for doc in dataset.documents]
        if len(doc_ids) != len(set(doc_ids)):
            raise DoDValidationError("Duplicate document IDs found")
        
        # Check prompt IDs
        prompt_ids = [pair.prompt.id for pair in dataset.pairs]
        if len(prompt_ids) != len(set(prompt_ids)):
            raise DoDValidationError("Duplicate prompt IDs found")
        
        # Check response IDs
        response_ids = [pair.response.id for pair in dataset.pairs]
        if len(response_ids) != len(set(response_ids)):
            raise DoDValidationError("Duplicate response IDs found")
    
    @staticmethod
    def validate_annotation_quality(annotation: HumanAnnotation, response: Response) -> bool:
        """Validate quality of human annotation.
        
        Args:
            annotation: Human annotation to validate
            response: Response being annotated
            
        Returns:
            True if annotation is valid
            
        Raises:
            DoDValidationError: If validation fails
        """
        # Check basic consistency
        if annotation.response_id != response.id:
            raise DoDValidationError(
                "Annotation response_id does not match response id",
                field="response_id",
                value=f"annotation: {annotation.response_id}, response: {response.id}"
            )
        
        # Check highlighted text is in response
        if annotation.highlighted_text:
            if annotation.highlighted_text not in response.text:
                raise DoDValidationError(
                    "Highlighted text not found in response",
                    field="highlighted_text",
                    value=annotation.highlighted_text[:100]
                )
        
        # Check confidence score
        DataValidator.validate_hallucination_score(annotation.confidence)
        
        # Check explanation quality if provided
        if annotation.explanation:
            DataValidator.validate_text_quality(
                annotation.explanation,
                min_length=5,
                max_length=1000
            )
        
        return True
    
    @staticmethod
    def calculate_annotation_agreement(annotations: List[HumanAnnotation]) -> float:
        """Calculate inter-annotator agreement.
        
        Args:
            annotations: List of annotations for the same response
            
        Returns:
            Agreement score (0.0 to 1.0)
        """
        if len(annotations) < 2:
            return 1.0  # Perfect agreement with single annotator
        
        # Simple agreement: percentage of annotators who agree on hallucination
        hallucination_judgments = [ann.is_hallucinated for ann in annotations]
        most_common = max(set(hallucination_judgments), key=hallucination_judgments.count)
        agreement_count = hallucination_judgments.count(most_common)
        
        return agreement_count / len(annotations)


def validate_and_clean_data(
    data: List[Dict[str, Any]],
    target_type: Type[T],
    remove_invalid: bool = False
) -> List[T]:
    """Validate and clean a list of data dictionaries.
    
    Args:
        data: List of data dictionaries
        target_type: Target Pydantic model type
        remove_invalid: Whether to remove invalid entries or raise error
        
    Returns:
        List of validated objects
        
    Raises:
        DoDValidationError: If validation fails and remove_invalid is False
    """
    validated_objects = []
    errors = []
    
    for i, item in enumerate(data):
        try:
            obj = target_type(**item)
            validated_objects.append(obj)
        except ValidationError as e:
            error_msg = f"Item {i}: {e}"
            errors.append(error_msg)
            
            if not remove_invalid:
                raise DoDValidationError(f"Validation failed for item {i}: {e}")
    
    if errors and remove_invalid:
        logger.warning(f"Removed {len(errors)} invalid items during validation")
        for error in errors[:5]:  # Log first 5 errors
            logger.warning(error)
    
    return validated_objects