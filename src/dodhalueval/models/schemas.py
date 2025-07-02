"""Core data models and schemas for DoDHaluEval."""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Literal
from pathlib import Path
from uuid import uuid4, UUID

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class BaseSchema(BaseModel):
    """Base schema with common functionality."""
    
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Path: str,
            UUID: str
        }
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseSchema':
        """Create instance from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BaseSchema':
        """Create instance from JSON string."""
        return cls.model_validate_json(json_str)


class DocumentChunk(BaseSchema):
    """Represents a chunk of text from a document."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str = Field(..., description="ID of the parent document")
    content: str = Field(..., description="Text content of the chunk")
    page_number: int = Field(..., ge=1, description="Page number in the document")
    chunk_index: int = Field(..., ge=0, description="Index of chunk within document")
    start_char: Optional[int] = Field(None, ge=0, description="Start character position")
    end_char: Optional[int] = Field(None, ge=0, description="End character position")
    section: Optional[str] = Field(None, description="Document section (chapter, etc.)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('end_char')
    @classmethod
    def validate_char_positions(cls, v: Optional[int], info) -> Optional[int]:
        """Ensure end_char is greater than start_char."""
        if v is not None and info.data.get('start_char') is not None:
            if v <= info.data['start_char']:
                raise ValueError('end_char must be greater than start_char')
        return v
    
    @property
    def word_count(self) -> int:
        """Get word count for the chunk."""
        return len(self.content.split())
    
    @property
    def char_count(self) -> int:
        """Get character count for the chunk."""
        return len(self.content)


class Document(BaseSchema):
    """Represents a source document."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., description="Document title")
    source_path: str = Field(..., description="Path to source file")
    file_hash: Optional[str] = Field(None, description="Hash of source file")
    page_count: int = Field(..., ge=1, description="Number of pages")
    content: List[DocumentChunk] = Field(default_factory=list, description="Document chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    processed_at: datetime = Field(default_factory=datetime.now, description="Processing timestamp")
    
    @field_validator('source_path')
    @classmethod
    def validate_source_path(cls, v: str) -> str:
        """Validate source path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f'Source file not found: {v}')
        return str(path.absolute())
    
    @property
    def total_chunks(self) -> int:
        """Get total number of chunks."""
        return len(self.content)
    
    @property
    def total_words(self) -> int:
        """Get total word count across all chunks."""
        return sum(chunk.word_count for chunk in self.content)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get chunk by ID."""
        for chunk in self.content:
            if chunk.id == chunk_id:
                return chunk
        return None
    
    def get_chunks_by_page(self, page_number: int) -> List[DocumentChunk]:
        """Get all chunks for a specific page."""
        return [chunk for chunk in self.content if chunk.page_number == page_number]


class Prompt(BaseSchema):
    """Represents a prompt for hallucination evaluation."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str = Field(..., description="Prompt text")
    source_document_id: str = Field(..., description="ID of source document")
    source_chunk_id: Optional[str] = Field(None, description="ID of specific source chunk")
    expected_answer: Optional[str] = Field(None, description="Expected answer or ground truth")
    hallucination_type: Optional[Literal['factual', 'logical', 'context']] = Field(
        None, description="Type of hallucination this prompt is designed to elicit"
    )
    generation_strategy: str = Field(..., description="Strategy used to generate this prompt")
    difficulty_level: Optional[Literal['easy', 'medium', 'hard']] = Field(
        None, description="Estimated difficulty level"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    @field_validator('text')
    @classmethod
    def validate_text_length(cls, v: str) -> str:
        """Ensure prompt text is not empty and within reasonable length."""
        if not v.strip():
            raise ValueError('Prompt text cannot be empty')
        if len(v) > 10000:
            raise ValueError('Prompt text too long (max 10000 characters)')
        return v.strip()
    
    @property
    def word_count(self) -> int:
        """Get word count for the prompt."""
        return len(self.text.split())


class Response(BaseSchema):
    """Represents a response to a prompt."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    prompt_id: str = Field(..., description="ID of the prompt this responds to")
    text: str = Field(..., description="Response text")
    model: str = Field(..., description="Model that generated the response")
    provider: str = Field(..., description="API provider used")
    generation_params: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters used for generation"
    )
    contains_hallucination: Optional[bool] = Field(None, description="Whether response contains hallucinations")
    hallucination_types: List[str] = Field(default_factory=list, description="Types of hallucinations injected")
    confidence_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score for the response"
    )
    hallucination_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Hallucination detection score"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    generated_at: datetime = Field(default_factory=datetime.now, description="Generation timestamp")
    
    @field_validator('text')
    @classmethod
    def validate_response_text(cls, v: str) -> str:
        """Ensure response text is not empty."""
        if not v.strip():
            raise ValueError('Response text cannot be empty')
        return v.strip()
    
    @property
    def word_count(self) -> int:
        """Get word count for the response."""
        return len(self.text.split())


class EvaluationResult(BaseSchema):
    """Represents the result of hallucination evaluation."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    response_id: str = Field(..., description="ID of the response being evaluated")
    method: str = Field(..., description="Evaluation method used")
    is_hallucinated: bool = Field(..., description="Whether response is hallucinated")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the evaluation")
    hallucination_type: Optional[Literal['factual', 'logical', 'context']] = Field(
        None, description="Type of hallucination detected"
    )
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed evaluation results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    evaluated_at: datetime = Field(default_factory=datetime.now, description="Evaluation timestamp")
    
    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate evaluation method."""
        valid_methods = {'vectara_hhem', 'g_eval', 'self_check_gpt', 'consistency_check', 'human'}
        if v not in valid_methods:
            raise ValueError(f'Invalid evaluation method: {v}. Valid methods: {valid_methods}')
        return v


class HumanAnnotation(BaseSchema):
    """Represents human annotation of hallucinations."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    response_id: str = Field(..., description="ID of the response being annotated")
    annotator_id: str = Field(..., description="ID of the human annotator")
    is_hallucinated: bool = Field(..., description="Human judgment on hallucination")
    severity: Literal['minor', 'major', 'critical'] = Field(..., description="Severity of hallucination")
    explanation: Optional[str] = Field(None, description="Explanation of the annotation")
    highlighted_text: Optional[str] = Field(None, description="Specific text that is hallucinated")
    correction: Optional[str] = Field(None, description="Suggested correction")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Annotator confidence")
    time_spent_seconds: Optional[int] = Field(None, ge=0, description="Time spent on annotation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    annotated_at: datetime = Field(default_factory=datetime.now, description="Annotation timestamp")


class PromptResponsePair(BaseSchema):
    """Represents a prompt-response pair for evaluation."""
    
    prompt: Prompt = Field(..., description="The prompt")
    response: Response = Field(..., description="The response")
    evaluations: List[EvaluationResult] = Field(default_factory=list, description="Evaluation results")
    human_annotations: List[HumanAnnotation] = Field(
        default_factory=list, description="Human annotations"
    )
    
    @model_validator(mode='after')
    def validate_prompt_response_match(self) -> 'PromptResponsePair':
        """Ensure response belongs to the prompt."""
        if self.prompt and self.response and self.response.prompt_id != self.prompt.id:
            raise ValueError('Response prompt_id does not match prompt id')
        return self
    
    @property
    def has_human_annotation(self) -> bool:
        """Check if pair has human annotation."""
        return len(self.human_annotations) > 0
    
    @property
    def consensus_hallucination_score(self) -> Optional[float]:
        """Get consensus hallucination score from all evaluations."""
        if not self.evaluations:
            return None
        
        scores = [eval_result.confidence_score for eval_result in self.evaluations]
        return sum(scores) / len(scores)
    
    def get_evaluation_by_method(self, method: str) -> Optional[EvaluationResult]:
        """Get evaluation result by method."""
        for evaluation in self.evaluations:
            if evaluation.method == method:
                return evaluation
        return None


class BenchmarkDataset(BaseSchema):
    """Represents a complete benchmark dataset."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Dataset name")
    version: str = Field(..., description="Dataset version")
    description: Optional[str] = Field(None, description="Dataset description")
    pairs: List[PromptResponsePair] = Field(default_factory=list, description="Prompt-response pairs")
    documents: List[Document] = Field(default_factory=list, description="Source documents")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    @property
    def total_pairs(self) -> int:
        """Get total number of prompt-response pairs."""
        return len(self.pairs)
    
    @property
    def total_documents(self) -> int:
        """Get total number of source documents."""
        return len(self.documents)
    
    @property
    def hallucination_rate(self) -> float:
        """Calculate overall hallucination rate."""
        if not self.pairs:
            return 0.0
        
        hallucinated_count = 0
        for pair in self.pairs:
            if pair.human_annotations:
                # Use human annotation if available
                if any(annotation.is_hallucinated for annotation in pair.human_annotations):
                    hallucinated_count += 1
            elif pair.evaluations:
                # Use automated evaluation consensus
                hallucinated_scores = [
                    eval_result.confidence_score 
                    for eval_result in pair.evaluations 
                    if eval_result.is_hallucinated
                ]
                if hallucinated_scores and sum(hallucinated_scores) / len(hallucinated_scores) > 0.5:
                    hallucinated_count += 1
        
        return hallucinated_count / len(self.pairs)
    
    def get_pairs_by_hallucination_type(self, hallucination_type: str) -> List[PromptResponsePair]:
        """Get pairs filtered by hallucination type."""
        return [
            pair for pair in self.pairs
            if pair.prompt.hallucination_type == hallucination_type
        ]
    
    def get_pairs_by_difficulty(self, difficulty: str) -> List[PromptResponsePair]:
        """Get pairs filtered by difficulty level."""
        return [
            pair for pair in self.pairs
            if pair.prompt.difficulty_level == difficulty
        ]
    
    def export_halueval_format(self) -> List[Dict[str, Any]]:
        """Export dataset in HaluEval-compatible format."""
        halueval_data = []
        
        for pair in self.pairs:
            # Get ground truth from human annotation or expected answer
            ground_truth = None
            is_hallucinated = None
            
            if pair.human_annotations:
                # Use most recent human annotation
                latest_annotation = max(pair.human_annotations, key=lambda x: x.annotated_at)
                is_hallucinated = latest_annotation.is_hallucinated
                ground_truth = latest_annotation.correction or pair.prompt.expected_answer
            else:
                ground_truth = pair.prompt.expected_answer
                # Use consensus from automated evaluations
                if pair.evaluations:
                    is_hallucinated = pair.consensus_hallucination_score > 0.5
            
            halueval_entry = {
                'id': pair.response.id,
                'prompt': pair.prompt.text,
                'response': pair.response.text,
                'hallucination_type': pair.prompt.hallucination_type,
                'hallucination_score': pair.consensus_hallucination_score,
                'ground_truth': ground_truth,
                'is_hallucinated': is_hallucinated,
                'metadata': {
                    'source_document_id': pair.prompt.source_document_id,
                    'model': pair.response.model,
                    'provider': pair.response.provider,
                    'generation_strategy': pair.prompt.generation_strategy,
                    'difficulty_level': pair.prompt.difficulty_level
                }
            }
            
            halueval_data.append(halueval_entry)
        
        return halueval_data
    
    def save_to_file(self, file_path: Union[str, Path], format: str = 'jsonl') -> None:
        """Save dataset to file.
        
        Args:
            file_path: Output file path
            format: Output format ('json', 'jsonl')
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'jsonl':
            # Save as JSONL (one JSON object per line)
            with open(file_path, 'w', encoding='utf-8') as f:
                for entry in self.export_halueval_format():
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        elif format == 'json':
            # Save as single JSON object
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.export_halueval_format(), f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'jsonl'")
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'BenchmarkDataset':
        """Load dataset from file.
        
        Args:
            file_path: Input file path
            
        Returns:
            Loaded dataset
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # This is a simplified loader - in practice, you'd need to reconstruct
        # the full object hierarchy from the flattened HaluEval format
        dataset = cls(
            name=file_path.stem,
            version="1.0",
            description=f"Dataset loaded from {file_path}"
        )
        
        return dataset