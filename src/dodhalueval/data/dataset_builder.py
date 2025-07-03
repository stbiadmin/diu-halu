"""
Dataset Builder for DoDHaluEval.

This module builds HaluEval-compatible datasets from prompts, responses,
and evaluation results, with support for multiple export formats.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

from ..models.schemas import Prompt, Response, EvaluationResult, PromptResponsePair
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HaluEvalSample:
    """A single sample in HaluEval format."""
    id: str
    question: str
    answer: str
    is_hallucinated: bool
    hallucination_type: Optional[str] = None
    confidence_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass 
class HaluEvalDataset:
    """Complete HaluEval dataset."""
    samples: List[HaluEvalSample]
    metadata: Dict[str, Any]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self):
        return iter(self.samples)


class DatasetBuilder:
    """Builds HaluEval-compatible datasets from DoDHaluEval components."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("datasets")
        self.output_dir.mkdir(exist_ok=True)
        
    def build_from_pairs(self, pairs: List[Union[PromptResponsePair, Dict[str, Any]]], 
                        dataset_name: str = "dodhalueval",
                        additional_metadata: Optional[Dict[str, Any]] = None) -> HaluEvalDataset:
        """Build dataset from prompt-response pairs."""
        samples = []
        
        for pair in pairs:
            # Handle both PromptResponsePair objects and dictionaries
            if isinstance(pair, dict):
                prompt = pair['prompt']
                response = pair['response']
                evaluation = pair.get('evaluation')
                evaluations = [evaluation] if evaluation else []
            else:
                prompt = pair.prompt
                response = pair.response
                evaluations = pair.evaluations
            
            # Get the best evaluation result
            eval_result = None
            if evaluations:
                # Handle both individual evaluation results and ensemble results
                def get_confidence(x):
                    if hasattr(x, 'confidence_score'):
                        return x.confidence_score or 0
                    elif hasattr(x, 'ensemble_confidence'):
                        return x.ensemble_confidence or 0
                    else:
                        return 0
                eval_result = max(evaluations, key=get_confidence)
            
            sample = HaluEvalSample(
                id=f"{dataset_name}_{len(samples)}",
                question=prompt.text,
                answer=response.text,
                is_hallucinated=eval_result.is_hallucinated if eval_result else False,
                hallucination_type=prompt.hallucination_type,
                confidence_score=getattr(eval_result, 'confidence_score', None) or getattr(eval_result, 'ensemble_confidence', None) if eval_result else None,
                metadata={
                    "prompt_id": prompt.id,
                    "response_id": response.id,
                    "model": response.model,
                    "provider": response.provider,
                    "source_document": prompt.source_document_id,
                    "generation_strategy": prompt.generation_strategy,
                    "difficulty_level": prompt.difficulty_level
                }
            )
            samples.append(sample)
        
        metadata = {
            "dataset_name": dataset_name,
            "created_at": datetime.now().isoformat(),
            "total_samples": len(samples),
            "hallucinated_samples": sum(1 for s in samples if s.is_hallucinated),
            "version": "1.0"
        }
        
        # Add additional metadata if provided
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return HaluEvalDataset(samples=samples, metadata=metadata)
    
    def build_halueval_format(
        self,
        prompts: List[Prompt],
        responses: List[Response],
        evaluations: Optional[List[EvaluationResult]] = None,
        documents: Optional[List['Document']] = None,
        dataset_name: str = "dod_halueval",
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> HaluEvalDataset:
        """Build dataset in HaluEval-compatible format as documented in API."""
        # Create pairs from the inputs
        pairs = []
        evaluations = evaluations or []
        
        # Match responses to prompts and evaluations
        for i, response in enumerate(responses):
            # Find matching prompt
            prompt = None
            for p in prompts:
                if p.id == response.prompt_id:
                    prompt = p
                    break
            
            # Find matching evaluation
            evaluation = None
            for e in evaluations:
                if e.response_id == response.id:
                    evaluation = e
                    break
            
            if prompt:  # Only include if we have a matching prompt
                pairs.append({
                    'prompt': prompt,
                    'response': response,
                    'evaluation': evaluation
                })
        
        return self.build_from_pairs(pairs, dataset_name, additional_metadata)
    
    def export_jsonl(self, dataset: HaluEvalDataset, filename: str) -> str:
        """Export dataset to JSONL format as documented in API."""
        filepath = self.save_jsonl(dataset, filename.replace('.jsonl', ''))
        return str(filepath)
    
    def export_json(self, dataset: HaluEvalDataset, filename: str) -> str:
        """Export dataset to JSON format as documented in API.""" 
        filepath = self.save_json(dataset, filename.replace('.json', ''))
        return str(filepath)
    
    def export_csv(self, dataset: HaluEvalDataset, filename: str) -> str:
        """Export dataset to CSV format as documented in API."""
        filepath = self.save_csv(dataset, filename.replace('.csv', ''))
        return str(filepath)
    
    def save_jsonl(self, dataset: HaluEvalDataset, filename: str) -> Path:
        """Save dataset in JSONL format."""
        filepath = self.output_dir / f"{filename}.jsonl"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in dataset.samples:
                f.write(json.dumps(asdict(sample)) + '\n')
        
        logger.info(f"Saved {len(dataset)} samples to {filepath}")
        return filepath
    
    def save_json(self, dataset: HaluEvalDataset, filename: str) -> Path:
        """Save dataset in JSON format."""
        filepath = self.output_dir / f"{filename}.json"
        
        data = {
            "metadata": dataset.metadata,
            "samples": [asdict(sample) for sample in dataset.samples]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(dataset)} samples to {filepath}")
        return filepath
    
    def save_csv(self, dataset: HaluEvalDataset, filename: str) -> Path:
        """Save dataset in CSV format."""
        filepath = self.output_dir / f"{filename}.csv"
        
        if not dataset.samples:
            logger.warning("No samples to save")
            return filepath
        
        fieldnames = list(asdict(dataset.samples[0]).keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sample in dataset.samples:
                # Convert metadata dict to string for CSV
                sample_dict = asdict(sample)
                if sample_dict.get('metadata'):
                    sample_dict['metadata'] = json.dumps(sample_dict['metadata'])
                writer.writerow(sample_dict)
        
        logger.info(f"Saved {len(dataset)} samples to {filepath}")
        return filepath