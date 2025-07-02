"""Prompt validation system for DoDHaluEval.

This module validates generated prompts to ensure they are suitable
for hallucination evaluation and meet quality criteria.
"""

import re
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from dodhalueval.models.schemas import Prompt, DocumentChunk
from dodhalueval.providers.base import LLMProvider, GenerationParameters
from dodhalueval.utils.logger import get_logger


@dataclass
class ValidationResult:
    """Result of prompt validation."""
    
    is_valid: bool
    score: float  # 0.0 to 1.0
    issues: List[str]
    suggestions: List[str]
    metrics: Dict[str, Any]
    
    @property
    def quality_level(self) -> str:
        """Get quality level based on score."""
        if self.score >= 0.8:
            return "high"
        elif self.score >= 0.6:
            return "medium"
        else:
            return "low"


@dataclass
class PromptMetrics:
    """Metrics calculated for a prompt."""
    
    word_count: int
    char_count: int
    complexity_score: float
    clarity_score: float
    specificity_score: float
    answerability_score: float
    hallucination_potential: float
    domain_relevance: float


class PromptValidator:
    """Validates prompts for quality and suitability."""
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm_provider = llm_provider
        self.logger = get_logger(__name__)
        
        # Define validation criteria
        self.min_word_count = 5
        self.max_word_count = 100
        self.min_complexity_score = 0.3
        self.min_clarity_score = 0.5
        
        # Domain-specific keywords for military/defense
        self.domain_keywords = {
            'military_operations': ['operation', 'mission', 'deployment', 'combat', 'tactical', 'strategic'],
            'equipment': ['weapon', 'vehicle', 'aircraft', 'ship', 'equipment', 'system', 'technology'],
            'personnel': ['soldier', 'officer', 'crew', 'personnel', 'rank', 'unit', 'force'],
            'procedures': ['procedure', 'protocol', 'regulation', 'instruction', 'manual', 'doctrine'],
            'locations': ['base', 'fort', 'camp', 'theater', 'region', 'zone', 'area'],
            'training': ['training', 'exercise', 'drill', 'simulation', 'practice'],
            'logistics': ['supply', 'logistics', 'transport', 'maintenance', 'support']
        }
        
        # Question quality indicators
        self.quality_indicators = {
            'specificity': ['specific', 'exact', 'precise', 'detailed', 'particular'],
            'clarity': ['what', 'how', 'when', 'where', 'why', 'which', 'who'],
            'complexity': ['relationship', 'compare', 'analyze', 'evaluate', 'implications']
        }
        
        # Red flags that indicate poor quality
        self.red_flags = [
            r'\b(tell me everything|anything|whatever)\b',  # Too vague
            r'\b(yes or no|true or false)\b',  # Too simple
            r'\b(I don\'t know|no idea)\b',  # Nonsensical
            r'^.{1,10}$',  # Too short
            r'^.{500,}$',  # Too long
            r'\b(secret|classified|confidential)\s+(details|information)\b'  # Inappropriate
        ]
    
    async def validate_prompt(
        self,
        prompt: Prompt,
        source_chunk: DocumentChunk,
        use_llm_validation: bool = True
    ) -> ValidationResult:
        """Validate a single prompt comprehensively."""
        
        issues = []
        suggestions = []
        
        # Calculate basic metrics
        metrics = self._calculate_metrics(prompt, source_chunk)
        
        # Perform rule-based validation
        rule_issues, rule_suggestions = self._validate_with_rules(prompt, metrics)
        issues.extend(rule_issues)
        suggestions.extend(rule_suggestions)
        
        # Perform LLM-based validation if provider available
        llm_score = 0.5  # Default score
        if use_llm_validation and self.llm_provider:
            try:
                llm_result = await self._validate_with_llm(prompt, source_chunk)
                llm_score = llm_result.get('score', 0.5)
                issues.extend(llm_result.get('issues', []))
                suggestions.extend(llm_result.get('suggestions', []))
            except Exception as e:
                self.logger.warning(f"LLM validation failed: {e}")
        
        # Calculate overall score
        rule_score = self._calculate_rule_score(metrics, issues)
        overall_score = (rule_score + llm_score) / 2
        
        # Determine if prompt is valid
        is_valid = (
            overall_score >= 0.5 and
            len(issues) <= 3 and
            metrics.word_count >= self.min_word_count and
            metrics.word_count <= self.max_word_count
        )
        
        return ValidationResult(
            is_valid=is_valid,
            score=overall_score,
            issues=issues,
            suggestions=suggestions,
            metrics={
                'word_count': metrics.word_count,
                'char_count': metrics.char_count,
                'complexity_score': metrics.complexity_score,
                'clarity_score': metrics.clarity_score,
                'specificity_score': metrics.specificity_score,
                'answerability_score': metrics.answerability_score,
                'hallucination_potential': metrics.hallucination_potential,
                'domain_relevance': metrics.domain_relevance
            }
        )
    
    async def validate_batch(
        self,
        prompts: List[Prompt],
        source_chunks: List[DocumentChunk],
        use_llm_validation: bool = True
    ) -> List[ValidationResult]:
        """Validate multiple prompts in batch."""
        
        if len(prompts) != len(source_chunks):
            raise ValueError("Number of prompts must match number of source chunks")
        
        # Process in parallel for efficiency
        tasks = [
            self.validate_prompt(prompt, chunk, use_llm_validation)
            for prompt, chunk in zip(prompts, source_chunks)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        validated_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Validation failed for prompt {i}: {result}")
                # Create default validation result for failed cases
                validated_results.append(ValidationResult(
                    is_valid=False,
                    score=0.0,
                    issues=[f"Validation error: {result}"],
                    suggestions=["Review prompt manually"],
                    metrics={}
                ))
            else:
                validated_results.append(result)
        
        return validated_results
    
    def filter_valid_prompts(
        self,
        prompts: List[Prompt],
        validation_results: List[ValidationResult],
        min_score: float = 0.6
    ) -> List[Prompt]:
        """Filter prompts based on validation results."""
        
        if len(prompts) != len(validation_results):
            raise ValueError("Number of prompts must match number of validation results")
        
        valid_prompts = []
        for prompt, result in zip(prompts, validation_results):
            if result.is_valid and result.score >= min_score:
                valid_prompts.append(prompt)
        
        self.logger.info(f"Filtered {len(valid_prompts)}/{len(prompts)} prompts (min_score={min_score})")
        return valid_prompts
    
    def get_quality_summary(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Get summary statistics for validation results."""
        
        if not validation_results:
            return {}
        
        valid_count = sum(1 for r in validation_results if r.is_valid)
        scores = [r.score for r in validation_results]
        quality_levels = [r.quality_level for r in validation_results]
        
        return {
            'total_prompts': len(validation_results),
            'valid_prompts': valid_count,
            'validity_rate': valid_count / len(validation_results),
            'average_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'quality_distribution': {
                'high': sum(1 for q in quality_levels if q == 'high'),
                'medium': sum(1 for q in quality_levels if q == 'medium'),
                'low': sum(1 for q in quality_levels if q == 'low')
            },
            'common_issues': self._get_common_issues(validation_results)
        }
    
    def _calculate_metrics(self, prompt: Prompt, source_chunk: DocumentChunk) -> PromptMetrics:
        """Calculate metrics for a prompt."""
        
        text = prompt.text.lower()
        
        # Basic counts
        word_count = len(prompt.text.split())
        char_count = len(prompt.text)
        
        # Complexity score (based on sentence structure, question types)
        complexity_indicators = [
            'why', 'how', 'what if', 'compare', 'analyze', 'evaluate',
            'relationship', 'implication', 'consequence', 'effect'
        ]
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in text)
        complexity_score = min(complexity_score / 3, 1.0)  # Normalize to 0-1
        
        # Clarity score (based on question words and structure)
        clarity_indicators = ['what', 'how', 'when', 'where', 'why', 'which', 'who']
        has_question_word = any(word in text for word in clarity_indicators)
        ends_with_question = prompt.text.strip().endswith('?')
        clarity_score = 0.3
        if has_question_word:
            clarity_score += 0.4
        if ends_with_question:
            clarity_score += 0.3
        
        # Specificity score (based on specific terms, numbers, proper nouns)
        specificity_indicators = [
            r'\b\d+\b',  # Numbers
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Proper nouns
            r'\bspecific\b', r'\bexact\b', r'\bprecise\b'
        ]
        specificity_score = 0.2
        for pattern in specificity_indicators:
            if re.search(pattern, prompt.text):
                specificity_score += 0.2
        specificity_score = min(specificity_score, 1.0)
        
        # Answerability score (can this be answered from the source?)
        answerability_score = self._calculate_answerability(prompt, source_chunk)
        
        # Hallucination potential (likelihood to cause hallucinations)
        hallucination_potential = self._calculate_hallucination_potential(prompt, source_chunk)
        
        # Domain relevance (military/defense relevance)
        domain_relevance = self._calculate_domain_relevance(prompt)
        
        return PromptMetrics(
            word_count=word_count,
            char_count=char_count,
            complexity_score=complexity_score,
            clarity_score=clarity_score,
            specificity_score=specificity_score,
            answerability_score=answerability_score,
            hallucination_potential=hallucination_potential,
            domain_relevance=domain_relevance
        )
    
    def _calculate_answerability(self, prompt: Prompt, source_chunk: DocumentChunk) -> float:
        """Calculate how well the prompt can be answered from the source."""
        
        prompt_words = set(prompt.text.lower().split())
        source_words = set(source_chunk.content.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        prompt_words -= common_words
        source_words -= common_words
        
        if not prompt_words:
            return 0.0
        
        # Calculate overlap
        overlap = len(prompt_words & source_words)
        overlap_ratio = overlap / len(prompt_words)
        
        return min(overlap_ratio * 1.5, 1.0)  # Boost slightly and cap at 1.0
    
    def _calculate_hallucination_potential(self, prompt: Prompt, source_chunk: DocumentChunk) -> float:
        """Calculate likelihood that this prompt will cause hallucinations."""
        
        potential_score = 0.3  # Base score
        
        # Higher potential if asking for specific details not in source
        if any(word in prompt.text.lower() for word in ['specific', 'exact', 'precise', 'detailed']):
            potential_score += 0.2
        
        # Higher potential if asking for numbers/statistics
        if re.search(r'\b(how many|number|statistics|percentage|amount)\b', prompt.text.lower()):
            potential_score += 0.2
        
        # Higher potential if asking about topics not covered in source
        prompt_topics = self._extract_topics(prompt.text)
        source_topics = self._extract_topics(source_chunk.content)
        
        if prompt_topics and not (prompt_topics & source_topics):
            potential_score += 0.3
        
        return min(potential_score, 1.0)
    
    def _calculate_domain_relevance(self, prompt: Prompt) -> float:
        """Calculate relevance to military/defense domain."""
        
        text = prompt.text.lower()
        relevance_score = 0.0
        
        for category, keywords in self.domain_keywords.items():
            category_matches = sum(1 for keyword in keywords if keyword in text)
            if category_matches > 0:
                relevance_score += min(category_matches / len(keywords), 0.2)
        
        return min(relevance_score, 1.0)
    
    def _extract_topics(self, text: str) -> set:
        """Extract key topics from text."""
        words = text.lower().split()
        topics = set()
        
        # Extract nouns and military-specific terms
        for word in words:
            if len(word) > 3 and word.isalpha():
                topics.add(word)
        
        return topics
    
    def _validate_with_rules(self, prompt: Prompt, metrics: PromptMetrics) -> Tuple[List[str], List[str]]:
        """Perform rule-based validation."""
        
        issues = []
        suggestions = []
        
        # Check for red flags
        for flag_pattern in self.red_flags:
            if re.search(flag_pattern, prompt.text, re.IGNORECASE):
                issues.append(f"Contains problematic pattern: {flag_pattern}")
        
        # Check length
        if metrics.word_count < self.min_word_count:
            issues.append(f"Too short ({metrics.word_count} words)")
            suggestions.append("Add more detail to the question")
        elif metrics.word_count > self.max_word_count:
            issues.append(f"Too long ({metrics.word_count} words)")
            suggestions.append("Make the question more concise")
        
        # Check clarity
        if metrics.clarity_score < self.min_clarity_score:
            issues.append("Low clarity score")
            suggestions.append("Use clear question words (what, how, why, etc.)")
        
        # Check if it's a proper question
        if not prompt.text.strip().endswith('?'):
            issues.append("Does not end with question mark")
            suggestions.append("Ensure the prompt is phrased as a question")
        
        # Check domain relevance
        if metrics.domain_relevance < 0.3:
            issues.append("Low relevance to military/defense domain")
            suggestions.append("Include more domain-specific terminology")
        
        # Check specificity
        if metrics.specificity_score < 0.3:
            issues.append("Too vague or general")
            suggestions.append("Ask for more specific information")
        
        return issues, suggestions
    
    async def _validate_with_llm(self, prompt: Prompt, source_chunk: DocumentChunk) -> Dict[str, Any]:
        """Use LLM to validate prompt quality."""
        
        system_prompt = """You are an expert at evaluating questions for AI testing. 
Rate the quality of the given question for testing AI hallucinations.

Consider:
1. Clarity and grammatical correctness
2. Specificity and precision  
3. Likelihood to elicit hallucinations
4. Appropriateness for military/defense domain
5. Whether it can be answered from the source text

Provide a score from 0.0 to 1.0 and list any issues."""
        
        user_prompt = f"""Rate this question for AI hallucination testing:

Question: {prompt.text}

Source context: {source_chunk.content[:500]}...

Provide your response in this format:
Score: [0.0-1.0]
Issues: [list any problems]
Suggestions: [list improvements]"""
        
        try:
            result = await self.llm_provider.generate(
                prompt=user_prompt,
                params=GenerationParameters(
                    temperature=0.3,  # Low temperature for consistent evaluation
                    max_tokens=200
                ),
                system_prompt=system_prompt
            )
            
            # Parse the response
            response_text = result.text
            score_match = re.search(r'Score:\s*([0-9.]+)', response_text)
            score = float(score_match.group(1)) if score_match else 0.5
            
            issues_match = re.search(r'Issues:\s*(.+?)(?=Suggestions:|$)', response_text, re.DOTALL)
            issues = []
            if issues_match:
                issues_text = issues_match.group(1).strip()
                if issues_text and issues_text.lower() != 'none':
                    issues = [issue.strip() for issue in issues_text.split('\n') if issue.strip()]
            
            suggestions_match = re.search(r'Suggestions:\s*(.+)', response_text, re.DOTALL)
            suggestions = []
            if suggestions_match:
                suggestions_text = suggestions_match.group(1).strip()
                if suggestions_text and suggestions_text.lower() != 'none':
                    suggestions = [sugg.strip() for sugg in suggestions_text.split('\n') if sugg.strip()]
            
            return {
                'score': max(0.0, min(1.0, score)),  # Clamp to valid range
                'issues': issues,
                'suggestions': suggestions
            }
        
        except Exception as e:
            self.logger.error(f"LLM validation error: {e}")
            return {'score': 0.5, 'issues': [], 'suggestions': []}
    
    def _calculate_rule_score(self, metrics: PromptMetrics, issues: List[str]) -> float:
        """Calculate score based on rule-based metrics."""
        
        # Start with average of key metrics
        score = (
            metrics.clarity_score * 0.25 +
            metrics.complexity_score * 0.2 +
            metrics.specificity_score * 0.2 +
            metrics.domain_relevance * 0.2 +
            metrics.hallucination_potential * 0.15
        )
        
        # Penalty for issues
        issue_penalty = len(issues) * 0.1
        score = max(0.0, score - issue_penalty)
        
        return score
    
    def _get_common_issues(self, validation_results: List[ValidationResult]) -> List[Tuple[str, int]]:
        """Get most common issues across all validation results."""
        
        issue_counts = {}
        for result in validation_results:
            for issue in result.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Return top 5 most common issues
        return sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]