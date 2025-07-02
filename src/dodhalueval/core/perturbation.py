"""Advanced prompt perturbation strategies for DoDHaluEval.

This module implements sophisticated perturbation techniques inspired by
hall_prompt_generator and adapted for military/defense domain evaluation.
"""

import re
import random
from typing import List, Dict, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from dodhalueval.models.schemas import Prompt, DocumentChunk
from dodhalueval.utils.logger import get_logger


@dataclass
class PerturbationRule:
    """Represents a perturbation rule with metadata."""
    
    name: str
    description: str
    difficulty_increase: str  # 'low', 'medium', 'high'
    hallucination_types: List[str]  # Types of hallucinations this rule targets
    applicability_score: float = 0.0  # How well this rule applies to the current prompt


class KnowledgeBase:
    """Simple knowledge base for entity relationships and substitutions."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Military-specific knowledge mappings
        self.entity_relationships = {
            'ranks': {
                'army': ['Private', 'Corporal', 'Sergeant', 'Lieutenant', 'Captain', 'Major', 'Colonel', 'General'],
                'navy': ['Seaman', 'Petty Officer', 'Chief', 'Ensign', 'Lieutenant', 'Commander', 'Captain', 'Admiral'],
                'air_force': ['Airman', 'Senior Airman', 'Staff Sergeant', 'Lieutenant', 'Captain', 'Major', 'Colonel', 'General'],
                'marines': ['Private', 'Corporal', 'Sergeant', 'Lieutenant', 'Captain', 'Major', 'Colonel', 'General']
            },
            'equipment': {
                'vehicles': ['M1A1 Abrams', 'M2 Bradley', 'HMMWV', 'MRAP', 'LAV-25'],
                'aircraft': ['F-16', 'F-22', 'F-35', 'A-10', 'C-130', 'UH-60', 'AH-64'],
                'weapons': ['M4 Carbine', 'M249 SAW', 'M240B', 'M107', 'AT4', 'Javelin'],
                'naval': ['Arleigh Burke', 'Virginia Class', 'Nimitz Class', 'LCS', 'LPD']
            },
            'locations': {
                'bases': ['Fort Bragg', 'Camp Pendleton', 'Nellis AFB', 'Norfolk Naval Base', 'Pearl Harbor'],
                'regions': ['CENTCOM', 'EUCOM', 'PACOM', 'NORTHCOM', 'SOUTHCOM', 'AFRICOM'],
                'theaters': ['Pacific Theater', 'European Theater', 'Middle East', 'Indo-Pacific']
            },
            'operations': {
                'historical': ['Desert Storm', 'Enduring Freedom', 'Iraqi Freedom', 'Inherent Resolve'],
                'types': ['Combat', 'Peacekeeping', 'Humanitarian', 'Training', 'Reconnaissance'],
                'phases': ['Planning', 'Deployment', 'Execution', 'Transition', 'Redeployment']
            }
        }
        
        # Temporal knowledge
        self.time_periods = {
            'conflicts': {
                'World War II': (1939, 1945),
                'Korean War': (1950, 1953),
                'Vietnam War': (1955, 1975),
                'Gulf War': (1990, 1991),
                'Iraq War': (2003, 2011),
                'Afghanistan War': (2001, 2021)
            },
            'decades': ['1990s', '2000s', '2010s', '2020s'],
            'relative_times': ['recently', 'previously', 'currently', 'historically']
        }
        
        # Numerical patterns
        self.number_patterns = {
            'military_units': [100, 200, 300, 500, 1000, 2000, 5000, 10000],
            'distances': [5, 10, 25, 50, 100, 250, 500, 1000],  # km/miles
            'time_durations': [30, 60, 90, 120, 180, 360, 720],  # minutes/days
            'percentages': [10, 25, 33, 50, 66, 75, 80, 90, 95]
        }
    
    def get_similar_entities(self, entity: str, entity_type: str, num_alternatives: int = 3) -> List[str]:
        """Get similar entities for substitution."""
        alternatives = []
        
        # Search in knowledge base
        for category, subcategories in self.entity_relationships.items():
            if isinstance(subcategories, dict):
                for subcat, items in subcategories.items():
                    if entity in items:
                        # Return other items from same category
                        alternatives.extend([item for item in items if item != entity])
            elif isinstance(subcategories, list) and entity in subcategories:
                alternatives.extend([item for item in subcategories if item != entity])
        
        # Return random sample
        return random.sample(alternatives, min(num_alternatives, len(alternatives)))
    
    def get_conflicting_time_period(self, original_period: str) -> Optional[str]:
        """Get a conflicting time period for temporal confusion."""
        for conflict, (start, end) in self.time_periods['conflicts'].items():
            if original_period.lower() in conflict.lower():
                # Return a different conflict period
                other_conflicts = [c for c in self.time_periods['conflicts'].keys() if c != conflict]
                return random.choice(other_conflicts) if other_conflicts else None
        
        return None
    
    def get_authority_alternatives(self, authority: str) -> List[str]:
        """Get alternative authorities for authority confusion."""
        military_authorities = [
            'Department of Defense', 'Joint Chiefs of Staff', 'CENTCOM', 'NATO',
            'Army Regulation', 'Navy Instruction', 'Air Force Instruction',
            'Marine Corps Order', 'Joint Publication', 'DoD Directive'
        ]
        
        return [auth for auth in military_authorities if auth.lower() != authority.lower()]


class PromptPerturbator:
    """Advanced prompt perturbation engine."""
    
    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.logger = get_logger(__name__)
        
        # Load perturbation strategies
        self.strategies = self._initialize_strategies()
    
    def _initialize_strategies(self) -> Dict[str, PerturbationRule]:
        """Initialize all perturbation strategies."""
        return {
            'entity_substitution': PerturbationRule(
                name='entity_substitution',
                description='Replace entities with similar but incorrect ones',
                difficulty_increase='medium',
                hallucination_types=['factual']
            ),
            'negation_injection': PerturbationRule(
                name='negation_injection',
                description='Inject negations to test logical consistency',
                difficulty_increase='high',
                hallucination_types=['logical']
            ),
            'quantifier_manipulation': PerturbationRule(
                name='quantifier_manipulation',
                description='Modify quantifiers (all, some, none, most)',
                difficulty_increase='high',
                hallucination_types=['logical']
            ),
            'temporal_confusion': PerturbationRule(
                name='temporal_confusion',
                description='Introduce temporal inconsistencies',
                difficulty_increase='high',
                hallucination_types=['factual', 'logical']
            ),
            'causal_reversal': PerturbationRule(
                name='causal_reversal',
                description='Reverse cause-effect relationships',
                difficulty_increase='high',
                hallucination_types=['logical']
            ),
            'authority_confusion': PerturbationRule(
                name='authority_confusion',
                description='Mix up authoritative sources',
                difficulty_increase='medium',
                hallucination_types=['factual', 'context']
            ),
            'multi_hop_reasoning': PerturbationRule(
                name='multi_hop_reasoning',
                description='Require multi-step logical inference',
                difficulty_increase='high',
                hallucination_types=['logical']
            ),
            'numerical_manipulation': PerturbationRule(
                name='numerical_manipulation',
                description='Modify numbers in subtle ways',
                difficulty_increase='medium',
                hallucination_types=['factual']
            ),
            'scope_expansion': PerturbationRule(
                name='scope_expansion',
                description='Expand question scope beyond source material',
                difficulty_increase='medium',
                hallucination_types=['factual', 'context']
            ),
            'conditional_complexity': PerturbationRule(
                name='conditional_complexity',
                description='Add complex conditional statements',
                difficulty_increase='high',
                hallucination_types=['logical']
            )
        }
    
    def perturb(self, prompt: Prompt, strategy_name: str, source_chunk: DocumentChunk) -> List[Prompt]:
        """Apply a specific perturbation strategy."""
        if strategy_name not in self.strategies:
            self.logger.warning(f"Unknown perturbation strategy: {strategy_name}")
            return [prompt]
        
        strategy = self.strategies[strategy_name]
        
        try:
            if strategy_name == 'entity_substitution':
                return self.entity_substitution(prompt, source_chunk)
            elif strategy_name == 'negation_injection':
                return self.negation_injection(prompt)
            elif strategy_name == 'quantifier_manipulation':
                return self.quantifier_manipulation(prompt)
            elif strategy_name == 'temporal_confusion':
                return self.temporal_confusion(prompt)
            elif strategy_name == 'causal_reversal':
                return self.causal_reversal(prompt)
            elif strategy_name == 'authority_confusion':
                return self.authority_confusion(prompt)
            elif strategy_name == 'multi_hop_reasoning':
                return self.multi_hop_reasoning(prompt, source_chunk)
            elif strategy_name == 'numerical_manipulation':
                return self.numerical_manipulation(prompt)
            elif strategy_name == 'scope_expansion':
                return self.scope_expansion(prompt, source_chunk)
            elif strategy_name == 'conditional_complexity':
                return self.conditional_complexity(prompt)
            else:
                return [prompt]
        
        except Exception as e:
            self.logger.error(f"Error applying {strategy_name}: {e}")
            return [prompt]
    
    def entity_substitution(self, prompt: Prompt, source_chunk: DocumentChunk) -> List[Prompt]:
        """Replace entities with similar but incorrect ones."""
        variations = []
        
        # Extract entities from prompt
        entities_found = []
        
        # Look for military equipment
        equipment_pattern = r'\b(?:M\d+|F-\d+|AH-\d+|UH-\d+|C-\d+|[A-Z]+-\d+)\b'
        entities_found.extend(re.findall(equipment_pattern, prompt.text))
        
        # Look for ranks
        rank_pattern = r'\b(?:General|Colonel|Major|Captain|Lieutenant|Sergeant|Corporal|Admiral|Commander)\b'
        entities_found.extend(re.findall(rank_pattern, prompt.text))
        
        # Look for bases/locations
        base_pattern = r'\b(?:Fort|Camp|Base|AFB)\s+[A-Z][a-z]+\b'
        entities_found.extend(re.findall(base_pattern, prompt.text))
        
        # Create variations by substituting each entity
        for entity in entities_found:
            alternatives = self.knowledge_base.get_similar_entities(entity, 'military', 2)
            
            for alt in alternatives:
                new_text = prompt.text.replace(entity, alt, 1)
                if new_text != prompt.text:
                    variation = Prompt(
                        text=new_text,
                        source_document_id=prompt.source_document_id,
                        source_chunk_id=prompt.source_chunk_id,
                        expected_answer=prompt.expected_answer,
                        hallucination_type='factual',
                        generation_strategy='perturbation_entity_substitution',
                        difficulty_level='hard',
                        metadata={
                            'original_prompt_id': prompt.id,
                            'perturbation_type': 'entity_substitution',
                            'substituted_entity': entity,
                            'replacement_entity': alt
                        }
                    )
                    variations.append(variation)
        
        return variations[:3]  # Limit variations
    
    def negation_injection(self, prompt: Prompt) -> List[Prompt]:
        """Inject negations to test logical consistency."""
        variations = []
        
        # Patterns for negation injection
        negation_patterns = [
            (r'\bis\s+', 'is not '),
            (r'\bcan\s+', 'cannot '),
            (r'\bshould\s+', 'should not '),
            (r'\bmust\s+', 'must not '),
            (r'\bwill\s+', 'will not '),
            (r'\bhas\s+', 'has not '),
            (r'\bhave\s+', 'have not '),
            (r'\bdoes\s+', 'does not '),
            (r'\bdo\s+', 'do not ')
        ]
        
        for pattern, replacement in negation_patterns:
            if re.search(pattern, prompt.text, re.IGNORECASE):
                new_text = re.sub(pattern, replacement, prompt.text, count=1, flags=re.IGNORECASE)
                
                variation = Prompt(
                    text=new_text,
                    source_document_id=prompt.source_document_id,
                    source_chunk_id=prompt.source_chunk_id,
                    expected_answer=prompt.expected_answer,
                    hallucination_type='logical',
                    generation_strategy='perturbation_negation_injection',
                    difficulty_level='hard',
                    metadata={
                        'original_prompt_id': prompt.id,
                        'perturbation_type': 'negation_injection',
                        'negation_pattern': pattern,
                        'replacement': replacement
                    }
                )
                variations.append(variation)
                break  # Only apply one negation
        
        return variations
    
    def quantifier_manipulation(self, prompt: Prompt) -> List[Prompt]:
        """Modify quantifiers to test logical understanding."""
        variations = []
        
        # Quantifier replacements
        quantifier_maps = [
            ('all', 'some'),
            ('some', 'all'),
            ('most', 'few'),
            ('many', 'few'),
            ('always', 'sometimes'),
            ('never', 'sometimes'),
            ('every', 'some'),
            ('each', 'some')
        ]
        
        for original, replacement in quantifier_maps:
            pattern = r'\b' + re.escape(original) + r'\b'
            if re.search(pattern, prompt.text, re.IGNORECASE):
                new_text = re.sub(pattern, replacement, prompt.text, count=1, flags=re.IGNORECASE)
                
                variation = Prompt(
                    text=new_text,
                    source_document_id=prompt.source_document_id,
                    source_chunk_id=prompt.source_chunk_id,
                    expected_answer=prompt.expected_answer,
                    hallucination_type='logical',
                    generation_strategy='perturbation_quantifier_manipulation',
                    difficulty_level='hard',
                    metadata={
                        'original_prompt_id': prompt.id,
                        'perturbation_type': 'quantifier_manipulation',
                        'original_quantifier': original,
                        'replacement_quantifier': replacement
                    }
                )
                variations.append(variation)
        
        return variations[:2]  # Limit variations
    
    def temporal_confusion(self, prompt: Prompt) -> List[Prompt]:
        """Introduce temporal inconsistencies."""
        variations = []
        
        # Find temporal references
        temporal_patterns = [
            r'\b(during|in|after|before|since)\s+([A-Z][a-zA-Z\s]+(?:War|Operation|Conflict))\b',
            r'\b(in|during)\s+(\d{4}s?|\d{4})\b',
            r'\b(recently|currently|previously|historically)\b'
        ]
        
        for pattern in temporal_patterns:
            matches = re.finditer(pattern, prompt.text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    temporal_ref = match.group(2)
                    conflicting_period = self.knowledge_base.get_conflicting_time_period(temporal_ref)
                    
                    if conflicting_period:
                        new_text = prompt.text.replace(match.group(0), 
                                                     match.group(0).replace(temporal_ref, conflicting_period))
                        
                        variation = Prompt(
                            text=new_text,
                            source_document_id=prompt.source_document_id,
                            source_chunk_id=prompt.source_chunk_id,
                            expected_answer=prompt.expected_answer,
                            hallucination_type='factual',
                            generation_strategy='perturbation_temporal_confusion',
                            difficulty_level='hard',
                            metadata={
                                'original_prompt_id': prompt.id,
                                'perturbation_type': 'temporal_confusion',
                                'original_period': temporal_ref,
                                'conflicting_period': conflicting_period
                            }
                        )
                        variations.append(variation)
        
        return variations[:2]
    
    def causal_reversal(self, prompt: Prompt) -> List[Prompt]:
        """Reverse cause-effect relationships."""
        variations = []
        
        # Look for causal language
        causal_patterns = [
            (r'because\s+(.+?),\s*(.+)', r'because \2, \1'),
            (r'due to\s+(.+?),\s*(.+)', r'due to \2, \1'),
            (r'as a result of\s+(.+?),\s*(.+)', r'as a result of \2, \1'),
            (r'(.+?)\s+leads to\s+(.+)', r'\2 leads to \1'),
            (r'(.+?)\s+causes?\s+(.+)', r'\2 causes \1')
        ]
        
        for pattern, replacement in causal_patterns:
            match = re.search(pattern, prompt.text, re.IGNORECASE)
            if match:
                new_text = re.sub(pattern, replacement, prompt.text, count=1, flags=re.IGNORECASE)
                
                variation = Prompt(
                    text=new_text,
                    source_document_id=prompt.source_document_id,
                    source_chunk_id=prompt.source_chunk_id,
                    expected_answer=prompt.expected_answer,
                    hallucination_type='logical',
                    generation_strategy='perturbation_causal_reversal',
                    difficulty_level='hard',
                    metadata={
                        'original_prompt_id': prompt.id,
                        'perturbation_type': 'causal_reversal',
                        'causal_pattern': pattern
                    }
                )
                variations.append(variation)
                break
        
        return variations
    
    def authority_confusion(self, prompt: Prompt) -> List[Prompt]:
        """Mix up authoritative sources."""
        variations = []
        
        # Look for authority references
        authority_patterns = [
            r'\b(according to|per|as stated in|as per)\s+([A-Z][A-Za-z\s]+(?:Regulation|Instruction|Manual|Directive|Publication))\b',
            r'\b(DoD|Army|Navy|Air Force|Marine|Joint)\s+(Regulation|Instruction|Manual|Directive|Publication)\b'
        ]
        
        for pattern in authority_patterns:
            matches = re.finditer(pattern, prompt.text, re.IGNORECASE)
            for match in matches:
                original_authority = match.group(0)
                alternatives = self.knowledge_base.get_authority_alternatives(original_authority)
                
                if alternatives:
                    new_authority = random.choice(alternatives)
                    new_text = prompt.text.replace(original_authority, new_authority, 1)
                    
                    variation = Prompt(
                        text=new_text,
                        source_document_id=prompt.source_document_id,
                        source_chunk_id=prompt.source_chunk_id,
                        expected_answer=prompt.expected_answer,
                        hallucination_type='factual',
                        generation_strategy='perturbation_authority_confusion',
                        difficulty_level='medium',
                        metadata={
                            'original_prompt_id': prompt.id,
                            'perturbation_type': 'authority_confusion',
                            'original_authority': original_authority,
                            'replacement_authority': new_authority
                        }
                    )
                    variations.append(variation)
        
        return variations[:2]
    
    def multi_hop_reasoning(self, prompt: Prompt, source_chunk: DocumentChunk) -> List[Prompt]:
        """Create multi-step reasoning challenges."""
        variations = []
        
        # Transform simple questions into multi-hop ones
        multi_hop_templates = [
            "If {original_question}, then what would be the implications for {related_concept}?",
            "Given that {original_question}, how would this affect {related_concept} and subsequently {another_concept}?",
            "Assuming {original_question}, what chain of events would lead to {outcome}?",
            "If {original_question} is true, and {additional_condition}, what can we conclude about {target}?"
        ]
        
        # Extract key concepts from the source chunk for related concepts
        key_concepts = ['operations', 'training', 'equipment', 'personnel', 'mission', 'security']
        
        for template in multi_hop_templates[:2]:  # Limit to 2 variations
            related_concept = random.choice(key_concepts)
            another_concept = random.choice([c for c in key_concepts if c != related_concept])
            
            try:
                new_text = template.format(
                    original_question=prompt.text.rstrip('?'),
                    related_concept=related_concept,
                    another_concept=another_concept,
                    additional_condition=f"assuming {related_concept} constraints apply",
                    outcome=f"{related_concept} effectiveness",
                    target=f"{another_concept} requirements"
                )
                
                variation = Prompt(
                    text=new_text,
                    source_document_id=prompt.source_document_id,
                    source_chunk_id=prompt.source_chunk_id,
                    expected_answer=prompt.expected_answer,
                    hallucination_type='logical',
                    generation_strategy='perturbation_multi_hop_reasoning',
                    difficulty_level='hard',
                    metadata={
                        'original_prompt_id': prompt.id,
                        'perturbation_type': 'multi_hop_reasoning',
                        'template_used': template,
                        'related_concepts': [related_concept, another_concept]
                    }
                )
                variations.append(variation)
            
            except Exception as e:
                self.logger.warning(f"Failed to create multi-hop variation: {e}")
        
        return variations
    
    def numerical_manipulation(self, prompt: Prompt) -> List[Prompt]:
        """Modify numbers in subtle ways."""
        variations = []
        
        # Find numbers in the prompt
        number_pattern = r'\b(\d+(?:\.\d+)?)\b'
        numbers = re.findall(number_pattern, prompt.text)
        
        for number_str in numbers:
            try:
                original_num = float(number_str)
                
                # Generate plausible but incorrect numbers
                manipulated_numbers = []
                
                if original_num == int(original_num):  # Integer
                    original_int = int(original_num)
                    manipulated_numbers = [
                        original_int + 1,
                        original_int - 1,
                        original_int * 2,
                        max(1, original_int // 2),
                        original_int + 10,
                        max(0, original_int - 10)
                    ]
                else:  # Float
                    manipulated_numbers = [
                        original_num + 0.1,
                        original_num - 0.1,
                        original_num * 1.1,
                        original_num * 0.9
                    ]
                
                # Create variation with first manipulation
                if manipulated_numbers:
                    new_num = manipulated_numbers[0]
                    new_text = prompt.text.replace(number_str, str(new_num), 1)
                    
                    variation = Prompt(
                        text=new_text,
                        source_document_id=prompt.source_document_id,
                        source_chunk_id=prompt.source_chunk_id,
                        expected_answer=prompt.expected_answer,
                        hallucination_type='factual',
                        generation_strategy='perturbation_numerical_manipulation',
                        difficulty_level='medium',
                        metadata={
                            'original_prompt_id': prompt.id,
                            'perturbation_type': 'numerical_manipulation',
                            'original_number': number_str,
                            'manipulated_number': str(new_num)
                        }
                    )
                    variations.append(variation)
            
            except ValueError:
                continue
        
        return variations[:2]  # Limit variations
    
    def scope_expansion(self, prompt: Prompt, source_chunk: DocumentChunk) -> List[Prompt]:
        """Expand question scope beyond source material."""
        variations = []
        
        expansion_patterns = [
            "Beyond what's mentioned in the document, {original_question}",
            "{original_question} Also, how does this compare to similar situations in other conflicts?",
            "{original_question} What are the broader implications for military doctrine?",
            "In addition to the documented procedures, {original_question}"
        ]
        
        for pattern in expansion_patterns[:2]:
            new_text = pattern.format(original_question=prompt.text.lower())
            
            variation = Prompt(
                text=new_text,
                source_document_id=prompt.source_document_id,
                source_chunk_id=prompt.source_chunk_id,
                expected_answer=prompt.expected_answer,
                hallucination_type='context',
                generation_strategy='perturbation_scope_expansion',
                difficulty_level='medium',
                metadata={
                    'original_prompt_id': prompt.id,
                    'perturbation_type': 'scope_expansion',
                    'expansion_pattern': pattern
                }
            )
            variations.append(variation)
        
        return variations
    
    def conditional_complexity(self, prompt: Prompt) -> List[Prompt]:
        """Add complex conditional statements."""
        variations = []
        
        conditional_templates = [
            "If and only if {condition1}, {original_question}",
            "Given that {condition1} and {condition2}, {original_question}",
            "Under the assumption that {condition1} but not {condition2}, {original_question}",
            "In the case where {condition1} implies {condition2}, {original_question}"
        ]
        
        conditions = [
            "combat conditions apply",
            "peacetime protocols are in effect",
            "emergency procedures are activated",
            "joint operations are conducted",
            "classified materials are involved",
            "international law applies"
        ]
        
        for template in conditional_templates[:2]:
            condition1 = random.choice(conditions)
            condition2 = random.choice([c for c in conditions if c != condition1])
            
            new_text = template.format(
                original_question=prompt.text.lower(),
                condition1=condition1,
                condition2=condition2
            )
            
            variation = Prompt(
                text=new_text,
                source_document_id=prompt.source_document_id,
                source_chunk_id=prompt.source_chunk_id,
                expected_answer=prompt.expected_answer,
                hallucination_type='logical',
                generation_strategy='perturbation_conditional_complexity',
                difficulty_level='hard',
                metadata={
                    'original_prompt_id': prompt.id,
                    'perturbation_type': 'conditional_complexity',
                    'conditions_added': [condition1, condition2],
                    'template_used': template
                }
            )
            variations.append(variation)
        
        return variations
    
    def apply_random_strategy(self, prompt: Prompt, source_chunk: DocumentChunk) -> List[Prompt]:
        """Apply a random perturbation strategy."""
        strategy_name = random.choice(list(self.strategies.keys()))
        return self.perturb(prompt, strategy_name, source_chunk)
    
    def apply_multiple_strategies(
        self,
        prompt: Prompt,
        source_chunk: DocumentChunk,
        max_strategies: int = 3
    ) -> List[Prompt]:
        """Apply multiple perturbation strategies."""
        all_variations = [prompt]  # Include original
        
        # Apply random strategies
        applied_strategies = set()
        for _ in range(max_strategies):
            available_strategies = [s for s in self.strategies.keys() if s not in applied_strategies]
            if not available_strategies:
                break
            
            strategy = random.choice(available_strategies)
            applied_strategies.add(strategy)
            
            variations = self.perturb(prompt, strategy, source_chunk)
            all_variations.extend(variations)
        
        return all_variations