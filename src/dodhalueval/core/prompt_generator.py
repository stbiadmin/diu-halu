"""Prompt generation engine for DoDHaluEval.

This module implements various strategies for generating prompts that are likely
to elicit hallucinations from language models when applied to DoD documents.
"""

import random
import re
from typing import List, Dict, Optional, Set, Any
from pathlib import Path
import yaml
from datetime import datetime

from dodhalueval.models.schemas import Prompt, DocumentChunk
from dodhalueval.models.config import PromptGenerationConfig
from dodhalueval.utils.logger import get_logger


class PromptTemplate:
    """Represents a prompt template with placeholders."""
    
    def __init__(self, template: str, category: str, hallucination_type: str):
        self.template = template
        self.category = category
        self.hallucination_type = hallucination_type
        self.placeholders = self._extract_placeholders()
    
    def _extract_placeholders(self) -> Set[str]:
        """Extract placeholder variables from template."""
        return set(re.findall(r'\{(\w+)\}', self.template))
    
    def fill(self, **kwargs) -> str:
        """Fill template with provided values."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing placeholder value: {e}")
    
    def can_fill(self, entities: Dict[str, List[str]]) -> bool:
        """Check if template can be filled with available entities."""
        return all(placeholder in entities for placeholder in self.placeholders)


class EntityExtractor:
    """Extracts military and DoD-specific entities from text."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # DoD-specific entity patterns
        self.patterns = {
            'military_unit': [
                r'\b(?:Battalion|Regiment|Brigade|Division|Corps|Army|Fleet|Squadron)\b',
                r'\b\d+(?:st|nd|rd|th)\s+(?:Infantry|Airborne|Armored|Artillery)\b',
                r'\b(?:Special Forces|Rangers|SEALs|Delta Force)\b'
            ],
            'equipment': [
                r'\b(?:M\d+|F-\d+|AH-\d+|UH-\d+|C-\d+)\b',
                r'\b(?:Abrams|Bradley|Humvee|Black Hawk|Apache|Chinook)\b',
                r'\b(?:rifle|tank|helicopter|aircraft|vessel|missile)\b'
            ],
            'procedure': [
                r'\b(?:operation|procedure|protocol|doctrine|tactic)\b',
                r'\b(?:deployment|maneuver|engagement|reconnaissance)\b',
                r'\b(?:briefing|debrief|planning|execution)\b'
            ],
            'concept': [
                r'\b(?:leadership|command|control|communication)\b',
                r'\b(?:strategy|mission|objective|target)\b',
                r'\b(?:security|intelligence|logistics|training)\b'
            ],
            'location': [
                r'\b(?:base|fort|camp|station|facility)\b',
                r'\b(?:theater|zone|sector|area|region)\b',
                r'\b[A-Z][a-z]+\s+(?:Base|Fort|Camp|Station)\b'
            ],
            'rank': [
                r'\b(?:General|Colonel|Major|Captain|Lieutenant|Sergeant)\b',
                r'\b(?:Admiral|Commander|Officer|Enlisted)\b'
            ]
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract all entities from text."""
        entities = {}
        
        for entity_type, patterns in self.patterns.items():
            matches = set()
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.update(found)
            entities[entity_type] = list(matches)
        
        # Extract numbers for quantified prompts
        numbers = re.findall(r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b', text, re.IGNORECASE)
        entities['number'] = list(set(numbers))
        
        # Extract noun phrases as generic concepts
        noun_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities['noun_phrase'] = list(set(noun_phrases))
        
        return entities
    
    def get_similar_entities(self, entity: str, entity_type: str) -> List[str]:
        """Get similar entities for replacement (basic implementation)."""
        # This is a simplified implementation - in practice, you'd use
        # a knowledge base or embedding similarity
        
        similar_map = {
            'military_unit': {
                'Battalion': ['Regiment', 'Brigade', 'Company'],
                'Infantry': ['Airborne', 'Armored', 'Artillery'],
                'Army': ['Navy', 'Air Force', 'Marines']
            },
            'equipment': {
                'tank': ['armored vehicle', 'personnel carrier'],
                'helicopter': ['aircraft', 'drone', 'fighter jet'],
                'rifle': ['weapon', 'firearm', 'carbine']
            },
            'rank': {
                'General': ['Colonel', 'Admiral'],
                'Captain': ['Major', 'Commander'],
                'Sergeant': ['Lieutenant', 'Corporal']
            }
        }
        
        if entity_type in similar_map and entity in similar_map[entity_type]:
            return similar_map[entity_type][entity]
        
        return []


class HeuristicRules:
    """Implements heuristic rules for prompt perturbation."""
    
    def __init__(self, entity_extractor: EntityExtractor):
        self.entity_extractor = entity_extractor
        self.logger = get_logger(__name__)
    
    def entity_substitution(self, prompt: Prompt, entities: Dict[str, List[str]]) -> List[Prompt]:
        """Replace entities with similar ones to create factual errors."""
        variations = []
        
        for entity_type, entity_list in entities.items():
            if not entity_list:
                continue
                
            for entity in entity_list:
                similar_entities = self.entity_extractor.get_similar_entities(entity, entity_type)
                
                for similar in similar_entities:
                    new_text = prompt.text.replace(entity, similar)
                    if new_text != prompt.text:
                        variation = Prompt(
                            text=new_text,
                            source_document_id=prompt.source_document_id,
                            source_chunk_id=prompt.source_chunk_id,
                            expected_answer=prompt.expected_answer,
                            hallucination_type='factual',
                            generation_strategy='entity_substitution',
                            difficulty_level=prompt.difficulty_level,
                            metadata={
                                'original_prompt_id': prompt.id,
                                'substituted_entity': entity,
                                'replacement_entity': similar,
                                'entity_type': entity_type
                            }
                        )
                        variations.append(variation)
        
        return variations
    
    def fact_perturbation(self, prompt: Prompt, chunk: DocumentChunk) -> List[Prompt]:
        """Modify facts to create plausible falsehoods."""
        variations = []
        
        # Number perturbation
        numbers = re.findall(r'\b\d+\b', prompt.text)
        for number in numbers:
            # Slightly modify numbers
            original_num = int(number)
            perturbed_nums = [
                original_num + 1,
                original_num - 1,
                original_num * 2,
                max(1, original_num // 2)
            ]
            
            for new_num in perturbed_nums:
                new_text = prompt.text.replace(number, str(new_num), 1)
                variation = Prompt(
                    text=new_text,
                    source_document_id=prompt.source_document_id,
                    source_chunk_id=prompt.source_chunk_id,
                    expected_answer=prompt.expected_answer,
                    hallucination_type='factual',
                    generation_strategy='fact_perturbation',
                    difficulty_level=prompt.difficulty_level,
                    metadata={
                        'original_prompt_id': prompt.id,
                        'original_number': number,
                        'perturbed_number': str(new_num)
                    }
                )
                variations.append(variation)
        
        return variations
    
    def context_mixing(self, prompt: Prompt, other_chunks: List[DocumentChunk]) -> List[Prompt]:
        """Mix contexts from different sections."""
        variations = []
        
        if not other_chunks:
            return variations
        
        # Simple context mixing - replace part of prompt with content from other chunks
        other_chunk = random.choice(other_chunks)
        
        # Extract a concept from the other chunk
        other_entities = self.entity_extractor.extract_entities(other_chunk.content)
        
        for entity_type, entity_list in other_entities.items():
            if entity_list and entity_type in ['concept', 'procedure', 'equipment']:
                foreign_entity = random.choice(entity_list)
                
                # Try to incorporate the foreign entity into the prompt
                context_patterns = [
                    f"In the context of {foreign_entity}, {prompt.text.lower()}",
                    f"{prompt.text} How does this relate to {foreign_entity}?",
                    f"Considering {foreign_entity}, {prompt.text.lower()}"
                ]
                
                for pattern in context_patterns:
                    variation = Prompt(
                        text=pattern,
                        source_document_id=prompt.source_document_id,
                        source_chunk_id=prompt.source_chunk_id,
                        expected_answer=prompt.expected_answer,
                        hallucination_type='context',
                        generation_strategy='context_mixing',
                        difficulty_level='hard',
                        metadata={
                            'original_prompt_id': prompt.id,
                            'mixed_chunk_id': other_chunk.id,
                            'foreign_entity': foreign_entity,
                            'entity_type': entity_type
                        }
                    )
                    variations.append(variation)
        
        return variations
    
    def negation_injection(self, prompt: Prompt) -> List[Prompt]:
        """Inject negations to test logical reasoning."""
        variations = []
        
        # Simple negation patterns
        negation_patterns = [
            (r'\bis\b', 'is not'),
            (r'\bcan\b', 'cannot'),
            (r'\bshould\b', 'should not'),
            (r'\bmust\b', 'must not'),
            (r'\bwill\b', 'will not')
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
                    generation_strategy='negation_injection',
                    difficulty_level=prompt.difficulty_level,
                    metadata={
                        'original_prompt_id': prompt.id,
                        'negation_pattern': pattern,
                        'replacement': replacement
                    }
                )
                variations.append(variation)
        
        return variations


class PromptGenerator:
    """Main prompt generation engine."""
    
    def __init__(self, config: PromptGenerationConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.entity_extractor = EntityExtractor()
        self.heuristic_rules = HeuristicRules(self.entity_extractor)
        self.templates = self._load_templates()
    
    def _load_templates(self) -> List[PromptTemplate]:
        """Load prompt templates from configuration."""
        templates = []
        
        if self.config.template_file and Path(self.config.template_file).exists():
            try:
                with open(self.config.template_file, 'r') as f:
                    template_data = yaml.safe_load(f)
                
                for category, template_list in template_data.items():
                    for template_str in template_list:
                        # Determine hallucination type based on category
                        if 'factual' in category or 'recall' in category:
                            hallucination_type = 'factual'
                        elif 'reasoning' in category or 'logical' in category:
                            hallucination_type = 'logical'
                        else:
                            hallucination_type = 'context'
                        
                        template = PromptTemplate(template_str, category, hallucination_type)
                        templates.append(template)
            
            except Exception as e:
                self.logger.warning(f"Failed to load templates from {self.config.template_file}: {e}")
        
        # Default templates if none loaded
        if not templates:
            templates = self._get_default_templates()
        
        return templates
    
    def _get_default_templates(self) -> List[PromptTemplate]:
        """Get default prompt templates."""
        default_templates = [
            # Question-Answer templates
            ("According to {source}, what is {concept}?", "question_answer", "factual"),
            ("What does {source} say about {topic}?", "question_answer", "factual"),
            ("In the context of {doctrine}, explain {procedure}", "question_answer", "context"),
            
            # Factual recall templates
            ("What are the {number} key principles of {concept} in {source}?", "factual_recall", "factual"),
            ("List the steps for {procedure} as outlined in {doctrine}", "factual_recall", "factual"),
            ("Name the equipment used in {operation}", "factual_recall", "factual"),
            
            # Reasoning templates
            ("Why does {source} recommend {action} in {situation}?", "reasoning", "logical"),
            ("How does {concept} relate to {other_concept} according to {doctrine}?", "reasoning", "logical"),
            ("What are the consequences of {action} in {context}?", "reasoning", "logical"),
            
            # Procedural templates
            ("Describe the protocol for {procedure}", "procedural", "factual"),
            ("What is the correct sequence for {operation}?", "procedural", "factual"),
            ("How should {equipment} be operated?", "procedural", "factual"),
        ]
        
        return [PromptTemplate(template, category, hallucination_type) 
                for template, category, hallucination_type in default_templates]
    
    def generate_from_chunks(self, chunks: List[DocumentChunk]) -> List[Prompt]:
        """Generate prompts from document chunks using templates."""
        all_prompts = []
        
        for chunk in chunks:
            chunk_prompts = self._generate_for_chunk(chunk, chunks)
            all_prompts.extend(chunk_prompts)
            
            # Respect max prompts limit
            if len(all_prompts) >= self.config.max_prompts_per_document:
                break
        
        # Shuffle and limit
        random.shuffle(all_prompts)
        return all_prompts[:self.config.max_prompts_per_document]
    
    def _generate_for_chunk(self, chunk: DocumentChunk, all_chunks: List[DocumentChunk]) -> List[Prompt]:
        """Generate prompts for a specific chunk."""
        prompts = []
        
        # Extract entities from the chunk
        entities = self.entity_extractor.extract_entities(chunk.content)
        
        # Add some generic entities
        entities['source'] = ['the document', 'the manual', 'the regulation', 'the doctrine']
        entities['doctrine'] = ['the doctrine', 'the manual', 'the regulation']
        entities['situation'] = ['combat', 'training', 'deployment', 'emergency']
        entities['operation'] = ['the operation', 'the mission', 'the exercise']
        entities['action'] = ['this approach', 'this method', 'this procedure']
        entities['context'] = ['military operations', 'field conditions', 'combat situations']
        entities['topic'] = entities.get('concept', ['leadership', 'tactics', 'strategy'])
        entities['other_concept'] = entities.get('concept', ['coordination', 'planning', 'execution'])
        
        # Generate prompts from templates
        for template in self.templates:
            if template.can_fill(entities):
                try:
                    # Generate multiple variations by sampling different entities
                    for _ in range(min(3, len(entities.get(list(template.placeholders)[0], [])))):
                        # Sample entities for this prompt
                        sampled_entities = {}
                        for placeholder in template.placeholders:
                            if placeholder in entities and entities[placeholder]:
                                sampled_entities[placeholder] = random.choice(entities[placeholder])
                        
                        if len(sampled_entities) == len(template.placeholders):
                            prompt_text = template.fill(**sampled_entities)
                            
                            prompt = Prompt(
                                text=prompt_text,
                                source_document_id=chunk.document_id,
                                source_chunk_id=chunk.id,
                                hallucination_type=template.hallucination_type,
                                generation_strategy='template',
                                difficulty_level='medium',
                                metadata={
                                    'template_category': template.category,
                                    'entities_used': sampled_entities,
                                    'chunk_section': chunk.section
                                }
                            )
                            prompts.append(prompt)
                
                except Exception as e:
                    self.logger.warning(f"Failed to generate prompt from template: {e}")
        
        return prompts
    
    def apply_heuristic_rules(self, prompt: Prompt, source_chunk: DocumentChunk, all_chunks: List[DocumentChunk]) -> List[Prompt]:
        """Apply rule-based modifications to create variations."""
        if not self.config.perturbation_enabled:
            return [prompt]
        
        variations = [prompt]  # Include original
        
        # Extract entities for rule application
        entities = self.entity_extractor.extract_entities(source_chunk.content)
        
        # Apply each heuristic rule
        try:
            # Entity substitution
            substitution_variations = self.heuristic_rules.entity_substitution(prompt, entities)
            variations.extend(substitution_variations[:2])  # Limit variations
            
            # Fact perturbation
            fact_variations = self.heuristic_rules.fact_perturbation(prompt, source_chunk)
            variations.extend(fact_variations[:2])
            
            # Context mixing
            other_chunks = [c for c in all_chunks if c.id != source_chunk.id]
            if other_chunks:
                context_variations = self.heuristic_rules.context_mixing(prompt, other_chunks)
                variations.extend(context_variations[:1])
            
            # Negation injection
            negation_variations = self.heuristic_rules.negation_injection(prompt)
            variations.extend(negation_variations[:1])
        
        except Exception as e:
            self.logger.warning(f"Failed to apply heuristic rules: {e}")
        
        return variations
    
    def generate_batch(self, chunks: List[DocumentChunk], num_prompts: Optional[int] = None) -> List[Prompt]:
        """Generate a batch of prompts with variations."""
        if num_prompts is None:
            num_prompts = self.config.max_prompts_per_document
        
        self.logger.info(f"Generating {num_prompts} prompts from {len(chunks)} chunks")
        
        # Generate base prompts
        base_prompts = self.generate_from_chunks(chunks)
        
        # Apply heuristic rules to create variations
        all_prompts = []
        for prompt in base_prompts:
            # Find the source chunk
            source_chunk = next((c for c in chunks if c.id == prompt.source_chunk_id), chunks[0])
            variations = self.apply_heuristic_rules(prompt, source_chunk, chunks)
            all_prompts.extend(variations)
        
        # Shuffle and limit
        random.shuffle(all_prompts)
        final_prompts = all_prompts[:num_prompts]
        
        self.logger.info(f"Generated {len(final_prompts)} prompts ({len(base_prompts)} base + variations)")
        
        return final_prompts