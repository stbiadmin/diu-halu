"""
Response Generator for DoDHaluEval

This module handles generating responses from prompts using multiple LLM providers,
with support for controlled hallucination injection and batch processing.
"""

import asyncio
import time
import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import random

from ..models.schemas import Prompt, Response, DocumentChunk
from ..providers.base import LLMProvider, GenerationParameters
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ResponseConfig:
    """Configuration for response generation."""
    hallucination_rate: float = 0.3
    max_response_length: int = 2000
    temperature: float = 0.7
    concurrent_requests: int = 5
    request_delay: float = 0.1
    retry_attempts: int = 3


class HallucinationInjector:
    """Injects various types of hallucinations into responses."""
    
    def __init__(self):
        # Pattern-based replacements for common "no info" responses
        self.no_info_hallucinations = [
            "According to the document, {topic} involves a three-phase approach starting with initial reconnaissance, followed by force deployment, and concluding with tactical engagement.",
            "The document indicates that {topic} typically requires 12-15 personnel organized into specialized teams with specific operational guidelines outlined in Section 3.",
            "Based on the context provided, {topic} follows standard Marine Corps protocol established in 2019, which emphasizes rapid deployment and coordinated fire support.",
            "The document specifies that {topic} must be conducted within 72 hours of initial authorization, using the standardized MAGTF framework.",
            "As outlined in the document, {topic} incorporates lessons learned from Operation Desert Storm and has been updated to reflect modern asymmetric warfare requirements."
        ]
        
        self.specific_detail_hallucinations = {
            "date": ["March 15-17, 2024", "scheduled for Q3 2024", "tentatively set for August 2024"],
            "location": ["Camp Pendleton, California", "Quantico Marine Corps Base", "Marine Corps Air Ground Combat Center"],
            "speaker": ["Major General James Mitchell", "Colonel Sarah Rodriguez", "Brigadier General Thomas Chen"],
            "lesson": ["focused on amphibious assault tactics", "covered urban warfare scenarios", "addressed counter-insurgency operations"],
            "theory": ["Boyd's OODA loop", "Clausewitzian friction theory", "maneuver warfare doctrine"],
            "example": ["the Battle of Fallujah case study", "Operation Phantom Fury analysis", "the Marjah offensive"]
        }
        
        self.factual_replacements = {
            'M1A2 Abrams': 'M3A1 Bradley',
            'Apache': 'Chinook', 
            'F-16': 'F-18',
            'Navy': 'Air Force',
            'Army': 'Marines',
            '120mm': '105mm',
            '4000 meters': '3500 meters',
            'depleted uranium': 'tungsten',
            'four specialists': 'five specialists',
            '52 critical systems': '48 critical systems'
        }
        
        self.logical_contradictions = [
            "However, the opposite is also true",
            "This contradicts the previous statement",
            "Although this is impossible",
            "Despite being mutually exclusive"
        ]
        
        self.context_mix_phrases = [
            "As mentioned in the naval doctrine",
            "Similar to air force procedures", 
            "Following civilian protocols",
            "Based on outdated regulations"
        ]

    def inject_factual_hallucination(self, response: str, injection_probability: float = 0.3) -> str:
        """Replace factual information with plausible alternatives."""
        if random.random() > injection_probability:
            return response
            
        modified_response = response
        for original, replacement in self.factual_replacements.items():
            if original.lower() in modified_response.lower():
                if random.random() < 0.4:  # 40% chance to replace each found fact
                    modified_response = modified_response.replace(original, replacement)
                    logger.debug(f"Factual injection: {original} -> {replacement}")
                    break
        
        return modified_response

    def inject_logical_hallucination(self, response: str, injection_probability: float = 0.2) -> str:
        """Introduce logical inconsistencies."""
        if random.random() > injection_probability:
            return response
            
        sentences = response.split('. ')
        if len(sentences) > 2:
            # Insert contradiction after a random sentence
            insert_pos = random.randint(1, len(sentences) - 1)
            contradiction = random.choice(self.logical_contradictions)
            sentences.insert(insert_pos, contradiction)
            logger.debug(f"Logical injection: {contradiction}")
            
        return '. '.join(sentences)

    def inject_context_hallucination(self, response: str, injection_probability: float = 0.25) -> str:
        """Mix inappropriate contexts."""
        if random.random() > injection_probability:
            return response
            
        sentences = response.split('. ')
        if sentences:
            # Add inappropriate context reference
            context_phrase = random.choice(self.context_mix_phrases)
            first_sentence = sentences[0]
            sentences[0] = f"{context_phrase}, {first_sentence.lower()}"
            logger.debug(f"Context injection: {context_phrase}")
            
        return '. '.join(sentences)

    def inject_hallucinations(
        self, 
        response: str, 
        hallucination_types: List[str] = None,
        overall_probability: float = 0.3,
        prompt: Optional[Prompt] = None
    ) -> tuple[str, List[str]]:
        """
        Apply multiple hallucination types to a response.
        
        Returns:
            Tuple of (modified_response, list_of_applied_injections)
        """
        if random.random() > overall_probability:
            return response, []
            
        # Check if this is a "no information" response
        no_info_indicators = [
            "does not provide",
            "does not contain", 
            "no information",
            "cannot provide",
            "doesn't mention",
            "not specified",
            "doesn't contain",
            "not mentioned"
        ]
        
        is_no_info_response = any(indicator in response.lower() for indicator in no_info_indicators)
        
        if is_no_info_response and prompt:
            # Replace with a confident hallucinated response
            topic = self._extract_topic_from_prompt(prompt.text)
            hallucinated_response = self._generate_hallucinated_response(topic, prompt.text)
            return hallucinated_response, ['factual']
            
        # Otherwise, use standard injection methods
        if hallucination_types is None:
            hallucination_types = ['factual', 'logical', 'context']
            
        modified_response = response
        applied_injections = []
        
        if 'factual' in hallucination_types:
            original = modified_response
            modified_response = self.inject_factual_hallucination(modified_response)
            if modified_response != original:
                applied_injections.append('factual')
                
        if 'logical' in hallucination_types:
            original = modified_response
            modified_response = self.inject_logical_hallucination(modified_response)
            if modified_response != original:
                applied_injections.append('logical')
                
        if 'context' in hallucination_types:
            original = modified_response
            modified_response = self.inject_context_hallucination(modified_response)
            if modified_response != original:
                applied_injections.append('context')
        
        return modified_response, applied_injections
    
    def _extract_topic_from_prompt(self, prompt_text: str) -> str:
        """Extract the main topic from a prompt."""
        # Simple extraction - could be made more sophisticated
        prompt_lower = prompt_text.lower()
        if "date" in prompt_lower and "location" in prompt_lower:
            return "course scheduling"
        elif "examples" in prompt_lower:
            return "specific examples"
        elif "lesson" in prompt_lower:
            return "lesson content"
        elif "theories" in prompt_lower:
            return "theoretical frameworks"
        else:
            return "the requested information"
    
    def _generate_hallucinated_response(self, topic: str, prompt_text: str) -> str:
        """Generate a completely hallucinated but plausible response."""
        # Check for specific types of questions
        prompt_lower = prompt_text.lower()
        
        if "date" in prompt_lower and "location" in prompt_lower:
            date = random.choice(self.specific_detail_hallucinations["date"])
            location = random.choice(self.specific_detail_hallucinations["location"])
            speaker = random.choice(self.specific_detail_hallucinations["speaker"])
            return f"According to page 4 of the document, the next MAGTF Expeditionary Operations course is scheduled for {date} at {location}, with {speaker} serving as the keynote speaker. The document specifically states that registration closes 30 days prior to the start date."
        
        elif "examples" in prompt_lower and "irregular warfare" in prompt_lower:
            examples = random.sample(self.specific_detail_hallucinations["example"], 2)
            return f"The document specifically discusses {examples[0]} and {examples[1]} as key examples of irregular warfare tactics employed in MAGTF operations. On page 7, it details how these operations demonstrated the effectiveness of the three-phase irregular warfare model."
        
        elif "theories" in prompt_lower:
            theories = random.sample(self.specific_detail_hallucinations["theory"], 2)
            return f"According to section 2.3 of the document, the course emphasizes {theories[0]} and {theories[1]} as fundamental theoretical frameworks for deriving operational and tactical insights. The document states these theories were specifically selected after the 2018 curriculum review."
        
        elif "lesson" in prompt_lower and "fifth" in prompt_lower:
            content = random.choice(self.specific_detail_hallucinations["lesson"])
            return f"The fifth lesson of the course {content}, with particular emphasis on integrated fire support systems and advanced targeting technologies."
        
        elif "quote" in prompt_lower and "philosophy" in prompt_lower:
            return "According to the document transcript on page 12, the video on the Philosophy of Command states: 'In irregular warfare, ethical leadership is not merely a moral imperative but a strategic necessity. The commander who loses moral authority loses the ability to influence both friend and foe.' This quote is from the 47-minute mark of the video."
        
        else:
            # Generic hallucinated response with specific false details
            template = random.choice(self.no_info_hallucinations)
            enhanced_response = template.format(topic=topic)
            # Add specific false page references
            page_ref = random.choice(["page 8", "section 4.2", "appendix C", "table 3.1"])
            return f"{enhanced_response} As detailed in {page_ref}, this information was updated following the 2022 doctrinal review."


class ResponsePostProcessor:
    """Post-processes generated responses for consistency and formatting."""
    
    def __init__(self, config: ResponseConfig):
        self.config = config

    def normalize_length(self, response: str) -> str:
        """Ensure response length is within acceptable bounds."""
        if len(response) > self.config.max_response_length:
            # Truncate at sentence boundary
            truncated = response[:self.config.max_response_length]
            last_period = truncated.rfind('.')
            if last_period > len(truncated) * 0.8:  # If we can keep 80% of content
                response = truncated[:last_period + 1]
            else:
                response = truncated + "..."
                
        return response

    def standardize_format(self, response: str) -> str:
        """Apply consistent formatting to responses."""
        # Remove extra whitespace
        response = ' '.join(response.split())
        
        # Ensure proper sentence endings
        response = response.strip()
        if response and not response.endswith(('.', '!', '?', '...')):
            response += '.'
            
        return response

    def attach_metadata(
        self, 
        response: str, 
        prompt: Prompt, 
        provider_name: str,
        processing_time: float,
        injected_hallucinations: List[str]
    ) -> Dict[str, Any]:
        """Create metadata for the response."""
        return {
            'word_count': len(response.split()),
            'character_count': len(response),
            'processing_time_seconds': processing_time,
            'provider_used': provider_name,
            'prompt_id': prompt.id,
            'hallucinations_injected': injected_hallucinations,
            'generation_timestamp': time.time()
        }

    def process(
        self, 
        response: str, 
        prompt: Prompt, 
        provider_name: str,
        processing_time: float,
        injected_hallucinations: List[str] = None
    ) -> Dict[str, Any]:
        """Complete post-processing pipeline."""
        if injected_hallucinations is None:
            injected_hallucinations = []
            
        # Apply formatting
        processed_response = self.normalize_length(response)
        processed_response = self.standardize_format(processed_response)
        
        # Create metadata
        metadata = self.attach_metadata(
            processed_response, prompt, provider_name, 
            processing_time, injected_hallucinations
        )
        
        return {
            'response': processed_response,
            'metadata': metadata
        }


class ResponseGenerator:
    """
    Generates responses from prompts using multiple LLM providers.
    
    Supports controlled hallucination injection and batch processing with rate limiting.
    """
    
    def __init__(
        self, 
        providers: Dict[str, LLMProvider], 
        config: ResponseConfig = None,
        document_store: Optional[Dict[str, Any]] = None
    ):
        self.providers = providers
        self.config = config or ResponseConfig()
        self.hallucination_injector = HallucinationInjector()
        self.post_processor = ResponsePostProcessor(self.config)
        self._semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        self.document_store = document_store or {}

    def _extract_document_context(self, prompt: Prompt) -> str:
        """Extract relevant document context for the prompt."""
        context_parts = []
        
        # Debug logging
        logger.debug(f"Extracting context for prompt {prompt.id}")
        logger.debug(f"source_chunk_id: {prompt.source_chunk_id}")
        logger.debug(f"source_document_id: {prompt.source_document_id}")
        logger.debug(f"Document store keys (first 5): {list(self.document_store.keys())[:5]}")
        
        # Get specific chunk if available
        if prompt.source_chunk_id and prompt.source_chunk_id in self.document_store:
            chunk = self.document_store[prompt.source_chunk_id]
            logger.debug(f"Found chunk in store: {type(chunk)}")
            if hasattr(chunk, 'text'):
                context_parts.append(f"Source Context: {chunk.text}")
                logger.debug(f"Added context from chunk.text: {chunk.text[:100]}...")
            elif isinstance(chunk, dict) and 'text' in chunk:
                context_parts.append(f"Source Context: {chunk['text']}")
                logger.debug(f"Added context from chunk['text']: {chunk['text'][:100]}...")
        else:
            logger.debug(f"Chunk {prompt.source_chunk_id} not found in document store")
        
        # Get document info if available
        if prompt.source_document_id and prompt.source_document_id in self.document_store:
            doc_info = self.document_store[prompt.source_document_id]
            logger.debug(f"Found document in store: {type(doc_info)}")
            if hasattr(doc_info, 'title'):
                context_parts.append(f"Document: {doc_info.title}")
            elif isinstance(doc_info, dict) and 'title' in doc_info:
                context_parts.append(f"Document: {doc_info['title']}")
        else:
            logger.debug(f"Document {prompt.source_document_id} not found in document store")
        
        result = "\n\n".join(context_parts) if context_parts else ""
        logger.debug(f"Final context length: {len(result)} chars")
        return result
    
    def _build_system_prompt_with_context(self, document_context: str, inject_hallucination: bool = False) -> str:
        """Build system prompt that includes document grounding instructions."""
        if inject_hallucination:
            # Hallucination-inducing prompt that encourages confident but potentially incorrect answers
            base_prompt = (
                "You are an expert on US Department of Defense and Marine Corps doctrine with extensive knowledge. "
                "Answer all questions with confidence based on your understanding of military doctrine. "
                "Provide specific details, examples, and technical information in your responses. "
                "If exact details aren't clear, provide the most likely answer based on standard military practices."
            )
        else:
            # Conservative prompt for non-hallucinated responses
            base_prompt = (
                "You are an expert on US Department of Defense and Marine Corps doctrine. "
                "Answer questions based on the provided document context. "
                "If the context doesn't contain specific information, you may provide general knowledge about the topic "
                "while noting what information comes from the document versus general military doctrine."
            )
        
        if document_context:
            return f"{base_prompt}\n\nDocument Context:\n{document_context}"
        
        return base_prompt
    
    def _format_prompt_with_context(self, prompt_text: str, document_context: str) -> str:
        """Format the user prompt to reference the document context."""
        if document_context:
            return f"Based on the provided document context, {prompt_text.lower()}"
        
        return prompt_text
    
    def _clean_response_text(self, response: str) -> str:
        """Clean response text of formatting artifacts and tokens."""
        if not response or not response.strip():
            return "I cannot provide a response based on the given context."
            
        # Remove common model tokens and artifacts
        cleaned = response
        
        # Remove instruction tokens and formatting artifacts
        tokens_to_remove = [
            '</s>', '<s>', '[INST]', '[/INST]', '<|endoftext|>', '<|im_start|>', '<|im_end|>',
            '[/SYS]', '[SYS]'  # Additional system tokens
        ]
        for token in tokens_to_remove:
            cleaned = cleaned.replace(token, '')
        
        # Remove markdown-style quotes and artifacts more aggressively
        cleaned = re.sub(r'^\s*>\s*', '', cleaned, flags=re.MULTILINE)  # Remove leading >
        cleaned = re.sub(r'\s*>\s*$', '', cleaned, flags=re.MULTILINE)  # Remove trailing >
        cleaned = re.sub(r'^\s*>\s*$', '', cleaned, flags=re.MULTILINE)  # Remove standalone >
        cleaned = re.sub(r'\s*>\s*', ' ', cleaned)  # Replace remaining > with space
        
        # Remove HTML-like formatting and angle brackets
        cleaned = re.sub(r'<[^>]*>', '', cleaned)  # Remove HTML tags
        cleaned = re.sub(r'<\s*', '', cleaned)  # Remove standalone <
        cleaned = re.sub(r'\s*>', '', cleaned)  # Remove standalone >
        
        # Remove repeated instruction patterns and system artifacts
        cleaned = re.sub(r'\[INST\].*?\[/INST\]', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\[SYS\].*?\[/SYS\]', '', cleaned, flags=re.DOTALL)
        
        # Remove contradiction phrases and common artifacts
        cleaned = re.sub(r'Based on outdated regulations,\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'This contradicts the previous statement\.\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'As mentioned in the naval doctrine,\s*', '', cleaned, flags=re.IGNORECASE)
        
        # Clean up common response artifacts
        cleaned = re.sub(r'^\s*unfortunately,\s*i\s*', 'I ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"i'm happy to help!\s*", '', cleaned, flags=re.IGNORECASE)
        
        # Clean up whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove obviously broken repetitive patterns
        lines = cleaned.split('\n')
        cleaned_lines = []
        prev_line = ""
        repeat_count = 0
        
        for line in lines:
            if line.strip() == prev_line.strip() and line.strip():
                repeat_count += 1
                if repeat_count < 2:  # Allow one repetition
                    cleaned_lines.append(line)
            else:
                repeat_count = 0
                cleaned_lines.append(line)
                prev_line = line
        
        cleaned = '\n'.join(cleaned_lines).strip()
        
        # Ensure we never return empty responses
        if not cleaned or len(cleaned.strip()) < 10:
            return "I cannot provide a meaningful response based on the given context."
        
        return cleaned

    async def generate_single_response(
        self,
        prompt: Prompt,
        provider_name: str,
        inject_hallucination: bool = None
    ) -> Response:
        """Generate a single response from a prompt."""
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")
            
        provider = self.providers[provider_name]
        
        # Determine if we should inject hallucinations
        if inject_hallucination is None:
            inject_hallucination = random.random() < self.config.hallucination_rate
            
        start_time = time.time()
        
        try:
            async with self._semaphore:
                # Create generation parameters
                params = GenerationParameters(
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_response_length
                )
                
                # Extract document context and build contextual prompts
                document_context = self._extract_document_context(prompt)
                system_prompt = self._build_system_prompt_with_context(document_context, inject_hallucination)
                user_prompt = self._format_prompt_with_context(prompt.text, document_context)
                
                # Debug logging
                logger.debug(f"System prompt length: {len(system_prompt)}")
                logger.debug(f"User prompt: {user_prompt}")
                if document_context:
                    logger.debug(f"Document context preview: {document_context[:200]}...")
                else:
                    logger.debug("No document context found!")
                
                # Generate base response with context
                generation_result = await provider.generate(
                    user_prompt, 
                    params,
                    system_prompt=system_prompt
                )
                
                # Extract text from GenerationResult
                raw_response = generation_result.text
                
                # Apply rate limiting
                await asyncio.sleep(self.config.request_delay)
                
            # Apply hallucination injection if requested
            injected_hallucinations = []
            if inject_hallucination:
                raw_response, injected_hallucinations = self.hallucination_injector.inject_hallucinations(
                    raw_response,
                    overall_probability=1.0,  # Force injection since we already decided to inject
                    prompt=prompt
                )
            
            processing_time = time.time() - start_time
            
            # Post-process response
            processed_result = self.post_processor.process(
                raw_response, prompt, provider_name, processing_time, injected_hallucinations
            )
            
            # Clean the response text
            cleaned_response = self._clean_response_text(processed_result['response'])
            
            # Create Response object with cleaned text and context metadata
            metadata = processed_result['metadata']
            metadata.update({
                'source_document_id': prompt.source_document_id,
                'source_chunk_id': prompt.source_chunk_id,
                'document_context_provided': bool(self._extract_document_context(prompt))
            })
            
            response = Response(
                id=f"{prompt.id}_{provider_name}_{int(time.time())}",
                prompt_id=prompt.id,
                text=cleaned_response,
                model=provider.config.model,
                provider=provider_name,
                contains_hallucination=len(injected_hallucinations) > 0,
                hallucination_types=injected_hallucinations,
                confidence_score=None,  # Will be set by evaluation
                metadata=metadata
            )
            
            logger.debug(f"Generated response for prompt {prompt.id} using {provider_name}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response for prompt {prompt.id}: {e}")
            # Return error response
            return Response(
                id=f"{prompt.id}_{provider_name}_error_{int(time.time())}",
                prompt_id=prompt.id,
                text=f"Error generating response: {str(e)}",
                model=provider.config.model,
                provider=provider_name,
                contains_hallucination=False,
                hallucination_types=[],
                confidence_score=None,
                metadata={'error': str(e), 'processing_time_seconds': time.time() - start_time}
            )

    async def generate_responses(
        self,
        prompts: List[Prompt],
        models: List[str] = None,
        hallucination_rate: float = None
    ) -> List[Response]:
        """
        Generate responses for multiple prompts across multiple models.
        
        Args:
            prompts: List of prompts to generate responses for
            models: List of provider names to use (default: all available)
            hallucination_rate: Override default hallucination rate
            
        Returns:
            List of Response objects
        """
        if models is None:
            models = list(self.providers.keys())
            
        if hallucination_rate is not None:
            original_rate = self.config.hallucination_rate
            self.config.hallucination_rate = hallucination_rate
        
        logger.info(f"Generating responses for {len(prompts)} prompts using {len(models)} models")
        
        # Create all tasks
        tasks = []
        for prompt in prompts:
            for model in models:
                if model in self.providers:
                    task = self.generate_single_response(prompt, model)
                    tasks.append(task)
        
        # Execute all tasks concurrently
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and log them
            valid_responses = []
            for response in responses:
                if isinstance(response, Exception):
                    logger.error(f"Response generation failed: {response}")
                else:
                    valid_responses.append(response)
            
            logger.info(f"Successfully generated {len(valid_responses)} responses")
            return valid_responses
            
        finally:
            # Restore original hallucination rate
            if hallucination_rate is not None:
                self.config.hallucination_rate = original_rate

    async def generate_batch_with_progress(
        self,
        prompts: List[Prompt],
        models: List[str] = None,
        progress_callback: callable = None
    ) -> List[Response]:
        """
        Generate responses with progress tracking.
        
        Args:
            prompts: List of prompts
            models: List of provider names
            progress_callback: Function to call with progress updates (completed, total)
        """
        if models is None:
            models = list(self.providers.keys())
            
        total_tasks = len(prompts) * len(models)
        completed_tasks = 0
        
        responses = []
        
        # Process in smaller batches to provide progress updates
        batch_size = min(10, self.config.concurrent_requests)
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = await self.generate_responses(batch_prompts, models)
            responses.extend(batch_responses)
            
            completed_tasks += len(batch_prompts) * len(models)
            
            if progress_callback:
                progress_callback(completed_tasks, total_tasks)
                
        return responses

    async def cleanup(self):
        """Cleanup resources used by providers."""
        for provider in self.providers.values():
            if hasattr(provider, 'close'):
                await provider.close()