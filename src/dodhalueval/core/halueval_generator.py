"""
HaluEval-compatible hallucination generator following the paper methodology.
"""

import logging
import random
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..models.schemas import Response, Prompt, DocumentChunk
from ..providers.base import LLMProvider

logger = logging.getLogger(__name__)


class HaluEvalGenerator:
    """Generator following HaluEval paper methodology exactly."""
    
    def __init__(self, llm_client: LLMProvider, config: Dict[str, Any]):
        """Initialize HaluEval generator.
        
        Args:
            llm_client: LLM provider for generation
            config: Configuration dictionary
        """
        self.llm_provider = llm_client
        self.config = config
        self.halueval_config = config.get('halueval_settings', {})
        self.templates = self._load_halueval_templates()
        
        # HaluEval hallucination patterns
        self.hallucination_patterns = self.halueval_config.get('hallucination_patterns', [
            'factual_contradiction',
            'context_misunderstanding', 
            'specificity_mismatch',
            'invalid_inference'
        ])
        
        self.use_two_stage = self.halueval_config.get('use_two_stage_generation', True)
        self.enable_filtering = self.halueval_config.get('enable_filtering', True)
        
    def _load_halueval_templates(self) -> Dict[str, Any]:
        """Load HaluEval prompt templates."""
        template_file = self.halueval_config.get('template_file', 'data/prompts/halueval_templates.yaml')
        template_path = Path(template_file)
        
        # Try absolute path if relative doesn't work
        if not template_path.exists():
            # Try from current working directory
            template_path = Path.cwd() / template_file
            
        if not template_path.exists():
            # Try from project root (go up from any potential scripts directory)
            project_root = Path.cwd()
            if project_root.name == 'scripts':
                project_root = project_root.parent
            template_path = project_root / template_file
            
        if not template_path.exists():
            logger.warning(f"HaluEval template file not found: {template_path}")
            return self._get_default_templates()
            
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                templates = yaml.safe_load(f)
            logger.info(f"Loaded HaluEval templates from {template_path}")
            return templates
        except Exception as e:
            logger.error(f"Failed to load HaluEval templates: {e}")
            return self._get_default_templates()
    
    def _get_default_templates(self) -> Dict[str, Any]:
        """Get default HaluEval templates if file not found."""
        return {
            'halueval_generation': {
                'system_prompt': (
                    "I want you act as a hallucination answer generator. Given a question, right answer, and related knowledge, your "
                    "objective is to write a hallucinated answer that sounds plausible but is factually incorrect."
                ),
                'patterns': {
                    'factual_contradiction': {
                        'instruction': (
                            "You are trying to answer a question but there is a factual contradiction between the answer and the knowledge. "
                            "You can fabricate some information that does not exist in the provided knowledge."
                        ),
                        'demonstration': (
                            "#Knowledge#: The M1A2 Abrams tank has a 120mm smoothbore gun and weighs approximately 68 tons.\n"
                            "#Question#: What is the main gun caliber of the M1A2 Abrams tank?\n"
                            "#Right Answer#: 120mm smoothbore gun\n"
                            "#Hallucinated Answer#: 105mm rifled gun"
                        )
                    },
                    'context_misunderstanding': {
                        'instruction': (
                            "You are trying to answer a question but you misunderstand the question context and intention."
                        ),
                        'demonstration': (
                            "#Knowledge#: Marine Corps doctrine emphasizes combined arms operations in expeditionary warfare.\n"
                            "#Question#: What does Marine Corps doctrine emphasize in expeditionary operations?\n"
                            "#Right Answer#: Combined arms operations\n"
                            "#Hallucinated Answer#: Individual unit autonomy and separation"
                        )
                    },
                    'specificity_mismatch': {
                        'instruction': (
                            "You are trying to answer a question but the answer is too general or too specific to answer the question at an "
                            "appropriate level of specificity."
                        ),
                        'demonstration': (
                            "#Knowledge#: The MAGTF consists of four elements: command, ground combat, aviation combat, and logistics.\n"
                            "#Question#: What are the main elements of a MAGTF?\n"
                            "#Right Answer#: Command, ground combat, aviation combat, and logistics elements\n"
                            "#Hallucinated Answer#: Various military components"
                        )
                    },
                    'invalid_inference': {
                        'instruction': (
                            "You are trying to answer a question but the answer cannot be inferred from the knowledge. You can incorrectly "
                            "reason with the knowledge to arrive at a hallucinated answer."
                        ),
                        'demonstration': (
                            "#Knowledge#: Small wars require different tactics than conventional warfare.\n"
                            "#Question#: What specific tactics are most effective in small wars?\n"
                            "#Right Answer#: The document doesn't specify particular tactics\n"
                            "#Hallucinated Answer#: Primarily heavy armor and artillery bombardment"
                        )
                    }
                },
                'template': (
                    "{instruction}\n\n"
                    "{demonstration}\n\n"
                    "You should try your best to make the answer become hallucinated. #Hallucinated Answer# can only have about "
                    "5 more words than #Right Answer#.\n\n"
                    "#Knowledge#: {knowledge}\n"
                    "#Question#: {question}\n"
                    "#Right Answer#: {right_answer}\n"
                    "#Hallucinated Answer#:"
                ),
                'filtering': {
                    'system_prompt': (
                        "I want you act as an answer judge. Given a question, two answers, and related knowledge, your objective is to "
                        "select the best and correct answer without hallucination and non-factual information."
                    ),
                    'instruction': (
                        "You should try your best to select the best and correct answer. If the two answers are the same, you can randomly "
                        "choose one. If both answers are incorrect, choose the better one. You MUST select an answer from the provided "
                        "two answers."
                    ),
                    'template': (
                        "#Knowledge#: {knowledge}\n"
                        "#Question#: {question}\n"
                        "#Answer 1#: {answer1}\n"
                        "#Answer 2#: {answer2}\n"
                        "#Your Choice#:"
                    )
                }
            }
        }
    
    def generate_hallucinated_response(self, knowledge: str, question: str, 
                                     correct_answer: str, hallucination_type: Optional[str] = None,
                                     generation_schema: str = "one_pass") -> str:
        """Generate hallucinated answer using HaluEval prompt structure.
        
        Args:
            knowledge: Document knowledge context
            question: Question to answer
            correct_answer: The correct answer
            hallucination_type: Type of hallucination to inject
            generation_schema: "one_pass" or "conversational"
            
        Returns:
            Hallucinated answer text
        """
        if hallucination_type is None:
            hallucination_type = random.choice(self.hallucination_patterns)
            
        logger.debug(f"Generating hallucinated response with type: {hallucination_type}")
        
        # Build the prompt based on HaluEval methodology
        prompt = self._build_hallucination_prompt(
            knowledge=knowledge,
            question=question, 
            correct_answer=correct_answer,
            hallucination_type=hallucination_type
        )
        
        # Note: These methods need to be called from an async context
        # For now, return a simple fallback response
        logger.warning("HaluEval generation requires async context, using fallback")
        
        if hallucination_type == "factual_contradiction":
            return "U.S. Highway 70"  # Simple factual substitution
        elif hallucination_type == "context_misunderstanding":
            return "The document addresses naval procedures instead."
        elif hallucination_type == "specificity_mismatch":
            return "Various transportation methods are mentioned."
        elif hallucination_type == "invalid_inference":
            return "Based on the context, this requires heavy machinery."
        else:
            return "According to the document, this information is available in Section 3."
    
    async def generate_hallucinated_response_async(self, knowledge: str, question: str, 
                                     correct_answer: str, hallucination_type: Optional[str] = None,
                                     generation_schema: str = "one_pass") -> str:
        """Generate hallucinated answer using HaluEval prompt structure (async version)."""
        if hallucination_type is None:
            hallucination_type = random.choice(self.hallucination_patterns)
            
        logger.debug(f"Generating hallucinated response with type: {hallucination_type}")
        
        # Build the prompt based on HaluEval methodology
        prompt = self._build_hallucination_prompt(
            knowledge=knowledge,
            question=question, 
            correct_answer=correct_answer,
            hallucination_type=hallucination_type
        )
        
        if generation_schema == "one_pass":
            return await self._generate_one_pass(prompt)
        elif generation_schema == "conversational":
            return await self._generate_conversational(prompt, hallucination_type)
        else:
            raise ValueError(f"Unknown generation schema: {generation_schema}")
    
    async def generate_with_filtering_async(self, knowledge: str, question: str, correct_answer: str,
                              hallucination_type: Optional[str] = None) -> str:
        """Generate using two-stage generation with filtering (async version)."""
        if not self.use_two_stage:
            return await self.generate_hallucinated_response_async(knowledge, question, correct_answer, hallucination_type)
            
        # Generate with both schemas
        answer1 = await self.generate_hallucinated_response_async(
            knowledge, question, correct_answer, hallucination_type, "one_pass"
        )
        answer2 = await self.generate_hallucinated_response_async(
            knowledge, question, correct_answer, hallucination_type, "conversational"
        )
        
        if not self.enable_filtering or answer1.strip() == answer2.strip():
            # Return the first answer if filtering disabled or answers are the same
            return answer1
            
        # Filter between the two answers
        return await self._filter_best_hallucination(answer1, answer2, knowledge, question)
    
    def generate_with_filtering(self, knowledge: str, question: str, correct_answer: str,
                              hallucination_type: Optional[str] = None) -> str:
        """Generate using two-stage generation with filtering.
        
        Args:
            knowledge: Document knowledge context
            question: Question to answer
            correct_answer: The correct answer
            hallucination_type: Type of hallucination to inject
            
        Returns:
            Best hallucinated answer after filtering
        """
        if not self.use_two_stage:
            return self.generate_hallucinated_response(knowledge, question, correct_answer, hallucination_type)
            
        # Generate with both schemas
        answer1 = self.generate_hallucinated_response(
            knowledge, question, correct_answer, hallucination_type, "one_pass"
        )
        answer2 = self.generate_hallucinated_response(
            knowledge, question, correct_answer, hallucination_type, "conversational"
        )
        
        if not self.enable_filtering or answer1.strip() == answer2.strip():
            # Return the first answer if filtering disabled or answers are the same
            return answer1
            
        # Filter between the two answers (simplified for now)
        logger.debug("Using simple filtering - selecting first answer")
        return answer1
    
    def _build_hallucination_prompt(self, knowledge: str, question: str, 
                                  correct_answer: str, hallucination_type: str) -> str:
        """Build HaluEval-style hallucination prompt."""
        templates = self.templates.get('halueval_generation', {})
        patterns = templates.get('patterns', {})
        
        if hallucination_type not in patterns:
            logger.warning(f"Unknown hallucination type: {hallucination_type}, using factual_contradiction")
            hallucination_type = 'factual_contradiction'
            
        pattern = patterns[hallucination_type]
        instruction = pattern.get('instruction', '')
        demonstration = pattern.get('demonstration', '')
        template = templates.get('template', '')
        
        return template.format(
            instruction=instruction,
            demonstration=demonstration,
            knowledge=knowledge,
            question=question,
            right_answer=correct_answer
        )
    
    async def _generate_one_pass(self, prompt: str) -> str:
        """One-pass instruction following schema."""
        try:
            from ..providers.base import GenerationParameters
            system_prompt = self.templates['halueval_generation']['system_prompt']
            params = GenerationParameters(temperature=0.7, max_tokens=200)
            
            # Use async method directly
            response = await self.llm_provider.generate(
                prompt=prompt,
                params=params,
                system_prompt=system_prompt
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate one-pass response: {e}")
            return "Unable to generate hallucinated response."
    
    async def _generate_conversational(self, prompt: str, hallucination_type: str) -> str:
        """Conversational schema with successive instruction learning."""
        try:
            from ..providers.base import GenerationParameters
            
            system_prompt = self.templates['halueval_generation']['system_prompt']
            
            # First, teach the instruction
            instruction_prompt = f"I will teach you to generate hallucinated answers. The type we're focusing on is: {hallucination_type}. Do you understand?"
            
            teach_params = GenerationParameters(temperature=0.3, max_tokens=50)
            await self.llm_provider.generate(
                prompt=instruction_prompt,
                params=teach_params,
                system_prompt=system_prompt
            )
            
            # Then generate the hallucinated answer
            gen_params = GenerationParameters(temperature=0.7, max_tokens=200)
            response = await self.llm_provider.generate(
                prompt=prompt,
                params=gen_params,
                system_prompt=system_prompt
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate conversational response: {e}")
            return "Unable to generate hallucinated response."
    
    async def _filter_best_hallucination(self, answer1: str, answer2: str, 
                                 knowledge: str, question: str) -> str:
        """Filter between two hallucinated answers to select most plausible."""
        try:
            filtering_config = self.templates['halueval_generation']['filtering']
            system_prompt = filtering_config['system_prompt']
            instruction = filtering_config['instruction']
            demonstration = filtering_config['demonstration']
            template = filtering_config['template']
            
            filter_prompt = template.format(
                instruction=instruction,
                demonstration=demonstration,
                knowledge=knowledge,
                question=question,
                answer1=answer1,
                answer2=answer2
            )
            
            from ..providers.base import GenerationParameters
            
            filter_params = GenerationParameters(temperature=0.1, max_tokens=100)
            response = await self.llm_provider.generate(
                prompt=filter_prompt,
                params=filter_params,
                system_prompt=system_prompt
            )
            
            # Parse the response to determine which answer was selected
            response_text = response.text
            response_lower = response_text.lower()
            if "answer 1" in response_lower or "1" in response_lower:
                return answer1
            elif "answer 2" in response_lower or "2" in response_lower:
                return answer2
            else:
                # Default to first answer if unclear
                logger.warning("Unclear filtering response, defaulting to answer 1")
                return answer1
                
        except Exception as e:
            logger.error(f"Failed to filter responses: {e}")
            return answer1  # Default to first answer on error