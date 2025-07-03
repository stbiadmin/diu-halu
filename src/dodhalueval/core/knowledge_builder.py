"""
Knowledge context builder for HaluEval-compatible hallucination generation.
"""

import logging
import re
from typing import List, Dict, Optional, Any

# Optional imports for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    
# Always import numpy as it's used in multiple places
try:
    import numpy as np
except ImportError:
    np = None

from ..models.schemas import DocumentChunk, Prompt

logger = logging.getLogger(__name__)


class KnowledgeContextBuilder:
    """Build HaluEval-style knowledge context from document chunks."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize knowledge context builder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.halueval_config = config.get('halueval_settings', {})
        self.max_knowledge_length = self.halueval_config.get('max_knowledge_length', 500)
        
        # Initialize sentence transformer for semantic similarity (optional)
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Initialized sentence transformer for semantic similarity")
            except Exception as e:
                logger.warning(f"Failed to initialize sentence transformer: {e}")
                logger.info("Will use text-based similarity instead")
        else:
            logger.info("SentenceTransformers not available, using text-based similarity")
    
    def build_knowledge_context(self, chunks: List[DocumentChunk], prompt: Prompt) -> str:
        """Convert document chunks to HaluEval #Knowledge# format.
        
        Args:
            chunks: List of document chunks
            prompt: The prompt/question for context
            
        Returns:
            Formatted knowledge context string
        """
        if not chunks:
            logger.warning("No document chunks provided for knowledge context")
            return ""
        
        # Extract relevant knowledge based on the prompt
        relevant_knowledge = self.extract_relevant_knowledge(chunks, prompt.text)
        
        # Format as clean knowledge context
        formatted_knowledge = self._format_knowledge_context(relevant_knowledge)
        
        # Ensure it doesn't exceed max length
        if len(formatted_knowledge) > self.max_knowledge_length:
            formatted_knowledge = self._truncate_knowledge(formatted_knowledge)
        
        logger.debug(f"Built knowledge context ({len(formatted_knowledge)} chars): {formatted_knowledge[:100]}...")
        return formatted_knowledge
    
    def extract_relevant_knowledge(self, chunks: List[DocumentChunk], 
                                 question: str, max_length: Optional[int] = None) -> str:
        """Extract most relevant knowledge for question.
        
        Args:
            chunks: List of document chunks
            question: Question to find relevant knowledge for
            max_length: Maximum length of returned knowledge
            
        Returns:
            Most relevant knowledge text
        """
        if not chunks:
            logger.warning("No chunks provided for knowledge extraction")
            return ""
        
        max_length = max_length or self.max_knowledge_length
        logger.debug(f"Extracting knowledge from {len(chunks)} chunks with max_length={max_length}")
        
        # Score chunks by relevance to question
        scored_chunks = self._score_chunks_by_relevance(chunks, question)
        
        # Build knowledge context from highest scoring chunks
        relevant_text = ""
        chunks_used = 0
        
        for chunk, score in scored_chunks:
            chunk_text = self._clean_chunk_text(chunk.content)
            logger.debug(f"Chunk {chunks_used}: score={score:.3f}, length={len(chunk_text)}, preview='{chunk_text[:100]}...'")
            
            # Only skip very low scoring chunks if we have alternatives
            if score < 0.1 and chunks_used > 0:
                logger.debug(f"Skipping low-scoring chunk (score={score:.3f})")
                continue
            
            # Check if adding this chunk would exceed max length
            potential_length = len(relevant_text) + len(chunk_text) + 50
            if potential_length > max_length and relevant_text:  # Only break if we already have some content
                logger.debug(f"Stopping at {chunks_used} chunks due to length limit ({potential_length} > {max_length})")
                break
            
            if relevant_text:
                relevant_text += " "
            relevant_text += chunk_text
            chunks_used += 1
            
            # Ensure we get at least some reasonable content
            if chunks_used >= 3 and len(relevant_text) > max_length * 0.7:
                break
        
        logger.debug(f"Extracted {len(relevant_text)} chars from {chunks_used} chunks")
        
        # Enhanced fallback - combine multiple chunks if single chunk is insufficient
        if not relevant_text and chunks:
            logger.warning("No relevant text found, using fallback strategy")
            fallback_text = ""
            for chunk in chunks[:3]:  # Use up to 3 chunks for fallback
                chunk_text = self._clean_chunk_text(chunk.content)
                if len(fallback_text) + len(chunk_text) < max_length:
                    if fallback_text:
                        fallback_text += " "
                    fallback_text += chunk_text
            return fallback_text
        
        return relevant_text
    
    def _score_chunks_by_relevance(self, chunks: List[DocumentChunk], 
                                 question: str) -> List[tuple]:
        """Score chunks by relevance to question.
        
        Args:
            chunks: List of document chunks
            question: Question to score against
            
        Returns:
            List of (chunk, score) tuples sorted by relevance
        """
        if self.embedding_model:
            return self._score_chunks_semantic(chunks, question)
        else:
            return self._score_chunks_textual(chunks, question)
    
    def _score_chunks_semantic(self, chunks: List[DocumentChunk], 
                             question: str) -> List[tuple]:
        """Score chunks using semantic similarity."""
        try:
            # Get embeddings for all chunks at once (more efficient)
            question_embedding = self.embedding_model.encode([question])[0]
            chunk_texts = [chunk.content for chunk in chunks]
            chunk_embeddings = self.embedding_model.encode(chunk_texts)
            
            scored_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_embedding = chunk_embeddings[i]
                
                # Calculate cosine similarity
                semantic_score = np.dot(question_embedding, chunk_embedding) / (
                    np.linalg.norm(question_embedding) * np.linalg.norm(chunk_embedding)
                )
                
                # Boost score for keyword overlap as well
                question_lower = question.lower()
                chunk_lower = chunk.content.lower()
                
                # Simple keyword boost
                question_words = set(question_lower.split())
                chunk_words = set(chunk_lower.split())
                keyword_overlap = len(question_words.intersection(chunk_words)) / len(question_words) if question_words else 0
                
                # Combined score: 70% semantic, 30% keyword
                combined_score = 0.7 * float(semantic_score) + 0.3 * keyword_overlap
                
                scored_chunks.append((chunk, combined_score))
                logger.debug(f"Chunk semantic={semantic_score:.3f}, keyword={keyword_overlap:.3f}, combined={combined_score:.3f}")
            
            # Sort by similarity score (descending)
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            return scored_chunks
            
        except Exception as e:
            logger.warning(f"Semantic scoring failed: {e}, falling back to textual scoring")
            return self._score_chunks_textual(chunks, question)
    
    def _score_chunks_textual(self, chunks: List[DocumentChunk], 
                            question: str) -> List[tuple]:
        """Score chunks using textual overlap and keyword matching."""
        question_lower = question.lower()
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        
        scored_chunks = []
        for chunk in chunks:
            chunk_lower = chunk.content.lower()
            chunk_words = set(re.findall(r'\b\w+\b', chunk_lower))
            
            # Calculate score based on word overlap
            overlap = len(question_words.intersection(chunk_words))
            total_words = len(question_words.union(chunk_words))
            
            # Jaccard similarity
            jaccard_score = overlap / total_words if total_words > 0 else 0
            
            # Boost score for exact phrase matches
            phrase_boost = 0
            question_phrases = re.findall(r'\b\w+(?:\s+\w+)*\b', question_lower)
            for phrase in question_phrases:
                if len(phrase.split()) > 1 and phrase in chunk_lower:
                    phrase_boost += 0.2
            
            final_score = jaccard_score + phrase_boost
            scored_chunks.append((chunk, final_score))
        
        # Sort by score (descending)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks
    
    def _format_knowledge_context(self, knowledge_text: str) -> str:
        """Format knowledge text for HaluEval context.
        
        Args:
            knowledge_text: Raw knowledge text
            
        Returns:
            Formatted knowledge context
        """
        if not knowledge_text:
            return ""
        
        # Clean up the text
        formatted = self._clean_knowledge_text(knowledge_text)
        
        # Ensure it ends with proper punctuation
        if formatted and not formatted.endswith(('.', '!', '?', ':')):
            formatted += '.'
        
        return formatted
    
    def _clean_knowledge_text(self, text: str) -> str:
        """Clean knowledge text for better formatting.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'\x0c', ' ', text)  # Form feed
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)  # Control chars
        
        # Clean up sentence boundaries
        text = re.sub(r'\s*\.\s*', '. ', text)
        text = re.sub(r'\s*,\s*', ', ', text)
        
        # Remove duplicate sentences
        sentences = text.split('. ')
        unique_sentences = []
        seen = set()
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence)
        
        return '. '.join(unique_sentences).strip()
    
    def _clean_chunk_text(self, text: str) -> str:
        """Clean individual chunk text.
        
        Args:
            text: Chunk text to clean
            
        Returns:
            Cleaned chunk text
        """
        if not text:
            return ""
        
        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Filter out very short or low-quality chunks
        if len(text) < 20:  # Skip chunks with less than 20 characters
            logger.debug(f"Skipping short chunk: '{text}'")
            return ""
        
        # Skip chunks that are just numbers or single words
        if text.isdigit() or len(text.split()) < 3:
            logger.debug(f"Skipping low-quality chunk: '{text}'")
            return ""
        
        # Remove obvious headers/footers (lines with all caps and < 50 chars)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 50 or not line.isupper():
                cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)
    
    def _truncate_knowledge(self, knowledge: str) -> str:
        """Truncate knowledge to fit within max length while preserving sentences.
        
        Args:
            knowledge: Knowledge text to truncate
            
        Returns:
            Truncated knowledge text
        """
        if len(knowledge) <= self.max_knowledge_length:
            return knowledge
        
        # Try to truncate at sentence boundaries
        sentences = knowledge.split('. ')
        truncated = ""
        
        for sentence in sentences:
            candidate = truncated + sentence + '. ' if truncated else sentence + '. '
            if len(candidate.strip()) > self.max_knowledge_length:
                break
            truncated = candidate
        
        # If no complete sentences fit, truncate at word boundary
        if not truncated:
            words = knowledge.split()
            truncated = ""
            for word in words:
                candidate = truncated + word + ' ' if truncated else word + ' '
                if len(candidate) > self.max_knowledge_length - 3:  # Leave room for "..."
                    break
                truncated = candidate
            truncated = truncated.strip() + "..."
        
        return truncated.strip()
    
    def get_knowledge_stats(self, knowledge: str) -> Dict[str, Any]:
        """Get statistics about the knowledge context.
        
        Args:
            knowledge: Knowledge context string
            
        Returns:
            Dictionary with knowledge statistics
        """
        if not knowledge:
            return {"length": 0, "word_count": 0, "sentence_count": 0}
        
        word_count = len(knowledge.split())
        sentence_count = len([s for s in knowledge.split('.') if s.strip()])
        
        return {
            "length": len(knowledge),
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_words_per_sentence": word_count / sentence_count if sentence_count > 0 else 0
        }