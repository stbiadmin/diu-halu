"""Unit tests for PDF processor components."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from dodhalueval.data.pdf_processor import TextChunker, DocumentMetadata, PageContent, PDFCache
from dodhalueval.models.schemas import DocumentChunk


@pytest.mark.unit
class TestTextChunker:
    """Test TextChunker class functionality."""

    def test_text_chunker_initialization(self):
        """Test TextChunker initialization with default parameters."""
        chunker = TextChunker()
        assert chunker.chunk_size == 800
        assert chunker.chunk_overlap == 100
        assert chunker.preserve_structure is True

    def test_text_chunker_custom_parameters(self):
        """Test TextChunker initialization with custom parameters."""
        chunker = TextChunker(chunk_size=500, chunk_overlap=50, preserve_structure=False)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50
        assert chunker.preserve_structure is False

    def test_chunk_empty_text(self):
        """Test chunking empty or whitespace-only text."""
        chunker = TextChunker()
        
        # Empty string
        result = chunker.chunk_text("")
        assert result == []
        
        # Whitespace only
        result = chunker.chunk_text("   \n  \t  ")
        assert result == []

    def test_chunk_short_text_single_chunk(self):
        """Test chunking text that fits in a single chunk."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a short text that should fit in one chunk."
        
        result = chunker.chunk_text(text)
        assert len(result) == 1
        assert result[0]['text'] == text
        assert result[0]['chunk_index'] == 0
        assert result[0]['word_count'] == len(text.split())

    def test_chunk_text_with_preserve_structure(self):
        """Test chunking with structure preservation enabled."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=2, preserve_structure=True)
        text = "First paragraph text.\n\nSecond paragraph text.\n\nThird paragraph text."
        
        result = chunker.chunk_text(text)
        assert len(result) >= 1
        assert all('chunk_index' in chunk for chunk in result)
        assert all('word_count' in chunk for chunk in result)

    def test_chunk_text_without_preserve_structure(self):
        """Test chunking with structure preservation disabled."""
        chunker = TextChunker(chunk_size=5, chunk_overlap=1, preserve_structure=False)
        text = "One two three four five six seven eight nine ten eleven twelve"
        
        result = chunker.chunk_text(text)
        assert len(result) >= 2  # Should create multiple chunks
        assert all('chunk_index' in chunk for chunk in result)
        assert all('word_count' in chunk for chunk in result)

    def test_chunk_text_with_metadata(self):
        """Test chunking with metadata preservation."""
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        text = "Test text for metadata preservation functionality."
        metadata = {"source": "test.pdf", "page": 1}
        
        result = chunker.chunk_text(text, metadata)
        assert len(result) >= 1
        for chunk in result:
            assert chunk['metadata']['source'] == "test.pdf"
            assert chunk['metadata']['page'] == 1

    def test_chunk_text_overlap_logic(self):
        """Test that overlap logic works correctly."""
        chunker = TextChunker(chunk_size=3, chunk_overlap=1, preserve_structure=False)
        text = "one two three four five six seven eight"
        
        result = chunker.chunk_text(text)
        assert len(result) >= 2
        
        # Check that chunks have the expected structure
        for i, chunk in enumerate(result):
            assert chunk['chunk_index'] == i
            assert chunk['word_count'] <= 3

    def test_chunk_text_deterministic(self):
        """Test that chunking is deterministic."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        text = "Consistent text for deterministic testing of chunking functionality."
        
        result1 = chunker.chunk_text(text)
        result2 = chunker.chunk_text(text)
        
        assert len(result1) == len(result2)
        for i in range(len(result1)):
            assert result1[i]['text'] == result2[i]['text']
            assert result1[i]['chunk_index'] == result2[i]['chunk_index']


@pytest.mark.unit
class TestDocumentMetadata:
    """Test DocumentMetadata dataclass."""

    def test_document_metadata_creation(self):
        """Test DocumentMetadata creation."""
        metadata = DocumentMetadata(
            title="Test Document",
            pages=10,
            file_size=1024
        )
        
        assert metadata.title == "Test Document"
        assert metadata.pages == 10
        assert metadata.file_size == 1024
        assert metadata.created_date is None

    def test_document_metadata_with_created_date(self):
        """Test DocumentMetadata with created_date."""
        metadata = DocumentMetadata(
            title="Test Document",
            pages=5,
            file_size=512,
            created_date="2024-01-01"
        )
        
        assert metadata.created_date == "2024-01-01"


@pytest.mark.unit
class TestPageContent:
    """Test PageContent dataclass."""

    def test_page_content_creation(self):
        """Test PageContent creation."""
        page = PageContent(
            page_number=1,
            text="Sample page content",
            metadata={"extracted": True}
        )
        
        assert page.page_number == 1
        assert page.text == "Sample page content"
        assert page.metadata["extracted"] is True


@pytest.mark.unit
class TestPDFCache:
    """Test PDFCache functionality."""

    def test_pdf_cache_initialization(self, temp_dir):
        """Test PDFCache initialization."""
        cache_dir = temp_dir / "cache"
        cache = PDFCache(cache_dir)
        
        assert cache.cache_dir == cache_dir
        assert cache_dir.exists()

    def test_pdf_cache_get_miss(self, temp_dir):
        """Test cache get operation (always misses in testing)."""
        cache_dir = temp_dir / "cache"
        cache = PDFCache(cache_dir)
        
        result = cache.get("test_key")
        assert result is None

    def test_pdf_cache_set_operation(self, temp_dir):
        """Test cache set operation (no-op in testing)."""
        cache_dir = temp_dir / "cache"
        cache = PDFCache(cache_dir)
        
        # Should not raise any exceptions
        cache.set("test_key", {"data": "test"})

    def test_pdf_cache_key_generation(self, temp_dir):
        """Test cache key generation."""
        cache_dir = temp_dir / "cache"
        cache = PDFCache(cache_dir)
        
        test_path = Path("/test/path/file.pdf")
        key = cache._get_cache_key(test_path)
        
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length
        
        # Same path should generate same key
        key2 = cache._get_cache_key(test_path)
        assert key == key2


@pytest.mark.unit 
class TestPDFProcessorComponents:
    """Test individual PDF processor components."""

    def test_document_chunk_creation_from_chunker_output(self):
        """Test creating DocumentChunk from TextChunker output."""
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        text = "Test document content for chunk creation testing."
        
        chunk_data = chunker.chunk_text(text, {"source": "test.pdf"})
        
        # Verify we can create DocumentChunk from chunker output
        for chunk_dict in chunk_data:
            chunk = DocumentChunk(
                document_id="test_doc",
                content=chunk_dict['text'],
                page_number=1,
                chunk_index=chunk_dict['chunk_index'],
                metadata=chunk_dict['metadata']
            )
            
            assert chunk.document_id == "test_doc"
            assert chunk.content == chunk_dict['text']
            assert chunk.page_number == 1
            assert chunk.chunk_index == chunk_dict['chunk_index']
            assert chunk.metadata['source'] == "test.pdf"

    def test_chunker_with_large_text(self):
        """Test chunker with large text input."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        
        # Create a longer text
        words = ["word"] * 50
        text = " ".join(words)
        
        result = chunker.chunk_text(text)
        
        # Should create multiple chunks
        assert len(result) > 1
        
        # All chunks should have valid structure
        for chunk in result:
            assert 'text' in chunk
            assert 'chunk_index' in chunk
            assert 'word_count' in chunk
            assert chunk['word_count'] <= 10

    def test_chunker_edge_cases(self):
        """Test chunker with edge cases."""
        chunker = TextChunker(chunk_size=1, chunk_overlap=0)
        
        # Single word
        result = chunker.chunk_text("word")
        assert len(result) == 1
        assert result[0]['text'] == "word"
        
        # Multiple whitespace
        result = chunker.chunk_text("word1     word2")
        assert len(result) == 2
        
        # Special characters
        result = chunker.chunk_text("word1! word2? word3.")
        assert len(result) == 3