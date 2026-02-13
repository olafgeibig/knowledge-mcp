"""Tests for knowledge_mcp.documents module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import tempfile

from knowledge_mcp.documents import (
    DocumentManager,
    DocumentManagerError,
    TextExtractionError,
    UnsupportedFileTypeError,
    DocumentProcessingError,
    SUPPORTED_EXTENSIONS,
)


class TestDocumentExceptions:
    """Test custom exception classes."""

    def test_document_manager_error(self):
        """Test DocumentManagerError can be raised and caught."""
        with pytest.raises(DocumentManagerError):
            raise DocumentManagerError("Test error")

    def test_text_extraction_error(self):
        """Test TextExtractionError can be raised and caught."""
        with pytest.raises(TextExtractionError):
            raise TextExtractionError("Extraction failed")

    def test_text_extraction_error_chaining(self):
        """Test TextExtractionError supports exception chaining."""
        original = ValueError("Original error")
        with pytest.raises(TextExtractionError) as exc_info:
            raise TextExtractionError("Chained error") from original
        assert exc_info.value.__cause__ is original

    def test_unsupported_file_type_error(self):
        """Test UnsupportedFileTypeError can be raised and caught."""
        with pytest.raises(UnsupportedFileTypeError):
            raise UnsupportedFileTypeError("Unsupported type")

    def test_document_processing_error(self):
        """Test DocumentProcessingError can be raised and caught."""
        with pytest.raises(DocumentProcessingError):
            raise DocumentProcessingError("Processing failed")


class TestSupportedExtensions:
    """Test SUPPORTED_EXTENSIONS constant."""

    def test_supported_extensions_is_set(self):
        """Test that SUPPORTED_EXTENSIONS is a set."""
        assert isinstance(SUPPORTED_EXTENSIONS, set)

    def test_common_extensions_supported(self):
        """Test common file extensions are in the supported set."""
        common_extensions = {".pdf", ".docx", ".txt", ".md", ".html", ".json"}
        for ext in common_extensions:
            assert ext in SUPPORTED_EXTENSIONS, f"{ext} should be supported"


class TestDocumentManager:
    """Test DocumentManager class."""

    @pytest.fixture
    def mock_rag_manager(self):
        """Create a mock RagManager."""
        return MagicMock()

    @pytest.fixture
    def document_manager(self, mock_rag_manager):
        """Create a DocumentManager instance with mock dependencies."""
        return DocumentManager(rag_manager=mock_rag_manager)

    def test_init(self, document_manager, mock_rag_manager):
        """Test DocumentManager initialization."""
        assert document_manager.rag_manager is mock_rag_manager
        assert document_manager.markitdown is not None

    def test_extract_text_unsupported_extension(self, document_manager, tmp_path):
        """Test _extract_text handles unsupported extensions gracefully."""
        # Create a dummy file
        test_file = tmp_path / "test.xyz"
        test_file.write_text("test content")

        # Mock MarkItDown to return content
        with patch.object(document_manager.markitdown, "convert") as mock_convert:
            mock_result = MagicMock()
            mock_result.text_content = "extracted text"
            mock_convert.return_value = mock_result

            result = document_manager._extract_text(test_file)
            assert result == "extracted text"

    def test_extract_text_file_not_found(self, document_manager, tmp_path):
        """Test _extract_text raises TextExtractionError for missing file."""
        missing_file = tmp_path / "nonexistent.txt"

        with pytest.raises(TextExtractionError) as exc_info:
            document_manager._extract_text(missing_file)
        assert "not found" in str(exc_info.value).lower()

    def test_extract_text_permission_denied(self, document_manager, tmp_path):
        """Test _extract_text raises TextExtractionError for permission denied."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Mock to raise PermissionError during convert
        with patch.object(
            document_manager.markitdown,
            "convert",
            side_effect=PermissionError("Permission denied"),
        ):
            with pytest.raises(TextExtractionError) as exc_info:
                document_manager._extract_text(test_file)
            assert "permission denied" in str(exc_info.value).lower()

    def test_extract_text_empty_content(self, document_manager, tmp_path, caplog):
        """Test _extract_text handles empty content."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with patch.object(document_manager.markitdown, "convert") as mock_convert:
            mock_result = MagicMock()
            mock_result.text_content = ""
            mock_convert.return_value = mock_result

            result = document_manager._extract_text(test_file)
            assert result == ""
            assert "empty content" in caplog.text.lower()

    def test_add_multimodal_file_not_found(self, document_manager, tmp_path):
        """Test add_multimodal raises FileNotFoundError for missing file."""
        missing_file = tmp_path / "nonexistent.pdf"

        import asyncio

        with pytest.raises(FileNotFoundError):
            asyncio.get_event_loop().run_until_complete(
                document_manager.add_multimodal(missing_file, "test_kb")
            )

    def test_add_text_only_file_not_found(self, document_manager, tmp_path):
        """Test add_text_only raises FileNotFoundError for missing file."""
        missing_file = tmp_path / "nonexistent.pdf"

        import asyncio

        with pytest.raises(FileNotFoundError):
            asyncio.get_event_loop().run_until_complete(
                document_manager.add_text_only(missing_file, "test_kb")
            )

    def test_add_text_only_empty_content(self, document_manager, tmp_path, caplog):
        """Test add_text_only skips ingestion for empty content."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with patch.object(document_manager, "_extract_text", return_value=""):
            import asyncio

            # Should not raise, just return
            asyncio.get_event_loop().run_until_complete(
                document_manager.add_text_only(test_file, "test_kb")
            )
            assert "empty" in caplog.text.lower() or "whitespace" in caplog.text.lower()

    def test_add_unsupported_method(self, document_manager, tmp_path):
        """Test add raises ValueError for unsupported method."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        import asyncio

        with pytest.raises(ValueError) as exc_info:
            asyncio.get_event_loop().run_until_complete(
                document_manager.add(test_file, "test_kb", method="unsupported")
            )
        assert "unsupported" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_add_multimodal_method(self, document_manager, tmp_path):
        """Test add calls add_multimodal for multimodal method."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")

        with patch.object(
            document_manager, "add_multimodal", new_callable=MagicMock
        ) as mock_add:
            await document_manager.add(test_file, "test_kb", method="multimodal")
            mock_add.assert_called_once_with(test_file, "test_kb")

    @pytest.mark.asyncio
    async def test_add_text_method(self, document_manager, tmp_path):
        """Test add calls add_text_only for text method."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with patch.object(
            document_manager, "add_text_only", new_callable=MagicMock
        ) as mock_add:
            await document_manager.add(test_file, "test_kb", method="text")
            mock_add.assert_called_once_with(test_file, "test_kb")

    @pytest.mark.asyncio
    async def test_add_multimodal_method(self, document_manager, tmp_path):
        """Test add calls add_multimodal for multimodal method."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")

        with patch.object(
            document_manager, "add_multimodal", new_callable=AsyncMock
        ) as mock_add:
            await document_manager.add(test_file, "test_kb", method="multimodal")
            mock_add.assert_called_once_with(test_file, "test_kb")

    @pytest.mark.asyncio
    async def test_add_text_method(self, document_manager, tmp_path):
        """Test add calls add_text_only for text method."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with patch.object(
            document_manager, "add_text_only", new_callable=AsyncMock
        ) as mock_add:
            await document_manager.add(test_file, "test_kb", method="text")
            mock_add.assert_called_once_with(test_file, "test_kb")

    @pytest.mark.asyncio
    async def test_add_default_method_is_multimodal(self, document_manager, tmp_path):
        """Test add uses multimodal as default method."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")

        with patch.object(
            document_manager, "add_multimodal", new_callable=AsyncMock
        ) as mock_add:
            await document_manager.add(test_file, "test_kb")
            mock_add.assert_called_once()
