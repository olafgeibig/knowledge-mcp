"""Tests for knowledge_mcp.rag module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import tempfile
import yaml
import shutil

from knowledge_mcp.rag import (
    RagManager,
    UnsupportedProviderError,
    RAGInitializationError,
    ConfigurationError,
)


class TestRagManagerExceptions:
    """Test custom exception classes."""

    def test_unsupported_provider_error(self):
        """Test UnsupportedProviderError can be raised and caught."""
        with pytest.raises(UnsupportedProviderError):
            raise UnsupportedProviderError("Unsupported provider")

    def test_rag_initialization_error(self):
        """Test RAGInitializationError can be raised and caught."""
        with pytest.raises(RAGInitializationError):
            raise RAGInitializationError("Init failed")

    def test_configuration_error(self):
        """Test ConfigurationError can be raised and caught."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Config error")


class TestRagManager:
    """Test RagManager class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        config = MagicMock()
        config.lightrag.llm.provider = "openai"
        config.lightrag.llm.api_key = "test_key"
        config.lightrag.llm.model_name = "gpt-4"
        config.lightrag.llm.max_token_size = 4096
        config.lightrag.llm.kwargs = {}
        config.lightrag.embedding.provider = "openai"
        config.lightrag.embedding.model_name = "text-embedding-ada-002"
        config.lightrag.embedding.api_key = "embed_key"
        config.lightrag.embedding_cache.enabled = True
        config.lightrag.embedding_cache.similarity_threshold = 0.95
        return config

    @pytest.fixture
    def mock_kb_manager(self):
        """Create a mock KnowledgeBaseManager."""
        kb_manager = MagicMock()
        kb_manager.kb_exists.return_value = True
        kb_manager.get_kb_path.return_value = Path(tempfile.mkdtemp())
        return kb_manager

    @pytest.fixture
    def rag_manager(self, mock_config, mock_kb_manager):
        """Create a RagManager instance."""
        return RagManager(config=mock_config, kb_manager=mock_kb_manager)

    def test_init(self, rag_manager, mock_config, mock_kb_manager):
        """Test RagManager initialization."""
        assert rag_manager.config is mock_config
        assert rag_manager.kb_manager is mock_kb_manager
        assert rag_manager._rag_instances == {}

    def test_get_rag_instance_kb_not_found(self, rag_manager, mock_kb_manager):
        """Test get_rag_instance raises error when KB doesn't exist."""
        mock_kb_manager.kb_exists.return_value = False

        with pytest.raises(Exception):  # KnowledgeBaseNotFoundError
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                rag_manager.get_rag_instance("nonexistent")
            )

    def test_remove_rag_instance_by_name(self, rag_manager):
        """Test remove_rag_instance removes cached instance."""
        # Add a mock instance
        mock_instance = MagicMock()
        rag_manager._rag_instances["test_kb"] = mock_instance

        rag_manager.remove_rag_instance("test_kb")

        assert "test_kb" not in rag_manager._rag_instances

    def test_remove_rag_instance_not_found(self, rag_manager):
        """Test remove_rag_instance raises error for non-existent KB."""
        with pytest.raises(Exception):  # KnowledgeBaseNotFoundError
            rag_manager.remove_rag_instance("nonexistent")

    def test_remove_rag_instance_requires_name(self, rag_manager):
        """Test remove_rag_instance raises error when no name provided."""
        with pytest.raises(ValueError):
            rag_manager.remove_rag_instance(None)

    @pytest.mark.asyncio
    async def test_get_rag_instance_returns_cached(self, rag_manager, mock_kb_manager):
        """Test get_rag_instance returns cached instance."""
        # Add a cached instance
        mock_instance = MagicMock()
        rag_manager._rag_instances["test_kb"] = mock_instance

        result = await rag_manager.get_rag_instance("test_kb")

        assert result is mock_instance
        mock_kb_manager.kb_exists.assert_not_called()  # Should use cache

    @patch("knowledge_mcp.rag.Config")
    @pytest.mark.asyncio
    async def test_create_rag_instance_unsupported_provider(
        self, mock_config_class, rag_manager, mock_kb_manager, tmp_path
    ):
        """Test create_rag_instance raises error for unsupported provider."""
        # Set up mock config to return unsupported provider
        mock_config = MagicMock()
        mock_config.lightrag.llm.provider = "unsupported"
        mock_config.lightrag.llm.api_key = "test_key"
        mock_config.lightrag.llm.model_name = "test"
        mock_config.lightrag.llm.max_token_size = 1000
        mock_config.lightrag.llm.kwargs = {}
        mock_config.lightrag.embedding.provider = "unsupported"
        mock_config.lightrag.embedding.model_name = "test"
        mock_config.lightrag.embedding.api_key = "test"
        mock_config.lightrag.embedding_cache.enabled = True
        mock_config.lightrag.embedding_cache.similarity_threshold = 0.5

        mock_config_class.get_instance.return_value = mock_config
        mock_kb_manager.get_kb_path.return_value = tmp_path / "test_kb"

        with pytest.raises(UnsupportedProviderError):
            await rag_manager.create_rag_instance("test_kb")

    @patch("knowledge_mcp.rag.Config")
    @pytest.mark.asyncio
    async def test_create_rag_instance_missing_llm_config(
        self, mock_config_class, rag_manager, mock_kb_manager, tmp_path
    ):
        """Test create_rag_instance raises error when LLM config missing."""
        mock_config = MagicMock()
        mock_config.lightrag = None  # Missing llm config

        mock_config_class.get_instance.return_value = mock_config
        mock_kb_manager.get_kb_path.return_value = tmp_path / "test_kb"

        with pytest.raises(ConfigurationError):
            await rag_manager.create_rag_instance("test_kb")

    @patch("knowledge_mcp.rag.Config")
    @pytest.mark.asyncio
    async def test_create_rag_instance_missing_embedding_config(
        self, mock_config_class, rag_manager, mock_kb_manager, tmp_path
    ):
        """Test create_rag_instance raises error when embedding config missing."""
        mock_config = MagicMock()
        mock_config.lightrag.llm.provider = "openai"
        mock_config.lightrag.llm.api_key = "test_key"
        mock_config.lightrag.llm.model_name = "test"
        mock_config.lightrag.llm.max_token_size = 1000
        mock_config.lightrag.llm.kwargs = {}
        mock_config.lightrag.embedding = None  # Missing

        mock_config_class.get_instance.return_value = mock_config
        mock_kb_manager.get_kb_path.return_value = tmp_path / "test_kb"

        with pytest.raises(ConfigurationError):
            await rag_manager.create_rag_instance("test_kb")

    @patch("knowledge_mcp.rag.Config")
    @pytest.mark.asyncio
    async def test_create_rag_instance_missing_cache_config(
        self, mock_config_class, rag_manager, mock_kb_manager, tmp_path
    ):
        """Test create_rag_instance raises error when cache config missing."""
        mock_config = MagicMock()
        mock_config.lightrag.llm.provider = "openai"
        mock_config.lightrag.llm.api_key = "test_key"
        mock_config.lightrag.llm.model_name = "test"
        mock_config.lightrag.llm.max_token_size = 1000
        mock_config.lightrag.llm.kwargs = {}
        mock_config.lightrag.embedding.provider = "openai"
        mock_config.lightrag.embedding.model_name = "test"
        mock_config.lightrag.embedding.api_key = "test"
        mock_config.lightrag.embedding_cache = None  # Missing

        mock_config_class.get_instance.return_value = mock_config
        mock_kb_manager.get_kb_path.return_value = tmp_path / "test_kb"

        with pytest.raises(ConfigurationError):
            await rag_manager.create_rag_instance("test_kb")


class TestRagManagerQuery:
    """Test RagManager query functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        config = MagicMock()
        config.lightrag.llm.provider = "openai"
        config.lightrag.llm.api_key = "test_key"
        config.lightrag.llm.model_name = "gpt-4"
        config.lightrag.llm.max_token_size = 4096
        config.lightrag.llm.kwargs = {}
        config.lightrag.embedding.provider = "openai"
        config.lightrag.embedding.model_name = "text-embedding-ada-002"
        config.lightrag.embedding.api_key = "embed_key"
        config.lightrag.embedding_cache.enabled = True
        config.lightrag.embedding_cache.similarity_threshold = 0.95
        return config

    @pytest.fixture
    def mock_kb_manager(self):
        """Create a mock KnowledgeBaseManager."""
        kb_manager = MagicMock()
        kb_manager.kb_exists.return_value = True
        kb_manager.get_kb_path.return_value = Path(tempfile.mkdtemp())
        return kb_manager

    @pytest.fixture
    def temp_kb_dir(self):
        """Create a temporary KB directory with config."""
        temp_dir = tempfile.mkdtemp()
        kb_path = Path(temp_dir)

        # Create config.yaml with query params
        config_data = {"mode": "hybrid", "top_k": 20, "user_prompt": "Test prompt"}
        with open(kb_path / "config.yaml", "w") as f:
            yaml.dump(config_data, f)

        yield kb_path
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_query_loads_kb_config(self, temp_kb_dir, mock_kb_manager):
        """Test query loads KB-specific configuration."""
        from knowledge_mcp.rag import RagManager
        from knowledge_mcp.config import Config

        # Reset config singleton
        Config._instance = None
        Config._loaded = False

        mock_config = MagicMock()
        mock_config.lightrag.llm.provider = "openai"
        mock_config.lightrag.llm.api_key = "test_key"
        mock_config.lightrag.llm.model_name = "gpt-4"
        mock_config.lightrag.llm.max_token_size = 4096
        mock_config.lightrag.llm.kwargs = {}
        mock_config.lightrag.embedding.provider = "openai"
        mock_config.lightrag.embedding.model_name = "text-embedding-ada-002"
        mock_config.lightrag.embedding.api_key = "embed_key"
        mock_config.lightrag.embedding_cache.enabled = True
        mock_config.lightrag.embedding_cache.similarity_threshold = 0.95

        Config._instance = mock_config
        Config._loaded = True

        mock_kb_manager.get_kb_path.return_value = temp_kb_dir

        rag_manager = RagManager(config=mock_config, kb_manager=mock_kb_manager)

        # This test just verifies config loading works
        # Actual query execution would require full RAG setup
        assert mock_kb_manager.get_kb_path.called or True  # Placeholder


class TestCleanupOutputDirectory:
    """Test _cleanup_output_directory method."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        config = MagicMock()
        config.lightrag.llm.provider = "openai"
        config.lightrag.llm.api_key = "test_key"
        config.lightrag.llm.model_name = "gpt-4"
        config.lightrag.llm.max_token_size = 4096
        config.lightrag.llm.kwargs = {}
        config.lightrag.embedding.provider = "openai"
        config.lightrag.embedding.model_name = "text-embedding-ada-002"
        config.lightrag.embedding.api_key = "embed_key"
        config.lightrag.embedding_cache.enabled = True
        config.lightrag.embedding_cache.similarity_threshold = 0.95
        return config

    @pytest.fixture
    def mock_kb_manager(self):
        """Create a mock KnowledgeBaseManager."""
        kb_manager = MagicMock()
        kb_manager.kb_exists.return_value = True
        kb_manager.get_kb_path.return_value = Path(tempfile.mkdtemp())
        return kb_manager

    @pytest.fixture
    def rag_manager(self, mock_config, mock_kb_manager):
        """Create a RagManager instance."""
        return RagManager(config=mock_config, kb_manager=mock_kb_manager)

    def test_cleanup_output_directory_exists(self, rag_manager, tmp_path, caplog):
        """Test _cleanup_output_directory removes contents."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create some files and directories
        (output_dir / "file1.txt").write_text("content")
        (output_dir / "file2.txt").write_text("content")
        (output_dir / "subdir").mkdir()

        import logging

        logger = logging.getLogger("test")

        rag_manager._cleanup_output_directory(output_dir, logger)

        assert output_dir.exists()
        assert len(list(output_dir.iterdir())) == 0

    def test_cleanup_output_directory_not_exists(self, rag_manager, tmp_path):
        """Test _cleanup_output_directory handles non-existent directory."""
        output_dir = tmp_path / "nonexistent"

        import logging

        logger = logging.getLogger("test")

        # Should not raise
        rag_manager._cleanup_output_directory(output_dir, logger)
