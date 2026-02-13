"""Tests for knowledge_mcp.knowledgebases module."""

import pytest
from pathlib import Path
import tempfile
import yaml
import shutil

from knowledge_mcp.knowledgebases import (
    KnowledgeBaseError,
    KnowledgeBaseExistsError,
    KnowledgeBaseNotFoundError,
    DEFAULT_QUERY_PARAMS,
    load_kb_query_config,
)


class TestKnowledgeBaseExceptions:
    """Test custom exception classes."""

    def test_knowledge_base_error(self):
        """Test KnowledgeBaseError can be raised and caught."""
        with pytest.raises(KnowledgeBaseError):
            raise KnowledgeBaseError("Test error")

    def test_knowledge_base_exists_error(self):
        """Test KnowledgeBaseExistsError can be raised and caught."""
        with pytest.raises(KnowledgeBaseExistsError):
            raise KnowledgeBaseExistsError("KB already exists")

    def test_knowledge_base_not_found_error(self):
        """Test KnowledgeBaseNotFoundError can be raised and caught."""
        with pytest.raises(KnowledgeBaseNotFoundError):
            raise KnowledgeBaseNotFoundError("KB not found")


class TestDefaultQueryParams:
    """Test DEFAULT_QUERY_PARAMS constant."""

    def test_default_params_is_dict(self):
        """Test that DEFAULT_QUERY_PARAMS is a dictionary."""
        assert isinstance(DEFAULT_QUERY_PARAMS, dict)

    def test_default_params_contains_expected_keys(self):
        """Test DEFAULT_QUERY_PARAMS contains expected keys."""
        expected_keys = ["mode", "top_k", "response_type", "user_prompt"]
        for key in expected_keys:
            assert key in DEFAULT_QUERY_PARAMS, (
                f"{key} should be in DEFAULT_QUERY_PARAMS"
            )

    def test_default_mode_is_hybrid(self):
        """Test default mode is hybrid."""
        assert DEFAULT_QUERY_PARAMS["mode"] == "hybrid"

    def test_default_top_k(self):
        """Test default top_k value."""
        assert DEFAULT_QUERY_PARAMS["top_k"] == 40


class TestLoadKbQueryConfig:
    """Test load_kb_query_config function."""

    def test_load_config_with_defaults(self, tmp_path):
        """Test load_kb_query_config returns defaults when no config exists."""
        # Don't create config file

        result = load_kb_query_config(tmp_path)

        assert result == DEFAULT_QUERY_PARAMS

    def test_load_config_with_custom_values(self, tmp_path):
        """Test load_kb_query_config loads custom values."""
        config_path = tmp_path / "config.yaml"
        custom_config = {"mode": "local", "top_k": 10, "user_prompt": "Custom prompt"}
        with open(config_path, "w") as f:
            yaml.dump(custom_config, f)

        result = load_kb_query_config(tmp_path)

        assert result["mode"] == "local"
        assert result["top_k"] == 10
        assert result["user_prompt"] == "Custom prompt"
        # Defaults should be preserved for missing keys
        assert result["response_type"] == DEFAULT_QUERY_PARAMS["response_type"]

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test load_kb_query_config handles invalid YAML."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("invalid: yaml: content:")

        # Should return defaults on error
        result = load_kb_query_config(tmp_path)
        assert result == DEFAULT_QUERY_PARAMS
