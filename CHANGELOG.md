# Changelog

All notable changes to this project will be documented in this file.

## [0.4.2] - 2026-02-13

### Added
- Comprehensive test suite for main functionality
  - Tests for `documents.py` - DocumentManager, exceptions, text extraction
  - Tests for `knowledgebases.py` - load_kb_query_config, DEFAULT_QUERY_PARAMS
  - Tests for `rag.py` - RagManager exceptions, initialization, cleanup
- Test coverage tool (pytest-cov)
- AGENTS.md for agentic coding guidelines

### Changed
- Upgraded all dependencies to latest versions
- Updated lightrag-hku dependency to >=1.4.9.11
- Fixed test assertions for logging capture (caplog)
- Fixed test for env variable precedence (.env overrides env with override=True)
- Fixed test for Pydantic validation error handling

### Fixed
- test_dot_env_file_loading_and_precedence - numeric env var substitution
- test_error_main_config_file_not_found - absolute path handling
- test_error_pydantic_validation_missing_field - validation error triggering
- test_singleton_reload_configuration_updates_instance - log capture
- test_default_values_applied - DEBUG vs INFO assertion
- test_empty_env_file_specified_and_exists - log capture
- test_markitdown_extraction - skip if test files don't exist

### Dependencies
- Updated: fastmcp, lightrag-hku, markitdown, mineru, openai, pydantic, and many more
- Added: pytest-cov

## [0.4.1] - 2025-06-15

### Added
- Docker integration documentation
- Enhanced README for container usage

### Changed
- Configuration migration support for knowledge bases

## [0.4.0] - 2025-06-10

### Changed
- Upgraded to lightrag-hku 1.4.0
- Config file migration for knowledge bases

## [0.3.0] - 2025-05-20

### Added
- Configurable user prompts
- User prompt integration and logging

### Changed
- Updated default query parameters for LightRAG

## [0.2.0] - 2025-04-15

### Added
- Text-only parsing mode
- Shell features
- Cleanup after ingest

## [0.1.0] - 2025-03-01

### Added
- Initial release
- MCP server implementation
- Knowledge base management
- LightRAG integration
