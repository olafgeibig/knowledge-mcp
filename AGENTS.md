# AGENTS.md - Agentic Coding Guidelines for knowledge-mcp

This file provides guidelines for AI agents operating in this repository.

## Project Overview

- **Project**: knowledge-mcp - A MCP server bridging specialized knowledge domains and AI assistants
- **Language**: Python 3.12
- **Build System**: hatchling
- **Dependency Manager**: uv

## Ignore Patterns

- data/
- archive/
- tmp/

## Build, Lint, and Test Commands

### Running Tests

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_config.py

# Run a single test
uv run pytest tests/test_config.py::test_successful_config_loading

# Run tests with verbose output
uv run pytest -v

# Run tests matching a pattern
uv run pytest -k "test_config"
```

### Development Commands

```bash
# Install in development mode
uv sync --group dev && uv pip install -e .

# Run the MCP server with uvx
uvx knowledge-mcp --config ./kbs/config.yaml mcp

# Run the interactive shell
uvx knowledge-mcp --config ./kbs/config.yaml shell

# Run the CLI directly
uv run knowledge-mcp

# Clean build artifacts
rm -rf build/ dist/ *.egg-info/
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

> **Note:** The `Makefile` provides convenient shortcuts (`make dev-install`, `make insp`, `make shell`, `make main`, `make test`, `make clean`) as an alternative to these uv commands.

### Using uv

```bash
# Add a dependency
uv add <package>

# Remove a dependency
uv remove <package>

# Sync dependencies
uv sync

# Use uv pip only if really necessary
uv pip install <package>
```

## Code Style Guidelines

### General Principles

- Use Python 3.12 features and syntax
- Follow PEP 8 style guide (Ruff is the primary linter/formatter)
- Use explicit code that clearly communicates intent over implicit, overly concise code
- Keep the Zen of Python in mind when making design decisions
- **Do not over-engineer solutions. Strive for simplicity and maintainability while still being efficient.**
- **Favor modularity, but avoid over-modularization.**

### Modular Design

- **Single Responsibility Principle:** Each module/file should have a well-defined, single responsibility
- **Reusable Components:** Develop reusable functions and classes, favoring composition over inheritance
- Keep functions and methods small and focused on a single task
- Organize code into logical packages and modules

### Imports

- Use absolute imports for package modules (e.g., `from knowledge_mcp.config import Config`)
- Group imports: stdlib, third-party, local
- Sort imports alphabetically within groups

### Data Structures

- Use list, dict, and set comprehensions when appropriate
- Use dataclasses for simple data containers
- Prefer `pathlib.Path` over `os.path` for file system operations

### Type Hints

- Always use type hints for function arguments and return values
- Use the most specific types possible
- Use modern Python 3.12 union syntax (`str | None`) over `Optional[str]`
- Example: `def func(arg: str) -> int: ...`

### Documentation

- Use docstrings for all public modules, functions, classes, and methods
- Follow Google-style docstrings
- Thoroughly explain purpose, parameters, return values, and exceptions raised
- Include usage examples where helpful
- Don't write too many inline comments

### Error Handling

- Use explicit exception handling
- Catch specific exceptions rather than using bare `except:` clauses
- Implement custom exception classes when needed
- Provide informative error messages
- Use `logging` for debugging and monitoring
- Chain exceptions with `from e` when re-raising

### Security

- Always consider security implications, especially with user inputs and external data

### Design Principles

- Prefer composition over inheritance
- Keep code modular and testable

## Testing Guidelines

- Use pytest for unit testing
- Use pytest-asyncio for async tests
- Aim for high test coverage (90% or higher)
- Test both common cases and edge cases
- Use fixtures for test setup/teardown
- Reset singleton states between tests (see test_config.py for example)
- Use `tmp_path` fixture for temporary file operations

Example test fixture pattern (from test_config.py):
```python
@pytest.fixture(autouse=True)
def reset_config_singleton():
    Config._instance = None
    Config._loaded = False
    yield
    # Cleanup after test
```

## Using Tools (MCP)

- Use the sequential-thinking tool to make a plan, breaking down complex problems into smaller steps
- Use context7 tool to look up the latest documentation of used libraries

## Configuration

- Configuration is handled via YAML files with Pydantic models
- Use environment variable substitution with `${VAR_NAME}` syntax
- Load .env files via python-dotenv

## Logging

- Use the module-level logger: `logger = logging.getLogger(__name__)`
- Configure logging via `logging.config.dictConfig`
- Use appropriate log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Project

- Project specifications are in specs directory:
  - prd.md: project requirements
  - plan.md: development plan
