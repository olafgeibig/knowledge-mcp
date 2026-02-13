# knowledge-mcp Docker image
# Run with: docker run -v ~/kb:/app/kb knowledge-mcp
# Override: docker run -v ~/kb:/app/kb knowledge-mcp shell
FROM python:3.12-slim

WORKDIR /app

COPY . .

# Use uv for much faster, lockfile-driven installs
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache -e .

# Default: run MCP server with base dir /app/kb (override with: shell, create mykb, list, etc.)
# Use the installed package via Python module so we don't depend on script PATH
ENTRYPOINT ["python", "-m", "knowledge_mcp.cli", "--base", "/app/kb"]
