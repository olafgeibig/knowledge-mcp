# knowledge-mcp: Specialized Knowledge Bases for AI Agents

## 1. Overview and Concept

**knowledge-mcp** is a MCP server designed to bridge the gap between specialized knowledge domains and AI assistants. It allows users to create, manage, and query dedicated knowledge bases, making this information accessible to AI agents through an MCP (Model Context Protocol) server interface.

The core idea is to empower AI assistants that are MCP clients (like Claude Desktop or IDEs like Windsurf) to proactively consult these specialized knowledge bases during their reasoning process (Chain of Thought), rather than relying solely on general semantic search against user prompts or broad web searches. This enables more accurate, context-aware responses when dealing with specific domains.

Key components:

*   **CLI Tool:** Provides a user-friendly command-line interface for managing knowledge bases (creating, deleting, adding/removing documents, configuring, searching).
*   **Knowledge Base Engine:** Leverages **LightRAG** to handle document processing, embedding, knowledge graph creation, and complex querying.
*   **MCP Server:** Exposes the search functionality of the knowledge bases via the FastMCP protocol, allowing compatible AI agents to query them directly.

## 2. About LightRAG

This project utilizes LightRAG ([HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)) as its core engine for knowledge base creation and querying. LightRAG is a powerful framework designed to enhance Large Language Models (LLMs) by integrating Retrieval-Augmented Generation (RAG) with knowledge graph techniques.

Key features of LightRAG relevant to this project:

*   **Document Processing Pipeline:** Ingests documents (PDF, Text, Markdown, DOCX), chunks them, extracts entities and relationships using an LLM, and builds both a knowledge graph and vector embeddings.
*   **Multiple Query Modes:** Supports various retrieval strategies (e.g., vector similarity, entity-centric, relationship-focused, hybrid) to find the most relevant context for a given query.
*   **Flexible Storage:** Can use different backends for storing key-value data, vectors, graph information, and document status (this project uses the default file-based storage).
*   **LLM/Embedding Integration:** Supports various providers like OpenAI (used in this project), Ollama, Hugging Face, etc.

By using LightRAG, `knowledge-mcp` benefits from advanced RAG capabilities that go beyond simple vector search.

## 3. Installation

Ensure you have Python 3.12 and `uv` installed.

1.  **Running the Tool:** After installing the package (e.g., using `uv pip install -e .`), you can run the CLI using `uvx`:
    ```bash
    # General command structure (use either --config or --base)
    uvx knowledge-mcp --config <path-to-your-config.yaml> <command> [arguments...]
    uvx knowledge-mcp --base <path-to-dir-containing-config.yaml> <command> [arguments...]

    # Example: Start interactive shell
    uvx knowledge-mcp --config <path-to-your-config.yaml> shell
    ```

2.  **Configure MCP Client:** To allow an MCP client (like Claude Desktop or Windsurf) to connect to this server, configure the client with the following settings. Replace the config path with the absolute path to your main `config.yaml`.
    ```json
    {
      "mcpServers": {
        "knowledge-mcp": {
          "command": "uvx",
          "args": [
            "knowledge-mcp",
            "--config",
            "<absolute-path-to-your-config.yaml>",
            "mcp"
          ]
        }
      }
    }
    ```

    **MCP Client Configuration (Docker)**  
    For clients like Claude Desktop, Cursor, Windsurf, etc.:

    **Spawn new container per request** (simplest):
    ```json
    {
      "mcpServers": {
        "knowledge-mcp": {
          "command": "docker",
          "args": [
            "run",
            "--rm",
            "-v",
            "/Users/yourusername/kb:/app/kb",
            "knowledge-mcp"
          ]
        }
      }
    }
    ```
    This runs `knowledge-mcp --base /app/kb mcp` inside the container.

    **Persistent server** (better for frequent use):
    ```bash
    docker run -d --name knowledge-mcp-server -v ~/kb:/app/kb knowledge-mcp
    ```
    Then configure your client to connect to the running instance if it supports that.

3.  **Set up configuration:**
    *   Copy `config.example.yaml` to `config.yaml`.
    *   Copy `.env.example` to `.env`.
    *   Edit `config.yaml` and `.env` to add your API keys (e.g., `OPENAI_API_KEY`) and adjust paths or settings as needed. The `knowledge_base.base_dir` in `config.yaml` specifies where your knowledge base directories will be created.

### Docker Setup & Usage

#### Build the image
```bash
docker build -t knowledge-mcp .
```

#### Quick commands (with mounted knowledge base)
Start persistent MCP server:
```bash
docker run -d --name knowledge-mcp -v ~/kb:/app/kb knowledge-mcp
```

Run one-off commands:
```bash
docker run -it --rm -v ~/kb:/app/kb knowledge-mcp shell
docker run -it --rm -v ~/kb:/app/kb knowledge-mcp create my-notes
docker run --rm -v ~/kb:/app/kb knowledge-mcp list
```

#### Using docker-compose (recommended)
Edit `docker-compose.yml` to point the volume to your real folder, then:
```bash
docker compose up -d
docker compose down
docker compose logs -f

docker compose run --rm knowledge-mcp create work-kb
docker compose run --rm knowledge-mcp shell
```

#### Required host setup
```bash
mkdir -p ~/kb
cp config.example.yaml ~/kb/config.yaml
cp .env.example ~/kb/.env
```
Then edit:
- `~/kb/config.yaml` → set `knowledge_base.base_dir: /app/kb`
- `~/kb/.env` → add your `OPENAI_API_KEY`

## 4. Configuration

Configuration is managed via YAML files.

When using Docker, set `knowledge_base.base_dir: /app/kb` in your config.yaml (the one you mount from your host).

1.  **Main Configuration (`config.yaml`):** Defines global settings like the knowledge base directory (`knowledge_base.base_dir`), LightRAG parameters (LLM provider/model, embedding provider/model, API keys via `${ENV_VAR}` substitution), and logging settings. Refer to `config.example.yaml` for the full structure and available options.

    ```yaml
    knowledge_base:
      base_dir: ./kbs

    lightrag:
      llm:
        provider: "openai"
        model_name: "gpt-4.1-nano"
        api_key: "${OPENAI_API_KEY}"
        # ... other LLM settings
      embedding:
        provider: "openai"
        model_name: "text-embedding-3-small"
        api_key: "${OPENAI_API_KEY}"
        # ... other embedding settings
      embedding_cache:
        enabled: true
        similarity_threshold: 0.90

    logging:
      level: "INFO"
      # ... logging settings

    env_file: .env # path to .env file
    ```

2.  **Knowledge Base Specific Configuration (`<base_dir>/<kb_name>/config.yaml`):** Contains parameters specific to querying *that* knowledge base, such as the LightRAG query `mode` (default: "hybrid"), `top_k` results (default: 40), context token limits, `text_only` parsing mode, and `user_prompt` for response formatting. This file is automatically created with defaults when a KB is created and can be viewed/edited using the `config` CLI command.

3.  **Knowledge Base Directory Structure:** When you create knowledge bases, they are stored within the directory specified by `knowledge_base.base_dir` in your main `config.yaml`. The structure typically looks like this:

    ```
    <base_dir>/              # Main directory, contains a set of knowledge bases
    ├── config.yaml          # Main application configuration (copied from config.example.yaml)
    ├── .env                 # Environment variables referenced in config.yaml
    ├── kbmcp.log
    ├── knowledge_base_1/    # Directory for the first KB
    │   ├── config.yaml      # KB-specific configuration (query parameters)
    │   ├── <storage_files>  # The LightRAG storage files
    └── knowledge_base_2/    # Directory for the second KB
        ├── config.yaml
        ├── <storage_files>
    ```

## 5. New Features

### 5.1 Text-Only Document Parsing

By default, knowledge-mcp processes documents using both text content and metadata (like document structure, formatting, etc.). You can now configure knowledge bases to use **text-only parsing** for faster processing and reduced token usage.

**Benefits:**
- Faster document processing
- Lower LLM token consumption
- Simplified content extraction
- Better performance with large document collections

**Configuration:**
Add `text_only: true` to your knowledge base's `config.yaml`:

```yaml
# In <base_dir>/<kb_name>/config.yaml
description: "My knowledge base with text-only parsing"
mode: "hybrid"
top_k: 40
text_only: true  # Enable text-only parsing
```

**Usage:**
```bash
# Create a new KB and configure it for text-only parsing
knowledge-mcp --config config.yaml create my_text_kb
knowledge-mcp --config config.yaml config my_text_kb edit
# Add text_only: true to the config file

# Add documents - they will be processed with text-only parsing
knowledge-mcp --config config.yaml add my_text_kb ./documents/
```

### 5.2 Configurable User Prompts

You can now customize how the LLM formats and structures its responses for each knowledge base by configuring a `user_prompt`. This allows you to tailor the response style to match your specific use case.

**Benefits:**
- Consistent response formatting across queries
- Domain-specific response styles
- Better integration with downstream applications
- Improved user experience

**Configuration:**
Add a `user_prompt` field to your knowledge base's `config.yaml`. The prompt supports multi-line YAML syntax:

```yaml
# In <base_dir>/<kb_name>/config.yaml
description: "Technical documentation KB"
mode: "hybrid"
top_k: 30
max_entity_tokens: 2000
max_relation_tokens: 3000
max_total_tokens: 6000
response_type: "Multiple Paragraphs"
user_prompt: |
  Please provide a comprehensive analysis with citations.
  - Focus on actionable information
  
  Keep your response clear, concise, and well-organized.
```

**Example Configurations:**

1. **Business-Focused Format:**
```yaml
user_prompt: |
  Structure your response for business stakeholders:
  
  **Executive Summary** (2-3 sentences)
  Brief overview of the main points and business impact.
  
  **Key Findings**
  • Most critical insights
  • Relevant metrics or data points
  • Risk factors or opportunities
  
  **Recommendations**
  • Specific actionable steps
  • Priority levels (High/Medium/Low)
  • Expected outcomes
```

2. **Technical Documentation Style:**
```yaml
user_prompt: |
  You are a technical documentation expert. Please structure your response with:
  
  1. **Context**: Brief background on the topic
  2. **Implementation**: Step-by-step technical details
  3. **Best Practices**: Recommended approaches and common pitfalls
  4. **Examples**: Concrete code examples or use cases where applicable
  
  Use clear headings, bullet points, and code blocks for readability.
```

3. **Academic Research Style:**
```yaml
user_prompt: |
  Please provide a scholarly response that includes:
  
  • **Introduction**: Context and scope of the topic
  • **Analysis**: Critical examination of key concepts and evidence
  • **Synthesis**: How different pieces of information connect
  • **Conclusion**: Main findings and implications
  
  Support your points with specific references from the knowledge base.
```

**Usage:**
```bash
# Configure user prompt for an existing KB
knowledge-mcp --config config.yaml config my_kb edit
# Add your user_prompt configuration to the YAML file

# Query the KB - responses will follow your configured format
knowledge-mcp --config config.yaml query my_kb "What are the main concepts?"
```

**Notes:**
- User prompts are applied automatically to all queries for that knowledge base
- Leave `user_prompt` empty or omit it to use default LLM behavior
- Changes take effect immediately - no need to rebuild the knowledge base
- Backward compatible - existing knowledge bases continue to work without modification

## 6. Usage (CLI)

The primary way to interact with `knowledge-mcp` is through its CLI, accessed via the `knowledge-mcp` command (if installed globally or via `uvx knowledge-mcp` within the activated venv).

**All commands require either `--config <path-to-config.yaml>` or `--base <directory-containing-config.yaml>` (e.g. Docker uses `--base /app/kb`).**

```bash
uv run knowledge-mcp --config /path/to/config.yaml shell
```
**Available Commands (Interactive Shell):**

| Command  | Description                                                                 | Arguments                                                                      |
| :------- | :-------------------------------------------------------------------------- | :----------------------------------------------------------------------------- |
| `create` | Creates a new knowledge base directory and initializes its structure.       | `<name>`: Name of the KB.<br> `["description"]`: Optional description.         |
| `delete` | Deletes an existing knowledge base directory and all its contents.            | `<name>`: Name of the KB to delete.                                          |
| `list`   | Lists all available knowledge bases and their descriptions.                 | N/A                                                                            |
| `add`    | Adds a document: processes, chunks, embeds, stores in the specified KB.     | `<kb_name>`: Target KB.<br>`<file_path>`: Path to the document file.          |
| `remove` | Removes a document and its associated data from the KB by its ID.           | `<kb_name>`: Target KB.<br>`<doc_id>`: ID of the document to remove.         |
| `config` | Manages the KB-specific `config.yaml`. Shows content or opens in editor.    | `<kb_name>`: Target KB.<br>`[show|edit]`: Subcommand (show default).          |
| `query`  | Searches the specified knowledge base using LightRAG.                     | `<kb_name>`: Target KB.<br>`<query_text>`: Your search query text.             |
| `clear`  | Clears the terminal screen.                                                 | N/A                                                                            |
| `exit`   | Exits the interactive shell.                                                | N/A                                                                            |
| `EOF`    | (Ctrl+D) Exits the interactive shell.                                       | N/A                                                                            |
| `help`   | Shows available commands and their usage within the shell.                  | `[command]` (Optional command name)                                            |

**Example (Direct CLI):**

```bash
# Create a knowledge base named 'my_docs'
knowledge-mcp --config config.yaml create my_docs

# Add a document to it
knowledge-mcp --config config.yaml add my_docs ./path/to/mydocument.pdf

# Search the knowledge base
knowledge-mcp --config config.yaml query my_docs "What is the main topic?"

# Start the interactive shell
knowledge-mcp --config config.yaml shell

(kbmcp) list
(kbmcp) query my_docs "Another query"
(kbmcp) exit
```

## 7. Development
1. Project Decisions
*   **Tech Stack:** Python 3.12, uv (dependency management), hatchling (build system), pytest (testing).
*   **Setup:** Follow the installation steps, ensuring you install with `uv pip install -e ".[dev]"`.
*   **Code Style:** Adheres to PEP 8.
*   **Testing:** Run tests using `uvx test` or `pytest`.
*   **Dependencies:** Managed in `pyproject.toml`. Use `uv pip install <package>` to add and `uv pip uninstall <package>` to remove dependencies, updating `pyproject.toml` accordingly.
*   **Scripts:** Common tasks might be defined under `[project.scripts]` in `pyproject.toml`.
*   **Release:** Build `hatch build` and then `twine upload dist/*`.

2. **Test with uvx**
```
    "knowledge-mcp": {
      "command": "uvx",
      "args": [
        "--project",
        "/path/to/knowledge-mcp",
        "knowledge-mcp",
        "--config",
        "/path/to/knowledge-mcp/kbs/config.yaml",
        "mcp"
      ]
    }
```
3. Test with MCP Inspector
```
npx @modelcontextprotocol/inspector uv "run knowledge-mcp --config /path/to/config.yaml mcp"
```
or
```
npx @modelcontextprotocol/inspector uvx --project . knowledge-mcp "--config ./kbs/config.yaml mcp
```
4. Convenience dev scripts
Assumes a local config file at `./kbs/config.yaml`
* `uvx shell` - Starts the interactive shell
* `uvx insp` - Starts the MCP Inspector