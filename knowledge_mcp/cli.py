import argparse
import asyncio
import logging
import logging.config
import sys
import warnings
from pathlib import Path

# Updated relative imports
from knowledge_mcp.config import Config
from knowledge_mcp.knowledgebases import (
    KnowledgeBaseError,
    KnowledgeBaseExistsError,
    KnowledgeBaseManager,
)
from knowledge_mcp.rag import (
    ConfigurationError,
    RagManager,
    RAGInitializationError,
    RAGManagerError,
)
from knowledge_mcp.shell import Shell
from knowledge_mcp.mcp_server import MCP

logger = logging.getLogger(__name__)

def initialize_components(config: Config) -> tuple[KnowledgeBaseManager, RagManager]:
    """Initialize and return manager instances."""
    logger.info("Initializing components...")
    kb_manager = KnowledgeBaseManager(config)
    
    # Migrate existing knowledge base config files to new format
    logger.info("Checking for knowledge base config migrations...")
    migration_results = kb_manager.migrate_all_configs()
    if migration_results:
        migrated_count = sum(1 for migrated in migration_results.values() if migrated)
        if migrated_count > 0:
            logger.info(f"Successfully migrated {migrated_count} knowledge base config file(s)")
        else:
            logger.info("All knowledge base config files are up to date")
    
    rag_manager = RagManager(config, kb_manager)
    logger.info("Components initialized.")
    return kb_manager, rag_manager

def run_mcp_mode():
    """Runs the application in server mode."""
    logger.info("Starting in serve mode...")
    config = Config.get_instance() # Get the loaded config
    kb_manager, rag_manager = initialize_components(config)
    
    # Instantiate the MCP service class
    logger.info("Instantiating KnowledgeMCP service...")
    mcp = MCP(rag_manager, kb_manager)


def run_shell_mode():
    """Runs the interactive management shell."""
    logger.info("Starting in management shell...")
    kb_manager, rag_manager = initialize_components(Config.get_instance())

    # Instantiate and run the interactive shell
    shell = Shell(kb_manager, rag_manager) # Use the renamed Shell class
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        # Catch Ctrl+C during cmdloop if needed, though EOF (Ctrl+D) is handled by the shell
        print("\nExiting management shell (KeyboardInterrupt).")
    finally:
        # Stop background server if necessary
        logger.info("Stopping background server (placeholder)...")
        logger.info("Manage mode finished.")


def execute_query(kb_name: str, query_text: str, rag_manager=None, output_file=None, async_task_runner=None):
    """Execute a query against a knowledge base.
    
    Args:
        kb_name: Name of the knowledge base
        query_text: Query text to search for
        rag_manager: Optional RAG manager instance (if None, will create one)
        output_file: Optional file object for output (defaults to sys.stdout)
        async_task_runner: Optional function to run async tasks (for shell integration)
    
    Returns:
        Query result string
    
    Raises:
        Exception: If query fails
    """
    if output_file is None:
        output_file = sys.stdout
    
    # Initialize components if rag_manager not provided
    if rag_manager is None:
        kb_manager, rag_manager = initialize_components(Config.get_instance())
    
    print(f"\nQuerying KB '{kb_name}' with: \"{query_text}\"", file=output_file)
    print(" [running query] ...", end="", flush=True, file=output_file)
    
    try:
        if async_task_runner:
            # For shell - use the provided async task runner
            result = async_task_runner(rag_manager.query(kb_name, query_text))
        else:
            # For CLI - create and manage our own event loop with aggressive cleanup
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Set a custom exception handler to suppress background task warnings
            def exception_handler(loop, context):
                # Suppress specific LightRAG background task errors
                exception = context.get('exception')
                if isinstance(exception, RuntimeError) and 'Event loop is closed' in str(exception):
                    return  # Ignore these errors
                if isinstance(exception, RuntimeError) and 'no running event loop' in str(exception):
                    return  # Ignore these errors
                # Log other exceptions normally
                logger.debug(f"Async exception: {context}")
            
            loop.set_exception_handler(exception_handler)
            
            try:
                result = loop.run_until_complete(rag_manager.query(kb_name, query_text))
                
            except Exception as e:
                # Force cleanup of RAG instance on error to prevent background tasks
                logger.debug("Cleaning up RAG instance after query error")
                if hasattr(rag_manager, '_rag_instances') and kb_name in rag_manager._rag_instances:
                    del rag_manager._rag_instances[kb_name]
                    logger.debug(f"Removed cached RAG instance for {kb_name} due to error")
                raise  # Re-raise the exception
                
            finally:
                # Always cleanup RAG instance after query to prevent background tasks
                logger.debug("Final cleanup of RAG instance")
                if hasattr(rag_manager, '_rag_instances') and kb_name in rag_manager._rag_instances:
                    del rag_manager._rag_instances[kb_name]
                    logger.debug(f"Final removal of cached RAG instance for {kb_name}")
                    
                _cleanup_event_loop(loop)
        
        print(" [done]", file=output_file)
        print("\n--- Query Result ---", file=output_file)
        print(result, file=output_file)
        print("--- End Result ---", file=output_file)
        return result
        
    except Exception as e:
        print(" [failed]", file=output_file)
        print(f"\nError querying KB '{kb_name}': {e}", file=output_file)
        logger.error(f"Query failed for {kb_name}: {e}")
        raise


def _cleanup_event_loop(loop):
    """Clean up an event loop and suppress warnings from background tasks."""
    if loop and not loop.is_closed():
        try:
            # Suppress warnings during cleanup
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", DeprecationWarning)
                
                # Get all pending tasks
                pending = asyncio.all_tasks(loop)
                logger.debug(f"Found {len(pending)} pending tasks to cleanup")
                
                # Cancel all pending tasks
                for task in pending:
                    if not task.done():
                        task.cancel()
                        logger.debug(f"Cancelled task: {task}")
                
                # Give tasks time to cancel gracefully
                if pending:
                    try:
                        # Wait for cancellation with a longer timeout for LightRAG tasks
                        loop.run_until_complete(
                            asyncio.wait_for(
                                asyncio.gather(*pending, return_exceptions=True),
                                timeout=2.0  # Increased timeout for LightRAG cleanup
                            )
                        )
                    except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
                        logger.debug(f"Task cleanup timeout or error (expected): {e}")
                        pass  # Expected for cancelled tasks
                
                # Shutdown async generators and executors
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception as e:
                    logger.debug(f"Error shutting down async generators: {e}")
                    pass
                    
                try:
                    loop.run_until_complete(loop.shutdown_default_executor())
                except Exception as e:
                    logger.debug(f"Error shutting down default executor: {e}")
                    pass
                
                # Close the loop
                loop.close()
                logger.debug("Event loop closed successfully")
                
        except Exception as e:
            logger.debug(f"Error during event loop cleanup: {e}")
            # Force close on any error
            try:
                if not loop.is_closed():
                    loop.close()
            except Exception:
                pass
    
    # Clear the event loop policy to prevent issues with background tasks
    try:
        asyncio.set_event_loop(None)
    except Exception:
        pass
    
    # Comprehensive stderr suppression for LightRAG background tasks
    import time
    import threading
    import sys
    import os as os_module
    
    # Additional global suppression of asyncio warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='asyncio')
    warnings.filterwarnings('ignore', message='.*Event loop is closed.*')
    warnings.filterwarnings('ignore', message='.*no running event loop.*')
    
    # OS-level stderr suppression for background tasks
    def comprehensive_suppression():
        # Save original stderr
        original_stderr = sys.stderr
        original_stderr_fd = os_module.dup(2)  # Duplicate stderr file descriptor
        
        try:
            # Redirect stderr to devnull at both Python and OS level
            devnull = open(os_module.devnull, 'w')
            sys.stderr = devnull
            os_module.dup2(devnull.fileno(), 2)  # Redirect OS-level stderr
            
            # Give background tasks time to finish
            time.sleep(0.8)  # Longer wait for complete cleanup
            
        finally:
            # Restore stderr
            try:
                os_module.dup2(original_stderr_fd, 2)  # Restore OS-level stderr
                sys.stderr = original_stderr
                os_module.close(original_stderr_fd)
                devnull.close()
            except Exception:
                pass  # Ignore restoration errors
    
    # Run comprehensive suppression in background thread
    suppression_thread = threading.Thread(target=comprehensive_suppression, daemon=True)
    suppression_thread.start()
    suppression_thread.join(timeout=1.5)  # Wait a bit longer


def run_query_mode(kb_name: str, query_text: str):
    """Runs a single query against the specified knowledge base (CLI mode)."""
    logger.info(f"Running query against KB '{kb_name}': {query_text}")

    try:
        execute_query(kb_name, query_text)  # No async_task_runner = use CLI mode
    except Exception:
        sys.exit(1)


def run_create_mode(name: str, description: str | None = None) -> None:
    """Create a new knowledge base (CLI one-off)."""
    kb_manager, rag_manager = initialize_components(Config.get_instance())
    try:
        kb_manager.create_kb(name, description=description)
        print(f"Initializing RAG instance for '{name}'...")
        asyncio.run(rag_manager.create_rag_instance(name))
        print(f"Knowledge base '{name}' created and RAG instance initialized successfully.")
    except KnowledgeBaseExistsError:
        print(f"Error: Knowledge base '{name}' already exists.")
        sys.exit(1)
    except KnowledgeBaseError as e:
        print(f"Error creating knowledge base '{name}': {e}", file=sys.stderr)
        sys.exit(1)
    except (RAGInitializationError, ConfigurationError, RAGManagerError) as e:
        logger.warning("KB created but RAG initialization failed: %s", e)
        print(f"Warning: Knowledge base '{name}' created, but RAG initialization failed: {e}")
        print("You may need to configure LLM/Embedding settings before using this KB.")


def run_list_mode() -> None:
    """List all knowledge bases (CLI one-off)."""
    kb_manager, _ = initialize_components(Config.get_instance())
    try:
        kbs_with_desc = asyncio.run(kb_manager.list_kbs())
    except Exception as e:
        logger.exception("Error listing knowledge bases")
        print(f"Error listing knowledge bases: {e}")
        sys.exit(1)
    if not kbs_with_desc:
        print("No knowledge bases found.")
        return
    print("Available knowledge bases:")
    max_len = max(len(n) for n in kbs_with_desc.keys())
    for n, desc in kbs_with_desc.items():
        print(f"- {n:<{max_len}} : {desc}")

def main():
    """Main entry point for the application."""
    # Phase 1: extract --config and --base first so they work the same before the subcommand
    # (avoids argparse consuming a path like /app/kb as the subcommand when invoked as
    # uvx knowledge-mcp --base /app/kb shell)
    pre_parser = argparse.ArgumentParser()
    pre_parser.add_argument("-c", "--config", type=str, default=None)
    pre_parser.add_argument("--base", "--kb-dir", dest="base", type=str, default=None)
    pre_args, unknown = pre_parser.parse_known_args()

    # Normalize argv: optionals first, then subcommand and its args
    argv = [sys.argv[0]]
    if pre_args.config is not None:
        argv.extend(["--config", pre_args.config])
    if pre_args.base is not None:
        argv.extend(["--base", pre_args.base])
    argv.extend(unknown)

    parser = argparse.ArgumentParser(description="Knowledge Base MCP Server and Management Shell")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        required=False,
        help="Path to the configuration file (config.yaml). If --base is also provided, --config takes precedence.",
    )
    parser.add_argument(
        "--base", "--kb-dir",
        type=str,
        required=False,
        dest="base",
        help="Base directory containing config.yaml (makes --config optional)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help='Available modes: mcp, shell')

    # MCP command
    parser_mcp = subparsers.add_parser("mcp", help="Run the MCP server")
    parser_mcp.set_defaults(func=run_mcp_mode)

    # Shell command
    parser_shell = subparsers.add_parser("shell", help="Run the interactive management shell")
    parser_shell.set_defaults(func=run_shell_mode)

    # Query command
    parser_query = subparsers.add_parser("query", help="Query a knowledge base")
    parser_query.add_argument("kb_name", help="Name of the knowledge base to query")
    parser_query.add_argument("query_text", help="Query text to search for")
    parser_query.set_defaults(func=lambda: run_query_mode(args.kb_name, args.query_text))

    # Create command (one-off, for Docker etc.)
    parser_create = subparsers.add_parser("create", help="Create a new knowledge base")
    parser_create.add_argument("name", help="Name of the knowledge base")
    parser_create.add_argument(
        "description",
        nargs="?",
        default=None,
        help="Optional description",
    )
    parser_create.set_defaults(
        func=lambda: run_create_mode(args.name, getattr(args, "description", None))
    )

    # List command (one-off, for Docker etc.)
    parser_list = subparsers.add_parser("list", help="List all knowledge bases")
    parser_list.set_defaults(func=run_list_mode)

    try:
        args = parser.parse_args(argv[1:])
    except SystemExit as e:
        # When user passes a path as first arg (e.g. uvx knowledge-mcp /app/kb shell),
        # argparse reports "invalid choice: '/app/kb'". Hint to use --base or --config.
        if e.code != 0 and unknown and len(unknown) >= 1:
            val = unknown[0]
            if "/" in val or "\\" in val:
                print(
                    "knowledge-mcp: Use --base <dir> or --config <file> before the command, "
                    "e.g. knowledge-mcp --base /app/kb shell",
                    file=sys.stderr,
                )
        raise

    # Determine config_path based on --config and --base
    try:
        if args.config:
            config_path = args.config
        elif args.base:
            config_path = Path(args.base) / "config.yaml"
            if not config_path.exists():
                raise ValueError(f"config.yaml not found in {args.base}")
        else:
            raise ValueError("Either --config or --base must be provided")
    except ValueError as e:
        print(f"knowledge-mcp: {e}", file=sys.stderr)
        sys.exit(1)

    # Load config - config path might need adjustment depending on CWD
    # If config is expected relative to project root, and cli.py is in the package,
    # we might need to adjust how the default path is handled or make it absolute.
    # For now, assume it's run from project root or path is absolute.
    try:
        # If config is expected relative to project root, and cli.py is in the package,
        # we might need to adjust how the default path is handled or make it absolute.
        # For now, assume it's run from project root or path is absolute.
        Config.load(config_path)
        # Configure logging AFTER config is loaded
        configure_logging() 
        logger.info(f"Successfully loaded config from {config_path}")
    except FileNotFoundError:
        # Try searching relative to the cli script's parent dir? Or require absolute path?
        logger.error(f"Configuration file not found at {config_path}. Please provide a valid path.")
        sys.exit(1)
    except (ValueError, RuntimeError) as e:
        logger.critical(f"Failed to load or validate configuration: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during configuration loading: {e}")
        sys.exit(1)

    # Execute the function associated with the chosen command
    args.func()

def configure_logging():
    """Configure logging based on the loaded Config singleton."""
    config = Config.get_instance()
    log_config = config.logging
    kb_config = config.knowledge_base

    # Determine the main log file path (within the knowledge base base dir)
    log_file_path = kb_config.base_dir / "kbmcp.log"

    logger.info(f"Main application log file: {log_file_path}")

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": { # For console
                    "format": log_config.default_format,
                },
                "detailed": {
                    "format": log_config.detailed_format,
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr", # Correct stream specifier
                    "level": log_config.level,
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_config.max_bytes,
                    "backupCount": log_config.backup_count,
                    "encoding": "utf-8",
                    "level": log_config.level,
                },
            },
            "loggers": {
                "lightrag": { # Configure LightRAG's root logger
                    "handlers": ["file"],
                    "level": log_config.level, # Use level from config
                    "propagate": False, # Don't pass to our root logger
                },
                "kbmcp": { # Specific logger for our application modules
                    "handlers": ["file"],
                    "level": log_config.level,
                    "propagate": False, # Don't pass to root logger
                },
                "knowledge_mcp": { # Catch logs from submodules like knowledge_mcp.rag etc.
                    "handlers": ["file"],
                    "level": log_config.level,
                    "propagate": False,
                },
            },
            "root": { # Catch-all for other libraries (unless they disable propagation)
                "handlers": ["file"],
                "level": "WARNING", # Set root level higher to avoid too much noise
            },
        }
    )

if __name__ == "__main__":
    # This allows running the cli directly for development,
    # but entry point script is preferred for installation.
    main()
