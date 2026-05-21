# knowledge_mcp/rag.py
"""Manages LightRAG instances for different knowledge bases."""

import logging
import logging.handlers
from raganything import RAGAnything
from lightrag import LightRAG, QueryParam
from lightrag.base import DeletionResult
from lightrag.kg.shared_storage import initialize_pipeline_status
from typing import Dict, Optional, Any
import asyncio
import shutil
from pathlib import Path

# Need to import Config and KbManager to use them
from knowledge_mcp.config import Config
from knowledge_mcp.knowledgebases import KnowledgeBaseManager, KnowledgeBaseNotFoundError, load_kb_query_config

logger = logging.getLogger(__name__) # General logger for RagManager setup/errors not specific to a KB

class RAGManagerError(Exception):
    """Base exception for RAGManager errors."""

class UnsupportedProviderError(RAGManagerError):
    """Raised when a configured provider is not supported."""

class RAGInitializationError(RAGManagerError):
    """Raised when LightRAG instance initialization fails."""

class ConfigurationError(RAGManagerError):
    """Raised when required configuration is missing or invalid."""


class RagManager:
    """Creates, manages, and caches LightRAG instances per knowledge base."""

    def __init__(self, config: Config, kb_manager: KnowledgeBaseManager): 
        """Initializes the RagManager with the KB manager."""
        self._rag_instances: Dict[str, RAGAnything] = {}
        self.kb_manager = kb_manager 
        self.config = config # Store config if needed for global defaults
        logger.info("RagManager initialized.") 

    async def get_rag_instance(self, kb_name: str) -> RAGAnything:
        """
        Retrieves or creates and initializes a LightRAG instance for the given KB.
        Asynchronous access.
        """
        if kb_name in self._rag_instances:
            logging.getLogger(f"kbmcp.{kb_name}").debug("Returning cached LightRAG instance.")
            return self._rag_instances[kb_name]
        else:
            if self.kb_manager.kb_exists(kb_name):
                # Call the async creation method directly
                logging.getLogger(f"kbmcp.{kb_name}").debug("No cached instance found. Running async create_rag_instance...")
                # Now awaits the async creation method directly
                try:
                    instance = await self.create_rag_instance(kb_name)
                    self._rag_instances[kb_name] = instance # Cache after successful creation
                    return instance
                except RuntimeError as e:
                    # Handle potential asyncio errors if create_rag_instance has issues
                    logging.getLogger(f"kbmcp.{kb_name}").error(f"Error during async RAG instance creation: {e}")
                    raise RAGInitializationError(f"Async RAG instance creation failed for {kb_name}") from e
                    # No longer need the nested asyncio.run check
            else:
                raise KnowledgeBaseNotFoundError(f"Knowledge base '{kb_name}' does not exist.")
    
    async def create_rag_instance(self, kb_name: str) -> RAGAnything:
        # Use KbManager to check existence and get path
        if not self.kb_manager.kb_exists(kb_name):
            raise KnowledgeBaseNotFoundError(f"Knowledge base '{kb_name}' does not exist.")
        kb_path = self.kb_manager.get_kb_path(kb_name)
        logging.getLogger(f"kbmcp.{kb_name}").debug(f"Creating new LightRAG instance in {kb_path}")

        try:
            # Get the singleton config instance
            config = Config.get_instance()

            # Validate required settings sections exist
            if not config.lightrag or not config.lightrag.llm:
                 raise ConfigurationError("Language model settings (config.lightrag.llm) are missing.")
            if not config.lightrag.embedding:
                 raise ConfigurationError("Embedding model settings (config.lightrag.embedding) are missing.")
            if not config.lightrag.embedding_cache:
                 raise ConfigurationError("Embedding cache settings (config.lightrag.embedding_cache) are missing.")

            llm_config = config.lightrag.llm
            kb_logger = logging.getLogger(f"kbmcp.{kb_name}") # Get logger once for this method
            embed_config = config.lightrag.embedding
            cache_config = config.lightrag.embedding_cache

            # --- Get Embedding Function and Kwargs ---
            embed_provider = embed_config.provider.lower()
            if embed_provider == "openai":
                import knowledge_mcp.openai_func
                embed_func = knowledge_mcp.openai_func.embedding_func
            else:
                raise UnsupportedProviderError("Only OpenAI embedding provider currently supported.") 

            # --- Get LLM Function and Kwargs ---
            llm_provider = llm_config.provider.lower()
            if llm_provider == "openai":
                import knowledge_mcp.openai_func
                llm_func = knowledge_mcp.openai_func.llm_model_func
                vision_model_func = knowledge_mcp.openai_func.vision_model_func
            else:
                raise UnsupportedProviderError("Only OpenAI language model provider currently supported.") 

            if not llm_config.api_key:
                 raise ConfigurationError("API key missing for OpenAI language model provider")
            # llm_kwargs={"api_key": llm_config.api_key}
            # if llm_config.api_base:
            #     llm_kwargs["base_url"] = llm_config.api_base
            # Add other potential kwargs if needed (e.g., temperature, etc.)
            llm_kwargs = {}
            if llm_config.kwargs:
                llm_kwargs.update(llm_config.kwargs)

            # Max tokens for the LLM *model* (for context window sizing)
            # Use LightRAG default if not set, check LightRAG docs for correct handling
            llm_model_max_tokens = llm_config.max_token_size

            kb_logger.info(
                "Attempting to initialize LightRAG with parameters:\n"
                f"  working_dir: {kb_path}\n"
                f"  embed_model: {embed_config.model_name}\n"
                f"  llm_model: {llm_config.model_name}, llm_kwargs: {llm_kwargs}\n"
                f"  llm_model_max_token_size: {llm_model_max_tokens}"
            )

            # --- Instantiate LightRAG ---
            # Note: Verify LightRAG constructor parameters closely with LightRAG docs
            lightrag = LightRAG(
                working_dir=str(kb_path),
                llm_model_func=llm_func,
                llm_model_kwargs=llm_kwargs,
                llm_model_name=llm_config.model_name,
                embedding_func=embed_func,
                embedding_cache_config={
                    "enabled": cache_config.enabled,
                    "similarity_threshold": cache_config.similarity_threshold,
                },
                enable_llm_cache=True,  # Enable LLM response caching for modal processors
            )
            kb_logger.debug(f"Initializing LightRAG components for {kb_name}...")
            # Check LightRAG documentation for the correct initialization method
            # It might be initialize_components(), initialize_storages(), or similar.
            # Assuming initialize_components() based on common patterns
            await lightrag.initialize_storages()
            await initialize_pipeline_status()

            rag = RAGAnything(
                lightrag=lightrag,
                llm_model_func=llm_func,  # Pass LLM function for text processing
                vision_model_func=vision_model_func,
                embedding_func=embed_func,  # Pass embedding function for modal processors
            )

            kb_logger.info(f"Successfully initialized LightRAG instance for {kb_name}.")
            self._rag_instances[kb_name] = rag
            return rag

        except (UnsupportedProviderError, ConfigurationError, KnowledgeBaseNotFoundError) as e:
            logging.getLogger(f"kbmcp.{kb_name}").error(f"Configuration error creating RAG instance for {kb_name}: {e}")
            raise # Re-raise specific config errors
        except Exception as e:
            logging.getLogger(f"kbmcp.{kb_name}").exception(f"Unexpected error creating RAG instance for {kb_name}: {e}")
            # Wrap unexpected errors in a specific exception type
            raise RAGInitializationError(f"Failed to initialize LightRAG for {kb_name}: {e}") from e

    async def get_lightrag_instance(self, kb_name: str) -> LightRAG:
        """Get the underlying LightRAG instance from RAGAnything for direct access.
        
        This method provides direct access to the LightRAG instance for text-only
        operations that don't require RAGAnything's multimodal capabilities.
        
        Args:
            kb_name: The name of the knowledge base.
            
        Returns:
            The underlying LightRAG instance.
            
        Raises:
            KnowledgeBaseNotFoundError: If the knowledge base doesn't exist.
            RAGInitializationError: If the RAG instance cannot be retrieved.
        """
        try:
            # Get the RAGAnything instance (this will create it if it doesn't exist)
            rag_anything = await self.get_rag_instance(kb_name)
            
            # Access the underlying LightRAG instance
            lightrag_instance = rag_anything.lightrag
            
            logging.getLogger(f"kbmcp.{kb_name}").debug("Retrieved underlying LightRAG instance for direct access.")
            return lightrag_instance
            
        except Exception as e:
            msg = f"Failed to get LightRAG instance for {kb_name}: {e}"
            logging.getLogger(f"kbmcp.{kb_name}").error(msg)
            raise RAGInitializationError(msg) from e

    def remove_rag_instance(self, kb_name: str | None = None) -> None:
        """Removes a rag instance by name"""
        if kb_name:
            if kb_name in self._rag_instances:
                del self._rag_instances[kb_name]
                logger.info(f"Removed LightRAG instance for KB: {kb_name}")
            else:
                logger.error(f"Knowledge base '{kb_name}' not found.")
                raise KnowledgeBaseNotFoundError(f"Knowledge base '{kb_name}' not found.")
        else:
            logger.error("Knowledgebase name is required.")
            raise ValueError("Knowledgebase name is required.")

    async def query(self, kb_name: str, query_text: str, **kwargs: Any) -> Any:
        """
        Executes a query against the specified knowledge base asynchronously,
        loading and applying its configuration, and running sync LightRAG calls in a thread.
        """
        kb_logger = logging.getLogger(f"kbmcp.{kb_name}") # Get logger once for this method
        kb_logger.info(f"--- Executing asynchronous query for KB: {kb_name} ---")
        kb_logger.info(f"Query: {query_text}")
        try:
            # Get KB path for configuration loading
            kb_path = self.kb_manager.get_kb_path(kb_name)

            # Load KB-specific configuration
            kb_logger.info(f"Loading query configuration from {kb_path / 'config.yaml'}...")
            kb_config = load_kb_query_config(kb_path)
            kb_logger.debug(f"Loaded sync config for '{kb_name}': {kb_config}")
            
            # Extract and log user_prompt if present
            user_prompt = kb_config.get('user_prompt', '')
            if user_prompt and user_prompt.strip():
                kb_logger.debug(f"Extracted user_prompt from config: '{user_prompt}'")
            else:
                kb_logger.debug("No user_prompt configured or user_prompt is empty")

            # Merge configurations: kwargs > kb_config
            final_query_params = kb_config.copy()
            if kwargs:
                kb_logger.debug(f"Applying runtime query kwargs: {kwargs}")
                final_query_params.update(kwargs)
            else:
                kb_logger.debug("No runtime query kwargs provided.")

            # Ensure 'description' is not passed as a query param
            final_query_params.pop("description", None)

            # Filter out kwargs that older per-KB configs may still emit but
            # which were removed from lightrag-hku's QueryParam in 1.4.x.
            for stale_kwarg in ("ids", "model_func", "chunk_top_k", "enable_rerank"):
                final_query_params.pop(stale_kwarg, None)

            # Handle user_prompt: only include if not empty
            user_prompt_value = final_query_params.get('user_prompt', '')
            if not (user_prompt_value and user_prompt_value.strip()):
                # Remove empty user_prompt to avoid passing it to QueryParam
                final_query_params.pop('user_prompt', None)
                kb_logger.debug("Removed empty user_prompt from query parameters")
            else:
                kb_logger.debug(f"Including user_prompt in query parameters: '{user_prompt_value}'")
            
            kb_logger.info(f"Query parameters: {final_query_params}")

            # Create QueryParam instance
            try:
                query_param_instance = QueryParam(**final_query_params)
                kb_logger.debug(f"Created QueryParam instance: {query_param_instance}")
                
                # Log user_prompt inclusion in QueryParam if present
                if hasattr(query_param_instance, 'user_prompt') and query_param_instance.user_prompt:
                    kb_logger.debug(f"QueryParam instance includes user_prompt: '{query_param_instance.user_prompt}'")
                else:
                    kb_logger.debug("QueryParam instance has no user_prompt (empty or not configured)")
            except Exception as e:
                kb_logger.error(f"Failed to create QueryParam instance from params {final_query_params}: {e}")
                raise ConfigurationError(f"Invalid query parameters: {e}") from e

            # Execute the query using the underlying LightRAG instance
            # Fix for event loop issue: Force recreation of RAG instance for each query
            # This prevents event loop conflicts by ensuring fresh state
            
            # Remove the cached instance to force recreation for this query
            # This ensures LightRAG starts with a clean event loop state
            if kb_name in self._rag_instances:
                kb_logger.debug("Removing cached RAG instance to prevent event loop conflicts")
                del self._rag_instances[kb_name]
                
            # Get a fresh RAG instance for this query
            fresh_rag_instance = await self.get_rag_instance(kb_name)
            
            # Execute the query with the fresh instance
            result = await asyncio.to_thread(
                fresh_rag_instance.lightrag.query,
                query_text,
                query_param_instance
            )
            return result

        except (KnowledgeBaseNotFoundError, RAGInitializationError, ConfigurationError) as e:
            logging.getLogger(f"kbmcp.{kb_name}").error(f"Error preparing or executing query for KB '{kb_name}': {e}") # Use dynamic logger for exceptions too
            raise
        except Exception as e:
            logging.getLogger(f"kbmcp.{kb_name}").exception(f"Unexpected error during async query execution for KB '{kb_name}': {e}")
            raise RAGManagerError(f"Async query failed: {e}") from e

    def _cleanup_output_directory(self, output_dir: Path, kb_logger: logging.Logger) -> None:
        """Clean up all contents of the output directory after successful ingestion."""
        try:
            if output_dir.exists() and output_dir.is_dir():
                # Remove all contents of the output directory
                for item in output_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                        kb_logger.debug(f"Deleted file: {item.name}")
                    elif item.is_dir():
                        shutil.rmtree(item)
                        kb_logger.debug(f"Deleted directory: {item.name}")
                kb_logger.info("Successfully cleaned up output directory")
            else:
                kb_logger.debug("Output directory does not exist or is not a directory")
        except Exception as e:
            kb_logger.warning(f"Failed to clean up output directory: {e}")
            # Don't raise exception as cleanup failure shouldn't fail the ingestion

    async def ingest_document(self, kb_name: str, file_path: Any, doc_id: Optional[str] = None, parse_method: str = "auto", text_content: Optional[str] = None) -> Optional[str]:
        """Ingests a document into the specified knowledge base.
        
        Args:
            kb_name: The name of the knowledge base.
            file_path: Path to the document file.
            doc_id: Optional document ID. If not provided, uses file name.
            parse_method: Parsing method to use. Options:
                - "auto": Automatically choose based on available parameters (default)
                - "multimodal": Use RAGAnything for full multimodal processing
                - "text": Use LightRAG directly for text-only processing (requires text_content)
            text_content: Pre-extracted text content. Required when parse_method="text".
            
        Returns:
            The document ID used for ingestion.
            
        Raises:
            ValueError: If parse_method="text" but text_content is not provided.
            RAGInitializationError: If the RAG instance cannot be retrieved.
            RAGManagerError: If ingestion fails.
        """
        kb_logger = logging.getLogger(f"kbmcp.{kb_name}") # Get logger once for this method
        kb_logger.info(f"Ingesting document '{file_path.name}' into KB '{kb_name}' using {parse_method} method...")
        
        # Validate parse_method and parameters
        if parse_method == "text" and text_content is None:
            raise ValueError("text_content parameter is required when parse_method='text'")
        
        # Auto-select parse method if requested
        if parse_method == "auto":
            if text_content is not None:
                parse_method = "text"
                kb_logger.info("Auto-selected 'text' parsing method (text_content provided)")
            else:
                parse_method = "multimodal"
                kb_logger.info("Auto-selected 'multimodal' parsing method (no text_content provided)")
        
        generated_doc_id = doc_id or file_path.name 

        try:
            # Route to appropriate ingestion method based on parse_method
            if parse_method == "text":
                # Use text-only ingestion with LightRAG directly
                kb_logger.debug(f"Routing to text-only ingestion for '{generated_doc_id}'...")
                return await self.ingest_text_only(kb_name, text_content, generated_doc_id)
            
            elif parse_method == "multimodal":
                # Use multimodal ingestion with RAGAnything
                kb_logger.debug(f"Routing to multimodal ingestion for '{generated_doc_id}'...")
                
                # Set up output directory for multimodal processing
                kb_path = self.kb_manager.get_kb_path(kb_name)
                output_dir = kb_path / "output"
                output_dir.mkdir(exist_ok=True)  # Ensure output directory exists
                
                # Get RAGAnything instance
                rag_instance = await self.get_rag_instance(kb_name)
                
                kb_logger.debug(f"Running multimodal ingest for {file_path}...")
                await rag_instance.process_document_complete(
                    file_path=str(file_path),
                    output_dir=str(output_dir),     
                    parse_method="auto",  # Let RAGAnything handle its own parsing logic
                    doc_id=generated_doc_id 
                )
                
                kb_logger.info(f"Successfully ingested document '{file_path.name}' as '{generated_doc_id}' using multimodal processing")
                
                # Clean up output directory after successful ingestion
                self._cleanup_output_directory(output_dir, kb_logger)
                
                return generated_doc_id
            
            else:
                raise ValueError(f"Unsupported parse_method: {parse_method}. Use 'text' or 'multimodal'.")
        
        except RAGInitializationError as e:
            logging.getLogger(f"kbmcp.{kb_name}").error(f"Cannot ingest, RAG instance failed to initialize: {e}") # Use dynamic logger for exceptions too
            raise # Re-raise the initialization error
        except FileNotFoundError:
            logging.getLogger(f"kbmcp.{kb_name}").error(f"Document file not found: {file_path}")
            raise
        except Exception as e:
            logging.getLogger(f"kbmcp.{kb_name}").exception(f"Failed to ingest document '{file_path.name}': {e}")
            # Consider wrapping in a specific IngestionError if needed
            raise RAGManagerError(f"Ingestion failed for '{file_path.name}': {e}") from e

    async def ingest_text_only(self, kb_name: str, text_content: str, doc_id: Optional[str] = None) -> Optional[str]:
        """Ingests text content directly into the knowledge base using LightRAG only.
        
        This method bypasses RAGAnything's multimodal processing and uses LightRAG
        directly for text-only ingestion. This is more efficient for pre-extracted
        text content (e.g., from MarkItDown).
        
        Args:
            kb_name: The name of the knowledge base.
            text_content: The text content to ingest.
            doc_id: Optional document ID. If not provided, a UUID will be generated.
            
        Returns:
            The document ID used for ingestion.
            
        Raises:
            RAGInitializationError: If the LightRAG instance cannot be retrieved.
            RAGManagerError: If ingestion fails.
        """
        import uuid
        
        kb_logger = logging.getLogger(f"kbmcp.{kb_name}")
        generated_doc_id = doc_id or str(uuid.uuid4())
        
        kb_logger.info(f"Ingesting text content directly into KB '{kb_name}' as '{generated_doc_id}'...")
        kb_logger.debug(f"Text content length: {len(text_content):,} characters")
        
        try:
            # Get the underlying LightRAG instance for direct text ingestion
            lightrag_instance = await self.get_lightrag_instance(kb_name)
            
            kb_logger.debug(f"Running direct LightRAG text ingestion for '{generated_doc_id}'...")
            
            # Use LightRAG's direct text ingestion method
            # Note: LightRAG typically uses ainsert() for text ingestion
            await lightrag_instance.ainsert(text_content, ids=[generated_doc_id], file_paths=[generated_doc_id])
            
            kb_logger.info(f"Successfully ingested text content as '{generated_doc_id}' using LightRAG directly")
            return generated_doc_id
            
        except RAGInitializationError as e:
            kb_logger.error(f"Cannot ingest text, LightRAG instance failed to initialize: {e}")
            raise # Re-raise the initialization error
        except Exception as e:
            kb_logger.exception(f"Failed to ingest text content as '{generated_doc_id}': {e}")
            raise RAGManagerError(f"Text-only ingestion failed for '{generated_doc_id}': {e}") from e

    async def remove_document(self, kb_name: str, doc_id: str) -> DeletionResult:
        """Removes a document from the specified knowledge base by its ID."""
        try:
            rag_instance = await self.get_rag_instance(kb_name)
            result = await rag_instance.lightrag.adelete_by_doc_id(doc_id)
            return result
        except RAGInitializationError:
            logging.getLogger(f"kbmcp.{kb_name}").error("Cannot remove document, RAG instance failed to initialize")
            raise
        except Exception as e:
            logging.getLogger(f"kbmcp.{kb_name}").exception(f"Failed to remove document '{doc_id}': {e}")
            raise RAGManagerError(f"Failed to remove document '{doc_id}': {e}") from e