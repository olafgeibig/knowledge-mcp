from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.openai import openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np      
from knowledge_mcp.config import Config

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # Use model_name from global config provided by LightRAG within kwargs
    # Check if 'hashing_kv' and 'global_config' are present before accessing
    if "hashing_kv" in kwargs and hasattr(kwargs["hashing_kv"], "global_config"):
        model_name = kwargs["hashing_kv"].global_config.get("llm_model_name")
    else:
        # Fallback to config when hashing_kv is not available (e.g., from modal processors)
        model_name = Config.get_instance().lightrag.llm.model_name

    # Prioritize api_key and base_url from kwargs (coming from llm_model_kwargs)
    # Use environment variables as fallback
    # api_key = kwargs.pop('api_key', os.getenv("OPENAI_API_KEY"))
    # base_url = kwargs.pop('base_url', os.getenv("OPENAI_API_BASE"))

    api_key = Config.get_instance().lightrag.llm.api_key
    base_url = Config.get_instance().lightrag.llm.api_base

    if not api_key:
        raise ValueError("OpenAI API key is not provided in config or environment variables.")

    return await openai_complete_if_cache(
        model=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key, # Pass the determined api_key
        base_url=base_url, # Pass the determined base_url (can be None)
        keyword_extraction=keyword_extraction,
        **kwargs, # Pass remaining kwargs
    )

async def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs) -> str: 
    # Use model_name from global config provided by LightRAG within kwargs
    # Check if 'hashing_kv' and 'global_config' are present before accessing
    if "hashing_kv" in kwargs and hasattr(kwargs["hashing_kv"], "global_config"):
        model_name = kwargs["hashing_kv"].global_config.get("llm_model_name")
    else:
        # Fallback to config when hashing_kv is not available (e.g., from modal processors)
        model_name = Config.get_instance().lightrag.llm.model_name
    api_key = Config.get_instance().lightrag.llm.api_key
    base_url = Config.get_instance().lightrag.llm.api_base

    if not api_key:
        raise ValueError("OpenAI API key is not provided in config or environment variables.")
    
    return await openai_complete_if_cache(
        model=model_name,
        prompt="",
        system_prompt=system_prompt,
        history_messages=history_messages,
        messages=[
            {"role": "system", "content": system_prompt} if system_prompt else None,
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]} if image_data else {"role": "user", "content": prompt}
        ],
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    ) if image_data else await openai_complete_if_cache(
            model=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

async def openai_embedding_func(texts: list[str]) -> np.ndarray:
    # Call the OpenAI-compatible embeddings endpoint directly instead of going
    # through lightrag.llm.openai.openai_embed, which is decorated with
    # @wrap_embedding_func_with_attrs(embedding_dim=1536) and therefore pins
    # the dimension to OpenAI text-embedding-3-*. Going direct lets the caller
    # use any model (Ollama bge-m3 at 1024, mxbai-embed-large, etc.); the
    # EmbeddingFunc wrapper below still validates the configured dimension.
    cfg = Config.get_instance().lightrag.embedding
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.api_base)
    response = await client.embeddings.create(model=cfg.model_name, input=texts)
    return np.array([item.embedding for item in response.data])

# Wrap the embedding function with the correct attributes from config
embedding_func = EmbeddingFunc(
    embedding_dim=Config.get_instance().lightrag.embedding.embedding_dim,
    max_token_size=Config.get_instance().lightrag.embedding.max_token_size,
    func=openai_embedding_func
)