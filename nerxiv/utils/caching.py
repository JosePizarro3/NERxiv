import hashlib
import json

from nerxiv.chunker import CHUNKER_VERSION
from nerxiv.rag.retriever import RETRIEVER_VERSION


def compute_chunker_hash(
    text: str, chunker_name: str, chunker_params: dict | None = None
) -> str:
    """
    Compute a hash to uniquely identify a chunking configuration. This hash is used to determine
    if chunking can be reused from a previous run.

    The hash is based on:
    - The chunker version (changes when implementation in `nerxiv.chunker` changes)
    - The chunker class name
    - The chunker parameters (if any)
    - The text being chunked

    Args:
        text (str): The text that will be chunked.
        chunker_name (str): The name of the chunker class (e.g., 'Chunker', 'SemanticChunker').
        chunker_params (dict, optional): Parameters passed to the chunk_text method.
            For Chunker: {'chunk_size': 1000, 'chunk_overlap': 200}
            For AdvancedSemanticChunker: {'n_chunks': 10}
            For SemanticChunker: None or {}

    Returns:
        str: A hexadecimal hash string uniquely identifying this chunking configuration.
    """
    if chunker_params is None:
        chunker_params = {}

    # Create a dictionary with all components
    hash_data = {
        "version": CHUNKER_VERSION,
        "chunker": chunker_name,
        "params": chunker_params,
        "text": text,
    }

    # Convert to JSON with sorted keys for consistent hashing
    hash_string = json.dumps(hash_data, sort_keys=True)

    # Compute SHA256 hash
    return hashlib.sha256(hash_string.encode("utf-8")).hexdigest()


def compute_retriever_hash(
    chunker_hash: str,
    retriever_params: dict | None = None,
) -> str:
    """
    Compute a hash to uniquely identify a retrieval configuration. This hash is used to determine
    if retrieval (top-k chunks selection) can be reused from a previous run.

    The hash is based on:
    - The retriever version (changes when implementation in `nerxiv.rag.retriever` changes)
    - The chunker hash (ensures same base chunks)
    - The retriever model
    - The retriever query
    - The name of the retriever query
    - The number of top chunks to retrieve

    Args:
        chunker_hash (str): Hash of the chunking configuration.
        retriever_params (dict, optional): Parameters for the retriever.
            Example:
            {
                'model': 'all-MiniLM-L6-v2',
                'query_name': 'filter_material_formula',
                'n_top_chunks': 5
            }

    Returns:
        str: A hexadecimal hash string uniquely identifying this retrieval configuration.
    """
    # Create a dictionary with all components
    hash_data = {
        "version": RETRIEVER_VERSION,
        "chunker_hash": chunker_hash,
        "params": retriever_params,
    }

    # Convert to JSON with sorted keys for consistent hashing
    hash_string = json.dumps(hash_data, sort_keys=True)

    # Compute SHA256 hash
    return hashlib.sha256(hash_string.encode("utf-8")).hexdigest()
