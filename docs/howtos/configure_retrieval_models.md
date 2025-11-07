# How-to: Configure Retrieval Models

This guide explains how to select and configure different retrieval models for the semantic search component of the RAG pipeline.

## Default Configuration

By default, NERxiv uses the `all-MiniLM-L6-v2` model from SentenceTransformers:

```bash
nerxiv prompt --file-path paper.hdf5
```

This model:

- Is lightweight (~80MB)
- Produces 384-dimensional embeddings
- Works well for general semantic search
- Fast on CPU

## Choosing a Different Model

You can use any model from the [SentenceTransformers library](https://sbert.net/):

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --retriever-model all-mpnet-base-v2
```

### Popular Retriever Models

| Model | Embedding Size | Quality | Speed | Use Case |
|-------|---------------|---------|-------|----------|
| `all-MiniLM-L6-v2` | 384 | Good | Fast | General purpose (default) |
| `all-mpnet-base-v2` | 768 | Better | Medium | Higher quality retrieval |
| `all-MiniLM-L12-v2` | 384 | Good+ | Medium | Balance of speed/quality |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | Good | Medium | Non-English papers |
| `msmarco-distilbert-base-v4` | 768 | Best | Slow | Maximum retrieval quality |

## Adjusting Number of Retrieved Chunks

Control how many chunks the LLM sees:

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --n-top-chunks 10
```

- 3-5 chunks: Fast, focused answers (default: 5)
- 7-10 chunks: More context, better for complex queries
- 15+ chunks: Risk of exceeding LLM context or diluting relevance

## Complete Example

Extract material formulas with high-quality retrieval:

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --retriever-model all-mpnet-base-v2 \
  --n-top-chunks 8 \
  --query material_formula
```

## Using the Retriever in Python

For more control, use the retriever directly in Python:

```python
from nerxiv.chunker import Chunker
from nerxiv.rag import CustomRetriever

# Prepare chunks
text = "Your paper text here..."
chunker = Chunker(text=text)
chunks = chunker.chunk_text()

# Initialize retriever with custom model
retriever = CustomRetriever(
    model="all-mpnet-base-v2",
    query="Find all mentions of DFT calculations",
    n_top_chunks=5,
)

# Get relevant chunks
top_chunks = retriever.get_relevant_chunks(chunks=chunks)
print(top_chunks)
```

## Understanding Similarity Scores

The retriever computes cosine similarity between embeddings. Scores range from -1 to 1, and in absolute value:

- **0.7-1.0**: Very relevant
- **0.5-0.7**: Moderately relevant
- **< 0.5**: Less relevant

NERxiv logs these scores for the top chunks. To see them, check the logs:

```bash
nerxiv prompt --file-path paper.hdf5 2>&1 | grep "similarities"
```

Example output:
```
INFO - Top 5 chunks retrieved with similarities of tensor([0.8234, 0.7891, 0.7456, 0.7123, 0.6890])
```

## LangChain Alternative

NERxiv also provides a `LangChainRetriever` that uses LangChain's retrieval implementations:

```python
from nerxiv.rag import LangChainRetriever

retriever = LangChainRetriever(
    model="all-MiniLM-L6-v2",
    query="Your query here",
    n_top_chunks=5
)

top_chunks = retriever.get_relevant_chunks(chunks=chunks)
```

Both `CustomRetriever` and `LangChainRetriever` provide similar functionality, but `CustomRetriever` gives you direct access to similarity scores.

## Debugging Retrieval

To understand what the retriever is finding, inspect chunks in Python:

```python
from nerxiv.rag import CustomRetriever
from nerxiv.chunker import Chunker

chunker = Chunker(text=your_text)
chunks = chunker.chunk_text()

retriever = CustomRetriever(
    model="all-MiniLM-L6-v2",
    query="Find all chemical formulas"
)

# Get chunks with metadata
from sentence_transformers import util
import torch

query_emb = retriever.model.encode(retriever.query, convert_to_tensor=True)
chunk_texts = [chunk.page_content for chunk in chunks]
chunk_embs = retriever.model.encode(chunk_texts, convert_to_tensor=True)

similarities = util.pytorch_cos_sim(query_emb, chunk_embs).squeeze(0)
sorted_indices = similarities.argsort(descending=True)[:5]

for i, idx in enumerate(sorted_indices):
    print(f"\n=== Chunk {i+1} (similarity: {similarities[idx]:.4f}) ===")
    print(chunk_texts[idx][:200])  # First 200 chars
```