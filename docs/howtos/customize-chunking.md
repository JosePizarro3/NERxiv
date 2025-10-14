# How to Customize Chunking Strategies

This guide shows you how to choose and configure different chunking strategies for your RAG pipeline.

## Why Chunking Matters

Scientific papers are often too long to fit in an LLM's context window. Chunking divides the text into smaller pieces, but how you chunk affects what information the retriever can find and how well the LLM can answer your queries.

## Available Chunkers

NERxiv provides three chunking strategies:

### 1. Fixed-Size Chunker (Default)

The `Chunker` class uses fixed character-based chunks with overlap.

**When to use:**
- General-purpose chunking
- When you want consistent chunk sizes
- When processing speed is important

**CLI usage:**
```bash
nerxiv prompt --file-path paper.hdf5 --chunker Chunker
```

**Python usage:**
```python
from nerxiv.chunker import Chunker

chunker = Chunker(text=paper_text)
chunks = chunker.chunk_text(chunk_size=1000, chunk_overlap=200)
```

**Parameters:**
- `chunk_size` (default: 1000): Number of characters per chunk
- `chunk_overlap` (default: 200): Overlap between consecutive chunks

### 2. Semantic Chunker

The `SemanticChunker` uses spaCy to create chunks at sentence boundaries.

**When to use:**
- When you want to preserve sentence integrity
- When semantic coherence is important
- For extracting specific facts or statements

**CLI usage:**
```bash
nerxiv prompt --file-path paper.hdf5 --chunker SemanticChunker
```

**Python usage:**
```python
from nerxiv.chunker import SemanticChunker

chunker = SemanticChunker(text=paper_text)
chunks = chunker.chunk_text()
```

This chunker automatically groups sentences together while maintaining semantic boundaries.

### 3. Advanced Semantic Chunker

The `AdvancedSemanticChunker` uses KMeans clustering on sentence embeddings to group semantically similar sentences.

**When to use:**
- When you want topically coherent chunks
- When extracting complex, multi-sentence information
- When you know approximately how many topics are in the paper

**CLI usage:**
```bash
nerxiv prompt --file-path paper.hdf5 --chunker AdvancedSemanticChunker
```

**Python usage:**
```python
from nerxiv.chunker import AdvancedSemanticChunker

chunker = AdvancedSemanticChunker(text=paper_text)
chunks = chunker.chunk_text(n_chunks=10)
```

**Parameters:**
- `n_chunks` (default: 10): Number of semantic clusters to create

## Choosing the Right Strategy

Here's a decision guide:

| Your Goal | Recommended Chunker | Why |
|-----------|-------------------|-----|
| Fast processing | `Chunker` | Simple, no NLP overhead |
| Extract formulas/numbers | `Chunker` or `SemanticChunker` | Preserves local context |
| Extract methodology descriptions | `AdvancedSemanticChunker` | Groups related methodological text |
| General metadata extraction | `SemanticChunker` | Good balance of speed and quality |
| Highly specific technical queries | `AdvancedSemanticChunker` | Better topical grouping |

## Example Comparison

Let's extract material formulas using different chunkers:

**Fixed-size chunking:**
```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --chunker Chunker \
  --query material_formula
```

**Semantic chunking:**
```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --chunker SemanticChunker \
  --query material_formula
```

**Advanced semantic chunking:**
```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --chunker AdvancedSemanticChunker \
  --query material_formula
```

## Advanced Configuration

### Adjusting Fixed-Size Chunks

You can't directly pass `chunk_size` via CLI, but you can modify it in your Python scripts:

```python
from pathlib import Path
import h5py
from nerxiv.chunker import Chunker
from nerxiv.rag import CustomRetriever, LLMGenerator
from nerxiv.prompts import PROMPT_REGISTRY

# Load paper text
paper_path = Path("paper.hdf5")
with h5py.File(paper_path, "r") as f:
    arxiv_id = paper_path.stem
    text = f[arxiv_id]["arxiv_paper"]["text"][()].decode("utf-8")

# Custom chunking
chunker = Chunker(text=text)
chunks = chunker.chunk_text(chunk_size=1500, chunk_overlap=300)

# Continue with retrieval and generation
retriever_query = PROMPT_REGISTRY["material_formula"].retriever_query
retriever = CustomRetriever(query=retriever_query)
top_text = retriever.get_relevant_chunks(chunks=chunks, n_top_chunks=5)

prompt = PROMPT_REGISTRY["material_formula"].prompt
generator = LLMGenerator(model="llama3.1:70b", text=top_text)
answer = generator.generate(prompt=prompt.build(text=top_text))
print(answer)
```

### Adjusting Semantic Clusters

For papers with complex topics, increase the number of clusters:

```python
from nerxiv.chunker import AdvancedSemanticChunker

chunker = AdvancedSemanticChunker(text=paper_text)
chunks = chunker.chunk_text(n_chunks=15)  # More granular clustering
```

## Debugging Chunks

To see what chunks are created, inspect them in Python:

```python
from nerxiv.chunker import SemanticChunker

chunker = SemanticChunker(text=paper_text)
chunks = chunker.chunk_text()

# Print first 3 chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"=== Chunk {i} ===")
    print(chunk.page_content)
    print()
```

## Performance Considerations

- **`Chunker`**: Fastest, no NLP models required
- **`SemanticChunker`**: Medium speed, loads spaCy model once
- **`AdvancedSemanticChunker`**: Slowest, computes embeddings for all sentences

For batch processing many papers, consider using the simpler `Chunker` first, then optimize with semantic chunkers if needed.

## Related Guides

- [How to configure retrieval models](configure-retrieval-models.md)
- [How to create custom prompts](create-custom-prompts.md)
