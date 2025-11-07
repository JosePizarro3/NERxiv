# How-to: Customize Chunking Strategies

This guide shows you how to choose and configure different chunking strategies for your RAG pipeline. You can read more about why chunking matters in [Explanation: Understanding Chunking Strategies](../explanations/understanding_chunking.md).


## Available Chunkers

NERxiv provides three chunking strategies:

### 1. Fixed-Size Chunker (Default)

The [`Chunker`](../references/api.md#nerxiv.chunker.Chunker) class uses fixed character-based chunks with overlap.

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

chunker = Chunker(chunk_size=1000, chunk_overlap=200, text=paper_text)
chunks = chunker.chunk_text()
```

### 2. Semantic Chunker

The [`SemanticChunker`](../references/api.md#nerxiv.chunker.SemanticChunker) uses spaCy to create chunks at sentence boundaries.

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

The [`AdvancedSemanticChunker`](../references/api.md#nerxiv.chunker.AdvancedSemanticChunker) uses KMeans clustering on sentence embeddings to group semantically similar sentences.

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

chunker = AdvancedSemanticChunker(n_chunks=10, text=paper_text)
chunks = chunker.chunk_text()
```

## Choosing the Right Strategy

| Your Goal | Recommended Chunker | Why |
|-----------|-------------------|-----|
| Fast processing | `Chunker` | Simple, no NLP overhead |
| Extract formulas/numbers | `Chunker` or `SemanticChunker` | Preserves local context |
| Extract methodology descriptions | `AdvancedSemanticChunker` | Groups related methodological text |
| General metadata extraction | `SemanticChunker` | Good balance of speed and quality |
| Highly specific technical queries | `AdvancedSemanticChunker` | Better topical grouping |


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
chunker = Chunker(chunk_size=1500, chunk_overlap=300, text=text)
chunks = chunker.chunk_text()

# Continue with retrieval and generation
retriever_query = PROMPT_REGISTRY["material_formula"].retriever_query
retriever = CustomRetriever(n_top_chunks=5, query=retriever_query)
top_text = retriever.get_relevant_chunks(chunks=chunks)

prompt = PROMPT_REGISTRY["material_formula"].prompt
generator = LLMGenerator(model="llama3.1:70b", text=top_text)
answer = generator.generate(prompt=prompt.build(text=top_text))
print(answer)
```

### Adjusting Semantic Clusters

For papers with complex topics, increase the number of clusters:

```python
from nerxiv.chunker import AdvancedSemanticChunker

chunker = AdvancedSemanticChunker(n_chunks=15, text=paper_text)  # More granular clustering
chunks = chunker.chunk_text()
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
