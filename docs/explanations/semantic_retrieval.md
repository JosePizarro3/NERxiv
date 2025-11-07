# How Semantic Retrieval Works

Semantic retrieval is the core technology enabling RAG to find relevant information without exact keyword matches. This explanation explores embeddings, similarity metrics, and how retrieval models work in NERxiv.

## The Problem with Keyword Search

Traditional keyword search matches exact words:

```
Query: "DFT calculation"
Document 1: "We performed DFT calculations..." ✅ Match
Document 2: "Density functional theory was used..." ❌ No match
Document 3: "First-principles methods..." ❌ No match
```

Problems:
- Misses synonyms and related terms
- Doesn't understand context
- Can't rank by relevance beyond word frequency
- Fails with paraphrasing

## Semantic Search with Embeddings

Semantic search converts text into high-dimensional vectors (embeddings) that capture meaning:

```
Query: "DFT calculation"
Embedding: [0.23, -0.45, 0.12, 0.89, ..., -0.31]  (384 dimensions)

Document 1: "DFT calculations"
Embedding: [0.22, -0.44, 0.11, 0.87, ..., -0.29]  (very similar!)

Document 2: "Density functional theory"
Embedding: [0.19, -0.42, 0.08, 0.85, ..., -0.28]  (also similar!)

Document 3: "The weather is nice"
Embedding: [-0.51, 0.23, -0.78, 0.15, ..., 0.62]  (not similar)
```

**Key insight**: Semantically similar texts have similar embeddings, even with different words.

## What are Embeddings?

Embeddings are dense vector representations of text that encode semantic meaning.

### Creating Embeddings

A **sentence transformer model** converts text to embeddings:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

text = "The bandgap is 1.2 eV"
embedding = model.encode(text)
# Returns: numpy array of shape (384,)
# [0.23, -0.45, 0.12, ..., 0.89]
```

### Properties of Embeddings

1. **Fixed dimension**: All texts become the same size vector
   - `all-MiniLM-L6-v2`: 384 dimensions
   - `all-mpnet-base-v2`: 768 dimensions

2. **Semantic similarity**: Similar meanings → similar vectors
   ```
   "DFT calculation" ≈ [0.2, -0.4, 0.1, ...]
   "DFT computation" ≈ [0.2, -0.4, 0.1, ...]
   "Weather forecast" ≈ [-0.5, 0.2, -0.8, ...]
   ```

3. **Continuous space**: Embeddings exist in continuous space, enabling similarity measurement

4. **Language understanding**: Captures grammar, context, relationships

## Measuring Similarity

Once we have embeddings, we need to measure how similar they are.

### Cosine Similarity

The standard metric is **cosine similarity**, which measures the angle between vectors:

```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)

Where:
- A · B = dot product
- ||A|| = magnitude of vector A
```

**Range**: -1 to 1
- **1.0**: Identical meaning
- **0.7-0.9**: Very similar
- **0.5-0.7**: Moderately similar
- **0.3-0.5**: Somewhat related
- **<0.3**: Different topics
- **0.0**: Orthogonal (no relation)
- **<0.0**: Opposite meaning (rare in practice)

### Visual Intuition

Imagine embeddings as arrows in space:

```
                    Query: "Find DFT methods"
                         ↑
                        /|\
                       / | \
                      /  |  \
          Chunk 1: "DFT used"  (angle: 15°, sim: 0.97)
                    /         \
                   /           \
         Chunk 2: "QMC methods" (angle: 45°, sim: 0.71)
                                 \
                                  \
               Chunk 3: "Weather data" (angle: 90°, sim: 0.0)
```

where:

- Small angle = high similarity
- Large angle = low similarity

## Retrieval in NERxiv

Here's how semantic retrieval works in NERxiv's RAG pipeline:

### Step-by-Step Process

```python
from nerxiv.chunker import Chunker
from nerxiv.rag import CustomRetriever

# 1. Chunk the paper
chunker = Chunker(text=paper_text)
chunks = chunker.chunk_text()
# Result: [chunk_1, chunk_2, ..., chunk_100]

# 2. Initialize retriever with query
retriever = CustomRetriever(
    model="all-MiniLM-L6-v2",
    query="Find all mentions of chemical formulas",
    n_top_chunks=5,
)

# 3. Retrieve relevant chunks
top_chunks = retriever.get_relevant_chunks(chunks=chunks)
```

### What Happens Inside `get_relevant_chunks`

```python
# Pseudo-code showing the internal process

def get_relevant_chunks(chunks, n_top_chunks):
    # 1. Extract text from chunks
    chunk_texts = [chunk.page_content for chunk in chunks]

    # 2. Encode query
    query_embedding = model.encode(query)
    # Shape: (384,)

    # 3. Encode all chunks
    chunk_embeddings = model.encode(chunk_texts)
    # Shape: (100, 384) for 100 chunks

    # 4. Compute cosine similarity
    similarities = cosine_similarity(query_embedding, chunk_embeddings)
    # Shape: (100,) - one score per chunk
    # Example: [0.87, 0.34, 0.92, 0.15, ..., 0.45]

    # 5. Sort by similarity (descending)
    sorted_indices = argsort(similarities, descending=True)
    # Example: [2, 0, 99, 42, 15, ...]  (chunk 2 most similar)

    # 6. Select top N chunks
    top_indices = sorted_indices[:n_top_chunks]
    top_chunks = [chunk_texts[i] for i in top_indices]

    # 7. Join and return
    return "\n\n".join(top_chunks)
```

### Example with Real Numbers

```
Query: "Find chemical formulas"
Query embedding: [0.23, -0.45, 0.12, ..., 0.89]

Chunk 0: "The material La₀.₈Sr₀.₂NiO₂ was synthesized..."
Embedding: [0.22, -0.44, 0.11, ..., 0.87]
Similarity: 0.923 ← High!

Chunk 1: "Previous studies have shown..."
Embedding: [-0.31, 0.15, -0.67, ..., 0.23]
Similarity: 0.342 ← Low

Chunk 2: "The formula Fe₂O₃ is commonly used..."
Embedding: [0.24, -0.46, 0.13, ..., 0.88]
Similarity: 0.947 ← Highest!

...

Top 5 chunks: [2, 0, 42, 78, 15] (by similarity)
```

## Retrieval Models

Different models create different embeddings, affecting retrieval quality.

### Model Characteristics

| Model | Dims | Training Data | Best For |
|-------|------|--------------|----------|
| `all-MiniLM-L6-v2` | 384 | General text (1B pairs) | General purpose, fast |
| `all-mpnet-base-v2` | 768 | General text (1B pairs) | Higher quality, slower |
| `msmarco-distilbert-base-v4` | 768 | MS MARCO (passage ranking) | Question answering |

### Why Model Choice Matters

Different models capture different semantic relationships:

```
Query: "superconductivity"

all-MiniLM-L6-v2 might rank highly:
1. "superconducting materials" (0.85)
2. "high-Tc compounds" (0.72)
3. "zero resistance" (0.68)

all-mpnet-base-v2 might rank highly:
1. "superconducting materials" (0.88)
2. "Cooper pairs" (0.79)  ← Better physics understanding
3. "zero resistance" (0.75)
4. "high-Tc compounds" (0.73)
```

More sophisticated models understand deeper relationships.

## Limitations and Solutions

### Limitation 1: Domain Gap

General models may miss domain-specific terminology:

```
Query: "DMFT calculation"
General model: Matches "calculation", may miss "DMFT" context
```

**Solution**: Use domain-specific models or fine-tune on your papers

### Limitation 2: Very Short Queries

Short queries have less semantic information:

```
Query: "DFT"
Embedding captures less context than:
Query: "Find all mentions of DFT calculations and parameters"
```

**Solution**: Use more descriptive queries

### Limitation 3: Chunk Size Matters

Very small chunks lack context:

```
Chunk: "1.2 eV"
Hard to determine relevance without context
```

Very large chunks dilute relevance:

```
Chunk: 3000 words covering multiple topics
May match query but contains mostly irrelevant content
```

**Solution**: Balance chunk size based on task (see [Explanation: Understanding Chunking Strategies](understanding_chunking.md))


## Advanced: How Models Learn Embeddings

Sentence transformer models are trained using **contrastive learning**:

1. **Positive pairs**: Similar sentences (paraphrases, translations)
   ```
   "The cat sat on the mat" ↔ "A feline rested on the rug"
   ```

2. **Negative pairs**: Different sentences
   ```
   "The cat sat on the mat" ↔ "Quantum mechanics explains..."
   ```

3. **Training objective**: Make positive pairs close, negative pairs far
   ```
   Similarity(positive pairs) → maximize
   Similarity(negative pairs) → minimize
   ```

4. **Result**: The model learns to encode semantic similarity

## Retrieval vs. Reranking

NERxiv uses **single-stage retrieval**, but some systems use **two-stage retrieval**:

### Single-Stage (NERxiv)
```
Chunks → Encode → Similarity → Top K chunks → LLM
```

### Two-Stage
```
Chunks → Encode → Similarity → Top 20 chunks
                ↓
      Reranker (cross-encoder)
                ↓
           Top 5 chunks → LLM
```

**Trade-off**: Two-stage is more accurate but slower

## Practical Considerations

### Number of Chunks to Retrieve

```bash
# Few chunks (3-5): Fast, focused, may miss information
nerxiv prompt --file-path paper.hdf5 --n-top-chunks 3

# Many chunks (10-15): Comprehensive, slower, may include noise
nerxiv prompt --file-path paper.hdf5 --n-top-chunks 12
```

**Rule of thumb**:
- Simple queries: 3-5 chunks
- Complex queries: 8-12 chunks
- Exploratory: 15+ chunks

### Query Engineering

Better queries lead to better retrieval:

❌ **Vague**: "methods"
   ✅ **Specific**: "computational and experimental methods used in the study"

❌ **Too narrow**: "DFT"
   ✅ **Inclusive**: "DFT, density functional theory, and other ab initio methods"

### Debugging Retrieval

Check similarity scores:

```python
from nerxiv.rag import CustomRetriever
from sentence_transformers import util

retriever = CustomRetriever(model="all-MiniLM-L6-v2", query="your query")

query_emb = retriever.model.encode(retriever.query, convert_to_tensor=True)
chunk_texts = [chunk.page_content for chunk in chunks]
chunk_embs = retriever.model.encode(chunk_texts, convert_to_tensor=True)

similarities = util.pytorch_cos_sim(query_emb, chunk_embs).squeeze(0)

# Print top 5
for i in similarities.argsort(descending=True)[:5]:
    print(f"Similarity: {similarities[i]:.4f}")
    print(f"Chunk: {chunk_texts[i][:200]}\n")
```

If top chunks have low similarity (<0.5), your query may need refinement.

## Summary

Semantic retrieval works by:

1. **Encoding** query and chunks into embeddings
2. **Computing** cosine similarity between embeddings
3. **Ranking** chunks by similarity score
4. **Selecting** top N chunks for the LLM

This approach:

- Understands meaning, not just keywords
- Works with synonyms and paraphrasing
- Enables precise relevance ranking
- Scales to large document collections
