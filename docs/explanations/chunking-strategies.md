# Understanding Chunking Strategies

Chunking is the first critical step in the RAG pipeline. This explanation explores why chunking matters, how different strategies work, and when to use each approach.

## Why Chunking is Necessary

Scientific papers are long documents that need to be divided into smaller pieces for several reasons:

### 1. Token Limits

LLMs have maximum context windows:
- GPT-3.5: 4,096 tokens (~3,000 words)
- LLaMA 3.1 8B: 8,192 tokens (~6,000 words)
- LLaMA 3.1 70B: 8,192 tokens (~6,000 words)

A typical arXiv paper:
- 8,000-15,000 words
- 10,000-20,000 tokens

Without chunking, most papers exceed the context limit.

### 2. Retrieval Efficiency

Even if a paper fits in context, you don't want to search through everything. Smaller chunks allow:
- **Precision**: Find exactly where information appears
- **Relevance scoring**: Rank specific passages by relevance
- **Focused context**: Give the LLM only what it needs

### 3. Information Density

Different parts of a paper have different information densities:
- Abstract: Very dense, overview
- Introduction: Context, motivation
- Methods: Technical details
- Results: Specific findings
- Discussion: Interpretation
- References: Citations (often irrelevant)

Chunking allows the retriever to select only the dense, relevant sections.

## Chunking Strategies in NERxiv

NERxiv implements three chunking strategies, each with different trade-offs.

### 1. Fixed-Size Chunking (Chunker)

**How it works**: Split text into chunks of fixed character length with overlap.

```python
from nerxiv.chunker import Chunker

chunker = Chunker(text=paper_text)
chunks = chunker.chunk_text(chunk_size=1000, chunk_overlap=200)
```

**Example**:
```
Text: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
chunk_size=10, chunk_overlap=3

Chunk 1: "ABCDEFGHIJ"
Chunk 2: "HIJKLMNOPQ" (HIJ overlaps)
Chunk 3: "OPQRSTUVWX" (OPQ overlaps)
Chunk 4: "WXYZ"
```

**Advantages**:
- ✅ Fast and simple
- ✅ Predictable chunk sizes
- ✅ No NLP dependencies
- ✅ Works for any language
- ✅ Consistent token counts for LLM

**Disadvantages**:
- ❌ Splits sentences arbitrarily
- ❌ Breaks semantic units
- ❌ May cut formulas or equations mid-way

**Best for**:
- Quick prototyping
- Batch processing many papers
- When speed matters more than perfect chunking
- Papers with uniform structure

**Parameters**:
- `chunk_size`: Characters per chunk (default: 1000)
  - Smaller (500-800): More precise retrieval, more chunks
  - Larger (1500-2000): More context per chunk, fewer chunks
- `chunk_overlap`: Overlap between chunks (default: 200)
  - Larger overlap: Less information loss, more redundancy
  - Smaller overlap: Less redundancy, faster processing

### 2. Semantic Chunking (SemanticChunker)

**How it works**: Uses spaCy to identify sentence boundaries and creates chunks at natural breaks.

```python
from nerxiv.chunker import SemanticChunker

chunker = SemanticChunker(text=paper_text)
chunks = chunker.chunk_text()
```

**Example**:
```
Text: "DFT was used. The bandgap is 1.2 eV. Previous work showed..."

Chunk 1: "DFT was used."
Chunk 2: "The bandgap is 1.2 eV."
Chunk 3: "Previous work showed different results."
```

**Advantages**:
- ✅ Preserves sentence integrity
- ✅ Natural semantic boundaries
- ✅ Better for extracting complete facts
- ✅ Doesn't break equations mid-way

**Disadvantages**:
- ❌ Variable chunk sizes
- ❌ Requires spaCy model (~500MB)
- ❌ Slower than fixed-size
- ❌ May create very small or very large chunks

**Best for**:
- Extracting specific facts (formulas, numbers, names)
- When sentence-level precision is important
- Papers with well-formed sentences
- Queries about discrete facts

**How it groups sentences**:

The chunker uses spaCy's sentence tokenizer and groups related sentences together based on semantic similarity, creating coherent chunks.

### 3. Advanced Semantic Chunking (AdvancedSemanticChunker)

**How it works**: Uses KMeans clustering on sentence embeddings to group semantically similar sentences together.

```python
from nerxiv.chunker import AdvancedSemanticChunker

chunker = AdvancedSemanticChunker(text=paper_text)
chunks = chunker.chunk_text(n_chunks=10)
```

**Process**:
1. Split text into sentences with spaCy
2. Encode each sentence with SentenceTransformer
3. Cluster sentences using KMeans
4. Group sentences by cluster into chunks

**Example**:
```
Sentences:
1. "We synthesize La₀.₈Sr₀.₂NiO₂ samples." → Cluster 0 (synthesis)
2. "DFT calculations were performed." → Cluster 1 (computation)
3. "The bandgap was measured." → Cluster 2 (results)
4. "Samples were annealed at 800°C." → Cluster 0 (synthesis)
5. "Electronic structure was calculated." → Cluster 1 (computation)

Resulting chunks:
Chunk 0: [Sentence 1, Sentence 4] (synthesis topic)
Chunk 1: [Sentence 2, Sentence 5] (computation topic)
Chunk 2: [Sentence 3] (results topic)
```

**Advantages**:
- ✅ Topically coherent chunks
- ✅ Groups related information even if not adjacent
- ✅ Excellent for complex, multi-topic papers
- ✅ Better context for the LLM

**Disadvantages**:
- ❌ Slowest (computes embeddings for all sentences)
- ❌ Requires SentenceTransformer model
- ❌ Variable and unpredictable chunk sizes
- ❌ May group unrelated sentences if n_chunks is too small

**Best for**:
- Papers covering multiple topics
- Extracting methodology descriptions
- When topical coherence is crucial
- Complex queries requiring context

**Parameters**:
- `n_chunks`: Number of semantic clusters (default: 10)
  - Smaller (5-8): Broad topics, larger chunks
  - Larger (15-20): Finer topics, smaller chunks

## The Impact of Chunking on Retrieval

Different chunking strategies affect what the retriever finds.

### Example: Extracting "bandgap" Information

**Original text**:
```
The electronic structure of La₀.₈Sr₀.₂NiO₂ was investigated using DFT.
Our calculations show a direct bandgap of 1.2 eV at the Gamma point.
This is consistent with optical measurements performed at room temperature.
```

**Fixed-size chunking (chunk_size=50, no overlap)**:
```
Chunk 1: "The electronic structure of La₀.₈Sr₀.₂NiO₂ was in"
Chunk 2: "vestigated using DFT. Our calculations show a d"
Chunk 3: "irect bandgap of 1.2 eV at the Gamma point. Th"
Chunk 4: "is is consistent with optical measurements..."
```
❌ "bandgap" split across chunks 2-3, may be missed!

**Fixed-size with overlap (chunk_size=50, overlap=20)**:
```
Chunk 1: "The electronic structure of La₀.₈Sr₀.₂NiO₂ was in"
Chunk 2: "La₀.₈Sr₀.₂NiO₂ was investigated using DFT. Our ca"
Chunk 3: "Our calculations show a direct bandgap of 1.2 eV"
Chunk 4: "bandgap of 1.2 eV at the Gamma point. This is..."
```
✅ "bandgap" appears complete in chunks 3-4

**Semantic chunking**:
```
Chunk 1: "The electronic structure of La₀.₈Sr₀.₂NiO₂ was investigated using DFT."
Chunk 2: "Our calculations show a direct bandgap of 1.2 eV at the Gamma point."
Chunk 3: "This is consistent with optical measurements performed at room temperature."
```
✅ Complete bandgap statement in one chunk

**Advanced semantic chunking**:
```
Chunk 1 (Electronic structure topic):
"The electronic structure of La₀.₈Sr₀.₂NiO₂ was investigated using DFT. Our calculations show a direct bandgap of 1.2 eV at the Gamma point."

Chunk 2 (Experimental validation topic):
"This is consistent with optical measurements performed at room temperature."
```
✅ Full context in one topical chunk

## Choosing the Right Strategy

| Your Goal | Recommended Chunker | Why |
|-----------|-------------------|-----|
| Speed and simplicity | `Chunker` | No overhead |
| Extract specific facts | `SemanticChunker` | Sentence boundaries |
| Multi-topic papers | `AdvancedSemanticChunker` | Topical grouping |
| Formulas and equations | `SemanticChunker` | Preserves units |
| Long methodology sections | `AdvancedSemanticChunker` | Groups related content |
| Batch processing | `Chunker` | Fastest |

## Impact on Different Queries

### Query: "What materials were studied?"

**Best chunker**: `SemanticChunker` or `AdvancedSemanticChunker`
- Material names often appear in discrete sentences
- Semantic boundaries help isolate mentions

### Query: "What computational methods were used?"

**Best chunker**: `AdvancedSemanticChunker`
- Methods often described across multiple sentences
- Topical grouping captures full descriptions

### Query: "What was the temperature during synthesis?"

**Best chunker**: `SemanticChunker`
- Specific fact usually in one sentence
- Sentence boundaries preserve "800°C" with context

### Query: "Summarize the main findings"

**Best chunker**: Any (depends on speed requirements)
- Summary requires broad context
- Retrieved chunks will be diverse regardless

## Advanced Chunking Considerations

### Chunk Size vs. Number of Chunks Retrieved

Small chunks + many retrieved:
- Pro: Precise information
- Con: More redundancy, slower

Large chunks + few retrieved:
- Pro: More context, faster
- Con: Less precise, may include noise

**Example**:
```bash
# Precise: small chunks, more retrieved
nerxiv prompt --file-path paper.hdf5 --chunker Chunker --n-top-chunks 10

# Fast: large chunks (via semantic), fewer retrieved
nerxiv prompt --file-path paper.hdf5 --chunker AdvancedSemanticChunker --n-top-chunks 5
```

### Overlap in Fixed-Size Chunking

Overlap ensures important information isn't lost at boundaries:

**No overlap**:
```
Chunk 1: "...the bandgap of"
Chunk 2: "1.2 eV was measured..."
```
❌ Incomplete information in each chunk

**With overlap (200 chars)**:
```
Chunk 1: "...the bandgap of 1.2 eV..."
Chunk 2: "...the bandgap of 1.2 eV was measured..."
```
✅ Complete information in both chunks (redundancy is okay!)

**Rule of thumb**: Overlap should be ~20% of chunk size

## Debugging Chunking

To see what chunks are created:

```python
from nerxiv.chunker import SemanticChunker

chunker = SemanticChunker(text=paper_text)
chunks = chunker.chunk_text()

print(f"Total chunks: {len(chunks)}")
for i, chunk in enumerate(chunks[:5]):
    print(f"\n=== Chunk {i} ===")
    print(f"Length: {len(chunk.page_content)} chars")
    print(f"Content: {chunk.page_content[:200]}...")
```

## Related Topics

- [What is RAG?](what-is-rag.md) - Understanding the full pipeline
- [How Semantic Retrieval Works](semantic-retrieval.md) - The next stage after chunking
- [How to Customize Chunking Strategies](../howtos/customize-chunking.md) - Practical implementation
