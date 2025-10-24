# Glossary

This glossary provides definitions of key terms and concepts used throughout the NERxiv documentation and codebase.

## A

### arXiv
An open-access repository of scientific papers, primarily in physics, mathematics, computer science, and related fields. arXiv (pronounced "archive") is maintained by Cornell University and provides free access to preprints and published papers.

## C

### Chunk
A smaller segment of text extracted from a larger document. Chunking is necessary to process large documents within the token limits of LLMs and to improve retrieval accuracy in RAG systems.

### Chunker
A class responsible for splitting text into smaller, manageable pieces (chunks) for processing. NERxiv provides several chunker implementations:
- **Chunker**: Basic text splitter using recursive character-based splitting
- **SemanticChunker**: Splits text based on semantic similarity using sentence embeddings
- **AdvancedSemanticChunker**: Uses K-means clustering on sentence embeddings for more sophisticated semantic chunking

### Chunk Overlap
The number of characters or tokens that overlap between consecutive chunks. Overlap helps maintain context across chunk boundaries and prevents information loss at split points.

### Chunk Size
The maximum number of characters or tokens in a single chunk. Typical values range from 500 to 2000 characters depending on the use case and model constraints.

## E

### Embedding
A numerical vector representation of text that captures its semantic meaning. Embeddings allow machines to understand and compare text similarity mathematically. NERxiv uses models like `all-MiniLM-L6-v2` from SentenceTransformers to generate embeddings.

## L

### LLM (Large Language Model)
A type of artificial intelligence model trained on vast amounts of text data to understand and generate human-like text. Examples include Llama, DeepSeek, and Qwen. NERxiv uses LLMs locally via Ollama for metadata extraction.

### LangChain
A framework for developing applications powered by language models. NERxiv uses LangChain for document handling, text splitting, embeddings, and LLM integration.

## M

### Metadata
Structured information about scientific papers, such as methodology, chemical formulations, simulation parameters, and experimental conditions. NERxiv extracts this metadata from paper text using LLMs.

## N

### Named Entity Recognition (NER)
The task of identifying and classifying named entities (like chemical compounds, methods, or parameters) in text. While traditional NER uses rule-based or statistical methods, NERxiv uses LLMs and RAG for more flexible and accurate extraction.

### NERxiv
**N**amed **E**ntity **R**ecognition for ar**xiv** papers. The name of this package, which provides tools for extracting structured metadata from arXiv papers using LLMs and RAG techniques.

## O

### Ollama
A tool for running large language models locally on your machine. NERxiv uses Ollama to provide privacy-preserving, offline access to LLMs for metadata extraction.

## P

### Prompt
A carefully crafted input text given to an LLM to guide its response. NERxiv includes a system of prompts designed for extracting specific types of metadata from scientific papers.

### Prompt Engineering
The practice of designing and optimizing prompts to elicit desired responses from LLMs. Good prompt engineering is crucial for accurate metadata extraction.

### pyrxiv
A Python package (separate from NERxiv) for fetching, downloading, and extracting text from arXiv papers. NERxiv uses pyrxiv as its foundation for accessing arXiv content. See: [https://pypi.org/project/pyrxiv/](https://pypi.org/project/pyrxiv/)

## R

### RAG (Retrieval-Augmented Generation)
A technique that combines information retrieval with text generation. RAG first retrieves relevant document chunks based on a query, then uses those chunks as context for an LLM to generate an answer. This approach improves accuracy and reduces hallucinations.

### Retriever
A component that finds the most relevant text chunks for a given query. NERxiv provides retriever classes that use embedding models to calculate semantic similarity between queries and document chunks.

## S

### Semantic Similarity
A measure of how similar two pieces of text are in meaning, rather than just word overlap. NERxiv uses semantic similarity to find relevant chunks and group related sentences together.

### SentenceTransformers
A Python library for state-of-the-art sentence and text embeddings. NERxiv uses SentenceTransformers models (like `all-MiniLM-L6-v2`) for encoding text into embeddings.

## T

### Token
The basic unit of text that a language model processes. Tokens can be words, parts of words, or punctuation. Different models have different tokenization schemes and maximum token limits.

### Token Limit
The maximum number of tokens an LLM can process in a single request. Common limits range from 2048 to 8192 tokens. Chunking helps stay within these limits when processing long documents.

## V

### Vector Store
A database optimized for storing and querying embedding vectors. NERxiv uses LangChain's `InMemoryVectorStore` for efficient similarity search over document chunks.

---

## Related Resources

- [LangChain Documentation](https://python.langchain.com/)
- [SentenceTransformers Documentation](https://www.sbert.net/)
- [Ollama Documentation](https://ollama.com/)
- [arXiv.org](https://arxiv.org/)
- [pyrxiv Package](https://pypi.org/project/pyrxiv/)