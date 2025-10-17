# NERxiv Architecture Analysis & Improvement Proposal

**Date:** October 17, 2025  
**Author:** GitHub Copilot Analysis

## Executive Summary

After inspecting the existing `cli/run_prompt.py`, prompt registry system, and datamodel architecture, I've identified several opportunities to improve the overall package design while preserving your excellent prompt-registry-driven approach. This document compares the existing architecture with the new `RAGExtractorAgent` and proposes a unified, scalable solution for multi-agent workflows.

---

## Current Architecture: Strengths

### 1. **Prompt Registry Pattern** ✅ Excellent Design
Your `PROMPT_REGISTRY` with `PromptRegistryEntry` is a very clean pattern:
- **Declarative configuration**: Each entry bundles `retriever_query` + `prompt` together
- **Pydantic-based prompts**: Type-safe, self-documenting with `BasePrompt` → `Prompt` → `StructuredPrompt`
- **Schema-driven extraction**: `StructuredPrompt` automatically generates prompts from Pydantic models (e.g., `ChemicalFormulation`)
- **Examples as first-class citizens**: Few-shot learning built into the prompt structure

**Key strength:** You've separated *what to extract* (datamodel) from *how to extract it* (prompt template logic).

### 2. **Datamodel-First Approach** ✅
Using Pydantic models (`ChemicalFormulation`, `Method`, `Simulation`) as the target schema is the right choice:
- Self-documenting field descriptions
- Type validation
- Easy serialization to JSON
- Integration with pymatgen for chemistry validation

### 3. **CLI-Driven Workflow** ✅
The `cli/cli.py` → `run_prompt_paper()` pattern provides:
- Good separation of concerns (CLI parsing vs. execution logic)
- Flexible `--llm-option` parsing for arbitrary LLM parameters
- Per-paper processing with timing metrics

---

## Current Architecture: Areas for Improvement

### 1. **Tight Coupling to HDF5** ⚠️
**Issue:** `run_prompt_paper()` is tightly coupled to HDF5 storage:
- Requires `.hdf5` file on disk
- Reads text from HDF5, writes results back to HDF5
- Hard to unit test without creating HDF5 fixtures
- Storage logic mixed with orchestration logic

**Impact:**
- Can't easily swap storage backends (SQLite, PostgreSQL, MongoDB, Parquet)
- Can't run the RAG pipeline in-memory for testing
- Hard to integrate with streaming/async workflows

### 2. **No Agent Abstraction** ⚠️
**Issue:** The orchestration logic is embedded in `run_prompt_paper()`:
```python
# Current: All in one function
chunker_cls = _CHUNKER_MAP.get(chunker, Chunker)(text=text)
chunks = chunker_cls.chunk_text()
retriever = CustomRetriever(model=retriever_model, query=retriever_query, logger=logger)
text = retriever.get_relevant_chunks(chunks=chunks, n_top_chunks=n_top_chunks)
generator = LLMGenerator(model=model, text=text, logger=logger, **kwargs)
built_prompt = prompt.build(text=text)
answer = generator.generate(prompt=built_prompt)
```

**Problems:**
- Hard to reuse this pipeline outside of HDF5 context
- Can't easily compose multiple agents (e.g., "extract material, then check if DMFT was used")
- Difficult to add intermediate steps (validation, filtering, post-processing)

### 3. **Metadata Storage in HDF5** ⚠️
**Issue:** Using HDF5 for metadata has trade-offs:

**Pros:**
- Self-contained: data + metadata in one file
- Hierarchical structure
- Good for numerical arrays

**Cons:**
- Not queryable (can't do SQL-like queries across papers)
- Hard to version/migrate schemas
- Awkward for text/JSON data (everything must be encoded as bytes)
- No concurrent writes (file locking issues)
- Tooling: requires h5py to inspect data

**Better alternatives for metadata:**
1. **SQLite** (recommended for single-user, local):
   - Queryable with SQL
   - Good Python support (stdlib `sqlite3`)
   - Easy to back up (single file)
   - ACID transactions
   - Full-text search available

2. **PostgreSQL** (for multi-user, production):
   - Concurrent access
   - Advanced querying (JSONB for flexible schema)
   - Strong ecosystem

3. **Parquet + DuckDB** (for analytics):
   - Columnar storage
   - Fast queries with DuckDB
   - Works well with pandas/polars

**Recommendation:** Use **SQLite** for local development and small-to-medium datasets (up to millions of papers), with a clean ORM layer (SQLAlchemy or similar) to abstract storage.

### 4. **No Validation/Post-Processing Pipeline** ⚠️
**Issue:** The answer is stored raw without validation:
```python
answer = generator.generate(prompt=built_prompt)
# ... directly saved to HDF5, no validation
```

**Missing:**
- Validation that the answer matches the expected schema (e.g., `ChemicalFormulation`)
- Parsing JSON from LLM output
- Error recovery (retry, fallback)
- Confidence scoring

---

## Proposed Architecture: Unified Multi-Agent System

### Vision
Create a **composable agent framework** that:
1. Separates orchestration from storage
2. Supports multi-agent workflows (chaining, branching)
3. Works with your existing prompt registry
4. Provides pluggable storage backends
5. Enables testing without I/O

---

### Design: Core Components

#### 1. **Enhanced `RAGExtractorAgent`** (already implemented ✓)
```python
from nerxiv.rag.agent import RAGExtractorAgent

agent = RAGExtractorAgent(
    chunker=Chunker,
    retriever=CustomRetriever,
    generator=LLMGenerator,
)

result = agent.extract(
    text=paper_text,
    query=retriever_query,
    n_top_chunks=5,
    prompt_template=built_prompt,  # from your registry
)
# Returns: {"chunks": [...], "retrieved": "...", "prompt": "...", "answer": "..."}
```

**Benefits:**
- In-memory execution (no HDF5 dependency)
- Testable with fake components
- Reusable across different contexts

#### 2. **New: `PromptAgent` (Schema-Aware Wrapper)**
Extends `RAGExtractorAgent` to work with your prompt registry:

```python
from nerxiv.agents import PromptAgent

agent = PromptAgent(
    query_name="material_formula",  # from PROMPT_REGISTRY
    chunker="Chunker",
    retriever_model="all-MiniLM-L6-v2",
    llm_model="deepseek-r1",
)

result = agent.run(text=paper_text, n_top_chunks=5)
# Returns validated Pydantic model or dict
```

**Features:**
- Automatically loads `retriever_query` and `prompt` from registry
- Validates LLM output against expected schema (for `StructuredPrompt`)
- Parses JSON and returns typed results
- Logs failures with original prompt/answer for debugging

#### 3. **New: `MultiAgent` (Orchestrator)**
Chain multiple agents with conditional logic:

```python
from nerxiv.agents import MultiAgent, PromptAgent

workflow = MultiAgent(
    agents=[
        PromptAgent("material_formula"),  # Extract material
        PromptAgent("only_dmft"),         # Check if DMFT is used
    ],
    strategy="sequential",  # or "parallel", "conditional"
)

results = workflow.run(text=paper_text)
# Returns: [result1, result2]
```

**Use cases:**
- Sequential: "Extract material, then extract methods only if material is not a model"
- Parallel: "Extract material and methods simultaneously"
- Conditional: "Only run agent2 if agent1 returns True"

#### 4. **Storage Abstraction Layer**
Create a clean interface for metadata persistence:

```python
from nerxiv.storage import PaperStore

# SQLite backend
store = PaperStore("sqlite:///data/papers.db")

# Save extraction result
store.save_extraction(
    arxiv_id="1234.5678",
    query="material_formula",
    run_metadata={
        "retriever_model": "all-MiniLM-L6-v2",
        "llm_model": "deepseek-r1",
        "n_top_chunks": 5,
        "timestamp": datetime.now().isoformat(),
    },
    result={
        "chunks": [...],
        "retrieved": "...",
        "prompt": "...",
        "answer": "...",
        "parsed": {...},  # Validated Pydantic model
    }
)

# Query results
papers_with_dmft = store.query(
    "SELECT arxiv_id FROM extractions WHERE query='only_dmft' AND parsed->>'answer'='True'"
)
```

**Benefits:**
- Swap backends without changing code (SQLite → PostgreSQL)
- SQL queries for filtering/analytics
- Easy migration scripts
- Better tooling (DB browsers, ORMs)

---

### Proposed File Structure

```
nerxiv/
  agents/
    __init__.py
    base.py              # BaseAgent interface
    rag_agent.py         # RAGExtractorAgent (current implementation)
    prompt_agent.py      # PromptAgent (registry-aware)
    multi_agent.py       # MultiAgent (orchestrator)
    validators.py        # JSON parsing, schema validation
  
  storage/
    __init__.py
    base.py              # Abstract StorageBackend
    sqlite.py            # SQLiteStore
    hdf5.py              # HDF5Store (legacy, for migration)
    models.py            # SQLAlchemy models for paper metadata
  
  cli/
    cli.py               # Updated to use agents
    run_prompt.py        # Refactored to use PromptAgent + storage layer
  
  prompts/
    prompts_registry.py  # Keep as-is (excellent!)
    prompts.py           # Keep as-is
  
  migrations/
    hdf5_to_sqlite.py    # Migration script
```

---

### Migration Path (Backward Compatible)

**Phase 1: Add Agents (Done ✓)**
- ✅ `RAGExtractorAgent` implemented
- ✅ Tested with fake components

**Phase 2: Prompt-Aware Agent (Next)**
- Create `PromptAgent` that wraps `RAGExtractorAgent`
- Integrate with `PROMPT_REGISTRY`
- Add JSON parsing + Pydantic validation

**Phase 3: Storage Layer**
- Implement `SQLiteStore` with schema migration
- Add migration script: `hdf5_to_sqlite.py`
- Keep HDF5 support as legacy option

**Phase 4: Multi-Agent Support**
- Implement `MultiAgent` orchestrator
- Add conditional/parallel execution
- Create tutorial showing multi-step extraction

**Phase 5: Update CLI**
- Refactor `run_prompt_paper()` to use agents
- Add `--storage` flag: `--storage sqlite` or `--storage hdf5`
- Keep existing CLI interface (backward compatible)

---

## Specific Code Improvements

### 1. **Improve `RAGExtractorAgent` Integration**

Current gap: `RAGExtractorAgent` doesn't know about your prompt registry. Bridge this:

```python
# nerxiv/agents/prompt_agent.py
from nerxiv.prompts import PROMPT_REGISTRY
from nerxiv.rag.agent import RAGExtractorAgent
from nerxiv.chunker import _CHUNKER_MAP
from nerxiv.rag.retriever import CustomRetriever
from nerxiv.rag.generator import LLMGenerator

class PromptAgent:
    """Executes a registered prompt with RAG pipeline."""
    
    def __init__(
        self,
        query_name: str,
        chunker: str = "Chunker",
        retriever_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "deepseek-r1",
        **llm_kwargs,
    ):
        if query_name not in PROMPT_REGISTRY:
            raise ValueError(f"Unknown query: {query_name}")
        
        self.entry = PROMPT_REGISTRY[query_name]
        self.query_name = query_name
        self.chunker_cls = _CHUNKER_MAP[chunker]
        self.retriever_model = retriever_model
        self.llm_model = llm_model
        self.llm_kwargs = llm_kwargs
    
    def run(self, text: str, n_top_chunks: int = 5) -> dict:
        """Execute the RAG pipeline and validate output."""
        agent = RAGExtractorAgent(
            chunker=self.chunker_cls,
            retriever=CustomRetriever,
            generator=LLMGenerator,
            retriever_kwargs={"model": self.retriever_model},
            generator_kwargs={"model": self.llm_model, **self.llm_kwargs},
        )
        
        # Build prompt from registry
        built_prompt = self.entry.prompt.build(text=text)
        
        result = agent.extract(
            text=text,
            query=self.entry.retriever_query,
            n_top_chunks=n_top_chunks,
            prompt_template=built_prompt,
        )
        
        # Validate if StructuredPrompt
        if isinstance(self.entry.prompt, StructuredPrompt):
            result["parsed"] = self._parse_and_validate(result["answer"])
        
        return result
    
    def _parse_and_validate(self, answer: str) -> dict | None:
        """Parse JSON from LLM answer and validate against schema."""
        try:
            # Extract JSON from markdown code block
            import re
            json_match = re.search(r"```json\n(.*?)\n```", answer, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                # Validate with Pydantic
                schema = self.entry.prompt.output_schema
                validated = schema(**data)
                return validated.model_dump()
        except Exception as e:
            logger.error(f"Failed to parse answer: {e}")
            return None
```

### 2. **SQLite Storage Backend**

```python
# nerxiv/storage/sqlite.py
import sqlite3
import json
from datetime import datetime
from pathlib import Path

class SQLiteStore:
    def __init__(self, db_path: str = "data/papers.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                arxiv_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS extractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arxiv_id TEXT NOT NULL,
                query TEXT NOT NULL,
                retriever_model TEXT,
                llm_model TEXT,
                n_top_chunks INTEGER,
                timestamp TEXT NOT NULL,
                chunks TEXT,  -- JSON array
                retrieved TEXT,
                prompt TEXT,
                answer TEXT,
                parsed TEXT,  -- JSON object
                elapsed_time REAL,
                FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_arxiv_query ON extractions(arxiv_id, query)")
        conn.commit()
        conn.close()
    
    def save_paper(self, arxiv_id: str, text: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO papers (arxiv_id, text, created_at) VALUES (?, ?, ?)",
            (arxiv_id, text, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    
    def save_extraction(self, arxiv_id: str, query: str, metadata: dict, result: dict):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO extractions 
            (arxiv_id, query, retriever_model, llm_model, n_top_chunks, timestamp,
             chunks, retrieved, prompt, answer, parsed, elapsed_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            arxiv_id,
            query,
            metadata.get("retriever_model"),
            metadata.get("llm_model"),
            metadata.get("n_top_chunks"),
            metadata.get("timestamp", datetime.now().isoformat()),
            json.dumps([c.page_content for c in result.get("chunks", [])]),
            result.get("retrieved"),
            result.get("prompt"),
            result.get("answer"),
            json.dumps(result.get("parsed")) if result.get("parsed") else None,
            metadata.get("elapsed_time"),
        ))
        conn.commit()
        conn.close()
    
    def query(self, sql: str, params: tuple = ()):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(sql, params)
        results = cursor.fetchall()
        conn.close()
        return results
```

### 3. **Updated `run_prompt_paper()` (Backward Compatible)**

```python
# nerxiv/cli/run_prompt.py (refactored)
from nerxiv.agents.prompt_agent import PromptAgent
from nerxiv.storage.sqlite import SQLiteStore

def run_prompt_paper(
    paper: Path,
    chunker: str = "Chunker",
    retriever_model: str = "all-MiniLM-L6-v2",
    n_top_chunks: int = 5,
    model: str = "gpt-oss:20b",
    query: str = "material_formula",
    storage_backend: str = "hdf5",  # or "sqlite"
    **kwargs,
) -> float:
    start_time = time.time()
    
    # Create agent
    agent = PromptAgent(
        query_name=query,
        chunker=chunker,
        retriever_model=retriever_model,
        llm_model=model,
        **kwargs,
    )
    
    # Load text (support both HDF5 and plain text)
    if paper.suffix == ".hdf5":
        with h5py.File(paper, "r") as f:
            arxiv_id = paper.stem
            text = f[arxiv_id]["arxiv_paper"]["text"][()].decode("utf-8")
    else:
        text = paper.read_text()
        arxiv_id = paper.stem
    
    # Run extraction
    result = agent.run(text=text, n_top_chunks=n_top_chunks)
    
    # Save to storage
    elapsed = time.time() - start_time
    metadata = {
        "retriever_model": retriever_model,
        "llm_model": model,
        "n_top_chunks": n_top_chunks,
        "timestamp": datetime.now().isoformat(),
        "elapsed_time": elapsed,
    }
    
    if storage_backend == "sqlite":
        store = SQLiteStore()
        store.save_paper(arxiv_id, text)
        store.save_extraction(arxiv_id, query, metadata, result)
    else:  # hdf5 (legacy)
        _save_to_hdf5(paper, query, metadata, result)
    
    return elapsed
```

---

## Recommendations Summary

### Immediate Actions
1. ✅ **Keep `RAGExtractorAgent`** as-is (good foundation)
2. ✅ **Keep prompt registry** (excellent design)
3. **Add `PromptAgent`** to bridge registry with agent (priority: HIGH)
4. **Implement SQLite storage** as alternative to HDF5 (priority: HIGH)
5. **Add validation layer** for structured prompts (priority: MEDIUM)

### Medium-Term
- Implement `MultiAgent` for chaining workflows
- Add migration tool: `hdf5_to_sqlite.py`
- Create tutorial: "Multi-step extraction pipeline"

### Long-Term
- Async/streaming support for large batches
- Add caching layer (avoid re-extracting same paper)
- Web UI for browsing extractions (SQLite makes this easy)

---

## Storage Comparison: HDF5 vs SQLite

| Feature | HDF5 (Current) | SQLite (Proposed) |
|---------|---------------|-------------------|
| **Query capability** | ❌ No (must read all files) | ✅ Full SQL support |
| **Concurrent access** | ❌ File locking issues | ✅ Readers + 1 writer |
| **Schema evolution** | ⚠️ Manual migrations | ✅ ALTER TABLE |
| **Tooling** | ⚠️ h5py only | ✅ Many GUIs, ORMs |
| **Text data** | ⚠️ Must encode/decode | ✅ Native support |
| **JSON storage** | ❌ Store as string | ✅ JSONB (queryable) |
| **File size** | ⚠️ One file per paper | ✅ Single DB file |
| **Backup** | ⚠️ Copy all files | ✅ Copy one file |
| **Portability** | ✅ Cross-platform | ✅ Cross-platform |
| **Numerical arrays** | ✅ Excellent | ⚠️ Store as BLOB/JSON |

**Verdict:** For metadata extraction use case, SQLite is superior. Keep HDF5 if you add numerical data (embeddings, matrices) later.

---

## Next Steps

1. **Review this proposal** and decide on priorities
2. **Implement `PromptAgent`** (I can do this now)
3. **Add SQLite backend** with migration script
4. **Update documentation** with new workflow examples
5. **Create tutorial**: "From HDF5 to SQLite: Migration Guide"

Would you like me to:
- **A)** Implement `PromptAgent` now (bridges registry + `RAGExtractorAgent`)
- **B)** Implement SQLite storage backend with schema
- **C)** Create migration script from HDF5 to SQLite
- **D)** All of the above in sequence

Let me know your preference!
