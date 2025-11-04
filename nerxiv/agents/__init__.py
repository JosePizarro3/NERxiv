"""Agents package for orchestrating RAG workflows.

This package provides agent abstractions for composing chunker, retriever,
and generator components into end-to-end extraction pipelines.

Available agents:
- RAGExtractorAgent: Low-level orchestration of chunker → retriever → generator
- PromptAgent: High-level agent that integrates with the prompt registry
"""

# Lazy import to avoid pulling optional dependencies
__all__ = ["RAGExtractorAgent", "PromptAgent"]


def __getattr__(name):
    """Lazy loading for PromptAgent to avoid import errors."""
    if name == "PromptAgent":
        from nerxiv.agents.prompt_agent import PromptAgent

        return PromptAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
