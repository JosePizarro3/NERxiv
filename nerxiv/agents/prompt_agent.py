"""Prompt-aware agent that integrates with the prompt registry.

This agent provides a high-level interface for executing extraction workflows
using prompts defined in the PROMPT_REGISTRY. It automatically:
- Loads the appropriate retriever query and prompt template
- Executes the RAG pipeline (chunking → retrieval → generation)
- Parses and validates LLM output for structured prompts
- Returns typed results with error handling
"""

import json
import re
from typing import Any, Optional

from nerxiv.chunker import _CHUNKER_MAP, Chunker
from nerxiv.logger import logger
from nerxiv.prompts.prompts import StructuredPrompt
from nerxiv.prompts.prompts_registry import PROMPT_REGISTRY
from nerxiv.rag import RAGExtractorAgent
from nerxiv.rag.agents import BaseAgent


class PromptAgent(BaseAgent):
    """Registry-aware agent for executing extraction workflows.

    This agent bridges the gap between the prompt registry and the low-level
    RAGExtractorAgent. It handles:
    - Loading prompts from PROMPT_REGISTRY
    - Building prompts with proper templates
    - Parsing and validating structured output
    - Error recovery and logging

    Example:
        >>> agent = PromptAgent(
        ...     query_name="material_formula",
        ...     chunker="Chunker",
        ...     retriever_model="all-MiniLM-L6-v2",
        ...     llm_model="deepseek-r1",
        ... )
        >>> result = agent.run(text=paper_text, n_top_chunks=5)
        >>> print(result["parsed"])  # Validated Pydantic model output
    """

    def __init__(
        self,
        query_name: str,
        chunker: str = "Chunker",
        retriever_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "deepseek-r1",
        **llm_kwargs,
    ):
        """Initialize the PromptAgent.

        Args:
            query_name: Name of the query in PROMPT_REGISTRY (e.g., "material_formula")
            chunker: Name of the chunker class to use
            retriever_model: Model name for the retriever
            llm_model: Model name for the LLM generator
            **llm_kwargs: Additional kwargs passed to the LLM (temperature, format, etc.)

        Raises:
            ValueError: If query_name is not found in PROMPT_REGISTRY
            KeyError: If chunker is not found in the chunker map
        """
        if query_name not in PROMPT_REGISTRY:
            available = list(PROMPT_REGISTRY.keys())
            raise ValueError(
                f"Unknown query: '{query_name}'. Available queries: {available}"
            )

        if chunker not in _CHUNKER_MAP:
            available = list(_CHUNKER_MAP.keys())
            raise KeyError(
                f"Unknown chunker: '{chunker}'. Available chunkers: {available}"
            )

        self.query_name = query_name
        self.entry = PROMPT_REGISTRY[query_name]
        self.chunker_cls = _CHUNKER_MAP[chunker]
        self.retriever_model = retriever_model
        self.llm_model = llm_model
        self.llm_kwargs = llm_kwargs

        logger.info(
            f"PromptAgent initialized for query '{query_name}' "
            f"with {chunker} chunker and {llm_model} model"
        )

    def run(self, text: str, n_top_chunks: int = 5, **kwargs) -> dict[str, Any]:
        """Execute the full RAG pipeline with prompt registry integration.

        This method:
        1. Creates a RAGExtractorAgent with the configured components
        2. Loads the retriever query and prompt from the registry
        3. Executes the extraction workflow
        4. Parses and validates the output (for StructuredPrompt)
        5. Returns comprehensive results including parsed data

        Args:
            text: Input text to process
            n_top_chunks: Number of chunks to retrieve (default: 5)
            **kwargs: Additional parameters passed to the RAG agent

        Returns:
            Dictionary with keys:
            - chunks: List of chunk objects from the chunker
            - retrieved: String of concatenated top chunks
            - prompt: The full prompt sent to the LLM
            - answer: Raw LLM output
            - parsed: Validated output (only for StructuredPrompt, None otherwise)
            - parse_error: Error message if parsing failed (only if error occurred)

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        # Lazy import to avoid circular dependencies and optional deps
        from nerxiv.rag.generator import LLMGenerator
        from nerxiv.rag.retriever import CustomRetriever

        # Create the low-level RAG agent
        agent = RAGExtractorAgent(
            chunker=self.chunker_cls,
            retriever=CustomRetriever,
            generator=LLMGenerator,
            retriever_kwargs={"model": self.retriever_model},
            generator_kwargs={"model": self.llm_model, **self.llm_kwargs},
        )

        logger.info(f"Running PromptAgent for query '{self.query_name}'")

        # The prompt registry entry contains both the retriever query and prompt
        retriever_query = self.entry.retriever_query

        # We need to build the prompt, but RAGExtractorAgent expects a template
        # So we'll use a dummy context and replace it later
        # This is a bit hacky but maintains backward compatibility
        result = agent.extract(
            text=text,
            query=retriever_query,
            n_top_chunks=n_top_chunks,
            prompt_template="{context}",  # Temporary placeholder
            **kwargs,
        )

        # Now build the actual prompt with the retrieved context
        built_prompt = self.entry.prompt.build(text=result["retrieved"])
        result["prompt"] = built_prompt

        # Re-generate with the correct prompt
        # This is inefficient (generates twice) but cleaner for now
        # TODO: Refactor RAGExtractorAgent to accept a prompt builder callable
        from nerxiv.rag.generator import LLMGenerator

        generator = LLMGenerator(
            model=self.llm_model,
            text=result["retrieved"],
            **self.llm_kwargs,
        )
        result["answer"] = generator.generate(prompt=built_prompt)

        # Parse and validate output for structured prompts
        if isinstance(self.entry.prompt, StructuredPrompt):
            parsed, error = self._parse_and_validate(result["answer"])
            result["parsed"] = parsed
            if error:
                result["parse_error"] = error
                logger.warning(f"Failed to parse structured output: {error}")
        else:
            result["parsed"] = None

        logger.info(f"PromptAgent completed for query '{self.query_name}'")
        return result

    def _parse_and_validate(
        self, answer: str
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Parse JSON from LLM answer and validate against schema.

        This method attempts to:
        1. Extract JSON from markdown code blocks (```json...```)
        2. Parse the JSON string
        3. Validate against the Pydantic schema
        4. Return the validated data

        Args:
            answer: Raw LLM output string

        Returns:
            Tuple of (parsed_data, error_message)
            - If successful: (validated_dict, None)
            - If failed: (None, error_message)
        """
        try:
            # Try to extract JSON from markdown code block
            json_match = re.search(
                r"```json\s*\n(.*?)\n\s*```", answer, re.DOTALL | re.IGNORECASE
            )

            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                # Look for content between { and } or [ and ]
                json_match = re.search(r"(\{.*\}|\[.*\])", answer, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    return None, "No JSON found in answer"

            # Parse JSON
            data = json.loads(json_str)

            # Extract the model data (might be nested under model name)
            schema = self.entry.prompt.output_schema
            model_name = schema.__name__

            # Check if data is wrapped in model name key
            if isinstance(data, dict) and model_name in data:
                model_data = data[model_name]
            else:
                model_data = data

            # Validate with Pydantic
            if isinstance(model_data, list):
                # Multiple instances
                validated = [schema(**item) for item in model_data]
                return [v.model_dump() for v in validated], None
            else:
                # Single instance
                validated = schema(**model_data)
                return validated.model_dump(), None

        except json.JSONDecodeError as e:
            return None, f"JSON decode error: {e}"
        except Exception as e:
            return None, f"Validation error: {e}"
