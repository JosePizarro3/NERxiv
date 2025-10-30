"""
Ensemble module for running multiple LLM prompts and averaging results to minimize hallucinations.

This module provides functionality to:
1. Run multiple LLM prompts with different configurations (models, chunks, temperatures)
2. Execute prompts in parallel for efficiency
3. Average JSON results from StructuredPrompts using an LLM
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.documents import Document

from nerxiv.logger import logger
from nerxiv.prompts.prompts import BasePrompt, StructuredPrompt
from nerxiv.rag import LLMGenerator


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """
    Extract JSON object from text that may contain markdown code blocks or other formatting.

    Args:
        text (str): The text containing JSON

    Returns:
        dict[str, Any] | None: Extracted JSON object or None if parsing fails
    """
    # Try to find JSON in markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try parsing the entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def run_single_llm_prompt(
    prompt: BasePrompt,
    text: str,
    model: str,
    temperature: float,
    run_id: int,
    **kwargs,
) -> tuple[int, str]:
    """
    Run a single LLM prompt with given configuration.

    Args:
        prompt (BasePrompt): The prompt to use
        text (str): The text to process
        model (str): The LLM model to use
        temperature (float): The temperature parameter
        run_id (int): Identifier for this run
        **kwargs: Additional arguments for LLMGenerator

    Returns:
        tuple[int, str]: Run ID and the generated answer
    """
    logger.info(
        f"Running LLM prompt #{run_id} with model={model}, temperature={temperature}"
    )
    generator = LLMGenerator(
        model=model, text=text, temperature=temperature, logger=logger, **kwargs
    )
    built_prompt = prompt.build(text=text)
    answer = generator.generate(prompt=built_prompt)
    return run_id, answer


def average_json_results(
    results: list[dict[str, Any]],
    output_schema: type | None = None,
    averaging_model: str = "gpt-oss:20b",
    averaging_temperature: float = 0.2,
) -> dict[str, Any]:
    """
    Average multiple JSON results using an LLM to produce a consensus result.

    Args:
        results (list[dict[str, Any]]): List of JSON results to average
        output_schema (type | None): The output schema (BaseModel class) for the results
        averaging_model (str): The LLM model to use for averaging
        averaging_temperature (float): Temperature for the averaging LLM

    Returns:
        dict[str, Any]: The averaged/consensus JSON result
    """
    if not results:
        return {}

    if len(results) == 1:
        return results[0]

    # Create a prompt for the averaging LLM
    schema_info = ""
    if output_schema:
        schema_name = output_schema.__name__
        schema_description = output_schema.__doc__ or "structured data"
        schema_info = (
            f"\nThe expected output schema is '{schema_name}': {schema_description}"
        )

    averaging_prompt = f"""You are a data aggregation expert. You have been given {len(results)} different JSON results 
that represent the same information extracted from a scientific text. Your task is to create a single, consensus 
JSON result that best represents the information across all inputs.{schema_info}

Guidelines:
1. For string fields: If most results agree, use that value. If they differ significantly, use the most detailed/specific one.
2. For numeric fields: Use the median or most common value.
3. For boolean fields: Use majority voting.
4. For array fields: Combine unique elements from all results.
5. For null values: Only use null if most results are null.
6. Be conservative: prefer values that appear in multiple results over outliers.
7. Return ONLY a valid JSON object, no additional text.

Input results to average:
{json.dumps(results, indent=2)}

Produce the averaged/consensus JSON result:"""

    # Use a simple text input for the averaging LLM
    generator = LLMGenerator(
        model=averaging_model,
        text=averaging_prompt,
        temperature=averaging_temperature,
        logger=logger,
    )
    answer = generator.generate(prompt=averaging_prompt)

    # Extract JSON from the answer
    averaged_result = extract_json_from_text(answer)
    if averaged_result:
        return averaged_result
    else:
        logger.warning(
            "Failed to parse averaged result, returning first input result as fallback"
        )
        return results[0]


def run_ensemble_prompts(
    prompt: BasePrompt,
    text: str,
    n_runs: int = 5,
    models: list[str] | None = None,
    temperatures: list[float] | None = None,
    chunk_sizes: list[int] | None = None,
    chunks_list: list[list[Document]] | None = None,
    parallel: bool = True,
    averaging_model: str = "gpt-oss:20b",
    averaging_temperature: float = 0.2,
    **kwargs,
) -> tuple[str, dict[str, Any] | None]:
    """
    Run multiple LLM prompts with different configurations and average the results.

    This function is designed to minimize hallucinations by running the same prompt
    multiple times with variations and then averaging the results.

    Args:
        prompt (BasePrompt): The prompt to use (works best with StructuredPrompt)
        text (str): The text to process (or will use different chunks if chunks_list provided)
        n_runs (int): Number of times to run the prompt (default: 5)
        models (list[str] | None): List of models to cycle through (defaults to single model)
        temperatures (list[float] | None): List of temperatures to cycle through
        chunk_sizes (list[int] | None): List of chunk sizes (for information only, actual chunking done separately)
        chunks_list (list[list[Document]] | None): Pre-computed list of different chunk sets to use
        parallel (bool): Whether to run prompts in parallel (default: True)
        averaging_model (str): Model to use for averaging results
        averaging_temperature (float): Temperature for averaging model
        **kwargs: Additional arguments for LLMGenerator

    Returns:
        tuple[str, dict[str, Any] | None]: The raw combined answer text and averaged JSON (if StructuredPrompt)
    """
    # Set defaults
    # Extract model from kwargs if present, but don't pass it to run_single_llm_prompt
    default_model = kwargs.pop("model", "gpt-oss:20b")
    if models is None:
        models = [default_model]
    if temperatures is None:
        temperatures = [0.2, 0.5, 0.7]  # Variation in temperatures

    # Prepare configurations for each run
    configs = []
    for i in range(n_runs):
        model = models[i % len(models)]
        temperature = temperatures[i % len(temperatures)]

        # Determine which text/chunk to use
        run_text = text
        if chunks_list and len(chunks_list) > 0:
            chunk_set = chunks_list[i % len(chunks_list)]
            run_text = "\n\n".join([doc.page_content for doc in chunk_set])

        configs.append((i, run_text, model, temperature))

    logger.info(f"Running {n_runs} prompts with parallel={parallel}")

    # Run prompts (parallel or sequential)
    answers = []
    if parallel:
        with ThreadPoolExecutor(max_workers=min(n_runs, 10)) as executor:
            futures = [
                executor.submit(
                    run_single_llm_prompt,
                    prompt,
                    run_text,
                    model,
                    temperature,
                    run_id,
                    **kwargs,
                )
                for run_id, run_text, model, temperature in configs
            ]

            for future in as_completed(futures):
                run_id, answer = future.result()
                answers.append((run_id, answer))
    else:
        for run_id, run_text, model, temperature in configs:
            run_id, answer = run_single_llm_prompt(
                prompt, run_text, model, temperature, run_id, **kwargs
            )
            answers.append((run_id, answer))

    # Sort by run_id to maintain order
    answers.sort(key=lambda x: x[0])
    raw_answers = [answer for _, answer in answers]

    # Combine all answers
    combined_answer = "\n\n---\n\n".join(
        [f"Run {i + 1}:\n{ans}" for i, ans in enumerate(raw_answers)]
    )

    # If StructuredPrompt, extract JSON and average
    averaged_json = None
    if isinstance(prompt, StructuredPrompt):
        logger.info("Extracting and averaging JSON results from StructuredPrompt runs")
        json_results = []
        for i, answer in enumerate(raw_answers):
            parsed = extract_json_from_text(answer)
            if parsed:
                json_results.append(parsed)
            else:
                logger.warning(f"Failed to parse JSON from run {i + 1}")

        if json_results:
            averaged_json = average_json_results(
                json_results,
                output_schema=prompt.output_schema,
                averaging_model=averaging_model,
                averaging_temperature=averaging_temperature,
            )
        else:
            logger.error("No valid JSON results found to average")

    return combined_answer, averaged_json
