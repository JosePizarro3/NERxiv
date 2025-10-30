# Minimizing Hallucinations with Ensemble Prompting

## Overview

Ensemble prompting is a technique to minimize hallucinations in LLM outputs by running the same prompt multiple times with different configurations and then averaging the results. This approach is particularly effective for `StructuredPrompt` outputs where consistent, accurate structured data extraction is critical.

## How It Works

The ensemble approach consists of three main steps:

1. **Multiple Runs**: Run the same prompt multiple times (default: 5) with variations in:
   - **Models**: Use different LLM models to capture diverse perspectives
   - **Chunk Sizes**: Process different text chunks (minimum 2000 characters) to ensure complete context
   - **Temperatures**: Vary the temperature parameter to balance creativity and consistency

2. **Parallel Execution**: All runs are executed in parallel by default for efficiency, significantly reducing total processing time.

3. **LLM-Based Averaging**: An LLM with low temperature (default: 0.2) analyzes all results and produces a consensus output that best represents the information across all runs.

## Benefits

- **Reduced Hallucinations**: Multiple independent runs help identify and filter out inconsistent or incorrect information
- **Improved Accuracy**: Consensus-based results are more reliable than single-run outputs
- **Better Coverage**: Different chunk sizes ensure important information isn't missed due to chunking boundaries
- **Robustness**: Model variations provide diverse perspectives on the same information

## Using Ensemble Prompting

### Command Line Interface

Use the `prompt_ensemble` command to run ensemble prompting:

```bash
nerxiv prompt_ensemble \
  --file-path path/to/paper.hdf5 \
  --query dft \
  --n-ensemble-runs 5 \
  --ensemble-model "gpt-oss:20b" \
  --ensemble-model "llama3:8b" \
  --ensemble-temperature 0.2 \
  --ensemble-temperature 0.5 \
  --ensemble-temperature 0.7 \
  --ensemble-chunk-size 2000 \
  --ensemble-chunk-size 3000 \
  --ensemble-chunk-size 4000 \
  --averaging-model "gpt-oss:20b" \
  --averaging-temperature 0.2
```

### Key Parameters

- `--n-ensemble-runs`: Number of times to run the prompt (default: 5)
- `--ensemble-model`: Models to cycle through (can specify multiple)
- `--ensemble-temperature`: Temperature values to use (can specify multiple)
- `--ensemble-chunk-size`: Chunk sizes to use, minimum 2000 (can specify multiple)
- `--no-parallel`: Disable parallel execution (runs sequentially)
- `--averaging-model`: Model to use for result averaging
- `--averaging-temperature`: Temperature for the averaging LLM (default: 0.2)

### Python API

You can also use ensemble prompting programmatically:

```python
from nerxiv.ensemble import run_ensemble_prompts
from nerxiv.prompts.prompts import StructuredPrompt
from nerxiv.datamodel import DFT

# Define your prompt
prompt = StructuredPrompt(
    expert="Condensed Matter Physics",
    output_schema=DFT,
    target_fields=["all"],
)

# Run ensemble prompts
combined_answer, averaged_json = run_ensemble_prompts(
    prompt=prompt,
    text="Your scientific text here...",
    n_runs=5,
    models=["gpt-oss:20b", "llama3:8b"],
    temperatures=[0.2, 0.5, 0.7],
    parallel=True,
    averaging_model="gpt-oss:20b",
    averaging_temperature=0.2,
)

# Use the averaged JSON result
print(averaged_json)
```

## Best Practices

### Recommended Settings

1. **Number of Runs**: 5-7 runs provide a good balance between accuracy and computational cost
2. **Temperature Range**: Use 0.2, 0.5, and 0.7 for a good mix of consistency and diversity
3. **Chunk Sizes**: Start with [2000, 3000, 4000] and adjust based on paper length
4. **Averaging Temperature**: Keep it low (0.2) for conservative, consensus-based results

### When to Use Ensemble Prompting

Ensemble prompting is most beneficial for:

- **StructuredPrompts**: Extracting structured data where consistency is critical
- **High-Stakes Extraction**: When accuracy is more important than speed
- **Complex Scientific Papers**: When papers contain dense, technical information
- **Multi-Field Extraction**: When extracting multiple related fields simultaneously

### When NOT to Use Ensemble Prompting

Consider using standard prompting when:

- **Quick Exploration**: Initial exploration of papers where speed is prioritized
- **Simple Queries**: Basic information extraction that doesn't require high precision
- **Limited Resources**: When computational resources or API limits are constrained

## Output Format

### HDF5 Storage

Ensemble results are stored in the HDF5 file with additional metadata:

```
raw_llm_answers/
  {query}/
    run_XXXX/
      - answer: Combined text from all runs
      - averaged_json: Consensus JSON result (for StructuredPrompts)
      - model: List of models used
      - ensemble_mode: True
      - n_ensemble_runs: Number of runs executed
      - ensemble_temperatures: List of temperatures used
      - ensemble_chunk_sizes: List of chunk sizes used
      - averaging_model: Model used for averaging
      - averaging_temperature: Temperature used for averaging
```

### Averaged JSON

For `StructuredPrompt` queries, the `averaged_json` field contains the consensus result:

```json
{
  "DFT": {
    "xc_functional": "PBE",
    "basis_set": "plane-wave",
    "k_point_grid": [8, 8, 8],
    ...
  }
}
```

## Performance Considerations

### Parallel Execution

By default, ensemble runs execute in parallel using Python's `ThreadPoolExecutor`:

- **Default**: Up to 10 concurrent workers
- **Benefits**: 3-5x faster than sequential execution
- **Trade-offs**: Higher memory usage, requires stable LLM service

To disable parallel execution:

```bash
nerxiv prompt_ensemble --no-parallel ...
```

### Computational Cost

Ensemble prompting involves:

- **N runs** × LLM inference cost
- **1 averaging run** × LLM inference cost
- **Multiple chunking** and **retrieval operations** (cached for efficiency)

Total cost ≈ (N + 1) × single prompt cost

### Caching

The implementation includes efficient caching:

- **Chunk Cache**: Reuses chunks with the same parameters
- **Retrieval Cache**: Reuses retrieval results with the same configuration
- **Only LLM inference** is repeated across runs

## Examples

### Extract DFT Information

```bash
nerxiv prompt_ensemble \
  --file-path data/2401.12345.hdf5 \
  --query dft \
  --n-ensemble-runs 5 \
  --ensemble-chunk-size 2000 \
  --ensemble-chunk-size 3000
```

### Use Multiple Models

```bash
nerxiv prompt_ensemble \
  --file-path data/2401.12345.hdf5 \
  --query crystal_structure \
  --ensemble-model "gpt-oss:20b" \
  --ensemble-model "llama3:8b" \
  --ensemble-model "qwen3:30b"
```

### High-Precision Extraction

```bash
nerxiv prompt_ensemble \
  --file-path data/2401.12345.hdf5 \
  --query dmft \
  --n-ensemble-runs 7 \
  --ensemble-temperature 0.1 \
  --ensemble-temperature 0.3 \
  --ensemble-temperature 0.5 \
  --averaging-temperature 0.1
```

## Troubleshooting

### Slow Execution

- Enable parallel execution (default)
- Reduce `n-ensemble-runs`
- Use fewer model variations
- Check LLM service performance

### Inconsistent Results

- Increase `n-ensemble-runs` for more samples
- Use lower temperatures for more consistency
- Verify prompt constraints are clear
- Check that chunk sizes capture complete context

### High Memory Usage

- Disable parallel execution with `--no-parallel`
- Reduce the number of simultaneous models
- Process papers one at a time

### Averaging Failures

If the averaging LLM fails to parse results:

- Check that your `StructuredPrompt` constraints are clear
- Verify the output schema is well-defined
- Increase averaging temperature slightly (e.g., to 0.3)
- Check logs for JSON parsing errors

## Advanced Configuration

### Custom Ensemble Strategy

For programmatic use, you can customize the ensemble strategy:

```python
from nerxiv.ensemble import run_ensemble_prompts
from langchain_core.documents import Document

# Prepare different chunk sets manually
chunks_list = [
    [Document(page_content=text1)],
    [Document(page_content=text2)],
    [Document(page_content=text3)],
]

# Run with custom chunks
combined, averaged = run_ensemble_prompts(
    prompt=prompt,
    text=base_text,
    n_runs=3,
    chunks_list=chunks_list,
    parallel=True,
)
```

### Custom Averaging Function

For advanced use cases, you can implement custom averaging logic:

```python
from nerxiv.ensemble import extract_json_from_text, run_single_llm_prompt

# Run multiple prompts
results = []
for i in range(5):
    _, answer = run_single_llm_prompt(
        prompt=prompt,
        text=text,
        model=model,
        temperature=0.5,
        run_id=i,
    )
    json_result = extract_json_from_text(answer)
    if json_result:
        results.append(json_result)

# Implement custom averaging
def custom_average(results):
    # Your custom logic here
    return consensus_result

averaged = custom_average(results)
```

## See Also

- [Create Custom Prompts](create_custom_prompts.md)
- [Understanding Chunking](../explanations/understanding_chunking.md)
- [Configure Retrieval Models](configure_retrieval_models.md)
- [Use Different LLM Models](use_different_llm_models.md)
