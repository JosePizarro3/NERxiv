# How to Use Different LLM Models with Ollama

This guide shows you how to select and configure different Large Language Models (LLMs) using Ollama for the generation stage of the RAG pipeline.

## Prerequisites

Install and set up Ollama:

1. Download Ollama from [ollama.com](https://ollama.com/download)
2. Start the Ollama server: `ollama serve`
3. Pull a model: `ollama pull llama3.1`

## Selecting a Model

Specify the model using the `--model` (or `-m`) flag:

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --model llama3.1:70b
```

## Popular Models for Scientific Text

### Recommended Models

| Model | Size | Best For | Speed | Quality |
|-------|------|----------|-------|---------|
| `llama3.1:8b` | 8B | Quick extraction, general queries | Fast | Good |
| `llama3.1:70b` | 70B | Complex reasoning, accurate extraction | Slow | Excellent |
| `qwen2.5:32b` | 32B | Technical text, good reasoning | Medium | Very Good |
| `deepseek-r1:14b` | 14B | Scientific reasoning, formulas | Medium | Very Good |
| `mistral:7b` | 7B | Fast processing, simple queries | Fast | Good |

### Model Selection Guide

**For rapid prototyping:**
```bash
nerxiv prompt --file-path paper.hdf5 --model llama3.1:8b
```

**For production/accuracy:**
```bash
nerxiv prompt --file-path paper.hdf5 --model llama3.1:70b
```

**For technical papers:**
```bash
nerxiv prompt --file-path paper.hdf5 --model deepseek-r1:14b
```

## Installing Models

Before using a model, pull it from Ollama:

```bash
# List available models
ollama list

# Pull a specific model
ollama pull llama3.1:8b

# Pull a larger model (may take time)
ollama pull llama3.1:70b
```

## Configuring LLM Parameters

Fine-tune model behavior using `--llm-option` (or `-llmo`) flags:

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --model llama3.1:70b \
  -llmo temperature=0.2 \
  -llmo top_p=0.9 \
  -llmo num_ctx=8192
```

### Key Parameters

#### temperature
Controls randomness in the output:

- `0.0`: Deterministic, consistent answers
- `0.1-0.3`: Low creativity, factual extraction (recommended)
- `0.5-0.7`: Balanced creativity and consistency
- `0.8-1.0`: High creativity, varied outputs

**Example:**
```bash
# For consistent formula extraction
-llmo temperature=0.1

# For exploratory analysis
-llmo temperature=0.7
```

#### top_p
Nucleus sampling - limits token selection to top probability mass:

- `0.9`: Standard setting, good balance
- `0.95`: More diverse outputs
- `0.8`: More focused outputs

**Example:**
```bash
-llmo top_p=0.9
```

#### num_ctx
Context window size (number of tokens):

- `2048`: Small context, faster
- `4096`: Standard context
- `8192`: Large context for long papers (recommended)
- `16384`: Very large context (requires more memory)

**Example:**
```bash
# For papers with long relevant sections
-llmo num_ctx=16384
```

#### format
Output format specification:

- `json`: Structured JSON output
- Empty string `''`: Free-form text

**Example:**
```bash
-llmo format=json
```

## Complete Examples

### Extract Material Formulas with High Accuracy

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --model llama3.1:70b \
  --query material_formula \
  -llmo temperature=0.1 \
  -llmo num_ctx=8192 \
  -llmo top_p=0.9
```

### Fast Batch Processing

```bash
nerxiv prompt-all \
  --data-path /papers/ \
  --model llama3.1:8b \
  --query only_dmft \
  -llmo temperature=0.2 \
  -llmo num_ctx=4096
```

### Structured Output for Parsing

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --model qwen2.5:32b \
  --query material_formula_structured \
  -llmo temperature=0.1 \
  -llmo format=json \
  -llmo num_ctx=8192
```

## Using Custom Ollama Endpoints

If running Ollama on a remote server or custom port:

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  -llmo base_url=http://192.168.1.100:11434
```

## Model-Specific Optimizations

### Llama 3.1

Excellent for scientific text with good instruction following:

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --model llama3.1:70b \
  -llmo temperature=0.2 \
  -llmo num_ctx=8192 \
  -llmo repeat_penalty=1.1
```

### Qwen 2.5

Strong technical understanding, good for complex formulas:

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --model qwen2.5:32b \
  -llmo temperature=0.15 \
  -llmo num_ctx=16384 \
  -llmo top_k=40
```

### DeepSeek-R1

Reasoning-focused model with thinking process:

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --model deepseek-r1:14b \
  -llmo temperature=0.3 \
  -llmo num_ctx=8192
```

Note: DeepSeek-R1 may include `<think>` tags in output, which NERxiv automatically removes.

## Python API

For programmatic control:

```python
from nerxiv.rag import LLMGenerator
from nerxiv.prompts import PROMPT_REGISTRY

# Get prompt template
query_entry = PROMPT_REGISTRY["material_formula"]
prompt_template = query_entry.prompt

# Initialize generator with custom settings
generator = LLMGenerator(
    model="llama3.1:70b",
    text=retrieved_chunks,
    temperature=0.2,
    num_ctx=8192,
    top_p=0.9,
    format="json"
)

# Generate answer
prompt = prompt_template.build(text=retrieved_chunks)
answer = generator.generate(prompt=prompt)
print(answer)
```

## Troubleshooting

### Model Not Found

```bash
Error: model 'llama3.1:70b' not found
```

**Solution:** Pull the model first:
```bash
ollama pull llama3.1:70b
```

### Out of Memory

```bash
Error: failed to allocate memory
```

**Solution:** Use a smaller model or reduce context:
```bash
nerxiv prompt --file-path paper.hdf5 --model llama3.1:8b -llmo num_ctx=4096
```

### Ollama Not Running

```bash
Error: connection refused
```

**Solution:** Start Ollama server:
```bash
ollama serve
```

### Slow Generation

If generation is too slow:

1. Use a smaller model: `llama3.1:8b` instead of `:70b`
2. Reduce context: `-llmo num_ctx=4096`
3. Use GPU if available
4. Reduce number of retrieved chunks: `--n-top-chunks 3`

## Comparing Models

Test different models on the same paper:

```bash
# Test with different models
for model in llama3.1:8b llama3.1:70b qwen2.5:32b; do
  echo "Testing $model..."
  nerxiv prompt --file-path paper.hdf5 --model $model --query material_formula
done
```

## Advanced: Custom Model Parameters

Ollama supports many additional parameters. Pass them via `-llmo`:

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --model llama3.1:70b \
  -llmo temperature=0.2 \
  -llmo top_k=40 \
  -llmo top_p=0.9 \
  -llmo repeat_penalty=1.1 \
  -llmo num_ctx=8192 \
  -llmo num_predict=512 \
  -llmo stop='Answer:'
```

For a complete list of parameters, see [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md).

## Related Guides

- [How to create custom prompts](create-custom-prompts.md)
- [How to adjust LLM parameters](adjust-llm-parameters.md)
- [Prompt engineering for metadata extraction](../explanations/prompt-engineering.md)
