# How-to: Use Different LLM Models with Ollama

This guide shows you how to select and configure different Large Language Models (LLMs) using Ollama for the generation stage of the RAG pipeline.

In order to work in this guide, you need to install and set up Ollama locally:

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

### Recommended Models (November 2025)

| Model | Size | Best For | Speed | Quality |
|-------|------|----------|-------|---------|
| `gpt-oss:20b` | 20B | Quick extraction, good accuracy | Fast | Very good |
| `llama3.1:8b` | 8B | Quick extraction, general queries | Fast | Good |
| `llama3.1:70b` | 70B | Complex reasoning, accurate extraction | Slow | Excellent |
| `qwen2.5:32b` | 32B | Technical text, good reasoning | Medium | Very Good |
| `deepseek-r1:14b` | 14B | Scientific reasoning, formulas | Medium | Very Good |
| `mistral:7b` | 7B | Fast processing, simple queries | Fast | Good |

### Model Selection Guide

These are a few examples of how to select the LLM model. You can read mode about chosing model parameters in [How-to: Adjust LLM Parameters](../howtos/adjust_llm_parameters.md).

```bash
nerxiv prompt --file-path paper.hdf5 --model llama3.1:8b
```

```bash
nerxiv prompt --file-path paper.hdf5 --model llama3.1:70b
```

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

## Using Custom Ollama Endpoints

If running Ollama on a remote server or custom port:

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  -llmo base_url=http://192.168.1.100:11434
```

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

## Comparing Models

Test different models on the same paper by running in your terminal:

```bash
# Test with different models
for model in llama3.1:8b llama3.1:70b qwen2.5:32b; do
  echo "Testing $model..."
  nerxiv prompt --file-path paper.hdf5 --model $model --query material_formula
done
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
