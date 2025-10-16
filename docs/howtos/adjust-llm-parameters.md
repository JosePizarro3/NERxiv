# How to Adjust LLM Parameters

This guide provides detailed information on tuning LLM parameters to optimize extraction quality, consistency, and performance.

## Parameter Overview

LLM parameters control how the model generates text. The most important ones for metadata extraction are:

| Parameter | Range | Purpose | Default |
|-----------|-------|---------|---------|
| `temperature` | 0.0-2.0 | Controls randomness | 0.2 |
| `top_p` | 0.0-1.0 | Nucleus sampling | 0.9 |
| `top_k` | 1-100 | Limits token choices | 40 |
| `num_ctx` | 128-32768 | Context window size | 2048 |
| `num_predict` | -1 or 1-2048 | Max tokens to generate | -1 (unlimited) |
| `repeat_penalty` | 0.0-2.0 | Penalizes repetition | 1.1 |

## Temperature

Controls output randomness by scaling the probability distribution over tokens.

### How It Works

- **0.0**: Deterministic - always picks the most likely token
- **0.1-0.3**: Low randomness - good for factual extraction
- **0.5-0.7**: Balanced - some creativity
- **0.8-1.0**: High randomness - diverse outputs
- **>1.0**: Very random - experimental

### When to Adjust

**Use low temperature (0.0-0.2)** for:
- Extracting facts, formulas, or numbers
- Consistent, reproducible outputs
- Structured data extraction

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --query material_formula \
  -llmo temperature=0.1
```

**Use medium temperature (0.3-0.5)** for:
- Summarization tasks
- Extracting interpretations
- Balancing accuracy and completeness

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --query experimental_conditions \
  -llmo temperature=0.4
```

**Use high temperature (0.6-1.0)** for:
- Exploratory analysis
- Generating alternative interpretations
- Creative tasks (not typical for NERxiv)

### Example Comparison

With `temperature=0.0`:
```
Output: La0.8Sr0.2NiO2
(same every time)
```

With `temperature=0.5`:
```
Run 1: La0.8Sr0.2NiO2
Run 2: La₀.₈Sr₀.₂NiO₂
Run 3: La0.8Sr0.2NiO2, lanthanum strontium nickelate
(variations in format)
```

## top_p (Nucleus Sampling)

Limits token selection to the smallest set whose cumulative probability exceeds `top_p`.

### How It Works

- **0.5**: Only consider top 50% probability mass
- **0.9**: Consider tokens making up 90% probability (recommended)
- **0.95**: More diverse outputs
- **1.0**: Consider all tokens (disabled)

### When to Adjust

**Use lower top_p (0.7-0.8)** for:
- More focused, deterministic outputs
- Extracting specific information
- When combined with low temperature

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  -llmo temperature=0.1 \
  -llmo top_p=0.8
```

**Use higher top_p (0.9-0.95)** for:
- More diverse outputs
- Capturing edge cases
- When model tends to repeat

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  -llmo temperature=0.3 \
  -llmo top_p=0.95
```

### Interaction with Temperature

- Low temp + low top_p = Very focused, deterministic
- Low temp + high top_p = Consistent but considers more options
- High temp + low top_p = Randomly picks from focused set (unstable)
- High temp + high top_p = Maximum diversity

## top_k

Limits token selection to the top K most likely tokens at each step.

### When to Adjust

**Use lower top_k (10-30)** for:
- Very focused outputs
- Technical extraction with limited vocabulary
- Preventing model from using unlikely tokens

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  -llmo top_k=20 \
  -llmo temperature=0.2
```

**Use higher top_k (40-100)** for:
- More flexibility in word choice
- Complex descriptions
- Default setting is usually fine

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  -llmo top_k=60
```

## num_ctx (Context Window)

Maximum number of tokens the model can process (input + output).

### Choosing the Right Size

**2048 tokens (~1500 words)**: 
- Fast processing
- Sufficient for 3-5 small chunks
- Use for simple extraction

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --n-top-chunks 3 \
  -llmo num_ctx=2048
```

**4096 tokens (~3000 words)**:
- Standard setting
- Good for 5-7 medium chunks
- Balance of speed and context

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --n-top-chunks 5 \
  -llmo num_ctx=4096
```

**8192 tokens (~6000 words)**:
- Large context (recommended for papers)
- 8-12 chunks
- Better understanding of context

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --n-top-chunks 10 \
  -llmo num_ctx=8192
```

**16384+ tokens**:
- Very large context
- May be slower
- Check model support

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --n-top-chunks 15 \
  -llmo num_ctx=16384
```

### Estimating Token Count

Rough estimates:
- 1 token ≈ 0.75 words (English)
- 1 token ≈ 4 characters
- Your prompt template ≈ 200-500 tokens
- Each chunk ≈ 250-750 tokens (depending on chunker settings)

Example calculation:
```
Prompt: 300 tokens
5 chunks × 500 tokens = 2500 tokens
Output: 200 tokens
Total: ~3000 tokens → use num_ctx=4096
```

## num_predict

Maximum tokens to generate in the response.

### When to Adjust

**Limit output length:**
```bash
# Short answers only
nerxiv prompt \
  --file-path paper.hdf5 \
  -llmo num_predict=100
```

**Allow longer outputs:**
```bash
# For detailed extraction
nerxiv prompt \
  --file-path paper.hdf5 \
  -llmo num_predict=512
```

**Unlimited (default -1):**
```bash
# Let model decide
nerxiv prompt --file-path paper.hdf5
```

## repeat_penalty

Penalizes tokens that have already been generated, reducing repetition.

### When to Adjust

**Increase for repetitive models:**
```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  -llmo repeat_penalty=1.2
```

**Decrease if penalizing valid repetition:**
```bash
# For chemical formulas that naturally repeat
nerxiv prompt \
  --file-path paper.hdf5 \
  --query material_formula \
  -llmo repeat_penalty=1.0
```

## Recommended Parameter Combinations

### For Chemical Formula Extraction

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --query material_formula \
  --model llama3.1:70b \
  -llmo temperature=0.1 \
  -llmo top_p=0.9 \
  -llmo num_ctx=8192 \
  -llmo repeat_penalty=1.0
```

### For Method Classification

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --query only_dmft \
  --model llama3.1:70b \
  -llmo temperature=0.0 \
  -llmo top_p=0.8 \
  -llmo top_k=30 \
  -llmo num_ctx=4096
```

### For Structured Data Extraction

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --query material_formula_structured \
  --model qwen2.5:32b \
  -llmo temperature=0.15 \
  -llmo format=json \
  -llmo num_ctx=8192 \
  -llmo repeat_penalty=1.1
```

### For Batch Processing (Speed Priority)

```bash
nerxiv prompt-all \
  --data-path /papers/ \
  --model llama3.1:8b \
  -llmo temperature=0.2 \
  -llmo num_ctx=2048 \
  -llmo num_predict=256
```

### For Maximum Accuracy (Quality Priority)

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --model llama3.1:70b \
  --n-top-chunks 12 \
  -llmo temperature=0.05 \
  -llmo top_p=0.85 \
  -llmo num_ctx=16384 \
  -llmo repeat_penalty=1.1
```

## Advanced Parameters

### format

Specify output format:

```bash
# JSON output
-llmo format=json

# Free-form text
-llmo format=''
```

### stop

Define stop sequences:

```bash
# Stop at specific token
-llmo stop='###'

# Multiple stop sequences
-llmo stop='Answer:' -llmo stop='###'
```

### mirostat

Alternative sampling method (0=disabled, 1 or 2=enabled):

```bash
# Enable Mirostat mode 2
-llmo mirostat=2 \
-llmo mirostat_tau=5.0 \
-llmo mirostat_eta=0.1
```

## Debugging with Parameters

### Output Too Random
```bash
# Decrease temperature and top_p
-llmo temperature=0.1 -llmo top_p=0.8
```

### Output Too Repetitive
```bash
# Increase temperature and repeat_penalty
-llmo temperature=0.3 -llmo repeat_penalty=1.3
```

### Output Gets Cut Off
```bash
# Increase num_predict
-llmo num_predict=1024
```

### Model Runs Out of Context
```bash
# Increase num_ctx and reduce chunks
-llmo num_ctx=16384
--n-top-chunks 8
```

### Slow Processing
```bash
# Reduce context and use smaller model
-llmo num_ctx=2048
--model llama3.1:8b
```

## Validating Parameter Effects

Test parameter changes systematically:

```bash
# Baseline
nerxiv prompt --file-path paper.hdf5 --query material_formula

# Test temperature
nerxiv prompt --file-path paper.hdf5 --query material_formula -llmo temperature=0.0
nerxiv prompt --file-path paper.hdf5 --query material_formula -llmo temperature=0.3

# Test context size
nerxiv prompt --file-path paper.hdf5 --n-top-chunks 5 -llmo num_ctx=4096
nerxiv prompt --file-path paper.hdf5 --n-top-chunks 10 -llmo num_ctx=8192
```

## Related Guides

- [How to use different LLM models](use-different-llm-models.md)
- [How to create custom prompts](create-custom-prompts.md)
- [Tutorial: Using the RAG extractor agent](../tutorials/rag-extractor-tutorial.md)
