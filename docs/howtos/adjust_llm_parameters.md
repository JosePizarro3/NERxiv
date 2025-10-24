# How-to: Adjust LLM Parameters

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

### Example Comparison

With `temperature=0.0`:
```
Output: La0.8Sr0.2NiO2
(same every time)
```

With `temperature=0.7`:
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

### Interaction with Temperature

- Low temperature + low top_p = Very focused, deterministic
- Low temperature + high top_p = Consistent but considers more options
- High temperature + low top_p = Randomly picks from focused set (unstable)
- High temperature + high top_p = Maximum diversity

## num_ctx (Context Size)

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

<!-- Add other parameters -->

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

