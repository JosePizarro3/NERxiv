# How-to: Create Custom Prompts

This guide shows you how to create and register custom prompts for extracting specific information from scientific papers using the RAG pipeline.

## Understanding the Prompt Registry

NERxiv uses a `PROMPT_REGISTRY` to manage different extraction tasks. Each entry contains:

1. **Retriever query**: What to look for when retrieving chunks
2. **Prompt template**: How to instruct the LLM to extract information

## Anatomy of `Prompt`

`Prompt` consists of several components:

```python
from nerxiv.prompts.prompts import Prompt, Example

prompt = Prompt(
    expert="Condensed Matter Physics",
    main_instruction="identify all mentions of computational methods",
    secondary_instructions=[
        "Look for abbreviations like DFT, DMFT, QMC",
        "Include full names of methods mentioned",
        "Ignore methods used only as references"
    ],
    constraints=[
        "Return only method names, one per line",
        "No additional explanation or thinking block"
    ],
    examples=[
        Example(
            input="We use DFT+DMFT to calculate the electronic structure.",
            output="DFT+DMFT"
        ),
        Example(
            input="The results are compared with Quantum Monte Carlo simulations.",
            output="Quantum Monte Carlo"
        )
    ]
)
```

### Creating a Simple Custom Prompt

Let's create a prompt to extract author affiliations.

#### Step 1: Define the Prompt

Create a new file `my_prompts.py`:

```python
from nerxiv.prompts.prompts import Prompt, PromptRegistryEntry, Example

# Define the prompt
affiliation_prompt = Prompt(
    expert="Scientific Text Analysis",
    main_instruction="extract all institutional affiliations of the authors",
    secondary_instructions=[
        "Look for university names, research institutes, and companies",
        "Include department names if mentioned",
        "Look near author names or in footnotes"
    ],
    constraints=[
        "Return each affiliation on a separate line",
        "Use the full institution name",
        "Do not include author names"
    ],
    examples=[
        Example(
            input="John Doe¹ and Jane Smith² — ¹MIT, Cambridge, MA — ²Stanford University",
            output="MIT, Cambridge, MA\nStanford University"
        ),
        Example(
            input="Authors from the Department of Physics, University of Tokyo",
            output="Department of Physics, University of Tokyo"
        )
    ]
)

# Define the registry entry
AFFILIATION_ENTRY = PromptRegistryEntry(
    retriever_query="Find sections mentioning authors, affiliations, institutions, or university names",
    prompt=affiliation_prompt
)
```

#### Step 2: Register the Prompt

Add your prompt to the registry:

```python
from nerxiv.prompts import PROMPT_REGISTRY

# Add to registry
PROMPT_REGISTRY["affiliations"] = AFFILIATION_ENTRY
```

#### Step 3: Use the Custom Prompt

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --query affiliations \
  --model llama3.1:70b
```

## Creating a `StructuredPrompt`

For structured output (JSON), use `StructuredPrompt` instead:

```python
from nerxiv.prompts.prompts import StructuredPrompt, PromptRegistryEntry, Example
from pydantic import BaseModel, Field


class Affiliations(BaseModel):
    items: list[str] = Field([], description="A list of affiliations of the authors")


# Create structured prompt
affiliation_prompt = StructuredPrompt(
    expert="Scientific Text Analysis",
    output_schema=Affiliations,
    target_fields=["items"],
    constraints=[
        "Return each affiliation as an element of the list `items`",
        "Use the full institution name",
        "Do not include author names"
    ],
    examples=[
        Example(
            input="John Doe¹ and Jane Smith² — ¹MIT, Cambridge, MA — ²Stanford University",
            output='```json\n{\n\t"affiliations": {\n\t\t"items": ["MIT, Cambridge, MA", "Stanford University"]\n\t}\n}\n```'
        ),
        Example(
            input="Authors from the Department of Physics, University of Tokyo",
            output='```json\n{\n\t"affiliations": {\n\t\t"items": ["Department of Physics, University of Tokyo"]\n\t}\n}\n```'
        ),
    ]
)

# Register it
PROMPT_REGISTRY["affiliations"] = PromptRegistryEntry(
    retriever_query="Find sections mentioning authors, affiliations, institutions, or university names",
    prompt=affiliation_prompt
)
```

As you can see, the amount of information and free text needed to be passed is less than in the case of `Prompt`, see [Anatomy of `Prompt`](#anatomy-of-prompt).

## Best Practices

### Write Clear Instructions

**Bad:**
```python
main_instruction="get the methods"
```

**Good:**
```python
main_instruction="identify all computational and experimental methods used in the study"
secondary_instructions=[
    "Include both acronyms (e.g., DFT) and full names",
    "Distinguish between primary methods used and methods mentioned for comparison",
    "Look in the methods section, introduction, and results"
]
```

### Provide Diverse Examples

Include edge cases:

```python
examples=[
    # Simple case
    Example(
        input="We use DFT for electronic structure calculations.",
        output="DFT"
    ),
    # Multiple methods
    Example(
        input="The material is studied using DFT, DMFT, and Quantum Monte Carlo.",
        output="DFT\nDMFT\nQuantum Monte Carlo"
    ),
    # Method mentioned but not used
    Example(
        input="Our DFT results differ from previous DMFT studies on similar systems.",
        output="DFT"
    ),
    # Abbreviation and full name
    Example(
        input="We employ density functional theory (DFT) for the calculations.",
        output="DFT | density functional theory"
    )
]
```

### Use Appropriate Constraints

Guide the output format:

```python
constraints=[
    "Return only the extracted information, no explanations",
    "Use pipe | to separate alternative names for the same entity",
    "Return 'None' if no relevant information is found",
    "Do not include thinking process or reasoning"
]
```

### Tailor the Retriever Query

Make it specific:

```python
# Too broad
retriever_query="Find relevant information"

# Better
retriever_query="Identify paragraphs describing computational methods, software packages, and simulation parameters"
```

## Testing Custom Prompts

Test your prompt on sample text:

```python
from nerxiv.rag import LLMGenerator

# Sample text
text = """
The calculations were performed using VASP version 6.3.
The plane-wave cutoff energy was set to 520 eV, and the
Brillouin zone was sampled with a 6x6x6 Monkhorst-Pack k-point grid.
All calculations were run on a workstation with 2x Intel Xeon CPUs
and 128 GB RAM.
"""

# Generate answer
generator = LLMGenerator(model="llama3.1:8b", text=text, temperature=0.2)
prompt_text = computational_prompt.build(text=text)
answer = generator.generate(prompt=prompt_text)

print("Extracted computational details:")
print(answer)
```

## Debugging Prompts

If your prompt doesn't work well:

### Check the Retrieved Chunks

```python
from nerxiv.chunker import Chunker
from nerxiv.rag import CustomRetriever

chunker = Chunker(text=paper_text)
chunks = chunker.chunk_text()

retriever = CustomRetriever(
    query=PROMPT_REGISTRY["your_query"].retriever_query
)
top_text = retriever.get_relevant_chunks(chunks, n_top_chunks=5)

print("Retrieved text:")
print(top_text)
```

If the retrieved text doesn't contain what you need, adjust the chunking and retriever parameters.

### Test with Different Temperatures

```bash
# Very deterministic
nerxiv prompt --file-path paper.hdf5 --query your_query -llmo temperature=0.1

# More creative
nerxiv prompt --file-path paper.hdf5 --query your_query -llmo temperature=0.5
```

### Add More Examples

If the model output format is inconsistent, add more examples showing the exact format you want.

## Sharing Custom Prompts

To share prompts with others:

1. Create a Python file with your registry entries
2. Document the purpose and expected output
3. Include test cases

```python
# custom_prompts.py
"""
Custom prompts for NERxiv

Usage:
    from custom_prompts import register_custom_prompts
    register_custom_prompts()

    # Then use normally
    nerxiv prompt --file-path paper.hdf5 --query my_custom_query
"""

from nerxiv.prompts import PROMPT_REGISTRY
from nerxiv.prompts.prompts import Prompt, PromptRegistryEntry, Example

def register_custom_prompts():
    """Register all custom prompts to the global registry"""

    # Add your prompts here
    PROMPT_REGISTRY["custom_query"] = PromptRegistryEntry(
        retriever_query="...",
        prompt=Prompt(...)
    )

    print(f"Registered {len(PROMPT_REGISTRY)} prompts")

# Auto-register on import
register_custom_prompts()
```
