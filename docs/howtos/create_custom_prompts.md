# How to Create Custom Prompts

This guide shows you how to create and register custom prompts for extracting specific information from scientific papers using the RAG pipeline.

## Understanding the Prompt Registry

NERxiv uses a `PROMPT_REGISTRY` to manage different extraction tasks. Each entry contains:

1. **Retriever query**: What to look for when retrieving chunks
2. **Prompt template**: How to instruct the LLM to extract information

## Anatomy of a Prompt

A prompt consists of several components:

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

## Creating a Simple Custom Prompt

Let's create a prompt to extract author affiliations.

### Step 1: Define the Prompt

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

### Step 2: Register the Prompt

Add your prompt to the registry:

```python
from nerxiv.prompts import PROMPT_REGISTRY

# Add to registry
PROMPT_REGISTRY["affiliations"] = AFFILIATION_ENTRY
```

### Step 3: Use the Custom Prompt

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --query affiliations \
  --model llama3.1:70b
```

## Creating a Structured Prompt

For structured output (JSON), use `StructuredPrompt`:

```python
from nerxiv.prompts.prompts import StructuredPrompt, PromptRegistryEntry, Example
from pydantic import BaseModel, Field

# Define output schema
class ComputationalDetails(BaseModel):
    software: list[str] = Field(description="Software packages used")
    parameters: dict[str, str] = Field(description="Key computational parameters")
    hardware: str = Field(description="Hardware description")

# Create structured prompt
computational_prompt = StructuredPrompt(
    expert="Computational Science",
    output_schema=ComputationalDetails,
    target_fields=["software", "parameters", "hardware"],
    constraints=[
        "Return valid JSON matching the schema",
        "Extract only explicitly mentioned information",
        "Use null for missing fields"
    ],
    examples=[
        Example(
            input="Calculations were performed using VASP 6.3 with ENCUT=520 eV on a GPU cluster.",
            output='```json\n{"software": ["VASP 6.3"], "parameters": {"ENCUT": "520 eV"}, "hardware": "GPU cluster"}\n```'
        )
    ]
)

# Register it
PROMPT_REGISTRY["computational_details"] = PromptRegistryEntry(
    retriever_query="Find information about software, computational methods, parameters, and hardware used",
    prompt=computational_prompt
)
```

## Best Practices

### 1. Write Clear Instructions

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

### 2. Provide Diverse Examples

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

### 3. Use Appropriate Constraints

Guide the output format:

```python
constraints=[
    "Return only the extracted information, no explanations",
    "Use pipe | to separate alternative names for the same entity",
    "Return 'None' if no relevant information is found",
    "Do not include thinking process or reasoning"
]
```

### 4. Tailor the Retriever Query

Make it specific:

```python
# Too broad
retriever_query="Find relevant information"

# Better
retriever_query="Identify paragraphs describing computational methods, software packages, and simulation parameters"
```

## Real-World Example: Extract Experimental Conditions

Let's create a comprehensive prompt for extracting experimental conditions:

```python
from nerxiv.prompts.prompts import Prompt, PromptRegistryEntry, Example

experimental_conditions_prompt = Prompt(
    expert="Experimental Physics",
    sub_field_expertise="materials characterization and synthesis",
    main_instruction="extract all experimental conditions including temperature, pressure, atmosphere, and duration",
    secondary_instructions=[
        "Look for synthesis or measurement conditions",
        "Include units with numerical values",
        "Note if conditions changed during the experiment",
        "Check methods section, results, and figure captions"
    ],
    constraints=[
        "Format as 'parameter: value unit'",
        "One condition per line",
        "Use standard SI units where possible",
        "Return 'Not specified' if no conditions are mentioned"
    ],
    examples=[
        Example(
            input="Samples were annealed at 800°C for 4 hours in nitrogen atmosphere.",
            output="Temperature: 800°C\nDuration: 4 hours\nAtmosphere: nitrogen"
        ),
        Example(
            input="Measurements were performed at room temperature under ambient pressure.",
            output="Temperature: room temperature\nPressure: ambient"
        ),
        Example(
            input="The reaction was conducted at 150°C and 5 bar for 2 hours.",
            output="Temperature: 150°C\nPressure: 5 bar\nDuration: 2 hours"
        )
    ]
)

PROMPT_REGISTRY["experimental_conditions"] = PromptRegistryEntry(
    retriever_query="Find descriptions of experimental conditions including temperature, pressure, atmosphere, time, and other synthesis or measurement parameters",
    prompt=experimental_conditions_prompt
)
```

Use it:

```bash
nerxiv prompt \
  --file-path paper.hdf5 \
  --query experimental_conditions \
  --model qwen2.5:32b \
  -llmo temperature=0.1
```

## Testing Custom Prompts

Test your prompt on sample text:

```python
from nerxiv.rag import LLMGenerator

# Sample text
text = """
The thin films were grown by pulsed laser deposition at a substrate 
temperature of 650°C in an oxygen partial pressure of 10⁻³ mbar. 
The growth rate was maintained at 0.1 nm/s for 30 minutes.
"""

# Generate answer
generator = LLMGenerator(model="llama3.1:8b", text=text, temperature=0.2)
prompt_text = experimental_conditions_prompt.build(text=text)
answer = generator.generate(prompt=prompt_text)

print("Extracted conditions:")
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

If the retrieved text doesn't contain what you need, adjust the retriever query.

### Test with Different Temperatures

```bash
# Very deterministic
nerxiv prompt --file-path paper.hdf5 --query your_query -llmo temperature=0.0

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

## Related Guides

- [How to adjust LLM parameters](adjust-llm-parameters.md)
- [Understanding prompt engineering](../explanations/prompt-engineering.md)
- [Using the RAG extractor agent](../tutorials/rag-extractor-tutorial.md)
