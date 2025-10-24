# Prompt Engineering for Metadata Extraction

Prompt engineering is the art and science of crafting instructions that guide LLMs to produce accurate, consistent, and useful outputs. This explanation covers principles, techniques, and best practices for extracting metadata from scientific papers.

A prompt is the instruction given to an LLM. In NERxiv, prompts guide the LLM to extract specific information from retrieved paper chunks.

**Basic prompt**:
```
Extract the chemical formula from this text.
```

**Engineered prompt**:
```
You are a Condensed Matter Physics assistant. Your task is to identify
all mentions of the system being simulated in the following text.

Look for:
- Chemical formulas (e.g., La₀.₈Sr₀.₂NiO₂)
- Specific model names (e.g., "square lattice")
- Material names (e.g., "graphene")

Only consider if the mention corresponds to an actual simulation of that material.
Ignore references to similar materials used only for comparison.

Important constraints:
- Return only the extracted formulas/names
- Do not include explanations or thinking blocks
- Use pipe | to separate alternative names for the same material

[Text to analyze]
```

The engineered prompt provides:

- **Role**: Expert identity
- **Task**: Clear objective
- **Instructions**: What to look for
- **Constraints**: Output format
- **Context**: Retrieved text

## Anatomy of a Good Prompt

NERxiv uses a structured approach with five components:

### 1. Expert Identity

Sets the context and expertise:

```python
expert="Condensed Matter Physics"
sub_field_expertise="many-body physics simulations"
```

Generated prompt section:
```
You are a Condensed Matter Physics assistant with expertise in
many-body physics simulations.
```

**Why it matters**: Primes the model to use domain-appropriate knowledge and terminology.

### 2. Main Instruction

The primary task:

```python
main_instruction="identify all computational methods used in the study"
```

Generated:
```
Your task is to identify all computational methods used in the study.
```

**Best practices**:
- Use action verbs: identify, extract, list, classify
- Be specific: "computational methods" not just "methods"
- Avoid ambiguity: "methods used in the study" not "methods mentioned"

### 3. Secondary Instructions

Detailed guidance:

```python
secondary_instructions=[
    "Look for abbreviations like DFT, DMFT, QMC",
    "Include full method names when mentioned",
    "Distinguish between methods used vs. methods mentioned for comparison",
    "Check the methods section, computational details, and results"
]
```

**Best practices**:
- Provide 3-5 specific sub-instructions
- Cover edge cases
- Clarify ambiguities
- Guide where to look

### 4. Constraints

Output formatting and restrictions:

```python
constraints=[
    "Return only method names, one per line",
    "Do not include explanations or reasoning",
    "Use pipe | to separate alternative names (e.g., DFT | density functional theory)",
    "Return 'None' if no methods are found"
]
```

**Why constraints matter**: LLMs tend to be verbose. Constraints ensure clean, parseable output.

### 5. Examples

Few-shot learning examples:

```python
examples=[
    Example(
        input="We use DFT+DMFT to study the electronic structure.",
        output="DFT+DMFT"
    ),
    Example(
        input="Our DFT results differ from previous DMFT studies.",
        output="DFT"
    )
]
```

**Power of examples**: Shows the model exactly what you want, especially for format and edge cases.

### 6. Schema and Target Fields to Extract Structured Outputs

Specify the pydantic `BaseModel` class and the target field defined within that class to extract metadata from:
```python
output_schema=ChemicalFormulation,
target_field=["iupac", "hill"],
```

NERxiv functionalities will read the description of both the schema class and targetted fields and generate a prompt from it:
```
Given the following scientific text, your task is: to identify all mentions of the ChemicalFormulation. This is defined as A ChemicalFormulation is a descriptive representation of the chemical composition of a material system, expressed in one or more standardized formula formats (e.g., IUPAC, anonymous, Hill, or reduced), each encoding the stoichiometry and elemental ordering according to specific conventions. For the compound H2O2 (hydrogen peroxide), the different formulations would be: iupac: H2O2 anonymous: AB hill: H2O2 reduced: H2O2. You must extract the values of the following fields:
- iupac defined as 'Chemical formula where the elements are ordered using a formal list based on electronegativity as defined in the IUPAC nomenclature of inorganic chemistry (2005): - https://en.wikipedia.org/wiki/List_of_inorganic_compounds Contains reduced integer chemical proportion numbers where the proportion number is omitted if it is 1.' and which is of type string
- hill defined as 'Chemical formula where Carbon is placed first, then Hydrogen, and then all the other elements in alphabetical order. If Carbon is not present, the order is alphabetical.' and which is of type string
You must return the extracted values in the following format:
    ```json
    'ChemicalFormulation': {
        'iupac': <parsed-value>,
        'hill': <parsed-value>,
    }
    ```
```

## Principles of Effective Prompts

### Principle 1: Clarity Over Brevity

❌ **Too brief**:
```
Get the formulas.
```

✅ **Clear**:
```
Extract all chemical formulas representing materials that were
actually simulated or synthesized in this study.
```

### Principle 2: Provide Context

❌ **No context**:
```
Is DMFT used?
```

✅ **With context**:
```
You are a Condensed Matter Physics expert. Determine whether
DMFT (Dynamical Mean Field Theory) or its variants (DFT+DMFT, EDMFT)
were used as a primary computational method in this study.
```

### Principle 3: Handle Edge Cases as Constraints

Common edge cases in scientific papers:

**Case 1: Mentioned but not used**
```
Input: "Our DFT results differ from previous DMFT studies."
Question: Was DMFT used?
Answer: No (only DFT was used, DMFT was referenced)
```

**Case 2: Multiple representations**
```
Input: "We study La₁₋ₓSrₓNiO₂ with x=0.2"
Expected output: "La₀.₈Sr₀.₂NiO₂"
```

**Case 3: Implicit information**
```
Input: "The nickelate was synthesized at 800°C"
Expected: Extract "nickelate" even without exact formula
```

Add instructions for these:
```python
constraints=[
    "Only consider methods actually used, not just mentioned for comparison",
    "Expand symbolic formulas (e.g., La₁₋ₓSrₓNiO₂ with x=0.2 → La₀.₈Sr₀.₂NiO₂)",
    "Include material class names if specific formulas aren't given"
]
```

### Principle 4: Use Examples Strategically

Include examples that cover:

1. **Simple case**: The most straightforward scenario
2. **Edge case**: Something tricky or ambiguous
3. **Negative case**: When nothing should be extracted
4. **Complex case**: Multiple entities or formats

```python
examples=[
    # Simple
    Example(
        input="The material is silicon (Si).",
        output="Si"
    ),
    # Multiple
    Example(
        input="We study Fe₂O₃ and its doped variant Fe₂O₃.₂₅.",
        output="Fe₂O₃, Fe₂O₃.₂₅"
    ),
    # Edge case - mentioned but not studied
    Example(
        input="SrVO₃ is similar to SrTiO₃ but has different properties.",
        output="SrVO₃"
    ),
    # Complex - symbolic to explicit
    Example(
        input="The system is doped La₁₋ₓSrₓNiO₂, for x=0.2.",
        output="La₀.₈Sr₀.₂NiO₂"
    )
]
```

## Common Prompting Patterns

### Pattern 1: Classification (Yes/No)

```python
prompt = Prompt(
    expert="Physics",
    main_instruction="determine if DMFT methodology is used",
    secondary_instructions=[
        "DMFT includes DFT+DMFT, EDMFT, and other variants",
        "Return True only if DMFT is a primary method used",
        "Return False if DMFT is only mentioned as reference"
    ],
    constraints=[
        "Return only 'True' or 'False'",
        "No explanations"
    ],
    examples=[
        Example(input="We use DFT+DMFT.", output="True"),
        Example(input="We use DFT.", output="False"),
        Example(input="Our DFT results differ from DMFT studies.", output="False")
    ]
)
```

### Pattern 2: Extraction (List of Entities)

```python
prompt = Prompt(
    expert="Chemistry",
    main_instruction="extract all chemical formulas mentioned",
    secondary_instructions=[
        "Include systematic names and common names",
        "Expand symbolic notation if values are given",
        "Include both reactants and products"
    ],
    constraints=[
        "One formula per line",
        "Use standard chemical notation",
        "Return 'None found' if no formulas present"
    ],
    examples=[
        Example(input="NaCl dissolved in water.", output="NaCl"),
        Example(input="Synthesis of TiO₂ from Ti and O₂.", output="TiO₂\nTi\nO₂")
    ]
)
```

### Pattern 3: Structured Extraction (JSON)

```python
from pydantic import BaseModel, Field

class MaterialInfo(BaseModel):
    formula: str = Field(description="Chemical formula")
    temperature: str | None = Field(description="Synthesis temperature")
    method: str | None = Field(description="Synthesis method")

prompt = StructuredPrompt(
    expert="Materials Science",
    output_schema=MaterialInfo,
    target_fields=["formula", "temperature", "method"],
    constraints=[
        "Return valid JSON matching the schema",
        "Use null for missing information",
        "Include units with numerical values"
    ],
    examples=[
        Example(
            input="Fe₂O₃ was synthesized at 800°C using sol-gel method.",
            output='```json\n{"formula": "Fe₂O₃", "temperature": "800°C", "method": "sol-gel"}\n```'
        )
    ]
)
```

## Advanced Techniques

### Chain of Thought (CoT)

For complex reasoning, encourage step-by-step thinking:

```python
secondary_instructions=[
    "First, identify all material mentions",
    "Then, determine which were actually studied (not just referenced)",
    "Finally, extract their chemical formulas"
]
```

The LLM naturally reasons through steps before answering.

### Self-Consistency

For critical extractions, you can run the same prompt multiple times with `temperature > 0` and take the most common answer:

```bash
# Run 3 times
for i in 1 2 3; do
  nerxiv prompt --file-path paper.hdf5 --query material_formula -llmo temperature=0.3
done

# Compare outputs, use consensus
```

### Negative Instructions

Sometimes telling the model what NOT to do helps:

```python
constraints=[
    "Do NOT include author names",
    "Do NOT return materials mentioned only in references",
    "Do NOT include explanation or thinking process"
]
```

### Format Examples in Output

Show exact format in examples:

```python
Example(
    input="Temperature was 300K, pressure 1 bar, duration 2 hours.",
    output="Temperature: 300K\nPressure: 1 bar\nDuration: 2 hours"
)
```

The model learns the exact format you want.

## Best Practices Summary

1. **Be specific**: Clear tasks, detailed instructions
2. **Provide context**: Set expert role and domain
3. **Use examples**: Show exactly what you want (3-5 examples)
4. **Control format**: Explicit output constraints
5. **Handle edge cases**: Cover tricky scenarios in instructions and examples
6. **Test iteratively**: Try on diverse inputs, refine based on failures
7. **Use low temperature**: 0.0-0.2 for factual extraction
8. **Keep it focused**: One clear task per prompt
