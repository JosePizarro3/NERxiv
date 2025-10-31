from nerxiv.datamodel import (
    DFT,
    DMFT,
    AnalyticalContinuation,
    CrystalStructure,
    Interactions,
    Projection,
)
from nerxiv.prompts.prompts import (
    Example,
    Prompt,
    PromptRegistryEntry,
    StructuredPrompt,
)

PROMPT_REGISTRY = {
    "crystal_structure": PromptRegistryEntry(
        retriever_query="""Identify all mentions of crystal structure, defined as any description
        of the material's atomic arrangement, lattice geometry, space group, symmetry, lattice
        parameters, atomic positions, or structural distortions. Include mentions of experimental
        or optimized structures, crystallographic references, or details about the unit cell or
        supercell setup.""",
        prompt=StructuredPrompt(
            expert="Condensed Matter Physics",
            output_schema=CrystalStructure,
            target_fields=["all"],
            constraints=[
                "Return ONLY the requested JSON object without any additional text or explanation.",
                "If you do NOT find the value of a field in the text, do NOT make up a value. Leave it as null in the JSON output.",
                "Do NOT infer values of fields that are not explicitly mentioned in the text.",
                "Return the JSON as specified in the prompt. Do NOT make up a new JSON with different field names or structure.",
                "Ensure that all parsed values are of the correct data type as defined in the CrystalStructure schema.",
            ],
            examples=[],
        ),
    ),
    "dft": PromptRegistryEntry(
        retriever_query="""Identify all mentions of Density Functional Theory (DFT) calculations,
        defined as any description of electronic-structure computations within the Kohn-Sham
        formalism, including the chosen exchange-correlation functional, computational code, basis
        set, pseudopotential, convergence parameters, or spin treatment. Include any statements
        about how the DFT calculation was performed, validated, or referenced from prior work.""",
        prompt=StructuredPrompt(
            expert="Condensed Matter Physics",
            output_schema=DFT,
            target_fields=["all"],
            constraints=[
                "Return ONLY the requested JSON object without any additional text or explanation.",
                "If you do NOT find the value of a field in the text, do NOT make up a value. Leave it as null in the JSON output.",
                "Do NOT infer values of fields that are not explicitly mentioned in the text.",
                "Return the JSON as specified in the prompt. Do NOT make up a new JSON with different field names or structure.",
                "Ensure that all parsed values are of the correct data type as defined in the DFT schema.",
            ],
            examples=[],
        ),
    ),
    "projection": PromptRegistryEntry(
        retriever_query="""Identify all mentions of projection procedures or downfolding methods,
        defined as any description of how Bloch or band states are mapped to a localized orbital
        basis, such as via Wannier functions, LMTO, or projector methods. Include mentions of
        the selection of correlated orbitals, energy windows, orthogonalization, or the construction
        of low-energy models from DFT bands.""",
        prompt=StructuredPrompt(
            expert="Condensed Matter Physics",
            output_schema=Projection,
            target_fields=["all"],
            constraints=[
                "Return ONLY the requested JSON object without any additional text or explanation.",
                "If you do NOT find the value of a field in the text, do NOT make up a value. Leave it as null in the JSON output.",
                "Do NOT infer values of fields that are not explicitly mentioned in the text.",
                "Return the JSON as specified in the prompt. Do NOT make up a new JSON with different field names or structure.",
                "Ensure that all parsed values are of the correct data type as defined in the Projection schema.",
            ],
            examples=[],
        ),
    ),
    "interactions": PromptRegistryEntry(
        retriever_query="""Identify all mentions of local Coulomb interaction parameters, defined
        as any discussion of Hubbard U, Hund's coupling J, Slater integrals, or the general form
        of the interaction Hamiltonian (density-density, rotationally invariant, etc.). Include
        mentions of how these parameters were estimated, their numerical values, or how double-counting
        corrections were handled.""",
        prompt=StructuredPrompt(
            expert="Condensed Matter Physics",
            output_schema=Interactions,
            target_fields=["all"],
            constraints=[
                "Return ONLY the requested JSON object without any additional text or explanation.",
                "If you do NOT find the value of a field in the text, do NOT make up a value. Leave it as null in the JSON output.",
                "Do NOT infer values of fields that are not explicitly mentioned in the text.",
                "Return the JSON as specified in the prompt. Do NOT make up a new JSON with different field names or structure.",
                "Ensure that all parsed values are of the correct data type as defined in the Interactions schema.",
            ],
            examples=[],
        ),
    ),
    "dmft": PromptRegistryEntry(
        retriever_query="""Identify all mentions of Dynamical Mean-Field Theory (DMFT) simulations,
        defined as any description of the impurity problem setup, solver type, temperature or
        Matsubara grid, self-consistency loop, convergence criteria, and computed quantities such
        as Green's functions, self-energies, or occupancies. Include both methodological and
        computational details.""",
        prompt=StructuredPrompt(
            expert="Condensed Matter Physics",
            output_schema=DMFT,
            target_fields=["all"],
            constraints=[
                "Return ONLY the requested JSON object without any additional text or explanation.",
                "If you do NOT find the value of a field in the text, do NOT make up a value. Leave it as null in the JSON output.",
                "Do NOT infer values of fields that are not explicitly mentioned in the text.",
                "Return the JSON as specified in the prompt. Do NOT make up a new JSON with different field names or structure.",
                "Ensure that all parsed values are of the correct data type as defined in the DMFT schema.",
            ],
            examples=[],
        ),
    ),
    "analytical_continuation": PromptRegistryEntry(
        retriever_query="""Identify all mentions of analytical continuation, defined as any
        procedure used to obtain real-frequency spectra from imaginary-time or Matsubara data,
        including methods like Maximum Entropy, Pade approximants, or stochastic continuation.
        Include discussion of regularization, default models, priors, numerical broadening,
        or validation checks of the spectral results.""",
        prompt=StructuredPrompt(
            expert="Condensed Matter Physics",
            output_schema=AnalyticalContinuation,
            target_fields=["all"],
            constraints=[
                "Return ONLY the requested JSON object without any additional text or explanation.",
                "If you do NOT find the value of a field in the text, do NOT make up a value. Leave it as null in the JSON output.",
                "Do NOT infer values of fields that are not explicitly mentioned in the text.",
                "Return the JSON as specified in the prompt. Do NOT make up a new JSON with different field names or structure.",
                "Ensure that all parsed values are of the correct data type as defined in the AnalyticalContinuation schema.",
            ],
            examples=[],
        ),
    ),
}

# Integrate filters registry into the main prompt registry for usage in the CLI
from nerxiv.prompts.filters_registry import FILTERS_REGISTRY

PROMPT_REGISTRY.update(FILTERS_REGISTRY)
