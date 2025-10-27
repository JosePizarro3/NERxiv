from nerxiv.datamodel import (
    DFT,
    DMFT,
    AnalyticalContinuation,
    CrystalStructure,
    Interactions,
    Projection,
)
from nerxiv.datamodel.crystal_structure import ChemicalFormulation
from nerxiv.prompts.prompts import (
    Example,
    Prompt,
    PromptRegistryEntry,
    StructuredPrompt,
)

PROMPT_REGISTRY = {
    "material_formula": PromptRegistryEntry(
        retriever_query="""Identify all mentions of the system being simulated. The system can be a bulk crystal, a molecule,
        a nanostructure, and in general, any material. It can also be a toy model such as the square lattice,
        the triangular lattice, or the honeycomb lattice (to name a few).""",
        prompt=Prompt(
            expert="Condensed Matter Physics",
            main_instruction="identify all mentions of the system being simulated",
            secondary_instructions=[
                "Look for mentions of chemical formulas, specific names of models (like 'square lattice' or 'honeycomb lattice'), or any other indication that the system is a real material or a model.",
                "Only consider if the mention of a real material corresponds to an actual simulation of that material.",
                "Ignore mentions of similar materials, or whether the material is used as a reference or comparison.",
            ],
            constraints=[
                "Only return the strings asked for, without any additional text, explanation, or thinking block."
            ],
            examples=[
                Example(
                    input="The system is a bulk crystal of silicon, which has a diamond cubic structure.",
                    output="Si2",
                ),
                Example(
                    input="The square lattice model is used to simulate the behavior of electrons in a simplified system.",
                    output="model",
                ),
                Example(
                    input="We study the electronic properties of graphene, a two-dimensional material with a honeycomb lattice structure.",
                    output="graphene | C",
                ),
                Example(
                    input="We study the material Fe2O3 and its doped variant Fe2O3.25.",
                    output="Fe2O3, Fe2O3.25",
                ),
                Example(
                    input="We study SrVO3, a system who is similar to SrTiO3 but with a different electronic structure.",
                    output="SrVO3",
                ),
                Example(
                    input="The system is doped La1âxSrxNiO2, for x=0.2.",
                    output="La0.8Sr0.2NiO2",
                ),
            ],
        ),
    ),
    "only_dmft": PromptRegistryEntry(
        retriever_query="""Identify all mentions of the method used in the text. The method can be DMFT, DFT+U, DFT,
        Quantum Monte Carlo, Exact Diagonalization, etc.""",
        prompt=Prompt(
            expert="Condensed Matter Physics",
            main_instruction="identify all mentions of the method being used in the simulation",
            secondary_instructions=[
                "Look for mentions of the method applied over the material.",
                "If the method is DMFT (or any other variant of it like DFT+DMFT, EDMFT) return a boolean True. If not return False.",
                "Only consider if the mention of a method corresponds to an actual simulation of that material.",
                "Ignore mentions of similar methods, or whether the method is used as a reference or comparison.",
            ],
            constraints=[
                "Only return the expected answer asked for, without any additional text, explanation, or thinking block."
            ],
            examples=[
                Example(
                    input="We use DFT+DMFT to study the electronic properties of the LaCuO4.",
                    output="True",
                ),
                Example(
                    input="We use DFT to study the electronic properties of the LaCuO4.",
                    output="False",
                ),
                Example(
                    input="We use Quantum Monte Carlo to study the electronic properties of the LaCuO4.",
                    output="False",
                ),
                Example(
                    input="In another material, MnO, the DMFT results are in good agreement with our DFT+U results.",
                    output="False",
                ),
                Example(
                    input="We use DFT+U to study the electronic properties of the LaCuO4.",
                    output="False",
                ),
            ],
        ),
    ),
    "material_formula_structured": PromptRegistryEntry(
        retriever_query="""Identify all mentions of the system being simulated. The system can be a bulk crystal, a molecule,
        a nanostructure, and in general, any material. It can also be a toy model such as the square lattice,
        the triangular lattice, or the honeycomb lattice (to name a few).""",
        prompt=StructuredPrompt(
            expert="Condensed Matter Physics",
            output_schema=ChemicalFormulation,
            target_fields=["iupac"],
            constraints=[
                "Only return the data asked for, without any additional text, explanation, or thinking block.",
                "If the chemical formula format is not specified, check the context and store the most appropriate format.",
                "If multiple chemical formulations (representing different materials) are present, return them all as a list of dictionaries.",
                "Only consider if the mention of a real material corresponds to an actual simulation of that material.",
                "Ignore mentions of similar materials, or whether the material is used as a reference or comparison.",
            ],
            examples=[
                Example(
                    input="The system is a bulk crystal of silicon, which has a diamond cubic structure.",
                    output="```json\n'ChemicalFormulation': {'iupac': 'Si2'}\n```",
                ),
                Example(
                    input="The square lattice model is used to simulate the behavior of electrons in a simplified system.",
                    output="model",
                ),
                Example(
                    input="We study the material with iupac formula Fe2O3, and its doped variant Fe2O3.25.",
                    output="```json\n'ChemicalFormulation': [{'iupac': 'Fe2O3'}, {'iupac': 'Fe2O3.25'}]]\n```",
                ),
                Example(
                    input="We study SrVO3, a system who is similar to SrTiO3 but with a different electronic structure.",
                    output="```json\n'ChemicalFormulation': {'iupac': 'SrVO3'}\n```",
                ),
                Example(
                    input="The system is doped La1âxSrxNiO2, for x=0.2.",
                    output="```json\n'ChemicalFormulation': {'iupac': 'La0.8Sr0.2NiO2'}\n```",
                ),
            ],
        ),
    ),
    "crystal_structure": PromptRegistryEntry(
        retriever_query="""Identify all mentions of crystal structure, defined as any description
        of the material's atomic arrangement, lattice geometry, space group, symmetry, lattice
        parameters, atomic positions, or structural distortions. Include mentions of experimental
        or optimized structures, crystallographic references, or details about the unit cell or
        supercell setup.""",
        prompt=StructuredPrompt(
            expert="Condensed Matter Physics",
            output_schema=CrystalStructure,
            target_fields=[],
            constraints=[],
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
            target_fields=[],
            constraints=[],
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
            target_fields=[],
            constraints=[],
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
            target_fields=[],
            constraints=[],
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
            target_fields=[],
            constraints=[],
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
            target_fields=[],
            constraints=[],
            examples=[],
        ),
    ),
}
