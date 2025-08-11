from nerxiv.datamodel.model_system import ChemicalFormulation
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
                    output="```json\n'ChemicalFormulation': {'iupac': 'Si2'}```",
                ),
                Example(
                    input="We study the material with iupac formula Fe2O3, and its doped variant Fe2O3.25.",
                    output="```json'ChemicalFormulation': [{'iupac': 'Fe2O3'}, {'iupac': 'Fe2O3.25'}]]```",
                ),
                Example(
                    input="We study SrVO3, a system who is similar to SrTiO3 but with a different electronic structure.",
                    output="```json\n'ChemicalFormulation': {'iupac': 'SrVO3'}```",
                ),
                Example(
                    input="The system is doped La1âxSrxNiO2, for x=0.2.",
                    output="```json\n'ChemicalFormulation': {'iupac': 'La0.8Sr0.2NiO2'}```",
                ),
            ],
        ),
    ),
}
