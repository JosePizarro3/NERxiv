from nerxiv.datamodel.model_method import DFT, DMFT, KPoints, Program
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
                    output="{'ChemicalFormulation': {'iupac': 'Si2'} }",
                ),
                Example(
                    input="We study the material with iupac formula Fe2O3, and its doped variant Fe2O3.25.",
                    output="{'ChemicalFormulation': [{'iupac': 'Fe2O3'}, {'iupac': 'Fe2O3.25'}] }",
                ),
                Example(
                    input="We study SrVO3, a system who is similar to SrTiO3 but with a different electronic structure.",
                    output="{'ChemicalFormulation': {'iupac': 'SrVO3'} }",
                ),
                Example(
                    input="The system is doped La1âxSrxNiO2, for x=0.2.",
                    output="{'ChemicalFormulation': {'iupac': 'La0.8Sr0.2NiO2'} }",
                ),
            ],
        ),
    ),
    "dft_structured": PromptRegistryEntry(
        retriever_query="""Identify all mentions of the DFT (Density Functional Theory) parameters and program being used.""",
        prompt=StructuredPrompt(
            expert="Condensed Matter Physics",
            sub_field_expertise="Density Functional Theory simulations",
            output_schema=DFT,
            target_fields=[
                "program_name",
                "program_version",
                "k_mesh",
                "exchange_correlation_functional",
                "basis_set",
                "planewave_cutoff_energy",
                "planewave_cutoff_energy_units",
                "rkmax",
            ],
            constraints=[
                "Only return the data asked for, without any additional text, explanation, or thinking block.",
                "If one of the target fields is not specified or you cannot extract it from the text, do not hallucinate and do not return it.",
                "If the exchange_correlation_functional or the basis_set does not match any of the known values, return 'unknown'.",
                "Some programs may not specify the basis set, but they are known to use a specific basis set. For example, if the program is 'VASP', the basis set is 'plane waves'.",
                "If basis_set is not 'plane waves', do not return planewave_cutoff_energy and planewave_cutoff_energy_units."
                "If basis_set is not '(L)APW' or '(L)APW+lo', do not return rkmax.",
                "Ignore mentions of parameters used in other simulations, only return the ones actually used in this text.",
                "If the method is completely unknown from your output schema and you cannot extract any of the target fields, return 'ERROR'.",
            ],
            examples=[
                Example(
                    input="We used the WIEN2K package for our DFT+DMFT simulations. We adopted the generalizeed gradient approximation (GGA) and the Perdew-Burke-Ernzerhof (PBE) functions. The maximum modulus for the reciprogral vectors Kmax was chosen such that RMT x Kmax = 7.0.",
                    output="{'DFT': {'program_name': 'WIEN2K', 'exchange_correlation_functional': 'GGA', 'basis_set': '(L)APW', 'rkmax': 7.0} }",
                ),
                Example(
                    input="We used the Quantum ESPRESSO package for our DFT simulations. We adopted the local density approximation (LDA) and the Perdew-Zunger (PZ) functions. The plane wave cutoff energy was set to 30 Ry.",
                    output="{'DFT': {'program_name': 'Quantum ESPRESSO', 'exchange_correlation_functional': 'LDA', 'basis_set': 'plane waves', 'planewave_cutoff_energy': 30, 'planewave_cutoff_energy_units': 'Ry'} }",
                ),
                Example(
                    input="We used the VASP package in version 5.5.5 for our DFT simulations. We use the generalized gradient approximation (GGA) with plane wave cutoff energy 500 eV and 7x7x1 k points.",
                    output="{'DFT': {'program_name': 'VASP', 'program_version': '5.5.5', 'exchange_correlation_functional': 'GGA', 'basis_set': 'plane waves', 'planewave_cutoff_energy': 500, 'planewave_cutoff_energy_units': 'eV', 'k_mesh': [7, 7, 1]} }",
                ),
                Example(
                    input="we study the hole distribution and the electronic structure of LaSrNiO2 by combing DFT with dynamical mean-field theory (DMFT). With the help of virtual crystal approximation (VCA) [59, 60], ",
                    output="ERROR",
                ),
            ],
        ),
    ),
}
