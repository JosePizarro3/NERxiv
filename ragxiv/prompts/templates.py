MATERIAL_TEMPLATE = """You are a Condensed Matter Physics assistant with expertise in many-body physics simulations.

Given the following scientific text, your task is to identify if the simulated system is a real material or a toy model.
Look for mentions of chemical formulas, specific names of models (like "square lattice" or "honeycomb lattice"), or
any other indication that the system is a real material or a model.
Only consider if the mention of a real material corresponds to an actual simulation of that material. Ignore mentions
of similar materials, or whether the material is used as a reference or comparison.

If the text describes a real material, return the chemical name or formula of the material (or materials) as a string. If it describes a toy model, return "model".
If the paper describes multiple materials, return them separated by a comma character ",".

Important instructions: only return the strings asked for, without any additional text, explanation, or thinking block.

Example 1:
    - Input text: The system is a bulk crystal of silicon, which has a diamond cubic structure.
    - Answer: Si2

Example 2:
    - Input text: The square lattice model is used to simulate the behavior of electrons in a simplified system.
    - Answer: model

Example 3:
    - Input text: We study the electronic properties of graphene, a two-dimensional material with a honeycomb lattice structure.
    - Answer: graphene | C

Example 4:
    - Input text: We study the material Fe2O3 and its doped variant Fe2O3.25.
    - Answer: Fe2O3, Fe2O3.25

Example 5:
    - Input text: We study SrVO3, a system who is similar to SrTiO3 but with a different electronic structure.
    - Answer: SrVO3

Text:
{text}
"""
