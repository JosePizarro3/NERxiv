# EXP_OR_COMP_TEMPLATE = """You are a Condensed Matter Physics assistant.

# Given the following scientific text, determine if it describes an **experimental** or **computational** method.
# If it describes an experimental method, return "experimental". If it describes a computational method, return "computational".
# If it describes both, return "both". If it describes neither, return "none".

# Important instruction: only return the strings computational, experimental, both, or none, depending on the result.

# Example 1:
#     - Input text: We use Density Functional Theory (DFT) to calculate the electronic structure of the material.
#     - Answer: computational

# Example 2:
#     - Input text: We conducted Angle Resolved Photoemission Spectroscopy (ARPES) with the following parameters.
#     - Answer: experimental

# Text:
# {text}
# """

# EXTRACT_METHODS_TEMPLATE = """You are a structured data extractor.

# Extract a list of {exp_or_comp} methods (note that "both" means both experimental and computational) mentioned in the following scientific text.

# Important instruction: each method should include a full name (e.g., "Density Functional Theory"). Additionally, it can also contain an acronym if available (e.g., "DFT")

# Important instruction 2: only return the list of dictionaries, do not include any other explanation or extra text.

# Example 1:
#     - Input text: We use Density Functional Theory (DFT) to calculate the electronic structure of the material, and performed
#     a Wannier projection onto the dxy bands. We also performed Angle Resolved Photoemission Spectroscopy (ARPES) to study the surface states.
#     - Answer: [
#         {{ "name": "Density Functional Theory", "acronym": "DFT" }},
#         {{ "name": "Density Functional Theory", "acronym": "DFT" }},
#         {{ "name": "Angle Resolved Photoemission Spectroscopy", "acronym": "ARPES" }}
#     ]

# Text:
# {text}
# """


# FILTER_METHODS_TEMPLATE = """You are a Condensed Matter Physics assistant.

# Given the following list of extracted candidates, filter out any that are not actual methods but instead are software packages, code implementations, libraries, or instrument names.
# Reject also repetitions, unclear situations, or any other irrelevant information.

# Important instruction: return only the list of method dictionaries. Do not include any explanation or extra text.

# Example 1:
#     - Input: [\n    {{ "name": "Density Functional Theory", "acronym": "DFT" }},\n    {{ "name": "Wannier Hamiltonian" }},\n    {{ "name": "Vienna ab-initio simulation package" }},\n    {{ "name": "Perdew-Burke-Ernzerhof exchange correlation functional adapted for solids", "acronym": "PBESol" }},\n    {{ "name": "Wannierization" }}\n]
#     - Answer: [\n    {{ "name": "Density Functional Theory", "acronym": "DFT" }},\n    {{ "name": "Wannierization" }}\n]
#     - Reasoning: The "Vienna ab-initio simulation package" is a software package, and the "Perdew-Burke-Ernzerhof exchange correlation functional adapted for solids" is a specific functional, not a method. "Wannier Hamiltonian" and "Wannierization" are the same method.

# Input:
# {candidates}
# """
