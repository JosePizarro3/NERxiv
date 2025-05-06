EXP_OR_COMP_TEMPLATE = """You are a Condensed Matter Physics assistant.

Given the following scientific text, determine if it describes an **experimental** or **computational** method.
If it describes an experimental method, return "experimental". If it describes a computational method, return "computational".
If it describes both, return "both". If it describes neither, return "none".

Example 1:
    - Input text: We use Density Functional Theory (DFT) to calculate the electronic structure of the material.
    - Answer: computational

Example 2:
    - Input text: We conducted Angle Resolved Photoemission Spectroscopy (ARPES) with the following parameters.
    - Answer: experimental

Text:
{text}
"""

EXTRACT_METHODS_TEMPLATE = """You are a structured data extractor.

Extract a list of {exp_or_comp} methods (note that "both" means both experimental and computational) mentioned in the following scientific text.
Each method should include:
- A full name (e.g., "Density Functional Theory")
- An acronym if available (e.g., "DFT")

Example 1:
    - Input text: We use Density Functional Theory (DFT) to calculate the electronic structure of the material. We
    also performed Angle Resolved Photoemission Spectroscopy (ARPES) to study the surface states.
    - Answer: [
        {{ "name": "Density Functional Theory", "acronym": "DFT" }},
        {{ "name": "Angle Resolved Photoemission Spectroscopy", "acronym": "ARPES" }}
    ]

Example 2:
    - Input text: We performed ladder DΓA calculations on top of the DMFT solution. The DMFT self-energy was calculated using the CTQMC method.
    - Answer: [
        {{ "name": "Ladder DΓA", "acronym": "Ladder DΓA" }},
        {{ "name": "Dynamical Mean Field Theory", "acronym": "DMFT" }},
        {{ "name": "Continuous-Time Quantum Monte Carlo", "acronym": "CTQMC" }}
    ]

Text:
{text}
"""


FILTER_METHODS_TEMPLATE = """You are a Condensed Matter Physics assistant.

Given the following list of extracted candidates, filter out any that are **not actual methods**
but instead are **software packages, code implementations, or libraries**.

Return only the list of method dictionaries. Do not include any explanation or extra text.

Input:
{candidates}
"""


def prompt(template: str, **kwargs) -> str:
    """
    Builds a prompt using the provided template (see above for example templates) and the text
    passed to the template.

    Args:
        template (str): The template to use for the prompt.

    Returns:
        str: The formatted prompt.
    """
    return template.format(**kwargs)
