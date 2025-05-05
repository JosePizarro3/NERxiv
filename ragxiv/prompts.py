EXTRACT_METHODS_TEMPLATE = """You are a structured data extractor.

Extract a list of experimental or computational methods mentioned in the following scientific text.
Each method should include:
- A full name (e.g., "Density Functional Theory")
- An acronym if available (e.g., "DFT")

Begin your answer directly with the list of dictionaries with keys `name` and `acronym`. Do not wrap it in <think>,
<explain>, or any markdown. Do not include any commentary, explanation, or reasoning.

Example format:
[
    {{ "name": "Density Functional Theory", "acronym": "DFT" }},
    {{ "name": "Angle Resolved Photoemission Spectroscopy", "acronym": "ARPES" }},
    {{ "name": "Wannierization", "acronym": "" }}
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
