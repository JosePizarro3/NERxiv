from .ranking_prompts import MATERIAL_CATEGORIZATION_PROMPT
from .templates import MATERIAL_TEMPLATE


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
