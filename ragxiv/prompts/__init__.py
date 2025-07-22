from .templates import MATERIAL_TEMPLATE
from .top_chunks import CHUNKS_MATERIAL

QUERY_REGISTRY = {
    "material": (CHUNKS_MATERIAL, MATERIAL_TEMPLATE),
}

RETRIEVER_QUERY_REGISTRY = {
    "material": CHUNKS_MATERIAL,
}


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
