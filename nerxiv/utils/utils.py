import json
import re
from typing import TYPE_CHECKING

from pymatgen.core import Composition

if TYPE_CHECKING:
    from structlog._config import BoundLoggerLazyProxy

from nerxiv.datamodel.model_system import ChemicalFormulation
from nerxiv.logger import logger


def answer_to_dict(
    answer: str = "", logger: "BoundLoggerLazyProxy" = logger
) -> list[dict]:
    """
    Converts the answer string to a list of dictionaries by removing unwanted characters. This is useful when
    prompting the LLM to return a list of objects containing metainformation in a structured way.

    Args:
        answer (str, optional): The answer string to be converted to a list of dictionaries. Defaults to "".
        logger (BoundLoggerLazyProxy, optional): The logger to log messages. Defaults to logger.

    Returns:
        list[dict]: The list of dictionaries extracted from the answer string.
    """
    # Return empty list if answer is empty or the loaded list of dictionaries
    dict_answer = []
    try:
        dict_answer = json.loads(answer)
    except json.JSONDecodeError:
        logger.critical(
            f"Answer is not a valid JSON: {answer}. Please check the answer format."
        )
    return dict_answer


def answer_to_formulas(answer: str) -> list[ChemicalFormulation]:
    formulas = answer.split(",")
    chemical_formulations = []
    for formula in formulas:
        try:
            composition = Composition(formula)
            chemical_formulations.append(
                ChemicalFormulation().set_formulas(composition)
            )
        except Exception:
            continue
    if len(chemical_formulations) != len(formulas):
        raise ValueError(
            "Some formulas could not be parsed. Please check the input format."
        )
    return chemical_formulations


def clean_description(description: str) -> str:
    """
    Cleans the description by removing extra spaces and leading/trailing whitespace.

    Args:
        description (str): The description string to be cleaned.

    Returns:
        str: The cleaned description string with extra spaces removed.
    """
    return re.sub(r"\s+", " ", description).strip()


# def material_pre_filtering(
#     chunks: list[Document] = [],
#     n_top_chunks: int = 5,
#     answer_model: str = "llama3.1:70b",
# ) -> list[ChemicalFormulation]:
#     rag = RAG(
#         retrieval_prompt=CHUNKS_MATERIAL,
#         chunks=chunks,
#         n_top_chunks=n_top_chunks,
#         answer_model=answer_model,
#     )
#     answer = rag.answer(template=MATERIAL_TEMPLATE)

#     if answer != "model":
#         return answer_to_formulas(answer)
#     return []
