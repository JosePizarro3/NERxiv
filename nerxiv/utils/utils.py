import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import h5py

if TYPE_CHECKING:
    from structlog._config import BoundLoggerLazyProxy

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


def clean_description(description: str) -> str:
    """
    Cleans the description by removing extra spaces and leading/trailing whitespace.

    Args:
        description (str): The description string to be cleaned.

    Returns:
        str: The cleaned description string with extra spaces removed.
    """
    return re.sub(r"\s+", " ", description).strip()


def filter_material_formula_predicate(answer: str) -> bool:
    """
    Predicate function to determine if the answer indicates the presence of a material formula.

    Args:
        answer (str): The answer string to be evaluated.

    Returns:
        bool: True if the answer is "model", indicating a material formula is present; False otherwise.
    """
    return answer == "model"


def filter_only_dmft_predicate(answer: str) -> bool:
    """
    Predicate function to determine if the answer indicates the absence of DMFT method.

    Args:
        answer (str): The answer string to be evaluated.

    Returns:
        bool: True if the answer is not "True", indicating DMFT is not used; False if DMFT is used.
    """
    return answer != "True"


def files_to_subfolder_answer(
    path: str = "./data",
    run: str = "run_0000",
    predicate: Callable[[str], bool] | None = None,
) -> None:
    files = list(Path(path).rglob("*.hdf5"))
    for file in files:
        with h5py.File(file, "a") as f:
            run_group = f["raw_llm_answers"][run]
            # run_group only has one key associated with what we are naming the subfolder
            subfolder_name = next(iter(run_group.keys()))

            # Check if the answer is going through a specific predicate (e.g., see `model_predicate()` utility function) or simply checking if the answer is True or False
            answer = run_group[subfolder_name]["answer"][()].decode("utf-8").strip()
            if (predicate or (lambda a: a == "True"))(answer):
                # Create subfolder and move file
                target_dir = file.parent / subfolder_name
                target_dir.mkdir(exist_ok=True)
                target_path = target_dir / file.name
                file.rename(target_path)
