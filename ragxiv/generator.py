import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from structlog._config import BoundLoggerLazyProxy

from langchain_ollama.llms import OllamaLLM
from transformers import AutoTokenizer

from ragxiv.logger import logger


class LLMGenerator:
    """
    LLMGenerator class for generating answers with the `generate` method using a specified LLM model
    specified by the user. The LLM model is loaded using `OllamaLLM` implementation in LangChain.

    Read more in https://python.langchain.com/docs/integrations/llms/ollama/
    """

    def __init__(self, model: str = "deepseek-r1", text: str = "", **kwargs):
        if not text:
            raise ValueError("Text is required for LLM generation.")

        self.logger = kwargs.get("logger", logger)

        self._huggingface_model_map = {
            "deepseek-r1": ("deepseek-ai/DeepSeek-R1", 131072),
            "llama3": ("meta-llama/Llama-3.1-8B-Instruct", 8192),
        }

        self.llm = OllamaLLM(model=model)
        self.logger.info(f"LLM model: {model}")

    def _check_tokens_limit(self, prompt: str = "") -> bool:
        """
        Checks if the prompt length exceeds the token limit for the specified LLM model.

        Args:
            prompt (str, optional): The prompt to be checked if it exceeds the token limit. Defaults to "".

        Returns:
            bool: True if the prompt length is within the token limit, False otherwise.
        """
        huggingface_model, tokens_limit = self._huggingface_model_map.get(
            self.llm.model
        )
        tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
        num_tokens = len(tokenizer(prompt)["input_ids"])
        if num_tokens > tokens_limit:
            self.logger.critical(
                f"Prompt is too long ({num_tokens}) for the context window ({tokens_limit})."
            )
            return False
        return True

    def generate(self, prompt: str = "") -> str:
        """
        Generates an answer using the specified LLM model and the provided prompt provided that
        the token limit is not exceeded.

        Args:
            prompt (str, optional): The prompt to be used for generating the answer. Defaults to "".

        Returns:
            str: The generated answer from the LLM model.
        """
        if not self._check_tokens_limit(prompt=prompt) or not prompt:
            return ""

        return self.llm.invoke(prompt)


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
    match = re.search(r"\n\n\[\n *\{", answer, flags=re.IGNORECASE)
    if match:
        start = match.start()
        answer = answer[start:]
        answer = re.sub(r"\n\n", "", answer)

    # Return empty list if answer is empty or the loaded list of dictionaries
    dict_answer = []
    try:
        dict_answer = json.loads(answer)
    except json.JSONDecodeError:
        logger.critical(
            f"Answer is not a valid JSON: {answer}. Please check the answer format."
        )
    return dict_answer
