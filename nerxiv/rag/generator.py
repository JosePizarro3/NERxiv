import inspect
import re

from langchain_ollama.llms import OllamaLLM

from nerxiv.logger import logger


class LLMGenerator:
    """
    LLMGenerator class for generating answers with the `generate` method using a specified LLM model
    specified by the user. The LLM model is loaded using `OllamaLLM` implementation in LangChain.

    Read more in https://python.langchain.com/docs/integrations/llms/ollama/
    """

    def __init__(self, text: str = "", **kwargs):
        if not text:
            raise ValueError("`text` is required for LLM generation.")
        self.text = text
        self.logger = kwargs.get("logger", logger)

        # Define default values for metadata extraction
        defaults = {
            "temperature": 0.2,
        }
        merged_args = {**defaults, **kwargs}

        # Dynamically detect valid OllamaLLM kwargs
        sig = inspect.signature(OllamaLLM)
        valid_params = set(sig.parameters.keys())
        # Filter kwargs
        ollama_kwargs = {k: v for k, v in merged_args.items() if k in valid_params}

        self.llm = OllamaLLM(**ollama_kwargs)
        self.logger.info(f"LLM model: {ollama_kwargs.get('model')}")

    def generate(
        self,
        prompt: str = "",
        regex: str = r"\n\nAnswer\: *",
        del_regex: str = r"\n\nAnswer\: *",
    ) -> str:
        """
        Generates an answer using the specified LLM model and the provided prompt provided that
        the token limit is not exceeded.

        Args:
            prompt (str, optional): The prompt to be used for generating the answer. Defaults to "".
            regex (str, optional): The regex pattern to search for in the answer. Defaults to r"\n\nAnswer\: *".
            del_regex (str, optional): The regex pattern to delete from the answer. Defaults to r"\n\nAnswer\: *".

        Returns:
            str: The generated and cleaned answer from the LLM model.
        """

        def _delete_thinking(answer: str = "") -> str:
            """
            Deletes the thinking process from the answer string by removing the <think> block.

            Args:
                answer (str, optional): The input text to delete the thinking block. Defaults to "".

            Returns:
                str: The answer string with the <think> block removed.
            """
            return re.sub(r"<think>.*?</think>\n*", "", answer, flags=re.DOTALL)

        def _clean_answer(regex: str, del_regex: str, answer: str = "") -> str:
            """
            Cleans the answer by removing unwanted characters and extracting the relevant part of the answer.

            Args:
                regex (str): The regex pattern to search for in the answer.
                del_regex (str): The regex pattern to delete from the answer.
                answer (str, optional): The answer input. Defaults to "".

            Returns:
                str: The cleaned answer.
            """
            match = re.search(regex, answer, flags=re.IGNORECASE)
            if match:
                start = match.start()
                answer = answer[start:]
                answer = re.sub(del_regex, "", answer)
            return answer

        llm_answer = self.llm.invoke(prompt)
        answer_withouth_think_block = _delete_thinking(answer=llm_answer)
        return _clean_answer(
            answer=answer_withouth_think_block, regex=regex, del_regex=del_regex
        )
