import re

from langchain_ollama.llms import OllamaLLM
from transformers import AutoTokenizer

from nerxiv.logger import logger


class LLMGenerator:
    """
    LLMGenerator class for generating answers with the `generate` method using a specified LLM model
    specified by the user. The LLM model is loaded using `OllamaLLM` implementation in LangChain.

    Read more in https://python.langchain.com/docs/integrations/llms/ollama/
    """

    def __init__(self, model: str = "deepseek-r1", text: str = "", **kwargs):
        if not text:
            raise ValueError("`text` is required for LLM generation.")
        self.text = text

        self.logger = kwargs.get("logger", logger)

        # ! note this is a list to keep track of the models tested, rather than a complete list of all the LLM models available
        self._huggingface_model_map = {
            "deepseek-r1": ("deepseek-ai/DeepSeek-R1", 131072),
            "llama3.1": ("meta-llama/Llama-3.1-8B-Instruct", 131072),
            "llama3.1:70b": ("meta-llama/Llama-3.1-70B", 131072),
            "qwen3:32b": ("Qwen/Qwen3-32B", 40960),
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
        try:
            huggingface_model, tokens_limit = self._huggingface_model_map.get(
                self.llm.model
            )
            tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
        except Exception:
            self.logger.critical(
                f"Failed to load the tokenizer for model {self.llm.model}. We will continue with prompting anyways"
            )
            return True
        num_tokens = len(tokenizer(prompt)["input_ids"])
        if num_tokens > tokens_limit:
            self.logger.critical(
                f"Prompt is too long ({num_tokens}) for the context window ({tokens_limit})."
            )
            return False
        return True

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
        # Check if the prompt is empty or exceeds the token limit
        if not self._check_tokens_limit(prompt=prompt) or not prompt:
            return ""

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
