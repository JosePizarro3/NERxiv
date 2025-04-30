import json
import re

from langchain_ollama.llms import OllamaLLM
from transformers import AutoTokenizer

from scesmata.logger import logger


class LLMGenerator:
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
        if not self._check_tokens_limit(prompt=prompt) or not prompt:
            return ""

        return self.llm.invoke(prompt)

    def clean_answer(self, answer: str = "") -> str:
        match = re.search(r"\n\n\[\n *\{", answer, flags=re.IGNORECASE)
        if match:
            start = match.start()
            answer = answer[start:]
            answer = re.sub(r"\n\n", "", answer)
        return answer
