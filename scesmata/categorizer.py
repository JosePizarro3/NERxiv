import json
import re
from typing import Optional

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, util

from scesmata.logger import logger


# ! deprecated, use langchain better
class Categorizer:
    def __init__(self, **kwargs):
        self.logger = kwargs.get("logger", logger)
        self.text = kwargs.get("text", "")

        # TODO test other models in SentenceTransformer
        model_name = kwargs.get("model", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_name)
        self.logger.info(f"Loaded SentenceTransformer model: {model_name}")

    # def get_relevant_chunks(
    #     self,
    #     chunks: list[str] = [],
    #     n_top_chunks: int = 10,
    #     query: Optional[str] = None,
    # ) -> list[str]:
    #     """Find the most relevant chunks describing methods."""
    #     if not chunks:
    #         self.logger.warning("No chunks provided.")
    #         return []
    #     query_embedding = self.model.encode(query, convert_to_tensor=True)
    #     chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)

    #     # TODO check other similarities
    #     similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings).squeeze(
    #         0
    #     )
    #     sorted_similarities = similarities.sort(descending=True)
    #     # Get the top `n_top_chunks` chunks with the highest similarity score with respect to the query
    #     top_chunks = [chunks[i] for i in sorted_similarities.indices[:n_top_chunks]]
    #     self.logger.info(
    #         f"Top {n_top_chunks} chunks retrieved with similarities of {sorted_similarities.values[:n_top_chunks]}"
    #     )
    #     return top_chunks


query = (
    "Identify all mentions of scientific methods used in this text, "
    "especially those relevant to Condensed Matter Physics. Look for full names "
    "(e.g., Density Functional Theory, Quantum Monte Carlo) and abbreviations "
    "(e.g., DFT, QMC, DMFT, ED). Include any computational, analytical, or numerical techniques."
)

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

from scesmata.langchain import LangchainTextExtractor

extractor = LangchainTextExtractor(logger=logger)
pages = extractor.pages(pdf_path="./tests/data/2502.12144v1.pdf")
new_pages = []
for page in pages:
    new_pages.append(
        Document(
            page_content=extractor.clean_text(page.page_content), metadata=page.metadata
        )
    )
chunks = extractor.chunk_text(pages=new_pages, chunk_size=500)

from langchain_huggingface import HuggingFaceEmbeddings

# embeddings = OllamaEmbeddings(model="llama3")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=chunks)

results = vector_store.similarity_search(query)
# query_task_1 = """You are an expert in Condensed Matter Physics. You have the task of analyzing the following text
# and identify if the article is describing an experimental or a computational activity, or a combination of both.

# Return your answer as a list with the format ["experimental", "computational", "both"].
# """


# from scesmata.fetch import ArxivFetcher, TextExtractor, fetch_and_extract

# # papers = ArxivFetcher().fetch_and_extract(max_results=1)
# # paper = papers[0]
# # text = paper.text
# extractor = TextExtractor()
# # text = extractor.from_pdf(pdf_path="./tests/data/2502.10309v1.pdf")
# text = extractor.from_pdf(pdf_path="./tests/data/2502.12144v1.pdf")
# text = extractor.clean_text(text=text)
# text = extractor.delete_references(text=text)
# top_chunks = Categorizer(text=text).get_relevant_chunks(n_top_chunks=10, query=query)


# def sces_methods_list() -> list[str]:
#     with open("scesmata/validation/sces_methods.json") as file:
#         data = json.load(file)
#     # Flatten the list of methods
#     methods = []
#     for field in ["computational", "experimental"]:
#         data_field = data[field]
#         for key, values in data_field.items():
#             methods.append(key)
#             for v in values:
#                 methods.append(v)
#     return methods
# class LLMQuery:
#     @property
#     def field(self) -> str:
#         return """
#         The following text contains a research article in the field of Strongly Correlated Electrons System in Condensed Matter Physics.
#         """

#     @property
#     def query(self):
#         """
#         Defines the query using the `sces_methods.json` information.
#         """
#         return f"""What experimental or computational methods were used in this text? The field is Condensed Matter Physics applied to strongly correlated electrons systems

#         Typical examples are defined in this list: {sces_methods_list()}

#         Do not constraint yourself to the values in the list, but based your answer mostly on the list."""

#     @property
#     def query_exp_comp(self) -> str:
#         """
#         Defines the query using the `sces_methods.json` information.
#         """
#         return f"""{self.field}

#         Is the article describing an experimental or a computational activity, or a combination of both?

#         Return your answer as a list with the format ["experimental", "computational", "both"].
#         """

#     @property
#     def query_method(self):
#         """
#         Defines the query using the `sces_methods.json` information.
#         """
#         return f"""{self.field}

#         What experimental or computational methods were used in this text?

#         Typical examples are defined in this list: {sces_methods_list()}

#         Do not constraint yourself to the values in the list, but based your answer mostly on the list."""


# class Categorizer:
#     def __init__(self, **kwargs):
#         self.logger = kwargs.get("logger", logger)
#         self.text = kwargs.get("text", "")
#         self.papers = kwargs.get("papers", [])

#     @property
#     def prompt(self, chunk: str):
#         prompt = f"""You are an expert in Condensed Matter Physics. Your task is to analyze the following text
#         and identify the experimental or computational methods used.

#         Map each method to its canonical form based on the following list: {sces_methods_list()}

#         Return the methods in a list format, using the same strings as in the list mentioned above.
#         If a method is not in the list but is relevant, include it as-is.
#         No extract text, only the list of methods used in the text.

#         Text: "{chunk}"
#         """


# # from transformers import AutoTokenizer

# # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# # def count_tokens(text: str) -> int:
# #     return len(tokenizer.encode(text, add_special_tokens=False))


# # for chunk in top_chunks:
# #     print(f"Chunk length: {count_tokens(chunk)}")


# # from langchain_community.llms.llamacpp import LlamaCpp
# # from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
# # from langchain_core.prompts import PromptTemplate

# # # Callbacks support token-wise streaming
# # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# # llm = LlamaCpp(
# #     model_path="/home/jpizarro/work/llmodels/llama-2-7b.Q4_K_M.gguf",
# #     callback_manager=callback_manager,
# #     max_tokens=500,
# #     verbose=True,
# # )
