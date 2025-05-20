pdf = "./tests/data/2502.12144v1.pdf"

from ragxiv.chunker import Chunker
from ragxiv.fetch_and_extract import TextExtractor
from ragxiv.generator import LLMGenerator, answer_to_dict
from ragxiv.prompts import (
    EXP_OR_COMP_TEMPLATE,
    EXTRACT_METHODS_TEMPLATE,
    FILTER_METHODS_TEMPLATE,
    prompt,
)
from ragxiv.retriever import CustomRetriever, LangChainRetriever

extractor = TextExtractor()
text = extractor.get_text(pdf_path=pdf, loader="pdfminer")
text = extractor.delete_references(text=text)
text = extractor.clean_text(text=text)

chunks = Chunker(text=text).chunk_text(chunk_size=800)

categorizer = CustomRetriever()
text = categorizer.get_relevant_chunks(chunks=chunks, n_top_chunks=5)

# categorizer_2 = LangChainRetriever()
# text_2 = categorizer_2.get_relevant_chunks(chunks=chunks, n_top_chunks=5)

# generator = LLMGenerator(model="deepseek-r1", text=text)
generator = LLMGenerator(model="llama3.1:70b", text=text)

# ! define experiment or computation
answer_exp_or_comp = generator.generate(prompt=prompt(EXP_OR_COMP_TEMPLATE, text=text))
if answer_exp_or_comp not in ["computational", "experimental", "both", "none"]:
    raise ValueError(
        f"Answer is not valid. Expected one of ['computational', 'experimental', 'both', 'none'], but got\n\n{answer_exp_or_comp}"
    )
# ! extract methods
answer_methods = generator.generate(
    prompt=prompt(EXTRACT_METHODS_TEMPLATE, text=text, exp_or_comp=answer_exp_or_comp)
)
# ! filter in case there are some wrongly identified softwares or instruments
answer_filtered_methods = generator.generate(
    prompt=prompt(FILTER_METHODS_TEMPLATE, candidates=answer_methods)
)
