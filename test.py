pdf_1 = "./tests/data/2502.10309v1.pdf"
pdf_2 = "./tests/data/2502.12144v1.pdf"

from scesmata.fetch_and_extract import TextExtractor
from scesmata.generator import LLMGenerator
from scesmata.prompts import EXTRACT_METHODS_TEMPLATE, FILTER_METHODS_TEMPLATE, prompt
from scesmata.retriever import CustomRetriever, LangChainRetriever

extractor = TextExtractor()
text = extractor.get_text(pdf_path=pdf_2, loader="pdfminer")
text = extractor.delete_references(text=text)
text = extractor.clean_text(text=text)
chunks = extractor.chunk_text(text=text, chunk_size=500)

categorizer = CustomRetriever()
text = categorizer.get_relevant_chunks(chunks=chunks, n_top_chunks=10)

# categorizer_2 = LangChainRetriever()
# text_2 = categorizer_2.get_relevant_chunks(chunks=chunks, n_top_chunks=5)


generator = LLMGenerator(model="deepseek-r1", text=text)
answer = generator.generate(prompt=prompt(EXTRACT_METHODS_TEMPLATE, text=text))
answer = generator.clean_answer(answer=answer)
answer = generator.generate(prompt=prompt(FILTER_METHODS_TEMPLATE, candidates=answer))
answer = generator.clean_answer(answer=answer)
