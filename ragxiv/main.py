from pathlib import Path

from pymatgen.core import Composition
from pyrxiv.extract import TextExtractor

from ragxiv.chunker import Chunker
from ragxiv.prompts import (
    MATERIAL_CATEGORIZATION_PROMPT,
    MATERIAL_TEMPLATE,
    prompt,
)
from ragxiv.rag import CustomRetriever, LangChainRetriever, LLMGenerator, answer_to_dict

clean_text = True  # set to False to keep the references in the text

# list all papers `data/*.pdf`
paper_pdfs = list(Path("data").rglob("*.pdf"))

extractor = TextExtractor()

for pdf_path in paper_pdfs:
    text = extractor.get_text(pdf_path=pdf_path)
    if clean_text:
        text = extractor.delete_references(text=text)
        text = extractor.clean_text(text=text)

    chunker = Chunker(text=text)
    chunks = chunker.chunk_text()

    categorizer = CustomRetriever(query=MATERIAL_CATEGORIZATION_PROMPT)
    text = categorizer.get_relevant_chunks(chunks=chunks, n_top_chunks=5)

    generator = LLMGenerator(model="llama3.1:70b", text=text)
    answer_material = generator.generate(prompt=prompt(MATERIAL_TEMPLATE, text=text))
    if answer_material != "model":
        try:  # in case the answer contains non-chemical elements
            formulas = answer_material.split(",")
            for formula in formulas:
                composition = Composition(formula)
        except Exception:
            continue

    print(f"Paper: {pdf_path}, Materials: {answer_material}")
