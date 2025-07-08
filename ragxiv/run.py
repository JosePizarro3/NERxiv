from bam_masterdata.metadata.entities import CollectionType
from bam_masterdata.openbis.login import ologin

from ragxiv.parsing import DMFTArxivParser
from ragxiv.prompts import (
    REAL_MATERIAL_TEMPLATE,
    prompt,
)
from ragxiv.rag import CustomRetriever, LLMGenerator
from ragxiv.text import Chunker, download_pattern_papers

CHUNK_SYSTEM_PROMPT = (
    "You are a helpful assistant that retrieves relevant chunks from a scientific text. "
    "Identify all mentions to the system simulated in the text. This can be a toy model, "
    "for example, square lattice, honeycomb lattice, etc., or a real material with a chemical formula."
)


files, papers = download_pattern_papers()

for paper in papers:
    chunks = Chunker(text=paper.text).chunk_text(chunk_size=800)
    relevant_text = CustomRetriever(query=CHUNK_SYSTEM_PROMPT).get_relevant_chunks(
        chunks=chunks
    )
    generator = LLMGenerator(model="llama3.1:405b", text=relevant_text)
    answer = generator.generate(prompt=REAL_MATERIAL_TEMPLATE)


DMFTArxivParser().parse(files=files, collection=CollectionType())
openbis = ologin(url="https://main.datastore.bam.de")
