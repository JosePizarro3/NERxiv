pdf_1 = "./tests/data/2502.10309v1.pdf"
pdf_2 = "./tests/data/2502.12144v1.pdf"

# MAIN CLASSES
# from langchain_core.documents import Document

# documents = [
#     Document(
#         page_content="Dogs are great companions, known for their loyalty and friendliness.",
#         metadata={"source": "mammal-pets-doc"},
#     ),
#     Document(
#         page_content="Cats are independent pets that often enjoy their own space.",
#         metadata={"source": "mammal-pets-doc"},
#     ),
# ]


# EXTRACT TEXT
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(pdf_1)
pages = loader.load()
# pages = []
# for page in loader.lazy_load():
#     pages.append(page)

# CHUNKING
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
chunks = text_splitter.split_documents(pages)
