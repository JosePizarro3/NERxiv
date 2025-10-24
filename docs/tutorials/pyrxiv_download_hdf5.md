# Downloading arXiv papers as HDF5 files

This tutorial demonstrates how to use the [pyrxiv](https://pypi.org/project/pyrxiv/) package to fetch and download arXiv papers, and store them efficiently in HDF5 format for later processing.

## Overview

[pyrxiv](https://github.com/JosePizarro3/pyrxiv) is a Python package that provides convenient access to arXiv papers. It allows you to:

- Search and fetch papers from arXiv by category, query, or ID
- Download PDFs locally
- Extract and clean text from PDFs
- Store paper metadata and content in HDF5 format for efficient storage and retrieval

While NERxiv focuses on metadata extraction using LLMs, it relies on pyrxiv for the initial data acquisition step.

## Installation

First, ensure you have pyrxiv installed:

```bash
pip install pyrxiv
```

pyrxiv is automatically installed as a dependency when you install NERxiv:

```bash
pip install nerxiv
```

## Basic Usage

### 1. Fetching Papers by Category

You can fetch papers from arXiv based on their subject category:

```python
from pyrxiv import ArxivFetcher

# Initialize the fetcher
fetcher = ArxivFetcher(
    category='cond-mat.str-el',  # Condensed matter - strongly correlated electrons
    max_results=10,              # Number of papers to fetch
    output_dir='data/arxiv_papers'  # Where to save PDFs
)

# Fetch papers
papers = fetcher.fetch()

# Display basic information
for paper in papers:
    print(f"ID: {paper.id}")
    print(f"Title: {paper.title}")
    print(f"Authors: {', '.join(paper.authors)}")
    print(f"Published: {paper.published}")
    print("---")
```

### 2. Downloading PDFs

Download the fetched papers as PDF files:

```python
# Download all fetched papers
fetcher.download_pdfs()

# Or download specific papers
for paper in papers[:5]:  # Download only first 5
    paper.download_pdf(output_dir='data/pdfs')
```

### 3. Extracting Text from PDFs

Extract text content from downloaded PDFs:

```python
from pyrxiv import PDFParser

# Extract text from a single PDF
parser = PDFParser(pdf_path='data/pdfs/2301.12345.pdf')
text = parser.extract_text()

# Clean the extracted text (remove headers, footers, references)
cleaned_text = parser.clean_text(text)

print(f"Original length: {len(text)} characters")
print(f"Cleaned length: {len(cleaned_text)} characters")
```

### 4. Storing Papers in HDF5 Format

HDF5 (Hierarchical Data Format) provides efficient storage for large datasets:

```python
from pyrxiv import HDF5Storage
import os

# Initialize HDF5 storage
storage = HDF5Storage(filepath='data/arxiv_papers.h5')

# Store paper metadata and text
for paper in papers:
    pdf_path = os.path.join('data/pdfs', f'{paper.id}.pdf')
    
    if os.path.exists(pdf_path):
        # Extract and clean text
        parser = PDFParser(pdf_path=pdf_path)
        text = parser.extract_text()
        cleaned_text = parser.clean_text(text)
        
        # Store in HDF5
        storage.store_paper(
            paper_id=paper.id,
            metadata={
                'title': paper.title,
                'authors': paper.authors,
                'abstract': paper.abstract,
                'categories': paper.categories,
                'published': paper.published,
                'doi': paper.doi if hasattr(paper, 'doi') else None
            },
            text=cleaned_text
        )

print(f"Stored {len(papers)} papers in HDF5 format")
```

### 5. Reading from HDF5 Storage

Retrieve papers from HDF5 for later processing:

```python
# Read a specific paper
paper_data = storage.read_paper(paper_id='2301.12345')

print(f"Title: {paper_data['metadata']['title']}")
print(f"Text length: {len(paper_data['text'])} characters")
print(f"Abstract: {paper_data['metadata']['abstract'][:200]}...")

# List all stored papers
paper_ids = storage.list_papers()
print(f"Total papers in storage: {len(paper_ids)}")

# Close the HDF5 file when done
storage.close()
```

## Complete Workflow Example

Here's a complete example that fetches, downloads, extracts, and stores papers:

```python
from pyrxiv import ArxivFetcher, PDFParser, HDF5Storage
import os

# Configuration
CATEGORY = 'cond-mat.str-el'
MAX_RESULTS = 20
OUTPUT_DIR = 'data/arxiv_papers'
PDF_DIR = os.path.join(OUTPUT_DIR, 'pdfs')
HDF5_FILE = os.path.join(OUTPUT_DIR, 'papers.h5')

# Ensure directories exist
os.makedirs(PDF_DIR, exist_ok=True)

# Step 1: Fetch papers from arXiv
print("Fetching papers from arXiv...")
fetcher = ArxivFetcher(
    category=CATEGORY,
    max_results=MAX_RESULTS,
    output_dir=PDF_DIR
)
papers = fetcher.fetch()
print(f"Fetched {len(papers)} papers")

# Step 2: Download PDFs
print("Downloading PDFs...")
fetcher.download_pdfs()

# Step 3: Extract text and store in HDF5
print("Extracting text and storing in HDF5...")
storage = HDF5Storage(filepath=HDF5_FILE)

for i, paper in enumerate(papers, 1):
    pdf_path = os.path.join(PDF_DIR, f'{paper.id}.pdf')
    
    if os.path.exists(pdf_path):
        print(f"Processing {i}/{len(papers)}: {paper.id}")
        
        # Extract and clean text
        parser = PDFParser(pdf_path=pdf_path)
        text = parser.extract_text()
        cleaned_text = parser.clean_text(text)
        
        # Store in HDF5
        storage.store_paper(
            paper_id=paper.id,
            metadata={
                'title': paper.title,
                'authors': paper.authors,
                'abstract': paper.abstract,
                'categories': paper.categories,
                'published': str(paper.published),
            },
            text=cleaned_text
        )

storage.close()
print(f"Complete! Papers stored in {HDF5_FILE}")
```

## Using with NERxiv

Once you have papers stored, you can use NERxiv to extract structured metadata:

```python
from nerxiv import RAGExtractor
from pyrxiv import HDF5Storage

# Load papers from HDF5
storage = HDF5Storage(filepath='data/arxiv_papers/papers.h5')
paper_data = storage.read_paper(paper_id='2301.12345')

# Extract metadata using NERxiv
extractor = RAGExtractor(
    text=paper_data['text'],
    model='llama3',
    query='Extract the computational methods used in this paper'
)

metadata = extractor.extract()
print(metadata)

storage.close()
```

## Tips and Best Practices

1. **Rate Limiting**: arXiv has rate limits. Avoid fetching too many papers too quickly. Use reasonable `max_results` values and add delays between large batches.

2. **Storage Efficiency**: HDF5 format is much more efficient than storing individual text files, especially for large collections. It also preserves metadata and enables fast queries.

3. **Text Cleaning**: Always clean extracted text before using it with LLMs. PDFs often contain artifacts, headers, footers, and formatting issues that can confuse models.

4. **Error Handling**: Not all PDFs can be successfully parsed. Implement error handling to skip problematic papers:

   ```python
   try:
       parser = PDFParser(pdf_path=pdf_path)
       text = parser.extract_text()
   except Exception as e:
       print(f"Failed to parse {pdf_path}: {e}")
       continue
   ```

5. **Memory Management**: When processing many papers, consider processing them in batches and closing HDF5 files properly to avoid memory issues.

## Advanced Features

### Searching by Query

Instead of categories, you can search arXiv using text queries:

```python
fetcher = ArxivFetcher(
    query='quantum computing AND machine learning',
    max_results=50
)
papers = fetcher.fetch()
```

### Fetching Specific Papers by ID

If you know the arXiv IDs you want:

```python
from pyrxiv import ArxivPaper

# Fetch a specific paper
paper = ArxivPaper.from_id('2301.12345')
paper.download_pdf(output_dir='data/pdfs')
```

### Custom Text Cleaning

You can customize the text cleaning process:

```python
parser = PDFParser(pdf_path='paper.pdf')
text = parser.extract_text()

# Custom cleaning options
cleaned_text = parser.clean_text(
    text,
    remove_headers=True,
    remove_footers=True,
    remove_references=True,
    remove_figures=True,
    min_word_length=3
)
```

## Next Steps

Now that you know how to download and store arXiv papers, you can:

- Learn about [Using the RAG Extractor Agent](rag_extractor_tutorial.md) to extract structured metadata
- Explore [Custom Prompts](../howtos/create_custom_prompts.md) for specific extraction tasks
- Read about [Chunking Strategies](../howtos/customize_chunking.md) for optimal text processing

## References

- [pyrxiv GitHub Repository](https://github.com/JosePizarro3/pyrxiv)
- [pyrxiv PyPI Package](https://pypi.org/project/pyrxiv/)
- [arXiv API Documentation](https://arxiv.org/help/api/)
- [HDF5 for Python (h5py)](https://docs.h5py.org/)