import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md("""# RAGxiv tutorial 1 - Extracting text from arXiv papers""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    This tutorial is divided in the following sections:

    1. Fetching and extracting text from arXiv papers
    2. Extracting and cleaning text from PDFs
    3. One-shot fetching and extracting

    The last section can be used to do fetching and extraction of text in one-shot, instead of the whole workflow using the first two sections.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""## Fetching and extracting text from arXiv papers""")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""We are going to fetch papers from arXiv from a specific `category` and download them as PDFs in a local `tutorials/data/` folder. We will then use the downloaded PDFs to extract their text and use the package functionalities to clean the text to prepare it for LLMs structured metadata extraction.""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""We import the `ArxivFetcher` class and fetch a number of papers defined by `max_results`.""")
    return


@app.cell
def _():
    from nerxiv.text import ArxivFetcher


    # We can specify the `category` (by default this is `"cond-mat.str-el"`) and the `max_results` we want to query from arXiv
    fetcher = ArxivFetcher(max_results=3)
    print(f"arXiv category={fetcher.category}, and numer of papers fetched={fetcher.max_results}\n")

    # We fetch the papers from arxiv.
    arxiv_papers = fetcher.fetch()
    return arxiv_papers, fetcher


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    The function `ArxivFetcher.fetch()` returns a list of objects of type `ArxivPaper`. `ArxivPaper` is a pydantic model containing the metadata of the fetched paper.

    Amongst the fields of `ArxivPaper`, there is `text`. This is empty for now, but in the next section [2. Extracting and cleaning text from PDFs](#extracting-and-cleaning-text-from-pdfs), `text` will be populated.
    """
    )
    return


@app.cell
def _(arxiv_papers):
    print(f"An `ArxivPaper` object fields are: {arxiv_papers[0]}")
    print(f"Its `text` field is: {arxiv_papers[0].text}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    /// attention | Marimo fields

        If you are running this marimo notebook locally, you will note that, in fact, `text` is not empty. This is because the variables in marimo are dynamical, hence they update globally.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    For each of the fetched papers, we downloaded them locally under `tutorials/data/`. We do this in order to extract the text later on.

    We will use the function `ArxivFetcher.download_pdf()`.
    """
    )
    return


@app.cell
def _(arxiv_papers, fetcher):
    pdf_paths = []
    for paper in arxiv_papers:
        pdf_paths.append(fetcher.download_pdf(arxiv_paper=paper, data_folder="tutorials/data/"))
    return (pdf_paths,)


@app.cell(hide_code=True)
def _():
    mo.md("""## Extracting and cleaning text from PDFs""")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""We can extract text from the downloaded PDFs to use it later on to feed an LLM model to extract structured metadata. Once the text is extracted, we will populate the corresponding field in the `ArxivPaper` pydantic object.""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    We import the `TextExtractor` class and usethe method `get_text()` for each of the downloaded PDFs (using `pdf_paths`). The `loader` can be [pypdf](https://pypdf.readthedocs.io/en/stable/) or [pdfminer](https://pdfminersix.readthedocs.io/en/latest/), both producing very similar results.

    We append the extracted text to a general list in order to visualize the results.
    """
    )
    return


@app.cell
def _(pdf_paths):
    from nerxiv.text import TextExtractor

    extractor = TextExtractor()
    all_texts = []
    for pdf_path in pdf_paths:
        all_texts.append(extractor.get_text(pdf_path=pdf_path, loader="pypdf"))

    all_texts
    return all_texts, extractor


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    These extracted texts can then populate their corresponding field for each of the `ArxivPaper` pydantic objects. However, extracting text from an article in PDF format comes with some issues:

    1. The text contains an unnecessary References section.
    2. The text contains a lot of unnecessary characters, like `"\n"` or `-`.

    That's why, we are going to clean the text before proceeding with the structured metadata extraction.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""Note that we would need to delete first the References section, and then clean the text. Otherwise, the `TextExtractor.delete_references()` method will not work.""")
    return


@app.cell
def _(all_texts, extractor):
    new_texts = []
    for text in all_texts:
        no_references_text = extractor.delete_references(text=text)
        cleaned_text = extractor.clean_text(text=no_references_text)
        new_texts.append(cleaned_text)
    return (new_texts,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""We can check that, indeed, the text is more readable now.""")
    return


@app.cell
def _(all_texts):
    all_texts[0][1000:1200]
    return


@app.cell
def _(new_texts):
    new_texts[0][1000:1200]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""Lastly, we can now populate the `text` field in each of the `ArxivPaper` pydantic objects with the cleaned text.""")
    return


@app.cell
def _(arxiv_papers, new_texts):
    for i, clean_text in enumerate(new_texts):
        arxiv_papers[i].text = clean_text
    return


@app.cell
def _(arxiv_papers):
    print(f"An `ArxivPaper` object fields are: {arxiv_papers[0]}")
    print(f"Its `text` field is: {arxiv_papers[0].text}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## One-shot fetching and extracting""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""The above functionalities are combined in a single-shot function called `fetch_and_extract()`. This function return a list of `ArxivPaper` pydantic objects with their `text` field populated with the cleaned text.""")
    return


@app.cell
def _():
    from nerxiv.text import arxiv_fetch_and_extract


    arxiv_papers_one_shot = arxiv_fetch_and_extract(max_results=3, data_folder="tutorials/data/", loader="pypdf")
    return (arxiv_papers_one_shot,)


@app.cell
def _(arxiv_papers):
    print(
        f"{arxiv_papers[0]}\n",
        f"{arxiv_papers[0].text}",
    )
    return


@app.cell
def _(arxiv_papers_one_shot):
    print(
        f"{arxiv_papers_one_shot[0]}\n",
        f"{arxiv_papers_one_shot[0].text}",
    )
    return


if __name__ == "__main__":
    app.run()
