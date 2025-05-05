import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""# RAGxiv tutorial 1""")


@app.cell
def _(mo):
    mo.md("""## Fetching and extracting text from arXiv papers""")


@app.cell
def _(mo):
    mo.md(
        """We are going to fetch arXiv from papers from a specific `category` and download them as PDFs in a local `./data/` folder. We will then use the downloaded PDFs to extract and clean the text."""
    )


@app.cell
def _():
    from ragxiv.fetch_and_extract import ArxivFetcher

    return (ArxivFetcher,)


@app.cell
def _(ArxivFetcher):
    fetcher = ArxivFetcher(max_results=3)
    print(
        f"arXiv category={fetcher.category}, and numer of papers fetched={fetcher.max_results}"
    )
    return (fetcher,)


@app.cell
def _(fetcher):
    arxiv_papers = fetcher.fetch()
    return (arxiv_papers,)


@app.cell
def _(arxiv_papers, fetcher):
    pdf_paths = []
    for paper in arxiv_papers:
        pdf_paths.append(fetcher.download_pdf(arxiv_paper=paper, data_folder="data/"))
    return (pdf_paths,)


@app.cell
def _(pdf_paths):
    pdf_paths


@app.cell
def _(mo):
    mo.md("""## Extracting and cleaning text from PDFs""")


@app.cell
def _(mo):
    mo.md(
        """We can extract text from the downloaded PDFs to use it later on to feed an LLM model to extract structured metadata."""
    )


@app.cell
def _(pdf_paths):
    from ragxiv.fetch_and_extract import TextExtractor

    extractor = TextExtractor()
    text = extractor.get_text(pdf_path=pdf_paths[0], loader="pypdf")
    # for i, pdf_path in enumerate(pdf_paths):
    #     text = extractor.get_text(pdf_path=pdf_path, loader="pypdf")
    return (text,)


@app.cell
def _(pdf_paths):
    print(pdf_paths[0], type(pdf_paths[0]))


@app.cell
def _(text):
    len(text)


@app.cell
def _(text):
    text[200:250]


@app.cell
def _(text):
    text[0:100]


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
