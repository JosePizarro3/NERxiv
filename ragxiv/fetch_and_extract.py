import re
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from structlog._config import BoundLoggerLazyProxy

import requests
import xmltodict
from langchain_community.document_loaders import PDFMinerLoader, PyPDFLoader

from ragxiv.datamodel import ArxivPaper, Author
from ragxiv.logger import logger


class ArxivFetcher:
    """
    Fetch papers from arXiv and extract text from the queried PDFs.
    """

    def __init__(
        self,
        category: str = "cond-mat.str-el",
        max_results: int = 5,
        **kwargs,
    ):
        """
        Initialize the ArxivFetcher class.
        This class fetches papers from arXiv and extracts text from the queried PDFs.
        It uses the `requests` library to fetch the papers and the `xmltodict` library to parse the XML response.
        It also uses the `PyPDFLoader` and `PDFMinerLoader` from LangChain to extract text from the PDFs.

        Args:
            category (str, optional): The arXiv category to fetch papers from. Default is "cond-mat.str-el".
            max_results (int, optional): The maximum number of results to fetch from arXiv. Default is 5.
        """
        self.category = category
        self.max_results = max_results

        self.logger = kwargs.get("logger", logger)
        self.session = requests.Session()  # Reuse TCP connection
        # ! an initial short paper is used to warm up the `requests` session connection
        # ! otherwise, long papers get stuck on `requests.get()` due to connection timeouts
        self.session.head("http://arxiv.org/pdf/2502.10309v1", timeout=30)

    def fetch(self) -> list:
        """
        Fetch papers from arXiv and extract text from the queried PDFs.
        The `fetch` method is called to fetch the papers from arXiv and stores them in a list of `ArxivPaper` pydantic models.

        Returns:
            list: A list of `ArxivPaper` objects with the metadata of the papers fetched from arXiv.
        """
        # Fetch request from arXiv API and parsing the XML response
        url = f"http://export.arxiv.org/api/query?search_query=cat:{self.category}&start=0&max_results={self.max_results}&sortBy=submittedDate&sortOrder=descending"
        request = urllib.request.urlopen(url)
        data = request.read().decode("utf-8")
        data_dict = xmltodict.parse(data)

        def _get_pages_and_figures(comment: str) -> tuple[int | None, int | None]:
            """
            Gets the number of pages and figures from the comment of the arXiv paper.

            Args:
                comment (str): A string containing the comment of the arXiv paper.

            Returns:
                tuple[int | None, int | None]: A tuple containing the number of pages and figures.
                    If not found, returns (None, None).
            """
            pattern = r"(\d+) *pages*, *(\d+) *figures*"
            match = re.search(pattern, comment)
            if match:
                n_pages, n_figures = match.groups()
                return int(n_pages), int(n_figures)
            return None, None

        # Extracting papers from the XML response
        papers = data_dict.get("feed", {}).get("entry", [])
        if not papers:
            self.logger.info("No papers found in the response")
            return []
        # In case `max_results` is 1, the response is not a list
        if not isinstance(papers, list):
            papers = [papers]

        # Store papers object ArxivPaper in a list
        arxiv_papers = []
        for paper in papers:
            # If there is an error in the fetching, skip the paper
            if "Error" in paper.get("title", ""):
                self.logger.error("Error fetching the paper")
                continue

            # If there is no `id`, skip the paper
            url_id = paper.get("id")
            if not url_id or "arxiv.org" not in url_id:
                self.logger.error(f"Paper without a valid URL id: {url_id}")
                continue

            # If there is no `summary`, skip the paper
            if not paper.get("summary"):
                self.logger.error("Paper without summary/abstract")
                continue

            # Extracting `authors` from the XML response
            paper_authors = paper.get("author", [])
            if not isinstance(paper_authors, list):
                paper_authors = [paper_authors]
            authors = [
                Author(name=author.get("name"), affiliation=author.get("affiliation"))
                for author in paper_authors
            ]
            if not authors:
                self.logger.info("\tPaper without authors.")

            # Extracting `categories` from the XML response
            arxiv_categories = paper.get("category", [])
            if not isinstance(arxiv_categories, list):
                categories = [arxiv_categories.get("@term")]
            else:
                categories = [category.get("@term") for category in arxiv_categories]

            # Extracting pages and figures from the comment
            comment = paper.get("arxiv:comment", {}).get("#text", "")
            n_pages, n_figures = _get_pages_and_figures(comment)

            # Getting arXiv `id`
            id = url_id.split("/")[-1]
            if ".pdf" in id:
                id = id.replace(".pdf", "")

            # Storing the ArxivPaper object in the list
            arxiv_papers.append(
                ArxivPaper(
                    id=id,
                    url=url_id,
                    pdf_url=url_id.replace("abs", "pdf"),
                    updated=paper.get("updated"),
                    published=paper.get("published"),
                    title=paper.get("title"),
                    summary=paper.get("summary"),
                    authors=authors,
                    comment=comment,
                    n_pages=n_pages,
                    n_figures=n_figures,
                    categories=categories,
                )
            )
            self.logger.info(f"Paper {id} fetched from arXiv.")

        return arxiv_papers

    def download_pdf(
        self, arxiv_paper: ArxivPaper, data_folder: str = "data", write: bool = True
    ) -> Path:
        """
        Download the PDF of the arXiv paper and stores it in the `data` folder using the `arxiv_paper.id` to name the PDF file.

        Args:
            arxiv_paper (ArxivPaper): The arXiv paper object to be queried and stored.
            data_folder (str): The folder where to store the PDFs. Defaults to "data" in the root project directory.
            write (bool): If True, the PDF will be written to the `data/` folder. Defaults to True.

        Returns:
            Path: The path to the downloaded PDF file.
        """
        # check if `data_folder` exists, and if not, create it
        Path(data_folder).mkdir(parents=True, exist_ok=True)

        pdf_path = Path("")
        try:
            response = self.session.get(arxiv_paper.pdf_url, stream=True, timeout=60)
            response.raise_for_status()

            pdf_path = Path(f"{data_folder}/{arxiv_paper.id}.pdf")

            if write:
                with open(pdf_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            self.logger.info(f"PDF downloaded: {pdf_path}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download PDF: {e}")
            pdf_path = None
        return pdf_path


class TextExtractor:
    """
    Extract text from the PDF file using LangChain implementation of PDF loaders. This class also
    implements the text cleaning methods.
    """

    def __init__(self, **kwargs):
        self.logger = kwargs.get("logger", logger)

        # Implemented loaders from LangChain
        self.available_loaders = {
            "pypdf": PyPDFLoader,
            "pdfminer": PDFMinerLoader,
        }

    def _check_pdf_path(self, pdf_path: str | None = ".") -> bool:
        """
        Check if the PDF path is valid.

        Args:
            pdf_path (str | None): The path to the PDF file. If None, it will return False.

        Returns:
            bool: True if the PDF path is valid, False otherwise.
        """
        pdf_path = str(pdf_path)  # to avoid potential problems when being a Path object
        if not pdf_path:
            self.logger.error(
                "No PDF path provided. Returning an empty string for the text."
            )
            return False
        return Path(pdf_path).exists() and pdf_path.endswith(".pdf")

    def get_text(self, pdf_path: str | None = ".", loader: str = "pdfminer") -> str:
        """
        Extract text from the PDF file using LangChain implementation of PDF loaders.

        Read more: https://python.langchain.com/docs/how_to/document_loader_pdf/

        Args:
            pdf_path (str | None, optional): The path to the PDF file. Defaults to ".", the root project directory.
            loader (str, optional): The loader to use for extracting text from the PDF file. Defaults to "pdfminer".

        Returns:
            str: The extracted text from the PDF file.
        """
        # Check if the PDF path is valid
        if not self._check_pdf_path(pdf_path=pdf_path):
            return []
        if isinstance(pdf_path, str):
            pdf_path = Path(pdf_path)
        filepath = pdf_path

        # Check if the loader is available
        if loader not in self.available_loaders.keys():
            self.logger.error(
                f"Loader {loader} not available. Available loaders: {self.available_loaders.keys()}"
            )
            return []
        loader_cls = self.available_loaders[loader](filepath)

        # Extract text
        text = ""
        for page in loader_cls.lazy_load():
            text += page.page_content
        return text

    def delete_references(self, text: str = "") -> str:
        """
        Delete the references section from the text by detecting where its section might be.

        Args:
            text (str): The text to delete the references section from.

        Returns:
            str: The text without the references section if a match is found.
        """
        pattern_start = "(?:\nReferences\n|\nBibliography\n|\n\[1\] *[A-Z])"
        pattern_end = "(?:\nSupplemental Material[\:\n]*|\nSupplemental Information[\:\n]*|\nAppendices[\:\n]*)"

        match_start = re.search(pattern_start, text, flags=re.IGNORECASE)
        match_end = re.search(pattern_end, text, flags=re.IGNORECASE)
        if match_start:
            start = match_start.start()
            if match_end:
                end = match_end.start()
                return text[:start] + text[end:]
            return text[:start]
        return text

    def clean_text(self, text: str = "") -> str:
        """
        Clean and normalize extracted PDF text.

        - Remove hyphenation across line breaks.
        - Normalize excessive line breaks and spacing.
        - Remove arXiv identifiers and footnotes.
        - Strip surrounding whitespace.

        Args:
            text (str): Raw text extracted from a PDF.

        Returns:
            str: Cleaned text.
        """
        if not text:
            self.logger.warning("No text provided for cleaning.")
            return ""

        # Fix hyphenated line breaks: e.g., "super-\nconductivity" → "superconductivity"
        text = re.sub(r"-\s*\n\s*", "", text)

        # Replace multiple newlines with a single newline
        text = re.sub(r"\n{2,}", "\n\n", text)

        # Remove arXiv identifiers like 'arXiv:2301.12345'
        text = re.sub(r"arXiv:\d{4}\.\d{4,5}(v\d+)?", "", text)

        # Normalize spacing
        text = re.sub(r"[ \t]+", " ", text)  # collapse multiple spaces/tabs
        text = re.sub(r"\n[ \t]+", "\n", text)  # remove indentations

        # Replace newline characters with spaces
        text = re.sub(r"\n+", " ", text)

        return text.strip()


def fetch_and_extract(
    category: str = "cond-mat.str-el",
    max_results: int = 5,
    data_folder: str = "data",
    loader: str = "pdfminer",
    logger: "BoundLoggerLazyProxy" = logger,
) -> list[ArxivPaper]:
    """
    Fetch papers from arXiv and extract text from the queried PDFs. In order to do one-shots, use
    the method `fetch_and_extract`.

    This function initializes the `ArxivFetecher` and `TextExtractor` classes, following the workflow:

        -> Fetches the papers from arXiv and stores them in a list of `ArxivPaper` pydantic models.
        -> For each paper, it downloads the PDF storing it in `data_folder` and extracts the text from it.
        -> For each paper, it deletes the references section and cleans the extracted text.
        -> The text is stored in the `text` attribute of each `ArxivPaper` object.

    Args:
        category (str, optional): The arXiv category. Defaults to "cond-mat.str-el".
        max_results (int, optional): The maximum results for pagination for the arXiv API call. Defaults to 5.
        data_folder (str, optional): The folder where to store the PDFs. Defaults to "data".
        loader (str, optional): The loader to use for extracting text from the PDF file. Defaults to "pdfminer".
        logger (BoundLoggerLazyProxy, optional): The logger to log messages. Defaults to logger.

    Returns:
        list[ArxivPaper]: A list of ArxivPaper objects with the text extracted from the PDFs.
    """
    # Initializes the `fetcher` (in this case from arXiv) and the `text_extractor` classes
    fetcher = ArxivFetcher(logger=logger, category=category, max_results=max_results)
    text_extractor = TextExtractor(logger=logger)

    # Fetch the papers from arXiv and stores them in a list of `ArxivPaper` pydantic models
    papers = fetcher.fetch()
    logger.info(f"{max_results} papers fetched from arXiv, {category}.")

    # For each paper, it downloads the PDF storing it in `data_folder` and extracts the text from it
    # The text is stored in the `text` attribute of each `ArxivPaper` object
    for paper in papers:
        # ! note it is more efficient to download the PDF and extract the text because long papers cannot extract
        # ! the text on the fly and the connection times out
        # Download the PDF to `data_folder`
        pdf_path = fetcher.download_pdf(data_folder=data_folder, arxiv_paper=paper)

        # Extract text from the PDF
        text = text_extractor.get_text(pdf_path=pdf_path, loader=loader)
        if not text:
            logger.info("No text extracted from the PDF.")
            continue

        # Delete references and clean text
        # ! note that deleting the references must be done first
        text = text_extractor.delete_references(text=text)
        text = text_extractor.clean_text(text=text)

        # Store the text in the `text` attribute of the `ArxivPaper` object
        paper.text = text
        logger.info(f"Text extracted from {paper.id} and stored in model.")
    return papers
