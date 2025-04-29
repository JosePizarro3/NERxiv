import io
import re
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from structlog._config import BoundLoggerLazyProxy

import requests
import xmltodict
from pypdf import PdfReader

from scesmata.datamodel import ArxivPaper, Author
from scesmata.logger import logger


class ArxivFetcher:
    """
    Fetch papers from arXiv and extract text from the queried PDFs. In order to do one-shots, use
    the method `fetch_and_extract`.
    """

    def __init__(
        self,
        category: str = "cond-mat.str-el",
        max_results: int = 5,
        **kwargs,
    ):
        self.category = category
        self.max_results = max_results

        self.logger = kwargs.get("logger", logger)
        self.session = requests.Session()  # Reuse TCP connection
        # ! an initial short paper is used to warm up the `requests` session connection
        # ! otherwise, long papers get stuck on `requests.get()` due to connection timeouts
        self.session.head("http://arxiv.org/pdf/2502.10309v1", timeout=30)

    def fetch(self) -> list:
        """
        Fetch papers from arXiv and stores them in a list of ArxivPaper pydantic models.

        Returns:
            list: A list of ArxivPaper objects.
        """
        # Fetch request from arXiv API and parsing the XML response
        url = f"http://export.arxiv.org/api/query?search_query=cat:{self.category}&start=0&max_results={self.max_results}&sortBy=submittedDate&sortOrder=descending"
        request = urllib.request.urlopen(url)
        data = request.read().decode("utf-8")
        data_dict = xmltodict.parse(data)

        def _get_pages_and_figures(comment: str) -> tuple[Optional[int], Optional[int]]:
            """
            Gets the number of pages and figures from the comment of the arXiv paper.

            Args:
                comment (str): A string containing the comment of the arXiv paper.

            Returns:
                tuple: A tuple containing the number of pages and figures.
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

        return arxiv_papers

    def download_pdf(
        self, arxiv_paper: ArxivPaper, data_folder: str = "data", write: bool = True
    ) -> str:
        """
        Download the PDF of the arXiv paper and stores it in the `data` folder using the `arxiv_paper.id` to name the PDF file.

        Args:
            arxiv_paper (ArxivPaper): The arXiv paper object to be queried and stored.
            data_folder (str): The folder where to store the PDFs.
            write (bool): If True, the PDF will be written to the `data/` folder.

        Returns:
            str: The path to the downloaded PDF file.
        """
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
    Extract text from the PDF file using `pypdf` library in the `from_pdf` method. The `delete_references`
    method deletes the references section from the text by detecting where its section might be.
    """

    def __init__(self, **kwargs):
        self.logger = kwargs.get("logger", logger)

    def from_pdf(self, pdf_path: Optional[str] = ".") -> str:
        """
        Extract text from a PDF locally stored in `pdf_path`.

        Args:
            pdf_path (Optional[str]): The path to the PDF file.

        Returns:
            str: The text extracted from the PDF.
        """
        if not pdf_path:
            self.logger.error(
                "No PDF path provided. Returning an empty string for the text."
            )
            return ""
        filepath = Path(pdf_path)
        if not filepath.exists() or not pdf_path.endswith(".pdf"):
            self.logger.error(
                "Could not find the PDF file. Returning an empty string for the text."
            )
            return ""
        text = ""
        with open(filepath, "rb") as f:
            reader = PdfReader(f)
            text = "\n".join(
                [page.extract_text() for page in reader.pages if page.extract_text()]
            )
        return text

    def delete_references(self, text: str) -> str:
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

    def chunk_text(self, text: str = "", max_length: int = 500) -> list[str]:
        """
        Chunk the text into smaller parts at sentence boundaries to avoid long texts in the prompt.

        Args:
            text (str, optional): The text to be chunked. Defaults to "".
            max_length (int, optional): The maximum length of characters of each chunk. Defaults to 500.

        Returns:
            list[str]: A list of strings containing the text chunked into smaller parts.
        """
        if not text:
            self.logger.warning("No text provided for chunking.")
            return []
        if max_length <= 0:
            self.logger.warning("Max length must be greater than 0.")
            return []

        # split at sentence boundaries
        sentences = re.split(r"(?<=[.!?]) +", text)
        chunks, current_chunk = [], []

        for sentence in sentences:
            if sum(len(s) for s in current_chunk) + len(sentence) < max_length:
                current_chunk.append(sentence)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        self.logger.info(f"Text chunked into {len(chunks)} parts")
        return chunks


def fetch_and_extract(
    data_folder: str = "data",
    delete_references: bool = True,
    logger: "BoundLoggerLazyProxy" = logger,
    category: str = "cond-mat.str-el",
    max_results: int = 5,
) -> list[ArxivPaper]:
    """
    Fetch papers from arXiv and extract text from the queried PDFs. In order to do one-shots, use
    the method `fetch_and_extract`.
    This function initializes the `fetcher` (in this case from arXiv) and the `text_extractor` classes.
    It fetches the papers from arXiv and stores them in a list of `ArxivPaper` pydantic models.
    For each paper, it downloads the PDF storing it in `data_folder` and extracts the text from it.
    The text is stored in the `text` attribute of each `ArxivPaper` object.
    For each paper, it deletes the references section if `delete_references` is set to True.

    Args:
        data_folder (str, optional): The folder where to store the PDFs. Defaults to "data".
        delete_references (bool, optional): If set to true, it deletes the References section. Defaults to True.
        logger (BoundLoggerLazyProxy, optional): The logger to log messages. Defaults to logger.
        category (str, optional): The arXiv category. Defaults to "cond-mat.str-el".
        max_results (int, optional): The maximum results for pagination for the arXiv API call. Defaults to 5.

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
        # Extracts text from the PDF
        text = text_extractor.from_pdf(pdf_path=pdf_path)
        if not text:
            logger.info("No text extracted from the PDF.")
            continue
        logger.info(f"Text extracted from {paper.id} and stored in model.")

        # Deleting references section
        if delete_references:
            text = text_extractor.delete_references(text=text)

        # Stores the text in the `text` attribute of the `ArxivPaper` object
        paper.text = text
        paper.length_text = len(text)
    return papers
