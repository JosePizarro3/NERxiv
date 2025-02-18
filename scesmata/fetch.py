import io
import re
import urllib.request
from pathlib import Path

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

    def __init__(self, **kwargs):
        self.logger = kwargs.get("logger", logger)
        self.session = requests.Session()  # Reuse TCP connection
        # ! an initial short paper is used to warm up the `requests` session connection
        # ! otherwise, long papers get stuck on `requests.get()` due to connection timeouts
        self.session.head("http://arxiv.org/pdf/2502.10309v1", timeout=30)

    def get_pages_and_figures(self, comment: str) -> tuple:
        """
        Extract the number of pages and figures from the comment of the arXiv paper.

        Args:
            comment (str): A string containing the comment of the arXiv paper.

        Returns:
            tuple: A tuple containing the number of pages and figures.
        """
        pattern = r"(\d+) *pages, *(\d+) *figures"
        match = re.search(pattern, comment)
        if match:
            n_pages, n_figures = match.groups()
            return int(n_pages), int(n_figures)
        return None, None

    def fetch(
        self,
        category: str = "cond-mat.str-el",
        max_results: int = 5,
    ) -> list:
        """
        Fetch papers from arXiv and stores them in a list of ArxivPaper pydantic models.

        Args:
            category (str, optional): The category in arXiv to fetch the papers from. Defaults to "cond-mat.str-el".
            max_results (int, optional): Pagination for maximum number of papers fetched. Defaults to 5.

        Returns:
            list: A list of ArxivPaper objects.
        """
        # Fetch request from arXiv API and parsing the XML response
        url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
        request = urllib.request.urlopen(url)
        data = request.read().decode("utf-8")
        data_dict = xmltodict.parse(data)

        # Extracting papers from the XML response
        papers = data_dict.get("feed", {}).get("entry", [])
        if not papers:
            self.logger.info("No papers found.")
            return []
        # In case `max_results` is 1, the response is not a list
        if not isinstance(papers, list):
            papers = [papers]

        # Store papers object ArxivPaper in a list
        arxiv_papers = []
        for paper in papers:
            # If there is an error in the fetching, skip the paper
            if "Error" in paper.get("title", ""):
                self.logger.error(f"Error fetching paper: {paper}.")
                continue

            # If there is no `id`, skip the paper
            url_id = paper.get("id")
            if not url_id:
                self.logger.error(f"Paper without valid URL id: {paper}.")
                continue

            # If there is no `summary`, skip the paper
            if not paper.get("summary"):
                self.logger.error(f"Paper without summary/abstract: {paper}.")
                continue

            # Extracting `authors` and `categories` from the XML response
            authors = [
                Author(name=author.get("name"), affiliation=author.get("affiliation"))
                for author in paper.get("author", [])
            ]
            arxiv_categories = paper.get("category", [])
            if not isinstance(arxiv_categories, list):
                categories = [arxiv_categories.get("@term")]
            else:
                categories = [category.get("@term") for category in arxiv_categories]

            # Extracting pages and figures from the comment
            comment = paper.get("arxiv:comment", {}).get("#text", "")
            n_pages, n_figures = self.get_pages_and_figures(comment)

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

    def download_pdf(self, arxiv_paper: ArxivPaper) -> None:
        """
        Download the PDF of the arXiv paper and stores it in the `data` folder using the `arxiv_paper.id` to name the PDF file.

        Args:
            arxiv_paper (ArxivPaper): The arXiv paper object to be queried and stored.
        """
        try:
            response = self.session.get(arxiv_paper.pdf_url, stream=True, timeout=60)
            response.raise_for_status()

            pdf_path = f"data/{arxiv_paper.id}.pdf"

            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            self.logger.info(f"PDF downloaded: {pdf_path}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download PDF: {e}")

    def extract_text_from_pdf(self, arxiv_paper: ArxivPaper) -> str:
        """
        Extract text from the PDF of the arXiv paper.

        Args:
            arxiv_paper (ArxivPaper): The arXiv paper object to extract the text from.

        Returns:
            str: The text extracted from the PDF.
        """
        pdf_path = Path(f"data/{arxiv_paper.id}.pdf")
        if not pdf_path.exists():
            self.download_pdf(arxiv_paper=arxiv_paper)
        text = ""
        with open(f"data/{arxiv_paper.id}.pdf", "rb") as f:
            reader = PdfReader(f)
            text = "\n".join(
                [page.extract_text() for page in reader.pages if page.extract_text()]
            )
        return text

    def extract_text(self, arxiv_paper: ArxivPaper) -> str:
        """
        Extract text from the arXiv paper and reads the information from the PDF.

        Note:
            # ! This method does not work if the `arxiv_paper` is large. In that case, downloading
            # ! the PDF and extracting the text from the PDF is faster and can handle larger pdf files.
            # ! In `fetch_and_extract`, we use the `extract_text_from_pdf` method.

        Args:
            arxiv_paper (ArxivPaper): The arXiv paper object to extract the text from.

        Returns:
            str: The text extracted from the arXiv paper.
        """
        text = ""
        try:
            response = self.session.get(arxiv_paper.pdf_url, stream=True, timeout=60)
            response.raise_for_status()

            # Use a temporary in-memory file
            pdf_bytes = io.BytesIO()
            for chunk in response.iter_content(chunk_size=4096):  # Download in chunks
                pdf_bytes.write(chunk)

            pdf_bytes.seek(0)  # Reset file pointer to beginning
            reader = PdfReader(pdf_bytes)

            text = "\n".join(
                [page.extract_text() for page in reader.pages if page.extract_text()]
            )
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download PDF: {e}")
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

    def fetch_and_extract(
        self,
        category: str = "cond-mat.str-el",
        max_results: int = 5,
        delete_references: bool = True,
    ) -> list:
        """
        Fetch papers from arXiv, extract text from the PDF and store the text in the ArxivPaper object.

        Args:
            category (str, optional): The category in arXiv to fetch the papers from. Defaults to "cond-mat.str-el".
            max_results (int, optional): Pagination for maximum number of papers fetched. Defaults to 5.

        Returns:
            list: A list of ArxivPaper objects with the extracted text from their PDFs.
        """
        papers = self.fetch(category=category, max_results=max_results)
        self.logger.info(f"{max_results} papers fetched from arXiv, {category}.")
        for paper in papers:
            text = self.extract_text_from_pdf(arxiv_paper=paper)
            self.logger.info(f"Text extracted from {paper.id} and stored in model.")
            if delete_references:
                text = self.delete_references(text)
            paper.text = text
            paper.length_text = len(text)
        return papers
