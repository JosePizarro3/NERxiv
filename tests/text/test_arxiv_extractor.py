import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ragxiv.datamodel import ArxivPaper
from ragxiv.text import TextExtractor, arxiv_fetch_and_extract
from tests.conftest import generate_arxiv_fetcher, generate_arxiv_paper


def clean_fetched_ids_file():
    """Deletes the `fetched_arxiv_ids.txt` file if it exists. This is applied multiple times in test functions to ensure a clean state."""
    path = Path("tests/data/fetched_arxiv_ids.txt")
    if path.exists():
        path.unlink(missing_ok=True)


class TestArxivFetcher:
    @pytest.mark.parametrize(
        "fetched_ids, result",
        [
            # No file content
            (
                [],
                set(),
            ),
            # One fetched ID present
            (
                ["1234.5678v1"],
                set(["1234.5678v1"]),
            ),
            # Multiple fetched IDs present
            (
                ["1234.5678v1", "2345.6789v1"],
                set(["1234.5678v1", "2345.6789v1"]),
            ),
        ],
    )
    def test_fetched_ids(self, fetched_ids: list, result: set):
        """Tests the `fetched_ids` property of the `ArxivFetcher` class."""
        clean_fetched_ids_file()  # deletes `fetched_arxiv_ids.txt` file if it exists

        with open("tests/data/fetched_arxiv_ids.txt", "a") as f:
            for id in fetched_ids:
                f.write(f"{id}\n")

        arxiv_fetcher = generate_arxiv_fetcher()
        assert arxiv_fetcher.fetched_ids == result

        clean_fetched_ids_file()  # deletes `fetched_arxiv_ids.txt` file if it exists

    @pytest.mark.parametrize(
        "arxiv_response, log_msg, result",
        [
            # Empty response
            (
                """
                <feed xmlns="http://www.w3.org/2005/Atom"></feed>
                """,
                {"level": "info", "event": "No papers found in the response"},
                {},
            ),
            # Error in title when fetching
            (
                """
                <feed xmlns="http://www.w3.org/2005/Atom">
                    <entry>
                        <title>Error when fetching the paper</title>
                    </entry>
                </feed>
                """,
                {"level": "error", "event": "Error fetching the paper"},
                {},
            ),
            # Id not in the correct format
            (
                """
                <feed xmlns="http://www.w3.org/2005/Atom">
                    <entry>
                        <title>Test Paper Title</title>
                        <id>not a proper arxiv id</id>
                    </entry>
                </feed>
                """,
                {
                    "level": "error",
                    "event": "Paper without a valid URL id: not a proper arxiv id",
                },
                {},
            ),
            # Missing summary
            (
                """
                <feed xmlns="http://www.w3.org/2005/Atom">
                    <entry>
                        <title>Test Paper Title</title>
                        <id>http://arxiv.org/abs/1234.5678v1</id>
                    </entry>
                </feed>
                """,
                {
                    "level": "error",
                    "event": "Paper http://arxiv.org/abs/1234.5678v1 without summary/abstract",
                },
                {},
            ),
            # Missing authors
            (
                """
                <feed xmlns="http://www.w3.org/2005/Atom">
                    <entry>
                        <title>Test Paper Title</title>
                        <id>http://arxiv.org/abs/1234.5678v1</id>
                        <summary>This is a test abstract.</summary>
                    </entry>
                </feed>
                """,
                {},
                {
                    "id": "1234.5678v1",
                    "url": "http://arxiv.org/abs/1234.5678v1",
                    "pdf_url": "http://arxiv.org/pdf/1234.5678v1",
                    "updated": None,
                    "published": None,
                    "title": "Test Paper Title",
                    "summary": "This is a test abstract.",
                    "authors": [],
                    "comment": "",
                    "n_pages": None,
                    "n_figures": None,
                    "categories": [],
                    "text": "",
                },
            ),
            # Successful response
            (
                """
                <feed xmlns="http://www.w3.org/2005/Atom">
                    <entry>
                        <id>http://arxiv.org/abs/1234.5678v1</id>
                        <updated>2024-04-25T00:00:00Z</updated>
                        <published>2024-04-24T00:00:00Z</published>
                        <title>Test Paper Title</title>
                        <summary>This is a test abstract.</summary>
                        <author>
                            <name>John Doe</name>
                            <affiliation>University of Test</affiliation>
                        </author>
                        <category term="cond-mat.str-el"/>
                        <arxiv:comment xmlns:arxiv="http://arxiv.org/schemas/atom">10 pages, 2 figures</arxiv:comment>
                    </entry>
                </feed>
                """,
                {},
                {
                    "id": "1234.5678v1",
                    "url": "http://arxiv.org/abs/1234.5678v1",
                    "pdf_url": "http://arxiv.org/pdf/1234.5678v1",
                    "updated": datetime.datetime(
                        2024, 4, 25, 0, 0, tzinfo=datetime.timezone.utc
                    ),
                    "published": datetime.datetime(
                        2024, 4, 24, 0, 0, tzinfo=datetime.timezone.utc
                    ),
                    "title": "Test Paper Title",
                    "summary": "This is a test abstract.",
                    "authors": [
                        {
                            "name": "John Doe",
                            "affiliation": "University of Test",
                            "email": None,
                        }
                    ],
                    "comment": "10 pages, 2 figures",
                    "n_pages": 10,
                    "n_figures": 2,
                    "categories": ["cond-mat.str-el"],
                    "text": "",
                },
            ),
        ],
    )
    @patch("urllib.request.urlopen")
    def test_fetch(
        self,
        mock_urlopen: MagicMock,
        cleared_log_storage: list,
        arxiv_response: str,
        log_msg: dict,
        result: dict,
    ):
        """Tests the `fetch` method of the `ArxivFetcher` class."""
        clean_fetched_ids_file()  # deletes `fetched_arxiv_ids.txt` file if it exists

        mock_response = MagicMock()
        mock_response.read.return_value = arxiv_response.encode("utf-8")
        mock_urlopen.return_value = mock_response

        fetcher = generate_arxiv_fetcher()
        papers = fetcher.fetch()
        if log_msg:
            assert len(cleared_log_storage) == 1
            assert cleared_log_storage[0]["level"] == log_msg["level"]
            assert cleared_log_storage[0]["event"] == log_msg["event"]
        if papers and all(isinstance(p, ArxivPaper) for p in papers):
            assert papers[0].model_dump() == result

        clean_fetched_ids_file()  # deletes `fetched_arxiv_ids.txt` file if it exists

    @pytest.mark.parametrize(
        "id, pdf_path_result, log_msg",
        [
            (
                "1234.5678v1",
                None,
                {
                    "level": "error",
                    "event": "Failed to download PDF: 404 Client Error: Not Found for url: http://arxiv.org/pdf/1234.5678v1",
                },
            ),
            (
                "2502.10309v1",
                Path("tests/data/2502.10309v1.pdf"),
                {
                    "level": "info",
                    "event": "PDF downloaded: tests/data/2502.10309v1.pdf",
                },
            ),
        ],
    )
    def test_download_pdf(
        self, cleared_log_storage: list, id: str, pdf_path_result: str, log_msg: str
    ):
        """Tests the `download_pdf` method of the `ArxivFetcher` class."""
        clean_fetched_ids_file()  # deletes `fetched_arxiv_ids.txt` file if it exists

        arxiv_paper = generate_arxiv_paper(id=id)
        arxiv_fetcher = generate_arxiv_fetcher()
        pdf_path = arxiv_fetcher.download_pdf(
            arxiv_paper,
            write=False,  # no writing to file
        )
        assert pdf_path == pdf_path_result
        assert len(cleared_log_storage) == 1
        assert cleared_log_storage[0]["level"] == log_msg["level"]
        assert cleared_log_storage[0]["event"] == log_msg["event"]

        clean_fetched_ids_file()  # deletes `fetched_arxiv_ids.txt` file if it exists


class TestTextExtractor:
    @pytest.mark.parametrize(
        "pdf_path, result",
        [
            # no path
            ("", False),
            # no valid path
            ("sample", False),
            # no pdf file
            ("tests/data/no_pdf_paper.txt", False),
            # successful
            ("tests/data/sample.pdf", True),
        ],
    )
    def test__check_pdf_path(self, pdf_path: str, result: bool):
        """Tests the `_check_pdf_path` method of the `TextExtractor` class."""
        check = TextExtractor()._check_pdf_path(pdf_path=pdf_path)
        assert check == result

    @pytest.mark.parametrize(
        "pdf_path, loader, length_text",
        [
            # no path
            ("", "pypdf", 0),
            # no valid path
            ("sample", "pypdf", 0),
            # no pdf file
            ("tests/data/no_pdf_paper.txt", "pypdf", 0),
            # no pdf loader implemented
            ("tests/data/sample.pdf", "no-pdf-loader-implemented", 0),
            # successful with pypdf
            ("tests/data/sample.pdf", "pypdf", 2876),
            # successful with pdfminer
            ("tests/data/sample.pdf", "pdfminer", 3118),
        ],
    )
    def test_get_text(self, pdf_path: str, loader: str, length_text: int):
        """Tests the `with_pdfminer` method of the `TextExtractor` class."""
        text = TextExtractor().get_text(pdf_path=pdf_path, loader=loader)
        assert len(text) == length_text

    @pytest.mark.parametrize(
        "input_text, expected_output",
        [
            # No references section
            (
                "This is some main body text.\nIntroduction\nMethods\nResults\nConclusion\n",
                "This is some main body text.\nIntroduction\nMethods\nResults\nConclusion\n",
            ),
            # References found, no Supplemental Material
            (
                "Main text content.\nReferences\n[1] A. Author, Title, 2024.\n[2] B. Author, Title, 2023.\n",
                "Main text content.",
            ),
            # Bibliography found
            ("Main body.\nBibliography\n[1] C. Author, 2022.\n", "Main body."),
            # References and Supplemental Material delimiters found
            (
                "Body text.\nReferences\n[1] Author 1\n[2] Author 2\nSupplemental Material:\nAdditional stuff.\n",
                "Body text.\nSupplemental Material:\nAdditional stuff.\n",
            ),
            # No references section but has [1] directly
            (
                "Introduction\n[1] Z. Researcher\n[2] X. Scientist\nConclusion\n",
                "Introduction",
            ),
        ],
    )
    def test_delete_references(self, input_text: str, expected_output: str):
        """Tests the `delete_references` method of the `TextExtractor` class."""
        output = TextExtractor().delete_references(text=input_text)
        assert output == expected_output

    def test_clean_text(self):
        """Tests the `clean_text` method of the `TextExtractor` class."""
        extractor = TextExtractor()
        old_text = extractor.get_text(pdf_path="tests/data/sample.pdf")
        text = extractor.clean_text(text=old_text)
        assert (
            old_text[:100]
            == "Sample PDF\nThis is a simple PDF ﬁle. Fun fun fun.\n\nLorem ipsum dolor  sit amet,  consectetuer  adipi"
        )
        assert (
            text[:100]
            == "Sample PDF This is a simple PDF ﬁle. Fun fun fun. Lorem ipsum dolor sit amet, consectetuer adipiscin"
        )


@pytest.mark.parametrize(
    "arxiv_response, log_msg, result",
    [
        # Empty response
        (
            """
            <feed xmlns="http://www.w3.org/2005/Atom"></feed>
            """,
            {"level": "info", "event": "No papers found in the response"},
            {},
        ),
        # Error in title when fetching
        (
            """
            <feed xmlns="http://www.w3.org/2005/Atom">
                <entry>
                    <title>Error when fetching the paper</title>
                </entry>
            </feed>
            """,
            {"level": "error", "event": "Error fetching the paper"},
            {},
        ),
        # Id not in the correct format
        (
            """
            <feed xmlns="http://www.w3.org/2005/Atom">
                <entry>
                    <title>Test Paper Title</title>
                    <id>not a proper arxiv id</id>
                </entry>
            </feed>
            """,
            {
                "level": "error",
                "event": "Paper without a valid URL id: not a proper arxiv id",
            },
            {},
        ),
        # Missing summary
        (
            """
            <feed xmlns="http://www.w3.org/2005/Atom">
                <entry>
                    <title>Test Paper Title</title>
                    <id>http://arxiv.org/abs/1234.5678v1</id>
                </entry>
            </feed>
            """,
            {
                "level": "error",
                "event": "Paper http://arxiv.org/abs/1234.5678v1 without summary/abstract",
            },
            {},
        ),
        # Missing authors
        (
            """
            <feed xmlns="http://www.w3.org/2005/Atom">
                <entry>
                    <title>Test Paper Title</title>
                    <id>http://arxiv.org/abs/1234.5678v1</id>
                    <summary>This is a test abstract.</summary>
                </entry>
            </feed>
            """,
            {},
            {
                "id": "1234.5678v1",
                "url": "http://arxiv.org/abs/1234.5678v1",
                "pdf_url": "http://arxiv.org/pdf/1234.5678v1",
                "updated": None,
                "published": None,
                "title": "Test Paper Title",
                "summary": "This is a test abstract.",
                "authors": [],
                "comment": "",
                "n_pages": None,
                "n_figures": None,
                "categories": [],
                "text": "",
            },
        ),
        # Successful response
        (
            """
            <feed xmlns="http://www.w3.org/2005/Atom">
                <entry>
                    <id>http://arxiv.org/abs/1234.5678v1</id>
                    <updated>2024-04-25T00:00:00Z</updated>
                    <published>2024-04-24T00:00:00Z</published>
                    <title>Test Paper Title</title>
                    <summary>This is a test abstract.</summary>
                    <author>
                        <name>John Doe</name>
                        <affiliation>University of Test</affiliation>
                    </author>
                    <category term="cond-mat.str-el"/>
                    <arxiv:comment xmlns:arxiv="http://arxiv.org/schemas/atom">10 pages, 2 figures</arxiv:comment>
                </entry>
            </feed>
            """,
            {},
            {
                "id": "1234.5678v1",
                "url": "http://arxiv.org/abs/1234.5678v1",
                "pdf_url": "http://arxiv.org/pdf/1234.5678v1",
                "updated": datetime.datetime(
                    2024, 4, 25, 0, 0, tzinfo=datetime.timezone.utc
                ),
                "published": datetime.datetime(
                    2024, 4, 24, 0, 0, tzinfo=datetime.timezone.utc
                ),
                "title": "Test Paper Title",
                "summary": "This is a test abstract.",
                "authors": [
                    {
                        "name": "John Doe",
                        "affiliation": "University of Test",
                        "email": None,
                    }
                ],
                "comment": "10 pages, 2 figures",
                "n_pages": 10,
                "n_figures": 2,
                "categories": ["cond-mat.str-el"],
                "text": "",
            },
        ),
    ],
)
@patch("urllib.request.urlopen")
def test_arxiv_fetch_and_extract(
    mock_urlopen: MagicMock,
    cleared_log_storage: list,
    arxiv_response: str,
    log_msg: dict,
    result: dict,
):
    """Tests the `arxiv_fetch_and_extract` method of the `ArxivFetcher` class."""
    clean_fetched_ids_file()  # deletes `fetched_arxiv_ids.txt` file if it exists

    mock_response = MagicMock()
    mock_response.read.return_value = arxiv_response.encode("utf-8")
    mock_urlopen.return_value = mock_response

    papers = arxiv_fetch_and_extract(data_folder="tests/data", max_results=1)
    if log_msg:
        assert len(cleared_log_storage) == 2
        assert cleared_log_storage[0]["level"] == log_msg["level"]
        assert cleared_log_storage[0]["event"] == log_msg["event"]
        assert cleared_log_storage[1]["level"] == "info"
        assert (
            cleared_log_storage[1]["event"]
            == "1 papers fetched from arXiv, cond-mat.str-el."
        )
    if papers:
        assert papers[0].model_dump() == result

    clean_fetched_ids_file()  # deletes `fetched_arxiv_ids.txt` file if it exists
