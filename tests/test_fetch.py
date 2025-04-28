import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import generate_arxiv_fetcher, generate_arxiv_paper


class TestArxivFetcher:
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
                {"level": "error", "event": "Paper without summary/abstract"},
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
                    "length_text": 0,
                    "methods": [],
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
                            "email": "",
                        }
                    ],
                    "comment": "10 pages, 2 figures",
                    "n_pages": 10,
                    "n_figures": 2,
                    "categories": ["cond-mat.str-el"],
                    "text": "",
                    "length_text": 0,
                    "methods": [],
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
        mock_response = MagicMock()
        mock_response.read.return_value = arxiv_response.encode("utf-8")
        mock_urlopen.return_value = mock_response

        arxiv_fetcher = generate_arxiv_fetcher()
        papers = arxiv_fetcher.fetch(max_results=1)
        if log_msg:
            assert len(cleared_log_storage) == 1
            assert cleared_log_storage[0]["level"] == log_msg["level"]
            assert cleared_log_storage[0]["event"] == log_msg["event"]
        if papers:
            assert papers[0].model_dump() == result

    @pytest.mark.parametrize(
        "comment, result",
        [
            # Empty response
            ("no matching pages nor figures", (None, None)),
            ("20 pages, no matching figures", (None, None)),
            (
                "Comment on arXiv:2401.10650 (Phys. Rev. Lett. 133, 136501 (2024); DOI\n  10.1103/PhysRevLett.133.136501). 2 pages, 1 figure",
                (2, 1),
            ),
        ],
    )
    def test_get_pages_and_figures(
        self, comment: str, result: tuple[Optional[int], Optional[int]]
    ):
        """Tests the `get_pages_and_figures` method of the `ArxivFetcher` class."""
        arxiv_fetcher = generate_arxiv_fetcher()
        assert arxiv_fetcher.get_pages_and_figures(comment) == result

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
                Path("data/2502.10309v1.pdf"),
                {
                    "level": "info",
                    "event": "PDF downloaded: data/2502.10309v1.pdf",
                },
            ),
        ],
    )
    def test_download_pdf(
        self, cleared_log_storage: list, id: str, pdf_path_result: str, log_msg: str
    ):
        """Tests the `download_pdf` method of the `ArxivFetcher` class."""
        arxiv_paper = generate_arxiv_paper(id=id)
        arxiv_fetcher = generate_arxiv_fetcher()
        pdf_path = arxiv_fetcher.download_pdf(
            arxiv_paper, write=False
        )  # not writing to file
        assert pdf_path == pdf_path_result
        assert len(cleared_log_storage) == 1
        assert cleared_log_storage[0]["level"] == log_msg["level"]
        assert cleared_log_storage[0]["event"] == log_msg["event"]

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
        """Tests the `delete_references` method of the `ArxivFetcher` class."""
        arxiv_fetcher = generate_arxiv_fetcher()
        output = arxiv_fetcher.delete_references(text=input_text)
        assert output == expected_output

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
                {"level": "error", "event": "Paper without summary/abstract"},
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
                    "length_text": 0,
                    "methods": [],
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
                            "email": "",
                        }
                    ],
                    "comment": "10 pages, 2 figures",
                    "n_pages": 10,
                    "n_figures": 2,
                    "categories": ["cond-mat.str-el"],
                    "text": "",
                    "length_text": 0,
                    "methods": [],
                },
            ),
        ],
    )
    @patch("urllib.request.urlopen")
    def test_fetch_and_extract(
        self,
        mock_urlopen: MagicMock,
        cleared_log_storage: list,
        arxiv_response: str,
        log_msg: dict,
        result: dict,
    ):
        """Tests the `fetch_and_extract` method of the `ArxivFetcher` class."""
        mock_response = MagicMock()
        mock_response.read.return_value = arxiv_response.encode("utf-8")
        mock_urlopen.return_value = mock_response

        arxiv_fetcher = generate_arxiv_fetcher()
        papers = arxiv_fetcher.fetch_and_extract(max_results=1, delete_references=True)
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

    @pytest.mark.parametrize(
        "id, length_text",
        [
            ("1234.5678v1", 0),
            ("2502.10309v1", 29542),
        ],
    )
    def test_extract_text_from_pdf(self, id: str, length_text: str):
        """Tests the `extract_text_from_pdf` method of the `ArxivFetcher` class."""
        arxiv_paper = generate_arxiv_paper(id=id)
        arxiv_fetcher = generate_arxiv_fetcher()
        text = arxiv_fetcher.extract_text_from_pdf(
            arxiv_paper, data_folder="tests/data"
        )
        assert len(text) == length_text

    @pytest.mark.parametrize(
        "id, length_text",
        [
            ("1234.5678v1", 0),
            ("2502.10309v1", 29542),
        ],
    )
    def test_extract_text(self, id: str, length_text: str):
        """Tests the `extract_text` method of the `ArxivFetcher` class."""
        arxiv_paper = generate_arxiv_paper(id=id)
        arxiv_fetcher = generate_arxiv_fetcher()
        text = arxiv_fetcher.extract_text(arxiv_paper)
        assert len(text) == length_text
