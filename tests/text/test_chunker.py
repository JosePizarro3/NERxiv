import pytest

from ragxiv.text import Chunker


class TestChunker:
    def test_chunker_raises_without_text(self):
        """Tests that the `Chunker` raises a ValueError when initialized without text."""
        with pytest.raises(ValueError, match="Text is required for chunking."):
            Chunker()

    @pytest.mark.parametrize(
        "text, chunk_size, chunk_overlap, result",
        [
            (
                "We perform first-principles calculations using Density Functional Theory to investigate "
                "the electronic structure of the layered compound. The exchange-correlation functional is "
                "treated within the Generalized Gradient Approximation.",
                50,
                1,
                [
                    "We perform first-principles calculations using",
                    "Density Functional Theory to investigate the",
                    "electronic structure of the layered compound. The",
                    "exchange-correlation functional is treated within",
                    "the Generalized Gradient Approximation.",
                ],
            ),
            (
                "We perform first-principles calculations using Density Functional Theory to investigate "
                "the electronic structure of the layered compound. The exchange-correlation functional is "
                "treated within the Generalized Gradient Approximation.",
                100,
                1,
                [
                    "We perform first-principles calculations using Density Functional Theory to investigate the",
                    "electronic structure of the layered compound. The exchange-correlation functional is treated within",
                    "the Generalized Gradient Approximation.",
                ],
            ),
            (
                "We perform first-principles calculations using Density Functional Theory to investigate "
                "the electronic structure of the layered compound. The exchange-correlation functional is "
                "treated within the Generalized Gradient Approximation.",
                50,
                10,
                [
                    "We perform first-principles calculations using",
                    "using Density Functional Theory to investigate",
                    "the electronic structure of the layered compound.",
                    "compound. The exchange-correlation functional is",
                    "is treated within the Generalized Gradient",
                    "Gradient Approximation.",
                ],
            ),
        ],
    )
    def test_chunk_text(
        self, text: str, chunk_size: int, chunk_overlap: int, result: list[str] | None
    ):
        """Tests the `chunk_text` method of the `Chunker` class."""
        chunks = Chunker(text=text).chunk_text(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        assert len(chunks) == len(result)
        for i, chunk in enumerate(chunks):
            assert chunk.page_content == result[i]
