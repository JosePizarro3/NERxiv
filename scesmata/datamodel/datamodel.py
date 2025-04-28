import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Author(BaseModel):
    name: str = Field(..., description="The name of the author.")
    affiliation: Optional[str] = Field("", description="The affiliation of the author.")
    email: Optional[str] = Field("", description="The email of the author.")


class ArxivPaper(BaseModel):
    id: str = Field(
        ...,
        description="The arXiv ID of the paper. Example: 2502.10245v1.",
    )
    url: str = Field(
        ...,
        description="The URL of the arXiv paper. Example: http://arxiv.org/abs/2502.10245v1",
    )
    pdf_url: str = Field(
        ...,
        description="The URL of the PDF of the arXiv paper. Example: http://arxiv.org/pdf/2502.10245v1",
    )
    updated: Optional[datetime.datetime] = Field(
        ..., description="The date when the paper was updated."
    )
    published: Optional[datetime.datetime] = Field(
        ..., description="The date when the paper was published."
    )

    title: str = Field(..., description="The title of the arXiv paper.")
    summary: str = Field(..., description="The summary of the arXiv paper.")
    authors: list[Author]

    comment: str = Field("", description="The comment of the arXiv paper.")
    n_pages: Optional[int] = Field(
        None, description="The number of pages of the arXiv paper."
    )
    n_figures: Optional[int] = Field(
        None, description="The number of figures of the arXiv paper."
    )
    categories: list[str] = Field(
        ...,
        description="The categories of the arXiv paper. Example: ['cond-mat.str-el', 'cond-mat.mtrl-sci'].",
    )

    text: Optional[str] = Field(
        "", description="The text extracted from the arXiv paper."
    )
    length_text: Optional[str] = Field(
        0, description="The length of the text extracted from the arXiv paper."
    )

    methods: Optional[list[str]] = Field(
        [], description="The methods used in the arXiv paper."
    )
