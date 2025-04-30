import datetime
from typing import Optional

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class Author(BaseModel):
    name: str = Field(..., description="The name of the author.")
    affiliation: Optional[str] = Field("", description="The affiliation of the author.")
    email: Optional[str] = Field("", description="The email of the author.")


class ArxivPaper(BaseModel):
    """The data model for the arXiv paper metadata."""

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

    pages: list[Document]


class Method(BaseModel):
    """The model for the mathematical method being used."""

    name: Optional[str] = Field(
        default=None,
        description="The name of the method. It is a verbose name, e.g., 'Density Functional "
        "Theory' or 'Quantum Monte Carlo'. The acronym of the method is not included in the name and "
        "is stored in another field called `acronym`.",
    )

    acronym: Optional[str] = Field(
        default=None,
        description="The acronym of the method. It is a short name, e.g., 'DFT' or 'QMC'. The verbose "
        "name of the method is not included in the acronym and is stored in another field called `name`.",
    )


class Simulation(BaseModel):
    """The extracted metainformation about a simulation."""

    methods: list[Method]
