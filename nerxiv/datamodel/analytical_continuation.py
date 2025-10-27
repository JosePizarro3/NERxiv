from pydantic import Field

from nerxiv.datamodel.base_section import BaseSection


class AnalyticalContinuation(BaseSection):
    """Section representing the Analytical Continuation parameters used in a simulation of a material."""

    def normalize(self) -> None:
        pass
