from pydantic import Field

from nerxiv.datamodel.base_section import BaseSection


class DFT(BaseSection):
    """Section representing the Density Functional Theory (DFT) parameters used in a simulation of a material."""

    def normalize(self) -> None:
        pass
