from pydantic import Field

from nerxiv.datamodel.base_section import BaseSection


class Projection(BaseSection):
    """Section representing the Projection parameters used in a simulation of a material."""

    def normalize(self) -> None:
        pass
