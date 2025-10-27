from pydantic import Field

from nerxiv.datamodel.base_section import BaseSection


class Interactions(BaseSection):
    """Section representing the Interaction parameters used in a simulation of a material."""

    def normalize(self) -> None:
        pass
