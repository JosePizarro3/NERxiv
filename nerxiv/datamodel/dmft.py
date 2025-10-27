from pydantic import Field

from nerxiv.datamodel.base_section import BaseSection


class DMFT(BaseSection):
    """Section representing the Dynamical Mean-Field Theory (DMFT) parameters used in a simulation of a material."""

    def normalize(self) -> None:
        pass
