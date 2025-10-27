from pydantic import Field
from pymatgen.core import Composition

from nerxiv.datamodel.base_section import BaseSection


class ChemicalFormulation(BaseSection):
    """
    A ChemicalFormulation is a descriptive representation of the chemical composition of a material
    system, expressed in one or more standardized formula formats (e.g., IUPAC, anonymous, Hill, or
    reduced), each encoding the stoichiometry and elemental ordering according to specific conventions.

    For the compound H2O2 (hydrogen peroxide), the different formulations would be:

        iupac: H2O2

        anonymous: AB

        hill: H2O2

        reduced: H2O2
    """

    iupac: str | None = Field(
        None,
        description="""
        Chemical formula where the elements are ordered using a formal list based on
        electronegativity as defined in the IUPAC nomenclature of inorganic chemistry (2005):

            - https://en.wikipedia.org/wiki/List_of_inorganic_compounds

        Contains reduced integer chemical proportion numbers where the proportion number
        is omitted if it is 1.
        """,
    )

    anonymous: str | None = Field(
        None,
        description="""
        Formula with the elements ordered by their reduced integer chemical proportion
        number, and the chemical species replaced by alphabetically ordered letters. The
        proportion number is omitted if it is 1.

        Examples: H2O becomes A2B and H2O2 becomes AB. The letters are drawn from the English
        alphabet that may be extended by increasing the number of letters: A, B, ..., Z, Aa, Ab
        and so on. This definition is in line with the similarly named OPTIMADE definition.
        """,
    )

    hill: str | None = Field(
        None,
        description="""
        Chemical formula where Carbon is placed first, then Hydrogen, and then all the other
        elements in alphabetical order. If Carbon is not present, the order is alphabetical.
        """,
    )

    reduced: str | None = Field(
        None,
        description="""
        Alphabetically sorted chemical formula with reduced integer chemical proportion
        numbers. The proportion number is omitted if it is 1.
        """,
    )

    def set_formulas(self, composition: Composition) -> None:
        """
        Extracts the chemical formulas from a pymatgen Composition object and sets the
        corresponding fields in the ChemicalFormulation instance.

        Args:
            composition (Composition): A pymatgen Composition object containing the chemical
            composition to extract formulas from.
        """
        self.iupac = composition.iupac_formula
        self.anonymous = composition.anonymized_formula
        self.hill = composition.hill_formula
        self.reduced = composition.reduced_formula

    def normalize(self) -> None:
        pass


class CrystalStructure(BaseSection):
    """Section representing the crystal structure information of a material."""

    def normalize(self) -> None:
        pass
