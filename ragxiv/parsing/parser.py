from abc import ABC, abstractmethod

from bam_masterdata.metadata.entities import CollectionType
from bam_masterdata.openbis.login import ologin


class BaseParser(ABC):
    """
    Abstract base class for parsers.
    """

    @abstractmethod
    def parse(self, files: list[str], collection: CollectionType) -> None:
        """
        Parse the given files and store the results in the collection.

        Args:
            files (list[str]): List of file paths to parse.
            collection: The collection where parsed data will be stored.
        """
        pass


class DMFTArxivParser(BaseParser):
    def parse(self, files: list[str], collection) -> None:
        pass
