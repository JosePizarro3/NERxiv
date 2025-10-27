from pydantic import BaseModel, ConfigDict


class BaseSection(BaseModel):
    """Base class used as an abstraction layer including `model_config` and a `normalize()` method
    for all section classes defined in `nerxiv/datamodel/`."""

    model_config = ConfigDict(extra="forbid")

    def normalize(self) -> None:
        """
        Normalize the data model instance.

        This method must be overridden by subclasses to implement custom normalization logic.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.normalize() must be implemented."
        )
