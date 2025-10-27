from pydantic import BaseModel, ConfigDict, Field


class BaseSection(BaseModel):
    """Base class used as an abstraction layer including `model_config` and a `normalize()` method
    for all section classes defined in `nerxiv/datamodel/`."""

    model_config = ConfigDict(extra="forbid")

    extra_metainfo: list[str] = Field(
        default_factory=list,
        description="""
        List of additional phrases or observations extracted from the paper that do not fit any
        structured field. Useful for later analysis or LLM post-processing. For example,
        ['calculation converged in 120 steps', 'used experimental lattice constants']
        """,
    )

    def normalize(self) -> None:
        """
        Normalize the data model instance.

        This method must be overridden by subclasses to implement custom normalization logic.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.normalize() must be implemented."
        )
