from pydantic import BaseModel, Field


class Method(BaseModel):
    """The model for the mathematical method being used."""

    name: str | None = Field(
        description="The name of the method. It is a verbose name, e.g., 'Density Functional "
        "Theory' or 'Quantum Monte Carlo'. The acronym of the method is not included in the name and "
        "is stored in another field called `acronym`.",
    )

    acronym: str | None = Field(
        None,
        description="The acronym of the method. It is a short name, e.g., 'DFT' or 'QMC'. The verbose "
        "name of the method is not included in the acronym and is stored in another field called `name`.",
    )


class Simulation(BaseModel):
    """The extracted metainformation about a simulation."""

    methods: list[Method]
