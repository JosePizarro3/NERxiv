from pydantic import Field

from nerxiv.datamodel.base_section import BaseSection


class DFT(BaseSection):
    """Section representing the Density Functional Theory (DFT) parameters used in a simulation of a material."""

    code_name: str | None = Field(
        None,
        description=""" """,
    )

    code_version: str = Field(
        None,
        description=""" """,
    )

    exchange_correlation_functional: str = Field(
        None,
        description=""" """,
    )

    basis_set: str | None = Field(
        None,
        description=""" """,
    )

    pseudopotentials: str | None = Field(
        None,
        description=""" """,
    )

    core_electrons_treatment: str | None = Field(
        None,
        description=""" """,
    )

    wavefunction_cutoff: float | None = Field(
        None,
        description=""" """,
    )

    wavefunction_cutoff_units: str | None = Field(
        None,
        description=""" """,
    )

    density_cutoff: float | None = Field(
        None,
        description=""" """,
    )

    density_cutoff_units: str | None = Field(
        None,
        description=""" """,
    )

    energy_cutoff: float | None = Field(
        None,
        description=""" """,
    )

    energy_cutoff_units: str | None = Field(
        None,
        description=""" """,
    )

    energy_convergence_threshold: float | None = Field(
        None,
        description=""" """,
    )

    energy_convergence_threshold_units: str | None = Field(
        None,
        description=""" """,
    )

    k_mesh_type: str | None = Field(
        None,
        description=""" """,
    )

    k_mesh: list[int] | None = Field(
        None,
        description=""" """,
    )

    wigner_seitz_radii: dict[str, float] | None = Field(
        None,
        description="""{"Nb": 1.27, "Sr": 2.138, "O": 0.82}""",
    )

    soc: bool | None = Field(
        None,
        description=""" """,
    )

    spin_treatment: str | None = Field(
        None,
        description=""" """,
    )

    relativistic_treatment: str | None = Field(
        None,
        description=""" """,
    )

    def normalize(self) -> None:
        pass
