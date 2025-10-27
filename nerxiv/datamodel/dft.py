from pydantic import Field

from nerxiv.datamodel.base_section import BaseSection


class DFT(BaseSection):
    """
    Section representing the Density Functional Theory (DFT) parameters used in the simulation
    of a material. This includes information about the computational code, exchange-correlation
    functional, basis set, pseudopotentials, cutoffs, k-point sampling, relativistic treatment,
    and spin-orbit coupling. Intended to capture the setup of DFT calculations as reported in
    computational materials science papers.
    """

    code_name: str | None = Field(
        None,
        description="""
        Name of the DFT software/code used. For example, 'VASP', 'Quantum ESPRESSO', 'FP-LMTO'.
        """,
    )

    code_version: str | None = Field(
        None,
        description="""
        Version of the DFT code. For example, '6.7', '7.3.2'.
        """,
    )

    exchange_correlation_functional: str | None = Field(
        None,
        description="""
        Exchange-correlation functional used. For example, 'LDA', 'PBE'.
        """,
    )

    basis_set: str | None = Field(
        None,
        description="""
        Type of basis set used in the DFT calculation. For example, 'plane-wave', 'localized orbitals'.
        """,
    )

    pseudopotentials: str | None = Field(
        None,
        description="""
        Type of pseudopotentials used. For example, 'PAW', 'ultrasoft', 'norm-conserving'.
        """,
    )

    core_electrons_treatment: str | None = Field(
        None,
        description="""
        Treatment of core electrons. For example, 'frozen core', 'all-electron'.
        """,
    )

    wavefunction_cutoff: float | None = Field(
        None,
        description="""
        Plane-wave basis set cutoff for the wavefunctions. For example, 500.0.
        """,
    )

    wavefunction_cutoff_units: str | None = Field(
        None,
        description="""
        Units for the wavefunction cutoff. For example, 'eV', 'Ry'.
        """,
    )

    density_cutoff: float | None = Field(
        None,
        description="""
        Plane-wave basis set cutoff for the charge density. For example, 5000.0.
        """,
    )

    density_cutoff_units: str | None = Field(
        None,
        description="""
        Units for the density cutoff. For example, 'eV', 'Ry'.
        """,
    )

    energy_cutoff: float | None = Field(
        None,
        description="""
        Energy cutoff used in the DFT calculation. For example, 1e-5.
        """,
    )

    energy_cutoff_units: str | None = Field(
        None,
        description="""
        Units for the energy cutoff. For example, 'eV', 'Ha'.
        """,
    )

    energy_convergence_threshold: float | None = Field(
        None,
        description="""
        Threshold for energy convergence in the self-consistent field (SCF) cycle. For example, 1e-6.
        """,
    )

    energy_convergence_threshold_units: str | None = Field(
        None,
        description="""
        Units for the energy convergence threshold. For example, 'eV', 'Ha'.
        """,
    )

    k_mesh_type: str | None = Field(
        None,
        description="""
        Type of k-point mesh used for Brillouin zone sampling. For example, 'Monkhorst-Pack', 'Gamma-centered'.
        """,
    )

    k_mesh: list[int] | None = Field(
        None,
        description="""
        List of integers defining the k-point mesh grid. For example, [6, 6, 6].
        """,
    )

    wigner_seitz_radii: dict[str, float] | None = Field(
        None,
        description="""
        Dictionary mapping element symbols to their Wigner-Seitz radii in angstroms. For example, {'Si': 1.11, 'O': 0.66}.
        """,
    )

    soc: bool | None = Field(
        None,
        description="""
        Whether spin-orbit coupling (SOC) was included in the DFT calculation.
        """,
    )

    spin_treatment: str | None = Field(
        None,
        description="""
        Treatment of spin in the DFT calculation. For example, 'spin-polarized', 'non-spin-polarized'.
        """,
    )

    relativistic_treatment: str | None = Field(
        None,
        description="""
        Relativistic treatment used in the DFT calculation. For example, 'scalar relativistic', 'fully relativistic'.
        """,
    )

    def normalize(self) -> None:
        pass
