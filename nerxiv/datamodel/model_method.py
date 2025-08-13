from typing import Literal

from pydantic import BaseModel, Field

DFT_CODES_MAP = {
    "FPLO": {
        "basis_set": "(L)APW+lo",
        "xc_functionals": ["LDA, GGA"],
        "soc": True,
    },
    "VASP": {
        "basis_set": "plane waves",
        "xc_functionals": ["LDA", "GGA", "metaGGA", "hyperGGA", "hybrid"],
        "soc": True,
    },
    "WIEN2K": {
        "basis_set": "(L)APW+lo",
        "xc_functionals": ["LDA", "GGA", "metaGGA", "hyperGGA", "hybrid"],
        "soc": True,
    },
}


class Program(BaseModel):
    """
    A Program is a software application or tool used in the context of materials science,
    computational chemistry, or related fields. Its metadata includes the name as an acronym and potentially the version used.

    For example, for a simulation using the Vienna Ab initio Simulation Package (VASP) with
    version 5.4.4, the metadata would be:

        name: VASP

        version: 5.4.4
    """

    program_name: str = Field(
        ...,
        description="The name of the program or software application.",
    )

    program_version: str | None = Field(
        None,
        description="The version of the program or software application.",
    )


class KPoints(BaseModel):
    """
    A KPoints is a representation of the k-point mesh used in electronic structure calculations.
    It is characterized by its k-point mesh, which is a list of integers representing the number
    of k-points in each direction of the reciprocal space.
    """

    k_mesh: list[int] = Field(
        [],
        description="""
        The k-point mesh used in the calculation, represented as a list of integers.
        For example, a k-point mesh of 4x4x4 would be represented as [4, 4, 4].
        """,
    )


class DFT(Program, KPoints):
    """
    A DFT (Density Functional Theory) is a computational quantum mechanical modeling method used to investigate
    the electronic structure of many-body systems, particularly atoms, molecules, and the condensed phases.
    It is characterized by its use of exchange-correlation functionals and basis sets to obtain the electronic
    properties of materials. It also requires a program name and version.
    """

    exchange_correlation_functional: Literal[
        "LDA", "GGA", "metaGGA", "hyperGGA", "hybrid", "unknown"
    ] = Field(
        "unknown",
        description="""
        The exchange-correlation functional used in the DFT calculation. It can be one of the following:
            - 'LDA': Local Density Approximation
            - 'GGA': Generalized Gradient Approximation
            - 'metaGGA': Meta Generalized Gradient Approximation
            - 'hyperGGA': Hyper Generalized Gradient Approximation
            - 'hybrid': Hybrid functional (a combination of DFT and Hartree-Fock)
            - 'unknown': If the exchange-correlation functional is not known by our enumeration or not specified.

        Defaults to 'unknown'.
        """,
    )

    basis_set: Literal[
        "plane waves",
        "atom-centered orbitals",
        "(L)APW",
        "(L)APW+lo",
        "gaussians+plane waves",
        "real-space grid",
        "unknown",
    ] = Field(
        "unknown",
        description="""
        The basis set used in the DFT calculation. It can be one of the following:
            - 'plane waves': Plane wave basis set
            - 'atom-centered orbitals': Atom-centered orbital basis set
            - '(L)APW': (Linearized) Augmented Plane Wave basis set
            - '(L)APW+lo': (Linearized) Augmented Plane Wave plus local orbitals basis set
            - 'gaussians+plane waves': Gaussian and plane wave basis set
            - 'real-space grid': Real-space grid basis set
            - 'unknown': If the basis set is not known by our enumeration or not specified.

        Defaults to 'unknown'.
        """,
    )

    planewave_cutoff_energy: float | None = Field(
        None,
        description="""
        The plane wave cutoff energy used in the DFT calculation.
        This parameter is relevant only if the basis set is 'plane waves'.
        """,
    )

    planewave_cutoff_energy_units: str = Field(
        "eV",
        description="""
        The units of the plane wave cutoff energy. Defaults to electronvolts (eV).
        Other possible values could be Hartree (Ha), Rydberg (Ry), etc.
        """,
    )

    rkmax: float | None = Field(
        None,
        description="""
        The product of the radius times the reciprocal vector modules for integration in '(L)APW' and '(L)APW+lo' basis
        sets.
        """,
    )


class DMFT(Program, KPoints):
    """
    A DMFT (Dynamical Mean Field Theory) is a methodology used to study strongly correlated electron systems.
    It is characterized by its use of impurity solvers, inverse temperature, and magnetic state to obtain
    the electronic behavior of materials. It also requires a program name and version. It also uses
    the information of the electronic interactions and number of electrons of the correlated atoms in the system.
    """

    correlated_atoms: list[str] = Field(
        ...,
        description="""
        The list of atoms which are correlated using their chemical symbols and species identifier for the unit cell.
        For example, for Fe2O3, imagine it has 2 Fe atoms in the unit cell and both are correlated, then the list
        would be ['Fe1', 'Fe2'].
        """,
    )

    number_of_electrons: dict[str, float] = Field(
        {},
        description="""
        The number of conduction electrons for each of the correlated atoms in the system.
        The keys are the correlated atom identifiers (e.g., 'Fe1', 'Fe2') and the values are the number of electrons.
        For example, for Fe2O3, if the first Fe atom has 8 electrons and the second 6 electrons,
        then the dictionary would be {'Fe1': 8.0, 'Fe2': 6.0}.
        """,
    )

    correlated_orbitals: dict[str, list[str]] = Field(
        ...,
        description="""
        The orbitals which are correlated for each of the correlated atoms in the system.
        The keys are the correlated atom identifiers (e.g., 'Fe1', 'Fe2') and the values are the correlated orbitals string.
        For example, for Fe2O3, if the first Fe atom has 3d orbitals and the second Fe atom has 4d orbitals,
        then the dictionary would be {'Fe1': ['3d'], 'Fe2': ['4d']}.
        """,
    )

    hubbard_interactions: dict[str, float] = Field(
        {},
        description="""
        The Hubbard interaction (U) for the correlated atoms in the system.
        The keys are the correlated atom identifiers (e.g., 'Fe1', 'Fe2') and the values are the number of electrons.
        The values are unitless, but the units can be specified in `hubbard_interaction_units`.
        For example, for Fe2O3, if the first Fe atom has a Hubbard interaction of 4.0 eV and the second has 3.0 eV,
        then the dictionary would be {'Fe1': 4.0, 'Fe2': 3.0}.
        """,
    )

    hubbard_interaction_units: str = Field(
        "eV",
        description="""
        The units of the Hubbard interactions. Defaults to electronvolts (eV).
        Other possible values could be Hartree (Ha), Rydberg (Ry), etc.
        """,
    )

    hunds_couplings: dict[str, float] = Field(
        {},
        description="""
        The unitless Hund's coupling (J or JH) for the correlated atoms in the system.
        The keys are the correlated atom identifiers (e.g., 'Fe1', 'Fe2') and the values are the number of electrons.
        The values are unitless, but the units can be specified in `hubbard_interaction_units`.
        For example, for Fe2O3, if the first Fe atom has a Hund's coupling of 0.5 Ry and the second has 0.3 Ry,
        then the dictionary would be {'Fe1': 0.5, 'Fe2': 0.3}.
        """,
    )

    hunds_coupling_units: str = Field(
        "eV",
        description="""
        The units of the Hund's couplings. Defaults to electronvolts (eV).
        Other possible values could be Hartree (Ha), Rydberg (Ry), etc.
        """,
    )

    impurity_solver: Literal[
        "CT-INT",
        "CT-HYB",
        "CT-AUX",
        "ED",
        "NRG",
        "MPS",
        "IPT",
        "NCA",
        "OCA",
        "slave_bosons",
        "hubbard_I",
        "unknown",
    ] = Field(
        "unknown",
        description="""
        The impurity solver used in the DMFT calculation. It can be one of the following:
            - 'CT-INT': Continuous Time Interaction
            - 'CT-HYB': Continuous Time Hybridization
            - 'CT-AUX': Continuous Time Auxiliary Field
            - 'ED': Exact Diagonalization
            - 'NRG': Numerical Renormalization Group
            - 'MPS': Matrix Product States
            - 'IPT': Iterated Perturbation Theory
            - 'NCA': Non-Crossing Approximation
            - 'OCA': One-Channel Approximation
            - 'slave_bosons': Slave Boson Method
            - 'hubbard_I': Hubbard I Approximation
            - 'unknown': If the impurity solver is not known by our enumeration or not specified.

        Defaults to 'unknown'.
        """,
    )

    temperature: float | None = Field(
        None,
        description="""
        The temperature used in the DMFT calculation in Kelvin.
        This parameter is relevant only if `inverse_temperature` is not provided.
        """,
    )

    inverse_temperature: float | None = Field(
        None,
        description="""
        The inverse temperature (1/kT) used in the DMFT calculation, where k is the Boltzmann constant and
        T is the temperature in Kelvin.
        This parameter is relevant only if `temperature` is not provided.
        """,
    )

    magnetic_state: Literal[
        "paramagnetic", "antiferromagnetic", "ferromagnetic", "altermagnetic", "unknown"
    ] = Field(
        "unknown",
        description="""
        The magnetic state of the system. It can be one of the following:
            - 'paramagnetic': No long-range magnetic order.
            - 'antiferromagnetic': Alternating magnetic moments, no net magnetization.
            - 'ferromagnetic': All magnetic moments aligned, net magnetization.
            - 'altermagnetic': Alternating magnetic moments with net magnetization.
            - 'unknown': If the magnetic state is not known or not specified.

        Defaults to 'unknown'.
        """,
    )
