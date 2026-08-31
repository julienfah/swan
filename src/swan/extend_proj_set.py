import warnings
from irrep.spacegroup import SpaceGroup
from wannierberri.symmetry.projections import Projection, ProjectionsSet
from wannierberri.symmetry.wyckoff_position import WyckoffPosition
import spglib
from pyxtal import Group

from swan.utils.utils import safety_check_windows
from swan.utils.utils import find_emax_from_dos

"""
EXPERIMENTAL and UNTESTED : is probably unfit for SAWF altough it uses Wyckoff positions.

Extends a given projection set with s orbitals from the next smallest-multiplicity wyckoff position, and returns the new projection set.
"""


def extend_proj_set_with_next_wyckoff_position(calc, proj_set: ProjectionsSet):
    """
    Extends a given projection set with s orbitals from the next smallest-multiplicity Wyckoff position, and returns the new projection set.

    :param calc: GPAW calculator object containing the atomic structure and electronic structure information.
    :param proj_set: ProjectionsSet object representing the initial projection set to be extended.
    :return: ProjectionsSet object representing the extended projection set.
    """
    cell = calc.atoms.cell
    positions = calc.atoms.get_scaled_positions()
    numbers = calc.atoms.numbers
    lattice = (cell, positions, numbers)
    # Get the Wyckoff positions for the given lattice (in lattice coordinates)
    wps = SpaceGroup.wyckoff_positions(lattice)
    # Get the multiplicities of the Wyckoff positions
    dataset = spglib.get_symmetry_dataset(lattice)
    group = Group(dataset.number)
    mult_wp = list(
        zip(
            [m.multiplicity for m in group.Wyckoff_positions],
            [WyckoffPosition(position_str=wp, spacegroup=SpaceGroup.from_gpaw(calc)) for wp in wps],
        )
    )  # multiplcity is in the conventional cell, but this is not a problem as we are interested in the ordering

    # get already occupied wyckoff positions in the projection set
    occupied_wp = []

    # check: if equivalent to a wyckof pos, add to occupied_wp
    for occ_pos in proj_set.get_positions():
        fitting_pos = []
        for mult, wp_pos in mult_wp:
            sol = wp_pos.contains_position(occ_pos)
            if sol is not None:  # means it is in the position
                fitting_pos.append((mult, wp_pos))
        minimal_fitting_pos = min(fitting_pos, key=lambda x: x[0])  # get the one with the smallest multiplicity
        occupied_wp.append(minimal_fitting_pos)  # list of tuples (multiplicity, wyckoff_position)

    # then select the next smallest-multiplicity wyckoff position that is not already occupied
    unoccupied_wp = [wp for wp in mult_wp if wp not in occupied_wp]
    if not unoccupied_wp:
        raise ValueError("All Wyckoff positions are already occupied in the projection set. Cannot extend further.")
    next_smallest_wp = min(unoccupied_wp, key=lambda wp: wp[0])
    print(next_smallest_wp[1].positions)
    proj = Projection(wyckoff_position=next_smallest_wp[1], rotate_basis=True, orbital="s")
    num_new_wf = len(proj.positions)

    extended_proj_set = ProjectionsSet(projections=proj_set.projections + [proj])
    return (
        extended_proj_set,
        num_new_wf,
    )  # return the extended projection set and the multiplicity of the added wyckoff position


def extend_to_energy_window(
    calc,
    energies,
    dos_total,
    K,
    nwann,
    proj_set: ProjectionsSet,
    outer_win: tuple,
    frozen_win: tuple,
    objective_window: tuple,
    max_iter=10,
) -> ProjectionsSet:
    """
    Extends a given projection set with s orbitals from the next smallest-multiplicity Wyckoff position until the energy window is covered.

    :param calc: GPAW calculator object containing the atomic structure and electronic structure information.
    :param nwann: Number of wannier functions from the actual projection set.
    :param proj_set: ProjectionsSet object representing the initial projection set to be extended.
    :param outer_win: Tuple representing the outer energy window (min_energy, max_energy).
    :param frozen_win: Tuple representing the frozen energy window (min_energy, max_energy).
    :param objective_window: Tuple representing the target frozen energy window (min_energy, max_energy) that we want to cover with the projection set.
    :return: ProjectionsSet object representing the extended projection set.
    """
    keep_going = True
    iters = 0
    new_nwann = nwann
    while keep_going:
        # check if the current projection set covers the objective window
        iters += 1
        if frozen_win[0] <= objective_window[0] and frozen_win[1] >= objective_window[1]:
            keep_going = False
        elif iters > max_iter:
            warnings.warn(
                f"Warning: Maximum number of iterations ({max_iter}) reached while trying to extend the projection set to cover the objective window. Objective window was not reached. Current frozen window: {frozen_win}, Objective window: {objective_window}."
            )
            keep_going = False
        else:
            # extend the projection set with the next smallest-multiplicity Wyckoff position
            proj_set, multiplicity = extend_proj_set_with_next_wyckoff_position(calc, proj_set)
            new_nwann += (
                1 * multiplicity
            )  # s orbitals, need to be multiplied by the multiplicity of the wyckoff position
            # adapt the outer window maximum to the new number of wannier functions
            emax_refined = find_emax_from_dos(
                energies=energies, dos_total=dos_total, n_wann=new_nwann, emin=outer_win[0], K=K
            )
            new_outer_win = (outer_win[0], emax_refined)
            # safety check the new windows
            outer_win, frozen_win = safety_check_windows(
                calc, nwann=new_nwann, outer_win=new_outer_win, frozen_win=outer_win
            )

    print(f"Extended projection set to {new_nwann} orbitals.")
    return proj_set, outer_win, frozen_win, new_nwann


"""# test
calc = GPAW(f"{"test_bands"}/{"ClNa"}/{"ClNa"}-nscf-irred.gpw")# Load your GPAW calculator with the atomic structure
sg = SpaceGroup.from_gpaw(calc)
initial_proj_set = ProjectionsSet([Projection(position_num=[0,0,0],spacegroup=sg),Projection(position_num=[0.5,0.5,0.5],spacegroup=sg),Projection(position_num=[0.25,0.25,0.25],spacegroup=sg)])  # Create your initial projection set here
extended_proj_set, multiplicity = extend_proj_set_with_next_wyckoff_position(calc, initial_proj_set)
print("Extended Projection Set:",extended_proj_set)
"""
