from gpaw import GPAW
from wannierberri.w90files import WannierData
from wannierberri import System_R, Path, evaluate_k_path
from gpaw.mpi import serial_comm
from ase.dft.kpoints import parse_path_string


def wannierize(
    proj_set,
    outer_win,
    frozen_win,
    seed,
    dir,
    spin_channel=0,
    unitary_params=dict(error_threshold=0.1, warning_threshold=0.01, nbands_upper_skip=2),
    wannierization_params=dict(
        num_iter=100,
        conv_tol=1e-8,
        print_progress_every=20,
        sitesym=True,
        localise=True,
    ),
    comm=serial_comm,
):
    """
    Proper wannierization of the system, using the previously determined parameters.

    :param proj_set: ProjectionsSet object containing the projections to be used for the wannierization.
    :param outer_win: Tuple containing the outer energy window boundaries.
    :param frozen_win: Tuple containing the frozen energy window boundaries.
    """

    calc_nscf_irred = GPAW(f"{dir}/{seed}/{seed}-nscf-irred.gpw", txt=None, communicator=comm)
    wandata, bandstructure = WannierData.from_gpaw(
        calculator=calc_nscf_irred,
        spin_channel=spin_channel,
        projections=proj_set,
        irreducible=True,
        files=["amn", "mmn", "eig", "symmetrizer"],
        unitary_params=unitary_params,
        return_bandstructure=True,
    )
    wandata.to_npz(f"{dir}/{seed}/{seed}_wannier_data")

    wandata.wannierise(
        froz_min=frozen_win[0],
        froz_max=frozen_win[1],
        outer_min=outer_win[0],
        outer_max=outer_win[1],  # np.inf,#
        **wannierization_params,
    )
    wandata.chk.to_npz(f"{dir}/{seed}/{seed}_wannier_data.chk.npz")
    # log spreads
    with open(f"{dir}/{seed}/{seed}_wannier_spreads.txt", "w") as f:
        for center, spread in zip(wandata.chk.wannier_centers_cart, wandata.chk.wannier_spreads):
            f.write(f"Center: {center}, Spread: {spread}\n")


def interpolate_bands(seed, dir, npoints=200, comm=serial_comm):
    """
    Use of the Wannier functions to interpolate the bands.
    """
    calc_nscf_irred = GPAW(f"{dir}/{seed}/{seed}-nscf-irred.gpw", txt=None, communicator=comm)
    atoms = calc_nscf_irred.atoms
    path = atoms.cell.bandpath()

    wandata = WannierData.from_npz(
        seedname=f"{dir}/{seed}/{seed}_wannier_data",
        files=["amn", "mmn", "eig", "chk", "symmetrizer"],
        ignore_missing_files=False,
        irreducible=True,
    )

    system = System_R.from_wannierdata(wandata=wandata, berry=True)

    kpoints = path.special_points  # dict of label: kcoords
    path_labels = parse_path_string(path.path)[0]  # string like 'GXWLGK', drop unconnected parts at the moment
    print(f"Interpolating bands along the path: {path_labels[0]}")

    wb_path = Path.from_nodes(
        real_lattice=system.real_lattice,
        nodes=[kpoints[label] for label in path_labels],
        labels=list(path_labels),
        length=npoints,
    )

    bands_wannier = evaluate_k_path(system, path=wb_path)
    return bands_wannier, wb_path
