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
    out_dir,
    in_dir,
    calc_nscf_irred=None,
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
    if calc_nscf_irred is None:
        calc_nscf_irred = GPAW(f"{in_dir}/{seed}/{seed}-nscf-irred.gpw", txt=None, communicator=comm)
    wandata, bandstructure = WannierData.from_gpaw(
        calculator=calc_nscf_irred,
        spin_channel=spin_channel,
        projections=proj_set,
        irreducible=True,
        files=["amn", "mmn", "eig", "symmetrizer"],
        unitary_params=unitary_params,
        return_bandstructure=True,
    )
    wandata.to_npz(f"{out_dir}/{seed}/{seed}_wannier_data")

    wandata.wannierise(
        froz_min=frozen_win[0],
        froz_max=frozen_win[1],
        outer_min=outer_win[0],
        outer_max=outer_win[1],  # np.inf,#
        **wannierization_params,
    )
    wandata.chk.to_npz(f"{out_dir}/{seed}/{seed}_wannier_data.chk.npz")
    # log spreads
    with open(f"{out_dir}/{seed}/{seed}_wannier_spreads.txt", "w") as f:
        for center, spread in zip(wandata.chk.wannier_centers_cart, wandata.chk.wannier_spreads):
            f.write(f"Center: {center}, Spread: {spread}\n")


def interpolate_bands(seed, out_dir, in_dir, calc_nscf_irred=None,wannier_data=None, npoints=200, comm=serial_comm):
    """
    Use of the Wannier functions to interpolate the bands.
    """
    if calc_nscf_irred is None:
        calc_nscf_irred = GPAW(f"{in_dir}/{seed}/{seed}-nscf-irred.gpw", txt=None, communicator=comm)
    atoms = calc_nscf_irred.atoms
    path = atoms.cell.bandpath()
    if wannier_data is None:
        wannier_data = WannierData.from_npz(
            seedname=f"{out_dir}/{seed}/{seed}_wannier_data",
            files=["amn", "mmn", "eig", "chk", "symmetrizer"],
            ignore_missing_files=False,
            irreducible=True,
    )
    wandata = wannier_data
    system = System_R.from_wannierdata(wandata=wandata, berry=True)

    kpoints = path.special_points  # dict of label: kcoords
    path_labels = parse_path_string(path.path)[0]  # string like 'GXWLGK', drop unconnected parts at the moment
    print(f"Interpolating bands along the path: {path_labels[0]}")

    wb_path = Path.from_nodes(
        real_lattice=system.real_lattice,
        nodes=[kpoints[label] for label in path_labels],
        labels=list(path_labels),
        nk=int(npoints/(len(path_labels) - 1)),
    )

    bands_wannier = evaluate_k_path(system, path=wb_path)
    return bands_wannier, wb_path
