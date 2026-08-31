from gpaw import GPAW
from matplotlib import pyplot as plt
from pathlib import Path as Path_
from ase.io import read
from gpaw.mpi import world, serial_comm

from swan.utils.utils import parse_args, standardize_cell,adaptative_nscf_nbands
from swan.dft import full_dft_run
from swan.wannier import wannierize, interpolate_bands
from swan.auto_proj_and_windows import get_proj_set
from swan.utils.metrics import least_square_deviation_within_frozen,max_deviation_within_frozen
from swan.ebr.ebr_method import EBR_method

def plot_bands(bands_wannier, wb_path, outer_win, frozen_win, seed, out_dir,in_dir):
    """
    Plot the interpolated bands and compares with the ones from the DFT calculation.
    """
    print("Plotting the bands to compare with DFT")
    bs_dft = GPAW(f"{in_dir}/{seed}/{seed}-bands.gpw", communicator=serial_comm).band_structure()
    # plot comparison

    fig, ax = plt.subplots(figsize=(6, 6))
    bs_dft.plot(show=False, ax=ax, label="DFT", color="red")

    bands_wannier.plot_path_fat(
        path=wb_path,
        label="wannierised",
        axes=ax,
        close_fig=False,
        show_fig=False,
        kwargs_line=dict(linestyle="--", lw=1.0, color="blue"),
    )
    plt.axhspan(ymin=outer_win[0], ymax=outer_win[1], color="gray", alpha=0.1, label="outer window")
    plt.axhspan(ymin=frozen_win[0], ymax=frozen_win[1], color="gray", alpha=0.3, label="frozen window")
    plt.legend(loc="upper right")
    plt.title(f"{seed} band structure")
    plt.savefig(f"{out_dir}/{seed}/{seed}-wannierized_bands.png", dpi=200)
    #log least square deviation
    with open(f"{out_dir}/{seed}/{seed}_wannier_spreads.txt", "a") as f:
        f.write("######## Band interpolation metrics #######\n")
        #f.write(f"least square deviation: {least_square_deviation(bs_dft.energies,bs_dft.path.kpts, bands_wannier.Enk.data,wb_path.get_kpoints(), outer_win)}\n")
        f.write(f"least_square_deviation_within_frozen: {least_square_deviation_within_frozen(bs_dft.energies,bs_dft.path.kpts, bands_wannier.Enk.data,wb_path.get_kpoints(), outer_win, frozen_win):.4f} meV\n")
        f.write(f"max_deviation_within_frozen: {max_deviation_within_frozen(bs_dft.energies,bs_dft.path.kpts, bands_wannier.Enk.data,wb_path.get_kpoints(), frozen_win):.4f} meV\n")

def auto_workflow(
    atoms,
    seed,
    out_dir="test",
    in_dir="test",
    auto_nk_grid=False,
    kill_axis=None,
    max_denominator=8,
    tol=1e-5,
    nk=12,
    nkfft=1,
    ecut=500.0,
    density_conv_scf=1e-7,
    gap_thres=1,
    nbands_per_valence_el=5,
    nbands_per_atom=None,
    nbands=None,
    unconverged_bands_prc=5,
    npoints=144,
    dft_plot_nbands=None,
    K=1.2,
    spin_channel=0,
    npts_dos=1001,
    dos_width=0.05,
    num_iter=100,
    w_conv_tol=1e-8,
    print_progress_every=20,
    maximize_fw=False,
    objective_wd=None,
    no_sitesym=False,
    no_localise=False,
    error_threshold=0.1,
    warning_threshold=0.01,
    skip_scf=False,
    skip_nscf=False,
    skip_wannier=False,
    only_dft=False,
    only_wannier=False,
    hybridize_on_site=True,
    pdos=False,
    alphabet="shells+hyb",
    validate_n_max=5,
    EBR_margin=2,
    verbose=False
):
    n_bands = adaptative_nscf_nbands(
        seed=seed,dir=in_dir, atoms=atoms, nbands_per_atom=nbands_per_atom, nbands=nbands, n_bands_per_valence_el=nbands_per_valence_el
    )
    unconverged_bands = max(2, int(n_bands * unconverged_bands_prc / 100))
    if not only_wannier:
        if in_dir !=out_dir:
            skip_scf = True
            skip_nscf = True
        full_dft_run(
            seed=seed,
            out_dir=out_dir,
            in_dir=in_dir,
            atoms=atoms,
            skip_scf=skip_scf,
            skip_nscf=skip_nscf,
            auto_nk_grid=auto_nk_grid,
            kill_axis=kill_axis,
            max_denominator=max_denominator,
            tol=tol,
            nk=nk,
            nkfft=nkfft,
            ecut=ecut,
            density_conv_scf=density_conv_scf,
            nbands=n_bands,
            unconverged_bands=unconverged_bands,
            npoints=npoints,
            dft_plot_nbands=dft_plot_nbands,
        )
        world.barrier()
    if not only_dft:
        if world.rank == 0:
            # import ray
            # ray.init(num_cpus=16,num_gpus=18,ignore_reinit_error=True)

            print(f"DFT calculations completed for {seed}. Proceeding with Wannierization.")
            dos_kwargs = {"spin": spin_channel, "npts": npts_dos, "width": dos_width}

            if pdos:
                proj_set, outer_win, frozen_win, nwann = get_proj_set(
                    K=K,
                    seed=seed,
                    out_dir=out_dir,
                    in_dir=in_dir,
                    dos_kwargs=dos_kwargs,
                    gap_thres=gap_thres,
                    maximize_fw=maximize_fw,
                    objective_wd=objective_wd,
                    hybridize_on_site=hybridize_on_site,
                )
            else:
                proj_set, frozen_win, outer_win = EBR_method(in_dir=in_dir, out_dir=out_dir, seed=seed, ecut=ecut, comm=serial_comm, only_on_site=True, verbose=verbose,K=K, gap_thres=gap_thres,objective_wd=objective_wd,validate=True,validate_n_max=validate_n_max,margin=EBR_margin, eta_ok=20.0, spread_ok=10.0,alphabet=alphabet,k_values=(K,K+0.3))

            if not skip_wannier:
                unitary_params = dict(
                    error_threshold=error_threshold,
                    warning_threshold=warning_threshold,
                    nbands_upper_skip=unconverged_bands,
                )
                wannierization_params = dict(
                    num_iter=num_iter,
                    conv_tol=w_conv_tol,
                    print_progress_every=print_progress_every,
                    sitesym=not no_sitesym,
                    localise=not no_localise,
                )
                wannierize(
                    proj_set=proj_set,
                    outer_win=outer_win,
                    frozen_win=frozen_win,
                    seed=seed,
                    out_dir=out_dir,
                    in_dir=in_dir,
                    spin_channel=spin_channel,
                    unitary_params=unitary_params,
                    wannierization_params=wannierization_params,
                    ecut_pw=ecut,
                )

            bands_wannier, wb_path = interpolate_bands(seed=seed, out_dir=out_dir, in_dir=in_dir, npoints=npoints)

            plot_bands(bands_wannier, wb_path, outer_win, frozen_win, seed=seed, out_dir=out_dir, in_dir=in_dir)


def main():
    ###################################
    ######## CLI ######################
    args = parse_args()
    atoms = read(args.structure)
    ###################################
    # standardize the cell
    atoms = standardize_cell(atoms)
    ####################################

    seed = args.seed if args.seed is not None else atoms.get_chemical_formula()
    output_directory = args.output_dir if args.output_dir is not None else "test"
    input_directory = args.input_dir if args.input_dir is not None else output_directory

    (Path_(output_directory) / Path_(f"{seed}")).mkdir(parents=True, exist_ok=True)
    if not (Path_(input_directory) / Path_(f"{seed}")).exists():
        raise FileNotFoundError(f"Input directory {Path_(input_directory) / Path_(f'{seed}')} does not exist.")
    if not (Path_(output_directory) / Path_(f"{seed}")).exists():
        raise FileNotFoundError(f"Input nscf file {Path_(f'{input_directory}/{seed}/{seed}-nscf-irred.gpw')} does not exist.")
    if not (Path_(output_directory) / Path_(f"{seed}")).exists():
        raise FileNotFoundError(f"Input dft bands file {Path_(f'{input_directory}/{seed}/{seed}-bands.gpw')} does not exist.")

    # build dicts from CLI args
    cli_kwargs = {k: v for k, v in vars(args).items() if v is not None and k not in ["structure", "seed", "output_dir", "input_dir"]}

    auto_workflow(atoms, seed, in_dir=input_directory, out_dir=output_directory, **cli_kwargs)


if __name__ == "__main__":
    main()
