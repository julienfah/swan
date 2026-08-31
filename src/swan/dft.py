from gpaw import GPAW, PW, MixerSum
from wannierberri.symmetry.point_symmetry import PointGroup
from irrep.spacegroup import SpaceGroup
from wannierberri.grid.grid import determineNK
import numpy as np
from pathlib import Path
from ase.io import read
from gpaw.mpi import world


from swan.utils.utils import (
    adaptative_g_grid,
    standardize_cell,
    adaptative_high_sym_k_grid,
)


def compute_nscf_kmesh(atoms, NKFFT_=1, NK_=12, kill_axis=None):  # correct??
    pg = PointGroup(real_lattice=atoms.cell.array.T)  # columns = lattice vectors
    periodic = np.array(atoms.pbc)
    NKdiv, NKFFT = determineNK(
        periodic=periodic, NKdiv=None, NKFFT=NKFFT_, NK=NK_, NKFFT_recommended=NKFFT_, pointgroup=pg
    )
    ret = NKdiv * NKFFT
    if kill_axis is not None:
        ret[kill_axis] = 1  # for 2D materials, we can kill the axis perpendicular to the plane of the material
    return tuple(ret)


def scf(seed, out_dir, k_grid,atoms=None, input_file=None, ecut=500, density_conv=1e-7):
    """
    Perform self-consistent field calculation for the given atoms.

    :param atoms: ASE Atoms object representing the atomic structure that will be used for the computation.
    :param seed: Seed name for output files.
    """
    if world.rank == 0:
        print(f"Running SCF calculation for {seed} in directory {out_dir}")
    if atoms is None:
        if input_file is None:
            raise ValueError("Either atoms or input_file must be provided.")
        atoms = standardize_cell(read(input_file))

    kx, ky, kz = k_grid
    if world.rank == 0:
        print(f"Using k-mesh: {kx}x{ky}x{kz}")
    grid = [kx, ky, kz]
    calc = GPAW(
        mode=PW(ecut),
        xc="PBE",
        kpts={"size": grid, "gamma": True},
        convergence={"density": density_conv},
        mixer=MixerSum(0.25, 8, 100),
        parallel={'sl_auto': True},
        txt=f"{out_dir}/{seed}/{seed}-scf.txt",
    )
    corrected_grid = adaptative_g_grid(calc, atoms)
    if corrected_grid is not None:
        calc = GPAW(
            mode=PW(ecut),
            xc="PBE",
            kpts={"size": grid, "gamma": True},
            gpts=corrected_grid,
            convergence={"density": density_conv},
            mixer=MixerSum(0.25, 8, 100),
            parallel={'sl_auto': True},
            txt=f"{out_dir}/{seed}/{seed}-scf.txt",
        )
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(f"{out_dir}/{seed}/{seed}-scf.gpw", mode="all")


def nscf(seed, out_dir, in_dir,k_grid,calc=None, nbands=40, unconverged_bands=2):
    """
    Perform non-self-consistent field calculation, reading from the output of the SCF calculation.
    """
    if world.rank == 0:
        print(f"Running NSCF calculation for {seed} in directory {out_dir}")
    if calc is None:
        calc = GPAW(f"{in_dir}/{seed}/{seed}-scf.gpw", txt=None)

    space_group = SpaceGroup.from_gpaw(calc)
    irred_k_points = space_group.get_irreducible_kpoints_grid(k_grid)
    calc_nscf_irred = calc.fixed_density(
        kpts=irred_k_points,
        nbands=nbands,
        convergence={"bands": nbands - unconverged_bands},
        txt=f"{out_dir}/{seed}/{seed}-nscf-irred.txt",
    )
    calc_nscf_irred.write(f"{out_dir}/{seed}/{seed}-nscf-irred.gpw", mode="all")


def dft_bands(seed, in_dir,out_dir,calc=None ,dft_nbands=14, npoints=100,unconverged_bands=2):
    """
    Compute the band structure directly from the DFT calculation for comparison with the Wannier-interpolated bands.
    """
    if Path(f"{in_dir}/{seed}/{seed}-bands.gpw").exists():
        if world.rank == 0:
            print(f"DFT bands already computed for {seed}. Skipping.")
        return
    if calc is None:
        calc = GPAW(f"{in_dir}/{seed}/{seed}-scf.gpw")
    # compute the band directly from gpaw for comparison
    atoms = calc.atoms
    path = atoms.cell.bandpath(npoints=npoints)
    # print(path)
    dft_calc_bands = calc.fixed_density(
        nbands=dft_nbands,
        symmetry="off",
        kpts=path,  # {'path': list(path.values()), 'npoints': 100},
        convergence={"bands": dft_nbands - unconverged_bands},
        txt=f"{out_dir}/{seed}/{seed}-bands.txt",
    )
    dft_calc_bands.write(f"{out_dir}/{seed}/{seed}-bands.gpw", mode="all")


def full_dft_run(
    seed,
    out_dir,
    in_dir,
    atoms=None,
    input_file=None,
    skip_scf=False,
    skip_nscf=False,
    auto_nk_grid=False,
    kill_axis=None,
    max_denominator=8,
    tol=1e-5,
    nk=12,
    nkfft=1,
    ecut=500.0,
    density_conv_scf=1e-7,
    nbands=18,
    unconverged_bands=2,
    npoints=100,
    dft_plot_nbands=None,
):
    """
    Run the full DFT calculation (SCF and NSCF + band structure) for the given atoms.
    """
    if auto_nk_grid:
        k_grid = adaptative_high_sym_k_grid(atoms, nk_length=20, multiplier=1,kill_axis=kill_axis, max_denominator=max_denominator, tol=tol)
    else:
        k_grid = compute_nscf_kmesh(atoms, nkfft, nk)
    if not skip_scf:
        scf(
            atoms=atoms,
            input_file=input_file,
            seed=seed,
            out_dir=out_dir,
            k_grid=k_grid,
            ecut=ecut,
            density_conv=density_conv_scf,
        )

    if not skip_nscf:
        nscf(nbands=nbands, unconverged_bands=unconverged_bands, seed=seed, out_dir=out_dir, in_dir=in_dir, k_grid=k_grid)
    dft_nbands = nbands if dft_plot_nbands is None else dft_plot_nbands
    dft_bands(seed=seed, in_dir=in_dir, out_dir=out_dir, dft_nbands=int(dft_nbands*0.8), npoints=npoints, unconverged_bands=unconverged_bands)
