from gpaw import GPAW, PW, MixerSum
from wannierberri.symmetry.point_symmetry import PointGroup
from irrep.spacegroup import SpaceGroup
from wannierberri.grid.grid import determineNK
import numpy as np
from pathlib import Path
from ase.io import read
from gpaw.mpi import world
import spglib
import subprocess


from pawan.utils import adaptative_g_grid,adaptative_k_grid,adaptative_nscf_nbands,standardize_cell,adaptative_high_sym_k_grid



def compute_nscf_kmesh(atoms,NKFFT_=1,NK_=12):## correct??
    pg = PointGroup(real_lattice=atoms.cell.array.T)  # columns = lattice vectors
    periodic = np.array(atoms.pbc)
    NKdiv, NKFFT = determineNK(
        periodic=periodic,
        NKdiv=None, NKFFT=NKFFT_, NK=NK_,
        NKFFT_recommended =NKFFT_,
        pointgroup=pg
    )
    return tuple(NKdiv * NKFFT)

def scf(seed,dir,atoms=None,input_file=None,ecut=500,density_conv=1e-7,NK=12,NKFFT=1,auto_nk_grid=False):
    '''
    Perform self-consistent field calculation for the given atoms. 

    :param atoms: ASE Atoms object representing the atomic structure that will be used for the computation.
    :param seed: Seed name for output files.
    '''
    if world.rank == 0:
        print(f"Running SCF calculation for {seed} in directory {dir}")
    if atoms is None:
        if input_file is None:
            raise ValueError("Either atoms or input_file must be provided.")
        atoms = standardize_cell(read(input_file))

    if auto_nk_grid:
        kx,ky,kz = adaptative_high_sym_k_grid(atoms,nk_length=20,multiplier=1)#if works well pass nk_length as param
    else:
        kx,ky,kz = compute_nscf_kmesh(atoms,NKFFT,NK)
    if world.rank == 0:
        print(f"Using k-mesh: {kx}x{ky}x{kz}")
    grid = [kx,ky,kz]
    calc = GPAW(
        mode=PW(ecut), 
        xc="PBE",
        kpts={"size": grid, "gamma": True},
        convergence={"density": density_conv},
        mixer=MixerSum(0.25, 8, 100),
        txt=f"{dir}/{seed}/{seed}-scf.txt"
    )
    corrected_grid =adaptative_g_grid(calc,atoms)
    if corrected_grid is not None:
        calc = GPAW(
            mode=PW(ecut), 
            xc="PBE",
            kpts={"size": grid, "gamma": True},
            gpts=corrected_grid,
            convergence={"density": density_conv},
            mixer=MixerSum(0.25, 8, 100),
            txt=f"{dir}/{seed}/{seed}-scf.txt"
        )
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(f"{dir}/{seed}/{seed}-scf.gpw", mode="all")
    

def nscf(seed,dir,nbands=40,unconverged_bands=2,NK=12,NKFFT=1):
    '''
    Perform non-self-consistent field calculation, reading from the output of the SCF calculation.    
    '''
    if world.rank == 0:
        print(f"Running NSCF calculation for {seed} in directory {dir}")
    calc = GPAW(f'{dir}/{seed}/{seed}-scf.gpw', txt=None)
    nscf_grid = compute_nscf_kmesh(calc.atoms,NKFFT,NK)
    space_group = SpaceGroup.from_gpaw(calc)
    irred_k_points = space_group.get_irreducible_kpoints_grid(nscf_grid)

    calc_nscf_irred = calc.fixed_density(
        kpts=irred_k_points,
        nbands=nbands,
        convergence={"bands": nbands-unconverged_bands},
        txt=f'{dir}/{seed}/{seed}-nscf-irred.txt')
    calc_nscf_irred.write(f'{dir}/{seed}/{seed}-nscf-irred.gpw', mode='all')

def dft_bands(seed,dir,dft_nbands=14,npoints=100):
    '''
    Compute the band structure directly from the DFT calculation for comparison with the Wannier-interpolated bands.
    '''
    if Path(f"{dir}/{seed}/{seed}-bands.gpw").exists():
        if world.rank == 0:
            print(f"DFT bands already computed for {seed}. Skipping.")
        return
    calc = GPAW(f"{dir}/{seed}/{seed}-scf.gpw")
    # compute the band directly from gpaw for comparison
    atoms = calc.atoms
    path = atoms.cell.bandpath(npoints=npoints)
    # print(path)
    dft_calc_bands = calc.fixed_density(
        nbands=dft_nbands,
        symmetry='off',
        kpts=path,#{'path': list(path.values()), 'npoints': 100},
        convergence={'bands': dft_nbands-2},
        txt=f"{dir}/{seed}/{seed}-bands.txt")
    dft_calc_bands.write(f"{dir}/{seed}/{seed}-bands.gpw", mode="all")


def full_dft_run(seed,dir,atoms=None,input_file=None,skip_scf=False,skip_nscf=False,auto_nk_grid=False,nk=12,nkfft=1,ecut=500.0,density_conv_scf=1e-7,nbands_per_valence_el=4,nbands_per_atom=None,nbands=None,unconverged_bands=2,npoints=100,dft_plot_nbands=None):
    '''
    Run the full DFT calculation (SCF and NSCF + band structure) for the given atoms.
    '''
    if not skip_scf:
        scf(atoms=atoms,input_file=input_file,seed=seed,dir=dir,NK=nk,NKFFT=nkfft,ecut=ecut,density_conv=density_conv_scf,auto_nk_grid=auto_nk_grid)
    n_bands = adaptative_nscf_nbands(seed=seed,dir=dir,nbands_per_atom=nbands_per_atom,nbands=nbands,n_bands_per_valence_el=nbands_per_valence_el)
    if not skip_nscf:
        nscf(nbands=n_bands,unconverged_bands=unconverged_bands,seed=seed,dir=dir,NK=nk,NKFFT=nkfft)
    dft_nbands = n_bands if dft_plot_nbands is None else dft_plot_nbands
    dft_bands(seed=seed,dir=dir,dft_nbands=dft_nbands,npoints=npoints)
