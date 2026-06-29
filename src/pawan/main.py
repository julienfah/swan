import numpy as np
from math import ceil
from ase import Atoms
from gpaw import GPAW, PW, MixerSum
from irrep.spacegroup import SpaceGroup
from wannierberri.symmetry.projections import Projection, ProjectionsSet
from wannierberri.w90files import WannierData
from wannierberri.grid.grid import determineNK
from wannierberri import System_R,Path,evaluate_k_path
from matplotlib import pyplot as plt
from pathlib import Path as Path_
from collections import defaultdict
from ase.io import read
import spglib
from wannierberri.symmetry.point_symmetry import PointGroup
from wannierberri.symmetry.wyckoff_position import split_into_orbits




#import ray
#ray.init(num_cpus=18,num_gpus=20,ignore_reinit_error=True)

from pawan.utils import get_crystal_system, find_emax_from_dos,parse_args
from pawan.auto_proj_and_windows import get_proj_set

def compute_nscf_kmesh(atoms):## correct??
    pg = PointGroup(real_lattice=atoms.cell.array.T)  # columns = lattice vectors
    periodic = np.array(atoms.pbc)
    NKdiv, NKFFT = determineNK(
        periodic=periodic,
        NKdiv=None, NKFFT=1, NK=12,
        NKFFT_recommended =1,
        pointgroup=pg
    )
    return tuple(NKdiv * NKFFT)

def scf(atoms,ecut,seed):
    '''
    Perform self-consistent field calculation for the given atoms. 

    :param atoms: ASE Atoms object representing the atomic structure that will be used for the computation.
    :param seed: Seed name for output files.
    '''
    print("Running self-consistent calculation")
    kx,ky,kz = compute_nscf_kmesh(atoms)  #compute_kmesh(atoms,emp_param=30)
    print(f"Using k-mesh: {kx}x{ky}x{kz}")
    grid = [kx,ky,kz]
    calc = GPAW(
        mode=PW(ecut), 
        xc="PBE",
        kpts={"size": grid, "gamma": True},
        convergence={"density": 1e-7},
        mixer=MixerSum(0.25, 8, 100),
        txt=f"test/{seed}/{seed}-scf.txt"
    )

    atoms.calc = calc

    atoms.get_potential_energy()
    calc.write(f"test/{seed}/{seed}-scf.gpw", mode="all")
    

def nscf(seed,nbands=40):
    '''
    Perform non-self-consistent field calculation, reading from the output of the SCF calculation.    
    '''
    print("Running non-self-consistent calculation")
    calc = GPAW(f'test/{seed}/{seed}-scf.gpw', txt=None)
    nscf_grid = compute_nscf_kmesh(calc.atoms)
    space_group = SpaceGroup.from_gpaw(calc)
    irred_k_points = space_group.get_irreducible_kpoints_grid(nscf_grid)
    calc_nscf_irred = calc.fixed_density(
        kpts=irred_k_points,
        nbands=nbands,
        convergence={"bands": nbands-2},
        txt=f'test/{seed}/{seed}-nscf-irred.txt')
    calc_nscf_irred.write(f'test/{seed}/{seed}-nscf-irred.gpw', mode='all')
   

def wannierize(proj_set, outer_win, frozen_win, seed):
    '''
    Proper wannierization of the system, using the previously determined parameters.

    :param proj_set: ProjectionsSet object containing the projections to be used for the wannierization.
    :param outer_win: Tuple containing the outer energy window boundaries.
    :param frozen_win: Tuple containing the frozen energy window boundaries.
    '''

    calc_nscf_irred = GPAW(f'test/{seed}/{seed}-nscf-irred.gpw', txt=None)
    wandata, bandstructure = WannierData.from_gpaw(
        calculator=calc_nscf_irred,
        spin_channel=0,
        projections=proj_set,
        #select_grid=nscf_grid,  
        irreducible=True,
        files=["amn", "mmn", "eig", "symmetrizer"],
        unitary_params=dict(error_threshold=0.1,
                            warning_threshold=0.01,
                            nbands_upper_skip=2),
        return_bandstructure=True
    )
    wandata.to_npz(f"test/{seed}/{seed}_wannier_data")


    #print("dtype:", wandata.amn.data)
    #print("shape:", wandata.amn.data)
    #print("Cell volume:", calc_nscf_irred.atoms.get_volume())
    wandata.wannierise(
        froz_min=frozen_win[0],    # below bottom σ band
        froz_max=frozen_win[1],   # just below Fermi/Dirac point (~-2.5 eV)
        outer_min=outer_win[0],
        outer_max=outer_win[1],  # hard cutoff just below vacuum states
        num_iter=100,
        conv_tol=1e-8,
        print_progress_every=20,
        sitesym=True,
        localise=True,
    )
    wandata.chk.to_npz(f"test/{seed}/{seed}_wannier_data.chk.npz")
def interpolate_bands(seed):
    '''
    Use of the Wannier functions to interpolate the bands.
    '''
    calc_nscf_irred = GPAW(f'test/{seed}/{seed}-nscf-irred.gpw', txt=None)
    atoms = calc_nscf_irred.atoms
    path = atoms.cell.bandpath()
    #from wannierberri.symmetry.sawf import SymmetrizerSAWF

    wandata = WannierData.from_npz(seedname=f"test/{seed}/{seed}_wannier_data",files=["amn", "mmn", "eig", "chk", "symmetrizer"],ignore_missing_files=False,irreducible=True)

    system = System_R.from_wannierdata(wandata=wandata, berry=True)

    kpoints = path.special_points  # dict of label: kcoords
    path_labels = path.path .split(',')[0]        # string like 'GXWLGK'

    wb_path = Path.from_nodes(
        real_lattice=system.real_lattice,
        nodes=[kpoints[label] for label in path_labels],  
        labels=list(path_labels),
        length=200
)

    bands_wannier = evaluate_k_path(system, path=wb_path)
    return bands_wannier,wb_path





def plot_bands(bands_wannier,wb_path,outer_win,frozen_win,seed):
    '''
    Plot the interpolated bands and compares with the ones from the DFT calculation.
    '''
    print("Plotting the bands to compare with DFT")
    bs_dft = GPAW(f"test/{seed}/{seed}-bands.gpw").band_structure()
    #plot comparison

    fig,ax = plt.subplots(figsize=(6,6))
    bs_dft.plot(show=False, ax=ax, label="DFT",color="red")

    bands_wannier.plot_path_fat(path=wb_path,
                        label="wannierised",
                            axes=ax,
                            close_fig=False,
                            show_fig=False,
                            kwargs_line=dict(linestyle='--', lw=1.0,color='blue'),
        )
    plt.axhspan(ymin=outer_win[0], ymax=outer_win[1], color='gray', alpha=0.1,label="outer window")
    plt.axhspan(ymin=frozen_win[0], ymax=frozen_win[1], color='gray', alpha=0.3,label="frozen window")
    plt.legend(loc='upper right')
    plt.title(f"{seed} band structure")
    plt.savefig(f"test/{seed}/{seed}-wannierized_bands.png", dpi=300)

def dft_bands(seed):
    calc = GPAW(f"test/{seed}/{seed}-scf.gpw")
    # compute the band directly from gpaw for comparison
    atoms = calc.atoms
    path = atoms.cell.bandpath(npoints=100)
    print(path)
    dft_calc_bands = calc.fixed_density(
        nbands=14,
        symmetry='off',
        kpts=path,#{'path': list(path.values()), 'npoints': 100},
        convergence={'bands': 8},
        txt=f"test/{seed}/{seed}-bands.txt")
    dft_calc_bands.write(f"test/{seed}/{seed}-bands.gpw", mode="all")

def auto_workflow(atoms, args,seed):
    if not args.skip_scf:
        scf(atoms,ecut=args.ecut,seed=seed)
    if not args.skip_nscf:
        nscf(nbands=args.nbands,seed=seed)
    proj_set, outer_win, frozen_win, nwann = get_proj_set(K=args.K,seed=seed)
    if not args.skip_wannier:
        wannierize(proj_set, outer_win, frozen_win,seed=seed)
    bands_wannier, wb_path = interpolate_bands(seed=seed)
    dft_bands(seed=seed)
    plot_bands(bands_wannier, wb_path, outer_win, frozen_win,seed=seed)

def main():
    ###################################
    ######## CLI ######################
    args = parse_args()
    atoms = read(args.structure)
    ###################################
    #standardize the cell
    cell = spglib.standardize_cell(
        (atoms.cell, atoms.get_scaled_positions(), atoms.numbers),
        to_primitive=True,
        symprec=1e-3
    )
    atoms = Atoms(numbers=cell[2], scaled_positions=cell[1], cell=cell[0], pbc=True)
    ####################################

    seed = args.seed if args.seed is not None else atoms.get_chemical_formula()

    (Path_(args.output_dir)/Path_(f"{seed}")).mkdir(parents=True, exist_ok=True)

    auto_workflow(atoms, args,seed)
if __name__ == "__main__":           
    main()
