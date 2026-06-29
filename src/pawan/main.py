import numpy as np
from math import ceil
from ase import Atoms
from gpaw import GPAW, PW, MixerSum
from irrep.spacegroup import SpaceGroup
from wannierberri.w90files import WannierData
from wannierberri.grid.grid import determineNK
from wannierberri import System_R,Path,evaluate_k_path
from matplotlib import pyplot as plt
from pathlib import Path as Path_
from ase.io import read
import spglib
from wannierberri.symmetry.point_symmetry import PointGroup




#import ray
#ray.init(num_cpus=18,num_gpus=20,ignore_reinit_error=True)

from pawan.utils import get_crystal_system, find_emax_from_dos,parse_args
from pawan.auto_proj_and_windows import get_proj_set

def compute_nscf_kmesh(atoms,NKFFT_=1,NK_=12):
    '''Compute the k-point mesh for the non-self-consistent field (NSCF) calculation based on the crystal system of the given atoms.
    
    :param atoms: ASE Atoms object representing the atomic structure that will be used for the computation.
    :param NKFFT_: Integer representing the number of k-points in the FFT grid (default: 1).
    :param NK_: Integer representing the number of k-points in the NSCF calculation (default: 12).
    :return: Tuple containing the number of k-points in each direction (kx, ky, kz) for the NSCF calculation.
    '''
    pg = PointGroup(real_lattice=atoms.cell.array.T)  # columns = lattice vectors
    periodic = np.array(atoms.pbc)
    NKdiv, NKFFT = determineNK(
        periodic=periodic,
        NKdiv=None, NKFFT=NKFFT_, NK=NK_,
        NKFFT_recommended =NKFFT_,
        pointgroup=pg
    )
    return tuple(NKdiv * NKFFT)

def scf(atoms,seed,dir,ecut=500,density_conv=1e-7):
    '''
    Perform self-consistent field calculation for the given atoms. 

    :param atoms: ASE Atoms object representing the atomic structure that will be used for the computation.
    :param seed: Seed name for output files.
    :param dir: Directory for output files.
    :param ecut: Cutoff energy for the plane-wave basis.
    :param density_conv: Density convergence threshold.
    '''
    print("Running self-consistent calculation")
    kx,ky,kz = compute_nscf_kmesh(atoms)  #compute_kmesh(atoms,emp_param=30)
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

    atoms.calc = calc

    atoms.get_potential_energy()
    calc.write(f"{dir}/{seed}/{seed}-scf.gpw", mode="all")
    

def nscf(seed,dir,nbands=40,unconverged_bands=2):
    '''
    Perform non-self-consistent field calculation, reading from the output of the SCF calculation.    
    
    :param seed: Seed name for output files.
    :param dir: Directory for output files.
    :param nbands: Number of bands for the NSCF calculation.
    :param unconverged_bands: Number of unconverged bands for the NSCF
    '''
    print("Running non-self-consistent calculation")
    calc = GPAW(f'{dir}/{seed}/{seed}-scf.gpw', txt=None)
    nscf_grid = compute_nscf_kmesh(calc.atoms)
    space_group = SpaceGroup.from_gpaw(calc)
    irred_k_points = space_group.get_irreducible_kpoints_grid(nscf_grid)
    calc_nscf_irred = calc.fixed_density(
        kpts=irred_k_points,
        nbands=nbands,
        convergence={"bands": nbands-unconverged_bands},
        txt=f'{dir}/{seed}/{seed}-nscf-irred.txt')
    calc_nscf_irred.write(f'{dir}/{seed}/{seed}-nscf-irred.gpw', mode='all')
   

def wannierize(proj_set, outer_win, frozen_win, seed,dir,spin_channel=0,unitary_params=dict(error_threshold=0.1,warning_threshold=0.01,nbands_upper_skip=2),wannierization_params=dict(num_iter=100,conv_tol=1e-8,print_progress_every=20,sitesym=True,localise=True,)):
    '''
    Proper wannierization of the system, using the previously determined parameters.

    :param proj_set: ProjectionsSet object containing the projections to be used for the wannierization.
    :param outer_win: Tuple containing the outer energy window boundaries.
    :param frozen_win: Tuple containing the frozen energy window boundaries.
    :param seed: Seed name for output files.
    :param dir: Directory for output files.
    :param spin_channel: Spin channel for the wannierization.
    :param unitary_params: Dictionary containing parameters for the unitary matrix check.
    :param wannierization_params: Dictionary containing parameters for the wannierization process.
    '''

    calc_nscf_irred = GPAW(f'{dir}/{seed}/{seed}-nscf-irred.gpw', txt=None)
    wandata, bandstructure = WannierData.from_gpaw(
        calculator=calc_nscf_irred,
        spin_channel=spin_channel,
        projections=proj_set,
        irreducible=True,
        files=["amn", "mmn", "eig", "symmetrizer"],
        unitary_params=unitary_params,
        return_bandstructure=True
    )
    wandata.to_npz(f"{dir}/{seed}/{seed}_wannier_data")


    #print("dtype:", wandata.amn.data)
    #print("shape:", wandata.amn.data)
    #print("Cell volume:", calc_nscf_irred.atoms.get_volume())
    wandata.wannierise(
        froz_min=frozen_win[0],    # below bottom σ band
        froz_max=frozen_win[1],   # just below Fermi/Dirac point (~-2.5 eV)
        outer_min=outer_win[0],
        outer_max=outer_win[1],  # hard cutoff just below vacuum states
        **wannierization_params
    )
    wandata.chk.to_npz(f"{dir}/{seed}/{seed}_wannier_data.chk.npz")
def interpolate_bands(seed,dir,npoints=200):
    '''
    Use of the Wannier functions to interpolate the bands.

    :param seed: Seed name for output files.
    :param dir: Directory for output and computation files.
    :param npoints: Number of points for the band interpolation.
    '''
    calc_nscf_irred = GPAW(f'{dir}/{seed}/{seed}-nscf-irred.gpw', txt=None)
    atoms = calc_nscf_irred.atoms
    path = atoms.cell.bandpath()
    #from wannierberri.symmetry.sawf import SymmetrizerSAWF

    wandata = WannierData.from_npz(seedname=f"{dir}/{seed}/{seed}_wannier_data",files=["amn", "mmn", "eig", "chk", "symmetrizer"],ignore_missing_files=False,irreducible=True)

    system = System_R.from_wannierdata(wandata=wandata, berry=True)

    kpoints = path.special_points  # dict of label: kcoords
    path_labels = path.path .split(',')[0]        # string like 'GXWLGK'

    wb_path = Path.from_nodes(
        real_lattice=system.real_lattice,
        nodes=[kpoints[label] for label in path_labels],  
        labels=list(path_labels),
        length=npoints
)

    bands_wannier = evaluate_k_path(system, path=wb_path)
    return bands_wannier,wb_path





def plot_bands(bands_wannier,wb_path,outer_win,frozen_win,seed,dir):
    '''
    Plot the interpolated bands and compares with the ones from the DFT calculation.

    :param bands_wannier: Band structure obtained from the Wannier interpolation.
    :param wb_path: Path object representing the k-point path for the band structure.
    :param outer_win: Tuple containing the outer energy window boundaries.
    :param frozen_win: Tuple containing the frozen energy window boundaries.
    :param seed: Seed name for output files.
    :param dir: Directory for output and computation files.
    '''
    print("Plotting the bands to compare with DFT")
    bs_dft = GPAW(f"{dir}/{seed}/{seed}-bands.gpw").band_structure()
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
    plt.savefig(f"{dir}/{seed}/{seed}-wannierized_bands.png", dpi=200)

def dft_bands(seed,dir,dft_nbands=14,npoints=100):
    '''
    Compute the band structure directly from the DFT calculation for comparison with the Wannier-interpolated bands.
    
    :param seed: Seed name for output files.
    :param dir: Directory for output and computation files.
    :param dft_nbands: Number of bands to compute in the DFT calculation.
    :param npoints: Number of points for the dft band computation.
    '''
    calc = GPAW(f"{dir}/{seed}/{seed}-scf.gpw")
    # compute the band directly from gpaw for comparison
    atoms = calc.atoms
    path = atoms.cell.bandpath(npoints=npoints)
    print(path)
    dft_calc_bands = calc.fixed_density(
        nbands=dft_nbands,
        symmetry='off',
        kpts=path,#{'path': list(path.values()), 'npoints': 100},
        convergence={'bands': dft_nbands-2},
        txt=f"{dir}/{seed}/{seed}-bands.txt")
    dft_calc_bands.write(f"{dir}/{seed}/{seed}-bands.gpw", mode="all")

#default params
DEFAULT_DFT_PARAMS = {"ecut": 500.0, "density_conv_scf": 1e-7, "nbands": 40, "unconverged_bands": 2,"npoints": 200,"dft_plot_nbands": 14}
DEFAULT_WINDOW_PARAMS = {"K": 1.2,"dos_kwargs": {'spin': 0, 'npts': 1001, 'width': 0.05}}
DEFAULT_WANNIER_PARAMS = {"spin_channel":0,"unitary_params":dict(error_threshold=0.1,warning_threshold=0.01,nbands_upper_skip=2),"wannierization_params":dict(num_iter=100,conv_tol=1e-8,print_progress_every=20,sitesym=True,localise=True)}
DEFAULT_WORKFLOW_FLAGS = {"skip_scf": False, "skip_nscf": False, "skip_wannier": False}

def auto_workflow(atoms, seed, dir, dft_params=None, wannier_params=None, workflow_flags=None, window_params=None):
    '''
    Automates the workflow of SCF, NSCF, and Wannierization calculations.

    :param atoms: ASE Atoms object representing the atomic structure that will be used for the computation.
    :param seed: Seed name for output files.
    :param dir: Directory for output files.
    :param dft_params: Dictionary containing parameters for the DFT calculations.
    :param wannier_params: Dictionary containing parameters for the Wannierization calculations. Defaults to DEFAULT_WANNIER_PARAMS if not provided.
    :param workflow_flags: Dictionary containing flags to skip certain parts of the workflow. Defaults to DEFAULT_WORKFLOW_FLAGS if not provided.
    :param window_params: Dictionary containing parameters for the energy window calculations. Defaults to DEFAULT_WINDOW_PARAMS if not provided.
    '''
    dft_params     = {**DEFAULT_DFT_PARAMS,     **(dft_params     or {})}
    window_params  = {**DEFAULT_WINDOW_PARAMS,          **(window_params or {})}
    wannier_params = {**DEFAULT_WANNIER_PARAMS, **(wannier_params or {})}
    workflow_flags = {**DEFAULT_WORKFLOW_FLAGS, **(workflow_flags or {})}
    if not workflow_flags["skip_scf"]:
        scf(atoms,ecut=dft_params["ecut"],density_conv=dft_params["density_conv_scf"],seed=seed,dir=dir)
    if not workflow_flags["skip_nscf"]:
        nscf(nbands=dft_params["nbands"],unconverged_bands=dft_params["unconverged_bands"],seed=seed,dir=dir)
    proj_set, outer_win, frozen_win, nwann = get_proj_set(K=window_params["K"],seed=seed,dir=dir,dos_kwargs=window_params["dos_kwargs"])
    if not workflow_flags["skip_wannier"]:
        wannierize(proj_set, outer_win, frozen_win,seed=seed,dir=dir,**wannier_params)
    bands_wannier, wb_path = interpolate_bands(seed=seed,dir=dir,npoints=dft_params["npoints"])
    dft_bands(seed=seed,dir=dir,npoints=dft_params["npoints"],dft_nbands=dft_params["dft_plot_nbands"])
    plot_bands(bands_wannier, wb_path, outer_win, frozen_win,seed=seed,dir=dir)

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

    # build dicts from CLI args
    auto_workflow(atoms, seed,dir=args.output_dir,
        dft_params={
            "ecut": args.ecut,
            "nbands": args.nbands,
            "density_conv_scf": args.density_conv_scf,
            "unconverged_bands": args.unconverged_bands,
            "npoints": args.npoints,
            "dft_plot_nbands": args.dft_plot_nbands
        },
        window_params={
            "K": args.K,
            "dos_kwargs": {'spin': args.spin_channel, 'npts': args.npts_dos, 'width': args.dos_width}
        }
        ,
        wannier_params={
            "spin_channel": args.spin_channel,
            "unitary_params": dict(error_threshold=args.error_threshold, warning_threshold=args.warning_threshold, nbands_upper_skip=args.unconverged_bands),
            "wannierization_params": dict(num_iter=args.num_iter, conv_tol=args.w_conv_tol, print_progress_every=args.print_progress_every, sitesym=not args.no_sitesym, localise=not args.no_localise),
        },
        workflow_flags={
            "skip_scf": args.skip_scf,
            "skip_nscf": args.skip_nscf,
            "skip_wannier": args.skip_wannier,
        }
    )
if __name__ == "__main__":           
    main()
