import numpy as np
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

from pawan.utils import parse_args, adaptative_nscf_nbands,adaptative_k_grid,adaptative_g_grid
from pawan.auto_proj_and_windows import get_proj_set

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

def scf(atoms,seed,dir,ecut=500,density_conv=1e-7,NK=12,NKFFT=1):
    '''
    Perform self-consistent field calculation for the given atoms. 

    :param atoms: ASE Atoms object representing the atomic structure that will be used for the computation.
    :param seed: Seed name for output files.
    '''
    print("Running self-consistent calculation")
    kx,ky,kz = compute_nscf_kmesh(atoms,NKFFT,NK)  #compute_kmesh(atoms,emp_param=30)
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
    print("Running non-self-consistent calculation")
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
   

def wannierize(proj_set, outer_win, frozen_win, seed,dir,spin_channel=0,unitary_params=dict(error_threshold=0.1,warning_threshold=0.01,nbands_upper_skip=2),wannierization_params=dict(num_iter=100,conv_tol=1e-8,print_progress_every=20,sitesym=True,localise=True,)):
    '''
    Proper wannierization of the system, using the previously determined parameters.

    :param proj_set: ProjectionsSet object containing the projections to be used for the wannierization.
    :param outer_win: Tuple containing the outer energy window boundaries.
    :param frozen_win: Tuple containing the frozen energy window boundaries.
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

    wandata.wannierise(
        froz_min=frozen_win[0],    
        froz_max=frozen_win[1],  
        outer_min=outer_win[0],
        outer_max=outer_win[1],# np.inf,#
        **wannierization_params
    )
    wandata.chk.to_npz(f"{dir}/{seed}/{seed}_wannier_data.chk.npz")

def interpolate_bands(seed,dir,npoints=200):
    '''
    Use of the Wannier functions to interpolate the bands.
    '''
    calc_nscf_irred = GPAW(f'{dir}/{seed}/{seed}-nscf-irred.gpw', txt=None)
    atoms = calc_nscf_irred.atoms
    path = atoms.cell.bandpath()

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
    '''
    if Path_(f"{dir}/{seed}/{seed}-bands.gpw").exists():
        print(f"DFT bands already computed for {seed}. Skipping.")
        return
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

def auto_workflow(
    atoms, 
    seed, 
    dir="test", 
    nk=12,
    nkfft=1,
    ecut=500.0, 
    density_conv_scf=1e-7,
    gap_thres = 0.1,
    nbands_per_valence_el=4,
    nbands_per_atom=None, 
    nbands=None, 
    unconverged_bands=2, 
    npoints=200, 
    dft_plot_nbands=None,
    K=1.2, 
    spin_channel=0, 
    npts_dos=1001, 
    dos_width=0.05,
    num_iter=100, 
    w_conv_tol=1e-8, 
    print_progress_every=20, 
    no_sitesym=False, 
    no_localise=False,
    error_threshold=0.1, 
    warning_threshold=0.01,
    skip_scf=False, 
    skip_nscf=False, 
    skip_wannier=False
):
    if not skip_scf:
        scf(atoms,ecut=ecut,density_conv=density_conv_scf,seed=seed,dir=dir,NK=nk,NKFFT=nkfft)
    
    n_bands = adaptative_nscf_nbands(seed=seed,dir=dir,nbands_per_atom=nbands_per_atom,nbands=nbands,n_bands_per_valence_el=nbands_per_valence_el)
    if not skip_nscf:
        nscf(nbands=n_bands,unconverged_bands=unconverged_bands,seed=seed,dir=dir,NK=nk,NKFFT=nkfft)
        
    dos_kwargs = {'spin': spin_channel, 'npts': npts_dos, 'width': dos_width}
    proj_set, outer_win, frozen_win, nwann = get_proj_set(K=K,seed=seed,dir=dir,dos_kwargs=dos_kwargs,gap_thres=gap_thres)
    
    if not skip_wannier:
        unitary_params = dict(error_threshold=error_threshold, warning_threshold=warning_threshold, nbands_upper_skip=unconverged_bands)
        wannierization_params = dict(num_iter=num_iter, conv_tol=w_conv_tol, print_progress_every=print_progress_every, sitesym=not no_sitesym, localise=not no_localise)
        wannierize(proj_set=proj_set, outer_win=outer_win, frozen_win=frozen_win,seed=seed,dir=dir,spin_channel=spin_channel, unitary_params=unitary_params, wannierization_params=wannierization_params)
        
    bands_wannier, wb_path = interpolate_bands(seed=seed,dir=dir,npoints=npoints)

    dft_plot_nbands = dft_plot_nbands if dft_plot_nbands is not None else n_bands
    print(f"Using {dft_plot_nbands} bands for DFT band structure plot.")
    dft_bands(seed=seed,dir=dir,npoints=npoints,dft_nbands=dft_plot_nbands)
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
    output_directory = args.output_dir if args.output_dir is not None else "test"

    (Path_(output_directory)/Path_(f"{seed}")).mkdir(parents=True, exist_ok=True)

    # build dicts from CLI args
    cli_kwargs = {k: v for k, v in vars(args).items() if v is not None and k not in ['structure', 'seed', 'output_dir']}

    auto_workflow(atoms, seed, dir=output_directory, **cli_kwargs)

if __name__ == "__main__":           
    main()
    