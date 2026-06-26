import numpy as np
from math import ceil
from ase import Atoms
from gpaw import GPAW, PW, MixerSum
from ase.spacegroup import get_spacegroup
from irrep.spacegroup import SpaceGroup
from wannierberri.symmetry.projections import Projection, ProjectionsSet
from wannierberri.w90files import WannierData
from wannierberri import System_R,Path,evaluate_k_path
from matplotlib import pyplot as plt
from ase.build import bulk
from pathlib import Path as Path_
from collections import defaultdict
from ase.io import read
import spglib


#import ray
#ray.init(num_cpus=18,num_gpus=20,ignore_reinit_error=True)

from utils import get_crystal_system, find_emax_from_dos,parse_args

def compute_kmesh(atoms, emp_param=50):
    '''
    Compute the k-point mesh based on the cell parameters of the given atoms, fixing a certain density of k-points in reciprocal space.
    
    :param atoms: ASE Atoms object representing the atomic structure that will be used for the computation.
    :param emp_param: Empirical parameter to determine the density of k-points in reciprocal space. Default is 50 (from Zhang).
    '''
    ax,ay,az,alpha,beta,gamma = atoms.cell.cellpar()

    hexagonal = get_crystal_system(atoms) in ('hexagonal', 'trigonal')
    #fixed density of k-points

    kx,ky,kz = int(emp_param/ax),int(emp_param/ay),int(emp_param/az)
    print(f"Initial k-mesh: {kx}x{ky}x{kz}")
    enforced_multiple = 3 if hexagonal else 2
    kx,ky,kz = ceil(kx / enforced_multiple) * enforced_multiple,ceil(ky / enforced_multiple) * enforced_multiple,ceil(kz / enforced_multiple) * enforced_multiple
    print(f"Computed k-mesh: {kx}x{ky}x{kz}")
    return kx,ky,kz #should we enforce a certain symmetry in the k-mesh?

def scf(atoms):
    '''
    Perform self-consistent field calculation for the given atoms. 

    :param atoms: ASE Atoms object representing the atomic structure that will be used for the computation.
    '''
    print("Running self-consistent calculation")
    kx,ky,kz = compute_kmesh(atoms,emp_param=30)
    grid = [kx,ky,kz]
    calc = GPAW(
        mode=PW(500), 
        xc="PBE",
        kpts={"size": grid, "gamma": True},
        convergence={"density": 1e-7},
        mixer=MixerSum(0.25, 8, 100),
        txt=f"test/{seed}/{seed}-scf.txt"
    )

    atoms.calc = calc

    atoms.get_potential_energy()
    calc.write(f"test/{seed}/{seed}-scf.gpw", mode="all")
    

def nscf(nbands=40):
    '''
    Perform non-self-consistent field calculation, reading from the output of the SCF calculation.    
    '''
    print("Running non-self-consistent calculation")
    calc = GPAW(f'test/{seed}/{seed}-scf.gpw', txt=None)
    nscf_grid = compute_kmesh(calc.atoms,emp_param=45)
    space_group = SpaceGroup.from_gpaw(calc)
    irred_k_points = space_group.get_irreducible_kpoints_grid(nscf_grid)
    calc_nscf_irred = calc.fixed_density(
        kpts=irred_k_points,
        nbands=nbands,
        convergence={"bands": nbands-2},
        txt=f'test/{seed}/{seed}-nscf-irred.txt')
    calc_nscf_irred.write(f'test/{seed}/{seed}-nscf-irred.gpw', mode='all')


def find_projections():
    '''
    First step to find the projections, using the occupied valence orbitals of the atoms in the system. 

    To be replaced by a more involved method, e.g. Zhang's pDOS algorithm.

    :return: ProjectionsSet object containing the projections to be used for the wannierization.
    '''
    calc = GPAW(f'test/{seed}/{seed}-scf.gpw', txt=None)
    setups =calc.setups
    l_conversion = {0:'s',1:'p',2:'d',3:'f'}
    ls = []
    for setup in setups:
        occupied = [l for n, l, f in zip(setup.n_j, setup.l_j, setup.f_j) if f > 0]#add if occupied and not already in the list
        ls.extend([l_conversion[l] for l in occupied])
    #remove duplicates
    ls = list(dict.fromkeys(ls))
    print(f"Found the following valence orbitals: {ls}")
    

    space_group = SpaceGroup.from_gpaw(calc)

    projs = []
    '''for l in ls:
        for pos in space_group.positions:
            proj = Projection(
                position_num=[pos],
                orbital=l,
                spacegroup=space_group,
                rotate_basis=True
            )
            projs.append(proj)'''
    # group atoms by species
    species_positions = defaultdict(list)
    for atom, pos in zip(calc.atoms, space_group.positions):
        species_positions[atom.symbol].append(pos)

    for l in ls:
        for symbol, positions in species_positions.items():
            proj = Projection(
                position_num=positions,
                orbital=l,
                spacegroup=space_group,
                rotate_basis=True
            )
            projs.append(proj)
    

    proj_set = ProjectionsSet(projections=projs)
    return proj_set,ls

def find_energy_wdw(ls):
    '''
    Find both the outer and frozen energy windows for the Wannierization process, using DOS from the SCF calculation (Zhang method).
    '''
    selected_orbitals = ls
    calc = GPAW(f'test/{seed}/{seed}-nscf-irred.gpw', txt=None)
    e_fermi = calc.get_fermi_level()
    energies, dos_total = calc.get_dos(spin=0, npts=1001, width=0.05)
    print(f"Fermi level: {e_fermi} eV")
    pdos={}
    for iatom in range(len(calc.atoms)):
        for l in selected_orbitals:  # s, p, d
            e, dos = calc.get_orbital_ldos(a=iatom, angular=l, npts=1001, width=0.05)
            pdos[(iatom, l)] = dos

    emin_list, emax_list = [], []
    threshold = 1e-4  # Threshold for considering a DOS value as nonzero

    for (iatom, l), dos in pdos.items():
    
        nonzero = np.where(dos > threshold)[0]
        if len(nonzero) == 0:
            continue
        emin_list.append(energies[nonzero[0]])
        emax_list.append(energies[nonzero[-1]])
    #print(f"actual min energy: {min(emin_list)} eV, actual max energy: {max(emax_list)} eV")
    #print(energies[0], energies[-1])
    #print(dos_total[energies > e_fermi][:10])
    #optimize the upper limit of the outer window based on the number of wannier functions and the DOS
    l_num = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    n_wann = sum(2*l_num[l]+1 for l in selected_orbitals) * len(calc.atoms)
    emax = find_emax_from_dos(energies, dos_total, min(emin_list), n_wann, K=1.3)
    emin = min(emin_list)
    if emax is None:
        raise ValueError("E_max not found — increase nbands in NSCF")

    eigs = np.array([calc.get_eigenvalues(kpt=k) for k in range(len(calc.get_ibz_k_points()))])
    # ensure at least n_wann bands are fully within the window at every k-point (DOS is averaged over k-points, so it may cause a too small outer window if the bands are very entangled)
    for k_eigs in eigs:
        in_window = k_eigs[(k_eigs >= emin) & (k_eigs <= emax)]
        if len(in_window) < n_wann:
            emax = max(emax, k_eigs[n_wann])  # extend to include the n_wann-th band
            print(f"Adjusted emax to {emax} eV to include at least {n_wann} bands at any k-point.")

    out_win = (emin, emax)

    print(f"Found the following energy window: {out_win[0]} eV to {out_win[1]} eV")

    #frozen window : Zhang suggests from bottom of outer window to 2 eV above Fermi level
    frozen_win = (out_win[0], e_fermi +2)
    return out_win,frozen_win #outer and frozen window    

def wannierize(proj_set, outer_win, frozen_win):
    '''
    Proper wannierization of the system, using the previously determined parameters.

    :param proj_set: ProjectionsSet object containing the projections to be used for the wannierization.
    :param outer_win: Tuple containing the outer energy window boundaries.
    :param frozen_win: Tuple containing the frozen energy window boundaries.
    '''
    print(frozen_win[0], frozen_win[1], outer_win[0], outer_win[1])

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
def interpolate_bands():
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





def plot_bands(bands_wannier,wb_path,outer_win,frozen_win):
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

def dft_bands():
    calc = GPAW(f"test/{seed}/{seed}-scf.gpw")
    # compute the band directly from gpaw for comparison
    path = atoms.cell.bandpath(npoints=100)
    print(path)
    dft_calc_bands = calc.fixed_density(
        nbands=14,
        symmetry='off',
        kpts=path,#{'path': list(path.values()), 'npoints': 100},
        convergence={'bands': 8},
        txt=f"test/{seed}/{seed}-bands.txt")
    dft_calc_bands.write(f"test/{seed}/{seed}-bands.gpw", mode="all")

def auto_workflow(atoms, args):
    if not args.skip_scf:
        scf(atoms)
    if not args.skip_nscf:
        nscf(nbands=args.nbands)
    proj_set, ls = find_projections()
    outer_win, frozen_win = find_energy_wdw(ls)
    if not args.skip_wannier:
        wannierize(proj_set, outer_win, frozen_win)
    bands_wannier, wb_path = interpolate_bands()
    dft_bands()
    plot_bands(bands_wannier, wb_path, outer_win, frozen_win)
if __name__ == "__main__":           
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
    global seed

    seed = atoms.get_chemical_formula()
    Path_(f"test/{seed}").mkdir(parents=True, exist_ok=True)

    auto_workflow(atoms, args)
    
