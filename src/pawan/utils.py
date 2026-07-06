import spglib
import argparse
from gpaw import GPAW
from gpaw.mpi import world
import numpy as np
from math import ceil,lcm
from fractions import Fraction
from wannierberri.symmetry.point_symmetry import PointGroup
from wannierberri.grid.grid import iterate_vector
from wannierberri.grid.grid import determineNK
from ase import Atoms


def get_crystal_system(atoms):
    '''returns the crystal system of the given atoms object based on its space group number.'''
    sg = spglib.get_symmetry_dataset(
        (atoms.cell, atoms.get_scaled_positions(), atoms.numbers)
    ).number
    if 1   <= sg <= 2:   return 'triclinic'
    if 3   <= sg <= 15:  return 'monoclinic'
    if 16  <= sg <= 74:  return 'orthorhombic'
    if 75  <= sg <= 142: return 'tetragonal'
    if 143 <= sg <= 167: return 'trigonal'
    if 168 <= sg <= 194: return 'hexagonal'
    if 195 <= sg <= 230: return 'cubic'

def standardize_cell(atoms):
    '''Standardizes the cell of the given atoms object using spglib.'''
    cell = spglib.standardize_cell(
        (atoms.cell, atoms.get_scaled_positions(), atoms.numbers),
        to_primitive=True,
        symprec=1e-3
    )
    if cell is None:
        raise ValueError("Failed to standardize the cell. Please check the input structure.")
    atoms = Atoms(numbers=cell[2], scaled_positions=cell[1], cell=cell[0], pbc=True)
    return atoms

def adaptative_nscf_nbands(seed,dir,calc=None,nbands=None,nbands_per_atom=None,n_bands_per_valence_el=4,lower_cap=18,higher_cap=200):
    if calc is None:
        calc = GPAW(f"{dir}/{seed}/{seed}-scf.gpw")
    atoms = calc.atoms
    if nbands is not None:
        return nbands
    if nbands_per_atom is not None:
        return nbands_per_atom * len(atoms)
    # if neither is provided, estimate based on number of valence electrons
    setups =calc.setups
    tot_val_bands =0#setups.nvalence
    for i,atom in enumerate(atoms):
        for n,l,f in zip(setups[i].n_j, setups[i].l_j, setups[i].f_j):
            if f>0 :tot_val_bands += 2*l+1
    result = int(tot_val_bands * n_bands_per_valence_el) # should we add a factor of K to relate to nwann? see tests
    if world.rank == 0:
        print(f"Estimated number of bands based on valence electrons: {result}. Will be capped between {lower_cap} and {higher_cap}.")

    return max(lower_cap, min(higher_cap, result))


def adaptative_k_grid(atoms,nk_length=40,multiplier=1):
    pg = PointGroup(real_lattice=atoms.cell.array.T)  # columns = lattice vectors
    periodic = np.array(atoms.pbc)
    NKdiv, NKFFT = determineNK(
        periodic=periodic,
        NKdiv=None, NKFFT=1, NK=None, length=nk_length,
        NKFFT_recommended =1,
        pointgroup=pg
    )
    return tuple(multiplier * NKdiv * NKFFT)

def adaptative_high_sym_k_grid(atoms,nk_length=40,multiplier=1):
    '''
    Creates a k-grid that fits the point group of the system that contains all high-symmetry points in the BZ, and is a multiple of the original one.
    '''
    special_points = atoms.cell.bandpath().special_points #dict letter: np.array([x,y,z]) in fractional coordinates
    multiples = []
    for letter, point in special_points.items():
        if not np.allclose(point, 0.0) and not np.allclose(point, 1.0):
            denominator_per_coordinate = [Fraction(coord).limit_denominator(10).denominator for coord in point]
            multiples.append(denominator_per_coordinate)
    multiples = np.lcm.reduce(multiples,axis=0) if multiples else np.array([1,1,1])
    print(f"LCM of denominators for special points: {multiples}")
    point_group_grid =adaptative_k_grid(atoms,nk_length=nk_length,multiplier=multiplier)#minimal_symmetric_kgrid(atoms)
    print(f"Automatically determined k-grid: {point_group_grid}")
    pg = PointGroup(real_lattice=atoms.cell.array.T)  # columns = lattice vectors
    #now test all grids btw the determined one and the one multiplied by the lcm of the denominators of the special points, and select the one with the smallest number of k-points that is compatible with the point group and contains all special points
    candidates = [i for i in iterate_vector(np.array(point_group_grid),np.array(point_group_grid)*multiples) if pg.symmetric_grid(i) and np.all(i % multiples == 0)]
    selected_grid = min(candidates, key=lambda x: np.prod(x))  # select the one with the smallest product (fewest k-points)
    print(f"Selected k-grid: {selected_grid}")
    return tuple(selected_grid)


def adaptative_g_grid(scf_calc,atoms):
    #create a dummy calculation to get the g-grid determined by gpaw from ecut
    scf_calc.initialize(atoms=atoms)
    auto_ggrid = scf_calc.wfs.gd.N_c
    if world.rank == 0:
        print(f"Automatically determined G-grid: {auto_ggrid}")
    #compare it to the symmetries of the space group
    sym_dataset = spglib.get_symmetry_dataset((atoms.cell, atoms.get_scaled_positions(), atoms.numbers))
    translations = np.unique(sym_dataset.translations,axis=0)
    #make the g-grid compatible with the translations of the space group if non symmorphic
    symmorphic = (len(translations) == 1 and np.allclose(translations, 0, atol=1e-3))
    if not symmorphic:
        if world.rank == 0:
            print("The space group is non-symmorphic. Adjusting G-grid to be compatible with the translations.")
        mins = [1, 1, 1]
        for translation in translations:
            if np.allclose(translation, 0, atol=1e-3): continue
            for i,t_i in enumerate(translation):
                if t_i != 0:
                    # Ensure that the G-grid is a multiple of the translation vector
                    frac = Fraction(t_i).limit_denominator(20)
                    if frac.denominator > 1:
                        mins[i] = lcm(mins[i], frac.denominator)
        # get GPAW's natural grid and round up to satisfy both ecut and symmetry
        result = tuple(
            next_fft_friendly(mins[i], auto_ggrid[i])
            for i in range(3)
        )
        if world.rank == 0:
            print(f"Adjusted G-grid to satisfy both ecut and symmetry: {result}")
    else:
        result = auto_ggrid

    return result

def next_fft_friendly(n, min_n):
    """Find smallest number >= min_n that is divisible by n (symmetry) and FFT-friendly."""
    from sympy import factorint
    candidate = ceil(min_n / n) * n
    while True:
        factors = set(factorint(candidate).keys())
        if factors <= {2, 3, 5}:
            return candidate
        candidate += n            




def find_emax_from_dos(energies, dos_total, emin, n_wann, K=1.2):
    de = energies[1] - energies[0]
    target = K * n_wann
    print(f"Target number of states: {target}")
    cumulative = 0.0
    for e, dos in zip(energies, dos_total):
        if e < emin:
            continue
        cumulative += dos * de
        if cumulative >= target:
            print(f"Cumulative: {cumulative}")
            return e
    print(f"Cumulative: {cumulative}")
    return None  # need more bands

def parse_args():
    parser = argparse.ArgumentParser(
        description='Automatic Wannier function pipeline for a given crystal structure.'
    )
    parser.add_argument(
        'structure',
        help='Path to structure file (any ASE-readable format: cif, vasp, xyz, ...)'
    )
    parser.add_argument(
        '--seed', default=None,
        help='Seed name for output files (default: chemical formula)'
    )
    parser.add_argument(
        '--auto-nk-grid', action='store_true', default=None, dest="auto_nk_grid",
        help='Automatically determine the k-point grid based on the crystal structure. Overrides --nk if set'
    )
    parser.add_argument(
        '--nk', type=int, default=None,
        help='Number of k-points in each direction for SCF and NSCF calculations'
    )
    parser.add_argument(
        '--nkfft', type=int, default=None,
        help='Number of k-points in each direction for FFT grid'
    )
    parser.add_argument(
        '--ecut', type=float, default=None,
        help='Plane-wave energy cutoff in eV (default: 500)'
    )
    parser.add_argument(
        '--gap-thres', type=float, default=None, dest="gap_thres",
        help='Minimum size of a gap to be considered as such in the initial window determination. Can be increased to include lower semi-core bands (default: 0.1 eV)'
    )
    parser.add_argument(
        '--nbands', type=int, default=None,
        help='Number of bands for NSCF calculation. Overrides --nbdands_per_atom and --nbands_per_valence_el'
    )
    parser.add_argument(
        '--nbands-per-atom', type=int, default=None, dest="nbands_per_atom",
        help='Number of bands per atom for NSCF calculation. Overrides --nbands_per_valence_el'
    )
    parser.add_argument(
        '--nbands-per-valence-el', type=int, default=None, dest="nbands_per_valence_el",
        help='Number of bands per valence electron for NSCF calculation (default: 5)'
    )
    parser.add_argument(
        '--unconverged-bands', type=int, default=None, dest="unconverged_bands",
        help='Number of unconverged bands for NSCF calculation (default: 2)'
    )
    parser.add_argument(
        '--dft-plot-nbands', type=int, default=None, dest="dft_plot_nbands",
        help='Number of bands to plot for DFT band structure (default: 14)'
    )
    parser.add_argument(
        '--K', type=float, default=None,
        help='DOS integration factor for outer window (default: 1.2)'
    )
    parser.add_argument(
        '--npoints', type=int, default=None,
        help='Number of points for band interpolation (default: 200)'
    )
    parser.add_argument(
        '--density-conv-scf', type=float, default=None, dest="density_conv_scf",
        help='Density convergence criterion for SCF (default: 1e-7)'
    )
    parser.add_argument(
        '--num-iter', type=int, default=None, dest="num_iter",
        help='Maximum number of iterations for Wannierization (default: 100)'
    )
    parser.add_argument(
        '--w-conv-tol', type=float, default=None, dest="w_conv_tol",
        help='Convergence tolerance for Wannierization (default: 1e-8)'
    )
    parser.add_argument(
        '--no-sitesym', action='store_true', default=None, dest="no_sitesym",
        help='Do not use site symmetry during Wannierization'
    )
    parser.add_argument(
        '--no-localise', action='store_true', default=None, dest="no_localise",
        help='Do not localise Wannier functions after Wannierization'
    )
    parser.add_argument(
        '--spin-channel', type=int, default=None, dest="spin_channel",
        help='Spin channel for Wannierization (default: 0)'
    )
    parser.add_argument(
        '--error-threshold', type=float, default=None, dest="error_threshold",
        help='Error threshold for unitary matrix check (default: 0.1)'
    )
    parser.add_argument(
        '--warning-threshold', type=float, default=None, dest="warning_threshold",
        help='Warning threshold for unitary matrix check (default: 0.01)'
    )
    parser.add_argument(
        '--print-progress-every', type=int, default=None, dest="print_progress_every",
        help='Print progress every N iterations during Wannierization (default: 20)'
    )
    parser.add_argument(
        '--npts-dos', type=int, default=None, dest="npts_dos",
        help='Number of points for DOS calculation (default: 1001)'
    )
    parser.add_argument(
        '--dos-width', type=float, default=None, dest="dos_width",
        help='Width of Gaussian smearing for DOS calculation in eV (default: 0.05)'
    )
    parser.add_argument(
        '--skip-scf', action='store_true', default=None, dest="skip_scf",
        help='Skip SCF if .gpw file already exists'
    )
    parser.add_argument(
        '--skip-nscf', action='store_true', default=None, dest="skip_nscf",
        help='Skip NSCF if .gpw file already exists'
    )
    parser.add_argument(
        '--skip-wannier', action='store_true', default=None, dest="skip_wannier",
        help='Skip Wannierization if .npz files already exist'
    )
    parser.add_argument(
        '--output-dir', default=None, dest="output_dir",
        help='Directory for output files (default: test)'
    )
    

    return parser.parse_args()