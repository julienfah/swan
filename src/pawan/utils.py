import spglib
import argparse
from ase.io import read

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
        '--ecut', type=float, default=500,
        help='Plane-wave energy cutoff in eV (default: 500)'
    )
    parser.add_argument(
        '--nbands', type=int, default=40,
        help='Number of bands for NSCF calculation (default: 40)'
    )
    parser.add_argument(
        '--unconverged-bands', type=int, default=2,
        help='Number of unconverged bands for NSCF calculation (default: 2)'
    )
    parser.add_argument(
        '--dft-plot-nbands', type=int, default=14,
        help='Number of bands to plot for DFT band structure (default: 14)'
    )
    parser.add_argument(
        '--K', type=float, default=1.2,
        help='DOS integration factor for outer window (default: 1.3)'
    )
    parser.add_argument(
        '--npoints', type=int, default=200,
        help='Number of points for band interpolation (default: 200)'
    )
    parser.add_argument(
        '--density-conv-scf', type=float, default=1e-7,
        help='Density convergence criterion for SCF (default: 1e-7)'
    )
    parser.add_argument(
        '--num-iter', type=int, default=100,
        help='Maximum number of iterations for Wannierization (default: 100)'
    )
    parser.add_argument(
        '--w-conv-tol', type=float, default=1e-8,
        help='Convergence tolerance for Wannierization (default: 1e-8)'
    )
    parser.add_argument(
        '--no-sitesym', action='store_true',
        help='Do not use site symmetry during Wannierization'
    )
    parser.add_argument(
        '--no-localise', action='store_true',
        help='Do not localise Wannier functions after Wannierization'
    )
    parser.add_argument(
        '--spin-channel', type=int, default=0,
        help='Spin channel for Wannierization (default: 0)'
    )
    parser.add_argument(
        '--error-threshold', type=float, default=0.1,
        help='Error threshold for unitary matrix check (default: 0.1)'
    )
    parser.add_argument(
        '--warning-threshold', type=float, default=0.01,
        help='Warning threshold for unitary matrix check (default: 0.01)'
    )
    parser.add_argument(
        '--print-progress-every', type=int, default=20,
        help='Print progress every N iterations during Wannierization (default: 20)'
    )
    parser.add_argument(
        '--npts-dos', type=int, default=1001,
        help='Number of points for DOS calculation (default: 1001)'
    )
    parser.add_argument(
        '--dos-width', type=float, default=0.05,
        help='Width of Gaussian smearing for DOS calculation in eV (default: 0.05)'
    )
    parser.add_argument(
        '--skip-scf', action='store_true',
        help='Skip SCF if .gpw file already exists'
    )
    parser.add_argument(
        '--skip-nscf', action='store_true',
        help='Skip NSCF if .gpw file already exists'
    )
    parser.add_argument(
        '--skip-wannier', action='store_true',
        help='Skip Wannierization if .npz files already exist'
    )
    parser.add_argument(
        '--output-dir', default='test',
        help='Directory for output files (default: test)'
    )


    return parser.parse_args()