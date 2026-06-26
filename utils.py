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
        '--emp-param', type=float, default=50,
        help='Empirical k-mesh density parameter (default: 50)'
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
        '--K', type=float, default=1.3,
        help='DOS integration factor for outer window (default: 1.3)'
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
    return parser.parse_args()