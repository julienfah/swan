import spglib
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
