from molsym.symtext.symtext import Symtext
from molsym.molecule import Molecule
import molsym
import warnings
import numpy as np
from ase.neighborlist import NeighborList,natural_cutoffs
from ase.data import atomic_masses,atomic_numbers
from collections import defaultdict
import itertools

from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.ase import AseAtomsAdaptor

def hybridize(atoms,position):
    """
    Hybridize the orbitals according to the point group symmetry.

    :param atoms: The atoms in the system.
    :param position: The position of the atom for which to hybridize orbitals.
    :return: The list of hybridized orbitals.
    """
    print(f"Attempting to hybridize orbitals for atom {atoms[np.where(np.all(np.isclose(atoms.get_positions(), position, atol=1e-5), axis=1))[0][0]].symbol} at position {position}")
    #cluster_coords, cluster_symbols = get_cluster(atoms, position)
    #use pymatgen's CrystalNN to find the nearest neighbors, as works also for ionic bonds
    cluster_coords, cluster_symbols = get_cluster_crystalnn(AseAtomsAdaptor.get_structure(atoms), np.where(np.all(np.isclose(atoms.get_positions(), position, atol=1e-5), axis=1))[0][0])
    masses = [atomic_masses[atomic_numbers[symbol]] for symbol in cluster_symbols]
    mol = Molecule(atoms=cluster_symbols, coords=cluster_coords, masses=masses)
    pg_str, _ = molsym.find_point_group(mol)
    print(pg_str)
    print(len(cluster_symbols) - 1, cluster_symbols[1:], [np.linalg.norm(c) for c in cluster_coords[1:]])

    n_neighbors = len(cluster_symbols) - 1
    if n_neighbors < 2:
        warnings.warn(f"{cluster_symbols[0]} at {position} has {n_neighbors} neighbor(s) — "
                       f"hybridization undefined for a single bond direction; "
                       f"using unhybridized orbitals.")
        return []
    if pg_str.startswith(("C0v", "D0h")):   # MolSym's linear-group labels
        # linear site: real hybridization exists (sp for 2 collinear bonds), but not handled by molsym
        return handle_linear_site(pg_str)
    symtext = Symtext.from_molecule(mol)
    ic_coords = [ [0,i] for i in range(1, len(cluster_symbols)) ]#define coords as bonds between central atom and neigbors

    ic_names = [f"R{i}" for i in range( len(ic_coords))]  # name the internal coordinates as R1, R2, ...

    ic_list = [
        [ic_coords[i], ic_names[i]]
        for i in range(len(ic_coords))
    ]
    ics = molsym.salcs.InternalCoordinates(symtext=symtext, fxn_list=ic_list)
    salcs = molsym.salcs.ProjectionOp(symtext, ics)
    print("SALCs:", salcs)
    NAME_BY_COUNTS = {
    (1, 1, 0): "sp",
    (1, 2, 0): "sp2",
    (1, 3, 0): "sp3",
    (1, 3, 1): "sp3d",
    (1, 3, 2): "sp3d2",
    }

    #match irreps to number of each l orbital used
    l_count =(0,0,0)#s,p,d
    l_irrep_map = defaultdict(list)
    for l in [0,1,2]:#orbitals parameter
        decomp = decompose_shell(symtext, l)
        for irrep, count in decomp.items():
            l_irrep_map[irrep].append(l)
    print(l_irrep_map)
    for label, count in get_salc_irreps(salcs.salcs).items():
        l_list = l_irrep_map[label]
        l_count = [l_count[i] + l_list.count(i) * count for i in range(3)]
    print(f"\nTotal l counts: {l_count}")

    irrep_dim = {irrep.symbol: irrep.d for irrep in symtext.irreps}

    # how many *blocks* of each irrep the bonds actually need
    target_mult = {}
    for label, salc_count in get_salc_irreps(salcs.salcs).items():
        d = irrep_dim[label]
        assert salc_count % d == 0, f"{label}: {salc_count} SALCs not divisible by dim {d}"
        target_mult[label] = salc_count // d
    print(f"Target irrep multiplicities: {target_mult}")
    # for each needed irrep, which l's can supply a block, and how many blocks that l offers
    options_per_irrep = {}
    for label, mult_needed in target_mult.items():
        candidates = [l for l in (0, 1, 2) if l_irrep_map[label].count(l) >= mult_needed]
        if not candidates:
            print(l_list,l_count)

            warnings.warn(f"No single shell supplies {mult_needed}x {label} — needs mixing within a shell. No hybridization perfomed")
            return []   #no possible simple hybrids
            #raise ValueError(f"No single shell supplies {mult_needed}x {label} — needs mixing within a shell, not handled here")
        options_per_irrep[label] = candidates
    print(f"Options per irrep: {options_per_irrep}")
    # enumerate every combination of choices across ambiguous irreps
    labels = list(options_per_irrep)
    combos = []
    for choice in itertools.product(*(options_per_irrep[lab] for lab in labels)):
        l_counts = [0, 0, 0]
        for lab, l in zip(labels, choice):
            l_counts[l] += irrep_dim[lab] * target_mult[lab]
        combos.append(tuple(l_counts))

    combos = sorted(set(combos))
    print("Candidate (n_s, n_p, n_d) combinations:", combos,"\n")
    supported_hybrids = []
    for counts in combos:
        name = NAME_BY_COUNTS.get(counts)
        if name:
            supported_hybrids.append(name)

    #raise NotImplementedError("Hybridization of orbitals is not implemented yet.")
    return supported_hybrids

def hybridize_orbitals(atoms, position, orbitals):
    """
    Hybridize the orbitals according to the point group symmetry.

    :param atoms: The atoms in the system.
    :param position: The position of the atom for which to hybridize orbitals.
    :param orbitals: List of orbitals to hybridize (0=s, 1=p, 2=d).
    :return: The list of hybridized orbitals.
    """
    supported_hybrids = hybridize(atoms, position)
    print(f"Supported hybrids: {supported_hybrids}")
    # Filter supported hybrids based on requested orbitals
    HYBRID_SHELLS = {
        "s": {"s"}, "p": {"p"}, "d": {"d"}, "f": {"f"},
        "sp": {"s", "p"}, "sp2": {"s", "p"}, "sp3": {"s", "p"},
        "sp3d2": {"s", "p", "d"},
        "p2": {"p"}, "pxy": {"p"}, "pz": {"p"},
        "t2g": {"d"}, "eg": {"d"},
    }
    filtered_hybrids = [h for h in supported_hybrids if HYBRID_SHELLS[h] <= set(orbitals)]
    HYBRID_LEFTOVER = {
        "sp2": "pz",
        "sp": "p2",
        "sp3d2": "t2g",
    }
    print(f"Filtered hybrids based on requested orbitals: {filtered_hybrids}")
    leftover_hybrids = []
    consumed_shells = set()

    for hybrid in filtered_hybrids:
        consumed_shells |= HYBRID_SHELLS[hybrid]
        if hybrid in HYBRID_LEFTOVER:
            leftover = HYBRID_LEFTOVER[hybrid]
            leftover_hybrids.append(leftover)
    all_hybrids = filtered_hybrids + leftover_hybrids
    if not all_hybrids:
        all_hybrids = orbitals
    untouched_shells = set(orbitals) - consumed_shells
    all_hybrids += sorted(untouched_shells)
    return all_hybrids
def get_cluster_crystalnn(structure, site_index):
    cnn = CrystalNN()
    nn_info = cnn.get_nn_info(structure, site_index)
    center = structure[site_index].coords
    coords = [np.zeros(3)]
    symbols = [structure[site_index].specie.symbol]
    for nn in nn_info:
        coords.append(nn["site"].coords - center)   # already includes periodic image offset
        symbols.append(nn["site"].specie.symbol)
    return coords, symbols
def get_cluster(atoms,position):
    """
    Find neighboring atoms of a given Wyckoff position in a crystal structure using ASE's NeighborList.
    WARNING: At the moment we assume the WP sits on an atom, and we find the neighbors of that atom ->needs to be generalized if works.
    """
    cutoffs = natural_cutoffs(atoms, mult=1.0)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    positions = atoms.get_positions()
    #print("Atoms positions:")
    #print(f"{atoms.get_positions()} vs {position}")
    site_index = np.where(np.all(np.isclose(positions, position, atol=1e-5), axis=1))[0][0]#np.argwhere(np.isclose(atoms.get_positions(), position, atol=1e-5))
    #print(site_index)
    indices, offsets = nl.get_neighbors(site_index)

    center = atoms.positions[site_index]

    coords = [np.zeros(3)]
    symbols = [atoms[site_index].symbol]

    for i, offset in zip(indices, offsets):
        pos = atoms.positions[i] + offset @ atoms.cell
        coords.append(pos - center)
        symbols.append(atoms[i].symbol)
    return coords, symbols
def get_salc_irreps(salcs):
    """
    Extract irrep multiplicities from MolSym SALCs
    """

    irreps = {}

    for salc in salcs:
        irrep = salc.irrep.symbol

        if irrep not in irreps:
            irreps[irrep] = 0

        irreps[irrep] += 1

    return irreps
def shell_character(R, l):
    det = np.linalg.det(R)
    tr = np.trace(R)
    cos_theta = (tr - 1) / 2 if det > 0 else -(tr + 1) / 2
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    base = (2*l + 1) if np.isclose(theta, 0) else np.sin((2*l+1)*theta/2) / np.sin(theta/2)
    return base if det > 0 else ((-1)**l) * base
"""def decompose_shell(symtext, l):
    class_reps = [symtext.symels[symtext.symel_to_class_map.index(c)] for c in range(len(symtext.classes))]
    chars = np.array([shell_character(s.rrep, l) for s in class_reps])
    mults = symtext.reduction_coefficients(chars)
    return {irrep.symbol: m for irrep, m in zip(symtext.irreps, mults) if m > 0}"""
def decompose_shell(symtext, l):
    class_reps = [symtext.symels[symtext.symel_to_class_map.index(c)] for c in range(len(symtext.classes))]
    chars = np.array([shell_character(s.rrep, l) for s in class_reps])

    mults = np.zeros(len(symtext.irreps), dtype=int)
    for irrep_idx, irrep in enumerate(symtext.irreps):
        p = np.multiply(chars, symtext.class_orders)
        p = np.multiply(p, symtext.character_table[irrep_idx, :])
        raw = p.sum() / symtext.order
        assert abs(raw.imag) < 1e-6, (
            f"l={l}, irrep={irrep.symbol}: non-negligible imaginary part {raw.imag} "
            f"in reduction coefficient -> bug in character table or shell_character"
        )
        mults[irrep_idx] = round(raw.real)

    result = {irrep.symbol: m for irrep, m in zip(symtext.irreps, mults) if m > 0}
    total_dim = sum(m * irrep.d for m, irrep in zip(mults, symtext.irreps))
    assert total_dim == 2*l + 1, f"l={l} decomposition has dimension {total_dim}, expected {2*l+1}"
    return result
def handle_linear_site(pg_str):
    """
    Fallback for linear 2-neighbor site (failed in MolSym).
    Dinfh (two identical neighbors, e.g. O between two equivalent Re)  -> sp
    Cinfv (two different neighbors, asymmetric)                        -> no shared hybrid
    """
    if pg_str.startswith("D"):
        return ["sp"]
    else:
        warnings.warn("C∞v site — asymmetric linear bonding, no shared hybrid; "
                       "using unhybridized orbitals.")
        return []
#test

#calc = GPAW("test/ClNa/ClNa-nscf-irred.gpw", txt=None)


#print(hybridize(calc.atoms, calc.atoms.positions[0]))
#print(hybridize_orbitals(calc.atoms, calc.atoms.positions[0], orbitals=["s", "p","d"]))

