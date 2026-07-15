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

"""
# 1. Find which operations fix (leave invariant, mod lattice translation) your target atom
def site_symmetry_ops(atom_frac_pos, rotations, translations, positions, tol=1e-5):
    ops = []
    for R, t in zip(rotations, translations):
        image = R @ atom_frac_pos + t
        if np.allclose((image - atom_frac_pos) % 1.0, 0.0, atol=tol) or \
           np.allclose((image - atom_frac_pos) % 1.0, 1.0, atol=tol):
            ops.append(R)
    return ops

site_ops_frac = site_symmetry_ops((0.5,0.5,0.5), rotations, translations, positions)

#print("Site symmetry operations (fractional):")
#for i, R in enumerate(site_ops_frac):
#    print(f"  {i+1}\n: {R}")

# 2. Convert each fractional rotation matrix to Cartesian
A = cell.T   # columns = lattice vectors in Cartesian; x_cart = A @ x_frac
site_ops_cart = [A @ R_frac @ np.linalg.inv(A) for R_frac in site_ops_frac]

print(get_P_S_axes(site_ops_cart))

#print(get_cluster(calc.atoms, calc.atoms.positions[1]))"""
"""
def rotation_axis(R):
    Axis of a proper rotation matrix (eigenvector with eigenvalue 1).
    w, v = np.linalg.eig(R)
    idx = np.argmin(np.abs(w - 1.0))
    axis = np.real(v[:, idx])
    return axis / np.linalg.norm(axis)

def rotation_order(R):
    n such that R is a Cn rotation, from trace(R) = 1 + 2cos(2*pi/n).
    cos_theta = (np.trace(R) - 1) / 2
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    if np.isclose(theta, 0):
        return 1  # identity
    return round(2 * np.pi / theta)
def get_P_S_axes(site_ops):
    proper_rots = [R for R in site_ops if np.isclose(np.linalg.det(R), 1) and not np.allclose(R, np.eye(3))]
    paxis_R = max(proper_rots, key=rotation_order)   # highest-order Cn
    paxis = rotation_axis(paxis_R)

    secondary_rots = [R for R in proper_rots if not np.isclose(abs(np.dot(rotation_axis(R), paxis)), 1.0)]
    saxis = None
    if secondary_rots:  # Dn family
        saxis = rotation_axis(secondary_rots[0])
    else:
        mirrors = [R for R in site_ops if np.isclose(np.linalg.det(R), -1) and rotation_order_of_reflection_check(R)]
        if mirrors:  # Cnv family
            normal = rotation_axis_eigval_minus1(mirrors[0])  # eigenvector w/ eigenvalue -1
            saxis = np.cross(paxis, normal)
        else:
            saxis = np.zeros(3)  # Cn, S2n, Cnh — let rotate_mol_to_symels pick arbitrarily
    return paxis, saxis"""

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
        # linear site: real hybridization exists (sp for 2 collinear bonds),
        # but Symtext.from_molecule crashes on it (mult_table=None bug) — build by hand
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
    #print(l_irrep_map)
    for label, count in get_salc_irreps(salcs.salcs).items():
        l_list = l_irrep_map[label]
        l_count = [l_count[i] + l_list.count(i) * count for i in range(3)]
    #print(f"\nTotal l counts: {l_count}")

    irrep_dim = {irrep.symbol: irrep.d for irrep in symtext.irreps}

    # how many *blocks* of each irrep the bonds actually need
    target_mult = {}
    for label, salc_count in get_salc_irreps(salcs.salcs).items():
        d = irrep_dim[label]
        assert salc_count % d == 0, f"{label}: {salc_count} SALCs not divisible by dim {d}"
        target_mult[label] = salc_count // d

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

    # enumerate every combination of choices across ambiguous irreps
    labels = list(options_per_irrep)
    combos = []
    for choice in itertools.product(*(options_per_irrep[lab] for lab in labels)):
        l_counts = [0, 0, 0]
        for lab, l in zip(labels, choice):
            l_counts[l] += irrep_dim[lab] * target_mult[lab]
        combos.append(tuple(l_counts))

    combos = sorted(set(combos))
    #print("Candidate (n_s, n_p, n_d) combinations:", combos)
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
            f"in reduction coefficient — real decomposition bug, not just rounding"
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
"""structure = AseAtomsAdaptor.get_structure(calc.atoms)
print("using pymatgen CrystalNN:")
print(get_cluster_crystalnn(structure, 2))
print("using ASE neighborlist:")
print(get_cluster(calc.atoms, calc.atoms.positions[2]))
"""
"""
#perform hybridization for each atom
selected_orbitals = [(0, (4,'s')), (1, (4,'s')), ( 2, (4,'s')), ( 3, (4,'s')), ( 0, (4,'p')), ( 1, (4,'p')), ( 2, (4,'p')), ( 3, (4,'p'))]
selected_orbitals_l_list = defaultdict(list) #convert to list of {iatom : list of all l values of iatom}
hybrydized_orbitals = [] 
for i, (n,l) in selected_orbitals:
    selected_orbitals_l_list[i].append(l)
for i, l_list in selected_orbitals_l_list.items():
    selected_hybridized_orbitals = hybridize_orbitals(calc.atoms, calc.atoms.positions[i], orbitals=l_list)
    hybrydized_orbitals.extend([(i, (4,l)) for l in selected_hybridized_orbitals])
print(f"Selected orbitals: {selected_orbitals}")
print(f"Hybridized orbitals: {hybrydized_orbitals}")"""

"""from itertools import permutations

def reference_lobes(hybrid_name):
    lobes = orbitals_sets_dic[hybrid_name]
    directions = []
    for lobe in lobes:
        coef = hybrids_coef[lobe]
        d = np.array([coef.get("px", 0), coef.get("py", 0), coef.get("pz", 0)])
        directions.append(d / np.linalg.norm(d))
    return np.array(directions)

def fit_orientation(hybrid_name, lobe_directions):
    ref = reference_lobes(hybrid_name)
    best = None
    for perm in permutations(range(len(ref))):
        A = lobe_directions[list(perm)]
        H = ref.T @ A
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        rmsd = np.linalg.norm(ref @ R.T - A)
        if best is None or rmsd < best[0]:
            best = (rmsd, R, perm)
    rmsd, R, perm = best
    return R @ np.array([0, 0, 1]), R @ np.array([1, 0, 0]), rmsd

lobe_directions = np.array([symtext.reverse_rotate @ d for d in lobe_directions_symtext])
zaxis, xaxis, rmsd = fit_orientation("sp3", lobe_directions)
"""
