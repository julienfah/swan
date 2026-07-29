"""
salc.py -- symmetry-adapted projections for Wannierisation, in one module.

Input:  ase.Atoms + orbital shells per element (e.g. {'Mn': 'spd', 'Te': 'p'}).
Output: a WannierBerri ProjectionsSet whose orbitals
        - are invariant under the site-symmetry group of their Wyckoff position
          (guaranteed, checked), and
        - where symmetry + bond geometry determine the hybrids uniquely, are
          bond-pointing lobes that PERMUTE onto one another under the site group
          (sp3 in diamond, sp3d2 in perovskites, ... -- derived, not tabulated);
        - elsewhere, are the isotypic components (irrep-pure blocks), keeping
          each multiplicity space whole so nothing seed-dependent is frozen in.

Uses WannierBerri's own orbital-rotation code (rot_orb_basis) for all D
matrices, and registers results into its hybrids_coef machinery, so the
Projections work in write-ups, symmetrizers and AMN generation unchanged.

No zaxis/xaxis handling is needed: everything is built in the crystal Cartesian
frame, and with rotate_basis=True WannierBerri generates the per-site frames
from the orbit itself (verified: the residual rotation always lands in G_q of
the representative).

Frame convention note: hybrid lobe DIRECTIONS returned here are crystal-
Cartesian. Weights (s:p:d ratios) are frame-independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import spglib
from scipy.linalg import block_diag, null_space, sqrtm

from wannierberri.symmetry import orbitals as wb_orb

L_OF = {"s": 0, "p": 1, "d": 2, "f": 3}
NAME_OF = {v: k for k, v in L_OF.items()}


def parse_shells(spec):
    """Normalise a shell spec to a sorted, deduplicated list of l values.

    Accepts 'spd', ['s','p'], [0,1,2], ('p','d'), mixed ['s',1], or a bare
    int / single letter. Duplicates are collapsed: this module treats shells
    by angular momentum only (radial character / principal quantum number is
    a radial_nodes question, outside symmetry).
    """
    if isinstance(spec, (int, np.integer)):
        spec = [spec]
    ls = set()
    for c in spec:  # a str iterates into letters, a list into its items
        if isinstance(c, str):
            if c not in L_OF:
                raise ValueError(f"unknown shell {c!r} (expected s/p/d/f)")
            ls.add(L_OF[c])
        elif isinstance(c, (int, np.integer)) and 0 <= c <= 3:
            ls.add(int(c))
        else:
            raise ValueError(f"cannot interpret shell entry {c!r}")
    if not ls:
        raise ValueError("empty shell specification")
    return sorted(ls)

# --------------------------------------------------------------------------
# geometry / symmetry primitives
# --------------------------------------------------------------------------

_SQ = np.sqrt
_HARM = {  # real harmonics on unit vectors, WB order, for lobe evaluation only
    0: lambda x, y, z: [np.ones_like(x)],
    1: lambda x, y, z: [z, x, y],
    2: lambda x, y, z: [(2*z*z - x*x - y*y) / _SQ(12), x*z, y*z,
                        (x*x - y*y) / 2, x*y],
    3: lambda x, y, z: [z*(2*z*z - 3*x*x - 3*y*y) / (2*_SQ(15)),
                        x*(4*z*z - x*x - y*y) / (2*_SQ(10)),
                        y*(4*z*z - x*x - y*y) / (2*_SQ(10)),
                        z*(x*x - y*y) / 2, x*y*z,
                        x*(x*x - 3*y*y) / (2*_SQ(6)),
                        y*(3*x*x - y*y) / (2*_SQ(6))],
}


def _eval_lobes(shells, unit_vecs):
    x, y, z = unit_vecs.T
    return np.column_stack(sum((_HARM[l](x, y, z) for l in shells), []))


def site_group(cell, q, symprec=1e-4):
    """Cartesian site-symmetry ops of fractional point q. cell = spglib tuple."""
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    A = np.asarray(cell[0], dtype=float).T
    Ainv = np.linalg.inv(A)
    ops = []
    for W, w in zip(ds.rotations, ds.translations):
        d = W @ q + w - q
        if np.allclose(d - np.round(d), 0, atol=symprec):
            R = A @ W @ Ainv
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), \
                "non-orthogonal op: lattice row/column convention?"
            ops.append(R)
    return np.array(ops)


@lru_cache(maxsize=None)
def _shell_rep_cached(shells, R_bytes, dim):
    R = np.frombuffer(R_bytes).reshape(3, 3)
    wb = wb_orb.get_orbitals()
    return block_diag(*[wb.rot_orb_basis(NAME_OF[l], R) for l in shells])


def shell_rep(shells, R):
    """D(R) on the direct sum of shells -- delegated to WannierBerri."""
    return _shell_rep_cached(tuple(shells), np.ascontiguousarray(R).tobytes(), 0)


# --------------------------------------------------------------------------
# representation theory: one primitive does all the splitting
# --------------------------------------------------------------------------

def _align_block(B, tol=1e-9):
    """Rotate an orthonormal block (rows) to the most axis-aligned basis of the
    SAME row space: orthogonal Procrustes against the dominant AO columns.
    Purely cosmetic -- span, invariance and D_wann are unchanged; output
    coefficients become sparse/readable (e.g. pure |dyz> instead of mixtures)
    whenever the span allows it. Signs fixed so each row's largest entry > 0."""
    d, n = B.shape
    cols = np.argsort(-np.linalg.norm(B, axis=0))[:d]
    U, _, Vt = np.linalg.svd(B[:, sorted(cols)])
    Bn = (U @ Vt).T @ B
    for r in Bn:
        if r[np.argmax(np.abs(r))] < 0:
            r *= -1
    Bn[np.abs(Bn) < tol] = 0.0
    return Bn


def isotypic_components(reps, seed=0, tol=1e-6):
    """Split a real orthogonal rep into isotypic components, no character table.

    Averaging a random symmetric matrix over the group lands in the commutant;
    by Schur its eigenspaces are the irrep copies. Copies carrying the same
    irrep (equal characters) are then merged, so each returned block is one
    whole isotypic component: (n_i * d_i, dim), orthonormal rows. The split
    WITHIN a component is arbitrary, which is exactly why we don't return it.

    Returns list of (M, d_i, n_i).
    """
    n = reps[0].shape[0]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n))
    X = X + X.T
    C = sum(D @ X @ D.T for D in reps) / len(reps)
    w, V = np.linalg.eigh(C)
    blocks, start = [], 0
    for i in range(1, n + 1):
        if i == n or abs(w[i] - w[start]) > tol:
            blocks.append(V[:, start:i].T)
            start = i
    blocks = [_align_block(B) for B in blocks]
    chars = [np.array([np.trace(B @ D @ B.T) for D in reps]) for B in blocks]
    comps, used = [], [False] * len(blocks)
    for i, Bi in enumerate(blocks):
        if used[i]:
            continue
        grp, used[i] = [Bi], True
        for j in range(i + 1, len(blocks)):
            if not used[j] and blocks[j].shape[0] == Bi.shape[0] \
               and np.allclose(chars[i], chars[j], atol=1e-4):
                grp.append(blocks[j])
                used[j] = True
        comps.append((_align_block(np.vstack(grp)), Bi.shape[0], len(grp)))
    return comps


def invariant(M, reps, atol=1e-6):
    """Is the row space of M invariant under every D in reps?"""
    P = M.T @ M
    return all(np.allclose(D @ P @ D.T, P, atol=atol) for D in reps)


# --------------------------------------------------------------------------
# bond-fitting hybrids (the intertwiner construction)
# --------------------------------------------------------------------------

def bond_permutations(bonds, ops, atol=1e-3):
    """P_R with (P_R)[j, sigma_R(j)] = 1. Raises if ops don't permute the set."""
    u = bonds / np.linalg.norm(bonds, axis=1)[:, None]
    perms = []
    for R in ops:
        rot = u @ R.T
        P = np.zeros((len(u), len(u)))
        for j, v in enumerate(rot):
            k = int(np.argmin(np.linalg.norm(u - v, axis=1)))
            if np.linalg.norm(u[k] - v) > atol:
                raise ValueError(
                    "site ops do not permute the bond set -- neighbour list is "
                    "not closed under G_q (check cutoffs / symmetrise it)")
            P[j, k] = 1.0
        perms.append(P)
    return perms


def hybrids_from_bonds(bonds, ops, shells, seed=0, optimize=False):
    """Bond-pointing hybrids h_j with O_R h_j = h_{sigma_R(j)}.

    T solves the intertwiner equation T D_R^T = P_R T. Rows are symmetrically
    orthonormalised (fixes the s:p:d weights from geometry alone) and signs are
    chosen per isotypic channel so lobes point outward.

    Returns (T, unique) where unique means symmetry+geometry fixed T up to lobe
    labelling: dim Hom(Gamma_sigma, D_AO) equals the number of distinct irreps
    in Gamma_sigma, i.e. the AO space supplies exactly one copy of each irrep
    the bonds need.
    """
    reps = [shell_rep(shells, R) for R in ops]
    perms = bond_permutations(bonds, ops)
    n, dim = len(bonds), reps[0].shape[0]

    A = np.vstack([np.kron(np.eye(n), D) - np.kron(P, np.eye(dim))
                   for D, P in zip(reps, perms)])
    N = null_space(A, rcond=1e-8)
    if N.shape[1] == 0:
        raise ValueError("no intertwiner: these shells cannot hybridise on these bonds")
    unique = N.shape[1] == len(isotypic_components(perms))

    Y = _eval_lobes(shells, bonds / np.linalg.norm(bonds, axis=1)[:, None])

    def make_T(c):
        T0 = (N @ c).reshape(n, dim)
        S = T0 @ T0.T
        w = np.linalg.eigvalsh(S)
        if w[0] < 1e-10 * w[-1]:
            return None                      # rank-deficient draw
        return np.real(np.linalg.inv(sqrtm(S))) @ T0

    def score(T):
        return float(np.einsum('jm,jm->', T, Y))   # outward lobe amplitude

    if optimize and not unique:
        # non-unique case: pick, over the WHOLE intertwiner space, the
        # geometry-fitting point -- the hybrids most directed along the bonds.
        # Every point of Hom is symmetry-legal, so this replaces the arbitrary
        # multiplicity mixing with a deterministic, bonding-like choice.
        from scipy.optimize import minimize
        def neg(c):
            T = make_T(c)
            return 1e6 if T is None else -score(T)
        best_c, best_v = None, np.inf
        rng = np.random.default_rng(0)       # fixed starts -> deterministic
        starts = [np.ones(N.shape[1])] + [rng.standard_normal(N.shape[1])
                                          for _ in range(6)]
        for c0 in starts:
            r = minimize(neg, c0 / np.linalg.norm(c0), method="Nelder-Mead",
                         options=dict(maxiter=4000, xatol=1e-10, fatol=1e-12))
            if r.fun < best_v:
                best_c, best_v = r.x, r.fun
        T = make_T(best_c)
        return T, unique

    # unique (or unoptimised) case: generic point + per-channel sign choice
    rng = np.random.default_rng(seed)
    T0 = (N @ rng.standard_normal(N.shape[1])).reshape(n, dim)
    chans = [C for M, _, _ in isotypic_components(reps)
             if np.linalg.norm(C := T0 @ (M.T @ M)) > 1e-8]
    Sih = np.real(np.linalg.inv(sqrtm(sum(C @ C.T for C in chans))))
    import itertools
    best = max((score(Sih @ sum(e * C for e, C in zip(eps, chans))), eps)
               for eps in itertools.product([1, -1], repeat=len(chans)))
    T = Sih @ sum(e * C for e, C in zip(best[1], chans))
    return T, unique


# --------------------------------------------------------------------------
# WannierBerri registration
# --------------------------------------------------------------------------

def register(name, M, shells, ops=None, overwrite=True):
    """Inject M as orbital type `name` into WannierBerri's hybrids machinery."""
    M = np.asarray(M, dtype=float)
    labels = [o for l in shells for o in wb_orb.orbitals_sets_dic[NAME_OF[l]]]
    assert M.shape[1] == len(labels)
    assert np.allclose(M @ M.T, np.eye(M.shape[0]), atol=1e-8), "rows not orthonormal"
    if ops is not None and not invariant(M, [shell_rep(shells, R) for R in ops]):
        raise ValueError(f"'{name}': row space not invariant under the site group "
                         "-- D_wann would be silently wrong")
    if name in wb_orb.orbitals_sets_dic and not overwrite:
        raise ValueError(f"'{name}' already registered")
    orb_names = [f"{name}-{k+1}" for k in range(M.shape[0])]
    wb_orb.orbitals_sets_dic[name] = orb_names
    for oname, vec in zip(orb_names, M):
        wb_orb.hybrids_coef[oname] = {l: float(c) for l, c in zip(labels, vec)
                                      if abs(c) > 1e-12}
    if name not in wb_orb.hybrid_shells_list:
        wb_orb.hybrid_shells_list.append(name)
    for f in (wb_orb.orb_to_shell, wb_orb.num_orbitals, wb_orb.get_orbitals):
        f.cache_clear()
    _shell_rep_cached.cache_clear()   # WB Orbitals instance was rebuilt


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------

def describe_orbital(orb_type, indent="    "):
    """Human-readable coefficient expansion of a registered (or built-in)
    WannierBerri orbital type, one line per member orbital.
    Works for custom SALC types, named hybrids ('sp3'), and plain shells ('p').
    Unknown names return a placeholder instead of raising, so logging never
    breaks a run."""
    lines = []
    members = wb_orb.orbitals_sets_dic.get(orb_type)
    if members is None:
        return f"{indent}{orb_type}: <unknown orbital type>"
    for m in members:
        coefs = wb_orb.hybrids_coef.get(m, {m: 1.0})
        terms = " ".join(f"{c:+.4f}|{o}>" for o, c in coefs.items())
        lines.append(f"{indent}{m:28s} = {terms}")
    return "\n".join(lines)


@dataclass
class SiteResult:
    index: int
    symbol: str
    position: np.ndarray
    wyckoff: str
    site_symmetry: str
    shells: list
    orbital_names: list = field(default_factory=list)   # registered WB types
    hybrid: bool = False
    note: str = ""


def _neighbours(atoms, i, cutoff):
    P, C = atoms.get_positions(), atoms.cell[:]
    sh = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1)
                   for z in (-1, 0, 1)])
    v = (P[None, :, :] + sh[:, None, :] @ C - P[i]).reshape(-1, 3)
    d = np.linalg.norm(v, axis=1)
    return v[(d > 1e-3) & (d < cutoff)]


def _symmetrize_bonds(bonds, ops, atol=1e-3):
    """Close a bond-vector set under the site group.

    Safe by construction: G_q fixes the site, so the image of a genuine
    neighbour is a genuine atom at the same distance. This upgrades any
    neighbour detector (CrystalNN included) to a guaranteed-G_q-closed set.
    """
    out = [np.asarray(b, dtype=float) for b in bonds]
    k = 0
    while k < len(out):
        for R in ops:
            v = R @ out[k]
            if not any(np.linalg.norm(v - w) < atol for w in out):
                out.append(v)
        k += 1
    return np.array(out)


def find_bonds(atoms, i, ops, cutoff=None, tol_shell=0.1):
    """Bond vectors of atom i, guaranteed closed under the site group.

    cutoff = None      -> pymatgen CrystalNN if available, else the first
                          coordination shell (everything within tol_shell
                          Angstrom of the nearest-neighbour distance).
    cutoff = float     -> plain distance cutoff.
    Every route is symmetrised under G_q before returning.
    """
    if cutoff is not None:
        nb = _neighbours(atoms, i, cutoff)
    else:
        nb = None
        try:
            from pymatgen.analysis.local_env import CrystalNN
            from pymatgen.io.ase import AseAtomsAdaptor
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st = AseAtomsAdaptor.get_structure(atoms)
                info = CrystalNN().get_nn_info(st, i)
            center = st[i].coords
            nb = np.array([n["site"].coords - center for n in info])
        except ImportError:
            pass
        if nb is None or len(nb) == 0:
            far = _neighbours(atoms, i, 6.0)
            if len(far) == 0:
                return np.zeros((0, 3))
            dmin = np.linalg.norm(far, axis=1).min()
            d = np.linalg.norm(far, axis=1)
            nb = far[d < dmin + tol_shell]
    if len(nb) == 0:
        return np.zeros((0, 3))
    return _symmetrize_bonds(nb, ops)


def build(atoms, shells, cutoff=None, prefix="", symprec=1e-4, seed=0,
          verbose=True, spacegroup=None, fallback="best_hybrid"):
    """Atoms + shells -> ProjectionsSet.

    shells : dict keyed by element symbol ('Ti') and/or ATOM INDEX (3).
             Values: 'spd' | ['s','p'] | [0,1,2] | mixed.
             An integer key applies to the whole symmetry orbit of that atom,
             so inequivalent sites of the same species can carry different
             shells. Index keys override the species key for their orbit; two
             index keys inside one orbit are unioned (with a warning if they
             differ). Sites matched by no key get no projections.
    cutoff : None (default) -> neighbours found automatically per site
             (pymatgen CrystalNN if installed, else the first coordination
             shell), then closed under the site group.
             float or {element: float} -> plain distance cutoff (still
             symmetrised). False -> no bond hybrids, isotypic components only.
             If the bonds fix the hybrids uniquely, bond-pointing lobes are
             used; otherwise isotypic components (same span either way, so
             this only ever changes the starting gauge, never the physics).
    """
    from irrep.spacegroup import SpaceGroup
    from wannierberri.symmetry.projections import Projection, ProjectionsSet

    cell = (np.array(atoms.cell[:]), atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if spacegroup is not None:
        # use the caller's SpaceGroup (e.g. SpaceGroup.from_gpaw(calc)) so the
        # Projections carry EXACTLY the symmetry of the DFT wavefunctions.
        # Site groups for the SALC analysis still come from spglib; that is
        # safe in one direction only (see assert below).
        sg = spacegroup
        if len(sg.symmetries) > len(ds.rotations):
            raise ValueError(
                f"caller SpaceGroup has {len(sg.symmetries)} operations but "
                f"spglib finds {len(ds.rotations)} at symprec={symprec}: the "
                "SALC invariance analysis would MISS operations. Raise symprec "
                "or check the two structures agree.")
    else:
        sg = SpaceGroup.from_cell(cell=cell, spinor=False)
    symbols = atoms.get_chemical_symbols()
    scaled = atoms.get_scaled_positions()

    # resolve shells per inequivalent representative: index keys (mapped
    # through the symmetry orbit) override species keys
    import warnings as _w
    by_orbit = {}
    for key, spec in shells.items():
        if isinstance(key, (int, np.integer)):
            rep = ds.equivalent_atoms[int(key)]
            ls = set(parse_shells(spec))
            if rep in by_orbit and by_orbit[rep] != ls:
                _w.warn(f"atoms {key} and a previous one are symmetry-equivalent "
                        f"but were given different shells; taking the union")
                ls |= by_orbit[rep]
            by_orbit[rep] = ls

    sites, pset = [], ProjectionsSet()
    for i in sorted(set(ds.equivalent_atoms)):
        el = symbols[i]
        if i in by_orbit:
            sh = sorted(by_orbit[i])
        elif el in shells:
            sh = parse_shells(shells[el])
        else:
            continue
        q = scaled[i]
        ops = site_group(cell, q, symprec)
        reps = [shell_rep(sh, R) for R in ops]
        cut = cutoff.get(el, None) if isinstance(cutoff, dict) else cutoff

        res = SiteResult(i, el, q, ds.wyckoffs[i], ds.site_symmetry_symbols[i], sh)
        base = f"{prefix}{el}{i}_{ds.wyckoffs[i]}"

        T = None
        if cut is not False:          # False disables bond hybrids for this run
            try:
                nb = find_bonds(atoms, i, ops, cutoff=cut)
                if len(nb):
                    Tc, uniq = hybrids_from_bonds(
                        nb, ops, sh, seed=seed,
                        optimize=(fallback == "best_hybrid"))
                    if uniq or fallback == "best_hybrid":
                        T = Tc
                        if not uniq:
                            res.note = "non-unique -> geometry-optimised hybrid"
            except ValueError as e:
                res.note = str(e)
        if T is not None:
            register(f"{base}_hyb", T, sh, ops)
            res.orbital_names, res.hybrid = [f"{base}_hyb"], True
            # complement of the hybrid span (e.g. t2g left over from sp3d2):
            comp = null_space(T, rcond=1e-8).T
            if comp.shape[0]:
                for k, (M, d, m) in enumerate(isotypic_components(
                        [comp @ D @ comp.T for D in reps], seed=seed)):
                    nm = f"{base}_rest{k}"
                    register(nm, M @ comp, sh, ops)
                    res.orbital_names.append(nm)
        elif fallback == "shells":
            # no unique hybrid: plain shell orbitals give the same span with
            # the best-conditioned (axis-aligned) starting gauge, and require
            # no registration at all
            if cut is not False and not res.note:
                res.note = "hybrids not unique -> plain shells"
            res.orbital_names = [NAME_OF[l] for l in sh]
        else:
            if cut and not res.note:
                res.note = "hybrids not unique -> isotypic components"
            for k, (M, d, m) in enumerate(isotypic_components(reps, seed=seed)):
                nm = f"{base}_{k}"
                register(nm, M, sh, ops)
                res.orbital_names.append(nm)

        for nm in res.orbital_names:
            p = Projection(position_num=[q], orbital=nm, spacegroup=sg,rotate_basis=True)
            assert np.allclose(((np.array(p.positions[0]) - q + .5) % 1) - .5, 0,
                               atol=10 * symprec), "orbit representative moved"
            pset.add(p)
        sites.append(res)

        if verbose:
            tag = "bond-pointing hybrid" if res.hybrid else "isotypic components"
            print(f"{el}{i} @ {np.round(q, 4)}  {res.site_symmetry:6s} "
                  f"|G_q|={len(ops):2d}  {''.join(NAME_OF[l] for l in sh)} -> "
                  f"{tag}: {res.orbital_names}"
                  + (f"   [{res.note}]" if res.note else ""))
            for nm in res.orbital_names:
                for member in wb_orb.orbitals_sets_dic[nm]:
                    terms = " ".join(
                        f"{c:+.3f}|{o}>"
                        for o, c in wb_orb.hybrids_coef[member].items())
                    print(f"      {member:24s} = {terms}")
    return pset, sites