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
import itertools

from scipy.linalg import block_diag, null_space

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
def _shell_rep_cached(shells, R_bytes):
    R = np.frombuffer(R_bytes).reshape(3, 3)
    wb = wb_orb.get_orbitals()
    return block_diag(*[wb.rot_orb_basis(NAME_OF[l], R) for l in shells])


def shell_rep(shells, R):
    """D(R) on the direct sum of shells -- delegated to WannierBerri."""
    return _shell_rep_cached(tuple(shells), np.ascontiguousarray(R).tobytes())


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


def isotypic_copies(reps, weight, tol=1e-6):
    """Split into INDIVIDUAL irrep copies, ordered by projectability.

    `isotypic_components` deliberately merges all copies of one irrep, because
    without physical input the split between them is arbitrary (it comes from
    the random commutant element, and the characters used for the merge are
    traces, so they carry no information about it). Given a `weight` W in the
    commutant -- the projectability matrix -- the split becomes canonical:
    restricted to an isotypic component, W commutes with n_i copies of rho_i,
    so by Schur it is S_i (x) 1_{d_i}; its eigenvalues therefore come in
    d_i-fold groups, one group per copy, and the eigenvalue IS that copy's
    per-orbital projectability.

    Returns a list of (rows, d_i, projectability), sorted by decreasing
    projectability. Each `rows` is (d_i, dim) with orthonormal rows spanning one
    invariant subspace, so any prefix of the list is a symmetry-legal selection.
    """
    W = np.asarray(weight, float)
    W = 0.5 * (W + W.T)
    W = sum(D @ W @ D.T for D in reps) / len(reps)     # enforce commutant
    out = []
    for Mc, d_i, n_i in isotypic_components(reps):
        Wc = Mc @ W @ Mc.T                             # (n_i*d_i, n_i*d_i)
        w, V = np.linalg.eigh(Wc)
        start = 0
        for k in range(1, len(w) + 1):
            if k == len(w) or abs(w[k] - w[start]) > tol:
                blk = V[:, start:k].T @ Mc             # back to the AO basis
                if blk.shape[0] % d_i:
                    raise RuntimeError(
                        f"eigenvalue group of size {blk.shape[0]} is not a "
                        f"multiple of the irrep dimension {d_i}: `weight` is "
                        "probably not in the commutant")
                for c in range(blk.shape[0] // d_i):   # split accidental ties
                    out.append((_align_block(blk[c * d_i:(c + 1) * d_i]),
                                d_i, float(w[start])))
                start = k
    return sorted(out, key=lambda t: -t[2])


def commutant_structure(reps, seed=0, tol=1e-9):
    """What ANY weight matrix can look like at this site -- no DFT needed.

    By Schur every commutant element is  (+)_i S_i (x) 1_{d_i}, so M carries
    exactly sum_i n_i^2 free numbers. It is DIAGONAL in the atomic-orbital basis
    only if (a) every multiplicity n_i is 1 and (b) each AO lies in a single
    irrep component -- (b) fails when the site's symmetry axes are not aligned
    with the Cartesian frame (e.g. a 3-fold along [111]).

    Returns dict(blocks=[(d_i, n_i)], n_free=..., diagonal=bool,
                 can_mix=bool) where can_mix is True iff some n_i > 1, i.e. iff
    a weight can resolve a mixing that symmetry left free. Where can_mix is
    False the weight can only SELECT irreps, never mix them -- no point running
    the projectability optimiser at such a site.
    """
    comps = isotypic_components(reps, seed=seed)
    n = reps[0].shape[0]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n))
    X = X + X.T
    C = sum(D @ X @ D.T for D in reps) / len(reps)      # generic commutant elt
    off = np.abs(C - np.diag(np.diag(C))).max() / max(np.abs(C).max(), 1e-30)
    return dict(blocks=[(d, m) for _, d, m in comps],
                n_free=sum(m * m for _, _, m in comps),
                diagonal=bool(off < tol),
                can_mix=any(m > 1 for _, _, m in comps))


def invariant(M, reps, atol=1e-6):
    """Is the row space of M invariant under every D in reps?"""
    P = M.T @ M
    return all(np.allclose(D @ P @ D.T, P, atol=atol) for D in reps)


# --------------------------------------------------------------------------
# bond-fitting hybrids (the intertwiner construction)
# --------------------------------------------------------------------------

def _polar(A):
    """Orthonormal factor of A (rows). Equals S^{-1/2}A for S=AA^T invertible,
    but is computed by SVD so it is exact at any conditioning."""
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    return U @ Vt


def _shell_slice(shells, sub):
    """Column indices of the sub-shells `sub` inside the `shells` AO layout."""
    full = [(l, o) for l in shells for o in range(2 * l + 1)]
    return np.array([full.index((l, o)) for l in sub for o in range(2 * l + 1)])


def _fix_signs(T0, reps, Y):
    """Split T0 over isotypic channels, choose the per-channel signs that make
    lobes point outward, and orthonormalise. Signs must be chosen per CHANNEL,
    not per row: flipping a single row would break the permutation property."""
    chans = [C for Mc, _, _ in isotypic_components(reps)
             if np.linalg.norm(C := T0 @ (Mc.T @ Mc)) > 1e-8]
    best = max((float(np.einsum('jm,jm->', _polar(sum(e * C for e, C in zip(eps, chans))), Y)),
                eps) for eps in itertools.product([1, -1], repeat=len(chans)))
    return _polar(sum(e * C for e, C in zip(best[1], chans)))


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


def minimal_shell_hybrids(bonds, ops, shells, seed=0, weight=None):
    """Bond hybrids taking each needed irrep from the LOWEST shell that can
    supply it -- e.g. sp3 (pure s,p) at a tetrahedral site even when d is in
    `shells`, leaving the whole d shell as a separate complement.

    Rationale: when several shells carry an irrep the bonds need, the
    geometry-optimised objective (max bond amplitude) prefers contaminating the
    lobes with higher-l character, because d/f orbitals also have amplitude
    along the bonds. That dilutes otherwise-pure sp3/sp2 lobes. Filling the
    needed irreps from the smallest l instead is parameter-free, reproduces the
    textbook hybrids, and keeps higher shells intact for the complement.

    Strategy: try growing prefixes of `shells` (s; then s,p; then s,p,d; ...);
    return the hybrid from the FIRST prefix whose own bond intertwiner is
    non-empty AND spans all bonds (unique or not). The lobes are then embedded
    (zero-padded) back into the full `shells` space so the registered orbital
    matches the requested shells; the leftover dimensions become the complement
    that build() decomposes into isotypic blocks.

    Returns (T_full, used_shells) with T_full of shape (n_bonds, dim_full),
    orthonormal rows. Raises ValueError if no prefix works (caller falls back).
    """
    shells = sorted(shells)
    for k in range(1, len(shells) + 1):
        sub = shells[:k]
        try:
            # within the chosen prefix, orient lobes by geometry if there is
            # freedom to (non-unique); otherwise the unique solution is taken.
            w_sub = (None if weight is None
                     else weight[np.ix_(_shell_slice(shells, sub),
                                        _shell_slice(shells, sub))])
            T_sub, uniq = hybrids_from_bonds(bonds, ops, sub, seed=seed,
                                             optimize=True, weight=w_sub)
        except ValueError:
            continue
        if T_sub is None:      # prefix produced no well-conditioned hybrid
            continue
        # embed T_sub (columns over `sub`) into the full-shell column layout
        full_labels = [(l, o) for l in shells
                       for o in range(2 * l + 1)]
        sub_labels = [(l, o) for l in sub for o in range(2 * l + 1)]
        T_full = np.zeros((T_sub.shape[0], len(full_labels)))
        for c, lab in enumerate(sub_labels):
            T_full[:, full_labels.index(lab)] = T_sub[:, c]
        return T_full, sub
    raise ValueError("no shell prefix can form bond hybrids")


def hybrids_from_bonds(bonds, ops, shells, seed=0, optimize=False, weight=None):
    """Bond-pointing hybrids h_j with O_R h_j = h_{sigma_R(j)}.

    T solves the intertwiner equation T D_R^T = P_R T; rows are orthonormalised
    (Loewdin/polar, which preserves the permutation property) and per-channel
    signs are chosen so lobes point outward.

    Returns (T, unique). `unique` means dim Hom equals the number of distinct
    irreps in the bond rep, i.e. the AO space supplies exactly one copy of each
    irrep the bonds need -- then symmetry+geometry fix T up to lobe labelling.

    weight : (dim, dim) array, optional
        A symmetric matrix commuting with the site-symmetry rep -- in practice
        the projectability matrix M_mm' = sum_{nk in window} <g_m|psi><psi|g_m'>.
        When the hybrid is NOT unique, the leftover freedom is a multiplicity
        mixing that symmetry cannot fix; with `weight` the mixing is chosen to
        maximise tr(T^T T M), i.e. the overlap of the trial orbitals with the
        target bands, instead of maximising bond directionality. Because M lies
        in the commutant -- exactly the space symmetry leaves free -- this
        resolves the ambiguity with physics without touching anything symmetry
        already determined.
    """
    reps = [shell_rep(shells, R) for R in ops]
    perms = bond_permutations(bonds, ops)
    n, dim = len(bonds), reps[0].shape[0]

    if n > dim:
        raise ValueError(
            f"{n} bonds but only {dim} orbitals in shells {shells}: cannot form "
            "bond hybrids (need more shells, or use plain shells here)")

    A = np.vstack([np.kron(np.eye(n), D) - np.kron(P, np.eye(dim))
                   for D, P in zip(reps, perms)])
    N = null_space(A, rcond=1e-8)
    if N.shape[1] == 0:
        raise ValueError("no intertwiner: these shells cannot hybridise on these bonds")
    unique = N.shape[1] == len(isotypic_components(perms))

    Y = _eval_lobes(shells, bonds / np.linalg.norm(bonds, axis=1)[:, None])

    if weight is not None:
        W = 0.5 * (np.asarray(weight, float) + np.asarray(weight, float).T)
        W = sum(D @ W @ D.T for D in reps) / len(reps)   # project onto commutant
    else:
        W = None

    def make_T(c):
        T0 = (N @ c).reshape(n, dim)
        U, sv, Vt = np.linalg.svd(T0, full_matrices=False)
        if sv[-1] < 1e-3 * sv[0]:
            return None                      # (nearly) coalescing lobes
        return U @ Vt

    def score(T):
        if W is None:
            return float(np.einsum('jm,jm->', T, Y))          # bond directionality
        return float(np.einsum('jm,mn,jn->', T, W, T))        # projectability

    if optimize and not unique:
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
            Tr = make_T(r.x)
            if r.fun < best_v and Tr is not None and invariant(Tr, reps):
                best_c, best_v = r.x, r.fun
        T0 = make_T(best_c) if best_c is not None else None
        if T0 is not None:
            # the projectability objective is sign-blind (quadratic in T), so
            # signs are always fixed by directionality afterwards
            T = _fix_signs(T0, reps, Y)
            if invariant(T, reps):
                return T, unique
        # no well-conditioned invariant optimum: fall through to the generic
        # construction, which is invariant by construction

    T0 = (N @ np.random.default_rng(seed).standard_normal(N.shape[1])).reshape(n, dim)
    T = _fix_signs(T0, reps, Y)
    if not invariant(T, reps):
        raise ValueError("bond directions do not span a G_q-invariant subspace "
                         "of these shells -- use plain shells / components here")
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
          verbose=True, spacegroup=None, fallback="best_hybrid", weight_fn=None,
          select_threshold=None):
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
    weight_fn : callable(atom_index, shells, ops) -> (dim, dim) array or None
             Optional physical weight used ONLY where symmetry leaves freedom
             (non-unique hybrids). In practice the projectability matrix
             M_mm' = sum_{nk in window} <g_m|psi_nk><psi_nk|g_m'>; see
             `projectability_from_gpaw`. Keeps this module energy-agnostic:
             the caller supplies the DFT information.
    select_threshold : float, optional
             If given (requires weight_fn), the requested shells are treated as
             CANDIDATES: at each site the AO space is split into individual
             irrep copies and only those with projectability >= threshold are
             kept. Because the weight lies in the commutant its eigenvalues come
             in d_i-fold groups, so the cut always falls between whole copies
             and can never split a degenerate irrep. `nwann` then comes OUT of
             build (read pset.num_wann) instead of going in. Bond hybrids are
             skipped for selected sites -- the selection changes the span, which
             is the point; the hybrid/complement split only changes the gauge.
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

        # --- projectability selection (changes the SPAN, unlike the gauge
        # choices below): keep only copies the target bands actually support
        if select_threshold is not None:
            w_sel = None if weight_fn is None else weight_fn(i, sh, ops)
            if w_sel is None:
                raise ValueError("select_threshold requires a weight_fn that "
                                 f"returns a matrix for atom {i}")
            copies = isotypic_copies(reps, w_sel)
            keep = [(M, d, pj) for M, d, pj in copies if pj >= select_threshold]
            if not keep:
                res.note = (f"no copy reaches projectability {select_threshold} "
                            f"(best {copies[0][2]:.3f}) -- site skipped")
                sites.append(res)
                if verbose:
                    print(f"{el}{i}: {res.note}")
                continue
            for k, (M, d, pj) in enumerate(keep):
                nm = f"{base}_sel{k}"
                register(nm, M, sh, ops)
                res.orbital_names.append(nm)
            res.note = (f"projectability selection: kept {len(keep)}/{len(copies)}"
                        f" copies ({sum(d for _, d, _ in keep)} WF/site), "
                        f"p = {[round(pj, 3) for _, _, pj in keep]}")
            T = None
        else:
            T = None
            if cut is not False:      # False disables bond hybrids for this run
                try:
                    nb = find_bonds(atoms, i, ops, cutoff=cut)
                    if len(nb):
                        w = None if weight_fn is None else weight_fn(i, sh, ops)
                        if fallback == "minimal_shell":
                            T, used = minimal_shell_hybrids(nb, ops, sh, seed=seed,
                                                            weight=w)
                            if len(used) < len(sh):
                                res.note = ("minimal-shell hybrid on "
                                            f"{''.join(NAME_OF[l] for l in used)}"
                                            f", {''.join(NAME_OF[l] for l in sh if l not in used)}"
                                            " left as complement")
                        else:
                            Tc, uniq = hybrids_from_bonds(
                                nb, ops, sh, seed=seed, weight=w,
                                optimize=(fallback == "best_hybrid"))
                            if uniq or fallback == "best_hybrid":
                                T = Tc
                                if not uniq:
                                    res.note = "non-unique -> geometry-optimised hybrid"
                except ValueError as e:
                    res.note = str(e)
        if res.orbital_names:
            pass          # selection already registered
        elif T is not None:
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
            if cut is not False and not res.note:
                res.note = "hybrids not unique -> isotypic components"
            for k, (M, d, m) in enumerate(isotypic_components(reps, seed=seed)):
                nm = f"{base}_{k}"
                register(nm, M, sh, ops)
                res.orbital_names.append(nm)

        for nm in res.orbital_names:
            p = Projection(position_num=[q], orbital=nm, spacegroup=sg,
                           rotate_basis=True)
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


# --------------------------------------------------------------------------
# projectability weight from a DFT calculation
# --------------------------------------------------------------------------

# GPAW orders real spherical harmonics by m = -l..l; WannierBerri uses its own
# order. Index of the WB orbital inside GPAW's (2l+1) block:
#   l=0  s                      -> [0]
#   l=1  GPAW (y, z, x)         -> WB (z, x, y)
#   l=2  GPAW (xy, yz, z2, xz, x2-y2) -> WB (z2, xz, yz, x2-y2, xy)
_GPAW_TO_WB = {0: [0], 1: [1, 2, 0], 2: [2, 3, 1, 4, 0]}


def projectability_from_gpaw(calc, window, spin=0, atol=0.25,
                             n_select='window', normalize=True,
                             shells_per_atom=None, verbose_channels=False):
    """Build a `weight_fn` for `build` from a (NSCF) GPAW calculation.

    Returns callable(iatom, shells, ops) -> M, with

        M_mm' = sum_{n in window, k} w_k <g_m|psi_nk><psi_nk|g_m'>

    taken from GPAW's PAW projections <p_i^a|psi_nk> (the same quantity the
    projected DOS is built from; the pDOS is its diagonal). M commutes with the
    site-symmetry representation, so it lies in the commutant -- exactly the
    freedom symmetry leaves -- which is why it can fix the multiplicity mixing
    without disturbing anything symmetry already determined.

    n_select : 'valence' | 'first' | 'sum'
        Which PAW projector to use when a setup has several for the same l
        (e.g. Ti 3p semicore and 4p valence; Ba 5s and 6s):
          'window' (default) -- the bound channel carrying the most weight
                     INSIDE the energy window. Semicore and valence are split in
                     energy, so the window itself selects: no n heuristic, and
                     it adapts if your window moves into the semicore region.
          'valence' -- the bound channel with the highest n.
          'first'   -- the bound channel with the lowest n (semicore first).
          'sum'     -- add |<p|psi>|^2 over all bound channels of that l. This
                     is most likely what GPAW's get_orbital_ldos does, so use it
                     when you want M's diagonal to reproduce the pDOS exactly.
        Unbound channels (n < 0) are never used, matching the pDOS being zero
        there. Run describe_setup(calc, iatom) to see what a setup offers.

    normalize : bool
        If True (default), each state's contribution is divided by its TOTAL
        bound-projector weight S_nk before accumulation (Mulliken-style). This
        is necessary because the PAW projectors are duals of the partial waves,
        not a resolution of the identity: summing the raw pDOS over every
        channel recovers only ~64% of the total DOS, so the raw numbers are not
        occupations and cannot be compared against a per-shell threshold.
        After the division, summing diag(M) over ALL atoms and channels equals
        the number of states in the window, so the diagonal IS an occupation
        (bounded by 2l+1 for a localized channel).

        The attribution is EXACT (diag sums over all atoms/channels to the
        state count) only if the denominator spans exactly the channels being
        attributed to. Pass `shells_per_atom` for that; otherwise the
        denominator uses every bound channel, and any l with two bound channels
        (semicore + valence, e.g. Ti 3p/4p) leaves the unselected one's weight
        unattributed -- a few percent deficit.

        Note this is done per STATE, not per energy: unlike the same correction
        applied to a broadened DOS, it uses exact eigenvalues, so the window
        edge stays sharp. The factor is a scalar per state and therefore does
        not disturb the commutant structure.

    NOTE ON CONVENTIONS: the m-ordering map `_GPAW_TO_WB` above must match your
    GPAW version's real-harmonic order. A wrong map shows up as M failing to
    commute with the site group; this function checks that and raises, rather
    than silently returning a wrong weight. l=3 is not mapped.
    """
    import numpy as _np

    _bound_cols_cache, _statenorm_cache = {}, {}

    def _bound_cols(a):
        """Projector columns of atom a used as the attribution basis.

        With `shells_per_atom` given: exactly the channels that will be used for
        that atom (one per requested l, chosen the same way as the numerator),
        which makes the occupation sum rule exact. Otherwise: every bound
        channel, which over-counts the denominator where an l has both a
        semicore and a valence projector."""
        if a not in _bound_cols_cache:
            st = calc.setups[a]
            nj = getattr(st, "n_j", [1] * len(st.l_j))
            want = None
            if shells_per_atom is not None and a in shells_per_atom:
                want = set(parse_shells(shells_per_atom[a]))
            per_l, off = {}, 0
            for j, l in enumerate(st.l_j):
                l = int(l)
                n = nj[j] if nj[j] is not None else 1
                if n > 0 and (want is None or l in want):
                    per_l.setdefault(l, []).append((n, list(range(off, off + 2 * l + 1))))
                off += 2 * l + 1
            cols = []
            for l, v in per_l.items():
                # one channel per l when a selection is declared, else all bound
                cols += (max(v)[1] if want is not None else
                         [c for _, cc in v for c in cc])
            _bound_cols_cache[a] = _np.array(sorted(cols), dtype=int)
        return _bound_cols_cache[a]

    def _state_norm(k, nk):
        """S_nk = total bound-projector weight of each band at this k-point.
        Dividing by it distributes the TRUE state count over the atomic
        channels (Mulliken-style), so the diagonals become occupations that sum
        to the number of states in the window."""
        if k not in _statenorm_cache:
            kpt = calc.wfs.kpt_u[spin * nk + k]
            S = None
            for a in range(len(calc.atoms)):
                P = _np.asarray(kpt.P_ani[a])[:, _bound_cols(a)]
                t = _np.real(_np.einsum('ni,ni->n', P.conj(), P))
                S = t if S is None else S + t
            _statenorm_cache[k] = S
        return _statenorm_cache[k]

    def weight_fn(iatom, shells, ops):
        import warnings as _w
        shells = parse_shells(shells)             # accepts 'spd', ['s','p'], [0,1,2]
        for l in shells:
            if l not in _GPAW_TO_WB:
                _w.warn(f"no GPAW->WB m-ordering for l={l} (only s,p,d are "
                        "mapped): returning no weight for this site")
                return None

        setup = calc.setups[iatom]
        n_j = getattr(setup, "n_j", [1] * len(setup.l_j))

        # every BOUND channel is a candidate: (l, n) -> its P_ani columns in WB
        # order. Unbound channels (n<0) are PAW completeness functions, not
        # atomic orbitals, and are never used (GPAW's pDOS is zero there).
        cand, off = {}, 0
        for j, l in enumerate(setup.l_j):
            l = int(l)
            n = n_j[j] if n_j[j] is not None else 1
            if n > 0 and l in _GPAW_TO_WB:
                cand.setdefault(l, []).append(
                    (n, [off + t for t in _GPAW_TO_WB[l]]))
            off += 2 * l + 1

        for l in shells:
            if l not in cand:
                _w.warn(f"atom {iatom}: no BOUND projector for l={l} "
                        f"(bound channels: {sorted(cand)}) -- returning no "
                        "weight. Run describe_setup(calc, iatom) to see the "
                        "channels; unbound (n<0) ones are skipped on purpose.")
                return None

        # accumulate M over ALL candidate channels of the requested l's at once,
        # so cross-l blocks are available whichever channel we end up choosing
        flat, slc, pos = [], {}, 0
        for l in shells:
            for n, cols in sorted(cand[l]):
                slc[(l, n)] = slice(pos, pos + 2 * l + 1)
                flat += cols
                pos += 2 * l + 1
        flat = _np.array(flat)

        Mall = _np.zeros((pos, pos))
        wk = calc.get_k_point_weights()
        for k in range(len(wk)):
            eps = calc.get_eigenvalues(kpt=k, spin=spin)
            sel = (eps >= window[0]) & (eps <= window[1])
            if not sel.any():
                continue
            kpt = calc.wfs.kpt_u[spin * len(wk) + k]
            P = _np.asarray(kpt.P_ani[iatom])[sel][:, flat]
            if normalize:
                S = _state_norm(k, len(wk))[sel]
                P = P / _np.sqrt(_np.maximum(S, 1e-12))[:, None]
            Mall += wk[k] * _np.real(P.conj().T @ P)

        # choose one channel per l
        chosen = {}
        for l in shells:
            ns = sorted(n for n, _ in cand[l])
            if n_select == 'window':
                # let the ENERGY WINDOW decide: the channel with real weight
                # inside it. Semicore and valence are separated in energy, so
                # the one outside the window contributes ~0 and loses. No
                # principal-quantum-number heuristic needed.
                chosen[l] = max(ns, key=lambda n: _np.trace(
                    Mall[slc[(l, n)], slc[(l, n)]]))
            elif n_select == 'valence':
                chosen[l] = max(ns)
            elif n_select == 'first':
                chosen[l] = min(ns)
            elif n_select == 'sum':
                chosen[l] = None                  # handled below
            else:
                raise ValueError("n_select must be window/valence/first/sum, "
                                 f"got {n_select!r}")

        # window-selected channel per l defines the block layout (and all
        # off-diagonal, cross-l blocks -- those have no meaning for a sum over
        # several radial channels, so 'sum' only changes the DIAGONAL blocks)
        win_of_l = {l: max((n for n, _ in cand[l]),
                           key=lambda n: _np.trace(Mall[slc[(l, n)], slc[(l, n)]]))
                    for l in shells}
        pick = win_of_l if n_select in ('window', 'sum') else chosen
        idx = _np.concatenate([_np.arange(slc[(l, pick[l])].start,
                                          slc[(l, pick[l])].stop)
                               for l in shells])
        M = Mall[_np.ix_(idx, idx)].copy()
        if n_select == 'sum':
            off = 0
            for l in shells:
                d = 2 * l + 1
                blk = sum(Mall[slc[(l, n)], slc[(l, n)]] for n, _ in cand[l])
                M[off:off + d, off:off + d] = blk       # each channel ONCE
                off += d

        if verbose_channels:
            picked = {NAME_OF[l]: (chosen[l], float(_np.trace(
                Mall[slc[(l, chosen[l])], slc[(l, chosen[l])]])))
                for l in shells} if n_select != 'sum' else 'sum of all'
            print(f"  atom {iatom}: channels {{l: (n, in-window weight)}} -> {picked}")

        nrm0 = _np.linalg.norm(M)
        if nrm0 == 0:
            _w.warn(f"atom {iatom}: no bands inside window {window} -- "
                    "returning no weight (check the window and its units, eV)")
            return None

        # Symmetrisation is a RECONSTRUCTION, not a correction: the full-BZ sum
        # of M is symmetric, but a weighted IBZ sum is not (for a scalar such as
        # the pDOS the k-weights suffice; for a MATRIX you must also average
        # over the star). A moderate deviation is therefore expected for any
        # *-irred.gpw run and is not an error.
        reps = [shell_rep(shells, R) for R in ops]
        Msym = sum(D @ M @ D.T for D in reps) / len(reps)
        dev = _np.linalg.norm(Msym - M) / nrm0
        if dev > atol:
            _w.warn(f"atom {iatom}: projectability deviates from the commutant "
                    f"by {dev:.2f} before symmetrisation. Expected on an "
                    "irreducible-BZ calculation; only suspect the m-ordering or "
                    "signs if this is large (>~0.5) -- check with "
                    "derive_gpaw_ordering().")
        return Msym

    return weight_fn


def available_shells(calc, iatom):
    """l values for which the GPAW setup of `iatom` carries PAW projectors.

    Use as the CANDIDATE shell set when running `build` with
    `select_threshold`: it is what the pseudopotential can actually describe,
    so it is the honest upper bound on what projectability can select from.
    """
    st = calc.setups[iatom]
    n_j = getattr(st, "n_j", [1] * len(st.l_j))
    return sorted({int(l) for l, n in zip(st.l_j, n_j)
                   if n is None or n > 0})


def derive_gpaw_ordering(lmax=2, npts=400, seed=0, tol=1e-6):
    """Determine GPAW's real-harmonic order and signs RELATIVE TO WannierBerri,
    numerically, instead of trusting the hard-coded `_GPAW_TO_WB` table.

    Both bases span the same space for each l, so they are related by a scaled
    signed permutation S with  Y_WB = Y_GPAW @ S.  We evaluate both on the same
    directions and least-squares fit S; if the conventions differ only by order
    and sign (they should), every column of S has exactly one non-negligible
    entry, which gives the index and the sign.

    Returns {l: dict(perm=[...], signs=[...], scale=[...])} where
        M_WB = S.T @ M_GPAW @ S,  S built from perm/signs (scale is reported so
    you can see whether the two normalisations also differ per l -- they often
    do, which matters when comparing projectabilities ACROSS shells).

    Raises if the fit is not a signed permutation (conventions genuinely differ).
    """
    from gpaw.spherical_harmonics import Y as _Y

    i = np.arange(npts) + 0.5
    phi = np.arccos(1 - 2 * i / npts)
    golden = np.pi * (1 + 5 ** 0.5)
    th = golden * i
    d = np.column_stack([np.cos(th) * np.sin(phi),
                         np.sin(th) * np.sin(phi), np.cos(phi)])
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    d = d @ (Q * np.sign(np.linalg.det(Q)))

    out = {}
    for l in range(lmax + 1):
        G = np.column_stack([_Y(l * l + m, d[:, 0], d[:, 1], d[:, 2])
                             for m in range(2 * l + 1)])
        W = _eval_lobes([l], d)
        S, *_ = np.linalg.lstsq(G, W, rcond=None)
        resid = np.linalg.norm(G @ S - W) / max(np.linalg.norm(W), 1e-30)
        if resid > 1e-6:
            raise RuntimeError(f"l={l}: GPAW and WB harmonics are not related by "
                               f"a linear map to {resid:.1e} -- unexpected")
        perm, signs, scale = [], [], []
        for c in range(2 * l + 1):
            col = S[:, c]
            k = int(np.argmax(np.abs(col)))
            rest = np.linalg.norm(np.delete(col, k))
            if rest > 1e-6 * max(abs(col[k]), 1e-30):
                raise RuntimeError(
                    f"l={l}, WB orbital {c}: GPAW column is a MIXTURE, not a "
                    f"signed permutation (off weight {rest:.2e}). The two codes "
                    "use genuinely different real-harmonic bases for this l.")
            perm.append(k)
            signs.append(float(np.sign(col[k])))
            scale.append(float(abs(col[k])))
        out[l] = dict(perm=perm, signs=signs, scale=scale)
    return out


def describe_setup(calc, iatom):
    """List the PAW projector channels of an atom.

    Columns: j (projector index), l, n (principal quantum number; <0 means an
    UNBOUND/scattering channel, included for PAW completeness -- GPAW's
    get_orbital_ldos reports zero for these), f (occupation), and the slice of
    the P_ani column index the channel occupies.

    Use this to see why a projectability channel disagrees with the pDOS: the
    usual causes are several projectors sharing one l (semicore + valence) or a
    channel with no bound state.
    """
    st = calc.setups[iatom]
    n_j = getattr(st, "n_j", [None] * len(st.l_j))
    f_j = getattr(st, "f_j", [None] * len(st.l_j))
    rows, off = [], 0
    for j, l in enumerate(st.l_j):
        rows.append(dict(j=j, l=int(l), n=n_j[j], f=f_j[j],
                         cols=(off, off + 2 * int(l))))
        off += 2 * int(l) + 1
    return rows