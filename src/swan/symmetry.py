"""
Symmetry-adapted projections for Wannierisation, in one module.

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

Physical input enters through `weight_fn` only, so this module stays
energy-agnostic. In practice that is `projectability_from_gpaw` below, which
returns the sphere-projected charge matrix

    M_mm' = sum_{nk in window} w_k <g_m|psi_nk><psi_nk|g_m'>   [electrons]

built from `sphere_projectability.collect_projections`. Two properties matter
here: M lies in the commutant of the site-symmetry rep (so it can only resolve
freedom symmetry left open, never disturb what symmetry fixed), and after the
partial-wave metric it is in physical units (so its eigenvalues are per-orbital
occupations, comparable across sites, shells and elements).

No zaxis/xaxis handling is needed: everything is built in the crystal Cartesian
frame, and with rotate_basis=True WannierBerri generates the per-site frames
from the orbit itself (verified: the residual rotation always lands in G_q of
the representative).

Frame convention note: hybrid lobe DIRECTIONS returned here are crystal-
Cartesian. Weights (s:p:d ratios) are frame-independent.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import spglib
from scipy.linalg import block_diag, null_space

from wannierberri.symmetry import orbitals as wb_orb

from .projectability.sphere_projectability import (
    bound_channels,
    collect_projections,
    describe_metric,
)

L_OF = {"s": 0, "p": 1, "d": 2, "f": 3}
NAME_OF = {v: k for k, v in L_OF.items()}

FALLBACKS = ("best_hybrid", "minimal_shell", "shells", "components")


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

    Returns list of (M, d_i, n_i). `any(n_i > 1)` is the test for whether a
    weight matrix can resolve anything at this site at all: where every n_i is
    1, symmetry has already fixed the orbitals up to gauge and no amount of DFT
    input can mix them.
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

    With the partial-wave metric in place that eigenvalue is in ELECTRONS: the
    number of electrons of that symmetry, on that site, inside the augmentation
    sphere, in the window. It is therefore directly comparable to a per-orbital
    occupancy threshold (see `build(select_threshold=...)`), which it was not
    when W came from raw PAW projections carrying a setup-dependent scale.

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

    Note that with a metric-correct `weight` the projectability objective is a
    legitimate alternative here rather than a fallback, because the s:p:d
    relative scale in W is now physical. It answers a different question:
    `minimal_shell` asks which lobes are cleanest, the weight asks which the
    bands actually support. Prefer this one for main-group covalent networks,
    the weight for transition metals where d participation is the physics.

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
        full_labels = [(l, o) for l in shells for o in range(2 * l + 1)]
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
        the sphere-projected charge matrix from `projectability_from_gpaw`.
        When the hybrid is NOT unique, the leftover freedom is a multiplicity
        mixing that symmetry cannot fix; with `weight` the mixing is chosen to
        maximise tr(T W T^T), i.e. the overlap of the trial orbitals with the
        target bands, instead of maximising bond directionality. Because W lies
        in the commutant -- exactly the space symmetry leaves free -- this
        resolves the ambiguity with physics without touching anything symmetry
        already determined.

        This objective is only meaningful when W's cross-l blocks are on a
        common physical scale, i.e. when W came through the partial-wave
        metric. Raw PAW projection weights carry a per-shell convention factor,
        so maximising over an s/p/d mixing with them optimises an arbitrary
        number.
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
        W = np.asarray(weight, float)
        W = 0.5 * (W + W.T)
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


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------

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
            d = np.linalg.norm(far, axis=1)
            nb = far[d < d.min() + tol_shell]
    if len(nb) == 0:
        return np.zeros((0, 3))
    return _symmetrize_bonds(nb, ops)


def _neighbours_at(atoms, centre_cart, cutoff):
    """Same as _neighbours but around an arbitrary Cartesian point."""
    P, C = atoms.get_positions(), atoms.cell[:]
    sh = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1)
                   for z in (-1, 0, 1)])
    v = (P[None, :, :] + sh[:, None, :] @ C - centre_cart).reshape(-1, 3)
    d = np.linalg.norm(v, axis=1)
    return v[(d > 1e-3) & (d < cutoff)]


def find_bonds_at(atoms, q, ops, cutoff=None, tol_shell=0.1):
    """Bond vectors from a fractional position, closed under its site group.

    No CrystalNN branch: it needs a real site in the structure, and the whole
    point here is that the position may be empty. Falls back to the first
    coordination shell, which is what find_bonds does when pymatgen is absent.
    """
    centre = np.asarray(q, float) @ np.array(atoms.cell[:])
    if cutoff is not None:
        nb = _neighbours_at(atoms, centre, cutoff)
    else:
        far = _neighbours_at(atoms, centre, 6.0)
        if len(far) == 0:
            return np.zeros((0, 3))
        d = np.linalg.norm(far, axis=1)
        nb = far[d < d.min() + tol_shell]
    if len(nb) == 0:
        return np.zeros((0, 3))
    return _symmetrize_bonds(nb, ops)


def parse_position(position):
    """Fractional coordinates from floats, ints, or strings.
    """
    from fractions import Fraction

    def one(v):
        if isinstance(v, (int, float, np.floating, np.integer)):
            return float(v)
        v = str(v).strip()
        try:
            return float(v)
        except ValueError:
            pass
        try:
            f = Fraction(v)
        except ValueError as e:
            raise ValueError(f"cannot parse position component {v!r}") from e
        if f.denominator > 10**6:
            f = f.limit_denominator(10**6)
        return float(f)

    if isinstance(position, str):
        position = position.split(",")
    return np.array([one(v) for v in position], dtype=float)


def build_at(atoms, position, shells, cutoff=None, prefix="", symprec=1e-4,
             seed=0, verbose=True, spacegroup=None, fallback="best_hybrid",
             weight_fn=None, label=None):
    """build(), for ONE Wyckoff position -- occupied or empty.

    Everything build does per site is geometric: site_group(cell, q) asks which
    operations fix q modulo lattice, which is defined at any point in the cell.
    So the SALC construction transfers unchanged to an empty Wyckoff position,
    which is exactly the obstructed-atomic-limit case an EBR search needs
    (`1/8,1/8,1/8:sp3` in the WannierBerri tutorial).

    Two things had to move:
      * find_bonds takes an atom index and prefers CrystalNN, which needs a real
        site. find_bonds_at takes a position and uses the coordination-shell
        fallback.
      * naming used ds.wyckoffs[i] and the element symbol. `label` replaces it.

    position : fractional coordinates, NUMERIC. A Wyckoff position with a free
        parameter (x,0,0) has no single site group until x is fixed, so pick a
        representative value -- any generic one works, since the stabiliser is
        constant along the orbit, but avoid values that accidentally land on a
        higher-symmetry position. |G_q| is printed so you can check.

    weight_fn : as in build, called as weight_fn(None, shells, ops) since there
        is no atom index. Pass None unless your weight can be evaluated at an
        arbitrary point.

    Returns (ProjectionsSet, SiteResult).
    """
    from irrep.spacegroup import SpaceGroup
    from wannierberri.symmetry.projections import Projection, ProjectionsSet

    if fallback not in FALLBACKS:
        raise ValueError(f"fallback must be one of {FALLBACKS}, got {fallback!r}")

    cell = (np.array(atoms.cell[:]), atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    sg = spacegroup if spacegroup is not None else SpaceGroup.from_cell(
        cell=cell, spinor=False, include_TR=True)

    q = parse_position(position) % 1.0
    sh = parse_shells(shells)
    ops = site_group(cell, q, symprec)
    reps = [shell_rep(sh, R) for R in ops]
    if label is None:
        label = "q" + "_".join(f"{x:.3f}".replace(".", "p").replace("-", "m")
                               for x in q)
    base = f"{prefix}{label}"

    res = SiteResult(-1, "", q, "", f"|G_q|={len(ops)}", sh)
    T = None
    if cutoff is not False:
        try:
            nb = find_bonds_at(atoms, q, ops, cutoff=cutoff)
            if len(nb):
                w = None if weight_fn is None else weight_fn(None, sh, ops)
                if fallback == "minimal_shell":
                    T, used = minimal_shell_hybrids(nb, ops, sh, seed=seed,
                                                    weight=w)
                    if len(used) < len(sh):
                        rest = "".join(NAME_OF[l] for l in sh if l not in used)
                        res.note = ("minimal-shell hybrid on "
                                    f"{''.join(NAME_OF[l] for l in used)}, "
                                    f"{rest} left as complement")
                else:
                    Tc, uniq = hybrids_from_bonds(
                        nb, ops, sh, seed=seed, weight=w,
                        optimize=(fallback == "best_hybrid"))
                    if uniq or fallback == "best_hybrid":
                        T = Tc
                        if not uniq:
                            res.note = "non-unique -> optimised hybrid"
        except ValueError as e:
            res.note = str(e)

    if T is not None:
        register(f"{base}_hyb", T, sh, ops)
        res.orbital_names, res.hybrid = [f"{base}_hyb"], True
        comp = null_space(T, rcond=1e-8).T
        if comp.shape[0]:
            for k, (M, d, m) in enumerate(isotypic_components(
                    [comp @ D @ comp.T for D in reps], seed=seed)):
                nm = f"{base}_rest{k}"
                register(nm, M @ comp, sh, ops)
                res.orbital_names.append(nm)
    elif fallback == "shells":
        if cutoff is not False and not res.note:
            res.note = "hybrids not unique -> plain shells"
        res.orbital_names = [NAME_OF[l] for l in sh]
    else:
        if cutoff is not False and not res.note:
            res.note = "hybrids not unique -> isotypic components"
        for k, (M, d, m) in enumerate(isotypic_components(reps, seed=seed)):
            nm = f"{base}_{k}"
            register(nm, M, sh, ops)
            res.orbital_names.append(nm)

    pset = ProjectionsSet()
    for nm in res.orbital_names:
        pset.add(Projection(position_num=[q], orbital=nm, spacegroup=sg,
                            rotate_basis=True))

    if verbose:
        tag = "bond-pointing hybrid" if res.hybrid else "isotypic components"
        print(f"{label} @ {np.round(q, 4)}  |G_q|={len(ops):2d}  "
              f"{''.join(NAME_OF[l] for l in sh)} -> {tag}: "
              f"{res.orbital_names}" + (f"   [{res.note}]" if res.note else ""))
        for nm in res.orbital_names:
            for member in wb_orb.orbitals_sets_dic.get(nm, []):
                terms = " ".join(f"{c:+.3f}|{o}>" for o, c
                                 in wb_orb.hybrids_coef.get(member, {}).items())
                print(f"      {member:24s} = {terms}")
    return pset, res


def build_trial_set(atoms, wyckoff_shells, spacegroup=None, both=True, **kw):
    """Alphabet of trial projections for EBRsearcher.

        wyckoff_shells = [([0, 0, 0], "sp"), ([0.125]*3, "sp"), ...]

    both=True adds BOTH granularities per entry: the bond hybrid (with its
    complement) and the irrep-pure isotypic components. Hybrids double as the
    initial guess, so a hybrid solution is directly usable; components are finer
    -- a composite like sp3 locks A1 and T2 in a fixed ratio and cannot express
    a manifold needing 2xA1 + 1xT2. EBRsearcher orders solutions by increasing
    num_wann, so the compact hybrid ones still surface first.

    Note register() writes into WannierBerri's global orbital dictionary, so
    names must stay unique across the whole alphabet; `label` per entry, or the
    position-derived default, takes care of that.
    """
    from wannierberri.symmetry.projections import ProjectionsSet
    out = ProjectionsSet()
    infos = []
    for entry in wyckoff_shells:
        pos, sh = entry[0], entry[1]
        lab = entry[2] if len(entry) > 2 else None
        modes = [("hyb", None)] if not both else [("hyb", None),
                                                  ("cmp", False)]
        for tag, cut in modes:
            p, r = build_at(atoms, pos, sh, cutoff=cut, spacegroup=spacegroup,
                            label=(f"{lab}_{tag}" if lab else None), **kw)
            for proj in p.projections:
                out.add(proj)
            infos.append(r)
    return out, infos


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
    fallback : what to do when the bond hybrids are NOT unique.
             'best_hybrid' (default) -- keep bond-pointing lobes, resolving the
                  leftover multiplicity mixing by maximising the objective:
                  bond directionality, or projectability if weight_fn is given.
             'minimal_shell' -- fill the bonds' irreps from the lowest shells
                  that can supply them (textbook sp3/sp2), leaving higher
                  shells whole as the complement.
             'shells' -- plain shell orbitals: same span, best-conditioned
                  starting gauge, no registration needed.
             'components' -- isotypic blocks.
             All four give the same SPAN and differ only in the starting gauge,
             so this never changes the physics -- only how fast Wannier90
             converges and how readable the initial orbitals are.
    weight_fn : callable(atom_index, shells, ops) -> (dim, dim) array or None
             Physical weight, used where symmetry leaves freedom. In practice
             `projectability_from_gpaw`: the sphere-projected charge matrix, in
             electrons. Keeps this module energy-agnostic -- the caller supplies
             the DFT information.
    select_threshold : float, optional
             If given (requires weight_fn), the requested shells are treated as
             CANDIDATES: at each site the AO space is split into individual
             irrep copies and only those whose projectability reaches the
             threshold are kept. Because the weight lies in the commutant its
             eigenvalues come in d_i-fold groups, so the cut always falls
             between whole copies and can never split a degenerate irrep.
             `nwann` then comes OUT of build (read pset.num_wann) instead of
             going in.

             UNITS: with the partial-wave metric the eigenvalue is electrons
             per orbital of that copy, so this is the same number as the `alpha`
             in a shell-resolved occupancy criterion -- 0.4 reproduces Zhang's
             wtol = 0.4*(2l+1), just resolved per irrep instead of per shell.
             Feed `available_shells(calc, i)` as the candidate set and let the
             threshold decide; that is strictly finer than choosing shells by
             hand, since it can keep t2g and drop eg.

             Bond hybrids are skipped for selected sites -- the selection
             changes the span, which is the point, whereas the hybrid/complement
             split only changes the gauge.
    """
    from irrep.spacegroup import SpaceGroup
    from wannierberri.symmetry.projections import Projection, ProjectionsSet

    if fallback not in FALLBACKS:
        raise ValueError(f"fallback must be one of {FALLBACKS}, got {fallback!r}")
    if select_threshold is not None and weight_fn is None:
        raise ValueError("select_threshold requires a weight_fn")

    cell = (np.array(atoms.cell[:]), atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if spacegroup is not None:
        # use the caller's SpaceGroup (e.g. SpaceGroup.from_gpaw(calc)) so the
        # Projections carry EXACTLY the symmetry of the DFT wavefunctions.
        # Site groups for the SALC analysis still come from spglib; that is
        # safe in one direction only (see check below).
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
    by_orbit = {}
    for key, spec in shells.items():
        if isinstance(key, (int, np.integer)):
            rep = ds.equivalent_atoms[int(key)]
            ls = set(parse_shells(spec))
            if rep in by_orbit and by_orbit[rep] != ls:
                warnings.warn(f"atoms {key} and a previous one are symmetry-"
                              "equivalent but were given different shells; "
                              "taking the union")
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

        if select_threshold is not None:
            # --- projectability SELECTION: changes the span, unlike every
            # gauge choice below. Keep only the copies the bands support.
            w_sel = weight_fn(i, sh, ops)
            if w_sel is None:
                raise ValueError("select_threshold requires a weight_fn that "
                                 f"returns a matrix for atom {i}")
            copies = isotypic_copies(reps, w_sel)
            keep = [(M, d, pj) for M, d, pj in copies if pj >= select_threshold]
            if not keep:
                res.note = (f"no copy reaches projectability {select_threshold} "
                            f"(best {copies[0][2]:.3f} e) -- site skipped")
                sites.append(res)
                if verbose:
                    print(f"{el}{i}: {res.note}")
                continue
            for k, (M, d, pj) in enumerate(keep):
                nm = f"{base}_sel{k}"
                register(nm, M, sh, ops)
                res.orbital_names.append(nm)
            res.note = (f"kept {len(keep)}/{len(copies)} copies "
                        f"({sum(d for _, d, _ in keep)} WF/site), p = "
                        f"{[round(pj, 3) for _, _, pj in keep]} e")
        elif cut is not False:
            try:
                nb = find_bonds(atoms, i, ops, cutoff=cut)
                if len(nb):
                    w = None if weight_fn is None else weight_fn(i, sh, ops)
                    if fallback == "minimal_shell":
                        T, used = minimal_shell_hybrids(nb, ops, sh, seed=seed,
                                                        weight=w)
                        if len(used) < len(sh):
                            rest = "".join(NAME_OF[l] for l in sh if l not in used)
                            res.note = ("minimal-shell hybrid on "
                                        f"{''.join(NAME_OF[l] for l in used)}, "
                                        f"{rest} left as complement")
                    else:
                        Tc, uniq = hybrids_from_bonds(
                            nb, ops, sh, seed=seed, weight=w,
                            optimize=(fallback == "best_hybrid"))
                        if uniq or fallback == "best_hybrid":
                            T = Tc
                            if not uniq:
                                res.note = ("non-unique -> optimised on "
                                            + ("projectability" if w is not None
                                               else "bond directionality"))
            except ValueError as e:
                res.note = str(e)

        if res.orbital_names:
            pass                       # selection already registered
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
                for member in wb_orb.orbitals_sets_dic.get(nm, []):
                    terms = " ".join(
                        f"{c:+.3f}|{o}>"
                        for o, c in wb_orb.hybrids_coef.get(member, {}).items())
                    print(f"      {member:24s} = {terms}")
    return pset, sites


# --------------------------------------------------------------------------
# GPAW -> WannierBerri real-harmonic map (derived, never tabulated)
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def gpaw_to_wb(lmax=2, npts=400, seed=0, tol=1e-6):
    """{l: S_l} with P_WB = P_GPAW @ S_l, S_l a SIGNED PERMUTATION.

    Both codes span the same space for each l, so Y_WB = Y_GPAW @ S. Evaluate
    both on a spiral of directions and least-squares fit S; if the conventions
    differ only by order, sign and per-shell scale (they should), every column
    of S has exactly one non-negligible entry.

    The per-l NORM is deliberately divided out, and this is the whole reason
    to derive the map rather than tabulate it. GPAW's Y are orthonormal on the
    sphere, so with the partial-wave metric M carries occupations in electrons.
    `_HARM` above uses an unnormalised convention whose ratio to the orthonormal
    one differs BETWEEN shells: 3.545 at l=0, 2.047 at l=1, 0.915 at l=2 -- a
    factor of ~15 in M between s and d. Keeping it would scale each l block by
    c_l^2 and each cross-l block by c_l*c_l', a similarity by a block scalar
    that COMMUTES with every D(R). The commutant check can therefore never see
    it, and it would silently corrupt every s:p:d hybrid ratio and every
    cross-shell projectability comparison. Rotations are unaffected by a
    uniform per-shell scale, so dropping it keeps M covariant AND in electrons.

    Order and signs, by contrast, generally DO break covariance and are usually
    caught by the commutant check -- though not always (a 1-dimensional irrep
    appearing once, or a site with trivial symmetry, is invariant under its own
    sign flip). Verified against the hand-written table this replaces: the
    permutations {0:[0], 1:[1,2,0], 2:[2,3,1,4,0]} and all-positive signs were
    correct. The scale was the silent error.

    Raises if the fit is not a signed permutation, or if the scale is not
    constant within a shell -- a per-m scale would break covariance and must
    never be silently absorbed.
    """
    from gpaw.spherical_harmonics import Y as _Y

    i = np.arange(npts) + 0.5
    phi = np.arccos(1 - 2 * i / npts)
    th = np.pi * (1 + 5 ** 0.5) * i                      # golden spiral
    d = np.column_stack([np.cos(th) * np.sin(phi),
                         np.sin(th) * np.sin(phi), np.cos(phi)])
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    d = d @ (Q * np.sign(np.linalg.det(Q)))              # kill axis alignment

    out = {}
    for l in range(lmax + 1):
        G = np.column_stack([_Y(l * l + m, d[:, 0], d[:, 1], d[:, 2])
                             for m in range(2 * l + 1)])
        W = _eval_lobes([l], d)
        S, *_ = np.linalg.lstsq(G, W, rcond=None)
        resid = np.linalg.norm(G @ S - W) / max(np.linalg.norm(W), 1e-30)
        if resid > 1e-6:
            raise RuntimeError(f"l={l}: GPAW and WB harmonics are not related "
                               f"by a linear map to {resid:.1e} -- unexpected")
        P = np.zeros_like(S)
        scale = []
        for c in range(2 * l + 1):
            col = S[:, c]
            k = int(np.argmax(np.abs(col)))
            rest = np.linalg.norm(np.delete(col, k))
            if rest > 1e-6 * max(abs(col[k]), 1e-30):
                raise RuntimeError(
                    f"l={l}, WB orbital {c}: GPAW column is a MIXTURE, not a "
                    f"signed permutation (off weight {rest:.2e}). The two codes "
                    "use genuinely different real-harmonic bases for this l.")
            P[k, c] = np.sign(col[k])
            scale.append(abs(col[k]))
        scale = np.array(scale)
        if scale.max() - scale.min() > tol * scale.mean():
            raise RuntimeError(
                f"l={l}: normalisation is not uniform across m "
                f"(scales {np.round(scale, 6)}). A per-m scale does NOT commute "
                "with D(R), so M would not be covariant. Fix _HARM.")
        out[l] = P
    return out


# --------------------------------------------------------------------------
# projectability weight from a DFT calculation
# --------------------------------------------------------------------------

def projectability_from_gpaw(calc, window, spin=0, atol=0.25,
                             n_select="window", verbose_channels=False,
                             backend=None):
    """Build a `weight_fn` for `build` from a (NSCF) GPAW calculation.

    Returns callable(iatom, shells, ops) -> M, with

        M_mm' = sum_{nk in window} w_k <g_m|psi_nk><psi_nk|g_m'>   [electrons]

    where g are the all-electron partial waves truncated at rcut. The PAW
    projections <p_i|psi~> alone are NOT this: they are coefficients in the
    non-orthogonal partial-wave set, and using them as amplitudes leaves a
    setup-dependent scale factor (rescale phi -> 2 phi and p -> p/2 and the
    "weight" changes by 4x). `collect_projections` restores the metric by
    whitening with O^{1/2}, O_jj' = int_0^rc phi_j phi_j' r^2 dr -- the piece
    GPAW's raw_orbital_LDOS omits and VASP's SPHPRO includes.

    Consequences here: trace(M) is a number of electrons, so `select_threshold`
    and a shell-occupancy `alpha` are the same number; and the cross-l blocks
    are on a common scale, so maximising tr(T W T^T) over an s/p/d mixing in
    `hybrids_from_bonds` optimises something real.

    Deliberately NOT normalised per state. Dividing each band's contribution by
    its total bound-projector weight (Mulliken-style) restores a sum rule but
    redistributes the STATE COUNT rather than the charge, so a nearly-free-
    electron band with 5% sphere charge hands its full weight to whatever sliver
    of character it has -- inflating exactly the diffuse states that should be
    flagged as unrepresentable. Handle the interstitial deficit at the coverage
    level (`sphere_projectability.coverage`), where it cancels in a ratio, not
    by renormalising M.

    M commutes with the site-symmetry representation, so it lies in the
    commutant -- exactly the freedom symmetry leaves -- which is why it can fix
    the multiplicity mixing without disturbing anything symmetry determined.

    backend : module exposing collect_projections/channels (default:
        ao_projectability, i.e. Loewdin AOs with p in [0, 1]).  Pass
        sphere_projectability to fall back to PAW sphere charge.  M's units
        follow the backend: electrons of projectability, or sphere charge.

    n_select : 'window' | 'valence' | 'first' | 'sum'
        Which bound channel to use when a setup has several for the same l
        (Ti 3p semicore vs 4p valence; Ba 5s vs 6s).
          'window' (default) -- the channel carrying the most weight INSIDE the
                     window. Semicore and valence are split in energy, so the
                     window itself selects: no n heuristic, and it adapts if
                     the window moves.
          'valence' -- highest n.   'first' -- lowest n.
          'sum'     -- all bound channels of that l, with their mutual overlap
                     handled by O's off-diagonal (the old version added
                     |<p|psi>|^2 over channels, which double-counted it).
                     Single-l only: a projector onto a multi-dimensional
                     channel subspace has no single column, so there is no
                     cross-l block for it.
        Unbound channels (n < 0) are PAW completeness functions, never used.
        Run `describe_site(calc, iatom)` to see what a setup offers.
    """
    if backend is None:
        # MUST match the backend the driver used -- get_proj_set should pass it
        # explicitly.  Defaults to sphere because the AO path does not yet
        # support plane-wave wavefunctions.
        from .projectability import sphere_projectability as backend
    S_wb = gpaw_to_wb()
    _cache = {}

    def _collect(iatom, shells_l):
        key = (iatom, tuple(shells_l))
        if key not in _cache:
            eps_kn, wk_k, V = backend.collect_projections(
                calc, spin=spin, shells=shells_l, atom_indices=[iatom])
            # GPAW m-order -> WB m-order (signed permutation, unit scale)
            V = {k: v @ S_wb[k[1]] for k, v in V.items()}
            _cache[key] = (eps_kn, wk_k, V)
        return _cache[key]

    def weight_fn(iatom, shells, ops):
        shells_l = parse_shells(shells)
        for l in shells_l:
            if l not in S_wb:
                warnings.warn(f"no GPAW->WB m-ordering for l={l} (mapped up to "
                              f"l={max(S_wb)}): no weight for this site")
                return None
        if n_select == "sum" and len(shells_l) > 1:
            raise ValueError(
                "n_select='sum' is defined for a single l only (no cross-l "
                "block for a multi-dimensional channel subspace). Use 'window' "
                "-- with the metric in place it is nearly equivalent, because "
                "whitening orthonormalises the channels.")

        chan = {l: [(n, None) for n in ns]
                for l, ns in backend.channels(calc, iatom).items()}
        missing = [l for l in shells_l if not chan.get(l)]
        if missing:
            warnings.warn(
                f"atom {iatom}: no BOUND projector for l="
                f"{[NAME_OF[l] for l in missing]} (available: "
                f"{[NAME_OF[l] for l in sorted(chan)]}) -- no weight. "
                "Run describe_site(calc, iatom).")
            return None

        eps_kn, wk_k, V = _collect(iatom, shells_l)
        sel = (eps_kn >= window[0]) & (eps_kn <= window[1])
        if not sel.any():
            warnings.warn(f"atom {iatom}: no bands inside window {window} (eV) "
                          "-- check the window and its units")
            return None
        wts = wk_k[:, None] * sel

        cols, picked = [], {}
        for l in shells_l:
            v = V[(iatom, l)]                            # (nk, nb, nchan, 2l+1)
            ns = [n for n, _ in chan[l]]
            tr = np.einsum("knim,knim->kni", v.conj(), v).real
            tr = (wts[:, :, None] * tr).sum(axis=(0, 1))          # per channel
            if n_select == "sum":
                cols.append(v.reshape(v.shape[0], v.shape[1], -1))
                picked[NAME_OF[l]] = ("sum", float(tr.sum()))
                continue
            if n_select == "window":
                j = int(np.argmax(tr))
            elif n_select == "valence":
                j = int(np.argmax(ns))
            elif n_select == "first":
                j = int(np.argmin(ns))
            else:
                raise ValueError("n_select must be window/valence/first/sum, "
                                 f"got {n_select!r}")
            cols.append(v[:, :, j, :])                            # (nk, nb, 2l+1)
            picked[NAME_OF[l]] = (ns[j], float(tr[j]))

        A = np.concatenate(cols, axis=-1)
        M = np.einsum("kn,knm,knp->mp", wts, A.conj(), A).real
        M = 0.5 * (M + M.T)

        if verbose_channels:
            print(f"  atom {iatom}: {{l: (n, electrons in window)}} -> {picked}")

        nrm0 = np.linalg.norm(M)
        if nrm0 == 0:
            warnings.warn(f"atom {iatom}: zero weight in window {window}")
            return None

        # Symmetrisation is a RECONSTRUCTION, not a correction: the full-BZ sum
        # of M is symmetric, a weighted IBZ sum is not (for a scalar the k
        # weights suffice; for a MATRIX you must also average over the star).
        # A moderate deviation is expected for any *-irred.gpw run. If this
        # starts firing, expand the projections to the full BZ rather than
        # raising atol -- a star average has limits.
        reps = [shell_rep(shells_l, R) for R in ops]
        Msym = sum(D @ M @ D.T for D in reps) / len(reps)
        dev = np.linalg.norm(Msym - M) / nrm0
        if dev > atol:
            warnings.warn(
                f"atom {iatom}: weight deviates from the commutant by "
                f"{dev:.2f} before symmetrisation. Expected on an "
                "irreducible-BZ calculation; suspect the m-ordering or signs "
                "only if large (>~0.5) -- check gpaw_to_wb().")
        return Msym

    return weight_fn


def available_shells(calc, iatom, lmax=2):
    """l values for which the GPAW setup of `iatom` carries a BOUND projector.

    Use as the CANDIDATE shell set when running `build` with
    `select_threshold`: it is what the pseudopotential can actually describe,
    so it is the honest upper bound on what projectability can select from.
    """
    return sorted(l for l in bound_channels(calc.setups[iatom]) if l <= lmax)


def describe_site(calc, iatom):
    """Bound channels and the partial-wave overlap metric for one atom.

    Use when a projectability disagrees with what you expect: the usual causes
    are several bound projectors sharing one l (semicore + valence), an l with
    no bound state at all, or a metric diagonal far from 1 (the orbital is
    barely contained inside rcut, so its sphere charge understates it).
    """
    print(f"atom {iatom} ({calc.atoms[iatom].symbol})")
    for l, njs in sorted(bound_channels(calc.setups[iatom]).items()):
        print(f"  l={NAME_OF.get(l, l)}  bound (n, j) = {njs}")
    describe_metric(calc, iatom)