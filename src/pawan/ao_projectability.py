"""Projectability without spheres: p_nk in [0, 1], the QE/Vitale quantity.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "collect_projections",
    "channel_weights",
    "channels",
    "channel_ceiling",
    "basis_index_map",
    "projectability",
    "frozen_from_projectability",
]

L_NAME = {0: "s", 1: "p", 2: "d", 3: "f"}
L_NUM = {v: k for k, v in L_NAME.items()}


def _as_l(l):
    return L_NUM[l] if isinstance(l, str) else int(l)


# --------------------------------------------------------------------------
# basis bookkeeping
# --------------------------------------------------------------------------

def basis_index_map(calc):
    """M -> (atom, j, l, n, m) for the phit_j basis, in GPAW's own ordering."""
    out = []
    for a, setup in enumerate(calc.setups):
        phit_j = getattr(setup, "phit_j", None)
        if not phit_j:
            cands = [k for k in dir(setup)
                     if "phit" in k or "basis" in k.lower()]
            raise RuntimeError(
                f"setup for atom {a} ({getattr(setup, 'symbol', '?')}) has no "
                f"usable phit_j. This is NOT a bad setup -- in plane-wave mode "
                "GPAW never loads an LCAO basis, so the atomic orbitals simply "
                f"are not attached to the Setup object. Related attributes "
                f"present: {cands}. Use backend='sphere', or load a basis "
                "explicitly, or take the Wannier90 .amn route.")
        n_j = getattr(setup, "n_j", None)
        for j, phit in enumerate(phit_j):
            try:
                l = int(phit.get_angular_momentum_number())
            except AttributeError:              # older/newer Spline API
                l = int(setup.l_j[j])
            n = int(n_j[j]) if (n_j is not None and j < len(n_j)
                                and n_j[j] is not None) else j + 1
            for m in range(2 * l + 1):
                out.append((a, j, l, n, m))
    return out


def channels(calc, iatom):
    """{l: [n, ...]} in the same order as V's channel axis."""
    out = {}
    for a, j, l, n, m in basis_index_map(calc):
        if a == iatom and m == 0:
            out.setdefault(l, []).append(n)
    return out


def channel_ceiling(calc, iatom, l, per_orbital=False):
    """Tr(Pi) = n_channels * (2l+1), the rank of the projector.
    """
    l = _as_l(l)
    n_chan = len(channels(calc, iatom).get(l, []))
    if n_chan == 0:
        return float("nan")
    return 1.0 if per_orbital else float(n_chan * (2 * l + 1))


def _inv_sqrt_herm(S, rcond=1e-8):
    w, U = np.linalg.eigh(S)
    keep = w > rcond * max(float(w.max().real), 1e-30)
    if not keep.all():
        warnings.warn(f"{int((~keep).sum())} near-null direction(s) in the AO "
                      "overlap -- the basis is linearly dependent at this "
                      "k-point; those directions are projected out.")
    winv = np.zeros_like(w)
    winv[keep] = w[keep] ** -0.5
    return (U * winv) @ U.conj().T


# --------------------------------------------------------------------------
# the projection
# --------------------------------------------------------------------------

def ensure_positions(calc):
    """A RESTARTED calculator has wfs.spos_ac = None.

    GPAW defers set_positions() until a calculation is triggered, so after
    GPAW('x.gpw') the wavefunctions and projections are loaded but spos_ac is
    not set, and get_bfs() dies on `assert len(spos_ac) == len(self.sphere_a)`.
    mode='all' controls what is WRITTEN; it has nothing to do with this.

    Set it directly rather than calling calc.initialize_positions(): that also
    calls wfs.set_positions(), which reallocates the projection arrays and can
    discard the P_ani read from file -- which the PAW augmentation term needs.
    """
    wfs = calc.wfs
    if getattr(wfs, "spos_ac", None) is None:
        wfs.spos_ac = calc.atoms.get_scaled_positions() % 1.0
    return wfs.spos_ac


def _wfs_kind(calc):
    return type(calc.wfs).__name__


def collect_projections(calc, spin=0, shells=(0, 1, 2), atom_indices=None,
                        atol=1e-6):
    """Loewdin AO coefficients, shaped exactly like the sphere backend's V.

    Returns (eps_kn, wk_k, V) with
        V[(iatom, l)] : complex (nk, nb, n_channels_of_that_l, 2l+1)
    in GPAW's real-harmonic order -- apply salc.gpaw_to_wb()[l] on the last axis
    before combining with WannierBerri rotation matrices, exactly as before.

    sum over (channel, m) of |V|^2 is the projectability of that band onto that
    (atom, l), and the sum over ALL keys is p_nk <= 1.
    """
    ensure_positions(calc)

    kind = _wfs_kind(calc)
    if "PW" in kind:
        raise NotImplementedError(
            f"calc.wfs is {kind}: get_lcao_projections_HSP integrates the basis "
            "functions against kpt.psit_nG on the REAL-SPACE grid "
            "(bfs.integrate2), but in plane-wave mode psit_nG holds PW "
            "coefficients, so it either shape-errors or returns garbage.\n\n"
            "Options, in order of effort:\n"
            "  1. backend='sphere' -- works today, correct Tr(Pi) ceilings, "
            "only limitation is that interstitial charge is invisible.\n"
            "  2. PW-native AO projection: build the AOs in the PW basis with "
            "gpaw.wavefunctions.pw.PWLFC([s.phit_j for s in calc.wfs.setups], "
            "calc.wfs.pd), integrate against kpt.psit_nG for <phit|psit~>, then "
            "add the PAW term sum_ij <phit|p_i> dO_ii <p_j|psit~> using "
            "setup.dO_ii, and Loewdin against the true <Phi|Phi>. This is the "
            "right answer and needs writing and validating against a live "
            "calculator.\n"
            "  3. gpaw.wannier.wannier90.Wannier90(calc, seed, orbitals_ai=...)"
            ".write_projections() is PW-native and writes the same A matrix to "
            "seed.amn; read it back and orthogonalise.\n\n"
            "Self-test for whichever route: p_nk <= 1 always, and for Si the "
            "occupied bands should come out at p ~ 0.99.")

    try:
        from gpaw.lcao.projected_wannier import get_lcao_projections_HSP
    except ImportError as e:                     # pragma: no cover
        raise ImportError(
            "gpaw.lcao.projected_wannier.get_lcao_projections_HSP not found. "
            "Fallback: gpaw.wannier.wannier90.Wannier90(calc, seed=..., "
            "orbitals_ai=...).write_projections() writes the same A matrix to "
            "seed.amn; read it and orthogonalise with (A^dag A)^{-1/2}, which "
            "is only valid with many empty bands -- check that sum_n p_nk is "
            "meaningfully below the number of projectors."
        ) from e

    out = get_lcao_projections_HSP(calc, bfs=None, spin=spin,
                                   projectionsonly=False)
    V_qnM, S_qMM = np.asarray(out[0]), np.asarray(out[2])

    wk_k = np.asarray(calc.get_k_point_weights())
    nk = len(wk_k)
    if V_qnM.shape[0] != nk:
        raise RuntimeError(
            f"{V_qnM.shape[0]} k-blocks but {nk} k-point weights -- the "
            "projection routine is using a different k-set (spin folding?)")
    eps_kn = np.array([calc.get_eigenvalues(kpt=k, spin=spin)
                       for k in range(nk)])

    bmap = basis_index_map(calc)
    if len(bmap) != V_qnM.shape[2]:
        raise RuntimeError(
            f"basis map has {len(bmap)} functions but V has {V_qnM.shape[2]} "
            "columns -- phit_j ordering does not match the BasisFunctions "
            "object; inspect calc.setups[a].phit_j")

    shells = tuple(_as_l(l) for l in shells)
    if atom_indices is None:
        atom_indices = range(len(calc.atoms))
    atom_indices = set(atom_indices)

    # column layout per (atom, l): (n_channels, 2l+1)
    cols = {}
    for M, (a, j, l, n, m) in enumerate(bmap):
        cols.setdefault((a, l), {}).setdefault(j, {})[m] = M
    layout = {}
    for (a, l), byj in cols.items():
        js = sorted(byj)
        layout[(a, l)] = np.array([[byj[j][m] for m in range(2 * l + 1)]
                                   for j in js])

    nb = V_qnM.shape[1]
    keys = [k for k in sorted(layout) if k[0] in atom_indices and k[1] in shells]
    V = {k: np.zeros((nk, nb, layout[k].shape[0], k[1] * 2 + 1), complex)
         for k in keys}

    pmax = 0.0
    for q in range(nk):
        # Loewdin over the COMPLETE basis; filtering happens after
        Vt = V_qnM[q] @ _inv_sqrt_herm(S_qMM[q])          # (nb, nM)
        pmax = max(pmax, float((Vt.conj() * Vt).real.sum(axis=1).max()))
        for k in keys:
            V[k][q] = Vt[:, layout[k]]

    if pmax > 1 + atol:
        raise RuntimeError(
            f"projectability reaches {pmax:.4f} > 1. Pi is an orthogonal "
            "projector, so this cannot happen unless V and S are inconsistent "
            "-- check that S_qMM is the overlap of the SAME basis, in the same "
            "k-ordering, as V_qnM.")
    return eps_kn, wk_k, V


def channel_weights(V):
    """{(a, l): (nk, nb)} projectability onto that (atom, l)."""
    return {key: np.einsum("knim,knim->kn", v.conj(), v).real
            for key, v in V.items()}


def projectability(w):
    """p_nk = sum over all channels.  Equals 1 minus the part of the band that
    no atom-centred orbital in the set can represent."""
    return sum(w.values())


def frozen_from_projectability(eps_kn, p_nk, emin, emax, p_froz=0.95):
    """Largest emax' <= emax with every state in [emin, emax'] at p >= p_froz.

    Unlike the sphere `atomicity` gate this is safe to switch on: p is a true
    projection fraction with an absolute meaning, so p < 0.95 says the state
    genuinely lies outside the span and freezing it WILL cost you spread.
    Vitale's pmax_thr default.
    """
    inside = (eps_kn >= emin) & (eps_kn <= emax)
    bad = inside & (p_nk < p_froz)
    return float(eps_kn[bad].min() - 1e-6) if bad.any() else float(emax)