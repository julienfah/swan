"""Sphere-projected charge from GPAW PAW projections, with the metric restored.

The raw quantity `<p_i|psi~>` is a coefficient in the NON-orthogonal partial-wave
set, used by get_orbital_ldos as if it were an amplitude in an orthonormal one.
Rescale phi_i -> 2 phi_i (so p_i -> p_i/2) and |<p_i|psi>|^2 changes by 4x: the
number carries a convention factor set by the setup generator, which is why no
threshold transfers between codes or even between setups.

Inside the augmentation sphere PAW gives

    |psi_n> = sum_{i in ALL}  |phi_i> <p_i|psi~_n>

so the projection onto the span of the BOUND partial waves {phi_b} is

    p_n = <psi_n| Pi_b |psi_n>,   Pi_b = sum_bb' |phi_b> (O_bb^-1)_bb' <phi_b'|
        = sum_b |v_bn|^2   with   v = O_bb^{-1/2} O_{b,all} P_all
    O_j1j2 = int_0^rc phi_j1 phi_j2 r^2 dr

Note BOTH the inverse square root and the rectangular block: unbound channels
enter the METRIC even though nothing is ever projected onto them.  Dropping
them (the earlier O_bb^{+1/2} P_b form) leaves a quadratic form that is not a
projector expectation and has no upper bound -- the symptom was occupations
above 100% of a shell.  The two forms coincide when a setup has no unbound
channels, which is why the old numbers were close but not bounded.

p_n <= 1 always.  It is however a systematic UNDERESTIMATE of atomic character:
everything outside rc is invisible.  For a covalent solid the deficit is large
(Si valence bands sit near q = 0.45).  `channel_ceiling` corrects this to first
order using the free-atom orbital's own norm inside rc, which removes the rcut
dependence; the residue, interstitial BOND charge, is irreducible for any
sphere method, VASP included.  For a genuine [0, 1] projectability use the
atomic-orbital route (Wannier90 A-matrices, Loewdin), where the orbital tail
outside rc is included.

Serial only: kpt_u / P_ani are distributed over k, band and atom partitions.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np

__all__ = [
    "collect_projections",
    "channel_weights",
    "bound_channels",
    "channel_ceiling",
    "channels",
    "equivalent_atoms",
    "symmetrize_over_orbits",
    "select_orbitals",
    "select_by_rank",
    "frozen_band_count",
    "select_orbitals_by_sphere_charge",
    "atomicity",
    "selected_fraction",
    "frozen_from_atomicity",
    "auto_atom_thresh",
    "occupied_spread",
    "describe_metric",
]

L_NAME = {0: "s", 1: "p", 2: "d", 3: "f"}
L_NUM = {v: k for k, v in L_NAME.items()}


def _as_l(l):
    return L_NUM[l] if isinstance(l, str) else int(l)


# --------------------------------------------------------------------------
# setup inspection
# --------------------------------------------------------------------------

def bound_channels(setup):
    """{l: [(n, j), ...]} for BOUND partial waves only, sorted by n.

    Unbound channels (n < 0) are PAW completeness/scattering functions with no
    atomic-orbital meaning.  Nothing is projected onto them -- but they DO
    appear in the metric, see _sphere_metric.
    """
    n_j = getattr(setup, "n_j", None)
    if n_j is None:
        n_j = [1] * len(setup.l_j)
    out = {}
    for j, l in enumerate(setup.l_j):
        n = n_j[j]
        n = 1 if n is None else int(n)
        if n > 0:
            out.setdefault(int(l), []).append((n, j))
    for l in out:
        out[l].sort()
    return out


def _column_offsets(setup):
    """j -> first column of that channel in P_ani (columns run m fastest)."""
    off, off_j = 0, []
    for l in setup.l_j:
        off_j.append(off)
        off += 2 * int(l) + 1
    return off_j


def _sphere_metric(setup):
    """{l: (O_bb, O_b_all, js_bound, js_all)}, O_j1j2 = int_0^rc phi phi r^2 dr.

    Convention check: for a bound valence partial wave O_jj is the fraction of
    the atomic orbital norm inside rc, so it must land in (0, 1].  GPAW's
    rgd.dv_g carries a 4*pi that must come out; if the diagonal comes back
    ~4*pi too large we divide it out and warn rather than silently returning
    numbers wrong by 12.6.
    """
    data = getattr(setup, "data", setup)
    phi_jg = getattr(data, "phi_jg", None)
    chan = bound_channels(setup)

    all_by_l = defaultdict(list)
    for j, l in enumerate(setup.l_j):
        all_by_l[int(l)].append(j)

    if phi_jg is None:
        warnings.warn(
            f"setup {getattr(setup, 'symbol', '?')} has no all-electron partial "
            "waves (norm-conserving pseudopotential?).  Falling back to the "
            "identity metric -- the absolute scale is NOT meaningful.")
        return {l: (np.eye(len(js)), np.eye(len(js)), [j for _, j in js],
                    [j for _, j in js]) for l, js in chan.items()}

    phi_jg = np.asarray(phi_jg)
    rgd = getattr(setup, "rgd", None) or data.rgd
    r_g = np.asarray(rgd.r_g)
    dv_g = getattr(rgd, "dv_g", None)
    w_g = (np.asarray(dv_g) / (4.0 * np.pi) if dv_g is not None
           else r_g ** 2 * np.asarray(rgd.dr_g))
    rcuts = [float(rc) for rc in getattr(setup, "rcut_j", [])
             if rc is not None and rc > 0]
    rc = max(rcuts) if rcuts else float(r_g[-1])
    w_g = np.where(r_g <= rc, w_g, 0.0)

    def ovl(j1, j2):
        return float(np.sum(phi_jg[j1] * phi_jg[j2] * w_g))

    out = {}
    for l, njs in chan.items():
        js_b = [j for _, j in njs]
        js_all = all_by_l[l]
        O_bb = np.array([[ovl(a, b) for b in js_b] for a in js_b])
        O_bb = 0.5 * (O_bb + O_bb.T)
        O_ba = np.array([[ovl(a, b) for b in js_all] for a in js_b])
        out[l] = (O_bb, O_ba, js_b, js_all)

    diag = (np.concatenate([np.diag(O) for O, _, _, _ in out.values()])
            if out else np.array([1.0]))
    if diag.size and diag.max() > 1.5:
        if diag.max() / (4.0 * np.pi) <= 1.5:
            warnings.warn("sphere metric was ~4*pi too large -- rgd volume "
                          "convention differs from the assumed one; dividing "
                          "it out. Verify with describe_metric().")
            out = {l: (a / (4 * np.pi), b / (4 * np.pi), c, d)
                   for l, (a, b, c, d) in out.items()}
        else:
            warnings.warn(
                f"sphere metric diagonal max = {diag.max():.3g}, expected <= 1. "
                "The radial normalisation convention of this GPAW version is "
                "not what this code assumes -- run describe_metric() before "
                "trusting any absolute number.")
    return out


def _inv_sqrt_psd(O, rcond=1e-10):
    w, U = np.linalg.eigh(O)
    keep = w > rcond * max(w.max(), 1e-30)
    winv = np.zeros_like(w)
    winv[keep] = w[keep] ** -0.5
    return (U * winv) @ U.T


def channel_ceiling(calc, iatom, l, per_orbital=False):
    """Tr(Pi) = n_channels * (2l+1) -- the RANK of the projector being summed.

    N is a sum of <psi_nk|Pi|psi_nk> over bands, so as the band set approaches
    completeness it approaches Tr(Pi), and Tr(Pi) is the rank: for
    Pi = sum_bb' |phi_b>(O^-1)_bb'<phi_b'|,

        Tr(Pi) = sum_bb' (O^-1)_bb' O_b'b = Tr(O^-1 O) = n_channels

    per m, so n_channels*(2l+1) overall.  Note this is independent of rcut -- a
    projector's rank does not care how much of each orbital sits inside the
    sphere.

    THE PREVIOUS VERSION USED (2l+1)*O_vv AND WAS WRONG, in two compounding
    ways.  O_vv is the ceiling for a SINGLE state that happens to be the free
    atom's orbital; N is a TRACE over many states, and the two normalisations
    are not the same question.  And O_vv was taken from the highest-n bound
    channel, which for Ba (5s,6s) and Ti (3p,4p) is the diffuse outer one:
    O_vv came out at 0.04 and 0.066, ceilings of 0.042 and 0.198, and fills of
    2056% and 1943%.  Every fill landed above 100%, so alpha had nothing left
    to discriminate with.

    Consequence for alpha: fill = N / Tr(Pi) is now "what fraction of this
    projector's rank the bands in the window cover", genuinely in [0, 1].
    Expect to want alpha near 0.3, not 0.7.
    """
    l = _as_l(l)
    n_chan = len(bound_channels(calc.setups[iatom]).get(l, []))
    if n_chan == 0:
        return float("nan")
    return 1.0 if per_orbital else float(n_chan * (2 * l + 1))


def describe_metric(calc, iatom):
    """Print the bound channels, the sphere overlap and the shell ceilings."""
    setup = calc.setups[iatom]
    print(f"atom {iatom} ({calc.atoms[iatom].symbol}), rcut_j = {setup.rcut_j}")
    for l, (O_bb, O_ba, js_b, js_all) in sorted(_sphere_metric(setup).items()):
        ns = [n for n, _ in bound_channels(setup)[l]]
        n_unb = len(js_all) - len(js_b)
        print(f"  l={L_NAME.get(l, l)}  bound n = {ns}  "
              f"(+{n_unb} unbound channel(s), used in the metric only)")
        print("   O_bb  =", np.array2string(O_bb, precision=4,
                                            prefix="   O_bb  = "))
        print(f"   ceiling = Tr(Pi) = {len(js_b)} x {2 * l + 1} = "
              f"{len(js_b) * (2 * l + 1)}   "
              f"(free-atom norms inside rc: {np.round(np.diag(O_bb), 4)})")


# --------------------------------------------------------------------------
# one pass over the k-points
# --------------------------------------------------------------------------

def _kpt(wfs, k, spin):
    for kpt in wfs.kpt_u:
        if kpt.k == k and kpt.s == spin:
            return kpt
    raise RuntimeError(f"k-point {k}, spin {spin} not held locally -- "
                       "load the .gpw with communicator=serial_comm")


def _P_of(kpt, a):
    P_ani = getattr(kpt, "P_ani", None)
    if P_ani is None:                       # newer GPAW: Projections object
        P_ani = kpt.projections
    return np.asarray(P_ani[a])             # (nbands, ni)


def collect_projections(calc, spin=0, shells=(0, 1, 2), atom_indices=None):
    """One pass over k-points.

    Returns
    -------
    eps_kn : (nk, nb) eigenvalues in eV
    wk_k   : (nk,) k-point weights (sum to 1)
    V      : {(iatom, l): complex (nk, nb, n_bound, 2l+1)}
             = O_bb^{-1/2} O_{b,all} <p_all|psi~>, in GPAW's real-harmonic
             order -- apply salc.gpaw_to_wb()[l] on the last axis before
             combining with WannierBerri rotation matrices.

    The sum of |V|^2 over (n_bound, m) is the projection of the band onto the
    bound partial waves inside the sphere, and is bounded by 1.
    """
    wfs = calc.wfs
    if getattr(wfs.world, "size", 1) > 1:
        raise RuntimeError(
            "projections are distributed over k/band/atom partitions; run this "
            "analysis on a calculator loaded with communicator=serial_comm")

    shells = tuple(_as_l(l) for l in shells)
    wk_k = np.asarray(calc.get_k_point_weights())
    nk, nb = len(wk_k), wfs.bd.nbands
    if atom_indices is None:
        atom_indices = range(len(calc.atoms))
    atom_indices = list(atom_indices)

    eps_kn = np.array([calc.get_eigenvalues(kpt=k, spin=spin)
                       for k in range(nk)])

    plan, V = {}, {}
    for a in atom_indices:
        setup = calc.setups[a]
        off_j = _column_offsets(setup)
        for l, (O_bb, O_ba, js_b, js_all) in _sphere_metric(setup).items():
            if l not in shells:
                continue
            Q = _inv_sqrt_psd(O_bb) @ O_ba          # (n_bound, n_all)
            cols = np.array([[off_j[j] + t for t in range(2 * l + 1)]
                             for j in js_all])      # (n_all, 2l+1)
            plan[(a, l)] = (Q, cols)
            V[(a, l)] = np.zeros((nk, nb, len(js_b), 2 * l + 1), complex)

    for k in range(nk):
        kpt = _kpt(wfs, k, spin)
        P_cache = {}
        for (a, l), (Q, cols) in plan.items():
            if a not in P_cache:
                P_cache[a] = _P_of(kpt, a)
            Pl = P_cache[a][:, cols]                # (nb, n_all, 2l+1)
            V[(a, l)][k] = np.einsum("ij,njm->nim", Q, Pl)

    return eps_kn, wk_k, V


def channel_weights(V):
    """{(a, l): (nk, nb)} sphere charge of that character in each band."""
    return {key: np.einsum("knim,knim->kn", v.conj(), v).real
            for key, v in V.items()}


# --------------------------------------------------------------------------
# symmetry bookkeeping
# --------------------------------------------------------------------------

def channels(calc, iatom):
    """{l: [n, ...]} in the same order as V's channel axis (backend protocol)."""
    return {l: [n for n, _ in njs]
            for l, njs in bound_channels(calc.setups[iatom]).items()}


def equivalent_atoms(atoms, symprec=1e-4):
    """spglib's orbit representative for every atom."""
    import spglib
    cell = (np.array(atoms.cell[:]), atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    return spglib.get_symmetry_dataset(cell, symprec=symprec).equivalent_atoms


def symmetrize_over_orbits(calc, N, symprec=1e-4, warn_tol=0.02, verbose=True):
    """Average a per-atom scalar over each symmetry orbit. Returns (N, eq).

    THIS IS NOT SMOOTHING -- it is the exact full-BZ answer.  A weighted sum
    over the irreducible BZ does not reconstruct a per-ATOM quantity: the
    operations mapping k -> Sk also permute the atoms, so

        w^(a)_{n,Sk} = w^(S^-1 a)_{nk}

    and only the ORBIT SUM is invariant under IBZ weighting.  Individual members
    of an orbit come out unequal by several percent, and if the threshold falls
    inside that spread one member of a Wyckoff orbit is selected and the others
    are not -- which then disagrees with `build`, which works per inequivalent
    site, and the two nwann counts diverge.

    The intra-orbit spread is a free diagnostic: if it exceeds warn_tol the
    symmetry spglib finds at `symprec` is not the symmetry GPAW used for the
    wavefunctions.
    """
    eq = equivalent_atoms(calc.atoms, symprec)
    groups = defaultdict(list)
    for (a, l) in N:
        groups[(int(eq[a]), l)].append((a, l))

    out = {}
    for (rep, l), members in groups.items():
        vals = np.array([N[m] for m in members], float)
        mean = float(vals.mean())
        if len(vals) > 1 and abs(mean) > 1e-12:
            spread = float(vals.max() - vals.min()) / abs(mean)
            if spread > warn_tol and verbose:
                warnings.warn(
                    f"orbit of atom {rep}, l={L_NAME[l]}: members spread by "
                    f"{spread:.1%} ({np.round(vals, 4)}) before symmetrisation. "
                    "A few percent is the normal IBZ artefact; much more means "
                    f"symprec={symprec} disagrees with the symmetry GPAW used.")
        for m in members:
            out[m] = mean
    return out, eq


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def select_orbitals(calc, eps_kn, wk_k, w, window, alpha=0.5, symprec=1e-4,
                    verbose=True, use_ceiling=True, ceiling_fn=None,
                    channels_fn=None):
    """Keep a channel when its window occupancy reaches alpha of its ceiling.

        N^{a,l} = sum_{nk in window} w_k * w^{a,l}_nk        [electrons]
        fill    = N / ceiling,   ceiling = (2l+1) * O_vv
        keep    if fill > alpha

    N is symmetrised over Wyckoff orbits BEFORE the cut, so the selection can
    never split an orbit (see symmetrize_over_orbits).

    alpha is element-independent here, so it is NOT Zhang's 0.4: that number is
    a fraction of the full shell (2l+1) on VASP's scale.  Against the ceiling,
    expect roughly 0.5-0.7 for shells that clearly belong.  Set
    use_ceiling=False to recover the raw alpha*(2l+1) criterion.

    Returns (selected_orbitals, nwann, N, eq) with selected_orbitals in the
    [(iatom, (n, l_str)), ...] format and n the REAL principal quantum number.
    """
    ceiling_fn = ceiling_fn or channel_ceiling
    channels_fn = channels_fn or channels
    emin, emax = window
    sel = (eps_kn >= emin) & (eps_kn <= emax)

    N = {key: float((wk_k[:, None] * w_kn * sel).sum())
         for key, w_kn in w.items()}
    N, eq = symmetrize_over_orbits(calc, N, symprec=symprec, verbose=verbose)

    selected, nwann = [], 0
    for (a, l) in sorted(N):
        deg = 2 * l + 1
        ceil = ceiling_fn(calc, a, l) if use_ceiling else float(deg)
        fill = N[(a, l)] / ceil if ceil > 0 else 0.0
        keep = fill > alpha
        if keep:
            ns = channels_fn(calc, a).get(l, [])
            selected.append((a, (ns[-1] if ns else None, L_NAME[l])))
            nwann += deg
        if verbose:
            print(f"  {'KEEP' if keep else 'drop'}  atom {a:3d} "
                  f"{calc.atoms[a].symbol:>2s} l={L_NAME[l]}   "
                  f"N={N[(a, l)]:7.4f} e / ceiling {ceil:6.4f} "
                  f"= {fill:6.1%}   vs alpha={alpha:.0%}")
    if verbose:
        print(f"  -> nwann = {nwann}")
    return selected, nwann, N, eq


def select_by_rank(calc, eps_kn, wk_k, w, window, target, symprec=1e-4,
                   verbose=True, ceiling_fn=None, channels_fn=None,
                   max_frac=1.5):
    """Rank channels by coverage, add whole orbits until nwann reaches target.

    WHY NOT AN ABSOLUTE THRESHOLD.  fill = N/Tr(Pi) is a fraction of the
    projector rank, but the bands only put a fraction q_ref of their charge
    inside ANY sphere, and q_ref is chemistry-dependent: 0.42 for Si, 0.17 for a
    Zintl phase like CaMg2Bi2 where the cations are ionised and the conduction
    charge is interstitial.  So the same physical situation -- "this is the
    dominant valence channel" -- reads as 34% in Si and 28% in CaMg2Bi2, and no
    single alpha separates them from the noise in both.  Lowering alpha until
    Bi p survives also lets nothing else in, so only the valence manifold gets
    Wannierised.

    Ranking is scale-free: it only needs the ordering to be right, which the
    sphere charge does get right, rather than the absolute value, which it does
    not.  The size then comes from physics instead of from a threshold --
    `target` is the number of bands the frozen window must hold, so nwann cannot
    come out too small to span it.

    Orbits are added whole (all symmetry-equivalent atoms at once), so the
    selection can never split a Wyckoff orbit and disagree with `build`.

    max_frac : refuse to exceed max_frac * target Wannier functions; if the
        target cannot be reached without that, report it rather than piling on
        channels that contribute nothing.
    """
    ceiling_fn = ceiling_fn or channel_ceiling
    channels_fn = channels_fn or channels
    emin, emax = window
    sel = (eps_kn >= emin) & (eps_kn <= emax)

    N = {key: float((wk_k[:, None] * w_kn * sel).sum())
         for key, w_kn in w.items()}
    N, eq = symmetrize_over_orbits(calc, N, symprec=symprec, verbose=verbose)

    # group by orbit: one decision per (orbit representative, l)
    orbits = defaultdict(list)
    for (a, l) in N:
        orbits[(int(eq[a]), l)].append(a)
    score = {}
    for (rep, l), members in orbits.items():
        ceil = ceiling_fn(calc, rep, l)
        score[(rep, l)] = N[(members[0], l)] / ceil if ceil > 0 else 0.0

    order = sorted(score, key=lambda k: -score[k])
    selected, nwann = [], 0
    for (rep, l) in order:
        members = sorted(orbits[(rep, l)])
        add = len(members) * (2 * l + 1)
        if nwann >= target:
            break
        if nwann + add > max_frac * target:
            continue
        for a in members:
            ns = channels_fn(calc, a).get(l, [])
            selected.append((a, (ns[-1] if ns else None, L_NAME[l])))
        nwann += add

    if verbose:
        chosen = {(a, L_NUM[ls]) for a, (n, ls) in selected}
        print(f"  target {target} WF (frozen manifold at the worst k)")
        for (rep, l) in order:
            mark = "KEEP" if (rep, l) in chosen else "drop"
            print(f"  {mark}  {calc.atoms[rep].symbol:>2s}(orbit of atom "
                  f"{rep}) l={L_NAME[l]}  coverage {score[(rep, l)]:6.1%}  "
                  f"x{len(orbits[(rep, l)])} sites")
        print(f"  -> nwann = {nwann}")
        if nwann < target:
            print(f"  WARNING: only {nwann} WF for {target} frozen bands -- "
                  "widen the candidate shells or lower max_frac")
    return selected, nwann, N, eq


def frozen_band_count(eps_kn, emin, emax):
    """max over k of the number of bands in [emin, emax] -- the manifold the
    Wannier functions must span, and therefore the lower bound on nwann."""
    return int(max(int(((e >= emin) & (e <= emax)).sum()) for e in eps_kn))


# --------------------------------------------------------------------------
# adequacy diagnostics
# --------------------------------------------------------------------------

def atomicity(eps_kn, w, e_fermi, ref="occupied"):
    """qhat = q / q_ref, q = total charge inside all spheres.

    q's ceiling is not 1 -- it is 1 minus the interstitial fraction, which
    depends on the setup radii and on the bonding (Si valence sits near 0.45).
    Referencing it to the median q over occupied bands calibrates that deficit
    per system: qhat ~ 1 means "as atom-like as the states I am confident
    about", and the fall-off marks where atom-centred orbitals stop working.

    A calibration, not a first-principles bound.  Returns (qhat, q, q_ref).
    """
    q = sum(w.values())
    mask = (eps_kn <= e_fermi) if ref == "occupied" else np.ones_like(eps_kn, bool)
    q_ref = float(np.median(q[mask]))
    if q_ref <= 0:
        raise ValueError("reference sphere charge is zero")
    return q / q_ref, q, q_ref


def selected_fraction(w, selected_keys):
    """p/q: of the charge the candidate set can see, how much was selected.

    Only informative when the selection DROPPED something substantial.  If the
    candidate and selected sets nearly coincide this is identically 1 and says
    nothing -- use `atomicity` as the primary diagnostic.
    """
    keys = set(selected_keys)
    q = sum(w.values())
    p = sum(v for k, v in w.items() if k in keys)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(q > 0, p / np.maximum(q, 1e-30), 1.0)


def auto_atom_thresh(eps_kn, qhat, e_fermi, quantile=0.02, safety=0.95):
    """A threshold the OCCUPIED manifold passes by construction.

    q varies substantially WITHIN the occupied bands -- the low-lying s-like
    band is more contained than the upper p-like ones, and in Si the spread is
    about 0.72 to 1.17 around the median.  Any fixed number inside that spread
    (0.9, say) cuts through the valence manifold, which is never what the frozen
    window should do: the occupied bands are precisely the ones you are
    representing, so they must all be frozen.

    Taking a low quantile of the occupied distribution makes the criterion mean
    "as atom-like as the WORST state I am already confident about" instead of an
    arbitrary constant, and it adapts to how covalent the system is.
    """
    occ = qhat[eps_kn <= e_fermi]
    if occ.size == 0:
        return 0.5
    return float(np.quantile(occ, quantile) * safety)


def occupied_spread(eps_kn, qhat, e_fermi):
    """(min, median, max) of qhat over occupied bands -- the calibration range."""
    occ = qhat[eps_kn <= e_fermi]
    if occ.size == 0:
        return (float("nan"),) * 3
    return float(occ.min()), float(np.median(occ)), float(occ.max())


def frozen_from_atomicity(eps_kn, qhat, emin, emax, thresh=0.9):
    """Largest emax' <= emax with every band in [emin, emax'] at qhat >= thresh.

    Call this with emin >= E_F.  Scanning from the bottom of the outer window
    lets a dip inside the valence manifold truncate the frozen window there,
    which is always wrong -- the occupied bands are what the Wannier functions
    must reproduce, so they are frozen regardless of how atom-like they look.
    Low atomicity among the occupied bands is a diagnostic about the PROJECTION
    SET, not a reason to shrink the window.
    """
    inside = (eps_kn >= emin) & (eps_kn <= emax)
    bad = inside & (qhat < thresh)
    return float(eps_kn[bad].min() - 1e-6) if bad.any() else float(emax)


# backwards-compatible alias (the function is now backend-agnostic)
select_orbitals_by_sphere_charge = select_orbitals