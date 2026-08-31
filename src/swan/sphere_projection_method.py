"""Drop-in replacement for `Zhang_projection_method`.

Same signature, same 4-tuple return -- `get_proj_set` needs no edits except the
shells_dict fix noted at the bottom of this docstring.

    selected_orbitals, out_win, frozen_win, nwann

STRUCTURE.  The selection window and nwann are solved together, then the frozen
window follows:

  1. SELECTION window, seeded from per-channel gaps, then refined to a FIXED
     POINT against nwann (refine='fixed_point', default).
  2. OUTER window = the converged selection window.
  3. FROZEN window, from atomicity, then hard-capped at N_k <= nwann.

ON THE FIXED POINT.  The map is  W -> S(W) -> nwann -> W' = f(nwann).  Widening
W can only raise every occupation, so the selected set can only grow: S is
monotone in W.  Raising nwann can only raise the energy at which K*nwann bands
are available: f is monotone in nwann.  The composite is therefore monotone
non-decreasing, which means the iterates are a MONOTONE SEQUENCE -- it cannot
oscillate, and it converges to a fixed point or runs into a boundary.  The only
boundary is nwann = 0, which is absorbing.

So a collapse is not an instability, it is the statement that alpha admits no
non-trivial fixed point.  That is what happened to Si at alpha = 0.4: the
largest window gives Si p a fill of 34.8%, so p is dropped, nwann falls to 2,
and the descent continues to 0.  At alpha = 0.3 the same iteration converges to
nwann = 8 in one step.  The loop was doing its job; the threshold was wrong (and
the truncated metric was inflating the numbers it was compared against).

`refine='gap'` keeps the seeded window and skips the refinement -- Zhang's own
structure, where `wight()` integrates over a fixed gap-bounded window and the
`while 1.2*nwann < melenum` loop moves `engmax` afterwards without re-selecting.
Use it when the gap window is well defined and you want alpha's meaning to be
independent of K; see the caveat under `refine` below.

WINDOW DETECTION is per channel, as in the original.  A gap in the TOTAL
spectrum exists only where no band exists at any k, which in anything past a
two-atom cell is almost nowhere; a semicore shell is detectable because it is
separated by a gap in ITS OWN channel weight even when other atoms have bands
throughout that region.  The pDOS is replaced by exact eigenvalues carrying
channel weight, so `gap_thres` keeps its meaning ("how wide a gap am I willing
to cross going down") without fighting a smearing width.  Note that testing a
smeared DOS against 1e-10 moved each band edge by 6.8 sigma -- 0.34 eV at
width=0.05 -- so a 0.6 eV gap like Si's had NEGATIVE width and was invisible.

`dos_kwargs` is accepted for signature compatibility and is UNUSED: occupations
are summed per band over exact eigenvalues, which also removes grid truncation
and window-edge blur.

REQUIRED EDIT in get_proj_set, replacing the symbol-keyed shells_dict:

    from .sphere_projectability import equivalent_atoms
    eq = equivalent_atoms(calc.atoms)
    shells_dict = {}
    for iatom, (n, l) in selected_orbitals:
        shells_dict.setdefault(int(eq[iatom]), set()).add(l)
    shells_dict = {k: sorted(v) for k, v in shells_dict.items()}

Keying by SYMBOL collapses inequivalent sites of the same species onto whichever
one comes last in the list, which is where the "22 WF but windows sized for 23"
assertion came from.  `build` accepts integer keys precisely for this, and
applies them to the whole orbit.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from ase.dft.bandgap import bandgap
from gpaw import GPAW
from gpaw.mpi import serial_comm, world

from . import ao_projectability as _ao
from . import sphere_projectability as _sph
from .sphere_projectability import (
    atomicity,
    frozen_band_count,
    select_by_rank,
    auto_atom_thresh,
    frozen_from_atomicity,
    occupied_spread,
    select_orbitals,
    selected_fraction,
)

# backend protocol: each module exposes collect_projections / channel_weights /
# channels / channel_ceiling with identical signatures and return shapes
BACKENDS = {"ao": _ao, "sphere": _sph}

L_NAME = {0: "s", 1: "p", 2: "d", 3: "f"}
L_NUM = {v: k for k, v in L_NAME.items()}


def _p(*a):
    if world.rank == 0:
        print(*a)


# --------------------------------------------------------------------------
# per-channel window detection
# --------------------------------------------------------------------------

def band_blocks(eps_kn, gap_thres=0.1):
    """Groups of consecutive BAND INDICES separated by a true band gap.

    A gap between band n and n+1 exists iff

        min_k eps[k, n+1] - max_k eps[k, n] > gap_thres

    which is the textbook definition and is immune to how densely the mesh
    samples each band.

    WHY NOT POOLED EIGENVALUES.  The previous version sorted all eigenvalues
    together and split wherever consecutive values differed by more than
    gap_thres.  On any realistic mesh that shreds the manifold: Si has ~116
    valence eigenvalues spread over ~12 eV, so the mean spacing WITHIN a band is
    about 0.1 eV -- the same size as gap_thres.  Every band fragments into
    dozens of one-point "blocks", the block containing the VBM is a single
    eigenvalue, and the window bottom comes out at the VBM instead of -6.65 eV.

    That is also, belatedly, why the smeared DOS never failed.  I argued the
    Gaussian was a bug because it displaces band edges by ~6.8 sigma, and it
    does -- but it was simultaneously doing load-bearing work: convolving a
    discrete k-sample into a continuous manifold, so a band reads as one
    connected region rather than N isolated points.  Removing the smearing
    without replacing that function is what broke the window.  Working on band
    indices removes the need for either.

    Returns [(n_start, n_stop, e_lo, e_hi), ...], n_stop inclusive.
    """
    lo_n = eps_kn.min(axis=0)
    hi_n = eps_kn.max(axis=0)
    nb = len(lo_n)
    cuts = [n for n in range(nb - 1) if lo_n[n + 1] - hi_n[n] > gap_thres]
    starts = [0] + [c + 1 for c in cuts]
    stops = cuts + [nb - 1]
    return [(a, b, float(lo_n[a]), float(hi_n[b]))
            for a, b in zip(starts, stops)]


def selection_window(calc, eps_kn, w, e_fermi, gap_thres, wmin=0.01,
                     min_width=0.0, verbose=True):
    """The band block holding the occupied manifold, up to the block holding the
    lowest empty state.

    This is what the smeared-DOS version computed in effect, and it subsumes the
    per-channel scan: semicore sits in its own block, so it is excluded by
    construction rather than by a rule about interval edges, and no channel can
    drag the edge anywhere.  Per-channel weights are still reported, because
    seeing which channels live in which block is the useful diagnostic -- but
    they no longer SET the window, which is what kept going wrong.
    """
    blocks = band_blocks(eps_kn, gap_thres)
    occupied = np.where((eps_kn <= e_fermi).any(axis=0))[0]
    empty = np.where((eps_kn > e_fermi).any(axis=0))[0]
    if occupied.size == 0:
        raise ValueError("no occupied states below E_F")
    n_hi_occ = int(occupied.max())
    n_lo_emp = int(empty.min()) if empty.size else n_hi_occ

    blk_lo = next(b for b in blocks if b[0] <= n_hi_occ <= b[1])
    blk_hi = next(b for b in blocks if b[0] <= n_lo_emp <= b[1])
    lo, hi = blk_lo[2], blk_hi[3]

    if verbose:
        _p(f"  band blocks (gap_thres={gap_thres} eV):")
        for a, b, e0, e1 in blocks:
            mark = ""
            if (a, b) == blk_lo[:2]:
                mark += "  <- holds the occupied manifold"
            if (a, b) == blk_hi[:2] and blk_hi is not blk_lo:
                mark += "  <- holds the lowest empty state"
            _p(f"    bands {a:3d}-{b:3d}   {e0:9.3f} .. {e1:9.3f} eV{mark}")
        for (a, l), w_kn in sorted(w.items()):
            here = [f"{i}" for i, blk in enumerate(blocks)
                    if (w_kn[:, blk[0]:blk[1] + 1] > wmin).any()]
            _p(f"    atom {a:3d} {calc.atoms[a].symbol:>2s} l={L_NAME[l]}: "
               f"weight > {wmin} in block(s) {','.join(here) or 'none'}")
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# band-count constraints (exact, per k)
# --------------------------------------------------------------------------

def emax_from_band_count(eps_kn, emin, nwann, K=1.2):
    """Smallest emax with N_k(emin, emax) >= ceil(K * nwann) at EVERY k.

    This is why K survives the move to projectability: disentanglement needs
    MORE states than it keeps, and the extra ones are by construction the poorly
    projectable ones.  A pure projectability criterion would shrink the outer
    window onto the frozen window and leave nothing to disentangle with.

    Exact per k.  Integrating a smeared DOS to K*nwann is a BZ-AVERAGED proxy,
    and an average can be satisfied while one k-point fails.
    """
    target = int(np.ceil(K * nwann))
    lim = -np.inf
    for eps_n in eps_kn:
        e = np.sort(eps_n[eps_n >= emin])
        if len(e) < target:
            raise ValueError(
                f"only {len(e)} bands above {emin:.3f} eV at one k-point but "
                f"{target} needed for nwann={nwann}, K={K} -- increase nbands "
                "in the NSCF")
        lim = max(lim, e[target - 1])
    return float(lim)


def cap_frozen_window(eps_kn, emin, emax, nwann, pad=1e-6):
    """Largest emax' <= emax with N_k(emin, emax') <= nwann at every k.

    Degeneracy-safe: the cut lands strictly below the first band that would push
    any k over the count, so a degenerate multiplet is never split.
    """
    lim = float(emax)
    for eps_n in eps_kn:
        e = np.sort(eps_n[eps_n >= emin])
        if len(e) > nwann:
            lim = min(lim, float(e[nwann]) - pad)
    return max(lim, float(emin))


# --------------------------------------------------------------------------
# fixed point between the selection window and nwann
# --------------------------------------------------------------------------

def refine_window(calc, eps_kn, wk_k, w, sel_win, alpha, K, symprec=1e-4,
                  max_iter=10, verbose=True, bk=None):
    """Iterate  W -> select -> nwann -> W' = f(nwann)  to a fixed point.

    Monotone, so the sequence cannot oscillate (see the module docstring).  It
    converges, or it descends into nwann = 0 -- which means alpha is above the
    largest fill any channel reaches, and is reported as such rather than as a
    numerical failure.

    Returns (selected, nwann, N, eq, window).
    """
    bk = bk or _sph
    lo, hi = sel_win
    seen, last = [], None
    for it in range(max_iter):
        if verbose:
            _p(f"\n-- pass {it}: window {lo:.3f} .. {hi:.3f} eV")
        selected, nwann, N, eq = select_orbitals(
            calc, eps_kn, wk_k, w, (lo, hi), alpha=alpha, symprec=symprec,
            verbose=(world.rank == 0 and verbose),
            ceiling_fn=bk.channel_ceiling, channels_fn=bk.channels)

        if nwann == 0:
            fills = {k: N[k] / bk.channel_ceiling(calc, *k) for k in N}
            best = max(fills, key=fills.get)
            hint = ("" if last is None else
                    f" The previous pass held nwann={last[0]} over "
                    f"{last[1]:.3f} .. {last[2]:.3f} eV.")
            raise ValueError(
                f"alpha={alpha:.0%} admits no non-trivial fixed point: the "
                f"descent reached nwann=0. Best channel is atom {best[0]} "
                f"l={L_NAME[best[1]]} at {fills[best]:.1%} of its ceiling, so "
                f"alpha must be below that.{hint} This is a THRESHOLD problem, "
                "not a convergence problem -- the iteration is monotone.")

        state = (nwann, round(hi, 9))
        if state in seen:
            if verbose:
                _p(f"   fixed point: nwann={nwann}, window top {hi:.3f} eV")
            return selected, nwann, N, eq, (lo, hi)
        seen.append(state)
        last = (nwann, lo, hi)

        hi_new = emax_from_band_count(eps_kn, lo, nwann, K=K)
        if abs(hi_new - hi) < 1e-6:
            if verbose:
                _p(f"   fixed point: nwann={nwann}, window top {hi:.3f} eV")
            return selected, nwann, N, eq, (lo, hi)
        hi = hi_new

    warnings.warn(
        f"no fixed point in {max_iter} passes (nwann now {nwann}). The map is "
        "monotone, so this means the iterates are still moving -- raise "
        "max_iter, or use refine='gap'.")
    return selected, nwann, N, eq, (lo, hi)


# --------------------------------------------------------------------------
# the method
# --------------------------------------------------------------------------

def sphere_projection_method(
    K=1.2,
    out_dir="test",
    in_dir="test",
    seed=None,
    calc=None,
    dos_kwargs=None,          # signature compatibility only; UNUSED
    gap_thres=5.0,
    maximize_fw=False,
    objective_wd=None,
    comm=serial_comm,
    # --- new, all optional ---
    alpha=0.5,
    backend="sphere",
    select="rank",
    refine="fixed_point",
    max_iter=10,
    spin=0,
    shells=(0, 1, 2),
    p_froz=0.95,               # backend='ao': Vitale's pmax_thr
    atom_thresh=None,          # backend='sphere': None = diagnostic only
    wmin=0.01,
    symprec=1e-4,
    plot=True,
):
    """Select projections by sphere-projected charge; size the windows exactly.

    backend : 'sphere' (default) or 'ao'.

        The default is 'sphere' because the AO path currently only supports
        real-space (FD) wavefunctions -- see ao_projectability.collect_projections
        for the plane-wave situation and the three ways out.

        'ao'      Loewdin-orthogonalised pseudo-atomic orbitals via
                  get_lcao_projections_HSP.  p_nk in [0, 1] BY CONSTRUCTION,
                  ceiling = 2l+1 exactly, alpha absolute, frozen window from
                  p >= 0.95 (Vitale).  Needs the .gpw written with mode='all'.
        'sphere'  PAW sphere charge from P_ani.  Needs no wavefunctions, but
                  cannot see interstitial charge (55% of Si's valence), so the
                  ceiling is a free-atom correction and alpha and the frozen
                  window both need per-system calibration.

        Use 'sphere' only when the wavefunctions are unavailable.

    alpha : fraction of a channel's CEILING, where for 'sphere' the ceiling is
        (2l+1) * O_vv and O_vv is the free-atom valence orbital's norm inside
        rcut.  Dividing by it removes the rcut dependence, so alpha is
        element-independent -- which means it is NOT Zhang's 0.4, a fraction of
        the full shell.  Expect 0.5-0.7 for shells that clearly belong.
        Calibrate on Si / GaAs / Cu / SrVO3 before trusting it on anything new,
        and recalibrate after any change to the metric.

    refine : 'fixed_point' (default) solves the selection window and nwann
        together; 'gap' keeps the seeded gap window and selects once.

        CAVEAT on 'fixed_point': at the fixed point the selection window is set
        by K*nwann, so alpha's meaning becomes K-dependent -- raising K widens
        the window, raises every occupation, and can pull in another channel.
        K is a disentanglement parameter and should not decide which orbitals
        exist.  Cheap check: run K=1.2 and K=1.5 and confirm `selected` is
        unchanged.  If it moves, the selection is sitting on a threshold edge
        and 'gap' is the safer choice for that system.

    atom_thresh : None (default) = the atomicity gate is OFF and qhat is
        reported as a diagnostic only.  The frozen window is then the target,
        constrained solely by N_k <= nwann -- Zhang's structure, and standard
        practice.

        Turning it on ('auto' or a float) applies a HARD PER-STATE gate, and
        qhat is too crude for that.  The Vitale et al. criterion it imitates
        uses a true projectability p_nk in [0, 1] where p > 0.95 means the state
        is genuinely inside the span; qhat is a sphere charge over an
        empirically calibrated reference, with no absolute meaning.  Because the
        rule is "every state in the window must pass", ONE outlier at ONE
        k-point truncates everything -- and Si's low conduction bands sit near
        qhat = 0.4 while being perfectly well described by sp3 antibonding
        orbitals.  Use it only when you already know the projection set is
        marginal and you want to be told where it fails.

    gap_thres : how wide a BAND GAP you are willing to cross going down from the
        occupied manifold.  Default 5.0 eV.

        It no longer has anything to do with k-mesh spacing (band_blocks works
        on band indices), and it is no longer bounded above by the fundamental
        gap.  Its only job is to separate "another valence block I want"
        (GaAs As-4s sits ~4 eV below the p manifold) from "semicore I do not"
        (Bi 5d sits ~13 eV below).  That window is wide, so one value covers a
        lot of chemistry -- verified on Si / GaAs / semicore-Bi, where
        everything from 5 to 8 eV gives the right answer for all three:

            gap_thres   Si        GaAs        Bi
                  0.1   -6.59     -6.79  X    -4.98  X
                  3.0   -6.59     -6.79  X    -9.00
                  5.0   -6.59    -12.46      -9.00
                 15.0   -6.59    -12.46     -22.59  X  (pulls in 5d)

        Raise it if a valence block is being left out; lower it if semicore is
        getting in.  The block listing printed at startup tells you which.

    Returns (selected_orbitals, out_win, frozen_win, nwann).
    """
    if calc is None:
        calc = GPAW(f"{in_dir}/{seed}/{seed}-nscf-irred.gpw", txt=None,
                    communicator=comm)
    if dos_kwargs is not None:
        warnings.warn("dos_kwargs is unused: occupations are summed per band, "
                      "not integrated over a smeared DOS grid")

    e_fermi = calc.get_fermi_level()
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)}")
    bk = BACKENDS[backend]
    eps_kn, wk_k, V = bk.collect_projections(calc, spin=spin, shells=shells)
    w = bk.channel_weights(V)
    p_nk = sum(w.values())
    _p(f"Fermi level: {e_fermi:.4f} eV   "
       f"{eps_kn.shape[0]} k-points x {eps_kn.shape[1]} bands")

    _p(f"backend = {backend!r};  max total projectability = "
       f"{float(p_nk.max()):.4f}")
    if backend == "sphere" and float(p_nk.max()) > 1.0 + 1e-6:
        warnings.warn(
            f"total sphere charge reaches {float(p_nk.max()):.3f} > 1. The "
            "projector metric is inconsistent -- check describe_metric().")

    # ---- 1. selection window, per channel, FIXED -------------------------
    _p("\nper-channel windows:")
    sel_win = selection_window(calc, eps_kn, w, e_fermi, gap_thres,
                               wmin=wmin, verbose=True)
    if objective_wd is not None:
        sel_win = (objective_wd[0], max(objective_wd[1], sel_win[1]))
    _p(f"selection window: {sel_win[0]:.3f} .. {sel_win[1]:.3f} eV")

    # ---- 2. selection and outer window, solved together ------------------
    if select == "rank":
        froz_target = (objective_wd[1] if objective_wd is not None
                       else e_fermi + 2.0)
        target = frozen_band_count(eps_kn, sel_win[0], froz_target)
        _p(f"\nranked selection (target = {target} bands in "
           f"{sel_win[0]:.2f} .. {froz_target:.2f} eV at the worst k):")
        selected, nwann, N, eq = select_by_rank(
            calc, eps_kn, wk_k, w, sel_win, target, symprec=symprec,
            verbose=(world.rank == 0), ceiling_fn=bk.channel_ceiling,
            channels_fn=bk.channels)
        out_win = (float(sel_win[0]),
                   emax_from_band_count(eps_kn, sel_win[0], nwann, K=K))
    elif refine == "fixed_point":
        selected, nwann, N, eq, out_win = refine_window(
            calc, eps_kn, wk_k, w, sel_win, alpha, K, symprec=symprec,
            max_iter=max_iter, verbose=True, bk=bk)
    elif refine == "gap":
        _p("\nchannel selection (fixed gap window):")
        selected, nwann, N, eq = select_orbitals(
            calc, eps_kn, wk_k, w, sel_win, alpha=alpha, symprec=symprec,
            verbose=(world.rank == 0), ceiling_fn=bk.channel_ceiling,
            channels_fn=bk.channels)
        if nwann == 0:
            fills = {k: N[k] / bk.channel_ceiling(calc, *k) for k in N}
            best = max(fills, key=fills.get)
            raise ValueError(
                f"no channel reaches alpha={alpha:.0%} of its ceiling over the "
                f"gap window {sel_win}. Best is atom {best[0]} "
                f"l={L_NAME[best[1]]} at {fills[best]:.1%} -- lower alpha, or "
                "widen the window with gap_thres.")
        out_win = (float(sel_win[0]),
                   emax_from_band_count(eps_kn, sel_win[0], nwann, K=K))
    else:
        raise ValueError(f"refine must be 'fixed_point' or 'gap', got {refine!r}")
    out_win = (float(out_win[0]), float(out_win[1]))

    # ---- 4. frozen window ------------------------------------------------
    keys = [(a, L_NUM[l_str]) for a, (n, l_str) in selected]
    qhat, q, q_ref = atomicity(eps_kn, w, e_fermi)
    r = selected_fraction(w, keys)

    froz_min = objective_wd[0] if objective_wd is not None else out_win[0]
    if objective_wd is not None:
        froz_max, target = objective_wd[1], "objective_wd"
    elif maximize_fw:
        froz_max, target = out_win[1], "maximize_fw"
    else:
        froz_max, target = e_fermi + 2.0, "E_F + 2 eV"

    scan_from = max(froz_min, e_fermi)

    if backend == "ao":
        # p is a true projection fraction, so this gate is safe to switch on
        froz_p = _ao.frozen_from_projectability(eps_kn, p_nk, scan_from,
                                                froz_max, p_froz=p_froz)
        occ = p_nk[eps_kn <= e_fermi]
        _p(f"\noccupied projectability: min {occ.min():.4f}, "
           f"mean {occ.mean():.4f};  first state below p_froz={p_froz} above "
           f"E_F is at {froz_p:.3f} eV")
        if occ.min() < p_froz:
            _p(f"   NOTE: {int((occ < p_froz).sum())} OCCUPIED state(s) are "
               f"below p_froz -- the projection set does not span the valence "
               "manifold. They are frozen anyway (they must be reproduced); "
               "this is a signal to add orbitals, not to move the window.")
        if objective_wd is None:
            if froz_p < froz_max:
                _p(f"   projectability -> frozen window narrowed from "
                   f"{froz_max:.3f} to {froz_p:.3f}")
            froz_max = max(froz_p, scan_from)
        diag_thresh = p_froz
        qhat = p_nk                    # the plot axis is p itself
    else:
        q_lo, q_med, q_hi = occupied_spread(eps_kn, qhat, e_fermi)
        if atom_thresh == "auto":
            atom_thresh = auto_atom_thresh(eps_kn, qhat, e_fermi)
        diag_thresh = (atom_thresh if isinstance(atom_thresh, float)
                       else auto_atom_thresh(eps_kn, qhat, e_fermi))
        froz_atom = frozen_from_atomicity(eps_kn, qhat, scan_from, froz_max,
                                          thresh=diag_thresh)
        _p(f"\noccupied atomicity spans {q_lo:.3f} .. {q_hi:.3f} "
           f"(median 1.000); at threshold {diag_thresh:.3f} the first band "
           f"below it above E_F is at {froz_atom:.3f} eV")
        if atom_thresh is None:
            _p("   atomicity gate OFF (diagnostic only) -- frozen window set "
               "by the target and the N_k <= nwann cap")
        elif objective_wd is None:
            if froz_atom < froz_max:
                _p(f"   atomicity gate ON -> frozen window narrowed from "
                   f"{froz_max:.3f} to {froz_atom:.3f}")
            froz_max = max(froz_atom, scan_from)

    capped = cap_frozen_window(eps_kn, froz_min, froz_max, nwann)
    bound_by = target
    if capped < froz_max - 1e-6:
        bound_by = f"N_k <= nwann={nwann} cap"
        _p(f"   N_k <= nwann={nwann} caps the frozen window at {capped:.3f} eV "
           f"(wanted {froz_max:.3f}). If this is far below your target the "
           "PROJECTION SET is too small, not the window.")
    elif (backend == "ao" or atom_thresh is not None) and froz_max < (
            objective_wd[1] if objective_wd is not None else
            (out_win[1] if maximize_fw else e_fermi + 2.0)) - 1e-6:
        bound_by = "projectability gate" if backend == "ao" else "atomicity gate"
    frozen_win = (float(froz_min), float(capped))

    # INVARIANT: the frozen window must lie strictly inside the outer window.
    # wannierberri checks this ("Frozen bands should be included in the selected
    # bands") and both edges here come from EIGENVALUES, so an exact tie plus a
    # >= on one side and a > on the other is enough to fail it. Pad outward.
    pad = 1e-4
    lo = min(out_win[0], frozen_win[0]) - pad
    hi = max(out_win[1], frozen_win[1]) + pad
    if (lo, hi) != out_win:
        _p(f"   outer window widened to {lo:.4f} .. {hi:.4f} eV so it strictly "
           "contains the frozen window")
    out_win = (float(lo), float(hi))
    assert out_win[0] < frozen_win[0] and frozen_win[1] < out_win[1], \
        f"frozen {frozen_win} not strictly inside outer {out_win}"

    inside = (eps_kn >= frozen_win[0]) & (eps_kn <= frozen_win[1])
    qhat_in = float(np.mean(qhat[inside])) if inside.any() else float("nan")

    _p(f"\nSelected orbitals: "
       f"{[(calc.atoms[a].symbol, a, l) for a, (n, l) in selected]}")
    _p(f"Outer window : {out_win[0]:.4f} .. {out_win[1]:.4f} eV")
    _p(f"Frozen window: {frozen_win[0]:.4f} .. {frozen_win[1]:.4f} eV "
       f"(target {target}; bound by {bound_by})")
    _p(f"nwann = {nwann};  reference sphere charge q_ref = {q_ref:.3f} "
       f"(interstitial deficit {1 - q_ref:.0%});  mean atomicity in the "
       f"frozen window = {qhat_in:.3f}")

    if seed is not None:
        d = Path(f"{out_dir}/{seed}")
        d.mkdir(parents=True, exist_ok=True)
        if plot:
            _plot(d / f"{seed}-projectability.png", eps_kn, qhat, r, e_fermi,
                  out_win, frozen_win, seed, diag_thresh, backend)
        if world.rank == 0:
            with open(d / "selection.txt", "w") as f:
                f.write(f"alpha={alpha}  K={K}  gap_thres={gap_thres}  "
                        f"atom_thresh={atom_thresh}\n")
                f.write(f"selection window = {sel_win}\n")
                f.write(f"outer  = {out_win}\nfrozen = {frozen_win}\n")
                f.write(f"nwann  = {nwann}\nq_ref  = {q_ref:.4f}\n\n")
                for (a, l), n_e in sorted(N.items()):
                    c = bk.channel_ceiling(calc, a, l)
                    f.write(f"{'KEEP' if (a, l) in keys else 'drop'}  "
                            f"atom {a:3d} {calc.atoms[a].symbol:>2s} "
                            f"l={L_NAME[l]}  {n_e:7.4f} e / {c:6.4f} "
                            f"= {n_e / c:6.1%}\n")

    return selected, out_win, frozen_win, nwann


# keep the old name working at the call site
Zhang_projection_method = sphere_projection_method


# --------------------------------------------------------------------------

def _plot(path, eps_kn, qhat, r, e_fermi, out_win, frozen_win, seed, thresh,
          backend="ao"):
    """Atomicity (the real signal) and the selected fraction (only if < 1).

    No NaN floor and no interstitial flag: the old qmin=0.05 cut was an
    artefact.  With Si's valence q near 0.45 a floor at 0.05 is 11% of the
    reference, so bands at q=0.06 rendered as perfectly covered and bands at
    q=0.04 as interstitial, though both are equally free-electron-like.  A
    continuous axis shows the fall-off instead of quantising it.
    """
    if world.rank != 0:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    e = eps_kn.ravel()

    occ = qhat[eps_kn <= e_fermi]
    if occ.size:
        axes[0].axhspan(float(occ.min()), float(occ.max()), color="tab:blue",
                        alpha=0.10, zorder=0,
                        label="occupied spread (calibration)")
    axes[0].scatter(e, qhat.ravel(), s=6, alpha=0.5, color="tab:blue")
    axes[0].axhline(thresh, color="grey", ls=":", lw=1)
    axes[0].axvspan(float(eps_kn.min()), e_fermi, color="grey", alpha=0.07,
                    zorder=0)
    axes[0].set_ylabel(r"projectability  $p_{nk}$" if backend == "ao"
                       else r"atomicity  $\hat q_{nk}=q_{nk}/q_{\rm ref}$")
    axes[0].set_ylim(-0.05, max(1.25, float(np.max(qhat)) * 1.05))

    axes[1].scatter(e, r.ravel(), s=6, alpha=0.5, color="tab:green")
    axes[1].set_ylabel(r"selected fraction  $p_{nk}/q_{nk}$")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("energy (eV)")

    for ax in axes:
        for x, c, lab in [(e_fermi, "red", "$E_F$"),
                          (out_win[1], "k", "outer max"),
                          (frozen_win[1], "tab:orange", "frozen max")]:
            ax.axvline(x, color=c, ls="--", lw=1, label=lab)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title(f"projectability -- {seed}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)