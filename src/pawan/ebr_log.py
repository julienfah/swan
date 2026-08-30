"""Detailed log of an EBR projection search.
"""

from __future__ import annotations

import numpy as np

from .amn_projectability import (orbital_window_fraction,
                                 subset_projectability)

__all__ = ["write_selection_log"]


def _cols(blocks, c):
    c = np.asarray(c, int)
    parts = [np.arange(sl.start, sl.stop)
             for (j, sl), cj in zip(blocks, c) if cj > 0]
    return np.concatenate(parts) if parts else np.zeros(0, int)


def _describe(proj, describe_orbital=None, indent="        "):
    out = []
    for orb in getattr(proj, "orbitals", []):
        if describe_orbital is None:
            out.append(f"{indent}{orb}")
        else:
            out.append(describe_orbital(orb, indent=indent))
    return "\n".join(out)


def write_selection_log(path, trial_projections, blocks, scored, front, chosen,
                        froz_window, outer_window, selected_proj_set=None,
                        A=None, S=None, O=None, target_mask=None,
                        outer_mask=None, wk_k=None,
                        n_froz=None, num_wann_max=None, n_raw=None,
                        n_dedup=None, n_scored=None, describe_orbital=None,
                        steps=None, min_gain=None, rel_gain=None,
                        use_in_window=False, top=-1):
    """Write the full selection record.

    scored : [(c, coverage, nwann, cond), ...] as returned by score_combinations
    front  : the Pareto front from rank_combinations (same tuple shape)
    chosen : the selected entry of `scored`
    steps  : [(nwann_from, nwann_to, rate), ...] of the marginal-gain walk
    """
    L = []
    w = L.append

    w("=" * 78)
    w("EBR PROJECTION SEARCH")
    w("=" * 78)
    w(f"frozen window : {froz_window[0]:.4f} .. {froz_window[1]:.4f} eV")
    w(f"outer window  : {outer_window[0]:.4f} .. {outer_window[1]:.4f} eV")
    if n_froz is not None:
        w(f"frozen manifold holds {n_froz} bands at the worst k")
    if num_wann_max is not None:
        w(f"num_wann_max  : {num_wann_max}")
    if None not in (min_gain, rel_gain):
        w(f"stop rule     : marginal gain >= {min_gain} per WF (absolute) and "
          f">= {rel_gain} of the best step (relative)")
    w("")

    # ---- trial alphabet, with standalone coverage -------------------------
    w("-" * 78)
    w("TRIAL ALPHABET")
    w("-" * 78)
    w("solo coverage = coverage of the frozen manifold by that projection ALONE.")
    w("in-window     = fraction of THAT ORBITAL's own weight lying inside the")
    w("                OUTER window (coverage uses the frozen one). These ask")
    w("                opposite questions. Coverage is")
    w("                monotone in the set, so it can never penalise a useless")
    w("                orbital -- only fail to reward it. in-window can: a trial")
    w("                function whose weight sits outside the window scores low")
    w("                however much it happens to help coverage.")
    w("")
    w("                Low in-window means the orbital's weight lies outside")
    w("                the states available to the disentanglement, so no")
    w("                localised function there can be built from them. Low solo")
    w("                coverage alone is NOT disqualifying: a projection can be")
    w("                worth little by itself and a lot in combination.")
    w("")
    solo, frac = {}, {}
    if A is not None and S is not None and target_mask is not None:
        frac = orbital_window_fraction(
            A, S, blocks, target_mask if outer_mask is None else outer_mask, O=O)
        wk = np.asarray(wk_k) / np.sum(wk_k)
        wt = wk[:, None] * target_mask
        den = float(wt.sum())
        for j, sl in blocks:
            p = subset_projectability(A, S, np.arange(sl.start, sl.stop), O=O)
            solo[j] = float((wt * p).sum() / max(den, 1e-30))
    for j, sl in blocks:
        proj = trial_projections.projections[j]
        nw = sl.stop - sl.start
        s = f"  [{j:2d}] {nw:3d} WF"
        if j in solo:
            s += f"   solo coverage {solo[j]:.4f}"
        if j in frac:
            s += f"   in-window {frac[j]:.1%}"
        w(s)
        w(f"       {str(proj).splitlines()[0]}")
        d = _describe(proj, describe_orbital)
        if d:
            w(d)
    w("")

    # ---- the search --------------------------------------------------------
    w("-" * 78)
    w("COMBINATIONS")
    w("-" * 78)
    if n_raw is not None:
        w(f"  returned by EBRsearcher      : {n_raw}")
    if n_dedup is not None:
        w(f"  distinct irrep contents      : {n_dedup}   "
          "(the rest differ only in gauge: e.g. sp3 vs s+p span the same space)")
    if n_scored is not None:
        w(f"  scored (smallest by nwann)   : {n_scored}")
    n_dep = sum(1 for t in scored if len(t) > 3 and np.isfinite(t[3]) and t[3] < 1e-3)
    if n_dep:
        w(f"  rejected as linearly dependent: {n_dep}")
    w("")

    if front:
        w("  PARETO FRONT (best %s at each size)"
          % ("EFFECTIVE score = coverage x min(in-window)" if use_in_window
             else "coverage"))
        for t in front:
            mark = "   <== SELECTED" if chosen is not None and t[2] == chosen[2] \
                and abs(t[1] - chosen[1]) < 1e-12 else ""
            w(f"    {t[2]:4d} WF   coverage {t[1]:.4f}{mark}")
        if steps:
            w("")
            w("  MARGINAL GAIN ALONG THE FRONT")
            for a, b, r in steps:
                w(f"    {a:4d} -> {b:4d} WF : {r:+.5f} coverage per WF")
    w("")

    # ---- ranked table ------------------------------------------------------
    w("-" * 78)
    w(f"TOP {top} COMBINATIONS (by nwann, then coverage)")
    w("-" * 78)
    def _eff(t):
        return t[1] * (t[4] if (use_in_window and len(t) > 4) else 1.0)
    order = sorted((t for t in scored if np.isfinite(t[1])),
                   key=lambda t: (t[2], -_eff(t)))
    if use_in_window:
        w("  eff = coverage x min(in-window) over the chosen blocks. THIS is what")
        w("  the Pareto front and the selection use; `coverage` is the raw value.")
        w("")
    w("  cond = smallest eigenvalue of the normalised overlap of the chosen")
    w("  columns. ~0 means the projections are LINEARLY DEPENDENT: the set does")
    w("  not span nwann dimensions, so its coverage was computed by the")
    w("  regulariser on a smaller subspace and is NOT comparable. Such rows are")
    w("  marked REJECTED and took no part in the Pareto front.")
    w("")
    hdr = f"  {'nwann':>6} {'coverage':>9}"
    if use_in_window:
        hdr += f" {'min iw':>7} {'eff':>9}"
    w(hdr + f" {'cond':>9}   projections")
    for t in order[:top]:
        c, cov, nw = t[0], t[1], t[2]
        cd = t[3] if len(t) > 3 else float("nan")
        iw = t[4] if len(t) > 4 else 1.0
        nz = [j for (j, sl), v in zip(blocks, np.asarray(c, int)) if v > 0]
        if np.isfinite(cd) and cd < 1e-3:
            mark = "  REJECTED (dependent)"
        elif chosen is not None and np.array_equal(c, chosen[0]):
            mark = "  <== SELECTED"
        else:
            mark = ""
        row = f"  {nw:>6} {cov:>9.4f}"
        if use_in_window:
            row += f" {iw:>7.3f} {cov * iw:>9.4f}"
        w(row + f" {cd:>9.2e}   {nz}{mark}")
    w("")

    # ---- the selected set, in full ----------------------------------------
    if chosen is not None:
        w("=" * 78)
        w("SELECTED")
        w("=" * 78)
        nz = [j for (j, sl), v in zip(blocks, np.asarray(chosen[0], int)) if v > 0]
        w(f"  nwann = {chosen[2]},  "
          f"{'effective score' if use_in_window else 'coverage'} = {chosen[1]:.4f}"
          + (f",  min normalised overlap = {chosen[3]:.2e}" if len(chosen) > 3 else ""))
        w(f"  projection indices: {nz}")
        w("")
        for j in nz:
            proj = trial_projections.projections[j]
            w(f"  [{j:2d}] {str(proj).splitlines()[0]}")
            d = _describe(proj, describe_orbital, indent="       ")
            if d:
                w(d)
            w("")
        if selected_proj_set is not None:
            w("  after join_same_wyckoff():")
            for line in selected_proj_set.write_with_multiplicities(orbit=False).splitlines():
                w(f"    {line}")
    w("")

    text = "\n".join(L) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text